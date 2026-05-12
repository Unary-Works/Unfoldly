"""
LLM factory functions — extracted from core/langgraph_agent.py Phase 1.
"""
from __future__ import annotations
import os, sys, time, json, re, gc, threading, struct
from typing import TypedDict, Literal, List, Dict, Any, Optional, Callable, Annotated, Tuple, Iterator, Union, Sequence
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.messages import AIMessageChunk

from services.inproc_openai_client import get_inproc_openai_client
from utils.logger import get_logger
logger = get_logger()
from config import settings
from config.prompts import get_prompt, normalize_prompt_language
from .utils import (
    _model_supports_system_role, _estimate_tokens, _chunk_text,
    _approx_tokens_from_text, _approx_tokens_from_messages,
    build_messages_for_model, _NO_SYSTEM_ROLE_MODELS, _model_system_role_cache,
)

# ── Global abort manager (session-level interruption) ────────────────────────
class _GlobalAbortManager:
    def __init__(self):
        self._flags: Dict[str, bool] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300

    def _cleanup_old_sessions(self):
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        with self._lock:
            if len(self._flags) > 100:
                keys_to_remove = [k for k, v in list(self._flags.items()) if not v][:50]
                for k in keys_to_remove:
                    self._flags.pop(k, None)
            self._last_cleanup = now

    def set(self, session_id: Optional[str] = None):
        key = session_id or "default"
        with self._lock:
            self._flags[key] = True
        self._cleanup_old_sessions()

    def clear(self, session_id: Optional[str] = None):
        key = session_id or "default"
        with self._lock:
            self._flags.pop(key, None)

    def is_aborted(self, session_id: Optional[str] = None) -> bool:
        key = session_id or "default"
        with self._lock:
            return self._flags.get(key, False)


_GLOBAL_ABORT_MANAGER = _GlobalAbortManager()


def get_global_abort_manager() -> _GlobalAbortManager:
    return _GLOBAL_ABORT_MANAGER


def get_llm(streaming: bool = False, session_id: Optional[str] = None):

    class _InprocChatModel:
        def __init__(self, *, streaming: bool = False, temperature: float = 0.1, tools: Optional[list] = None, session_id: Optional[str] = None):
            self._streaming = bool(streaming)
            self._temperature = float(temperature)
            self._tools = tools or []
            self._session_id = session_id
            self.force_text_model = False # Added to support forcing text model from outside

            self._client = get_inproc_openai_client()

        def bind_tools(self, tools: list):
            return _InprocChatModel(streaming=self._streaming, temperature=self._temperature, tools=tools or [], session_id=self._session_id)

        def _tool_schemas_text(self) -> str:
            parts = []
            for t in (self._tools or []):
                try:
                    name = getattr(t, "name", "") or ""
                    desc = getattr(t, "description", "") or ""
                    args = None
                    # LangChain Tool: args_schema -> Pydantic model
                    if getattr(t, "args_schema", None):
                        try:
                            args = t.args_schema.schema()  # type: ignore
                        except Exception:
                            args = None
                    if args is None:
                        # fallback: common attribute
                        args = getattr(t, "args", None)
                    parts.append(json.dumps({"name": name, "description": desc, "args_schema": args}, ensure_ascii=False))
                except Exception:
                    continue
            return "\n".join(parts)

        def _messages_to_openai(self, messages: list) -> list:
            out = []
            for m in (messages or []):
                # LangChain message classes
                if isinstance(m, SystemMessage):
                    out.append({"role": "system", "content": m.content or ""})
                elif isinstance(m, HumanMessage):
                    out.append({"role": "user", "content": m.content or ""})
                elif isinstance(m, ToolMessage):
                    out.append({"role": "user", "content": f"[TOOL_RESULT id={getattr(m, 'tool_call_id', '')}]\n{m.content or ''}"})
                elif isinstance(m, AIMessage):
                    tc = getattr(m, "tool_calls", None) or []
                    if tc and not (m.content or "").strip():
                        out.append({"role": "assistant", "content": f"[TOOL_CALLS]\n{json.dumps(tc, ensure_ascii=False)}"})
                    else:
                        out.append({"role": "assistant", "content": m.content or ""})
                else:
                    # best-effort dict
                    try:
                        role = getattr(m, "role", None) or "user"
                        content = getattr(m, "content", None) or ""
                        out.append({"role": role, "content": content})
                    except Exception:
                        pass
            return out

        @staticmethod
        def _is_ctx_overflow_error(err: Exception) -> bool:
            s = str(err or "").lower()
            return (
                ("context window" in s and "requested tokens" in s)
                or ("exceed context window" in s)
                or ("n_ctx" in s and "too large" in s)
                or ("context shift" in s)
                or ("max context length" in s)
            )

        @staticmethod
        def _smart_shrink_text(text: str, target_len: int) -> str:
            raw = str(text or "")
            if len(raw) <= target_len:
                return raw
            if target_len <= 96:
                return raw[:target_len]
            keep_head = int(target_len * 0.7)
            keep_tail = max(0, target_len - keep_head - 24)
            return raw[:keep_head] + "\n...[truncated]...\n" + raw[-keep_tail:]

        def _ctx_window_tokens(self) -> int:
            try:
                from services.local_llm import get_local_llm_manager
                mgr = get_local_llm_manager()
                n_ctx = int(getattr(mgr, "default_n_ctx", 5120) or 5120)
            except Exception:
                n_ctx = int(os.getenv("FILEAGENT_LLM_N_CTX", "5120") or 5120)
            return max(1024, min(n_ctx, 32768))

        def _prompt_budget_tokens(self) -> int:
            n_ctx = self._ctx_window_tokens()
            reserve = int(os.getenv("FILEAGENT_PROMPT_RESERVE_TOKENS", "960") or 960)
            reserve = max(192, min(reserve, n_ctx // 2))
            return max(512, n_ctx - reserve)

        def _estimate_prompt_tokens_oa(self, msgs: List[Dict[str, Any]]) -> int:
            try:
                chars_per_token = float(os.getenv("LLM_CHARS_PER_TOKEN_EST", "1.6") or 1.6)
            except Exception:
                chars_per_token = 1.6
            chars_per_token = max(1.1, min(chars_per_token, 4.0))
            total = 24
            for m in (msgs or []):
                c = str((m or {}).get("content") or "")
                total += 8 + int(len(c) / chars_per_token)
            return total

        def _output_budget_tokens(self, msgs: List[Dict[str, Any]]) -> int:
            n_ctx = self._ctx_window_tokens()
            prompt_tokens = self._estimate_prompt_tokens_oa(msgs)
            hard_cap = int(os.getenv("FILEAGENT_MAX_OUTPUT_TOKENS", "1600") or 1600)
            hard_cap = max(128, min(hard_cap, 4096))
            available = max(96, n_ctx - prompt_tokens - 64)
            return max(96, min(hard_cap, available))

        @staticmethod
        def _looks_tail_incomplete(text: str) -> bool:
            import re
            t = str(text or "").rstrip()
            if len(t) < 80:
                return False
            if t.endswith(("。", "！", "？", ".", "!", "?", "”", "\"", "’", "」", "』", "）", ")", "】", "]", "`")):
                return False
            tail = t[-24:]
            if any(k in tail for k in ["例如", "比如", "包括", "其中", "以及", "和", "或", "等", "如", "备注", "：", ":"]):
                return True
            return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]$", t))

        @staticmethod
        def _merge_with_overlap(base: str, extra: str) -> str:
            a = str(base or "")
            b = str(extra or "")
            if not a:
                return b
            if not b:
                return a
            max_k = min(240, len(a), len(b))
            for k in range(max_k, 0, -1):
                if a.endswith(b[:k]):
                    return a + b[k:]
            return a + b

        def _shrink_oa_messages_once(self, msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out = [dict(m or {}) for m in (msgs or [])]
            if not out:
                return out
            if len(out) > 2:
                out.pop(1)
                return out
            if len(out) >= 1 and out[0].get("role") == "system":
                c = str(out[0].get("content") or "")
                if len(c) > 1000:
                    out[0]["content"] = self._smart_shrink_text(c, int(len(c) * 0.82))
                    return out
            for i in range(len(out) - 1, -1, -1):
                if out[i].get("role") == "user":
                    c = str(out[i].get("content") or "")
                    if len(c) > 1200:
                        out[i]["content"] = self._smart_shrink_text(c, int(len(c) * 0.8))
                        return out
            return out

        def _clip_oa_messages_to_budget(self, msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out = [dict(m or {}) for m in (msgs or [])]
            target = self._prompt_budget_tokens()
            guard = 0
            while self._estimate_prompt_tokens_oa(out) > target and guard < 40:
                nxt = self._shrink_oa_messages_once(out)
                if nxt == out:
                    break
                out = nxt
                guard += 1
            if guard > 0:
                logger.info(f"[LLM] clipped prompt for context budget: steps={guard} est_tokens={self._estimate_prompt_tokens_oa(out)}")
            return out

        def _parse_tool_calls_best_effort(self, text: str) -> Optional[list]:
            if not text:
                return None
            s = text.strip()

            def _try_json(payload: str) -> Optional[list]:
                try:
                    obj = json.loads(payload)
                except Exception:
                    return None
                if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
                    return obj.get("tool_calls")
                if isinstance(obj, list):
                    # allow direct list
                    return obj
                return None

            # 1) raw json
            if s.startswith("{") or s.startswith("["):
                tc = _try_json(s)
                if tc is not None:
                    return tc

            # 2) fenced json block
            try:
                import re

                m = re.search(r"```json\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
                if m:
                    tc = _try_json(m.group(1).strip())
                    if tc is not None:
                        return tc
            except Exception:
                pass

            # 3) tag block
            if "<tool_calls>" in s and "</tool_calls>" in s:
                inner = s.split("<tool_calls>", 1)[1].split("</tool_calls>", 1)[0].strip()
                tc = _try_json(inner)
                if tc is not None:
                    return tc

            return None

        def invoke(self, messages: list):
            oa_msgs = self._messages_to_openai(messages)
    
            is_vision_needed = False
            for m in oa_msgs:
                if m.get("role") == "user":
                    c = m.get("content", "")
                    if isinstance(c, list):
                        for item in c:
                            if isinstance(item, dict) and item.get("type") == "image_url":
                                is_vision_needed = True
                                break
                                
            extra_params = {}
            if getattr(self, "force_text_model", False):
                extra_params["force_text_model"] = True
    
            if self._tools:
                tool_text = self._tool_schemas_text()
                tool_inst = (
                    "You may call tools to complete the task.\n"
                    "If a tool call is needed, output only valid JSON in exactly this shape:\n"
                    '{"tool_calls":[{"id":"call_1","name":"TOOL_NAME","args":{}}]}\n'
                    "Do not output Markdown, explanations, or any text outside the JSON object.\n"
                    "Available tools are listed below (name/description/args_schema):\n"
                    f"{tool_text}\n"
                    "If no tool is needed, answer normally in the user's language.\n"
                )
                oa_msgs = [{"role": "system", "content": tool_inst}] + oa_msgs

            oa_msgs = self._clip_oa_messages_to_budget(oa_msgs)
            max_retry = int(os.getenv("FILEAGENT_CTX_RETRY_MAX", "3") or 3)
            resp = None
            for attempt in range(max_retry + 1):
                abort_mgr = get_global_abort_manager()
                if abort_mgr.is_aborted(self._session_id):
                    from langchain_core.messages import AIMessage
                    return AIMessage(content="")
                try:
                    it = self._client.chat.completions.create(
                        model=None,
                        messages=oa_msgs,
                        tools=self._tools if self._tools else None,
                        tool_choice="auto" if self._tools else None,
                        temperature=self._temperature,
                        max_tokens=self._output_budget_tokens(oa_msgs),
                        stream=True,
                        **extra_params
                    )
                    collected = []
                    for ev in it:
                        if abort_mgr.is_aborted(self._session_id):
                            logger.info(f"检测到中断标志，停止 invoke 生成 (session={self._session_id})")
                            try:
                                it.close()
                            except Exception:
                                pass
                            from langchain_core.messages import AIMessage
                            return AIMessage(content="".join(collected).strip())
                        try:
                            delta = ((ev.choices or [{}])[0].delta.content) or ""
                            if delta:
                                collected.append(delta)
                        except Exception:
                            pass
                    content = "".join(collected).strip()
                    break
                except Exception as e:
                    if self._is_ctx_overflow_error(e) and attempt < max_retry:
                        logger.warning(f"[LLM] invoke context overflow, shrink and retry ({attempt + 1}/{max_retry})")
                        nxt = self._shrink_oa_messages_once(oa_msgs)
                        if nxt == oa_msgs:
                            raise
                        oa_msgs = nxt
                        continue
                    raise
            
            tool_calls = self._parse_tool_calls_best_effort(content) if self._tools else None

            if tool_calls:
                from langchain_core.messages import AIMessage
                # type: ignore
                return AIMessage(content="", tool_calls=tool_calls)  # type: ignore

            from langchain_core.messages import AIMessage
            return AIMessage(content=content)
    
        def stream(self, messages: list):
            if self._tools:
                msg = self.invoke(messages)
                text = getattr(msg, "content", "") or ""
                if not text:
                    yield AIMessageChunk(content="")
                    return
                chunk_size = int(os.getenv("LOCAL_LLM_STREAM_CHUNK_CHARS", "120"))
                for i in range(0, len(text), chunk_size):
                    yield AIMessageChunk(content=text[i : i + chunk_size])
                return

            abort_mgr = get_global_abort_manager()
            
            if abort_mgr.is_aborted(self._session_id):
                logger.info(f"session已被中断，拒绝启动新生成 (session={self._session_id})")
                yield AIMessageChunk(content="")
                return

            oa_msgs = self._clip_oa_messages_to_budget(self._messages_to_openai(messages))
            stream = None
            emitted_text = ""
            finish_reason = None
            try:
                extra_params = {}
                if getattr(self, "force_text_model", False):
                    extra_params["force_text_model"] = True
                max_retry = int(os.getenv("FILEAGENT_CTX_RETRY_MAX", "3") or 3)
                for attempt in range(max_retry + 1):
                    try:
                        stream = self._client.chat.completions.create(
                            model=None,
                            messages=oa_msgs,
                            temperature=self._temperature,
                            max_tokens=self._output_budget_tokens(oa_msgs),
                            stream=True,
                            **extra_params
                        )
                        break
                    except Exception as e:
                        if self._is_ctx_overflow_error(e) and attempt < max_retry:
                            logger.warning(f"[LLM] stream context overflow, shrink and retry ({attempt + 1}/{max_retry})")
                            nxt = self._shrink_oa_messages_once(oa_msgs)
                            if nxt == oa_msgs:
                                raise
                            oa_msgs = nxt
                            continue
                        raise
                
                if abort_mgr.is_aborted(self._session_id):
                    logger.info(f"创建stream后检测到中断，立即关闭 (session={self._session_id})")
                    try:
                        stream.close()
                    except Exception:
                        pass
                    import time
                    time.sleep(0.1)
                    return
                
                for ch in stream:
                    if abort_mgr.is_aborted(self._session_id):
                        logger.info(f"检测到中断标志，停止流式生成 (session={self._session_id})")
                        try:
                            stream.close()
                        except Exception:
                            pass
                        import time
                        time.sleep(0.1)
                        return
                    try:
                        fr = getattr(ch.choices[0], "finish_reason", None)
                        if fr is not None:
                            finish_reason = str(fr).strip().lower()
                    except Exception:
                        pass
                    delta = ""
                    try:
                        delta = ch.choices[0].delta.content or ""
                    except Exception:
                        delta = ""
                    if delta:
                        emitted_text += delta
                        yield AIMessageChunk(content=delta)

                auto_continue = str(os.getenv("FILEAGENT_AUTO_CONTINUE_ON_TRUNCATION", "true")).strip().lower() in {"1", "true", "yes", "on"}
                was_len_truncated = str(finish_reason or "").strip().lower() in {"length", "max_tokens"}
                if auto_continue and (was_len_truncated or self._looks_tail_incomplete(emitted_text)):
                    max_continue_rounds = max(1, min(4, int(os.getenv("FILEAGENT_CONTINUE_ROUNDS", "2") or 2)))
                    max_continue_chars = max(120, int(os.getenv("FILEAGENT_CONTINUE_MAX_CHARS", "1200") or 1200))
                    continue_budget = max(96, min(1024, int(os.getenv("FILEAGENT_CONTINUE_OUTPUT_TOKENS", "520") or 520)))
                    appended = 0
                    need_continue = True
                    for _ in range(max_continue_rounds):
                        if (not need_continue) or appended >= max_continue_chars:
                            break
                        if abort_mgr.is_aborted(self._session_id):
                            return
                        tail = emitted_text[-1200:]
                        if any("\u4e00" <= ch <= "\u9fff" for ch in tail):
                            cont_prompt = (
                                "你上一段回答在中途结束了。请从中断处继续补全，保证语义完整。\n"
                                "- 不要重复已写内容；只续写缺失部分。\n"
                                "- 继续使用当前语言。\n\n"
                                f"<已输出内容末尾>\n{tail}\n</已输出内容末尾>"
                            )
                        else:
                            cont_prompt = (
                                "Your previous answer appears to end mid-sentence. Continue from where it stopped.\n"
                                "- Do not repeat previous content; only provide the missing continuation.\n"
                                "- Keep the same language.\n\n"
                                f"<Tail of emitted answer>\n{tail}\n</Tail of emitted answer>"
                            )

                        cont_stream = None
                        cont_piece = ""
                        cont_finish_reason = None
                        try:
                            cont_stream = self._client.chat.completions.create(
                                model=None,
                                messages=[{"role": "user", "content": cont_prompt}],
                                temperature=self._temperature,
                                max_tokens=continue_budget,
                                stream=True,
                                **extra_params
                            )
                            for ch2 in cont_stream:
                                if abort_mgr.is_aborted(self._session_id):
                                    try:
                                        cont_stream.close()
                                    except Exception:
                                        pass
                                    return
                                try:
                                    fr2 = getattr(ch2.choices[0], "finish_reason", None)
                                    if fr2 is not None:
                                        cont_finish_reason = str(fr2).strip().lower()
                                except Exception:
                                    pass
                                d2 = ""
                                try:
                                    d2 = ch2.choices[0].delta.content or ""
                                except Exception:
                                    d2 = ""
                                if not d2:
                                    continue
                                remain = max_continue_chars - appended - len(cont_piece)
                                if remain <= 0:
                                    break
                                seg = d2[:remain]
                                if not seg:
                                    continue
                                cont_piece += seg
                                yield AIMessageChunk(content=seg)
                        finally:
                            if cont_stream is not None:
                                try:
                                    cont_stream.close()
                                except Exception:
                                    pass
                        if not cont_piece:
                            break
                        emitted_text = self._merge_with_overlap(emitted_text, cont_piece)
                        appended += len(cont_piece)
                        need_continue = (
                            str(cont_finish_reason or "").strip().lower() in {"length", "max_tokens"}
                            or self._looks_tail_incomplete(emitted_text)
                        )
            except Exception as e:
                # fallback: non-stream
                logger.error(f"Stream error: {e}, falling back to non-stream")
                if abort_mgr.is_aborted(self._session_id):
                    logger.info(f"session已中断，跳过fallback")
                    return
                msg = self.invoke(messages)
                text = getattr(msg, "content", "") or ""
                if text:
                    yield AIMessageChunk(content=text)
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                    import time
                    time.sleep(0.05)

    return _InprocChatModel(streaming=bool(streaming), temperature=0.1, session_id=session_id)

def get_llm_with_tools():
    llm = get_llm()
    tools = get_all_tools()
    if not tools:
        tools = [count_documents, search_documents]
    return llm.bind_tools(tools)



class ToolAgentState(TypedDict):
    messages: List[Any]
    question: str
    final_answer: str
    source_files: List[Dict]
    trace: List[Dict[str, Any]]
    steps: int
    require_tools: bool
