from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Generator, List, Optional

from utils.logger import get_logger

logger = get_logger()


class ActionExecutor:
    """
    Executes structured, non-search actions that were previously handled inline
    inside dispatch. Keeping these branches here makes dispatch focus on
    orchestration while preserving the legacy runtime behavior.
    """

    def __init__(self, agent: Any):
        self.agent = agent

    def execute_structured_action(
        self,
        *,
        action: str,
        params: Dict[str, Any],
        question: str,
        active_paths: Optional[List[str]],
        opened_file_path: Optional[str] = None,
        session_id: Optional[str],
        hist_ref: List[Dict[str, Any]],
        user_lang: str,
        response_language_label: str,
        emit_status_enabled: bool,
        collect_or_emit_stream: Callable[..., Generator[Dict[str, Any], None, Optional[str]]],
        emit_files_from_sources: Callable[..., Generator[Dict[str, Any], None, None]],
        emit_status_fn: Callable[..., Generator[Dict[str, Any], None, None]],
        to_user_text: Callable[[str], str],
        stream_natural_count_reply: Callable[..., str],
        icon_type_for_path: Callable[[str], str],
        request_started_at: Optional[float] = None,
    ) -> Generator[Dict[str, Any], None, bool]:
        q = (question or "").strip()
        params = dict(params or {})
        action_started_at = time.time()
        opened_file_path = str(opened_file_path or params.get("opened_file_path") or params.get("_opened_file_path") or "").strip()

        def _effective_active_paths(*, selected_scope: bool = False, file_hint: str = "") -> Optional[List[str]]:
            paths = list(active_paths or [])
            if paths:
                if selected_scope:
                    hint = str(file_hint or "").strip().lower()
                    if hint:
                        hint_matches = [
                            path
                            for path in paths
                            if hint in os.path.basename(str(path or "")).lower()
                            or os.path.basename(str(path or "")).lower() in hint
                        ]
                        if len(hint_matches) == 1:
                            return hint_matches
                    if opened_file_path and len(paths) > 50:
                        return [opened_file_path]
                return paths
            if selected_scope and opened_file_path:
                return [opened_file_path]
            return active_paths

        def _last_result_paths() -> List[str]:
            getter = getattr(self.agent, "_get_last_search_results_ref", None)
            rows = getter(session_id) if callable(getter) else []
            out: List[str] = []
            seen = set()
            for row in list(rows or []):
                if not isinstance(row, dict):
                    continue
                fp = str(row.get("file_path") or "").strip()
                if not fp or fp in seen:
                    continue
                seen.add(fp)
                out.append(fp)
            return out

        def _dedupe_source_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for row in list(rows or []):
                if not isinstance(row, dict):
                    continue
                key = str(row.get("file_path") or row.get("file_name") or "").strip()
                if key:
                    if key in seen:
                        continue
                    seen.add(key)
                out.append(row)
            return out

        def _last_result_media_paths() -> List[str]:
            try:
                from core.media.media_expert import MEDIA_EXTENSIONS
            except Exception:
                MEDIA_EXTENSIONS = {
                    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
                    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv",
                }
            out: List[str] = []
            seen = set()
            for fp in _last_result_paths():
                if os.path.splitext(fp)[1].lower() not in MEDIA_EXTENSIONS:
                    continue
                if fp in seen:
                    continue
                seen.add(fp)
                out.append(fp)
            return out

        def _media_export_active_paths() -> Optional[List[str]]:
            file_hint = str(
                (params or {}).get("file_hint")
                or (params or {}).get("focused_file")
                or ""
            ).strip()
            selection_scope = bool(
                file_hint
                or (params or {}).get("_selection_media_scope")
                or str((params or {}).get("_scope") or "").lower() in {"selected", "selected_items", "selected_folder"}
                or str((params or {}).get("scope") or "").lower() in {"selected", "selected_items", "selected_folder"}
            )
            scoped_paths = _effective_active_paths(selected_scope=selection_scope, file_hint=file_hint)
            if file_hint or opened_file_path:
                return scoped_paths

            prior_media_paths = _last_result_media_paths()
            current_paths = list(scoped_paths or [])
            # In normal app sessions active_paths can mean the whole indexed corpus.
            # For media follow-ups like "what is said at 10s in it?", the prior
            # media results are the conversational scope and should not be diluted
            # by the full selected corpus.
            if prior_media_paths and (not current_paths or len(current_paths) > 50):
                return prior_media_paths
            return scoped_paths

        def _elapsed_ms() -> int:
            base = request_started_at or action_started_at
            return int((time.time() - base) * 1000)

        def _trace_done(*, query_type: str, answer_length: int = 0, sources_count: int = 0) -> Dict[str, Any]:
            return {
                "type": "trace_append",
                "item": {
                    "stage": "structured_action_done",
                    "type": "handler",
                    "action": action,
                    "query_type": query_type,
                    "elapsed_ms": _elapsed_ms(),
                    "duration_ms": int((time.time() - action_started_at) * 1000),
                    "answer_length": int(answer_length or 0),
                    "sources_count": int(sources_count or 0),
                },
            }

        yield {
            "type": "trace_append",
            "item": {
                "stage": "structured_action_start",
                "type": "handler",
                "action": action,
                "elapsed_ms": _elapsed_ms(),
                "question": q[:120],
            },
        }

        if action == "translate_response":
            yield {"type": "thinking", "delta": "Mode: translate_response, translating previous answer...\n"}
            yield from emit_status_fn("running", "Translating...")
            target_lang = str((params or {}).get("lang") or "en").strip().lower()
            if target_lang.startswith("zh"):
                target_lang_label = "Chinese (Simplified)"
            else:
                target_lang_label = "English"
            prev_answer = ""
            try:
                hist_snapshot = list(hist_ref)
                if len(hist_snapshot) >= 2:
                    prev_answer = str(hist_snapshot[-2].get("a") or "").strip()
                elif len(hist_snapshot) == 1:
                    prev_answer = str(hist_snapshot[-1].get("a") or "").strip()
            except Exception:
                pass
            if not prev_answer:
                action = "process_previous"
                yield {"type": "thinking", "delta": "No prior response to translate; falling back to process_previous.\n"}
            else:
                translate_prompt = (
                    f"Translate the following text to {target_lang_label}. "
                    "Keep formatting (bullet points, headers) intact. "
                    "Output the translation only, no explanation:\n\n"
                    f"{prev_answer}"
                )
                llm = self.agent._get_llm_service(
                    detailed=False,
                    session_id=session_id,
                    prompt_language=user_lang,
                )
                resp_text = yield from collect_or_emit_stream(llm, translate_prompt)
                if resp_text is None:
                    resp_text = ""
                try:
                    hist_ref[-1]["a"] = resp_text
                except Exception:
                    pass
                yield _trace_done(query_type="translate", answer_length=len(resp_text or ""))
                yield {"type": "done", "ok": True, "query_type": "translate", "sources": [], "trace": []}
                return True

        if action == "summarize_all":
            summarize_paths = _effective_active_paths(
                selected_scope=bool(
                    (params or {}).get("_scope") == "selected"
                    or (params or {}).get("_preserve_selected_scope")
                    or (params or {}).get("_selection_media_scope")
                ),
                file_hint=str((params or {}).get("file_hint") or (params or {}).get("focused_file") or ""),
            )
            for ev in self.agent._handle_summarize_all(
                q,
                params,
                summarize_paths,
                session_id,
                emit_status_enabled,
                prompt_language=user_lang,
            ):
                if ev.get("type") == "sources":
                    yield from emit_files_from_sources(
                        ev.get("content") or [],
                        total_matches=ev.get("total_matches"),
                        shown_count=ev.get("shown_count"),
                    )
                elif ev.get("type") == "text":
                    piece = ev.get("delta") or ev.get("content") or ""
                    if piece:
                        yield {"type": "text", "delta": piece}
                elif ev.get("type") == "done":
                    yield _trace_done(
                        query_type=str(ev.get("query_type") or "summarize_all"),
                        sources_count=len(ev.get("sources") or []),
                    )
                    yield ev
                    return True
                else:
                    yield ev
            yield _trace_done(query_type="summarize_all")
            return True

        if action == "db_clear":
            if user_lang == "zh":
                msg = (
                    "你确定要清空数据库中的所有索引数据吗？\n\n"
                    "这将删除所有已索引文件的记录和向量数据（不会删除原始文件）。\n"
                    "操作不可逆。如果确认，请回复\"确认清空\"。"
                )
            else:
                msg = (
                    "Are you sure you want to clear all indexed data?\n\n"
                    "This will remove indexed records and vectors (original files are not deleted).\n"
                    'This action is irreversible. If confirmed, reply with "confirm clear".'
                )
            yield {"type": "text", "delta": msg}
            try:
                hist_ref[-1]["a"] = msg
            except Exception:
                pass
            yield _trace_done(query_type="clarify", answer_length=len(msg))
            yield {"type": "done", "ok": True, "query_type": "clarify", "sources": [], "trace": []}
            return True

        if action == "clarify":
            fallback_msg = (
                "我还没法判断你具体想找什么。你可以补充关键词、文件类型、文件名，或者说明你想让我总结什么。"
                if user_lang == "zh"
                else "I can't tell what you want me to look for yet. You can provide keywords, file type, a file name, or say what you want summarized."
            )
            clarify_sources: List[Dict[str, Any]] = []
            for raw_path in list(active_paths or []):
                fp = os.path.abspath(os.path.expanduser(str(raw_path or "").strip()))
                if not fp:
                    continue
                clarify_sources.append(
                    {
                        "file_path": fp,
                        "file_name": os.path.basename(fp),
                        "relevance_score": 1.0,
                    }
                )
            msg = str(params.get("question") or "")
            clarify_context = params.get("_clarify_context") if isinstance(params.get("_clarify_context"), dict) else {}
            if clarify_context or params.get("_clarify_with_context"):
                context_json = json.dumps(clarify_context or {}, ensure_ascii=False, indent=2, default=str)
                fallback_context_msg = msg.strip() or fallback_msg
                clarify_prompt = (
                    f"You are a file assistant generating a clarification question. Reply in {response_language_label}.\n"
                    "A separate ClarifyGate has already decided that the user's request needs clarification.\n"
                    "Use the context below. Do not answer the user's underlying request. Ask the smallest useful clarification question.\n"
                    "If this is a scope ambiguity, ask the user to choose among the previous relevant files, the current selected files, or another explicit file scope.\n"
                    "Mention concrete context when useful, such as previous-result count, selected-file count, or example file names.\n"
                    "Keep it brief and natural.\n\n"
                    f"<Clarify Context>\n{context_json}\n</Clarify Context>\n\n"
                    f"<User Question>\n{q}\n</User Question>\n\n"
                    f"Fallback clarification if context is insufficient: {fallback_context_msg}\n"
                )
                llm = self.agent._get_llm_service(
                    detailed=False,
                    session_id=session_id,
                    prompt_language=user_lang,
                )
                resp_text = yield from collect_or_emit_stream(llm, clarify_prompt)
                msg = str(resp_text or "").strip() or fallback_context_msg
                try:
                    hist_ref[-1]["a"] = msg
                except Exception:
                    pass
                yield _trace_done(query_type="clarify", answer_length=len(msg))
                if clarify_sources:
                    yield {
                        "type": "sources",
                        "content": clarify_sources,
                        "total_matches": len(clarify_sources),
                        "shown_count": len(clarify_sources),
                    }
                yield {"type": "done", "ok": True, "query_type": "clarify", "sources": clarify_sources, "trace": []}
                return True

            msg_lower = msg.strip().lower()
            _generic_clarify = (
                not msg.strip()
                or msg_lower in {
                    "what would you like to search for?",
                    "what do you want to find about",
                }
                or msg_lower.startswith("what would you like to search for")
                or msg_lower.startswith("what do you want to find about")
            )
            if _generic_clarify:
                clarify_prompt = (
                    f"You are a helpful file assistant. Reply in {response_language_label}.\n"
                    "The user's request is too ambiguous to answer safely.\n"
                    "Do not guess the intent and do not provide file content.\n"
                    "Write a brief, natural response that says you still can't tell what they mean, "
                    "then gently guide them to provide one of: a keyword, file type, exact file name, "
                    "or what they want summarized/explained.\n"
                    "Keep it short and warm. Do not use bullet points. Do not sound robotic.\n\n"
                    f"<User Question>\n{q}\n</User Question>\n"
                )
                llm = self.agent._get_llm_service(
                    detailed=False,
                    session_id=session_id,
                    prompt_language=user_lang,
                )
                resp_text = yield from collect_or_emit_stream(llm, clarify_prompt)
                msg = str(resp_text or "").strip() or fallback_msg
            if user_lang == "zh":
                if not any("\u4e00" <= ch <= "\u9fff" for ch in msg):
                    msg = fallback_msg
            else:
                if any("\u4e00" <= ch <= "\u9fff" for ch in msg):
                    msg = fallback_msg
            if not _generic_clarify:
                yield {"type": "text", "delta": msg}
            try:
                hist_ref[-1]["a"] = msg
            except Exception:
                pass
            yield _trace_done(query_type="clarify", answer_length=len(msg))
            if clarify_sources:
                yield {
                    "type": "sources",
                    "content": clarify_sources,
                    "total_matches": len(clarify_sources),
                    "shown_count": len(clarify_sources),
                }
            yield {"type": "done", "ok": True, "query_type": "clarify", "sources": clarify_sources, "trace": []}
            return True

        if action == "chat":
            yield {"type": "thinking", "delta": "Mode: chat, generating response...\n"}
            yield from emit_status_fn("thinking", "Generating response...")
            llm = self.agent._get_llm_service(
                detailed=False,
                session_id=session_id,
                prompt_language=user_lang,
            )
            chat_prompt = (
                f"You are a helpful file assistant. Answer in {response_language_label}.\n"
                "STRICT RULES:\n"
                "1. You may ONLY answer general usage/help questions (e.g. how to use this app, greetings, meta questions).\n"
                "2. If the user is asking about files, documents, or specific content in their knowledge base, "
                "do NOT answer from memory or conversation history — instead tell them to search using a keyword.\n"
                "3. Do NOT fabricate, infer, or assume any file content, person details, or statistics "
                "that were not explicitly provided by a database search result in THIS message.\n"
                "4. Keep your reply concise (2-3 sentences max).\n\n"
                f"<User Question>\n{q}\n</User Question>\n"
            )
            resp_text = yield from collect_or_emit_stream(llm, chat_prompt)
            if resp_text is None:
                return True
            try:
                hist_ref[-1]["a"] = resp_text
            except Exception:
                pass
            yield _trace_done(query_type="chat", answer_length=len(resp_text or ""))
            yield {"type": "done", "ok": True, "query_type": "chat", "sources": [], "trace": []}
            return True

        if action not in {
            "count",
            "summarize",
            "process_previous",
            "view_detail",
            "open_file",
            "media_export",
            "media_content_search",
        }:
            return False

        yield {"type": "thinking", "delta": f"Mode: {action}, executing...\n"}

        if action == "process_previous":
            yield from emit_status_fn("running", "Processing previous results...")
            resp_text = ""
            last_done_event: Optional[Dict[str, Any]] = None
            for ev in self.agent._handle_process_previous(
                q,
                session_id=session_id,
                prompt_language=user_lang,
                active_paths=active_paths,
                params=params,
            ):
                if ev.get("type") == "files":
                    preview = ev.get("preview") or []
                    all_items = _dedupe_source_rows(list(ev.get("all") or preview or []))
                    if session_id:
                        try:
                            self.agent._set_last_search_results(session_id, list(all_items or [])[:50])
                        except Exception:
                            pass
                    yield from emit_files_from_sources(
                        all_items,
                        total_matches=ev.get("total_matches"),
                        shown_count=ev.get("shown_count"),
                    )
                    continue
                if ev.get("type") == "sources":
                    source_items = _dedupe_source_rows(list(ev.get("content") or []))
                    if session_id:
                        try:
                            self.agent._set_last_search_results(
                                session_id,
                                list(source_items or [])[:50],
                            )
                        except Exception:
                            pass
                    yield from emit_files_from_sources(
                        source_items,
                        total_matches=ev.get("total_matches"),
                        shown_count=ev.get("shown_count"),
                    )
                    continue
                if ev.get("type") == "text":
                    d = ev.get("content") or ev.get("delta") or ""
                    if d:
                        yield {"type": "text", "delta": d}
                        resp_text += d
                    continue
                if ev.get("type") == "done":
                    last_done_event = ev
                    done_query_type = str(ev.get("query_type") or "process")
                    done_sources = _dedupe_source_rows(list(ev.get("sources") or []))
                    if done_query_type in {"search", "media_content_search", "summarize", "summarize_all"}:
                        # These are implementation details inside a scoped
                        # follow-up; the user-facing routed intent is still
                        # process_previous/process.
                        done_query_type = "process"
                    if session_id and ev.get("sources") is not None:
                        try:
                            self.agent._set_last_search_results(
                                session_id,
                                _dedupe_source_rows(list(ev.get("sources") or []))[:50],
                            )
                        except Exception:
                            pass
                    try:
                        hist_ref[-1]["a"] = resp_text
                    except Exception:
                        pass
                    yield _trace_done(
                        query_type=done_query_type,
                        answer_length=len(resp_text or ""),
                        sources_count=len(done_sources),
                    )
                    yield {
                        "type": "done",
                        "ok": bool(ev.get("ok", True)),
                        "query_type": done_query_type,
                        "sources": done_sources,
                        "trace": list(ev.get("trace") or []),
                    }
                    return True
            try:
                hist_ref[-1]["a"] = resp_text
            except Exception:
                pass
            yield _trace_done(
                query_type=(
                    "process"
                    if str((last_done_event or {}).get("query_type") or "process")
                    in {"search", "media_content_search", "summarize", "summarize_all"}
                    else str((last_done_event or {}).get("query_type") or "process")
                ),
                answer_length=len(resp_text or ""),
                sources_count=len(_dedupe_source_rows(list((last_done_event or {}).get("sources") or []))),
            )
            done_query_type = str((last_done_event or {}).get("query_type") or "process")
            if done_query_type in {"search", "media_content_search", "summarize", "summarize_all"}:
                done_query_type = "process"
            done_sources = _dedupe_source_rows(list((last_done_event or {}).get("sources") or []))
            yield {
                "type": "done",
                "ok": bool((last_done_event or {}).get("ok", True)),
                "query_type": done_query_type,
                "sources": done_sources,
                "trace": list((last_done_event or {}).get("trace") or []),
            }
            return True

        if action == "media_export":
            status_msg = "Analyzing media timestamp query..."
            if str((params or {}).get("sub_intent") or "") == "range_summary":
                status_msg = "Analyzing requested media interval..."
            yield from emit_status_fn("running", status_msg)
            resp_text = ""
            last_done_event: Optional[Dict[str, Any]] = None
            for ev in self.agent._handle_media_export(
                q,
                params,
                session_id=session_id,
                prompt_language=user_lang,
                active_paths=_media_export_active_paths(),
            ):
                if ev.get("type") == "files":
                    preview = ev.get("preview") or []
                    all_items = _dedupe_source_rows(list(ev.get("all") or preview or []))
                    if session_id:
                        try:
                            self.agent._set_last_search_results(session_id, list(all_items or [])[:50])
                        except Exception:
                            pass
                    yield from emit_files_from_sources(
                        all_items,
                        total_matches=ev.get("total_matches"),
                        shown_count=ev.get("shown_count"),
                    )
                    continue
                if ev.get("type") == "sources":
                    source_items = _dedupe_source_rows(list(ev.get("content") or []))
                    if session_id:
                        try:
                            self.agent._set_last_search_results(session_id, list(source_items or [])[:50])
                        except Exception:
                            pass
                    yield from emit_files_from_sources(
                        source_items,
                        total_matches=ev.get("total_matches"),
                        shown_count=ev.get("shown_count"),
                    )
                    continue
                if ev.get("type") == "thinking":
                    d = ev.get("delta") or ev.get("content") or ""
                    if d:
                        yield {"type": "thinking", "delta": d}
                    continue
                if ev.get("type") == "status":
                    phase = str(ev.get("phase") or "running")
                    msg = str(ev.get("message") or ev.get("content") or "")
                    if msg:
                        yield from emit_status_fn(phase, msg)
                    continue
                if ev.get("type") == "text":
                    d = ev.get("content") or ev.get("delta") or ""
                    if d:
                        yield {"type": "text", "delta": d}
                        resp_text += d
                    continue
                if ev.get("type") == "done":
                    last_done_event = ev
                    done_sources = _dedupe_source_rows(list(ev.get("sources") or []))
                    if session_id and ev.get("sources") is not None:
                        try:
                            self.agent._set_last_search_results(session_id, done_sources[:50])
                        except Exception:
                            pass
                    try:
                        hist_ref[-1]["a"] = resp_text
                    except Exception:
                        pass
                    yield _trace_done(
                        query_type=str(ev.get("query_type") or "media_export"),
                        answer_length=len(resp_text or ""),
                        sources_count=len(done_sources),
                    )
                    yield {
                        "type": "done",
                        "ok": bool(ev.get("ok", True)),
                        "query_type": ev.get("query_type", "media_export"),
                        "sources": done_sources,
                        "trace": list(ev.get("trace") or []),
                    }
                    return True
            try:
                hist_ref[-1]["a"] = resp_text
            except Exception:
                pass
            yield _trace_done(
                query_type=str((last_done_event or {}).get("query_type") or "media_export"),
                answer_length=len(resp_text or ""),
                sources_count=len(_dedupe_source_rows(list((last_done_event or {}).get("sources") or []))),
            )
            done_sources = _dedupe_source_rows(list((last_done_event or {}).get("sources") or []))
            yield {
                "type": "done",
                "ok": bool((last_done_event or {}).get("ok", True)),
                "query_type": (last_done_event or {}).get("query_type", "media_export"),
                "sources": done_sources,
                "trace": list((last_done_event or {}).get("trace") or []),
            }
            return True

        if action == "media_content_search":
            yield from emit_status_fn("running", "Searching indexed media content...")
            resp_text = ""
            last_done_event: Optional[Dict[str, Any]] = None
            for ev in self.agent._handle_media_content_search(
                q,
                params,
                session_id=session_id,
                prompt_language=user_lang,
                active_paths=active_paths,
            ):
                if ev.get("type") == "files":
                    preview = ev.get("preview") or []
                    all_items = ev.get("all") or preview
                    yield from emit_files_from_sources(
                        all_items,
                        total_matches=ev.get("total_matches"),
                        shown_count=ev.get("shown_count"),
                    )
                    continue
                if ev.get("type") == "sources":
                    yield from emit_files_from_sources(
                        ev.get("content") or [],
                        total_matches=ev.get("total_matches"),
                        shown_count=ev.get("shown_count"),
                    )
                    continue
                if ev.get("type") == "text":
                    d = ev.get("content") or ev.get("delta") or ""
                    if d:
                        yield {"type": "text", "delta": d}
                        resp_text += d
                    continue
                if ev.get("type") == "done":
                    last_done_event = ev
                    try:
                        hist_ref[-1]["a"] = resp_text
                    except Exception:
                        pass
                    yield _trace_done(
                        query_type=str(ev.get("query_type") or "media_content_search"),
                        answer_length=len(resp_text or ""),
                        sources_count=len(ev.get("sources") or []),
                    )
                    yield {
                        "type": "done",
                        "ok": bool(ev.get("ok", True)),
                        "query_type": ev.get("query_type", "media_content_search"),
                        "sources": list(ev.get("sources") or []),
                        "trace": list(ev.get("trace") or []),
                    }
                    return True
            try:
                hist_ref[-1]["a"] = resp_text
            except Exception:
                pass
            yield _trace_done(
                query_type=str((last_done_event or {}).get("query_type") or "media_content_search"),
                answer_length=len(resp_text or ""),
                sources_count=len((last_done_event or {}).get("sources") or []),
            )
            yield {
                "type": "done",
                "ok": bool((last_done_event or {}).get("ok", True)),
                "query_type": (last_done_event or {}).get("query_type", "media_content_search"),
                "sources": list((last_done_event or {}).get("sources") or []),
                "trace": list((last_done_event or {}).get("trace") or []),
            }
            return True

        if action == "count":
            cat = self.agent._normalize_category_name(str(params.get("category") or "all"))
            selection_mode = str((params or {}).get("_selection_mode") or "").strip().lower()
            needs_fallback = False
            if cat == "other" and ("其他" not in q and "other" not in q.lower()):
                cat = "all"
                needs_fallback = False
            if cat == "all" and active_paths == []:
                needs_fallback = False

            if selection_mode == "selected_items":
                yield from emit_status_fn("running", "Listing selected items")
                selected_sources: List[Dict[str, Any]] = []
                seen_paths = set()
                media_type_filter = str((params or {}).get("media_type") or "").strip().lower()
                media_scope_category = self.agent._normalize_category_name(str((params or {}).get("category") or ""))
                selected_paths = _effective_active_paths(
                    selected_scope=True,
                    file_hint=str((params or {}).get("file_hint") or (params or {}).get("focused_file") or ""),
                ) or []

                if not media_type_filter and media_scope_category in {"audio", "video"}:
                    media_type_filter = media_scope_category

                def _selected_item_matches_scope(file_path: str, *, is_dir: bool) -> bool:
                    if is_dir and (media_type_filter or media_scope_category == "audio/video"):
                        return False
                    if not (media_type_filter or media_scope_category == "audio/video"):
                        return True
                    try:
                        from core.media.media_expert import AUDIO_EXTENSIONS as _AUDIO_EXTS, VIDEO_EXTENSIONS as _VIDEO_EXTS
                    except Exception:
                        _AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape"}
                        _VIDEO_EXTS = {".mp4", ".m4v", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".ts"}
                    ext = os.path.splitext(file_path.lower())[1]
                    if media_type_filter == "video":
                        return ext in _VIDEO_EXTS
                    if media_type_filter == "audio":
                        return ext in _AUDIO_EXTS
                    return ext in _AUDIO_EXTS or ext in _VIDEO_EXTS

                for raw_path in list(selected_paths or []):
                    file_path = str(raw_path or "").strip()
                    if not file_path or file_path in seen_paths:
                        continue
                    seen_paths.add(file_path)
                    file_name = os.path.basename(file_path.rstrip("/")) or file_path
                    is_dir = False
                    try:
                        is_dir = os.path.isdir(file_path)
                    except Exception:
                        is_dir = False
                    if not _selected_item_matches_scope(file_path, is_dir=is_dir):
                        continue
                    selected_sources.append(
                        {
                            "file_name": file_name,
                            "file_path": file_path,
                            "type": "folder" if is_dir else icon_type_for_path(file_path),
                            "iconType": "folder" if is_dir else icon_type_for_path(file_path),
                            "doc_category": "folder" if is_dir else (media_type_filter or ("audio/video" if media_scope_category == "audio/video" else "")),
                        }
                    )

                yield from emit_files_from_sources(
                    selected_sources,
                    total_matches=len(selected_sources),
                    shown_count=len(selected_sources),
                )

                folder_count = sum(1 for item in selected_sources if item.get("type") == "folder")
                file_count = max(0, len(selected_sources) - folder_count)
                media_label_zh = ""
                media_label_en = ""
                if media_type_filter == "video":
                    media_label_zh = "视频文件"
                    media_label_en = "video file"
                elif media_type_filter == "audio":
                    media_label_zh = "音频文件"
                    media_label_en = "audio file"
                elif media_scope_category == "audio/video":
                    media_label_zh = "音视频文件"
                    media_label_en = "media file"
                if user_lang == "zh":
                    if media_label_zh:
                        if not selected_sources:
                            resp_text = f"当前选中的条目里没有{media_label_zh}。"
                        else:
                            resp_text = f"当前选中的条目里共有 {len(selected_sources)} 个{media_label_zh}。"
                    elif not selected_sources:
                        resp_text = "当前没有选中的文件或文件夹。"
                    elif folder_count and file_count:
                        resp_text = f"当前共选中 {len(selected_sources)} 个条目，其中 {folder_count} 个文件夹、{file_count} 个文件。"
                    elif folder_count:
                        resp_text = f"当前共选中 {len(selected_sources)} 个文件夹。"
                    else:
                        resp_text = f"当前共选中 {len(selected_sources)} 个文件。"
                else:
                    if media_label_en:
                        if not selected_sources:
                            resp_text = f"There are no selected {media_label_en}s right now."
                        else:
                            resp_text = f"There {'is' if len(selected_sources) == 1 else 'are'} {len(selected_sources)} selected {media_label_en}{'' if len(selected_sources) == 1 else 's'}."
                    elif not selected_sources:
                        resp_text = "There are no selected files or folders right now."
                    elif folder_count and file_count:
                        resp_text = (
                            f"You currently have {len(selected_sources)} selected items: "
                            f"{folder_count} folder{'s' if folder_count != 1 else ''} and "
                            f"{file_count} file{'s' if file_count != 1 else ''}."
                        )
                    elif folder_count:
                        resp_text = f"You currently have {len(selected_sources)} selected folder{'s' if len(selected_sources) != 1 else ''}."
                    else:
                        resp_text = f"You currently have {len(selected_sources)} selected file{'s' if len(selected_sources) != 1 else ''}."

                yield {"type": "text", "delta": resp_text}
                try:
                    hist_ref[-1]["a"] = resp_text
                except Exception:
                    pass
                yield _trace_done(query_type="count", answer_length=len(resp_text), sources_count=len(selected_sources))
                yield {"type": "done", "ok": True, "query_type": "count", "sources": list(selected_sources), "trace": []}
                return True

            if not needs_fallback:
                yield from emit_status_fn("running", f"Counting: {cat}")
                resp_text = ""
                raw_count_text = ""
                count_sources: List[Dict[str, Any]] = []
                has_ext_filter = bool((params or {}).get("file_extensions"))
                scope_value = str(
                    (params or {}).get("_scope")
                    or (params or {}).get("scope")
                    or (params or {}).get("_context_scope")
                    or ""
                ).strip().lower()
                scope_reason = str((params or {}).get("_scope_disambiguation") or "").strip().lower()
                scope_selected = scope_value in {"selected", "selected_items", "selected_folder"}
                scope_previous = (
                    scope_value in {"previous", "last_results", "previous_results", "prior_results", "prior", "results"}
                    or scope_reason in {"explicit_previous_scope", "personal_attribute_previous_scope", "contextual_followup"}
                )
                if cat == "all" and not has_ext_filter and not scope_selected and not scope_previous:
                    count_paths = None
                else:
                    if scope_selected:
                        count_paths = _effective_active_paths(
                            selected_scope=True,
                            file_hint=str((params or {}).get("file_hint") or (params or {}).get("focused_file") or ""),
                        )
                    elif scope_previous:
                        previous_paths = _last_result_paths()
                        count_paths = previous_paths if previous_paths else []
                    else:
                        count_paths = active_paths or None
                for ev in self.agent._handle_count(
                    cat,
                    q,
                    allowed_paths=count_paths,
                    session_id=session_id,
                    params=params,
                ):
                    if ev.get("type") == "fallback_to_search":
                        needs_fallback = True
                        break
                    if ev.get("type") == "sources":
                        count_sources = list(ev.get("content") or [])
                        yield from emit_files_from_sources(
                            count_sources,
                            total_matches=ev.get("total_matches"),
                            shown_count=ev.get("shown_count"),
                        )
                        continue
                    if ev.get("type") == "text":
                        d = ev.get("content") or ""
                        if d:
                            raw_count_text += d
                        continue
                    if ev.get("type") == "done":
                        resp_text = raw_count_text
                        if resp_text:
                            for ln in resp_text.splitlines(True):
                                yield {"type": "text", "delta": ln}
                        explain = stream_natural_count_reply(
                            user_question=q,
                            structured_count_text=raw_count_text,
                            file_preview=count_sources,
                        )
                        if explain:
                            yield {"type": "text", "delta": "\n\n" + explain}
                            resp_text = (resp_text + "\n\n" + explain).strip()
                        if count_sources:
                            self.agent._set_followup_hint(
                                session_id,
                                action="process_previous",
                                params={},
                                ttl_turns=2,
                                uses=2,
                            )
                        else:
                            self.agent._clear_followup_hint(session_id, reason="count_no_sources")
                        try:
                            hist_ref[-1]["a"] = resp_text
                        except Exception:
                            pass
                        yield _trace_done(query_type="count", answer_length=len(resp_text or ""))
                        yield {"type": "done", "ok": True, "query_type": "count", "sources": [], "trace": []}
                        return True

            if needs_fallback:
                return False

            try:
                hist_ref[-1]["a"] = resp_text
            except Exception:
                pass
            yield _trace_done(query_type="count", answer_length=len(resp_text or ""))
            yield {"type": "done", "ok": True, "query_type": "count", "sources": [], "trace": []}
            return True

        if action == "summarize":
            cat = self.agent._normalize_category_name(str(params.get("category") or ""))
            if not cat or cat in {"all", "other"}:
                last_results_for_summary = self.agent._get_last_search_results_ref(session_id)
                if last_results_for_summary:
                    yield {"type": "thinking", "delta": "No category provided; summarizing previous results...\n"}
                    yield from emit_status_fn("running", "Summarizing previous results...")
                    resp_text = ""
                    for ev in self.agent._handle_process_previous(
                        q,
                        session_id=session_id,
                        prompt_language=user_lang,
                        active_paths=active_paths,
                        params=params,
                    ):
                        if ev.get("type") == "sources":
                            yield from emit_files_from_sources(
                                ev.get("content") or [],
                                total_matches=ev.get("total_matches"),
                                shown_count=ev.get("shown_count"),
                            )
                            continue
                        if ev.get("type") == "text":
                            d = ev.get("content") or ev.get("delta") or ""
                            if d:
                                yield {"type": "text", "delta": d}
                                resp_text += d
                            continue
                        if ev.get("type") == "done":
                            try:
                                hist_ref[-1]["a"] = resp_text
                            except Exception:
                                pass
                            yield _trace_done(query_type="summarize", answer_length=len(resp_text or ""))
                            yield {"type": "done", "ok": True, "query_type": "summarize", "sources": [], "trace": []}
                            return True
                    try:
                        hist_ref[-1]["a"] = resp_text
                    except Exception:
                        pass
                    yield _trace_done(query_type="summarize", answer_length=len(resp_text or ""))
                    yield {"type": "done", "ok": True, "query_type": "summarize", "sources": [], "trace": []}
                    return True
                for ev in self.agent._handle_count("all", q, allowed_paths=active_paths, session_id=session_id):
                    if ev.get("type") == "sources":
                        yield from emit_files_from_sources(
                            ev.get("content") or [],
                            total_matches=ev.get("total_matches"),
                            shown_count=ev.get("shown_count"),
                        )
                        continue
                    if ev.get("type") == "text":
                        d = to_user_text(ev.get("content") or "")
                        yield {"type": "text", "delta": d}
                yield {"type": "done", "ok": True, "query_type": "count", "sources": [], "trace": []}
                return True

            yield from emit_status_fn("running", f"Summarizing topics: {cat}")
            resp_text = ""
            raw_kw = str(params.get("query") or params.get("keyword") or "").strip()
            kw = self.agent._normalize_summarize_keyword(cat, raw_kw, q) or ""
            if raw_kw and not kw:
                logger.info(f"总结关键词已忽略（泛意图词）：raw_kw={raw_kw}")
            for ev in self.agent._handle_summarize(
                cat,
                q,
                keyword=kw or None,
                allowed_paths=active_paths,
                session_id=session_id,
                prompt_language=user_lang,
                params=params,
            ):
                if ev.get("type") == "sources":
                    yield from emit_files_from_sources(
                        ev.get("content") or [],
                        total_matches=ev.get("total_matches"),
                        shown_count=ev.get("shown_count"),
                    )
                    continue
                if ev.get("type") == "text":
                    d = ev.get("content") or ev.get("delta") or ""
                    if d:
                        yield {"type": "text", "delta": d}
                        resp_text += d
                    continue
                if ev.get("type") == "done":
                    try:
                        hist_ref[-1]["a"] = resp_text
                    except Exception:
                        pass
                    yield _trace_done(query_type="summarize", answer_length=len(resp_text or ""))
                    yield {"type": "done", "ok": True, "query_type": "summarize", "sources": [], "trace": []}
                    return True
            try:
                hist_ref[-1]["a"] = resp_text
            except Exception:
                pass
            yield _trace_done(query_type="summarize", answer_length=len(resp_text or ""))
            yield {"type": "done", "ok": True, "query_type": "summarize", "sources": [], "trace": []}
            return True

        if action == "view_detail":
            idx = int(params.get("index") or 1)
            fname = str(params.get("file_name") or params.get("file") or "")
            yield from emit_status_fn("running", f"Viewing details: item #{idx}")
            for ev in self.agent._handle_view_detail(idx, fname, session_id=session_id, prompt_language=user_lang):
                if ev.get("type") == "sources":
                    yield from emit_files_from_sources(
                        ev.get("content") or [],
                        total_matches=ev.get("total_matches"),
                        shown_count=ev.get("shown_count"),
                    )
                    continue
                if ev.get("type") == "text":
                    d = ev.get("delta") or ev.get("content") or ""
                    if d:
                        yield {"type": "text", "delta": d}
                    continue
                if ev.get("type") == "done":
                    yield _trace_done(query_type="detail")
                    yield {"type": "done", "ok": True, "query_type": "detail", "sources": [], "trace": []}
                    return True
            yield _trace_done(query_type="detail")
            yield {"type": "done", "ok": True, "query_type": "detail", "sources": [], "trace": []}
            return True

        if action == "open_file":
            explicit_open_kws = {"打开", "开启", "启动", "open", "launch"}
            if not any(kw in q.lower() for kw in explicit_open_kws):
                fname = str(params.get("file_name") or params.get("file") or "")
                logger.info(
                    f"[open_file-guard] no explicit open keyword in '{q[:60]}', "
                    f"redirecting to search: '{fname or q}'"
                )
                return False

            fname = str(params.get("file_name") or params.get("file") or "")
            yield from emit_status_fn("running", f"Attempting to open file: {fname}")
            found_path = ""
            try:
                from tools.document_tools import search_files

                res_str = search_files(fname, limit=5)
                res = json.loads(res_str)
                if isinstance(res, dict) and res.get("files"):
                    found_path = res["files"][0].get("file_path", "")
            except Exception:
                pass

            if found_path:
                try:
                    from tools.file_management_tools import open_file

                    content_str = open_file(found_path)
                    icon = icon_type_for_path(found_path)
                    truncated = False

                    if icon == "image" and content_str.strip().startswith("{"):
                        try:
                            payload = json.loads(content_str)
                            if payload.get("kind") == "image":
                                content_str = payload.get("data_url", "")
                                truncated = bool(payload.get("truncated"))
                            elif payload.get("kind") == "image_too_large":
                                content_str = "[Image too large to preview]"
                                truncated = True
                        except Exception:
                            pass
                    elif icon != "image":
                        max_len = int(os.getenv("OPEN_FILE_MAX_CHARS", "60000"))
                        if len(content_str) > max_len:
                            content_str = content_str[:max_len]
                            truncated = True

                    yield {
                        "type": "opened_file",
                        "file": {
                            "file_name": os.path.basename(found_path),
                            "file_path": found_path,
                            "type": icon,
                            "iconType": icon,
                            "doc_category": "opened",
                            "doc_summary": "",
                        },
                        "content": content_str,
                        "truncated": truncated,
                    }
                    msg = f"已为您打开文件：{os.path.basename(found_path)}"
                    if user_lang == "en":
                        msg = f"Opened file: {os.path.basename(found_path)}"
                    yield {"type": "text", "delta": msg}
                    try:
                        hist_ref[-1]["a"] = msg
                    except Exception:
                        pass
                    yield _trace_done(query_type="open_file", answer_length=len(msg))
                except Exception as e:
                    emsg = f"打开文件失败：{e}" if user_lang == "zh" else f"Failed to open file: {e}"
                    yield {"type": "text", "delta": emsg}
            else:
                miss = (
                    f"未能在当前选中的范围内找到文件：{fname}"
                    if user_lang == "zh"
                    else f'Cannot find "{fname}" in the currently selected sources.'
                )
                yield {"type": "text", "delta": miss}

            yield _trace_done(query_type="open_file")
            yield {"type": "done", "ok": True, "query_type": "open_file", "sources": [], "trace": []}
            return True

        return False
