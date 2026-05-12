"""
Agent dispatch module — extracted from FileAgent for modularity.
"""
from __future__ import annotations
import os, re, time, json, uuid, gc
from typing import Any, Dict, List, Optional, Generator, Iterator

from utils.logger import get_logger
logger = get_logger()

from config.prompts import get_prompt, normalize_prompt_language
from core.handlers.context import HandlerContext


def query_stream_live(
    self,
    question: str,
    active_source_ids: Optional[List[str]] = None,
    active_paths: Optional[List[str]] = None,
    model_id: Optional[str] = None,
    files_preview_k: int = 5,
    session_id: Optional[str] = None,
    prompt_language: Optional[str] = None,
):
    import os
    logger.info("[Agent] mode=intent_dispatch (query_stream_live)")
    yield from self._query_stream_intent_dispatch(
        question,
        active_paths=active_paths,
        session_id=session_id,
        emit_status=True,
        prompt_language=prompt_language,
    )
    return

    try:
        from tools.file_management_tools import FILE_TOOLS_ROOT  # noqa: WPS433
    except Exception:
        FILE_TOOLS_ROOT = None

    tool_names = []
    try:
        for t in get_all_tools():
            n = getattr(t, "name", None)
            if n:
                tool_names.append(n)
    except Exception:
        tool_names = []

    sys_prompt = (
        "You are an intelligent file assistant Agent.\n"
        "Your job is to understand the request, plan tool usage, and complete the task.\n\n"
        "[Safety and Scope]\n"
    )
    
    if active_paths:
        sys_prompt += (
            "Important: you may access and retrieve files only from these directories (including subdirectories):\n"
            f"{json.dumps(active_paths, ensure_ascii=False, indent=2)}\n"
            "Unless the user explicitly provides another absolute path, do not access directories outside this list.\n\n"
        )
    else:
        sys_prompt += "**Warning**: no data source is selected. Ask the user to select files or folders first.\n\n"

    sys_prompt += (
        "[Core Rules]\n"
        "1. **Understand file-type intent precisely**:\n"
        "   - For requests like 'which database files' or 'all images', use `count_documents_files` with `file_extensions` (for example .db,.sql,.sqlite or .jpg,.png).\n"
        "   - Do not rely on filename contains checks such as 'database'.\n"
        "   - Common database suffixes: .db, .sqlite, .sqlite3, .sql, .csv, .parquet, .mdb, .accdb\n"
        "2. **Prefer indexed knowledge tools first**: for content/topic/summary questions, use `search_documents` or `count_documents`.\n"
        "2.1 **Hybrid retrieval (filename + content)**:\n"
        "   - For requests like 'find / related materials / what files' with person/company/term keywords (for example 'Person Name product keyword'), prioritize `search_documents(query=...)`.\n"
        "   - If the user explicitly asks for file path/location, then use `search_files(keyword=...)` to complement retrieval.\n"
        "   - Do not conclude 'not found' with only `search_files`; many matches come from file content.\n"
        "3. **No disk search for retrieval**: never use physical disk grep/traversal for search. Use indexed DB tools only: `search_documents`, `search_files`, `count_documents_files`.\n"
        "4. **Pick the right tool family**: file ops use read/write/delete/move/copy; knowledge queries use search/count.\n"
        "5. **Be transparent**: briefly explain the plan before calling tools.\n"
        "6. **Default file/folder listing within selected Sources**:\n"
        "   - When the user asks 'what files do I have' without an explicit absolute path, treat it as selected Sources scope.\n"
        "   - `list_directory` default should return selected Sources list, not the user's home root.\n"
        "7. **For 'all files/how many files' questions, query the index**:\n"
        "   - Use `count_documents_files` without filters for total and distribution.\n"
        "   - Do not replace this with disk directory traversal via `list_directory`.\n"
        "8. **Open-file requests must really open/read**:\n"
        "   - For explicit 'open this file/show original content' requests, prioritize `open_file(file_path=...)`.\n"
        "   - If only filename is provided, first resolve real path using `search_files(keyword=filename)`, then call `open_file`.\n"
        "   - Do not replace open-file with summary via `search_documents`.\n"
        "9. **Path/location questions must use real indexed paths first**:\n"
        "   - For 'where/path/which folder' requests, first call `search_files` to get real `file_path` values, then answer.\n"
        "   - If multiple file names are provided, run multiple `search_files` calls and list all hits (including duplicates with same name).\n"
        "   - Never guess paths (for example ~/Documents or ~/Downloads). If not found, explicitly say 'not found within indexed scope'.\n"
        "   - Do not replace `search_files` with `list_directory`/physical disk search.\n"
        "10. **Follow-up QA on currently opened file**:\n"
        "   - For requests like summarize/key points/answer based on this file, first call `get_opened_file_text(file_path=...)` and then answer.\n"
    )
    if FILE_TOOLS_ROOT:
        sys_prompt += f"\n[File Tool Safety Root]\n{FILE_TOOLS_ROOT}\n"
    if tool_names:
        sys_prompt += "\n[Available Tools]\n- " + "\n- ".join(sorted(set(tool_names))) + "\n"

    history_ref = self._get_history_ref(session_id)
    history_dicts = [{"q": h.get("q", ""), "a": h.get("a", "")} for h in history_ref[-5:]]
    messages = build_messages_for_model(sys_prompt, history_dicts, question)

    tools = get_all_tools()
    if not tools:
        tools = [count_documents, search_documents]
    
    llm = get_llm(streaming=False).bind_tools(tools)
    llm_plain = get_llm(streaming=False)

    trace: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    steps = 0
    MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "12"))
    final_answer = ""
    executed_tool_sigs: set = set()
    compressed_once = False
    force_no_tools = False

    def emit_status(phase: str, message: str, tool: Optional[str] = None):
        ev = {"type": "status", "phase": phase, "message": message}
        if tool:
            ev["tool"] = tool
        return ev

    def handle_ui_side_effects(tool_name: str, result_str: str, tool_args: dict):

        supported_tools = {
            "count_documents_files", "search_documents", "search_files",
            "get_opened_file_text",
            "count_documents", "summarize_topics", "get_document_content",
            "list_directory",
            "read_file",
            "open_file",
        }
        if tool_name in supported_tools:
            try:
                import json
                import re
                
                if tool_name in ("read_file", "open_file"):
                    fp = (tool_args.get("file_path") or "").strip()
                    fname = os.path.basename(fp) if fp else "unknown"
                    icon = "doc"
                    if fp:
                        ext = os.path.splitext(fp.lower())[1].lstrip(".")
                        if ext == "pdf":
                            icon = "pdf"
                        elif ext in {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"}:
                            icon = "image"
                        elif ext in {"xls", "xlsx", "csv"}:
                            icon = "sheet"
                    max_len = int(os.getenv("OPEN_FILE_MAX_CHARS", "60000"))
                    content = result_str or ""
                    truncated = False
                    if icon == "image" and isinstance(content, str) and content.strip().startswith("{"):
                        try:
                            payload = json.loads(content)
                            if isinstance(payload, dict) and payload.get("kind") == "image" and payload.get("data_url"):
                                content = str(payload.get("data_url") or "")
                                truncated = bool(payload.get("truncated"))
                            elif isinstance(payload, dict) and payload.get("kind") == "image_too_large":
                                sz = payload.get("size_bytes")
                                mx = payload.get("max_bytes")
                                content = (
                                    f"[Image too large to preview]\n"
                                    f"Path: {payload.get('file_path','')}\n"
                                    f"Size: {sz} bytes\n"
                                    f"Max preview: {mx} bytes\n"
                                    f"Tip: open it in system viewer or reduce size."
                                )
                                truncated = True
                        except Exception:
                            pass
                    if icon != "image":
                        if len(content) > max_len:
                            content = content[:max_len]
                            truncated = True
                    return {
                        "type": "opened_file",
                        "file": {
                            "file_name": fname,
                            "file_path": fp,
                            "type": icon,
                            "iconType": icon,
                            "doc_category": "opened",
                            "doc_summary": "",
                        },
                        "content": content,
                        "truncated": truncated,
                    }

                files = []
                
                if result_str.strip().startswith("{") or result_str.strip().startswith("["):
                    try:
                        payload = json.loads(result_str)
                        if isinstance(payload, dict):
                            files = payload.get("files", []) or []
                        elif isinstance(payload, list):
                            files = payload
                    except Exception:
                        pass
                
                if not files and tool_name == "list_directory":
                    lines = result_str.split('\n')
                    dir_path = tool_args.get("directory_path", ".")
                    if dir_path.startswith("~"):
                        dir_path = os.path.expanduser(dir_path)
                        
                    count_added = 0
                    for line in lines:
                        line = line.strip()
                        if not line or count_added >= 100: continue
                        
                        if "Error" in line or "Exception" in line or "[NO_MATCHES]" in line: continue
                        
                        full_path = os.path.join(dir_path, line)
                        
                        ftype = "doc"
                        if tool_name == "list_directory":
                            if not os.path.exists(full_path):
                                ftype = "folder" if "." not in line else "doc"
                            else:
                                ftype = "folder" if os.path.isdir(full_path) else "doc"
                        
                        files.append({
                            "file_name": line,
                            "file_path": full_path,
                            "type": ftype,
                            "iconType": ftype,
                            "doc_category": "文件浏览",
                            "doc_summary": f"位于 {dir_path}"
                        })
                        count_added += 1

                if not files:
                    pattern = r'(?:(?:\d+\.|【\d+】)\s*)([^\n]+?\.(?:pdf|docx?|xlsx?|pptx?|txt|md|csv))'
                    matches = re.findall(pattern, result_str, re.IGNORECASE)
                    if matches:
                        seen = set()
                        for m in matches[:50]:
                            fname = m.strip()
                            if fname and fname not in seen:
                                seen.add(fname)
                                files.append({
                                    "file_name": fname,
                                    "file_path": fname,
                                    "type": "doc",
                                    "iconType": "doc"
                                })
                
                if files and isinstance(files, list):
                    preview = files[: max(1, int(files_preview_k))]
                    sources.extend(files)
                    return {"type": "files", "total": len(files), "preview": preview, "all": files}
            except Exception as e:
                logger.error(f"[Agent] handle_ui_side_effects error: {e}")
        return None

    # ==================== Two-pass Planner + DAG Executor (primary path) ====================
    def _extract_json_from_text(text: str) -> Optional[str]:
        if not text:
            return None
        s = text.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            return s
        try:
            a = s.find("{")
            b = s.rfind("}")
            if a != -1 and b != -1 and b > a:
                return s[a : b + 1]
        except Exception:
            pass
        return None

    def _validate_plan(plan: dict, available_tool_names: List[str]) -> Optional[str]:

        if not isinstance(plan, dict):
            return "plan is not a JSON object"
        if "steps" not in plan or not isinstance(plan.get("steps"), list):
            return "missing steps[]"
        max_steps = int(os.getenv("TOOL_PLANNER_MAX_STEPS", "8"))
        if len(plan["steps"]) <= 0:
            intent = str(plan.get("intent") or "").strip().lower()
            if intent == "other":
                return None
            return "empty steps"
        if len(plan["steps"]) > max_steps:
            return f"too many steps: {len(plan['steps'])} > {max_steps}"
        seen_ids = set()
        for st in plan["steps"]:
            if not isinstance(st, dict):
                return "step is not an object"
            sid = str(st.get("id") or "").strip()
            tool = str(st.get("tool") or "").strip()
            args = st.get("args", {})
            deps = st.get("depends_on", []) or []
            if not sid:
                return "step missing id"
            if sid in seen_ids:
                return f"duplicate step id: {sid}"
            seen_ids.add(sid)
            if not tool:
                return f"step {sid} missing tool"
            if tool not in set(available_tool_names):
                return f"unknown tool in step {sid}: {tool}"
            if not isinstance(args, (dict, list)):
                return f"invalid args for step {sid}: must be object/array"
            if deps and (not isinstance(deps, list) or any(not str(x).strip() for x in deps)):
                return f"invalid depends_on for step {sid}"
        for st in plan["steps"]:
            sid = str(st.get("id") or "").strip()
            deps = st.get("depends_on", []) or []
            for d in deps:
                if d not in seen_ids:
                    return f"step {sid} depends_on missing id: {d}"
        return None

    def _topo_layers(steps_list: List[dict]) -> List[List[dict]]:
        # build maps
        by_id = {str(s.get("id")): s for s in steps_list}
        indeg: Dict[str, int] = {}
        outs: Dict[str, List[str]] = {}
        for sid, st in by_id.items():
            indeg[sid] = 0
            outs[sid] = []
        for sid, st in by_id.items():
            deps = st.get("depends_on", []) or []
            for d in deps:
                indeg[sid] += 1
                outs[d].append(sid)
        layers: List[List[dict]] = []
        ready = [sid for sid, k in indeg.items() if k == 0]
        while ready:
            layer_ids = list(ready)
            ready = []
            layer_steps = [by_id[i] for i in layer_ids]
            layers.append(layer_steps)
            for u in layer_ids:
                for v in outs.get(u, []):
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        ready.append(v)
        if any(k > 0 for k in indeg.values()):
            raise ValueError("plan has cycle in depends_on")
        return layers

    def _jsonpath_get(obj: Any, path: str) -> Any:
        cur = obj
        p = path.strip()
        if not p:
            return cur
        # split by '.' but keep bracket parts
        parts: List[str] = []
        buf = ""
        i = 0
        while i < len(p):
            ch = p[i]
            if ch == ".":
                if buf:
                    parts.append(buf)
                    buf = ""
                i += 1
                continue
            buf += ch
            i += 1
        if buf:
            parts.append(buf)

        for part in parts:
            # handle bracket(s)
            name = part
            idxs: List[int] = []
            if "[" in part and part.endswith("]"):
                # e.g. files[0][1]
                base = part.split("[", 1)[0]
                name = base
                rest = part[len(base):]
                # parse all [n]
                j = 0
                num = ""
                while j < len(rest):
                    if rest[j] == "[":
                        num = ""
                        j += 1
                        while j < len(rest) and rest[j] != "]":
                            num += rest[j]
                            j += 1
                        try:
                            idxs.append(int(num))
                        except Exception:
                            idxs.append(0)
                    j += 1
            if name:
                if isinstance(cur, dict):
                    cur = cur.get(name)
                else:
                    return None
            for k in idxs:
                if isinstance(cur, list) and 0 <= k < len(cur):
                    cur = cur[k]
                else:
                    return None
        return cur

    def _resolve_placeholders(args_obj: Any, outputs: Dict[str, Any]) -> Any:
        import re

        if isinstance(args_obj, dict):
            return {k: _resolve_placeholders(v, outputs) for k, v in args_obj.items()}
        if isinstance(args_obj, list):
            return [_resolve_placeholders(v, outputs) for v in args_obj]
        if isinstance(args_obj, str):
            pat_full = r"^\{\{\s*([a-zA-Z0-9_\-]+)\.([^\}]+?)\s*\}\}$"
            mfull = re.match(pat_full, args_obj.strip())
            if mfull:
                sid = mfull.group(1)
                pth = mfull.group(2)
                return _jsonpath_get(outputs.get(sid), pth)

            pat = r"\{\{\s*([a-zA-Z0-9_\-]+)\.([^\}]+?)\s*\}\}"
            def _rep(m):
                sid = m.group(1)
                pth = m.group(2)
                val = _jsonpath_get(outputs.get(sid), pth)
                if val is None:
                    return ""
                if isinstance(val, (dict, list)):
                    try:
                        return json.dumps(val, ensure_ascii=False)
                    except Exception:
                        return str(val)
                return str(val)
            return re.sub(pat, _rep, args_obj)
        return args_obj

    def _summarize_step_result(tool_name: str, result_str: str, result_json: Any) -> str:

        max_chars = int(os.getenv("STEP_RESULT_MAX_CHARS", "4000"))
        try:
            if isinstance(result_json, dict):
                if "files" in result_json and isinstance(result_json.get("files"), list):
                    files = result_json.get("files") or []
                    preview = []
                    for f in files[:20]:
                        if isinstance(f, dict):
                            preview.append({"file_name": f.get("file_name", ""), "file_path": f.get("file_path", "")})
                    return json.dumps(
                        {"count": result_json.get("count") or result_json.get("total") or len(files), "files_preview": preview},
                        ensure_ascii=False,
                    )[:max_chars]
                return json.dumps(result_json, ensure_ascii=False)[:max_chars]
        except Exception:
            pass
        s = (result_str or "").strip()
        if len(s) > max_chars:
            return s[:max_chars] + "...(truncated)"
        return s

    # ==================== Debug logging（planner/tool args/mid states） ====================
    _DEBUG_AGENT = os.getenv("AGENT_DEBUG_LOGS", "true").lower() in {"1", "true", "yes", "on"}
    _DEBUG_MAX_CHARS = int(os.getenv("AGENT_DEBUG_MAX_CHARS", "2000"))

    def _clip(v: Any, max_chars: int = None) -> str:
        mc = _DEBUG_MAX_CHARS if max_chars is None else int(max_chars)
        try:
            if isinstance(v, (dict, list)):
                s = json.dumps(v, ensure_ascii=False, indent=2)  # readable for logs
            else:
                s = str(v)
        except Exception:
            s = str(v)
        if len(s) > mc:
            return s[:mc] + f"...(truncated, total={len(s)})"
        return s

    def _dlog(title: str, payload: Any = None):
        if not _DEBUG_AGENT:
            return
        try:
            if payload is None:
                logger.info(f"[AgentDebug] {title}")
            else:
                logger.info(f"[AgentDebug] {title}\n{_clip(payload)}")
        except Exception:
            pass

    def _run_tool_once(tool_name: str, tool_args: dict) -> Dict[str, Any]:
        tool_obj = get_tool(tool_name)
        if tool_obj is None:
            return {"ok": False, "error": f"Unknown tool: {tool_name}", "result_str": "", "result_json": None, "ui_event": None}
        try:
            _dlog("tool_execute.start", {"tool": tool_name, "args": tool_args})
            if hasattr(tool_obj, "invoke"):
                res = tool_obj.invoke(tool_args)
            elif hasattr(tool_obj, "run"):
                res = tool_obj.run(**tool_args)
            else:
                res = tool_obj(**tool_args)
            result_str = str(res)
        except Exception as e:
            _dlog("tool_execute.error", {"tool": tool_name, "args": tool_args, "error": str(e)})
            return {"ok": False, "error": f"Error executing {tool_name}: {e}", "result_str": "", "result_json": None, "ui_event": None}
        result_json = None
        if isinstance(result_str, str) and result_str.strip().startswith("{"):
            try:
                result_json = json.loads(result_str)
            except Exception:
                result_json = None
        ui_event = None
        try:
            ui_event = handle_ui_side_effects(tool_name, result_str, tool_args)
        except Exception:
            ui_event = None
        _dlog(
            "tool_execute.done",
            {
                "tool": tool_name,
                "ok": True,
                "result_len": len(result_str or ""),
                "result_preview": (result_str or "")[:300],
                "parsed_json": isinstance(result_json, (dict, list)),
            },
        )
        return {"ok": True, "result_str": result_str, "result_json": result_json, "ui_event": ui_event, "error": None}

    def _should_skip_planner(q: str) -> bool:
        s = (q or "").strip()
        if not s:
            return True

        task_words = [
            "查", "找", "搜索", "多少", "哪些", "统计", "路径", "打开", "索引", "删除", "移动", "复制", "总结", "运行", "执行",
            "find", "search", "count", "how many", "which", "path", "open", "index", "delete", "move", "copy", "summarize", "run", "execute",
        ]
        if any(w in s for w in task_words):
            return False

        try:
            if len(s) <= 4:
                analyzed = self._analyze_query_intent(s, session_id=session_id)
                if isinstance(analyzed, dict) and analyzed.get("clear") is False:
                    return True
        except Exception:
            pass

        return False

    planner_enabled = (os.getenv("ENABLE_TOOL_PLANNER", "true").lower() in {"1", "true", "yes", "on"}) and (not _should_skip_planner(question))
    if planner_enabled:
        try:
            # ========== Planner step ==========
            yield emit_status("thinking", "正在生成执行计划...")
            category_info = self._get_category_stats(prompt_language="en")
            planner_prompt = (
                "You are a tool orchestration planner. Please output an executable JSON plan for the user's question (do not output any explanatory text).\n\n"
                "Requirements:\n"
                "1) You may arrange multiple tool steps in one plan.\n"
                "2) Express dependencies with depends_on; maximize parallelism when there is no dependency.\n"
                "3) args must be a JSON object or array.\n"
                "4) You may reference previous step outputs with placeholders like {{stepId.path}}, for example {{find1.files[0].file_path}}.\n"
                "5) Retrieval must use index/database tools only (search_documents/search_files/count_documents_files, etc.); do not plan disk traversal or physical file_search.\n"
                "6) Shell command execution is not available in public builds; call only tools listed under Available tools.\n"
                "7) If a step is destructive (write/delete/move/copy), set dangerous=true on that step.\n"
                "8) If the question is capability intro / tool explanation / small talk and no tools are needed, set intent to other and allow steps to be [].\n"
                "9) **When a tool requires category (for example search_documents), choose category strictly from the real categories below:**\n"
                f"{category_info}\n\n"
                f"Available tools: {', '.join(sorted(set(tool_names)))}\n\n"
                "Output strict JSON with this schema:\n"
                "{\n"
                '  \"intent\": \"search|path|open_file|summarize|count|other\",\n'
                "  \"steps\": [\n"
                "    {\"id\":\"s1\",\"tool\":\"search_documents\",\"args\":{\"query\":\"...\", \"category\": \"resume\"},\"depends_on\":[]},\n"
                "    {\"id\":\"s2\",\"tool\":\"search_files\",\"args\":{\"keyword\":\"...\",\"limit\":20},\"depends_on\":[]}\n"
                "  ]\n"
                "}\n\n"
                f"User question: {question}\n"
            )
            supports_system_now = _model_supports_system_role()
            if supports_system_now:
                planner_messages = [SystemMessage(content=planner_prompt), HumanMessage(content=question)]
            else:
                planner_messages = [HumanMessage(content=f"[Instructions]\n{planner_prompt}\n\n[User Question]\n{question}")]

            planner_resp = llm_plain.invoke(planner_messages)
            raw = (getattr(planner_resp, "content", "") or "").strip()
            _dlog("planner.raw_model_output", raw)
            json_text = _extract_json_from_text(raw) or raw
            _dlog("planner.extracted_json_text", json_text)
            plan_obj = json.loads(json_text)
            _dlog("planner.plan_obj", plan_obj)

            err = _validate_plan(plan_obj, tool_names)
            if err:
                _dlog("planner.plan_invalid", {"error": err, "plan_obj": plan_obj})
                raise ValueError(f"planner plan invalid: {err}")

            try:
                intent_now = str(plan_obj.get("intent") or "").strip().lower()
                steps_now = plan_obj.get("steps") or []
            except Exception:
                intent_now = ""
                steps_now = []
            if intent_now == "other" and isinstance(steps_now, list) and len(steps_now) == 0:
                yield emit_status("thinking", "正在回答...")
                final_answer = ""
                for chunk in llm_plain.stream([HumanMessage(content=question)]):
                    c = getattr(chunk, "content", "") or ""
                    if isinstance(c, str) and c:
                        yield {"type": "text", "delta": c}
                        final_answer += c
                _dlog("reducer.output_final_answer", final_answer)
                try:
                    if question and final_answer:
                        history_ref.append({"q": question, "a": final_answer})
                        if len(history_ref) > self.max_history:
                            del history_ref[:-self.max_history]
                except Exception:
                    pass
                yield {"type": "done", "ok": True, "query_type": "agent", "sources": sources, "trace": trace}
                return

            trace_plan = {"type": "plan", "title": "执行计划", "preview": (plan_obj.get("intent") or "")[:80], "plan": plan_obj}
            trace.append(trace_plan)
            yield {"type": "trace_append", "item": trace_plan}

            # ========== Execute DAG ==========
            yield emit_status("running", "正在执行计划...")
            max_parallel_tools = int(os.getenv("MAX_PARALLEL_TOOLS", "4"))
            dangerous_tools = {"write_file", "delete_file", "move_file", "copy_file"}

            import threading
            import concurrent.futures

            sem_tools = threading.Semaphore(max(1, max_parallel_tools))

            step_outputs: Dict[str, Any] = {}
            step_summaries: Dict[str, str] = {}
            step_status: Dict[str, str] = {}
            executed_sigs: set = set()

            layers = _topo_layers(plan_obj["steps"])
            _dlog("planner.topo_layers", {"layers": [[str(s.get("id")) for s in layer] for layer in layers]})
            for layer_idx, layer in enumerate(layers):
                _dlog("planner.execute.layer_start", {"layer_idx": layer_idx, "steps": [str(s.get("id")) for s in layer]})
                runnable: List[dict] = []
                for st in layer:
                    sid = str(st.get("id"))
                    tool = str(st.get("tool"))
                    args = st.get("args", {})
                    deps = st.get("depends_on", []) or []
                    dep_failed = False
                    for d in deps:
                        ds = step_status.get(str(d))
                        if ds and ds not in {"ok", "skipped_duplicate"}:
                            dep_failed = True
                            break
                    if dep_failed:
                        step_status[sid] = "skipped_due_to_dep"
                        step_outputs[sid] = {"ok": False, "skipped": True, "reason": "dependency failed"}
                        step_summaries[sid] = "skipped due to dependency failure"
                        continue
                    # resolve placeholders using finished outputs
                    try:
                        resolved_args = _resolve_placeholders(args, step_outputs)
                        if isinstance(resolved_args, str):
                            try:
                                resolved_args = json.loads(resolved_args)
                            except Exception:
                                pass
                        st["_resolved_args"] = resolved_args
                    except Exception:
                        st["_resolved_args"] = args
                    _dlog(
                        "planner.step.resolved_args",
                        {"id": sid, "tool": tool, "depends_on": deps, "args_raw": args, "args_resolved": st.get("_resolved_args")},
                    )

                    # dangerous protection
                    if tool in dangerous_tools and not bool(st.get("dangerous", False)):
                        step_status[sid] = "blocked_dangerous"
                        step_outputs[sid] = {"ok": False, "error": "dangerous tool requires dangerous=true"}
                        step_summaries[sid] = "blocked: dangerous tool requires dangerous=true"
                        _dlog("planner.step.blocked_dangerous", {"id": sid, "tool": tool})
                        continue

                    # de-dupe
                    try:
                        sig = f"{tool}:{json.dumps(st.get('_resolved_args', {}), ensure_ascii=False, sort_keys=True)}"
                    except Exception:
                        sig = f"{tool}:{str(st.get('_resolved_args', {}))}"
                    if sig in executed_sigs:
                        step_status[sid] = "skipped_duplicate"
                        step_outputs[sid] = {"ok": True, "skipped": True, "reason": "duplicate tool+args"}
                        step_summaries[sid] = "skipped duplicate tool+args"
                        _dlog("planner.step.skipped_duplicate", {"id": sid, "tool": tool, "sig": sig})
                        continue
                    executed_sigs.add(sig)
                    runnable.append(st)

                if not runnable:
                    continue

                # Submit runnable steps
                futures = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_tools) as ex:
                    for st in runnable:
                        sid = str(st.get("id"))
                        tool = str(st.get("tool"))
                        args = st.get("_resolved_args", {})

                        yield emit_status("running", f"执行: {tool}", tool=tool)
                        trace_item = {"type": "plan", "title": f"Step {sid}", "preview": f"{tool}", "tool_calls": [{"name": tool, "args": args}]}
                        trace.append(trace_item)
                        yield {"type": "trace_append", "item": trace_item}

                        def _task(tool_name=tool, tool_args=args):
                            sem_tools.acquire()
                            try:
                                return _run_tool_once(tool_name, tool_args)
                            finally:
                                sem_tools.release()

                        futures[ex.submit(_task)] = st

                    for fut in concurrent.futures.as_completed(list(futures.keys())):
                        st = futures[fut]
                        sid = str(st.get("id"))
                        tool = str(st.get("tool"))
                        args = st.get("_resolved_args", {})
                        try:
                            r = fut.result()
                        except Exception as e:
                            r = {"ok": False, "error": str(e), "result_str": "", "result_json": None, "ui_event": None}

                        step_status[sid] = "ok" if r.get("ok") else "error"
                        step_outputs[sid] = r.get("result_json") if r.get("result_json") is not None else {"ok": r.get("ok"), "result": r.get("result_str"), "error": r.get("error")}
                        step_summaries[sid] = _summarize_step_result(tool, r.get("result_str", ""), r.get("result_json"))
                        _dlog(
                            "planner.step.done",
                            {
                                "id": sid,
                                "tool": tool,
                                "args": args,
                                "status": step_status.get(sid),
                                "result_len": len((r.get("result_str") or "")),
                                "error": r.get("error"),
                                "summary_preview": (step_summaries.get(sid) or "")[:400],
                            },
                        )

                        # UI events
                        ui_event = r.get("ui_event")
                        if ui_event:
                            yield ui_event

                        # Trace tool result
                        trace_tool = {
                            "type": "tool",
                            "tool": tool,
                            "args": args,
                            "result_preview": (r.get("result_str") or "")[:500] + ("..." if len((r.get("result_str") or "")) > 500 else ""),
                            "ok": bool(r.get("ok")),
                            "error": r.get("error"),
                            "step_id": sid,
                        }
                        trace.append(trace_tool)
                        yield {"type": "trace_append", "item": trace_tool}

            # ========== Reduce: final answer ==========
            user_is_zh = (
                normalize_prompt_language(prompt_language, fallback="en") == "zh"
                or bool(re.search(r"[\u4e00-\u9fff]", question or ""))
            )
            yield emit_status("thinking", "正在汇总结果..." if user_is_zh else "Summarizing results...")
            reduce_payload = {
                "question": question,
                "intent": plan_obj.get("intent", "other"),
                "steps": [
                    {
                        "id": str(s.get("id")),
                        "tool": str(s.get("tool")),
                        "depends_on": s.get("depends_on", []) or [],
                        "status": step_status.get(str(s.get("id")), "unknown"),
                        "summary": step_summaries.get(str(s.get("id")), ""),
                    }
                    for s in plan_obj.get("steps", [])
                ],
            }
            _dlog("reducer.input_payload", reduce_payload)
            reducer_sys = (
                "You are a tool execution result summarizer. Please output the final answer based on the user's question and the summaries of each step.\n"
                "Requirements:\n"
                "- Detect the user's language from <UserQuestion>. If the user wrote Chinese, answer in Chinese. If the user wrote English, answer in English. Follow any explicit language request from the user.\n"
                "- Lead with the conclusion, then give supporting evidence.\n"
                "- Do not paste long file lists because the UI displays them separately.\n"
                "- If any step failed or was blocked, explain the reason and a practical alternative.\n"
            )
            reducer_question = f"<UserQuestion>\n{question}\n</UserQuestion>\n\n<PlanAndSummaries>\n{json.dumps(reduce_payload, ensure_ascii=False, indent=2)}\n</PlanAndSummaries>\n"
            supports_system_now = _model_supports_system_role()
            if supports_system_now:
                reducer_messages = [SystemMessage(content=reducer_sys), HumanMessage(content=reducer_question)]
            else:
                reducer_messages = [HumanMessage(content=f"[Instructions]\n{reducer_sys}\n\n[User Question]\n{reducer_question}")]

            final_answer = ""
            for chunk in llm_plain.stream(reducer_messages):
                c = getattr(chunk, "content", "") or ""
                if isinstance(c, str) and c:
                    yield {"type": "text", "delta": c}
                    final_answer += c
            _dlog("reducer.output_final_answer", final_answer)

            # save history
            try:
                if question and final_answer:
                    history_ref.append({"q": question, "a": final_answer})
                    if len(history_ref) > self.max_history:
                        del history_ref[:-self.max_history]
            except Exception:
                pass

            yield {"type": "done", "ok": True, "query_type": "agent", "sources": sources, "trace": trace}
            return
        except Exception as e:
            logger.warning(f"[Agent] Planner/DAG fallback due to: {e}")
            trace.append({"type": "error", "title": "planner_fallback", "preview": str(e)[:200]})
            yield {"type": "trace_append", "item": {"type": "error", "title": "planner_fallback", "preview": str(e)[:200]}}

    # 3) Agent Loop
    supports_system = _model_supports_system_role()
    
    while steps < MAX_TOOL_STEPS:
        yield emit_status("thinking", "正在思考..." if steps == 0 else "正在分析工具结果...")
        
        try:
            MAX_CTX_TOKENS = int(os.getenv("LOCAL_LLM_CTX_TOKENS", "32768"))
            PROMPT_BUDGET_TOKENS = int(os.getenv("LOCAL_LLM_PROMPT_BUDGET_TOKENS", str(int(MAX_CTX_TOKENS * 0.75))))
            TOOL_CHUNK_CHARS = int(os.getenv("LOCAL_LLM_TOOL_CHUNK_CHARS", "24000"))
            TOOL_CHUNK_SUMMARY_CHARS = int(os.getenv("LOCAL_LLM_TOOL_CHUNK_SUMMARY_CHARS", "2200"))
            TOOL_DIGEST_MAX_CHARS = int(os.getenv("LOCAL_LLM_TOOL_DIGEST_MAX_CHARS", "9000"))

            approx_tokens = _approx_tokens_from_messages(messages)
            if (not compressed_once) and approx_tokens > PROMPT_BUDGET_TOKENS:
                tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
                if tool_msgs:
                    summarizer = get_llm(streaming=False)
                    tool_text = "\n\n".join([(m.content or "") for m in tool_msgs])
                    # Map
                    chunks = _chunk_text_by_newlines(tool_text, TOOL_CHUNK_CHARS)
                    chunk_summaries: List[str] = []
                    for i, ch in enumerate(chunks):
                        yield emit_status("thinking", f"工具结果过长，正在分段整理 {i+1}/{len(chunks)}...")
                        prompt = (
                            "You are consolidating long tool outputs. Based only on the chunk below, extract information most relevant to the user's question.\n\n"
                            f"<User Question>\n{question}\n</User Question>\n\n"
                            f"<Tool Output Chunk {i+1}/{len(chunks)}>\n{ch}\n</Tool Output Chunk>\n\n"
                            "Output requirements:\n"
                            "- Output in English.\n"
                            "- Keep key numbers, file names/paths, categories, and conclusions.\n"
                            "- Do not repeat long raw passages.\n"
                            f"- Keep within {TOOL_CHUNK_SUMMARY_CHARS} characters.\n"
                        )
                        try:
                            resp = summarizer.invoke([HumanMessage(content=prompt)])
                            s = (resp.content or "").strip()
                        except Exception as e:
                            s = f"[chunk_summarization_failed:{e}]"
                        if s:
                            chunk_summaries.append(s)

                    # Reduce
                    yield emit_status("thinking", "正在合并分段整理结果...")
                    reduce_prompt = (
                        "Below are chunk summaries generated from long tool outputs. Merge them into one concise 'tool result digest' for downstream answering/planning.\n\n"
                        f"<User Question>\n{question}\n</User Question>\n\n"
                        f"<Chunk Summaries>\n" + "\n\n---\n\n".join(chunk_summaries) + "\n</Chunk Summaries>\n\n"
                        "Merge requirements:\n"
                        "- Deduplicate overlapping points and keep key numbers, file names/paths, categories, and conclusions.\n"
                        "- If still too long, prioritize the parts most relevant to the question and mention that deeper file-level drill-down is possible.\n"
                        f"- Keep the final digest within {TOOL_DIGEST_MAX_CHARS} characters.\n"
                    )
                    try:
                        digest_resp = summarizer.invoke([HumanMessage(content=reduce_prompt)])
                        digest = (digest_resp.content or "").strip()
                    except Exception as e:
                        digest = f"[digest_merge_failed:{e}]"

                    messages = [m for m in messages if not isinstance(m, ToolMessage)]
                    messages.append(
                        ToolMessage(
                            content=f"[COMPRESSED_TOOL_RESULTS]\n{digest}",
                            tool_call_id="tool_digest",
                        )
                    )
                    compressed_once = True
                    force_no_tools = True
                    messages.append(
                        HumanMessage(
                            content=(
                                "[Important]\n"
                                "Tool outputs were chunked and merged into COMPRESSED_TOOL_RESULTS.\n"
                                "Now answer the user directly based on this digest and provide conclusion/suggestions.\n"
                                "Do not call any more tools to avoid repeated retrieval and context overflow."
                            )
                        )
                    )

            messages_to_send = messages
            if not supports_system:
                messages_to_send = convert_messages_for_gemma(messages)
            
            llm_to_stream = llm_plain if force_no_tools else llm

            response = None
            # Buffer for merging chunks
            # LangChain's AIMessageChunk supports addition to merge content and tool_calls
            for chunk in llm_to_stream.stream(messages_to_send):
                if response is None:
                    response = chunk
                else:
                    response += chunk
                
                if chunk.content:
                    content_str = ""
                    if isinstance(chunk.content, str):
                        content_str = chunk.content
                    elif isinstance(chunk.content, list):
                        # Handle multimodal content list if necessary
                        for item in chunk.content:
                            if isinstance(item, str):
                                content_str += item
                            elif isinstance(item, dict) and item.get("type") == "text":
                                content_str += item.get("text", "")
                    
                    if content_str:
                        yield {"type": "text", "delta": content_str}
                        final_answer += content_str
                        
        except Exception as e:
            yield {"type": "text", "delta": f"调用模型出错: {e}"}
            break

        if not response:
            break
            
        ai_content = getattr(response, "content", "") or ""
        tool_calls = getattr(response, "tool_calls", None) or []
        _dlog(
            "model.mid_state",
            {
                "steps": steps,
                "force_no_tools": bool(force_no_tools),
                "ai_content_preview": (ai_content or "")[:400],
                "tool_calls": tool_calls,
            },
        )

        if force_no_tools:
            break

        if not tool_calls:
            break

        plan_item = {
            "type": "plan",
            "title": "调用工具",
            "preview": "、".join([tc.get("name", "") for tc in tool_calls])[:160],
            "tool_calls": tool_calls,
        }
        trace.append(plan_item)
        yield {"type": "trace_append", "item": plan_item}

        messages.append(response)

        for tc in tool_calls:
            t_name = tc.get("name", "")
            t_args = tc.get("args", {}) or {}
            t_id = tc.get("id", "")

            try:
                import json as _json
                sig = f"{t_name}:{_json.dumps(t_args, ensure_ascii=False, sort_keys=True)}"
            except Exception:
                sig = f"{t_name}:{str(t_args)}"
            if sig in executed_tool_sigs:
                messages.append(
                    ToolMessage(
                        content=f"[Tool:{t_name}]\n[SKIPPED_REPEAT] 相同参数的工具调用已执行过一次，为避免死循环与上下文超限，本次跳过。请基于已有结果直接回答。",
                        tool_call_id=t_id or "skipped_repeat",
                    )
                )
                force_no_tools = True
                continue
            executed_tool_sigs.add(sig)
            
            yield emit_status("running", f"执行: {t_name}", tool=t_name)
            
            tool_obj = get_tool(t_name)
            result_str = ""
            
            if tool_obj is None:
                result_str = f"Error: Tool '{t_name}' not found."
            else:
                try:
                    if hasattr(tool_obj, "invoke"):
                        res = tool_obj.invoke(t_args)
                    elif hasattr(tool_obj, "run"):
                        res = tool_obj.run(**t_args)
                    else:
                        res = tool_obj(**t_args)
                    result_str = str(res)
                except Exception as e:
                    result_str = f"Error executing {t_name}: {e}"

            ui_event = handle_ui_side_effects(t_name, result_str, t_args)
            if ui_event:
                yield ui_event

            trace_item = {
                "type": "tool",
                "tool": t_name,
                "args": t_args,
                "result_preview": result_str[:500] + ("..." if len(result_str)>500 else ""),
            }
            trace.append(trace_item)
            yield {"type": "trace_append", "item": trace_item}

            tool_msg_content = result_str
            if t_name in ("open_file", "read_file"):
                fp = (t_args.get("file_path") or "").strip()
                tool_msg_content = f"[{t_name}] opened file: {fp} (content delivered to UI; not included here)"
            messages.append(ToolMessage(content=f"[Tool:{t_name}]\n{tool_msg_content}", tool_call_id=t_id))

        steps += 1

    if question and final_answer:
        history_ref.append({"q": question, "a": final_answer})
        if len(history_ref) > self.max_history:
            del history_ref[:-self.max_history]

    yield {"type": "done", "ok": True, "query_type": "agent", "sources": sources, "trace": trace}
