"""
Agent dispatch module — extracted from FileAgent for modularity.
"""
from __future__ import annotations
import os, re, time, json, uuid, gc
from typing import Any, Dict, List, Optional, Generator, Iterator

from utils.logger import get_logger
logger = get_logger()

from config import settings
from config.prompts import get_prompt, normalize_prompt_language
from core.handlers.context import HandlerContext
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from core.llm.builder import get_llm, ToolAgentState
from core.llm.utils import (
    _model_supports_system_role,
    build_messages_for_model,
    convert_messages_for_gemma,
)
from tools import get_all_tools, get_tool
from tools.document_tools import count_documents, get_kb_instance, search_documents


def _handle_tools(self, task: str, session_id: Optional[str] = None):
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
        "You are an 'Intent Recognition + Planning + Tool Calling' intelligent file assistant.\n"
        "Your task: infer the user's goal (retrieve info / revise resume / update code / file operation / debugging), then call tools as needed.\n\n"
        "## Core Instructions\n"
        "0) Must-follow principles\n"
        "- For any conclusion based on file content or directory structure, gather evidence via tools first; never fabricate.\n"
        "- If retrieval returns chunks, aggregate by file_path/file_name to file level before counting files.\n"
        "- Write/delete/move are destructive operations: explain planned changes first and prefer creating a new file or backup copy.\n\n"
        "1) Tool strategy\n"
        "- Information retrieval: use indexed DB tools (`search_documents`, `search_files`, `count_documents_files`); never use physical disk search/traversal for retrieval.\n"
        "- Engineering/file operations: locate via indexed `search_files` when possible, then call only tools listed under [Available Tools].\n"
        "- If a requested operation needs a tool that is not listed under [Available Tools], explain the limitation instead of inventing a tool call.\n"
        "- Resume editing: clarify target role/JD/which resume/optimization focus before retrieval and edits.\n\n"
        "2) Output behavior\n"
        "- Start with a short 1-3 step plan (which tools and why).\n"
        "- If user confirmation is needed (especially destructive changes), ask before executing.\n"
    )
    if FILE_TOOLS_ROOT:
        sys_prompt += f"\n[Safety Scope] File operations are allowed only under: {FILE_TOOLS_ROOT}\n"
    if tool_names:
        sys_prompt += "\n[Available Tools]\n- " + "\n- ".join(sorted(set(tool_names))) + "\n"

    yield from self._run_tool_agent(task, require_tools=True, session_id=session_id)
    return

def _run_tool_agent(self, task: str, require_tools: bool = False, session_id: Optional[str] = None):
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

    category_info = self._get_category_stats(prompt_language="en")
    sys_prompt = (
        "You are a 'Planning + Tool Calling' intelligent file assistant.\n"
        "Never fabricate file/database contents.\n\n"
        "[Known Indexed Categories and Counts]\n"
        f"{category_info}\n\n"
        "[Database / Specific File-Type Tasks]\n"
        "- Count/list a specific category: use `count_documents_files(category='category_name')`, and category_name must come from the known list above.\n"
        "- Count/list database files: use `count_documents_files(file_extensions='.db,.sqlite,.sql,.csv')`.\n"
        "- Count/list images/code/docs: also use `count_documents_files` with explicit file extensions.\n"
        "- For requests like 'which database files do I have', prioritize extension-based matching (.db, .sqlite, .sql, etc.) instead of filename contains heuristics.\n\n"
        "[Hybrid Retrieval (Filename + Content)]\n"
        "- For requests like 'find / related materials / what files' with person/term keywords, prioritize `search_documents` on indexed content.\n"
        "- If user specifies categories like resume/report, pass that category into `search_documents` for precise retrieval.\n"
        "- For follow-up path/location questions, use `search_files` to get real file paths (multiple keywords can be queried separately).\n"
        "- Do not conclude 'not found' based only on `search_files`; many hits come from file content.\n\n"
        "[Path/Location Queries - Important]\n"
        "- For 'path / where / location / which folder' requests: call `search_files` first to retrieve real file_path values, then answer. Do not guess paths.\n"
        "- If multiple file names are provided in one request, plan multiple `search_files` calls and aggregate all hits (including duplicate names).\n"
        "- If no hits, explicitly say 'not found within indexed scope' and suggest checking Sources/indexing.\n\n"
        "[Other Operations]\n"
        "- Semantic content retrieval: `search_documents`\n"
        "- Topic synthesis: `summarize_topics`\n"
        "- File opening or preview: use `open_file` only when it appears under [Available Tools].\n"
        "- For any other operation, call only tools listed under [Available Tools]; do not invent unavailable shell, read, write, or list tools.\n"
        "- Destructive ops like move/delete: present dry-run plan and change list first, execute only after user confirmation.\n"
    )
    if FILE_TOOLS_ROOT:
        sys_prompt += f"\n[File Tool Safety Root]\n{FILE_TOOLS_ROOT}\n"
    if tool_names:
        sys_prompt += "\n[Available Tools]\n- " + "\n- ".join(sorted(set(tool_names))) + "\n"

    hist_ref = self._get_history_ref(session_id)
    history_dicts = [{"q": h.get("q", ""), "a": h.get("a", "")} for h in hist_ref[-5:]]
    messages = build_messages_for_model(sys_prompt, history_dicts, task)

    init_state: ToolAgentState = {
        "messages": messages,
        "question": task,
        "final_answer": "",
        "source_files": [],
        "trace": [],
        "steps": 0,
        "require_tools": bool(require_tools),
    }

    out = self.graph.invoke(init_state)
    answer = (out or {}).get("final_answer", "") or ""
    sources = (out or {}).get("source_files", []) or []
    trace = (out or {}).get("trace", []) or []

    if sources:
        yield {"type": "sources", "content": sources}
    if trace:
        yield {"type": "trace", "content": trace}
    yield {"type": "text", "content": answer}
    try:
        hist_ref[-1]["a"] = answer
    except Exception:
        pass
    yield {"type": "done", "query_type": "tools" if require_tools else "agent"}

def _build_graph(self):
    import os

    MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "12"))
    
    tools = get_all_tools()
    if not tools:
        tools = [count_documents, search_documents]
    llm_with_tools = get_llm().bind_tools(tools)
    
    supports_system = _model_supports_system_role()
    
    def call_model(state: ToolAgentState) -> ToolAgentState:
        messages = state["messages"]
        require_tools = bool(state.get("require_tools", False))
        trace = state.get("trace", []) or []

        messages_to_send = messages
        if not supports_system:
            messages_to_send = convert_messages_for_gemma(messages)
        
        response = llm_with_tools.invoke(messages_to_send)

        if require_tools and (not getattr(response, "tool_calls", None)):
            nudge = HumanMessage(
                content=(
                    "[Important] A tool call is required for this task. Call one of the available tools "
                    "that fits the request. Do not answer with command text or an explanation only; "
                    "return tool_calls directly."
                )
            )
            nudge_messages = messages_to_send + [nudge]
            response = llm_with_tools.invoke(nudge_messages)

        try:
            tool_calls = getattr(response, "tool_calls", None) or []
            if tool_calls:
                trace.append(
                    {
                        "type": "plan",
                        "title": "准备执行",
                        "preview": "、".join([tc.get("name", "") for tc in tool_calls if tc.get("name")])[:160],
                        "tool_calls": tool_calls,
                    }
                )
        except Exception:
            pass

        return {
            "messages": messages + [response],
            "require_tools": require_tools,
            "trace": trace,
            "source_files": state.get("source_files", []) or [],
            "steps": int(state.get("steps", 0)),
            "question": state.get("question", ""),
            "final_answer": state.get("final_answer", ""),
        }
    
    def call_tools(state: ToolAgentState) -> ToolAgentState:

        messages = state["messages"]
        last_message = messages[-1]
        steps = int(state.get("steps", 0))
        trace = state.get("trace", []) or []
        
        tool_results = []
        sources = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logger.info(f"[Agent] 执行工具: {tool_name}, 参数: {tool_args}")
            
            tool_obj = get_tool(tool_name)
            if tool_obj is None:
                result = f"未知工具: {tool_name}"
            else:
                try:
                    if hasattr(tool_obj, "invoke"):
                        result = tool_obj.invoke(tool_args)
                    elif hasattr(tool_obj, "run"):
                        result = tool_obj.run(**tool_args)
                    else:
                        result = tool_obj(**tool_args)  # type: ignore
                except Exception as e:
                    result = f"工具执行失败({tool_name}): {e}"

            try:
                args_preview = str(tool_args)
                if len(args_preview) > 260:
                    args_preview = args_preview[:260] + "..."
                res_preview = str(result)
                if len(res_preview) > 420:
                    res_preview = res_preview[:420] + "..."
                trace.append(
                    {
                        "type": "tool",
                        "tool": tool_name,
                        "args": tool_args,
                        "args_preview": args_preview,
                        "result_preview": res_preview,
                    }
                )
            except Exception:
                pass
            
            if tool_name == "search_documents" and str(result) != "[NO_RELEVANT_DOCS]":
                kb = get_kb_instance()
                search_results = kb.vector_search(
                    tool_args.get("query", ""), 
                    n_results=settings.VECTOR_SEARCH_TOP_K
                )
                reranked = kb.rerank(
                    tool_args.get("query", ""), 
                    search_results, 
                    top_k=settings.RERANK_TOP_K
                )
                for doc in reranked:
                    if doc.get('rerank_score', 0) >= settings.RELEVANCE_THRESHOLD:
                        sources.append({
                            'file_name': doc.get('file_name', ''),
                            'file_path': doc.get('file_path', ''),
                            'doc_summary': doc.get('doc_summary', ''),
                            'doc_category': doc.get('doc_category', ''),
                            'rerank_score': doc.get('rerank_score', 0),
                            'text': doc.get('text', '')[:300],
                        })

            if tool_name == "count_documents_files":
                try:
                    import json
                    payload = json.loads(str(result))
                    files = payload.get("files", []) or []
                    if files:
                        sources.extend(files[:50])
                except Exception:
                    pass
            
            if tool_name == "list_directory":
                try:
                    lines = str(result).split('\n')
                    dir_path = tool_args.get("directory_path", ".")
                    if dir_path.startswith("~"):
                        dir_path = os.path.expanduser(dir_path)
                        
                    count_added = 0
                    for line in lines:
                        line = line.strip()
                        if not line or count_added >= 50: continue
                        
                        if "Error" in line or "Exception" in line: continue

                        full_path = os.path.join(dir_path, line)
                        
                        icon_type = "folder" if os.path.isdir(full_path) else "doc"
                        if not os.path.exists(full_path):
                            icon_type = "folder" if "." not in line else "doc"

                        sources.append({
                            "file_name": line,
                            "file_path": full_path,
                            "doc_category": "文件浏览",
                            "doc_summary": f"位于 {dir_path}",
                            "type": icon_type
                        })
                        count_added += 1
                except Exception as e:
                    logger.error(f"[Agent] 解析 list_directory 结果失败: {e}")

            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        
        return {
            "messages": messages + tool_results,
            "source_files": sources,
            "steps": steps + max(1, len(getattr(last_message, "tool_calls", []) or [])),
            "require_tools": bool(state.get("require_tools", False)),
            "trace": trace,
        }
    
    def generate_final(state: ToolAgentState) -> ToolAgentState:
        messages = state["messages"]
        question = state["question"]
        llm = get_llm()
        source_files = state.get("source_files", []) or []
        require_tools = bool(state.get("require_tools", False))
        trace = state.get("trace", []) or []
        
        tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]

        if require_tools and not tool_results:
            return {
                "final_answer": "No tool has been executed yet. I will call tools first to collect directory/file evidence, then continue.",
                "source_files": source_files,
                "trace": trace,
            }
        
        has_valid_results = False
        valid_results = []
        for result in tool_results:
            if result != "[NO_RELEVANT_DOCS]":
                has_valid_results = True
                valid_results.append(result)
        
        if has_valid_results and valid_results:
            context = "\n\n".join(valid_results)
            
            final_prompt = f"""Answer the user's question based on the tool results below.

<Tool Results>
{context}
</Tool Results>

<User Question>
{question}
</User Question>

[Answer Requirements]
1. Briefly summarize the tool outcome (for example: found X files, mainly about ...).
2. **Do not** output a long raw file list because the UI already renders it.
3. Guide the user to the file list/sources panel in the UI.
4. If results are very few (<5), listing them directly is acceptable.
"""

            response = llm.invoke([HumanMessage(content=final_prompt)])
            return {"final_answer": response.content, "source_files": source_files, "trace": trace}
        
        elif tool_results and not has_valid_results:
            logger.info(f"[Agent] 检索无结果，进入 chat 模式")
            
            chat_prompt = f"""User asked: "{question}"

No relevant information was found in the local indexed content.
Please answer directly if possible, or suggest that additional external information may be needed.
Do not mention internal implementation details such as indexing or retrieval."""

            response = llm.invoke([HumanMessage(content=chat_prompt)])
            return {"final_answer": response.content, "source_files": source_files, "trace": trace}
        
        else:
            last_ai = [m for m in messages if isinstance(m, AIMessage)]
            if last_ai:
                return {"final_answer": last_ai[-1].content, "source_files": source_files, "trace": trace}
            return {"final_answer": "Sorry, I cannot answer this question right now.", "source_files": source_files, "trace": trace}
    
    def should_use_tools(state: ToolAgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        steps = int(state.get("steps", 0))

        if steps >= MAX_TOOL_STEPS:
            return "final"
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "final"
    
    workflow = StateGraph(ToolAgentState)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    workflow.add_node("final", generate_final)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_use_tools,
        {
            "tools": "tools",
            "final": "final"
        }
    )
    
    workflow.add_edge("tools", "agent")
    workflow.add_edge("final", END)
    
    return workflow.compile()
