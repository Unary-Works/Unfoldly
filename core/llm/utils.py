"""
LLM utility functions — extracted from core/langgraph_agent.py Phase 1.
"""
from __future__ import annotations
import os, sys, time, json, re, uuid, hashlib, gc, math, struct, threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TypedDict, Literal, List, Dict, Any, Optional, Callable, Annotated, Tuple, Iterator, Union, Sequence

from config import settings
from config.prompts import (
    CLASSIFY_PROMPT, SUMMARY_PROMPT, IMAGE_SUMMARY_PROMPT, IMAGE_OCR_PROMPT,
    INTENT_DETECTION_PROMPT, INTENT_DETECTION_SYSTEM_PROMPT,
    VIEW_DETAIL_INTENT_PROMPT, REWRITE_QUERY_PROMPT, AMBIGUOUS_QUERY_PROMPT,
    SUMMARIZE_SINGLE_FILE_PROMPT, SUMMARIZE_TOPICS_PROMPT, SUMMARIZE_ALL_PROMPT,
    CAPABILITY_QUERY_PROMPT, NO_RESULT_PROMPT, CHAT_FALLBACK_PROMPT,
    PLANNER_PROMPT, FINAL_ANSWER_PROMPT, EFFICIENT_ASSISTANT_PROMPT,
    get_prompt, normalize_prompt_language,
)
from utils.logger import get_logger
logger = get_logger()
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

_NO_SYSTEM_ROLE_MODELS = {"gemma", "gemma2", "gemma3", "gemma-3"}
_model_system_role_cache: Dict[str, bool] = {}

def _model_supports_system_role() -> bool:
    try:
        from services.preference_manager import PreferenceManager
        pref_manager = PreferenceManager()
        current_model_id = pref_manager.get_selected_model_id()
        
        if current_model_id:
            if current_model_id in _model_system_role_cache:
                return _model_system_role_cache[current_model_id]
            
            model_id_lower = current_model_id.lower()
            
            if "gemma-4" in model_id_lower or "gemma4" in model_id_lower:
                _model_system_role_cache[current_model_id] = True
                return True
                
            for pattern in _NO_SYSTEM_ROLE_MODELS:
                if pattern in model_id_lower:
                    logger.info(f"Model '{current_model_id}' does NOT support system role (using Gemma-compatible mode)")
                    _model_system_role_cache[current_model_id] = False
                    return False
            
            _model_system_role_cache[current_model_id] = True
            return True
    except Exception as e:
        logger.error(f"_model_supports_system_role error: {e}")
    return True



MAX_TOKENS_PER_CHUNK = 6000
MAX_TOTAL_TOKENS = 20000


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text) // 3


def _chunk_text(text: str, max_chars: int = 18000) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    lines = text.split('\n')
    current_chunk = ""
    
    for line in lines:
        if len(current_chunk) + len(line) + 1 <= max_chars:
            current_chunk += line + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(line) > max_chars:
                for i in range(0, len(line), max_chars):
                    chunks.append(line[i:i+max_chars])
                current_chunk = ""
            else:
                current_chunk = line + "\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text[:max_chars]]


def summarize_long_tool_result(
    tool_result: str,
    question: str,
    llm,
    max_chars: int = 18000,
) -> str:
    estimated_tokens = _estimate_tokens(tool_result)
    
    if estimated_tokens <= MAX_TOTAL_TOKENS:
        return tool_result
    
    logger.info(f"Tool result too long ({estimated_tokens} tokens), applying Map-Reduce summarization...")
    
    chunks = _chunk_text(tool_result, max_chars)
    chunk_summaries = []
    
    for i, chunk in enumerate(chunks):
        map_prompt = MAP_PROMPT.format(question=question, chunk=chunk)

        try:
            response = llm.invoke([HumanMessage(content=map_prompt)])
            summary = response.content.strip()
            if summary and summary != "无相关信息":
                chunk_summaries.append(f"[片段{i+1}摘要]\n{summary}")
        except Exception as e:
            logger.error(f"Map phase error for chunk {i+1}: {e}")
            chunk_summaries.append(f"[片段{i+1}节选]\n{chunk[:1000]}...")
    
    if not chunk_summaries:
        return tool_result[:max_chars] + f"\n\n[注意：原始结果过长（约{estimated_tokens}tokens），已截断]"
    
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]
    
    combined_summaries = "\n\n".join(chunk_summaries)
    
    if _estimate_tokens(combined_summaries) > MAX_TOKENS_PER_CHUNK:
        reduce_prompt = REDUCE_PROMPT.format(question=question, combined_summaries=combined_summaries)

        try:
            response = llm.invoke([HumanMessage(content=reduce_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Reduce phase error: {e}")
            return combined_summaries[:max_chars]
    
    return combined_summaries


def convert_messages_for_gemma(messages: List[Any]) -> List[Any]:
    converted: List[Any] = []
    system_content = ""
    
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_content += (msg.content or "") + "\n"
        elif isinstance(msg, HumanMessage):
            content = msg.content or ""
            if system_content and not converted:
                content = f"[Instructions]\n{system_content.strip()}\n\n[User Question]\n{content}"
                system_content = ""
            converted.append(HumanMessage(content=content))
        elif isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                tool_desc = "我将调用以下工具：\n"
                for tc in tool_calls:
                    tool_desc += f"- {tc.get('name', 'unknown')}({tc.get('args', {})})\n"
                converted.append(AIMessage(content=tool_desc))
            else:
                content = msg.content or ""
                if content:
                    converted.append(AIMessage(content=content))
        elif isinstance(msg, ToolMessage):
            tool_result = f"[Tool Result]\n{msg.content or ''}"
            converted.append(HumanMessage(content=tool_result))
    
    if system_content:
        converted.insert(0, HumanMessage(content=f"[Instructions]\n{system_content.strip()}"))
    
    final_messages: List[Any] = []
    for msg in converted:
        if not final_messages:
            final_messages.append(msg)
        else:
            last = final_messages[-1]
            same_role = (isinstance(last, HumanMessage) and isinstance(msg, HumanMessage)) or \
                        (isinstance(last, AIMessage) and isinstance(msg, AIMessage))
            if same_role:
                combined_content = (last.content or "") + "\n\n" + (msg.content or "")
                if isinstance(last, HumanMessage):
                    final_messages[-1] = HumanMessage(content=combined_content)
                else:
                    final_messages[-1] = AIMessage(content=combined_content)
            else:
                final_messages.append(msg)
    
    if final_messages and isinstance(final_messages[0], AIMessage):
        final_messages.insert(0, HumanMessage(content="[System] 开始对话"))
    
    
    return final_messages


def _approx_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return max(0, int(len(text) / 4))


def _approx_tokens_from_messages(msgs: List[Any]) -> int:
    total = 0
    for m in msgs:
        try:
            c = getattr(m, "content", "") or ""
            if isinstance(c, list):
                s = ""
                for it in c:
                    if isinstance(it, str):
                        s += it
                    elif isinstance(it, dict) and it.get("type") == "text":
                        s += it.get("text", "")
                total += _approx_tokens_from_text(s)
            else:
                total += _approx_tokens_from_text(str(c))
        except Exception:
            pass
    return total


def _chunk_text_by_newlines(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    if max_chars <= 0:
        return [text]
    lines = text.splitlines(keepends=True)
    chunks: List[str] = []
    buf = ""
    for ln in lines:
        if len(buf) + len(ln) > max_chars and buf:
            chunks.append(buf)
            buf = ""
        buf += ln
    if buf:
        chunks.append(buf)
    return chunks


def build_messages_for_model(
    sys_prompt: str,
    history: List[Dict[str, str]],
    current_question: str,
) -> List[Any]:
    supports_system = _model_supports_system_role()
    messages: List[Any] = []
    
    valid_pairs = []
    for h in history:
        q = (h.get("q") or "").strip()
        a = (h.get("a") or "").strip()
        if q and a:
            valid_pairs.append((q, a))
    
    if supports_system:
        messages.append(SystemMessage(content=sys_prompt))
        for q, a in valid_pairs:
            messages.append(HumanMessage(content=q))
            messages.append(AIMessage(content=a))
        messages.append(HumanMessage(content=current_question))
    else:
        sys_prefix = f"[Instructions]\n{sys_prompt}\n\n[User Question]\n"
        
        if valid_pairs:
            first_q, first_a = valid_pairs[0]
            messages.append(HumanMessage(content=sys_prefix + first_q))
            messages.append(AIMessage(content=first_a))
            
            for q, a in valid_pairs[1:]:
                messages.append(HumanMessage(content=q))
                messages.append(AIMessage(content=a))
            
            messages.append(HumanMessage(content=current_question))
        else:
            messages.append(HumanMessage(content=sys_prefix + current_question))
    
    return messages
def stream_replace_markdown_links(stream_chunks, link_map: dict):
    import re
    buffer = ""
    pattern = re.compile(r'^\[([^\]]+)\]\((\d+)\)')
    
    def is_valid_prefix(s: str) -> bool:
        if not s.startswith('['): return False
        cb = s.find(']')
        if cb == -1:
            if '\n' in s: return False
            return True
        if len(s) > cb + 1:
            if s[cb + 1] != '(': return False
            rest = s[cb + 2:]
            for i, char in enumerate(rest):
                if char.isdigit(): continue
                elif char == ')':
                    if i != len(rest) - 1: return False
                else: return False
        return True

    for chunk in stream_chunks:
        if not chunk: continue
        buffer += str(chunk)
        
        while True:
            idx = buffer.find('[')
            if idx == -1:
                yield buffer
                buffer = ""
                break
                
            if idx > 0:
                yield buffer[:idx]
                buffer = buffer[idx:]
                
            match = pattern.match(buffer)
            if match:
                text = match.group(1)
                file_idx = match.group(2)
                if file_idx in link_map:
                    yield link_map[file_idx]
                else:
                    yield match.group(0)
                buffer = buffer[match.end():]
                continue
                
            if len(buffer) < 200 and is_valid_prefix(buffer):
                break
            else:
                yield "["
                buffer = buffer[1:]
                continue
                
    if buffer:
        yield buffer
