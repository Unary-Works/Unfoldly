"""
view_detail handler — extracted from FileAgent._handle_view_detail.
Handler module — extracted from FileAgent for modularity.
Each handler is a generator function that yields stream events.
"""
from __future__ import annotations
import os, re, time, json, uuid
from typing import Any, Dict, List, Optional, Generator

from utils.logger import get_logger
logger = get_logger()

from config.prompts import (
    SUMMARIZE_ALL_PROMPT, SUMMARIZE_TOPICS_PROMPT,
    SUMMARIZE_SINGLE_FILE_PROMPT, NO_RESULT_PROMPT,
    get_prompt, normalize_prompt_language,
)
from core.kb.knowledge_base import FileKnowledgeBase

_MEDIA_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".ts",
               ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}


def _handle_view_detail(
    self,
    index: int,
    file_name: str,
    session_id: Optional[str] = None,
    prompt_language: Optional[str] = None,
):
    doc = self._find_file_by_name_or_index(index, file_name, session_id=session_id)
    lang = self._resolve_prompt_language(prompt_language, question=file_name, session_id=session_id)
    
    if not doc:
        msg = (
            "找不到对应的文件，请先搜索或指定正确的序号"
            if lang == "zh"
            else "Cannot find the target file. Please search first or provide a valid index."
        )
        yield {"type": "text", "content": msg}
        yield {"type": "done", "query_type": "error"}
        return
    
    file_path = doc.get('file_path', '')
    fname = doc.get('file_name', '')
    summary = doc.get('doc_summary', '')
    category = doc.get('doc_category', '')
    text_preview = doc.get('text', '')[:1000]
    
    logger.info(f"查看详情: {fname}")
    logger.info(f"摘要: {summary}")
    self._log(f"查看详情: {fname}, 摘要: {summary}")

    ext = os.path.splitext(fname)[1].lower()
    is_media = ext in _MEDIA_EXTS
    if is_media:
        try:
            from core.kb import get_kb_instance
            kb = get_kb_instance()
            all_chunks = kb.collection.get(
                where={"file_path": file_path},
                include=["documents", "metadatas"],
            )
            meta_lines: List[str] = []
            kf_entries: List[tuple] = []   # (time_sec, doc)
            asr_lines: List[str] = []
            for chunk_doc, chunk_meta in zip(
                all_chunks.get("documents") or [],
                all_chunks.get("metadatas") or [],
            ):
                ctype = chunk_meta.get("chunk_type", "")
                if ctype == "media_summary":
                    for line in chunk_doc.splitlines():
                        if not line.startswith("内容摘要:"):
                            meta_lines.append(line)
                elif ctype == "keyframe":
                    t = float(chunk_meta.get("keyframe_time_sec", 0))
                    kf_entries.append((t, chunk_doc))
                elif ctype == "asr_transcript":
                    asr_lines.append(chunk_doc)

            kf_entries.sort(key=lambda x: x[0])
            metadata_text = "\n".join(l for l in meta_lines if l.strip()) or fname
            keyframes_text = "\n\n".join(
                f"[{t:.0f}s] {doc}" for t, doc in kf_entries
            ) if kf_entries else (
                "\n".join(asr_lines) if asr_lines else summary
            )
            logger.info(f"[view_detail] media={fname} keyframe_chunks={len(kf_entries)} asr_chunks={len(asr_lines)}")

            prompt = self._prompt("SUMMARIZE_VIDEO_FILE_PROMPT", lang).format(
                metadata=metadata_text,
                keyframes=keyframes_text,
            )
        except Exception as _e:
            logger.warning(f"[view_detail] Failed to load media chunks: {_e}, falling back to default prompt")
            prompt = self._prompt("SUMMARIZE_SINGLE_FILE_PROMPT", lang).format(summary=summary, text_preview=text_preview)
    else:
        prompt = self._prompt("SUMMARIZE_SINGLE_FILE_PROMPT", lang).format(summary=summary, text_preview=text_preview)

    llm = self._get_llm_service(detailed=True, session_id=session_id, prompt_language=lang)
    full_response = ""
    for chunk in llm.generate_stream(prompt):
        if self.is_aborted(session_id):
            logger.info(f"检测到中断标志，停止生成 (session={session_id})")
            yield {"type": "done", "ok": False, "query_type": "interrupted", "message": "生成已中断"}
            return
        full_response += chunk
        yield {"type": "text", "delta": chunk}
    
    source_doc = [{
        'file_name': fname,
        'file_path': file_path,
        'doc_category': category,
        'doc_summary': summary
    }]
    yield {"type": "sources", "content": source_doc}
    
    self.current_viewing_file = file_path
    try:
        hist_ref = self._get_history_ref(session_id)
        if hist_ref:
            hist_ref[-1]["a"] = full_response
    except Exception:
        pass
    self._log(f"详情回复: {full_response}")
    yield {"type": "done", "query_type": "detail"}
