"""
summarize handler — extracted from FileAgent._handle_summarize.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
logger = get_logger()

from config import settings
from core.kb.knowledge_base import FileKnowledgeBase
from tools.document_tools import get_kb_instance

def _handle_summarize(
    self,
    category: str,
    original_question: str,
    *,
    keyword: Optional[str] = None,
    allowed_paths: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    prompt_language: Optional[str] = None,
    params: dict = None,
):
    import os
    try:
        lang = self._resolve_prompt_language(prompt_language, question=original_question, session_id=session_id)
        category = self._normalize_category_name(category)
        kb = get_kb_instance()
        media_type_filter = str((params or {}).get("media_type") or "").strip().lower()
        if media_type_filter not in {"audio", "video"}:
            media_type_filter = ""

        def _infer_media_type(meta: Dict[str, Any]) -> str:
            declared = str(meta.get("media_type") or "").strip().lower()
            if declared in {"audio", "video"}:
                return declared
            file_path = str(meta.get("file_path") or meta.get("file_name") or "").strip().lower()
            try:
                from core.media.media_expert import AUDIO_EXTENSIONS as _AUDIO_EXTS, VIDEO_EXTENSIONS as _VIDEO_EXTS
            except Exception:
                _AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape"}
                _VIDEO_EXTS = {".mp4", ".m4v", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".ts"}
            ext = os.path.splitext(file_path)[1]
            if ext in _VIDEO_EXTS:
                return "video"
            if ext in _AUDIO_EXTS:
                return "audio"
            return ""

        def _matches_media_type(meta: Dict[str, Any]) -> bool:
            if not media_type_filter:
                return True
            return _infer_media_type(meta) == media_type_filter

        category_label = category
        if category == "video":
            category_label = "视频文件" if lang == "zh" else "video files"
            media_type_filter = media_type_filter or "video"
        elif category == "audio":
            category_label = "音频文件" if lang == "zh" else "audio files"
            media_type_filter = media_type_filter or "audio"
        elif category == "audio/video":
            if media_type_filter == "video":
                category_label = "视频文件" if lang == "zh" else "video files"
            elif media_type_filter == "audio":
                category_label = "音频文件" if lang == "zh" else "audio files"
            else:
                category_label = "音视频文件" if lang == "zh" else "media files"

        logger.info(f"总结{category_label}主题分布")

        kw = self._normalize_summarize_keyword(category, keyword, original_question) or ""
        kw_retrieval = (
            self._augment_query_for_retrieval(kw, prompt_language=lang, session_id=session_id)
            if kw
            else ""
        )
        file_summaries: Dict[str, Dict[str, str]] = {}  # {file_path: {file_name, doc_summary}}

        def _path_allowed(fp: str) -> bool:
            return FileKnowledgeBase._is_path_allowed(fp, allowed_paths)

        if kw:
            try:
                query_for_search = self._augment_query_for_retrieval(
                    f"{kw_retrieval or kw} {category_label}",
                    prompt_language=lang,
                    session_id=session_id,
                )
                results2 = kb.vector_search(
                    query_for_search,
                    n_results=max(settings.VECTOR_SEARCH_TOP_K, 50),
                    allowed_paths=allowed_paths,
                )
                if results2:
                    reranked = kb.rerank(query_for_search, results2, top_k=max(settings.RERANK_TOP_K, 20))
                    for d in reranked:
                        if d.get("rerank_score", 0) < settings.RELEVANCE_THRESHOLD:
                            continue
                        if self._normalize_category_name(str(d.get("doc_category") or "")) != category:
                            continue
                        if not _matches_media_type(d):
                            continue
                        fp = d.get("file_path") or ""
                        if not fp or (fp in file_summaries) or (not _path_allowed(fp)):
                            continue
                        file_summaries[fp] = {
                            "file_name": d.get("file_name", "") or os.path.basename(fp) or "",
                            "doc_summary": d.get("doc_summary", "") or "",
                        }
            except Exception:
                file_summaries = {}

        if not file_summaries:
            results = kb.collection.get(include=["metadatas"])
            if not results.get("metadatas"):
                msg = f"未找到{category}类文档。" if lang == "zh" else f'No "{category}" documents were found.'
                yield {"type": "text", "content": msg}
                yield {"type": "done", "query_type": "summarize"}
                return

            kw_l = (kw_retrieval or kw).lower()
            for meta in results["metadatas"]:
                if self._normalize_category_name(meta.get("doc_category", "other")) != category:
                    continue
                if not _matches_media_type(meta):
                    continue
                file_path = meta.get("file_path", "") or ""
                if not file_path or file_path in file_summaries:
                    continue
                if not _path_allowed(file_path):
                    continue
                fn = meta.get("file_name", "") or os.path.basename(file_path) or ""
                ds = meta.get("doc_summary", "") or ""
                if kw:
                    hay = f"{fn}\n{file_path}\n{ds}".lower()
                    if kw_l not in hay:
                        continue
                file_summaries[file_path] = {"file_name": fn, "doc_summary": ds}

        total = len(file_summaries)
        if total == 0:
            if allowed_paths == []:
                msg = (
                    f"当前未选中任何文档，无法总结【{category_label}】。\n请在右侧（Sources）勾选需要检索的文件。"
                    if lang == "zh"
                    else f'No sources are selected, so I cannot summarize "{category_label}". Please select files/folders in Sources.'
                )
                yield {"type": "text", "content": msg}
            else:
                msg = f"未找到符合条件的{category_label}。" if lang == "zh" else f'No "{category_label}" files matched your condition.'
                yield {"type": "text", "content": msg}
            yield {"type": "done", "query_type": "summarize"}
            return

        if kw:
            logger.info(f"找到 {total} 份{category_label}（过滤词={kw}），准备总结主题")
        else:
            logger.info(f"找到 {total} 份{category_label}，准备总结主题")

        if total > 0:
            sources = [
                {
                    "file_name": info.get("file_name", ""),
                    "file_path": path,
                    "doc_category": category_label,
                    "doc_summary": (info.get("doc_summary", "") or "")[:200],
                }
                for path, info in list(file_summaries.items())[:50]
            ]
            yield {"type": "sources", "content": sources}
            if kw:
                progress_text = (
                    f"正在归纳「{kw}」相关的 {category_label}（共 {total} 份），请稍等…\n"
                    if lang == "zh"
                    else f'Summarizing "{category_label}" related to "{kw}" ({total} files)...\n'
                )
                yield {
                    "type": "text",
                    "content": progress_text,
                }
            else:
                progress_text = (
                    f"正在归纳 {category_label} 的主题分布（共 {total} 份），请稍等…\n"
                    if lang == "zh"
                    else f'Summarizing topic distribution of "{category_label}" ({total} files)...\n'
                )
                yield {"type": "text", "content": progress_text}

        max_docs = int(os.getenv("SUMMARIZE_MAX_DOCS", "80"))
        max_sum_chars = int(os.getenv("SUMMARIZE_DOC_SUMMARY_MAX_CHARS", "160"))
        items = list(file_summaries.items())[:max_docs]
        summary_list = []
        for i, (path, info) in enumerate(items, 1):
            name = info['file_name']
            summary = (info.get('doc_summary') or "")
            if summary and max_sum_chars > 0:
                summary = summary[:max_sum_chars]
            if summary:
                summary_list.append(f"{i}. {name}: {summary}")
            else:
                summary_list.append(f"{i}. {name}")

        all_summaries = "\n".join(summary_list)

        llm = self._get_llm_service(detailed=True, session_id=session_id, prompt_language=lang)
        used_n = len(items)
        header = (
            f"以下是 {used_n} 份{category_label}的摘要（从全部 {total} 份中抽取，已做截断以便快速归纳）："
            if lang == "zh"
            else f"Below are {used_n} summaries for category '{category_label}' (sampled/truncated from {total} files):"
        )
        if kw:
            header = (
                f"以下是 {used_n} 份与「{kw}」相关的{category_label}摘要（已做截断以便快速归纳）："
                if lang == "zh"
                else f"Below are {used_n} summaries for '{category_label}' related to '{kw}':"
            )
        prompt = self._prompt("SUMMARIZE_TOPICS_PROMPT", lang).format(
            header=f"{header}\n\n{all_summaries}",
            category=category_label,
        )

        if lang == "zh":
            prompt += "\n\n【重要指令】请务必使用中文进行归纳总结。即使外部参考信息包含英文，你的最终输出也必须完全使用中文。"
        else:
            prompt += "\n\n[IMPORTANT] You MUST respond in English. Even if the source summaries contain Chinese, your final response and topic categorization must be entirely in English."

        full_response = ""
        for chunk in llm.generate_stream(prompt):
            if self.is_aborted(session_id):
                logger.info(f"检测到中断标志，停止生成 (session={session_id})")
                yield {"type": "done", "ok": False, "query_type": "interrupted", "message": "生成已中断"}
                return
            full_response += chunk
            yield {"type": "text", "content": chunk}

        try:
            hist_ref = self._get_history_ref(session_id)
            if hist_ref:
                hist_ref[-1]["a"] = full_response
        except Exception:
            pass

        yield {"type": "done", "query_type": "summarize"}

    except Exception as e:
        error_msg = f"总结主题失败: {e}"
        logger.error(f"{error_msg}")
        yield {"type": "text", "content": error_msg}
        yield {"type": "done", "query_type": "error"}
