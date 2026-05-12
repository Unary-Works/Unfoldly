"""
count handler — extracted from FileAgent._handle_count.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
logger = get_logger()

from config import settings
from core.kb.knowledge_base import FileKnowledgeBase
from core.retrieval.category_engine import normalize_meta_category_name, normalize_stored_category_name
from tools.document_tools import get_kb_instance

def _handle_count(self, category: str, original_question: str, allowed_paths: Optional[List[str]] = None, session_id: Optional[str] = None, params: dict = None):
    import os
    try:
        def _safe_label(v: Any, fallback: str = "other") -> str:
            s = str(v or "").strip()
            s = "".join(ch for ch in s if ch.isprintable())
            s = " ".join(s.split())
            return s if s else fallback

        def _build_count_scope_context(
            *,
            total_files: int,
            sorted_categories: List[Tuple[str, int]],
            sorted_items: List[Dict[str, Any]],
        ) -> Dict[str, Any]:

            top_categories = []
            for cat, cnt in sorted_categories:
                top_categories.append(
                    {
                        "category": _safe_label(cat),
                        "count": int(cnt or 0),
                    }
                )

            keep_cats_limit = max(1, int(os.getenv("COUNT_SCOPE_KEEP_CATEGORIES", "24")))
            sample_limit_per_cat = max(1, int(os.getenv("COUNT_SCOPE_SAMPLES_PER_CATEGORY", "10")))
            keep_cats = [str(x.get("category") or "").strip() for x in top_categories[:keep_cats_limit] if x.get("category")]
            samples_by_category: Dict[str, List[Dict[str, str]]] = {c: [] for c in keep_cats}
            for it in sorted_items:
                cat = _safe_label(
                    normalize_stored_category_name(
                        it.get("doc_category") or "other",
                        media_type=it.get("media_type") or "",
                    )
                )
                if cat not in samples_by_category:
                    continue
                bucket = samples_by_category.get(cat) or []
                if len(bucket) >= sample_limit_per_cat:
                    continue
                bucket.append(
                    {
                        "file_name": str(it.get("file_name") or ""),
                        "file_path": str(it.get("file_path") or ""),
                        "doc_summary": str(it.get("doc_summary") or "")[:200],
                    }
                )
                samples_by_category[cat] = bucket
                if all(len(v) >= sample_limit_per_cat for v in samples_by_category.values()):
                    break

            return {
                "kind": "count_all",
                "total_files": int(max(0, total_files)),
                "category_counts": top_categories,
                "samples_by_category": samples_by_category,
                "stored_at": float(time.time()),
            }

        lang = self._resolve_prompt_language(None, question=original_question, session_id=session_id)
        is_zh = (lang == "zh")

        category = self._normalize_category_name(category)
        if self._is_generic_file_scope_category(category):
            category = "all"
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
            category_label = "视频文件" if is_zh else "video"
            media_type_filter = media_type_filter or "video"
        elif category == "audio":
            category_label = "音频文件" if is_zh else "audio"
            media_type_filter = media_type_filter or "audio"
        elif category == "audio/video":
            if media_type_filter == "video":
                category_label = "视频文件" if is_zh else "video"
            elif media_type_filter == "audio":
                category_label = "音频文件" if is_zh else "audio"
            else:
                category_label = "音视频文件" if is_zh else "media"
        kb = get_kb_instance()
        total_count = kb.collection.count()

        ext_filter: Optional[set] = None
        min_file_size_bytes = None
        max_file_size_bytes = None
        if params:
            try:
                if params.get("min_file_size_bytes") is not None:
                    min_file_size_bytes = int(params.get("min_file_size_bytes") or 0)
                if params.get("max_file_size_bytes") is not None:
                    max_file_size_bytes = int(params.get("max_file_size_bytes") or 0)
            except (TypeError, ValueError):
                min_file_size_bytes = None
                max_file_size_bytes = None

        def _file_size_bytes(meta: Dict[str, Any], file_path: str) -> Optional[int]:
            for key in ("file_size_bytes", "size_bytes", "file_size", "size"):
                value = meta.get(key)
                if value is None:
                    continue
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    continue
            if file_path:
                try:
                    return int(os.path.getsize(file_path))
                except Exception:
                    return None
            return None

        def _matches_size_filter(meta: Dict[str, Any], file_path: str) -> bool:
            if min_file_size_bytes is None and max_file_size_bytes is None:
                return True
            size = _file_size_bytes(meta, file_path)
            if size is None:
                return False
            if min_file_size_bytes is not None and size <= min_file_size_bytes:
                return False
            if max_file_size_bytes is not None and size >= max_file_size_bytes:
                return False
            return True

        if params:
            raw_exts = str(params.get("file_extensions") or "").strip()
            if raw_exts:
                ext_filter = set(
                    e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
                    for e in raw_exts.replace(",", " ").split()
                    if e.strip()
                )
        if not ext_filter:
            auto_exts = self._extract_ext_filters(original_question, params)
            if auto_exts:
                ext_filter = set(auto_exts)
                logger.info(f"[count] auto-extracted ext_filter from question: {ext_filter}")

        indexed_inventory_items: Optional[List[Dict[str, Any]]] = None
        if hasattr(kb, "indexed_file_inventory"):
            try:
                inventory_pack = kb.indexed_file_inventory(
                    allowed_paths=allowed_paths,
                    file_extensions=sorted(ext_filter) if ext_filter else None,
                    limit=0,
                    hydrate=True,
                )
                if inventory_pack.get("ready"):
                    indexed_inventory_items = []
                    for item in inventory_pack.get("files") or []:
                        meta = dict(item.get("metadata") or {})
                        meta["file_path"] = str(item.get("file_path") or meta.get("file_path") or "")
                        meta["file_name"] = str(item.get("file_name") or meta.get("file_name") or os.path.basename(meta["file_path"]))
                        meta["doc_summary"] = str(item.get("doc_summary") or meta.get("doc_summary") or "")
                        meta["doc_category"] = str(item.get("doc_category") or meta.get("doc_category") or "other")
                        indexed_inventory_items.append(meta)
                    logger.info(
                        "[count] using indexed inventory: files=%d ext_filter=%s",
                        len(indexed_inventory_items),
                        sorted(ext_filter) if ext_filter else None,
                    )
                else:
                    indexed_inventory_items = []
                    logger.warning("[count] indexed inventory not ready; refusing query-time metadata scan")
            except Exception as exc:
                indexed_inventory_items = []
                logger.warning("[count] indexed inventory failed: %s", exc)

        def _iter_count_metadata():
            yield from (indexed_inventory_items or [])

        if category == "all":
            file_items = {}
            for meta in _iter_count_metadata():
                file_path = meta.get('file_path', '') or ''

                if ext_filter:
                    import os as _os
                    _fext = _os.path.splitext(file_path.lower())[1]
                    if _fext not in ext_filter:
                        continue

                if not _matches_media_type(meta):
                    continue
                if not _matches_size_filter(meta, file_path):
                    continue

                if allowed_paths is not None:
                    is_allowed = False
                    for allowed in allowed_paths:
                        if file_path.startswith(allowed):
                            is_allowed = True
                            break
                    if not is_allowed:
                        continue

                file_name = meta.get('file_name', '') or ''
                file_key = file_path or file_name
                if file_key and file_key not in file_items:
                    file_items[file_key] = {
                        "file_path": file_path,
                        "file_name": file_name or (os.path.basename(file_path) if file_path else file_name),
                        "doc_category": normalize_meta_category_name(meta),
                        "media_type": _infer_media_type(meta),
                        "doc_summary": meta.get('doc_summary', '') or '',
                    }
                elif file_key in file_items and not file_items[file_key].get("doc_summary"):
                    s = meta.get('doc_summary', '') or ''
                    if s:
                        file_items[file_key]["doc_summary"] = s

            category_counts = {}
            for it in file_items.values():
                cat = _safe_label(
                    normalize_stored_category_name(
                        it.get("doc_category") or "other",
                        media_type=it.get("media_type") or "",
                    )
                )
                category_counts[cat] = category_counts.get(cat, 0) + 1

            total = len(file_items)
            sorted_cats = sorted(category_counts.items(), key=lambda x: -x[1])

            if total > 0:
                try:
                    priority = {"resume": 0, "contract": 1, "report": 2, "paper": 3, "manual": 4, "document": 5, "note": 6, "data": 7, "presentation": 8, "email": 9, "image": 10, "audio": 11, "video": 12, "audio/video": 13, "book": 14, "other": 99}
                    items_sorted = sorted(
                        file_items.values(),
                        key=lambda x: (
                            priority.get(
                                normalize_stored_category_name(
                                    str(x.get("doc_category") or "other"),
                                    media_type=x.get("media_type") or "",
                                ),
                                50,
                            ),
                            str(x.get("file_name") or ""),
                        ),
                    )
                    sources = []
                    for it in items_sorted:
                        sources.append(
                            {
                                "file_name": it.get("file_name") or "",
                                "file_path": it.get("file_path") or "",
                                "doc_category": _safe_label(
                                    normalize_stored_category_name(
                                        it.get("doc_category") or "other",
                                        media_type=it.get("media_type") or "",
                                    )
                                ),
                                "doc_summary": (it.get("doc_summary") or "")[:200],
                            }
                        )
                    if sources:
                        self._set_last_search_results(session_id, sources)
                        yield {
                            "type": "sources",
                            "content": sources,
                            "total_matches": total,
                            "shown_count": len(sources),
                        }
                    self._set_count_scope_context(
                        session_id,
                        _build_count_scope_context(
                            total_files=total,
                            sorted_categories=sorted_cats,
                            sorted_items=items_sorted,
                        ),
                    )
                except Exception:
                    pass
            else:
                self._clear_count_scope_context(session_id, reason="count_all_empty")

            wants_content = any(k in original_question for k in ["内容", "讲什么", "关于什么", "说什么", "摘要", "概要", "简介", "总结"])
            if total == 0:
                if total_count > 0:
                    response = (
                        "📊 当前未选中任何文档，请在右侧（Sources）勾选需要检索的文件或文件夹。"
                        if is_zh
                        else "📊 No files are currently selected. Please select files/folders in Sources first."
                    )
                else:
                    response = (
                        "📊 在您选中的范围内，暂时没有找到任何文件。"
                        if is_zh
                        else "📊 No files were found in the selected scope."
                    )

                yield {"type": "sources", "content": []}
            else:
                if wants_content:
                    response = (
                        f"📊 在您选中的范围内共有 {total} 份文档。\n"
                        if is_zh
                        else f"📊 There are {total} documents in the selected scope.\n"
                    )
                    if sorted_cats:
                        top = sorted_cats[:8]
                        if is_zh:
                            top_desc = "、".join([f"{_safe_label(cat)} {cnt}份" for cat, cnt in top])
                            response += f"主要分类：{top_desc}"
                        else:
                            top_desc = ", ".join([f"{_safe_label(cat)} {cnt}" for cat, cnt in top])
                            response += f"Main categories: {top_desc}"
                        if len(sorted_cats) > len(top):
                            rem_files = max(0, total - sum(cnt for _, cnt in top))
                            if is_zh:
                                response += f"；其余 {len(sorted_cats) - len(top)} 个分类合计 {rem_files} 份。"
                            else:
                                response += f"; the remaining {len(sorted_cats) - len(top)} categories contain {rem_files} files."
                    response += "\n"
                    response += ("\n---\n\n📄 **文件内容概览：**\n\n" if is_zh else "\n---\n\n📄 **Content overview:**\n\n")
                    priority2 = {"resume": 0, "contract": 1, "report": 2, "paper": 3, "manual": 4, "document": 5, "note": 6, "data": 7, "presentation": 8, "email": 9, "image": 10, "audio": 11, "video": 12, "audio/video": 13, "book": 14, "other": 99}
                    items_for_summary = sorted(
                        file_items.values(),
                        key=lambda x: (
                            priority2.get(
                                normalize_stored_category_name(
                                    str(x.get("doc_category") or "other"),
                                    media_type=x.get("media_type") or "",
                                ),
                                50,
                            ),
                            str(x.get("file_name") or ""),
                        ),
                    )
                    try:
                        _cnt_sum_cap_raw = str(
                            os.getenv("COUNT_RESPONSE_FILE_SUMMARY_MAX_CHARS", "") or ""
                        ).strip()
                        _cnt_sum_cap: Optional[int] = None
                        if _cnt_sum_cap_raw and _cnt_sum_cap_raw != "0":
                            _cnt_sum_cap = max(200, int(_cnt_sum_cap_raw))
                    except (TypeError, ValueError):
                        _cnt_sum_cap = None
                    for idx_s, it in enumerate(items_for_summary[:30], 1):
                        fname = it.get("file_name") or ("未知文件" if is_zh else "Unknown file")
                        cat_label = normalize_stored_category_name(
                            it.get("doc_category") or "other",
                            media_type=it.get("media_type") or "",
                        )
                        summary = (it.get("doc_summary") or "").strip()
                        if summary:
                            if _cnt_sum_cap is not None and len(summary) > _cnt_sum_cap:
                                summary_short = summary[: max(0, _cnt_sum_cap - 1)].rstrip() + "…"
                            else:
                                summary_short = summary
                        else:
                            summary_short = ("暂无摘要" if is_zh else "No summary")
                        if is_zh:
                            response += f"**{idx_s}. {fname}**（{cat_label}）\n   {summary_short}\n\n"
                        else:
                            response += f"**{idx_s}. {fname}** ({cat_label})\n   {summary_short}\n\n"
                    if total > 30:
                        response += (f"…… 还有 {total - 30} 份文件未列出。" if is_zh else f"... and {total - 30} more files are not shown.")
                    response += ("\n需要我详细介绍某一份文件吗？" if is_zh else "\nDo you want me to explain one specific file in detail?")
                else:
                    if is_zh:
                        response = f"📊 在您选中的范围内共有 {total} 份文档，分类如下：\n\n"
                        response += "| 分类 | 数量 |\n| --- | ---: |\n"
                        for cat, cnt in sorted_cats:
                            response += f"| {_safe_label(cat)} | {cnt}份 |\n"
                    else:
                        response = f"📊 There are {total} documents in the selected scope, categorized as:\n\n"
                        response += "| Category | Count |\n| --- | ---: |\n"
                        for cat, cnt in sorted_cats:
                            response += f"| {_safe_label(cat)} | {cnt} |\n"

        else:
            unique_files = {}
            unique_items = []
            for meta in _iter_count_metadata():
                if self._normalize_category_name(meta.get('doc_category', 'other')) == category:
                    if not _matches_media_type(meta):
                        continue
                    file_path = meta.get('file_path', '') or ''
                    if not _matches_size_filter(meta, file_path):
                        continue

                    if allowed_paths is not None:
                        is_allowed = False
                        for allowed in allowed_paths:
                            if file_path.startswith(allowed):
                                is_allowed = True
                                break
                        if not is_allowed:
                            continue

                    file_name = meta.get('file_name', '') or ''
                    file_key = file_path or file_name
                    if file_key and file_key not in unique_files:
                        unique_files[file_key] = file_name
                        unique_items.append(
                            {
                                "file_path": file_path,
                                "file_name": file_name or (os.path.basename(file_path) if file_path else file_name),
                                "doc_category": category_label,
                                "doc_summary": (meta.get("doc_summary") or "")[:200],
                            }
                        )

            count = len(unique_files)
            sample_docs = list(unique_files.values())[:3]

            if count > 0:
                sources = [
                    {
                        "file_name": name,
                        "file_path": path,
                        "doc_category": category_label,
                        "doc_summary": "",
                    }
                    for path, name in list(unique_files.items())
                ]
                self._set_last_search_results(session_id, sources)
                yield {
                    "type": "sources",
                    "content": sources,
                    "total_matches": count,
                    "shown_count": len(sources),
                }
                try:
                    self._set_count_scope_context(
                        session_id,
                        _build_count_scope_context(
                            total_files=count,
                            sorted_categories=[(category_label, count)],
                            sorted_items=unique_items,
                        ),
                    )
                except Exception:
                    pass
            else:
                self._clear_count_scope_context(session_id, reason="count_specific_category_empty")

            if count == 0:
                response = (
                    f"📊 在您选中的范围内，没有找到 **{category_label}**。"
                    if is_zh
                    else f"📊 No **{category_label}** files were found in the selected scope."
                )
            elif count > 10:
                if is_zh:
                    response = f"📊 选中的范围内共有 **{count}** 份{category_label}。\n"
                    response += f"\n文件较多，无法一一列出。"
                    response += f"\n您可以：\n- 搜索特定条件（如：某公司的{category_label}）\n- 或告诉我要查看第几份"
                else:
                    response = f"📊 There are **{count}** {category_label} files in the selected scope.\n"
                    response += "\nThere are many matching files, so I can't list all of them."
                    response += f"\nYou can:\n- Search with a specific condition (e.g., a company-related {category_label})\n- Or tell me which one to open"
            else:
                response = (
                    f"📊 选中的范围内共有 **{count}** 份{category_label}。\n"
                    if is_zh
                    else f"📊 There are **{count}** {category_label} files in the selected scope.\n"
                )
                if sample_docs:
                    response += ("\n部分文件：\n" if is_zh else "\nSample files:\n")
                    for doc in sample_docs:
                        response += f"- {doc}\n"
                    response += ("\n需要查看哪一份？" if is_zh else "\nWhich one do you want to open?")

        logger.info(f"count: {category_label}={count if category != 'all' else total}")

        try:
            hist_ref = self._get_history_ref(session_id)
            if hist_ref:
                hist_ref[-1]["a"] = response
        except:
            pass

        yield {"type": "text", "content": response}
        yield {"type": "done", "query_type": "count"}

    except Exception as e:
        error_msg = f"统计失败: {e}"
        logger.error(f"{error_msg}")
        import traceback
        traceback.print_exc()
        yield {"type": "text", "content": error_msg}
        yield {"type": "done", "query_type": "error"}
