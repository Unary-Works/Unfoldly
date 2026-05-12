"""
process_previous handler — extracted from FileAgent._handle_process_previous.

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
from langchain_core.messages import HumanMessage
from core.kb.knowledge_base import FileKnowledgeBase
from core.llm.builder import get_llm
from core.llm.utils import stream_replace_markdown_links
from core.retrieval import compute_lookup_overlap_score, lookup_match_quality
from core.retrieval.filename_canonicalizer import (
    classify_reference_target,
    extract_filename_query_surfaces,
)
from core.skills import ContextualRefineSkill, MediaFollowupSkill

_FOLLOWUP_TOPIC_STOPWORDS = {
    "a", "an", "and", "are", "be", "can", "content", "detail", "detailed",
    "details", "do", "does", "explain", "file", "files", "for", "give", "have",
    "how", "in", "is", "it", "its", "me", "more", "next", "of", "on", "please",
    "show", "summarize", "summary", "tell", "the", "them",
    "then", "these", "they", "those", "what", "which", "why", "with", "you",
}


def _build_candidate_affinity_blob(doc: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(doc.get("file_name") or ""),
            str(doc.get("file_path") or ""),
            str(doc.get("lookup_aliases") or ""),
            str(doc.get("doc_summary") or ""),
            str(doc.get("text") or ""),
        ]
    )


def _focused_file_affinity(doc: Dict[str, Any], focused_file: str) -> tuple[int, int]:
    needle = str(focused_file or "").strip()
    if not needle:
        return (0, 0)

    file_name = str(doc.get("file_name") or "").strip()
    file_path = str(doc.get("file_path") or "").strip()
    path_base = os.path.basename(file_path).strip()
    file_stem = os.path.splitext(file_name or path_base)[0].strip()
    needle_lower = needle.lower()
    stem_lower = file_stem.lower()
    name_lower = file_name.lower()
    base_lower = path_base.lower()

    exact, overlap = lookup_match_quality(
        needle,
        _build_candidate_affinity_blob(doc),
    )

    if needle_lower and needle_lower in {name_lower, base_lower, stem_lower}:
        return (3, max(overlap, 8))
    if stem_lower and needle_lower == stem_lower:
        return (3, max(overlap, 8))
    if exact:
        return (2, max(overlap, 6))
    return (1 if overlap >= 2 else 0, overlap)


def _apply_focused_file_bias(
    docs: List[Dict[str, Any]],
    *,
    focused_file: str,
    subagent_query: str,
) -> List[Dict[str, Any]]:
    if not docs or not focused_file:
        return docs

    def _score(doc: Dict[str, Any]) -> tuple[tuple[int, int], int]:
        focus_affinity = _focused_file_affinity(doc, focused_file)
        query_affinity = compute_lookup_overlap_score(
            subagent_query,
            _build_candidate_affinity_blob(doc),
        )
        return (focus_affinity, query_affinity)

    return sorted(docs, key=_score, reverse=True)


def _doc_identity(doc: Dict[str, Any]) -> str:
    return str(doc.get("file_path") or doc.get("file_name") or "").strip()


def _dedupe_docs_by_file_identity(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    by_key: Dict[str, Dict[str, Any]] = {}
    for doc in list(docs or []):
        if not isinstance(doc, dict):
            continue
        key = _doc_identity(doc)
        if not key:
            deduped.append(doc)
            continue
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = doc
            deduped.append(doc)
            continue
        for field in ("doc_summary", "text", "lookup_aliases", "file_name_en", "folder_name_en"):
            if not str(existing.get(field) or "").strip() and str(doc.get(field) or "").strip():
                existing[field] = doc.get(field)
    return deduped


def _resolve_explicit_focus_doc_from_results(
    query: str,
    docs: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    candidates = _dedupe_docs_by_file_identity(list(docs or []))
    if not query or len(candidates) <= 1:
        return None

    ref = classify_reference_target(query)
    needles: List[str] = []
    if str(ref.get("kind") or "") == "explicit" and str(ref.get("target") or "").strip():
        needles.append(str(ref.get("target") or "").strip())
    for surface in extract_filename_query_surfaces(query, max_candidates=3):
        surface = str(surface or "").strip()
        if surface:
            needles.append(surface)
    needles.append(str(query or "").strip())

    ranked: List[tuple[int, int, int, Dict[str, Any]]] = []
    for idx, doc in enumerate(candidates):
        blob = _build_candidate_affinity_blob(doc)
        best_exact = 0
        best_overlap = 0
        for needle in needles:
            exact, overlap = lookup_match_quality(needle, blob)
            best_exact = max(best_exact, 1 if exact else 0)
            best_overlap = max(best_overlap, int(overlap or 0))
        ranked.append((best_exact, best_overlap, -idx, doc))

    ranked.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))
    top_exact, top_overlap, _, top_doc = ranked[0]
    second_overlap = ranked[1][1] if len(ranked) > 1 else 0
    if top_exact or (top_overlap >= 4 and top_overlap >= second_overlap + 2):
        return top_doc
    return None


def _path_to_selected_scope_doc(path: str) -> Dict[str, Any]:
    file_path = str(path or "").strip()
    file_name = os.path.basename(file_path.rstrip(os.sep)) or file_path
    is_dir = False
    try:
        is_dir = os.path.isdir(file_path)
    except Exception:
        is_dir = False
    return {
        "file_name": file_name,
        "file_path": file_path,
        "doc_category": "folder" if is_dir else "",
        "doc_category_family": "folder" if is_dir else "",
        "type": "folder" if is_dir else "file",
        "iconType": "folder" if is_dir else "file",
    }


def _params_request_selected_scope(params: Optional[Dict[str, Any]]) -> bool:
    raw = dict(params or {})
    scope_values = {
        str(raw.get("_scope") or "").strip().lower(),
        str(raw.get("scope") or "").strip().lower(),
        str(raw.get("_scope_kind") or "").strip().lower(),
    }
    if scope_values & {"selected", "selected_items", "selected_folder"}:
        return True
    skill_name = str(raw.get("_skill_name") or "").strip().lower()
    return skill_name in {"list_selected", "summarize_selected"}


_CROSS_RESULT_FOLLOWUP_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|between|both|either|"
    r"who\s+is\s+more|which\s+(?:one\s+)?is\s+more|who\s+fits\s+better|"
    r"which\s+(?:one\s+)?fits\s+better|stronger\s+at|better\s+at|"
    r"difference|different|commonalit(?:y|ies)|similarit(?:y|ies)|"
    r"rank|ranking)\b"
    r"|比较|对比|区别|差异|共同点|相似|谁更|哪个更|哪一个更|更适合|排序",
    re.IGNORECASE,
)


def _looks_like_cross_result_followup(query: str, params: Optional[Dict[str, Any]]) -> bool:
    if _params_request_selected_scope(params):
        return False
    text = str(query or "").strip()
    if not text:
        return False
    return bool(_CROSS_RESULT_FOLLOWUP_RE.search(text))


def _merge_recent_result_sets_for_followup(
    current_results: List[Dict[str, Any]],
    recent_sets: List[List[Dict[str, Any]]],
    *,
    max_files: int = 24,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _add(doc: Dict[str, Any]) -> None:
        if not isinstance(doc, dict):
            return
        ident = _doc_identity(doc)
        if not ident or ident in seen:
            return
        seen.add(ident)
        merged.append(doc)

    for doc in list(current_results or []):
        _add(doc)
    for batch in reversed(list(recent_sets or [])):
        for doc in list(batch or []):
            _add(doc)
            if len(merged) >= max_files:
                return merged
    return merged


def _merge_selected_active_scope_results(
    results: List[Dict[str, Any]],
    active_paths: Optional[List[str]],
) -> List[Dict[str, Any]]:
    selected_paths: List[str] = []
    seen_selected = set()
    for raw_path in list(active_paths or []):
        path = str(raw_path or "").strip()
        if not path or path in seen_selected:
            continue
        seen_selected.add(path)
        selected_paths.append(path)
    if not selected_paths:
        return list(results or [])

    by_path: Dict[str, Dict[str, Any]] = {}
    for doc in list(results or []):
        path = str(doc.get("file_path") or "").strip()
        if path and path not in by_path:
            by_path[path] = doc

    merged: List[Dict[str, Any]] = []
    seen_out = set()
    for path in selected_paths:
        doc = by_path.get(path) or _path_to_selected_scope_doc(path)
        ident = _doc_identity(doc) or path
        if ident in seen_out:
            continue
        seen_out.add(ident)
        merged.append(doc)
    return merged


def _is_primary_source_anchor(doc: Dict[str, Any]) -> bool:
    role = str(doc.get("doc_role") or "").strip().lower()
    if role in {"summary", "explainer", "analysis", "transcript", "ocr_result", "generated_doc"}:
        return False
    if role == "primary_source":
        return True

    file_name = str(doc.get("file_name") or doc.get("file_path") or "").strip()
    ext = os.path.splitext(file_name)[1].lower()

    # When indexing metadata is coarse or missing, treat common raw file types as
    # source anchors unless they are explicitly marked as derived artifacts.
    if ext in {
        ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".csv", ".tsv",
        ".txt", ".md", ".rtf", ".epub", ".jpg", ".jpeg", ".png", ".webp", ".heic",
        ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".mp4", ".mov", ".avi",
        ".mkv", ".webm",
    }:
        return True

    return bool(str(doc.get("file_path") or "").strip())


def _extract_followup_focus_terms(query: str) -> List[str]:
    ql = str(query or "").strip().lower()
    if not ql:
        return []

    tokens = re.findall(r"[a-z0-9]+", ql)
    terms: List[str] = []
    seen = set()
    for token in tokens:
        if len(token) < 3 or token in _FOLLOWUP_TOPIC_STOPWORDS:
            continue
        for candidate in (token, token[:-1] if token.endswith("s") and len(token) > 4 else ""):
            cand = candidate.strip()
            if not cand or cand in _FOLLOWUP_TOPIC_STOPWORDS or cand in seen:
                continue
            seen.add(cand)
            terms.append(cand)
    return terms[:8]


def _filter_results_by_followup_topic(
    results: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    if len(results or []) < 2:
        return list(results or [])

    focus_terms = _extract_followup_focus_terms(query)
    if len(focus_terms) < 2:
        return list(results or [])

    filtered: List[Dict[str, Any]] = []
    debug_rows: List[str] = []
    focus_query = " ".join(focus_terms)
    for doc in list(results or []):
        blob = _build_candidate_affinity_blob(doc)
        blob_lower = blob.lower()
        matched_terms = sum(1 for term in focus_terms if term in blob_lower)
        overlap_score = compute_lookup_overlap_score(focus_query, blob)
        if matched_terms >= 2 or overlap_score >= 2:
            filtered.append(doc)
            debug_rows.append(
                f"{str(doc.get('file_name') or '')}: matched_terms={matched_terms} overlap={overlap_score}"
            )

    if not filtered or len(filtered) == len(results):
        return list(results or [])

    logger.info(
        "[process_previous] topic-focused scoped filter: %s -> %s using terms=%s hits=%s",
        len(results),
        len(filtered),
        focus_terms,
        debug_rows[:8],
    )
    return filtered


def _anchor_scoped_followup_query(
    followup_query: str,
    *,
    previous_user_query: str = "",
    focused_file: str = "",
) -> str:
    query = str(followup_query or "").strip()
    prev = str(previous_user_query or "").strip()
    focus = str(focused_file or "").strip()
    if not query or not prev:
        return query

    try:
        from core.retrieval.lookup_terms import extract_strong_lookup_anchors
    except Exception:
        return query

    anchors = extract_strong_lookup_anchors(prev, max_terms=6)
    if focus:
        anchors.extend(extract_strong_lookup_anchors(focus, max_terms=4))

    kept: List[str] = []
    seen = set()
    query_l = query.lower()
    for raw in anchors:
        anchor = str(raw or "").strip()
        if not anchor:
            continue
        key = anchor.lower()
        if key in seen or key in query_l:
            continue
        seen.add(key)
        kept.append(anchor)

    if not kept:
        return query
    return f"{' '.join(kept[:6])} {query}".strip()


def _detect_media_followup_timestamp_params(
    query: str,
    *,
    focused_file: str = "",
    focused_file_path: str = "",
) -> Optional[Dict[str, Any]]:
    q = str(query or "").strip()
    if not q:
        return None

    from core.intent.media_query_expert import MediaQueryExpert

    ql = q.lower()
    time_sec = MediaQueryExpert._extract_time(ql)
    time_end_sec = MediaQueryExpert._extract_time_range_end(ql)
    if time_sec is None:
        return None

    file_hint = str(focused_file or os.path.basename(str(focused_file_path or "")) or "").strip()
    augmented_query = q
    if file_hint and file_hint.lower() not in ql:
        augmented_query = f"{q} in {file_hint}"

    analyzed = MediaQueryExpert.analyze(augmented_query)
    if analyzed and str(analyzed.get("action") or "").strip() == "media_export":
        params = dict(analyzed.get("params") or {})
        params["query"] = q
        if file_hint and not params.get("file_hint"):
            params["file_hint"] = file_hint
        return params

    params: Dict[str, Any] = {
        "query": q,
        "time_sec": float(time_sec),
        "target_type": MediaQueryExpert._detect_target_type(ql),
        "sub_intent": "point_lookup",
    }
    if time_end_sec is not None:
        params["time_end_sec"] = float(time_end_sec)
    if file_hint:
        params["file_hint"] = file_hint
    return params


def _preserve_primary_source_context(
    refined_docs: List[Dict[str, Any]],
    *,
    original_docs: List[Dict[str, Any]],
    subagent_query: str,
) -> List[Dict[str, Any]]:
    if not refined_docs or not original_docs:
        return refined_docs

    primary_anchors = [doc for doc in original_docs if _is_primary_source_anchor(doc)]
    if len(primary_anchors) < 3:
        return refined_docs

    refined_primary = [doc for doc in refined_docs if _is_primary_source_anchor(doc)]
    if len(refined_primary) >= min(3, len(primary_anchors)):
        return refined_docs

    def _anchor_score(doc: Dict[str, Any]) -> tuple[int, int]:
        return (
            compute_lookup_overlap_score(subagent_query, _build_candidate_affinity_blob(doc)),
            -original_docs.index(doc),
        )

    ranked_anchors = sorted(primary_anchors, key=_anchor_score, reverse=True)
    target_anchor_count = min(len(primary_anchors), max(4, len(refined_primary) + 3), 6)

    merged = list(refined_docs)
    seen = {_doc_identity(doc) for doc in merged if _doc_identity(doc)}
    preserved = len(refined_primary)
    for doc in ranked_anchors:
        identity = _doc_identity(doc)
        if identity and identity in seen:
            continue
        merged.append(doc)
        if identity:
            seen.add(identity)
        preserved += 1
        if preserved >= target_anchor_count:
            break

    if preserved > len(refined_primary):
        logger.info(
            "[process_previous] preserved primary-source context: "
            f"original_primary={len(primary_anchors)} refined_primary={len(refined_primary)} merged_primary={preserved}"
        )
    return merged


def _extract_followup_focus_category(self, query: str) -> str:
    ql = str(query or "").strip().lower()
    if not ql:
        return ""
    if re.search(r"\btext files?\b|文本文件|文本文档", ql, re.IGNORECASE):
        return "document"
    if re.search(r"\bimages?\b|\bpictures?\b|\bphotos?\b|图片|照片", ql, re.IGNORECASE):
        return "image"
    if re.search(r"\baudio\b|\baudios\b|音频|录音", ql, re.IGNORECASE):
        return "audio/video"
    if re.search(r"\bvideos?\b|\bclips?\b|视频|影片", ql, re.IGNORECASE):
        return "audio/video"
    for ck in self._get_rule_category_keywords():
        if ck and ck.lower() in ql:
            cat = self._normalize_category_name(ck)
            if cat and cat not in {"all", "unknown", "other"}:
                return cat
    return ""


def _detect_followup_rewrite_mode(query: str) -> str:
    ql = str(query or "").strip().lower()
    if not ql:
        return ""
    if re.search(r"\b(make it shorter|shorten that|shorter|shorten|briefer|more concise|tldr)\b|缩短|更短|简短一点|更精简", ql, re.IGNORECASE):
        return "shorten"
    if re.search(r"\b(make it more detailed|more detailed|add more detail|expand|elaborate|go on|what next|and then|then what)\b|更详细|展开讲|详细一点|补充细节|然后呢|接着呢|还有呢|后面呢|继续说|继续讲|再展开", ql, re.IGNORECASE):
        return "detail"
    if re.search(r"\b(strongest sources|supporting files|what evidence supports|evidence supports|evidence for)\b|最强来源|支撑文件|支持该总结的证据|证据支持", ql, re.IGNORECASE):
        return "evidence"
    if re.search(r"\bfocus only on\b|只看|只聚焦|只关注", ql, re.IGNORECASE):
        return "focus"
    return ""


_PROCESS_PREVIOUS_MEDIA_EXTS = {
    ".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".ts",
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
}

_MEDIA_EMPTY_AUDIO_PATTERNS = (
    "blank audio",
    "only blank audio",
    "silent audio",
    "empty audio",
    "empty transcript",
    "no speech",
    "no spoken",
    "no dialogue",
    "without speech",
    "without dialogue",
    "静音",
    "无语音",
    "没有语音",
    "空白音频",
    "未检测到语音",
)


def _is_media_followup_doc(doc: Dict[str, Any]) -> bool:
    from core.retrieval.category_engine import is_media_category_value

    file_path = str(doc.get("file_path") or "")
    file_name = str(doc.get("file_name") or os.path.basename(file_path) or "")
    ext = os.path.splitext(file_name)[1].lower()
    if ext in _PROCESS_PREVIOUS_MEDIA_EXTS:
        return True
    return is_media_category_value(doc.get("doc_category") or doc.get("doc_category_family"))


def _looks_like_empty_audio_summary(text: str) -> bool:
    tl = str(text or "").strip().lower()
    if not tl:
        return True
    return any(pat in tl for pat in _MEDIA_EMPTY_AUDIO_PATTERNS)


def _merge_media_chunks_for_followup(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    *,
    max_chars: int = 6000,
) -> Dict[str, str]:
    meta_lines: List[str] = []
    summary_line = ""
    keyframe_entries: List[tuple[float, str]] = []
    asr_entries: List[str] = []

    for chunk_doc, chunk_meta in zip(documents or [], metadatas or []):
        ctype = str((chunk_meta or {}).get("chunk_type") or "")
        chunk_doc = str(chunk_doc or "").strip()
        if not chunk_doc:
            continue
        if ctype == "media_summary":
            for line in chunk_doc.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("内容摘要:"):
                    summary_line = line[len("内容摘要:"):].strip()
                elif line.lower().startswith("content summary:"):
                    summary_line = line.split(":", 1)[-1].strip()
                else:
                    meta_lines.append(line)
        elif ctype == "media_audio_summary":
            audio_line = ""
            for line in chunk_doc.splitlines():
                line = line.strip()
                if line.startswith("音频摘要:"):
                    audio_line = line[len("音频摘要:"):].strip()
                elif line.lower().startswith("audio summary:"):
                    audio_line = line.split(":", 1)[-1].strip()
            if audio_line and not _looks_like_empty_audio_summary(audio_line):
                asr_entries.append(audio_line)
        elif ctype == "media_visual_summary":
            visual_line = ""
            for line in chunk_doc.splitlines():
                line = line.strip()
                if line.startswith("画面摘要:"):
                    visual_line = line[len("画面摘要:"):].strip()
                elif line.lower().startswith("visual summary:"):
                    visual_line = line.split(":", 1)[-1].strip()
            if visual_line:
                keyframe_entries.append((-1.0, visual_line))
        elif ctype == "keyframe":
            try:
                t = float((chunk_meta or {}).get("keyframe_time_sec", 0) or 0)
            except Exception:
                t = 0.0
            keyframe_entries.append((t, chunk_doc))
        elif ctype == "asr_transcript":
            asr_entries.append(chunk_doc)

    keyframe_entries.sort(key=lambda item: item[0])

    visual_snippets = [desc for _, desc in keyframe_entries if desc][:2]
    nonempty_asr_entries = [seg for seg in asr_entries if seg and not _looks_like_empty_audio_summary(seg)]
    speech_snippets = nonempty_asr_entries[:2]

    composed_summary_parts: List[str] = []
    if summary_line and not _looks_like_empty_audio_summary(summary_line):
        composed_summary_parts.append(summary_line)
    if visual_snippets:
        composed_summary_parts.append("Visuals: " + " ".join(visual_snippets))
    if speech_snippets:
        composed_summary_parts.append("Audio: " + " ".join(speech_snippets))
    if not composed_summary_parts and summary_line:
        composed_summary_parts.append(summary_line)

    merged_parts: List[str] = []
    if summary_line:
        merged_parts.append(f"[Media Summary]\n{summary_line}")
    if meta_lines:
        merged_parts.append("[Media Metadata]\n" + "\n".join(meta_lines))
    if keyframe_entries:
        merged_parts.append(
            "[Visual Timeline]\n" + "\n\n".join(entry for _, entry in keyframe_entries if entry)
        )
    if asr_entries:
        merged_parts.append(
            "[Speech Transcript Segments]\n" + "\n".join(seg for seg in asr_entries if seg)
        )

    merged_text = "\n\n".join(part for part in merged_parts if part.strip()).strip()
    return {
        "summary": " ".join(part.strip() for part in composed_summary_parts if part.strip())[:1200].strip(),
        "text": merged_text[:max_chars].strip(),
    }


def _build_media_followup_context(
    kb: Any,
    file_path: str,
    *,
    max_chars: int = 6000,
) -> Dict[str, str]:
    all_chunks = kb.collection.get(
        where={"file_path": file_path},
        include=["documents", "metadatas"],
    )
    return _merge_media_chunks_for_followup(
        list(all_chunks.get("documents") or []),
        list(all_chunks.get("metadatas") or []),
        max_chars=max_chars,
    )


def _hydrate_media_results_for_followup(
    results: List[Dict[str, Any]],
    *,
    kb: Any,
    max_files: int = 4,
    max_chars: int = 6000,
) -> int:
    hydrated = 0
    for doc in results or []:
        if hydrated >= max_files:
            break
        if not _is_media_followup_doc(doc):
            continue
        file_path = str(doc.get("file_path") or "").strip()
        if not file_path:
            continue
        merged = _build_media_followup_context(kb, file_path, max_chars=max_chars)
        merged_text = str(merged.get("text") or "").strip()
        merged_summary = str(merged.get("summary") or "").strip()
        if not merged_text and not merged_summary:
            continue
        if merged_text:
            doc["_media_followup_text"] = merged_text
            doc["text"] = merged_text
        if merged_summary:
            doc["_media_followup_summary"] = merged_summary
            existing_summary = str(doc.get("doc_summary") or "").strip()
            if not existing_summary or _looks_like_empty_audio_summary(existing_summary):
                doc["doc_summary"] = merged_summary
        hydrated += 1
    return hydrated


def _select_followup_doc_summary(doc: Dict[str, Any], path_to_summary: Dict[str, str]) -> str:
    if _is_media_followup_doc(doc):
        merged_summary = str(doc.get("_media_followup_summary") or "").strip()
        if merged_summary:
            return merged_summary
    file_path = str(doc.get("file_path") or "")
    return str(
        doc.get("doc_summary")
        or path_to_summary.get(file_path, "")
        or doc.get("text")
        or ""
    ).strip()


def _select_followup_doc_text(doc: Dict[str, Any]) -> str:
    if _is_media_followup_doc(doc):
        merged_text = str(doc.get("_media_followup_text") or "").strip()
        if merged_text:
            return merged_text
    return str(doc.get("text") or "").strip()


def _is_collective_summary_request(query: str) -> bool:
    q = str(query or "").strip()
    if not q:
        return False
    ql = q.lower()
    summary_signal = bool(
        re.search(
            r'\b(summary|summarize|overview|recap|wrap\s*up|conclusions?|findings?|'
            r'main\s+points?|key\s+points?|key\s+takeaways?)\b'
            r'|总结|概括|归纳|汇总|结论|要点',
            ql,
            re.IGNORECASE,
        )
    )
    collective_ref = bool(
        re.search(
            r'\b(it|this|that|this file|that file|current file|them|these|those|'
            r'all of them|all these files|these files|those files)\b'
            r'|它|这个|这份|该文件|这份文件|这个文件|他们|它们|这些|那些|这批|这几个|这几份|这些文件|这些视频|这些音频',
            ql,
            re.IGNORECASE,
        )
    )
    direct_collective_overview = bool(
        re.search(
            r'\b(tell\s+me\s+(?:more\s+)?about|describe|explain|introduce)\s+'
            r'(them|they|these|those|these\s+files|those\s+files|all\s+of\s+them)\b'
            r'|\bwhat\s+are\s+(they|these|those|these\s+files|those\s+files)\s+about\b'
            r'|介绍一下(它们|他们|这些|那些|这些文件|那些文件|这几份文件)'
            r'|说说(它们|他们|这些|那些|这些文件|那些文件|这几份文件)'
            r'|讲讲(它们|他们|这些|那些|这些文件|那些文件|这几份文件)'
            r'|(这些|那些|这几份|这批)文件.*(讲了什么|是什么|主要内容|内容是什么)',
            ql,
            re.IGNORECASE,
        )
    )
    return (summary_signal and collective_ref) or direct_collective_overview


def _is_simple_collective_followup(query: str) -> bool:
    ref_target = classify_reference_target(query)
    return ref_target.get("kind") == "deictic"


def _build_scoped_collective_summary_prompt(
    self,
    *,
    docs: List[Dict[str, Any]],
    original_question: str,
    user_lang: str,
    path_to_summary: Dict[str, str],
) -> str:
    from collections import defaultdict

    conclusion_requested = bool(
        re.search(
            r"\b(?:conclusion|conclusions|bottom\s+line|takeaway|takeaways)\b"
            r"|结论|最终结论|核心结论",
            str(original_question or ""),
            re.IGNORECASE,
        )
    )
    few_files_mode = len(list(docs or [])) <= 5
    grouped_files = defaultdict(list)
    for doc in list(docs or []):
        cat = self._normalize_category_name(str(doc.get("doc_category") or "other"))
        grouped_files[cat].append(doc)

    header = (
        "你是文件分析助手。用户正在追问上一轮检索出的当前相关文件集合，请基于这些文件做整体总结。\n\n"
        if user_lang == "zh"
        else "You are a file analysis assistant. The user is asking for a collective summary of the current relevant files from the previous turn.\n\n"
    )

    context_lines: List[str] = []
    context_lines.append(
        f"当前文件数: {len(docs)}" if user_lang == "zh" else f"Current file count: {len(docs)}"
    )

    for cat, items in grouped_files.items():
        if user_lang == "zh":
            context_lines.append(f"[{cat}] 共 {len(items)} 份")
        else:
            context_lines.append(f"[{cat}] {len(items)} file(s)")
        for idx, doc in enumerate(items[:12], 1):
            fname = str(doc.get("file_name") or os.path.basename(str(doc.get("file_path") or "")) or "Unknown")
            summary = _select_followup_doc_summary(doc, path_to_summary)
            line = f"- {fname}"
            if summary:
                line += f": {summary[:420]}"
            context_lines.append(line)
        if len(items) > 12:
            remain = len(items) - 12
            context_lines.append(
                f"... 以及另外 {remain} 份同类文件"
                if user_lang == "zh"
                else f"... plus {remain} more file(s) in this category"
            )
        context_lines.append("")

    if user_lang == "zh":
        if few_files_mode:
            instructions = (
                "请直接输出针对当前这批文件的总结。\n"
                "要求：\n"
                "1. 当前文件数不超过 5 份，请按文件逐个介绍，每份文件单独一小段。\n"
                "2. 每份文件都要说明核心内容、主要信息和它的大致用途/价值。\n"
                "3. 先不要做泛泛的总体概括来替代逐个介绍。\n"
                "4. 只基于下面列出的当前文件回答，不要扩展到其他文件。\n"
                "5. 不要写路由说明、任务说明，也不要说“根据你的请求/我将为你”。\n"
                "6. 必须使用中文回答。\n"
            )
        else:
            instructions = (
                "请直接输出针对当前这批文件的整体总结。\n"
                "要求：\n"
                "1. 当前文件数超过 5 份，请先给一句总体概括，说明这批 relevant files 整体在讲什么。\n"
                "2. 再给 3-6 条归纳后的关键点，尽量按主题或类型分组，而不是逐个文件罗列。\n"
                "3. 如果有代表性的文件，可以点名 1-3 个作为例子，但不要退化成逐文件流水账。\n"
                "4. 结论必须覆盖当前文件集合的共性与差异，不能只总结单个文件。\n"
                "5. 只基于下面列出的当前文件回答，不要扩展到其他文件。\n"
                "6. 不要写路由说明、任务说明，也不要说“根据你的请求/我将为你”。\n"
                "7. 必须使用中文回答。\n"
            )
    else:
        if few_files_mode:
            instructions = (
                "Produce a summary of the CURRENT scoped files.\n"
                "Requirements:\n"
                "1. There are 5 or fewer files, so describe them one by one, with a short paragraph for each file.\n"
                "2. For each file, explain the core content, main information, and likely value/use.\n"
                "3. Do not replace the per-file explanation with only a high-level overview.\n"
                "4. Stay grounded in the files listed below only.\n"
                "5. Do not include routing/meta commentary such as 'Based on your request' or 'I will provide'.\n"
                "6. Respond in English.\n"
            )
        else:
            instructions = (
                "Produce a collective summary of the CURRENT scoped files.\n"
                "Requirements:\n"
                "1. There are more than 5 files, so start with one overall overview sentence explaining what this relevant file set is about.\n"
                "2. Then provide 3-6 synthesized takeaways, grouped by themes or file types when possible, instead of listing files one by one.\n"
                "3. You may cite 1-3 representative files as examples, but do not turn this into a file-by-file list.\n"
                "4. Your answer must cover the common patterns and important differences across the current file set, not just a single file.\n"
                "5. Stay grounded in the files listed below only.\n"
                "6. Do not include routing/meta commentary such as 'Based on your request' or 'I will provide'.\n"
                "7. Respond in English.\n"
            )

    if conclusion_requested:
        if user_lang == "zh":
            instructions += "8. 用户明确在问“结论”，请先输出一行以“结论：”开头的核心结论，再补充支撑要点。\n"
        else:
            instructions += "8. The user explicitly asked for the conclusion. Start with one line beginning with 'Conclusion:' before the supporting details.\n"

    return (
        header
        + (f"<用户追问>\n{original_question}\n</用户追问>\n\n" if user_lang == "zh" else f"<User follow-up>\n{original_question}\n</User follow-up>\n\n")
        + ("<当前相关文件>\n" if user_lang == "zh" else "<Current relevant files>\n")
        + "\n".join(context_lines)
        + ("\n</当前相关文件>\n\n" if user_lang == "zh" else "\n</Current relevant files>\n\n")
        + instructions
    )


def _handle_process_previous(
    self,
    original_question: str,
    session_id: Optional[str] = None,
    prompt_language: Optional[str] = None,
    active_paths: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
):
    import os
    try:
        def _done_event(query_type: str = "process", *, ok: bool = True, sources_payload: Optional[List[Dict[str, Any]]] = None):
            payload: Dict[str, Any] = {"type": "done", "query_type": query_type, "ok": ok}
            if sources_payload is not None:
                payload["sources"] = _dedupe_docs_by_file_identity(list(sources_payload))
            return payload

        lang = self._resolve_prompt_language(prompt_language, question=original_question, session_id=session_id)
        user_lang = lang
        results = self._get_last_search_results_ref(session_id)
        count_scope_ctx = self._get_count_scope_context(session_id) or {}
        structured_plan = None
        media_followup_plan = None

        if results and _looks_like_cross_result_followup(original_question, params):
            recent_getter = getattr(self, "_get_recent_search_result_sets", None)
            recent_sets = recent_getter(session_id, limit=4) if callable(recent_getter) else []
            if recent_sets:
                before_recent = len(results or [])
                merged_recent = _merge_recent_result_sets_for_followup(
                    list(results or []),
                    list(recent_sets or []),
                )
                if len(merged_recent) > before_recent:
                    logger.info(
                        "[process_previous] cross-result follow-up widened context: %s -> %s using %s recent set(s)",
                        before_recent,
                        len(merged_recent),
                        len(recent_sets or []),
                    )
                    results = merged_recent

        if active_paths is not None:
            filtered = []
            for doc in (results or []):
                path = str(doc.get("file_path") or "")
                if any(path == ap or path.startswith(ap.rstrip(os.sep) + os.sep) for ap in active_paths):
                    filtered.append(doc)
            logger.info(f"[process_previous] active_paths filter: {len(results or [])} -> {len(filtered)} results")
            if filtered:
                results = filtered
            elif results:
                logger.info(
                    f"[process_previous] active_paths filter produced 0 results but session has {len(results)} results; keeping session-scoped context"
                )

        if _params_request_selected_scope(params) and active_paths:
            before_count = len(results or [])
            results = _merge_selected_active_scope_results(list(results or []), active_paths)
            logger.info(
                "[process_previous] selected active scope materialized: %s -> %s results",
                before_count,
                len(results or []),
            )

        source_scope_results = list(results or [])

        try:
            skill_name = str((params or {}).get("_skill_name") or "").strip().lower()
            if skill_name in {"contextual_refine", "list_selected", "summarize_selected", "process_previous"}:
                structured_plan = ContextualRefineSkill.plan_from_params(
                    original_question,
                    params,
                    active_paths=active_paths,
                    last_results=results,
                )
                if structured_plan.focus_extension:
                    filtered_results = ContextualRefineSkill.filter_results(
                        results,
                        focus_extension=structured_plan.focus_extension,
                        focus_extensions=structured_plan.focus_extensions,
                    )
                    if filtered_results:
                        logger.info(
                            "[process_previous] structured focus filter %s: %s -> %s",
                            ",".join(structured_plan.focus_extensions or ([structured_plan.focus_extension] if structured_plan.focus_extension else [])),
                            len(results or []),
                            len(filtered_results),
                        )
                        results = filtered_results
                logger.info(
                    "[process_previous] structured plan scope=%s operation=%s rewrite=%s reason=%s",
                    structured_plan.scope,
                    structured_plan.operation,
                    structured_plan.rewrite_mode,
                    structured_plan.reason[:120],
                )

            if skill_name in {"media_followup", "media_timequery"}:
                media_followup_plan = MediaFollowupSkill.plan_from_params(
                    original_question,
                    params,
                    last_results=results,
                    active_paths=active_paths,
                )
                logger.info(
                    "[process_previous] media follow-up plan operation=%s media_type=%s file_hint=%s",
                    media_followup_plan.operation,
                    media_followup_plan.media_type,
                    media_followup_plan.file_hint,
                )
        except Exception as exc:
            logger.warning("[process_previous] structured follow-up planning failed: %s", exc)
            structured_plan = None
            media_followup_plan = None

        collective_summary_request = _is_collective_summary_request(original_question)
        followup_rewrite_mode = _detect_followup_rewrite_mode(original_question)
        force_contextual_answer = bool(
            (params or {}).get("_force_answer_rewrite")
            or (params or {}).get("_single_media_summary")
        )

        explicit_focus_doc = None
        if (
            results
            and not collective_summary_request
            and not followup_rewrite_mode
            and not _params_request_selected_scope(params)
            and not _looks_like_cross_result_followup(original_question, params)
        ):
            explicit_focus_doc = _resolve_explicit_focus_doc_from_results(original_question, list(results or []))
            if explicit_focus_doc is not None:
                logger.info(
                    "[process_previous] narrowed follow-up scope to explicit file focus: %s",
                    str(explicit_focus_doc.get("file_name") or explicit_focus_doc.get("file_path") or "").strip(),
                )
                results = [explicit_focus_doc]

        total_scope_files = int(count_scope_ctx.get("total_files") or 0)
        shown_files = len(results or [])
        emit_total = total_scope_files if total_scope_files > 0 and explicit_focus_doc is None else shown_files

        display_results = list(results or [])
        if structured_plan and structured_plan.scope in {"selected_items", "selected_folder"} and source_scope_results:
            display_results = list(source_scope_results)

        if display_results:
            display_payload = _dedupe_docs_by_file_identity(display_results)
            yield {
                "type": "files",
                "preview": display_payload[:50],
                "all": display_payload,
                "total_matches": emit_total if emit_total > len(display_payload) else len(display_payload),
                "shown_count": len(display_payload),
            }

        # ================= NEW SUBAGENT LOGIC =================
        subagent_action = "search"
        subagent_query = original_question
        subagent_media_type = "all"
        subagent_search_concept = ""
        subagent_media_params: Dict[str, Any] = {}
        focused_file = ""
        focused_file_path = ""
        ref_target = classify_reference_target(original_question)

        can_stay_in_context = bool(
            collective_summary_request
            or followup_rewrite_mode
            or (structured_plan and structured_plan.operation in {"rewrite", "summary", "overview", "qa", "support"})
            or media_followup_plan
        )
        if not can_stay_in_context and ref_target.get("kind") == "deictic":
            try:
                from core.intent.context_followup_expert import ContextFollowupExpert

                can_stay_in_context = bool(
                    ContextFollowupExpert._CONTENT_QUESTION_RE.search(original_question.strip())
                    or ContextFollowupExpert._POST_COUNT_CONTENT_RE.search(original_question.lower())
                    or any(p.search(original_question.lower()) for p in ContextFollowupExpert._IMPLICIT_PATTERNS)
                )
            except Exception as exc:
                logger.debug("[process_previous] deictic contextual QA heuristic failed: %s", exc)
        if (
            ref_target.get("kind") == "deictic"
            and ref_target.get("number") == "singular"
            and len(results or []) != 1
            and not can_stay_in_context
        ):
            msg = (
                "我还没法确定你指的是哪一份文件。请直接说出文件名，我就继续帮你看。"
                if user_lang == "zh"
                else "I can't tell which file you mean yet. Please name the file and I'll continue."
            )
            yield {"type": "text", "delta": msg}
            yield _done_event("clarify", sources_payload=list(source_scope_results or results or []))
            return

        # ── Detect if context is predominantly audio/video ──────────────────
        _MEDIA_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
                       ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv"}
        from core.retrieval.category_engine import is_media_category_value

        _media_files = [
            d for d in (results or [])
            if os.path.splitext(str(d.get("file_name") or ""))[1].lower() in _MEDIA_EXTS
               or is_media_category_value(d.get("doc_category") or d.get("doc_category_family"))
        ]
        _ctx_is_media = len(_media_files) > 0 and len(_media_files) >= len(results or []) // 2

        try:
            from core.intent.context_followup_expert import ContextFollowupExpert
            # Personal attribute follow-ups are allowed to stay in process_previous,
            # but they should bypass the subagent LLM router and go straight to a
            # scoped search with the inherited entity anchor.
            if ContextFollowupExpert._ATTR_LOOKUP_RE.search(original_question.lower()):
                resolved_attr = self._resolve_pronoun_query(
                    original_question,
                    session_id=session_id,
                    prompt_language=user_lang,
                )
                subagent_action = "search"
                subagent_query = str((resolved_attr or {}).get("resolved_query") or original_question).strip()
                subagent_search_concept = subagent_query
                logger.info(
                    f"[process_previous] personal attribute fast-path → search: "
                    f"query_chars={len(subagent_query or '')} entity_chars={len(str((resolved_attr or {}).get('entity') or '').strip())}"
                )
            else:
                hist_ref = self._get_history_ref(session_id)
                prev_user_query = ""
                prev_answer_preview = ""
                prev_answer_text = ""
                if hist_ref:
                    prev_user_query = str(hist_ref[-1].get("q") or "").strip()
                    prev_answer_text = str(hist_ref[-1].get("a") or "").strip()
                    prev_answer_preview = prev_answer_text[:180]

                if media_followup_plan and media_followup_plan.operation in {"time_lookup", "range_summary"}:
                    subagent_action = "media_export"
                    subagent_query = media_followup_plan.query or original_question
                    subagent_search_concept = ""
                    subagent_media_type = media_followup_plan.media_type
                    subagent_media_params = {
                        "time_sec": media_followup_plan.time_sec,
                        "target_type": media_followup_plan.target_type,
                        "media_type": media_followup_plan.media_type,
                    }
                    if media_followup_plan.file_hint:
                        subagent_media_params["file_hint"] = media_followup_plan.file_hint
                    if media_followup_plan.time_end_sec is not None:
                        subagent_media_params["time_end_sec"] = media_followup_plan.time_end_sec
                        subagent_media_params["sub_intent"] = "range_summary"
                    logger.info(
                        "[process_previous] structured media follow-up → media_export: %s",
                        {
                            "time_sec": subagent_media_params.get("time_sec"),
                            "time_end_sec": subagent_media_params.get("time_end_sec"),
                            "target_type": subagent_media_params.get("target_type"),
                            "file_hint": subagent_media_params.get("file_hint"),
                        },
                    )
                elif media_followup_plan and media_followup_plan.operation == "topic_search":
                    subagent_action = "media_content_search"
                    subagent_query = media_followup_plan.query or original_question
                    subagent_search_concept = subagent_query
                    subagent_media_type = media_followup_plan.media_type
                    if media_followup_plan.file_hint:
                        subagent_media_params["file_hint"] = media_followup_plan.file_hint
                    logger.info(
                        "[process_previous] structured media follow-up → media_content_search: query=%r media_type=%s",
                        subagent_query,
                        subagent_media_type,
                    )
                elif media_followup_plan and media_followup_plan.operation in {"summary", "rewrite"}:
                    subagent_action = "global_summary" if media_followup_plan.operation == "summary" else "answer_rewrite"
                    subagent_query = media_followup_plan.query or original_question
                    subagent_search_concept = ""
                    logger.info(
                        "[process_previous] structured media follow-up → %s: file_hint=%s",
                        subagent_action,
                        media_followup_plan.file_hint,
                    )
                elif structured_plan:
                    if (
                        structured_plan.operation == "rewrite"
                        or structured_plan.rewrite_mode
                        or force_contextual_answer
                        or (
                            _ctx_is_media
                            and len(results or []) <= 3
                            and structured_plan.operation in {"qa", "support"}
                        )
                        or (
                            structured_plan.scope in {"selected_items", "selected_folder"}
                            and structured_plan.operation in {"qa", "support"}
                        )
                    ):
                        subagent_action = "answer_rewrite"
                        subagent_query = original_question
                        subagent_search_concept = structured_plan.query or original_question
                    elif structured_plan.operation in {"summary", "overview"}:
                        subagent_action = "global_summary"
                        subagent_query = ""
                        subagent_search_concept = ""
                    else:
                        subagent_action = "search"
                        subagent_query = structured_plan.query or original_question
                        subagent_search_concept = subagent_query
                    logger.info(
                        "[process_previous] structured contextual route → %s: scope=%s query=%r",
                        subagent_action,
                        structured_plan.scope,
                        subagent_query,
                    )
                elif followup_rewrite_mode:
                    subagent_action = "answer_rewrite"
                    subagent_query = original_question
                    subagent_search_concept = original_question
                    logger.info(
                        f"[process_previous] rewrite fast-path → answer_rewrite: mode={followup_rewrite_mode} "
                        f"query_chars={len(original_question or '')}"
                    )
                elif _is_simple_collective_followup(original_question):
                    if collective_summary_request:
                        subagent_action = "global_summary"
                        subagent_query = ""
                        subagent_search_concept = ""
                        logger.info(
                            f"[process_previous] collective summary fast-path → global_summary: "
                            f"query_chars={len(original_question or '')}"
                        )
                    else:
                        subagent_action = "search"
                        subagent_query = ""
                        subagent_search_concept = ""
                        logger.info(
                            f"[process_previous] simple collective follow-up fast-path → answer current scoped files: "
                            f"query_chars={len(original_question or '')}"
                        )
                else:
                    from langchain_core.messages import SystemMessage
                    import json
                    import re

                    subagent_llm = get_llm(streaming=False, session_id=session_id)
                    if hasattr(subagent_llm, "force_text_model"):
                        subagent_llm.force_text_model = True

                    preview_docs = list(results or [])[:8]
                    docs_preview = "\n".join([f"- {d.get('file_name', 'Unknown')}" for d in preview_docs])
                    if preview_docs:
                        focused_file = str(
                            preview_docs[0].get("file_name")
                            or os.path.basename(str(preview_docs[0].get("file_path") or ""))
                            or ""
                        ).strip()
                        focused_file_path = str(preview_docs[0].get("file_path") or "").strip()

                    media_timestamp_params = (
                        _detect_media_followup_timestamp_params(
                            original_question,
                            focused_file=focused_file,
                            focused_file_path=focused_file_path,
                        )
                        if _ctx_is_media
                        else None
                    )
                    if media_timestamp_params:
                        subagent_action = "media_export"
                        subagent_query = original_question
                        subagent_search_concept = ""
                        subagent_media_params = media_timestamp_params
                        logger.info(
                            "[process_previous] media timestamp fast-path → media_export: %s",
                            {
                                "time_sec": subagent_media_params.get("time_sec"),
                                "time_end_sec": subagent_media_params.get("time_end_sec"),
                                "target_type": subagent_media_params.get("target_type"),
                                "file_hint": subagent_media_params.get("file_hint"),
                            },
                        )
                    elif _ctx_is_media:
                        # ── Media-aware routing prompt ───────────────────────────────
                        subagent_prompt = f"""You are a smart context router for audio/video files.
The user is asking a follow-up about {emit_total} audio/video file(s).
Files in context:
{docs_preview}

Previous user query: "{prev_user_query}"
Focused file in context: "{focused_file or '(none)'}"
Previous answer preview: "{prev_answer_preview}"

User Query: "{original_question}"

Analyze the intent. Output ONLY a valid JSON object:
{{
   "action": "media_content_search" | "media_export" | "global_summary" | "search",
   "query": "the search concept or topic if action is media_content_search or search",
   "time_sec": null or float (seconds into the media),
   "target_type": "audio_content" | "video_audio" | "video_visual",
   "media_type": "audio" | "video" | "all",
   "reason": "..."
}}

Rules:
- Use "media_content_search" if the user asks what was discussed/mentioned/said about a topic across the audio/video files (e.g. '\u63d0\u5230\u4e86\u4ec0\u4e48', 'what about X', '\u4e3b\u8981\u5185\u5bb9', '\u8ba8\u8bba\u4e86\u4ec0\u4e48').
- Use "media_export" if the user asks about a specific timestamp (e.g. '\u7b2c30\u79d2', 'at 1:20', '\u5728X\u5206\u949f\u5904').
- Use "global_summary" if the user wants an overall summary or overview of the media content.
- Use "search" only if the user is looking for specific files or non-media content.
- Prefer interpreting the question relative to the current media files in context, not as a brand-new global search.
- For media_content_search: fill in "query" with the topic/concept to find.
- For media_export: fill in "time_sec" (float) and "target_type".
OUTPUT JSON ONLY, NO MARKDOWN FORMATTING:"""
                    else:
                        # ── Generic (non-media) routing prompt ────────────────────────
                        subagent_prompt = f"""You are a smart context router subagent.
The user is asking a follow-up query based on a previous context of {emit_total} files.
Sample files in context:
{docs_preview}

Previous user query: "{prev_user_query}"
Focused file in context: "{focused_file or '(none)'}"
Previous answer preview: "{prev_answer_preview}"
Previous answer excerpt:
\"\"\"
{prev_answer_text[:1200]}
\"\"\"

User Query: "{original_question}"

Analyze the intent. Output ONLY a valid JSON object:
{{
   "action": "global_summary" | "search" | "answer_rewrite",
   "query": "the exact anchored keywords/entity to search for inside the current files if action is search",
   "reason": "..."
}}

Rules:
- Use "global_summary" if the user wants an overview, statistics, or summarization of the context (e.g. '\u4ecb\u7ecd\u4e00\u4e0b', '\u603b\u7ed3', 'what is this about', 'tell me about them', 'what are these files about').
- Use "global_summary" ALSO if there are very few files (e.g. 1-3) and the user wants to understand their content in general.
- Use "search" ONLY if looking for a specific entity, topic, detail, or concept (e.g. 'does it mention sensevoice').
- Use "answer_rewrite" if the user is revising the PREVIOUS ANSWER itself: making it shorter/longer, focusing it on a subset, or asking for evidence/strongest sources supporting that summary.
- Important: this is still a follow-up on the PREVIOUS files. If the user asks about a detail such as writing style, target user, revenue model, brand wins, node version, or whether the file mentions something, keep the interpretation anchored to the current file context.
- If there is a focused file or a small set of files in context, assume abstract references like "this article", "the profile", "the report", "the guide", or omitted pronouns refer to that context.
- When action is "search", preserve distinctive anchors from the previous user query or focused file in the query: proper names, product codes, invoice/order numbers, amounts, exact filenames, company names, and bilingual title terms. Do not search only generic detail words such as "amount", "model", "setup", "phone", "address", or "conclusion".
- Examples of good search queries: "MODEL-123 model", "VendorName 123.45 invoice amount", "PersonName phone". Bad search queries: "model", "amount", "phone".
- Do NOT broaden to unrelated files outside the current context. If action is "search", the query should be concise keywords for searching WITHIN the current files only.
OUTPUT JSON ONLY, NO MARKDOWN FORMATTING:"""
                
                    yield {"type": "thinking", "delta": f"\U0001f914 \u6b63\u5728\u7406\u89e3\u60a8\u7684\u4e0a\u4e0b\u6587\u610f\u56fe...\n\n" if user_lang == "zh" else f"\U0001f914 Analyzing your context intent...\n\n"}
                    resp = subagent_llm.invoke([SystemMessage(content=subagent_prompt)])
                    r_text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
                    
                    match = re.search(r'\{.*?\}', r_text, re.DOTALL)
                    parsed_json = json.loads(match.group(0)) if match else json.loads(r_text)
                    
                    subagent_action = parsed_json.get("action", "search")
                    if subagent_action in ("search", "media_content_search") and parsed_json.get("query"):
                        subagent_query = str(parsed_json.get("query")).strip()
                        subagent_search_concept = subagent_query
                    if subagent_action == "search" and subagent_query:
                        anchored_query = _anchor_scoped_followup_query(
                            subagent_query,
                            previous_user_query=prev_user_query,
                            focused_file=focused_file,
                        )
                        if anchored_query != subagent_query:
                            logger.info(
                                "[process_previous] anchored scoped follow-up query: %r -> %r",
                                subagent_query,
                                anchored_query,
                            )
                            subagent_query = anchored_query
                            subagent_search_concept = anchored_query
                    if _ctx_is_media and subagent_action == "media_content_search":
                        subagent_media_type = str(parsed_json.get("media_type") or "all")
                    
                    logger.info(
                        f"[Subagent Router] Intent: {subagent_action}, Query: {subagent_query}, "
                        f"media={_ctx_is_media}, Reason: {parsed_json.get('reason', '')}"
                    )
        except Exception as e:
            logger.warning(f"[process_previous] Subagent routing failed: {e}")
            subagent_action = "global_summary" if emit_total > 50 else "search"
        # ==========================================================

        derived_scope_ctx: Dict[str, Any] = {}
        topic_focused_results = list(results or [])
        if results and (collective_summary_request or bool(followup_rewrite_mode)):
            topic_focused_results = _filter_results_by_followup_topic(
                list(results or []),
                original_question,
            )

        if not count_scope_ctx and topic_focused_results and collective_summary_request:
            try:
                derived_scope_ctx = self._build_count_scope_from_sources(list(topic_focused_results or []))
            except Exception as e:
                logger.warning(f"[process_previous] derive scoped summary context failed: {e}")
                derived_scope_ctx = {}
            if derived_scope_ctx:
                count_scope_ctx = derived_scope_ctx
                total_scope_files = int(count_scope_ctx.get("total_files") or len(topic_focused_results or []))
                shown_files = len(topic_focused_results or [])
                emit_total = total_scope_files if total_scope_files > 0 else shown_files

        if topic_focused_results and (
            collective_summary_request or subagent_action == "answer_rewrite"
        ):
            if len(topic_focused_results) != len(results or []):
                results = list(topic_focused_results)
                file_payload = _dedupe_docs_by_file_identity(results)
                yield {
                    "type": "files",
                    "preview": file_payload[:50],
                    "all": file_payload,
                    "total_matches": len(file_payload),
                    "shown_count": len(file_payload),
                }
            else:
                results = list(topic_focused_results)

        use_count_scope_summary = False
        if subagent_action == "global_summary" and bool(count_scope_ctx):
            if derived_scope_ctx:
                use_count_scope_summary = (
                    total_scope_files > max(0, shown_files)
                    or total_scope_files >= 12
                )
            else:
                use_count_scope_summary = (
                    total_scope_files > max(0, shown_files)
                    or total_scope_files > 50
                    or collective_summary_request
                )

        _pp_matched_cat = ""

        if use_count_scope_summary:
            _cat_kws = self._get_rule_category_keywords()
            _specific_cat_kws = [k for k in _cat_kws if not self._is_generic_file_scope_category(k)]
            _oq_lower = subagent_query.lower()
            _pp_matched_cat = ""
            for ck in _specific_cat_kws:
                if ck.lower() in _oq_lower:
                    _pp_matched_cat = self._normalize_category_name(ck)
                    break
            if _pp_matched_cat and _pp_matched_cat not in {"", "all", "other", "unknown", "document"}:
                logger.info(
                    f"[process_previous] user-specified category '{_pp_matched_cat}'; routing to search(category={_pp_matched_cat}) instead of full summary"
                )
                use_count_scope_summary = False

        if use_count_scope_summary:
            category_counts = list(count_scope_ctx.get("category_counts") or [])
            samples_by_category = dict(count_scope_ctx.get("samples_by_category") or {})
            prompt_cats_limit = max(1, int(os.getenv("PROCESS_PREV_PROMPT_CATEGORIES", "14")))
            prompt_samples_per_cat = max(1, int(os.getenv("PROCESS_PREV_PROMPT_SAMPLES_PER_CATEGORY", "8")))
            prompt_summary_chars = max(80, int(os.getenv("PROCESS_PREV_PROMPT_SUMMARY_CHARS", "260")))
            cat_lines = []
            for item in category_counts[:20]:
                cat = str(item.get("category") or "").strip()
                cnt = int(item.get("count") or 0)
                if cat:
                    cat_lines.append(f"- {cat}: {cnt}")
            if not cat_lines:
                cat_lines.append("- other: unknown")

            sample_lines: List[str] = []
            for c in category_counts[:prompt_cats_limit]:
                cat = str(c.get("category") or "").strip()
                if not cat:
                    continue
                sitems = list(samples_by_category.get(cat) or [])[:prompt_samples_per_cat]
                if not sitems:
                    continue
                sample_lines.append(f"[{cat}]")
                for s in sitems:
                    name = str(s.get("file_name") or "").strip()
                    summ = str(s.get("doc_summary") or "").strip()
                    if summ:
                        sample_lines.append(f"- {name}: {summ[:prompt_summary_chars]}")
                    else:
                        sample_lines.append(f"- {name}")
            if not sample_lines:
                sample_lines.append("(no sample summaries)")

            if user_lang == "zh":
                prompt = (
                    "你是文件分析助手。用户正在追问“上一轮 count(all) 的全部文件”。\n"
                    "请基于“全量统计 + 分类代表样本”做结论归纳。\n"
                    "- 先给总览结论，再给3-6条关键发现。\n"
                    "- 必须覆盖全量范围趋势（不是仅50条样本）。\n"
                    "- 不要逐条复述样本，也不要声称你阅读了每一份原文。\n"
                    "- 明确这是基于元数据与分类样本的归纳。\n\n"
                    f"<用户要求>\n{original_question}\n</用户要求>\n\n"
                    f"<全量范围>\n总文件数: {total_scope_files}\n"
                    "分类统计:\n"
                    + "\n".join(cat_lines)
                    + "\n\n分类代表样本:\n"
                    + "\n".join(sample_lines)
                    + "\n</全量范围>\n\n"
                    "【重要】请务必使用中文回答。即使参考的文件内容包含英文，你的总结和输出也必须完全使用中文。"
                )
            else:
                prompt = (
                    "You are a file analytics assistant. The user is following up on previous count(all) results.\n"
                    "Generate conclusions using full-scope statistics plus category-level representative samples.\n"
                    "- Start with an overall conclusion, then provide 3-6 key findings.\n"
                    "- Cover full-scope trends (not just the shown 50 samples).\n"
                    "- Do not restate every sample and do not claim you read every file in full.\n"
                    "- Clearly state this is inferred from metadata and representative samples.\n\n"
                    f"<User Request>\n{original_question}\n</User Request>\n\n"
                    f"<Full Scope>\nTotal files: {total_scope_files}\n"
                    "Category counts:\n"
                    + "\n".join(cat_lines)
                    + "\n\nRepresentative samples by category:\n"
                    + "\n".join(sample_lines)
                    + "\n</Full Scope>\n\n"
                    "IMPORTANT: You MUST respond in English. Even if the referenced file content is in Chinese, your summary and response must be entirely in English."
                )

            llm = get_llm(streaming=True, session_id=session_id)
            llm.force_text_model = True  # type: ignore
            if user_lang == "zh":
                yield {
                    "type": "text",
                    "delta": f"说明：当前文件总量为 {total_scope_files} 份，无法逐份详细展开；下面基于全量统计与更多分类样本进行归纳总结。\n\n",
                }
            else:
                yield {
                    "type": "text",
                    "delta": f"Note: there are {total_scope_files} files in scope, so it is not feasible to provide one-by-one detailed answers. The summary below is based on full-scope statistics plus broader category samples.\n\n",
                }
            for ch in llm.stream([HumanMessage(content=prompt)]):
                if self.is_aborted(session_id):
                    yield {"type": "text", "delta": "\n\n(已中断)"}
                    break
                delta = getattr(ch, "content", "") or ""
                if delta:
                    yield {"type": "text", "delta": delta}
            yield _done_event("process", sources_payload=list(results or []))
            return

        _path_to_summary: dict = {}
        for _d in (results or []):
            _fp = _d.get("file_path") or ""
            _ds = _d.get("doc_summary") or _d.get("text") or ""
            if _fp and _ds:
                _path_to_summary[_fp] = _ds

        _media_context_kb = None

        def _maybe_hydrate_media_context(docs: List[Dict[str, Any]]) -> None:
            nonlocal _media_context_kb
            if not docs or not any(_is_media_followup_doc(doc) for doc in docs):
                return
            try:
                if _media_context_kb is None:
                    from core.kb import get_kb_instance
                    _media_context_kb = get_kb_instance()
                hydrated = _hydrate_media_results_for_followup(
                    docs,
                    kb=_media_context_kb,
                    max_files=max(1, int(os.getenv("PROCESS_PREV_MEDIA_CONTEXT_MAX_FILES", "4"))),
                    max_chars=max(1200, int(os.getenv("PROCESS_PREV_MEDIA_CONTEXT_MAX_CHARS", "6000"))),
                )
                if hydrated:
                    logger.info(
                        "[process_previous] hydrated media follow-up context for %s file(s)",
                        hydrated,
                    )
            except Exception as exc:
                logger.warning("[process_previous] media context hydration failed: %s", exc)

        if not results:
            if _pp_matched_cat:
                msg = (
                    f"正在为您检索 {_pp_matched_cat} 类别的文件…"
                    if user_lang == "zh"
                    else f"Searching for {_pp_matched_cat} files..."
                )
                yield {"type": "text", "content": msg}
                yield _done_event("process", sources_payload=list(results or []))
                return
            if not results:
                msg = (
                    "抱歉，没有找到上一轮的查询结果，无法进行操作。请先执行一次搜索或统计。"
                    if user_lang == "zh"
                    else "Sorry, I couldn't find previous results to operate on. Please run a search or count first."
                )
                yield {"type": "text", "content": msg}
                yield _done_event("process", sources_payload=[])
                return

        if results and subagent_action == "answer_rewrite":
            focused_results = list(results or [])
            focus_category = _extract_followup_focus_category(self, original_question)
            if focus_category:
                filtered_results = [
                    d for d in focused_results
                    if self._normalize_category_name(str(d.get("doc_category") or "")) == focus_category
                ]
                if filtered_results:
                    focused_results = filtered_results

            _maybe_hydrate_media_context(focused_results)

            file_payload = _dedupe_docs_by_file_identity(focused_results)
            yield {
                "type": "files",
                "preview": file_payload[:50],
                "all": file_payload,
                "total_matches": len(file_payload),
                "shown_count": len(file_payload),
            }

            prompt = (
                "You are revising a previous grounded answer about the currently scoped files.\n"
                "The user is not asking for a brand-new global search.\n\n"
                f"<User follow-up>\n{original_question}\n</User follow-up>\n\n"
                f"<Previous answer>\n{prev_answer_text[:4000] if prev_answer_text else '(not available)'}\n</Previous answer>\n\n"
                "<Scoped files>\n"
            )
            for idx, doc in enumerate(focused_results[:30], 1):
                fname = doc.get("file_name", "") or os.path.basename(doc.get("file_path", ""))
                summary = _select_followup_doc_summary(doc, _path_to_summary)
                prompt += f"[{idx}] {fname}\n"
                if summary:
                    prompt += f"Summary: {summary[:800]}\n"
                content_excerpt = _select_followup_doc_text(doc)
                if content_excerpt:
                    prompt += f"Context: {content_excerpt[:1600]}\n"
                prompt += "\n"
            prompt += (
                "</Scoped files>\n\n"
                "Instructions:\n"
                "- Revise the previous answer according to the user's follow-up.\n"
                "- If the follow-up is a specific question, answer it directly from the scoped file evidence instead of starting a new search.\n"
                "- If the evidence is not enough to identify something precisely, say that clearly and give only the safest visible/audio description.\n"
                "- Directly produce the revised answer. Do not ask the user for more context, do not explain your routing, and do not say you need clarification unless the scoped files contain no usable evidence at all.\n"
                "- Stay grounded in the scoped files only.\n"
                "- Do not restart the task from scratch and do not give a generic preamble like 'Based on your request'.\n"
            )
            if followup_rewrite_mode == "shorten":
                prompt += "- Return a shorter version of the previous answer, preserving the key conclusion only.\n"
            elif followup_rewrite_mode == "detail":
                prompt += "- Expand the previous answer using only scoped evidence.\n"
            elif followup_rewrite_mode == "focus":
                prompt += "- Focus only on the subset implied by the user's follow-up, and ignore files outside that subset.\n"
            elif followup_rewrite_mode == "evidence":
                prompt += "- Start with the strongest supporting files by name, then explain the evidence they provide.\n"
            else:
                prompt += "- If the follow-up asks to shorten, return a shorter version of the previous answer.\n"
                prompt += "- If the follow-up asks to make it more detailed, expand the previous answer using only scoped evidence.\n"
                prompt += "- If the follow-up asks to focus on a subset such as text files or images, keep only that subset in the answer.\n"
                prompt += "- If the follow-up asks for strongest sources or evidence, explicitly name the supporting files first, then summarize the evidence.\n"
            if user_lang == "zh":
                prompt += "\n【重要】请用中文回答。"
            else:
                prompt += "\nIMPORTANT: Respond in English."

            llm = get_llm(streaming=True, session_id=session_id)
            llm.force_text_model = True  # type: ignore
            for ch in llm.stream([HumanMessage(content=prompt)]):
                if self.is_aborted(session_id):
                    yield {"type": "text", "delta": "\n\n(已中断)"}
                    break
                delta = getattr(ch, "content", "") or ""
                if delta:
                    yield {"type": "text", "delta": delta}
            done_sources = (
                list(source_scope_results)
                if structured_plan and structured_plan.scope in {"selected_items", "selected_folder"} and source_scope_results
                else list(focused_results or [])
            )
            yield _done_event("process", sources_payload=done_sources)
            return

        if results and subagent_action == "global_summary":
            scoped_results = list(results or [])
            _maybe_hydrate_media_context(scoped_results)

            file_payload = _dedupe_docs_by_file_identity(scoped_results)
            yield {
                "type": "files",
                "preview": file_payload[:50],
                "all": file_payload,
                "total_matches": len(file_payload),
                "shown_count": len(file_payload),
            }

            prompt = _build_scoped_collective_summary_prompt(
                self,
                docs=scoped_results,
                original_question=original_question,
                user_lang=user_lang,
                path_to_summary=_path_to_summary,
            )

            llm = get_llm(streaming=True, session_id=session_id)
            llm.force_text_model = True  # type: ignore
            for ch in llm.stream([HumanMessage(content=prompt)]):
                if self.is_aborted(session_id):
                    yield {"type": "text", "delta": "\n\n(已中断)"}
                    break
                delta = getattr(ch, "content", "") or ""
                if delta:
                    yield {"type": "text", "delta": delta}
            done_sources = (
                list(source_scope_results)
                if structured_plan and structured_plan.scope in {"selected_items", "selected_folder"} and source_scope_results
                else list(scoped_results or [])
            )
            yield _done_event("process", sources_payload=done_sources)
            return

        def _build_clickable_file_link(file_name: str, file_path: str) -> str:

            from urllib.parse import quote
            name = str(file_name or os.path.basename(file_path) or "file")
            name = name.replace("[", r"\[").replace("]", r"\]")
            path = str(file_path or "").strip()
            if not path:
                return f"`{name}`"
            return f"[{name}](unfoldly://open?path={quote(path, safe='')})"

        # 🔥 Full retrieval within previous results when subagent chose "search".
        # Uses the complete pipeline: vector + BM25 + RRF fusion + keyword/anchor
        # supplement + filename matching, all scoped to the previous round's files.
        if results and subagent_action == "media_content_search" and _ctx_is_media and subagent_search_concept:
            # ── Delegate to media_content_search handler ─────────────────
            allowed_paths = [d.get("file_path") for d in results if d.get("file_path")]
            try:
                for ev in self._handle_media_content_search(
                    subagent_search_concept,
                    params={"query": subagent_search_concept, "media_type": subagent_media_type},
                    session_id=session_id,
                    prompt_language=user_lang,
                    active_paths=allowed_paths or None,
                ):
                    if self.is_aborted(session_id):
                        yield _done_event("interrupted", ok=False, sources_payload=list(results or []))
                        return
                    yield ev
                return
            except Exception as _me:
                logger.warning(f"[process_previous] media_content_search delegation failed: {_me}")
                # Fallback to regular search below
                subagent_action = "search"

        if results and subagent_action == "media_export" and _ctx_is_media and subagent_media_params:
            allowed_paths = [d.get("file_path") for d in results if d.get("file_path")]
            try:
                for ev in self._handle_media_export(
                    original_question,
                    params=subagent_media_params,
                    session_id=session_id,
                    prompt_language=user_lang,
                    active_paths=allowed_paths or None,
                ):
                    if self.is_aborted(session_id):
                        yield _done_event("interrupted", ok=False, sources_payload=list(results or []))
                        return
                    yield ev
                return
            except Exception as _me:
                logger.warning(f"[process_previous] media_export delegation failed: {_me}")
                subagent_action = "search"

        if results and subagent_action == "search" and subagent_query:
            from core.kb.knowledge_base import get_kb_instance
            try:
                kb = get_kb_instance()
                if hasattr(kb, "vector_search"):
                    original_scoped_results = list(results or [])
                    allowed_paths = [doc.get("file_path") for doc in results if doc.get("file_path")]
                    if allowed_paths:
                        yield {"type": "text", "delta": "正在从上述文件中检索相关内容...\n\n" if user_lang == "zh" else "Searching within previous results...\n\n"}
                        refined = kb.vector_search(query=subagent_query, n_results=min(50, len(allowed_paths)), allowed_paths=allowed_paths)
                        if refined:
                            refined = _apply_focused_file_bias(
                                refined,
                                focused_file=focused_file,
                                subagent_query=subagent_query,
                            )
                            refined = _preserve_primary_source_context(
                                refined,
                                original_docs=original_scoped_results,
                                subagent_query=subagent_query,
                            )
                            results = refined
                            for _r in results:
                                _fp = _r.get("file_path") or ""
                                if _fp and not (_r.get("doc_summary") or "").strip():
                                    _r["doc_summary"] = _path_to_summary.get(_fp, "")
                            _maybe_hydrate_media_context(results)
                            file_payload = _dedupe_docs_by_file_identity(results)
                            yield {
                                "type": "files",
                                "preview": file_payload[:50],
                                "all": file_payload,
                                "total_matches": len(file_payload),
                                "shown_count": len(file_payload),
                            }
                        else:
                            fallback_results = _apply_focused_file_bias(
                                list(results or []),
                                focused_file=focused_file,
                                subagent_query=subagent_query,
                            )
                            fallback_results = _preserve_primary_source_context(
                                fallback_results,
                                original_docs=original_scoped_results,
                                subagent_query=subagent_query,
                            )
                            if fallback_results:
                                results = fallback_results[: min(8, len(fallback_results))]
                                if focused_file_path:
                                    try:
                                        focused_content = str(self._read_file_content(focused_file_path) or "").strip()
                                    except Exception:
                                        focused_content = ""
                                    if focused_content:
                                        for _r in results:
                                            if str(_r.get("file_path") or "").strip() == focused_file_path:
                                                existing_text = str(_r.get("text") or "").strip()
                                                if len(existing_text) < 200:
                                                    _r["text"] = focused_content[:5000]
                                                existing_summary = str(_r.get("doc_summary") or "").strip()
                                                if not existing_summary:
                                                    _r["doc_summary"] = focused_content[:800]
                                                break
                                for _r in results:
                                    _fp = _r.get("file_path") or ""
                                    if _fp and not (_r.get("doc_summary") or "").strip():
                                        _r["doc_summary"] = _path_to_summary.get(_fp, "")
                                _maybe_hydrate_media_context(results)
                                note = (
                                    "在已聚焦的文件范围内没有找到精确片段，将基于最相关文件继续回答。\n\n"
                                    if user_lang == "zh"
                                    else "No exact snippet match was found inside the scoped results; answering from the most relevant focused files instead.\n\n"
                                )
                                yield {"type": "text", "delta": note}
                                file_payload = _dedupe_docs_by_file_identity(results)
                                yield {
                                    "type": "files",
                                    "preview": file_payload[:50],
                                    "all": file_payload,
                                    "total_matches": len(file_payload),
                                    "shown_count": len(file_payload),
                                }
                            else:
                                msg = "在上述文件中没有找到相关信息。" if user_lang == "zh" else "No relevant information found in the previous results."
                                yield {"type": "text", "delta": msg}
                                yield _done_event("process", sources_payload=list(results or []))
                                return
            except Exception as e:
                logger.warning(f"process_previous vector search fallback failed: {e}")

        _maybe_hydrate_media_context(results)

        link_map = {}
        prompt = (
            "You are an AI assistant answering a follow-up about the current scoped files.\n"
            "Produce exactly one direct final answer for the user.\n"
            "CRITICAL RULE: If the user asks about a specific file or topic, you MUST ONLY discuss that specific file/topic. IGNORE all other files in the list.\n"
            "If the user asks about 'them', 'these files', '这些文件', or '他们/它们', provide a concise grounded summary of the current file list.\n"
            "Do NOT mention routing, previous-context handling, or phrases like 'Based on your request' / 'I will provide'.\n"
            f"<用户要求>\n{original_question}\n</用户要求>\n\n"
            "<文件列表>\n"
        )
        for idx, doc in enumerate(results[:50], 1):
            fname = doc.get("file_name", "") or os.path.basename(doc.get("file_path", ""))
            fpath = doc.get("file_path", "")
            if fpath:
                link_map[str(idx)] = _build_clickable_file_link(fname, fpath)
            summary = _select_followup_doc_summary(doc, _path_to_summary)
            prompt += f"[{idx}] file: {fname}\n"
            if summary:
                prompt += f"摘要内容: {summary[:500]}\n"
            content_excerpt = _select_followup_doc_text(doc)
            if content_excerpt:
                prompt += f"正文片段: {content_excerpt[:1200]}\n"
            prompt += "\n"
        if user_lang == "zh":
            prompt += "</文件列表>\n\n【核心要求】请务必使用中文回答！\n1. 直接回答，不要写路由说明、任务说明、或“根据你的请求/我将为你”等前言。\n2. 仔细阅读用户要求，**只回答**用户提到的那个文件或主题相关的信息。\n3. 如果用户是在追问“它们/这些文件”，给出基于当前文件范围的简洁归纳，不要机械重复整份文件清单。\n4. 绝对不要把列表里的其他无关文件也总结出来。严禁全文罗列！\n5. 如果需要引用具体的文件，请使用 Markdown 链接格式并附上它的编号（例如 [文件名](13)）。"
        else:
            prompt += "</文件列表>\n\n[CRITICAL INSTRUCTION] You MUST answer in English!\n1. Answer directly. Do not include routing/meta preambles such as 'Based on your request' or 'I will provide'.\n2. Based strictly on the user's request, address ONLY the files or topics they explicitly mentioned.\n3. If the user is asking about 'them' or 'these files', give a concise grounded summary of the current files instead of restating the whole file list.\n4. Do NOT summarize or list unrelated files. Ignore any files that aren't relevant to what the user asked.\n5. If you need to cite a file, use the markdown link format with its index number (e.g. [Filename](13))."


        llm = get_llm(streaming=True, session_id=session_id)
        llm.force_text_model = True # type: ignore
        
        def _raw_stream():
            for ch in llm.stream([HumanMessage(content=prompt)]):
                yield getattr(ch, "content", "") or ""
                
        for chunk in stream_replace_markdown_links(_raw_stream(), link_map):
            if self.is_aborted(session_id):
                yield {"type": "text", "delta": "\n\n(已中断)"}
                break
            if chunk:
                yield {"type": "text", "delta": chunk}
        
        yield _done_event("process", sources_payload=list(results or []))
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        yield {"type": "text", "content": f"处理失败: {str(e)}"}
        yield _done_event("error", ok=False, sources_payload=[])
