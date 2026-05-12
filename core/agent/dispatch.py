"""
Agent dispatch module — extracted from FileAgent for modularity.
"""
from __future__ import annotations
import os, re, time, json, uuid, gc
from typing import Any, Dict, List, Optional, Generator, Iterator, Tuple

from utils.logger import get_logger
logger = get_logger()

from config import settings
from config.prompts import get_prompt, normalize_prompt_language
from core.agent.response_formatter import build_clickable_file_link
from core.domain import QueryExecutionMode
from core.handlers.context import HandlerContext
from core.llm.builder import get_llm
from core.llm.utils import stream_replace_markdown_links
from core.orchestration import ActionExecutor, QueryOrchestrator
from core.retrieval.filename_canonicalizer import (
    classify_explicit_filename_match_mode,
    compact_filename_key,
    filename_stem_key_matches_query,
    has_plausible_filename_extension,
    normalize_filename_candidate,
    score_filename_surface_match,
)
from core.retrieval import (
    apply_media_query_constraints,
    compute_lookup_overlap_score,
    extract_lookup_terms,
    filter_sources_to_scope,
    is_lookup_heavy_query,
    lookup_match_quality,
    narrow_candidates_by_topic_overlap,
    ensure_path_scope_matcher,
    sort_candidates_by_topic_overlap,
)
from tools.document_tools import get_kb_instance


def _build_lexical_query_text(*parts: str) -> str:
    """Merge original/model query surfaces without duplicating the same lookup anchor."""
    merged: List[str] = []
    seen_keys: set[str] = set()
    for raw in parts:
        text = re.sub(r"\s+", " ", str(raw or "").strip())
        if not text:
            continue
        canonical = normalize_filename_candidate(text) or text
        key = compact_filename_key(canonical) or canonical.casefold()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(text)
    return " ".join(merged).strip()


def _collapse_repeated_lookup_phrase(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    parts = text.split(" ")
    if len(parts) <= 1:
        return text
    for width in range(1, (len(parts) // 2) + 1):
        if len(parts) % width != 0:
            continue
        unit = parts[:width]
        if unit and all(parts[idx : idx + width] == unit for idx in range(0, len(parts), width)):
            return " ".join(unit)
    return text


def _active_scope_lacks_category_files(active_paths: Optional[List[str]], category: str, *, max_narrow_scope: int = 20) -> bool:
    """Return True when a narrow file scope plainly cannot contain the requested category."""
    paths = [str(path or "").strip() for path in list(active_paths or []) if str(path or "").strip()]
    if not paths or len(paths) > max_narrow_scope:
        return False
    exts = _CATEGORY_COMPATIBLE_EXTS.get(str(category or "").strip().lower())
    if not exts:
        return False
    for path in paths:
        try:
            if os.path.isdir(os.path.expanduser(path)):
                return False
        except Exception:
            return False
        ext = os.path.splitext(path)[1].lower()
        if not ext:
            return False
        if ext in exts:
            return False
    return True


_IDENTIFIER_FILENAME_ANCHOR_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]{1,8}[-_]?)?\d{6,}[A-Za-z0-9_-]*(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])(?=[A-Za-z0-9_-]{6,}(?![A-Za-z0-9]))"
    r"(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)"
    r"[A-Za-z0-9][A-Za-z0-9_-]{5,}(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_DATA_LIKE_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".numbers", ".json", ".jsonl", ".sql", ".xml"}
_CATEGORY_COMPATIBLE_EXTS = {
    "image": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tiff", ".tif", ".svg"},
    "audio/video": {
        ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape",
        ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv", ".wmv", ".ts",
    },
    "audio": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv", ".wmv", ".ts"},
    "data": set(_DATA_LIKE_EXTS),
    "spreadsheet": set(_DATA_LIKE_EXTS),
    "presentation": {".ppt", ".pptx", ".key", ".odp"},
    "document": {".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".pages"},
    "manual": {".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".pages"},
    "report": {".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".pages"},
    "paper": {".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".pages"},
    "resume": {".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".pages"},
    "invoice": {".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".pages", ".csv", ".xlsx", ".xls", ".numbers"},
    "book": {".pdf", ".epub", ".mobi", ".azw", ".azw3", ".txt", ".md", ".doc", ".docx"},
    "code": {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".swift", ".go", ".rs", ".c", ".h",
        ".cpp", ".hpp", ".cs", ".rb", ".php", ".scala", ".sh", ".zsh", ".bash", ".html",
        ".css", ".scss", ".json", ".yaml", ".yml", ".toml", ".xml", ".sql",
    },
}
_NEGATIVE_CATEGORY_ALIASES = {
    "image": "image",
    "images": "image",
    "photo": "image",
    "photos": "image",
    "picture": "image",
    "pictures": "image",
    "pic": "image",
    "pics": "image",
    "video": "video",
    "videos": "video",
    "movie": "video",
    "movies": "video",
    "clip": "video",
    "clips": "video",
    "audio": "audio",
    "audios": "audio",
    "recording": "audio",
    "recordings": "audio",
    "music": "audio",
    "song": "audio",
    "songs": "audio",
    "document": "document",
    "documents": "document",
    "doc": "document",
    "docs": "document",
    "pdf": "document",
    "pdfs": "document",
    "spreadsheet": "data",
    "spreadsheets": "data",
    "table": "data",
    "tables": "data",
    "data": "data",
    "csv": "data",
    "csvs": "data",
    "excel": "data",
    "resume": "resume",
    "resumes": "resume",
    "invoice": "invoice",
    "invoices": "invoice",
}


def _extract_negative_category_inventory(question: str) -> str:
    """Detect inventory-style requests like "files that are not images"."""
    ql = re.sub(r"\s+", " ", str(question or "").strip().lower())
    if not ql:
        return ""
    if not re.search(r"\b(?:find|search|show|list|display|browse|get|retrieve)\b", ql):
        return ""
    if not re.search(r"\b(?:files?|documents?|docs?|items?)\b", ql):
        return ""
    patterns = [
        r"\bnon[-\s]*(?P<cat>[a-z][a-z0-9_-]*)(?:\s+files?)?\b",
        r"\b(?:not|except|excluding|exclude|without)\s+(?P<cat>[a-z][a-z0-9_-]*)(?:\s+files?)?\b",
    ]
    for pat in patterns:
        m = re.search(pat, ql)
        if not m:
            continue
        raw = str(m.group("cat") or "").strip().lower().strip(".,;:!?")
        if not raw or raw in {"a", "an", "the", "any", "all"}:
            continue
        normalized = _NEGATIVE_CATEGORY_ALIASES.get(raw)
        if normalized:
            return normalized
    return ""
_OPAQUE_DISPLAY_STEM_RE = re.compile(
    r"^(?:[a-f0-9]{12,}|[0-9][0-9a-z._-]{5,}|u=\d+[0-9a-z=&_-]*)$",
    re.IGNORECASE,
)
_DISPLAY_ALIAS_STOPWORDS = {
    "analysis", "paper", "papers", "report", "reports", "document", "documents",
    "file", "files", "study", "model", "models", "method", "methods",
    "prediction", "learning", "network", "networks", "deep", "cross",
}
_DISPLAY_ALIAS_TITLE_RE = re.compile(r"\b([A-Z][A-Za-z0-9+-]{2,15})\s*\(([^()]{6,120})\)")
_DISPLAY_ALIAS_PAREN_RE = re.compile(r"\(([A-Z][A-Za-z0-9+-]{1,15})\)")
_DISPLAY_ALIAS_COLON_RE = re.compile(r"\b([A-Z][A-Za-z0-9+-]{2,15})\s*:")


def _source_text_value(src: Dict[str, Any], *keys: str) -> str:
    meta = src.get("metadata") if isinstance(src.get("metadata"), dict) else {}
    for key in keys:
        val = src.get(key)
        if val is None and isinstance(meta, dict):
            val = meta.get(key)
        text = str(val or "").strip()
        if text:
            return text
    return ""


def _should_suppress_folder_chain_for_refined_file_chain(
    *,
    sources_file_chain: List[Dict[str, Any]],
    sources_folder_chain: List[Dict[str, Any]],
    direct_folder_seed_sources: List[Dict[str, Any]],
    folder_listing_route: bool,
    folder_filter: Optional[str],
    category_inventory_mode: bool,
    query_text: str = "",
    folder_cards: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """File-chain LLM refine is the narrowed result set; folder-chain must not widen it.

    Folder cards alone are not enough to widen a refined file result set; broad
    content searches can produce coincidental folder candidates.
    """
    if not sources_file_chain or not sources_folder_chain:
        return False
    if direct_folder_seed_sources or folder_listing_route or folder_filter:
        return False
    generic_terms = {
        "find", "search", "show", "list", "display", "retrieve", "locate",
        "file", "files", "folder", "folders", "document", "documents", "doc", "docs",
        "report", "reports", "plan", "plans", "paper", "papers",
        "image", "images", "photo", "photos", "video", "videos",
        "audio", "data", "table", "tables", "all", "my", "the",
        "a", "an", "of", "for", "with", "about", "inside", "under",
        "finds", "listing", "listings",
        "找", "搜索", "显示", "文件", "文档", "资料", "报告", "图片", "照片",
        "视频", "音频", "文件夹", "目录", "全部", "所有", "我的", "里面", "下面",
        "简历", "名片", "手册", "方案", "计划", "数据", "表格",
    }
    distinctive_terms = [
        str(term or "").strip().lower()
        for term in extract_lookup_terms(query_text or "", max_terms=32)
        if len(str(term or "").strip()) >= 2 and str(term or "").strip().lower() not in generic_terms
    ]
    if category_inventory_mode and not distinctive_terms:
        return False
    for card in folder_cards or []:
        if not (
            bool(card.get("_folder_literal_hit"))
            or bool(card.get("_inferred_folder_match"))
        ):
            continue
        if not distinctive_terms:
            return False
        folder_blob = " ".join(
            [
                str(card.get("file_name") or ""),
                str(card.get("file_path") or ""),
                str(card.get("doc_summary") or ""),
            ]
        ).lower()
        if any(term in folder_blob for term in distinctive_terms):
            return False
    return any(str(src.get("file_path") or "").strip() for src in sources_file_chain)


def _should_run_folder_chain_recall(
    *,
    direct_folder_seed_sources: List[Dict[str, Any]],
    folder_listing_route: bool,
    folder_listing_query: bool,
    folder_filter: Optional[str],
    category_inventory_mode: bool,
) -> bool:
    """Folder-name recall is part of normal search; expansion is gated later."""
    return True


def _count_display_sources_by_kind(sources: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Count matched folder cards separately from displayed file results."""
    folder_count = 0
    file_paths: set[str] = set()
    anonymous_file_count = 0
    for src in list(sources or []):
        if (
            src.get("is_matched_folder")
            or src.get("_is_folder_hit")
            or str(src.get("type") or "").strip().lower() == "folder"
            or str(src.get("doc_category") or "").strip().lower() == "folder"
        ):
            folder_count += 1
            continue
        file_path = str(src.get("file_path") or "").strip()
        if file_path:
            file_paths.add(file_path)
        else:
            anonymous_file_count += 1
    return folder_count, len(file_paths) + anonymous_file_count


def _format_listing_found_answer(
    sources: List[Dict[str, Any]],
    *,
    user_lang: str,
) -> str:
    folder_count, file_count = _count_display_sources_by_kind(sources)
    if str(user_lang or "").lower().startswith("zh"):
        if folder_count and file_count:
            return f"已找到 {folder_count} 个相关文件夹，{file_count} 个相关文件。"
        if folder_count:
            return f"已找到 {folder_count} 个相关文件夹。"
        return f"已找到 {file_count} 个相关文件。"
    if folder_count and file_count:
        return f"Found {folder_count} relevant folder(s) and {file_count} relevant file(s)."
    if folder_count:
        return f"Found {folder_count} relevant folder(s)."
    return f"Found {file_count} relevant file(s)."


def _looks_opaque_display_name(name: str) -> bool:
    stem = os.path.splitext(os.path.basename(str(name or "").strip()))[0]
    if not stem:
        return False
    if 1 <= len(stem) <= 2 and any("\u4e00" <= ch <= "\u9fff" for ch in stem):
        return True
    if re.fullmatch(r"[0-9.]+(?:v\d+)?", stem, re.IGNORECASE):
        return True
    return bool(_OPAQUE_DISPLAY_STEM_RE.fullmatch(stem))


def _extract_display_alias_terms(*texts: str, max_terms: int = 2) -> List[str]:
    raw_texts = [str(text or "") for text in texts if str(text or "").strip()]
    ranked: List[Tuple[int, str]] = []
    seen: set[str] = set()

    def _register(token: str, *, priority: int) -> None:
        cleaned = str(token or "").strip("()[]{}.,;:!?")
        if not cleaned:
            return
        lowered = cleaned.lower()
        if lowered in _DISPLAY_ALIAS_STOPWORDS:
            return
        if lowered in seen:
            return
        seen.add(lowered)
        ranked.append((priority, cleaned))

    for raw in raw_texts:
        for match in re.finditer(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s*\(([\u4e00-\u9fff]{2,4})\)", raw):
            _register(match.group(2).strip(), priority=130)
            _register(match.group(1).strip(), priority=120)
        for match in _DISPLAY_ALIAS_TITLE_RE.finditer(raw):
            alias = match.group(1).strip()
            descriptor = match.group(2).strip()
            if alias and descriptor:
                if re.search(r"[\u4e00-\u9fff]", descriptor):
                    _register(descriptor, priority=125)
                _register(alias, priority=120)
        for match in _DISPLAY_ALIAS_PAREN_RE.finditer(raw):
            alias = match.group(1).strip()
            has_upper = any(ch.isupper() for ch in alias)
            has_lower = any(ch.islower() for ch in alias)
            if alias.isupper() or (has_upper and has_lower):
                _register(alias, priority=110 if alias.isupper() else 105)
        for match in _DISPLAY_ALIAS_COLON_RE.finditer(raw):
            alias = match.group(1).strip()
            has_upper = any(ch.isupper() for ch in alias)
            has_lower = any(ch.islower() for ch in alias)
            if alias.isupper() or (has_upper and has_lower):
                _register(alias, priority=100 if alias.isupper() else 98)

    for raw in raw_texts:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]{2,}", raw):
            cleaned = token.strip("()[]{}.,;:!?")
            if not cleaned:
                continue
            has_upper = any(ch.isupper() for ch in cleaned)
            has_lower = any(ch.islower() for ch in cleaned)
            is_acronym = cleaned.isupper() and 2 <= len(cleaned) <= 10
            is_mixed_case = has_upper and has_lower and not cleaned.islower()
            if not (is_mixed_case or is_acronym):
                continue
            _register(cleaned, priority=96 if is_mixed_case else 92)

    ranked.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return [token for _, token in ranked[: max(1, max_terms)]]


def _best_display_name_for_source(src: Dict[str, Any]) -> str:
    fp = str(src.get("file_path") or "").strip()
    fn = str(src.get("file_name") or os.path.basename(fp) or "").strip()
    if not fn:
        return fn

    if not _looks_opaque_display_name(fn):
        return fn

    alias_terms = _extract_display_alias_terms(
        _source_text_value(src, "lookup_aliases"),
        _source_text_value(src, "file_name_en"),
        _source_text_value(src, "doc_summary"),
    )
    if alias_terms:
        return f"{fn} - {' / '.join(alias_terms)}"

    file_name_en = _source_text_value(src, "file_name_en")
    if file_name_en:
        file_name_en = os.path.splitext(os.path.basename(file_name_en))[0].strip()
        if file_name_en:
            stem_key = compact_filename_key(os.path.splitext(fn)[0])
            alias_key = compact_filename_key(file_name_en)
            if alias_key and alias_key != stem_key:
                return f"{fn} - {file_name_en}"
    return fn


_PROFILE_POSITIVE_TERMS = {
    "candidate", "profile", "resume", "cv", "recommendation", "professional",
    "experience", "career", "background", "work", "leadership", "portfolio",
    "bio", "biography", "contact", "card",
}
_PROFILE_WEAK_CONTEXT_TERMS = {
    "company", "strategy", "strategic", "analysis", "relationship", "insight",
    "interview", "question", "questions", "notes", "general", "generic",
    "market", "project",
}
_PROFILE_NAME_STOPWORDS = {
    "candidate", "profile", "resume", "recommendation", "report", "document",
    "company", "strategy", "analysis", "relationship", "insight", "global",
    "regional", "leadership", "professional", "experience", "career", "brand",
    "gtm", "general", "topic", "notes", "acme",
}


def _profile_source_blob(src: Dict[str, Any]) -> str:
    return " ".join(
        [
            _source_text_value(src, "file_name"),
            os.path.basename(str(src.get("file_path") or "")),
            _source_text_value(src, "file_name_en"),
            _source_text_value(src, "folder_name_en"),
            _source_text_value(src, "lookup_aliases"),
            _source_text_value(src, "doc_summary"),
            _source_text_value(src, "doc_category"),
        ]
    ).strip()


def _extract_resume_subject_anchor(src: Dict[str, Any]) -> str:
    blob = _profile_source_blob(src)
    if not blob:
        return ""

    alias_match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s*\(([\u4e00-\u9fff]{2,4}|[A-Z][A-Za-z]+)\)", blob)
    if alias_match:
        return alias_match.group(1).strip().lower()

    name_candidates: List[str] = []
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", blob):
        candidate = match.group(1).strip()
        parts = [part.lower() for part in candidate.split()]
        if not parts:
            continue
        if all(part in _PROFILE_NAME_STOPWORDS for part in parts):
            continue
        if any(part in _PROFILE_NAME_STOPWORDS for part in parts) and len(parts) > 1:
            parts = [part for part in parts if part not in _PROFILE_NAME_STOPWORDS]
        if parts:
            name_candidates.append(" ".join(parts))
    if name_candidates:
        return name_candidates[0]

    cjk_match = re.search(r"[\u4e00-\u9fff]{2,4}", blob)
    return cjk_match.group(0) if cjk_match else ""


def _has_explicit_english_person_name(src: Dict[str, Any]) -> bool:
    label = " ".join(
        part
        for part in [
            _source_text_value(src, "file_name"),
            os.path.basename(str(src.get("file_path") or "")),
            _source_text_value(src, "file_name_en"),
        ]
        if str(part or "").strip()
    )
    if not label:
        return False
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", label):
        parts = [part.lower() for part in match.group(1).split()]
        if parts and not any(part in _PROFILE_NAME_STOPWORDS for part in parts):
            return True
    return False


def _resume_candidate_specificity(src: Dict[str, Any]) -> int:
    blob = _profile_source_blob(src).lower()
    if not blob:
        return 0
    score = 0
    category = _source_text_value(src, "doc_category").lower()
    positive_hits = 0
    if category in {"resume", "cv", "contact", "profile"}:
        score += 4
    subject_anchor = _extract_resume_subject_anchor(src)
    if subject_anchor:
        score += 2
    for term in _PROFILE_POSITIVE_TERMS:
        if term in blob:
            score += 1
            positive_hits += 1
    if category in {"report", "document", "note"} and subject_anchor and positive_hits >= 3:
        score += 3
    if _has_explicit_english_person_name(src):
        score += 2
    if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", _profile_source_blob(src)):
        score += 1
    weak_hits = sum(1 for term in _PROFILE_WEAK_CONTEXT_TERMS if term in blob)
    if weak_hits and not any(term in blob for term in ("candidate", "profile", "resume", "cv", "professional", "experience")):
        score -= min(4, weak_hits)
    return score


def _filter_resume_profile_noise(sources: List[Dict[str, Any]], *, query_text: str = "") -> List[Dict[str, Any]]:
    rows = list(sources or [])
    if len(rows) < 4:
        return rows
    scored = [(_resume_candidate_specificity(src), src) for src in rows]
    profile_like = [src for score, src in scored if score >= 5]
    if len(profile_like) < 3:
        return rows
    return profile_like


def _collect_resume_profile_anchor_sources(
    sources: List[Dict[str, Any]],
    query_text: str,
    *,
    limit: int = 4,
) -> List[Dict[str, Any]]:
    rows = list(sources or [])
    if not rows:
        return []
    generic_terms = {
        "find", "search", "show", "list", "candidate", "candidates", "profile",
        "profiles", "resume", "resumes", "cv", "cvs", "document", "documents",
        "file", "files", "about", "for", "the", "a", "an",
    }
    distinctive_terms = [
        str(term or "").strip().lower()
        for term in extract_lookup_terms(query_text or "", max_terms=32)
        if len(str(term or "").strip()) >= 2 and str(term or "").strip().lower() not in generic_terms
    ]
    ranked: List[Tuple[Tuple[int, int, int, float], Dict[str, Any]]] = []
    for src in rows:
        specificity = _resume_candidate_specificity(src)
        if specificity < 5:
            continue
        blob = _profile_source_blob(src).lower()
        term_hits = sum(1 for term in distinctive_terms if term in blob)
        if distinctive_terms and term_hits <= 0:
            continue
        ranked.append(
            (
                (
                    term_hits,
                    specificity,
                    compute_lookup_overlap_score(query_text, blob),
                    float(src.get("rerank_score", 0.0) or 0.0),
                ),
                src,
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [src for _, src in ranked[: max(1, limit)]]


def _merge_category_inventory_candidates(
    primary: List[Dict[str, Any]],
    supplemental: List[Dict[str, Any]],
    *,
    query_text: str,
    max_files: int,
) -> List[Dict[str, Any]]:
    merged_by_path: Dict[str, Dict[str, Any]] = {}
    ordered_paths: List[str] = []

    def _remember(item: Dict[str, Any], *, keyword_hit: bool) -> None:
        fp = str(item.get("file_path") or "").strip()
        fn = str(item.get("file_name") or os.path.basename(fp) or "").strip()
        key = fp or fn
        if not key:
            return
        if key not in merged_by_path:
            merged = dict(item)
            merged["_inventory_keyword_hit"] = bool(keyword_hit)
            merged_by_path[key] = merged
            ordered_paths.append(key)
            return
        if keyword_hit:
            merged_by_path[key]["_inventory_keyword_hit"] = True
        merged_by_path[key]["hit_chunks"] = max(
            int(merged_by_path[key].get("hit_chunks") or 0),
            int(item.get("hit_chunks") or 0),
        )

    for item in list(primary or []):
        _remember(item, keyword_hit=True)
    for item in list(supplemental or []):
        _remember(item, keyword_hit=False)

    merged = [merged_by_path[key] for key in ordered_paths]
    if not merged:
        return []

    query_blob = str(query_text or "").strip()

    def _sort_key(src: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
        if query_blob:
            match_blob = " ".join(
                [
                    str(src.get("file_name") or ""),
                    str(src.get("file_path") or ""),
                    str(src.get("doc_summary") or ""),
                    str(src.get("file_name_en") or ""),
                    str(src.get("folder_name_en") or ""),
                    str(src.get("lookup_aliases") or ""),
                    str(src.get("doc_role") or ""),
                    str(src.get("doc_category_leaf") or ""),
                ]
            )
            overlap = int(compute_lookup_overlap_score(query_blob, match_blob))
        else:
            overlap = 0
        keyword_hit = 1 if src.get("_inventory_keyword_hit") else 0
        hit_chunks = int(src.get("hit_chunks") or 0)
        summary_len = min(len(str(src.get("doc_summary") or "")), 400)
        name_len = min(len(str(src.get("file_name") or "")), 160)
        return (keyword_hit, overlap, hit_chunks, summary_len, name_len)

    if query_blob:
        merged = sort_candidates_by_topic_overlap(merged, query_blob)
    merged = sorted(merged, key=_sort_key, reverse=True)
    return merged[: max(1, int(max_files or 1))]


def _display_stem_dedupe_key(src: Dict[str, Any]) -> str:
    fp = str(src.get("file_path") or "").strip()
    name = str(src.get("file_name") or os.path.basename(fp) or "").strip()
    stem_key = compact_filename_key(os.path.splitext(name)[0])
    if not stem_key or len(stem_key) < 8 or stem_key.isdigit():
        return ""
    return stem_key


def _dedupe_sources_by_display_stem(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_paths: set[str] = set()
    for src in list(sources or []):
        fp = str(src.get("file_path") or "").strip()
        if fp and fp in seen_paths:
            continue
        dedupe_key = _display_stem_dedupe_key(src)
        if dedupe_key and dedupe_key in seen_keys:
            continue
        if fp:
            seen_paths.add(fp)
        if dedupe_key:
            seen_keys.add(dedupe_key)
        deduped.append(src)
    return deduped


def _dedupe_sources_by_file_identity(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for src in list(sources or []):
        if not isinstance(src, dict):
            continue
        fp = str(src.get("file_path") or "").strip()
        name = str(src.get("file_name") or os.path.basename(fp) or "").strip()
        key = fp or name
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(src)
    return deduped


def _collapse_exact_filename_focus_hits(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse chunk-level exact filename hits to distinct file identities."""
    ranked: List[Dict[str, Any]] = [
        src for src in list(sources or []) if isinstance(src, dict)
    ]
    ranked.sort(
        key=lambda src: (
            int(src.get("_direct_score", 0) or 0),
            float(src.get("_bm25_score", 0.0) or 0.0),
            float(src.get("rerank_score", 0.0) or src.get("score", 0.0) or 0.0),
            str(src.get("file_name") or "").lower(),
            str(src.get("file_path") or "").lower(),
        ),
        reverse=True,
    )
    return _dedupe_sources_by_file_identity(ranked)


def _folder_preview_diversity_key(src: Dict[str, Any]) -> str:
    fp = str(src.get("file_path") or "").strip()
    name = str(src.get("file_name") or os.path.basename(fp) or "").strip()
    stem_key = compact_filename_key(os.path.splitext(name)[0])
    if stem_key:
        return stem_key
    return fp.lower()


def _select_folder_chain_preview(
    sources_folder_chain: List[Dict[str, Any]],
    *,
    query_text: str = "",
    max_total: int = 6,
    max_per_root: int = 2,
) -> List[Dict[str, Any]]:
    if not sources_folder_chain:
        return []
    by_root: Dict[str, List[Dict[str, Any]]] = {}
    root_order: List[str] = []
    for src in list(sources_folder_chain or []):
        root = str(src.get("folder_chain_root") or os.path.dirname(str(src.get("file_path") or "")) or "").strip()
        if root not in by_root:
            by_root[root] = []
            root_order.append(root)
        by_root[root].append(src)

    preview_query = str(query_text or "").strip()

    profile_query = bool(re.search(r"\b(candidate|candidates|profile|profiles|resume|resumes|cv|cvs)\b", preview_query, re.IGNORECASE))

    def _preview_sort_key(src: Dict[str, Any]) -> Tuple[int, int, int, float]:
        blob = " ".join(
            [
                str(src.get("file_name") or ""),
                str(src.get("file_path") or ""),
                str(src.get("doc_summary") or ""),
            ]
        )
        overlap = int(compute_lookup_overlap_score(preview_query, blob)) if preview_query else 0
        profile_score = _resume_candidate_specificity(src) if profile_query else 0
        direct_score = int(src.get("_direct_score", 0) or 0)
        rerank_score = float(src.get("rerank_score", 0.0) or 0.0)
        return (profile_score, overlap, direct_score, rerank_score)

    for root in root_order:
        by_root[root].sort(key=_preview_sort_key, reverse=True)
        diversified: List[Dict[str, Any]] = []
        spillover: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        for item in by_root[root]:
            diversity_key = _folder_preview_diversity_key(item)
            if diversity_key and diversity_key in seen_keys:
                spillover.append(item)
                continue
            diversified.append(item)
            if diversity_key:
                seen_keys.add(diversity_key)
        by_root[root] = diversified + spillover

    picked: List[Dict[str, Any]] = []
    per_root_counts: Dict[str, int] = {}
    round_idx = 0
    while len(picked) < max_total:
        progressed = False
        for root in root_order:
            root_items = by_root.get(root) or []
            used = per_root_counts.get(root, 0)
            if used >= max_per_root or used >= len(root_items):
                continue
            picked.append(root_items[used])
            per_root_counts[root] = used + 1
            progressed = True
            if len(picked) >= max_total:
                break
        round_idx += 1
        if not progressed or round_idx > max(1, len(root_order)) * max_total:
            break
    return picked


def _compose_search_sources_for_display(
    *,
    folder_cards: List[Dict[str, Any]],
    sources_folder_chain: List[Dict[str, Any]],
    sources_file_chain: List[Dict[str, Any]],
    direct_folder_seed_sources: List[Dict[str, Any]],
    folder_listing_route: bool,
    folder_filter: Optional[str],
    explicit_filename_mode: bool,
    query_text: str = "",
) -> List[Dict[str, Any]]:
    file_sources = list(sources_file_chain or [])
    if not explicit_filename_mode:
        file_sources = _dedupe_sources_by_display_stem(file_sources)

    generic_terms = {
        "find", "search", "show", "list", "display", "retrieve", "locate",
        "file", "files", "folder", "folders", "document", "documents", "doc", "docs",
        "report", "reports", "plan", "plans", "paper", "papers",
        "image", "images", "photo", "photos", "video", "videos",
        "audio", "data", "table", "tables", "all", "my", "the",
        "a", "an", "of", "for", "with", "about", "inside", "under",
        "找", "搜索", "显示", "文件", "文档", "资料", "报告", "图片", "照片",
        "视频", "音频", "文件夹", "目录", "全部", "所有", "我的", "里面", "下面",
    }
    distinctive_terms = [
        str(term or "").strip().lower()
        for term in extract_lookup_terms(query_text or "", max_terms=32)
        if len(str(term or "").strip()) >= 2 and str(term or "").strip().lower() not in generic_terms
    ]
    folder_direct_target = False
    if distinctive_terms:
        required_hits = len(distinctive_terms) if len(distinctive_terms) <= 2 else max(2, (len(distinctive_terms) * 3 + 3) // 4)
        for card in list(folder_cards or []):
            folder_blob = " ".join(
                [
                    str(card.get("file_name") or ""),
                    str(card.get("file_path") or ""),
                    str(card.get("doc_summary") or ""),
                ]
            ).lower()
            hits = sum(1 for term in distinctive_terms if term and term in folder_blob)
            if bool(card.get("_folder_literal_hit")) and len(distinctive_terms) <= 2 and hits >= 1:
                folder_direct_target = True
                break
            if hits >= required_hits:
                folder_direct_target = True
                break

    keep_full_folder_expansion = bool(
        direct_folder_seed_sources
        or folder_listing_route
        or folder_filter
        or folder_direct_target
        or not file_sources
    )
    if keep_full_folder_expansion:
        return list(folder_cards or []) + list(sources_folder_chain or []) + file_sources

    folder_roots = {
        str(src.get("folder_chain_root") or os.path.dirname(str(src.get("file_path") or "")) or "").strip()
        for src in list(sources_folder_chain or [])
        if str(src.get("folder_chain_root") or os.path.dirname(str(src.get("file_path") or "")) or "").strip()
    }
    preview_max_total = max(4, min(8, len(sources_folder_chain or [])))
    preview_max_per_root = 4 if distinctive_terms or len(file_sources) >= 8 else 2
    if len(folder_roots) <= 1 and (distinctive_terms or len(file_sources) >= 8):
        preview_max_total = max(preview_max_total, min(10, len(sources_folder_chain or [])))
        preview_max_per_root = max(preview_max_per_root, min(6, len(sources_folder_chain or [])))

    folder_preview = _select_folder_chain_preview(
        list(sources_folder_chain or []),
        query_text=query_text,
        max_total=preview_max_total,
        max_per_root=preview_max_per_root,
    )
    return list(folder_cards or []) + file_sources + folder_preview


def _extract_identifier_filename_anchors(text: str, *, max_anchors: int = 4) -> List[str]:
    """Extract long id-like anchors that often live in filenames or paths."""
    anchors: List[str] = []
    seen: set[str] = set()
    for match in _IDENTIFIER_FILENAME_ANCHOR_RE.finditer(str(text or "")):
        raw = re.sub(r"\s+", "", str(match.group(0) or "").strip(" \"'“”‘’.,;:!?()[]{}"))
        if not raw or not any(ch.isdigit() for ch in raw):
            continue
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        anchors.append(raw)
        if len(anchors) >= max_anchors:
            break
    return anchors


def _query_contains_filename_needle(raw_query: str, candidate: str) -> bool:
    """Return True only when the query convincingly references the candidate filename.

    Used to gate the "exact filename hit" branch so an entity query is not
    treated as an exact match for an unrelated one-character CJK filename whose
    stem happens to be a substring of the query.
    """

    rq = str(raw_query or "").strip().lower()
    cand = str(candidate or "").strip().lower()
    if not rq or not cand:
        return False
    if cand in rq:
        return True
    cand_base = os.path.basename(cand)
    cand_stem = os.path.splitext(cand_base)[0]
    if cand_stem and cand_stem in rq:
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in cand_stem)
        # Short Latin stems such as "img" or "dog" are usually topical/generic
        # tokens.  Do not let them promote broad queries like "CAM_1234" or
        # "find dog" into exact filename matches for img.jpeg/dog.jpg.
        if has_cjk and len(cand_stem) >= 2:
            return True
        if len(cand_stem) >= 4:
            return True
    q_compact = re.sub(r"[\s\\/_\-.]+", "", rq)
    for token in {cand_base, cand_stem}:
        token_compact = re.sub(r"[\s\\/_\-.]+", "", token or "")
        if token_compact and len(token_compact) >= 4 and token_compact in q_compact:
            return True
    query_terms = set(extract_lookup_terms(rq, max_terms=64))
    cand_parts = [
        part.strip().lower()
        for part in re.split(r"[\s\\/_\-.]+", cand_stem)
        if part and part.strip()
    ]
    strong_parts = [
        part
        for part in cand_parts
        if (
            (
                len(part) >= 4
                or any("\u4e00" <= ch <= "\u9fff" for ch in part)
                or any(ch.isdigit() for ch in part)
            )
            and part not in {
            "file", "files", "image", "images", "photo", "photos",
            "video", "videos", "audio", "music", "doc", "docs",
            "report", "reports", "paper", "papers", "data",
            }
        )
    ]
    if strong_parts:
        q_space = f" {rq} "
        if all((part in query_terms) or (f" {part} " in q_space) or (part in q_compact) for part in strong_parts):
            return True
    return False


def _query_is_unambiguous_filename_stem_reference(raw_query: str, candidate: str) -> bool:
    """Return True when a no-extension query is essentially the filename stem.

    This is intentionally narrower than generic lexical overlap.  It lets
    "tell me project notes" focus `project notes.pdf`, while keeping broad
    topical searches like "find dog" or "find project notes tutorials" in normal
    semantic search.
    """
    if not _query_contains_filename_needle(raw_query, candidate):
        return False

    query = str(raw_query or "").strip().lower()
    cand_base = os.path.basename(str(candidate or "").strip().lower())
    cand_stem = os.path.splitext(cand_base)[0].strip()
    if not query or not cand_stem:
        return False

    stop_terms = {
        "a", "an", "and", "about", "all", "by", "for", "from", "in", "inside",
        "into", "me", "my", "named", "of", "on", "open", "please", "show",
        "summarize", "summary", "tell", "the", "this", "that", "to", "with",
        "file", "files", "document", "documents", "doc", "docs", "pdf", "txt",
        "word", "excel", "image", "images", "photo", "photos", "video", "videos",
        "audio", "report", "reports", "paper", "papers",
        "找", "搜索", "显示", "打开", "总结", "看看", "关于", "这个", "那个", "文件",
        "文档", "资料", "报告", "图片", "照片", "视频", "音频", "所有", "全部", "我的",
    }
    broad_search_verbs = {"find", "search", "list", "locate", "retrieve"}

    def _parts(text: str) -> List[str]:
        return [
            part.lower()
            for part in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text.lower())
            if part
        ]

    stem_parts = _parts(cand_stem)
    if not stem_parts:
        return False

    stem_terms = set(stem_parts)
    distinctive_stem_terms = [
        part
        for part in stem_parts
        if part not in stop_terms
        and (
            len(part) >= 4
            or any(ch.isdigit() for ch in part)
            or any("\u4e00" <= ch <= "\u9fff" for ch in part)
        )
    ]
    # Multi-word stems such as "project notes" are distinctive as a phrase even if
    # one component is short. A one-token short English stem stays topical.
    if len(stem_parts) >= 2:
        distinctive_stem_terms = distinctive_stem_terms or stem_parts
    if not distinctive_stem_terms:
        return False

    query_parts = _parts(query)
    query_core = [part for part in query_parts if part not in stop_terms and part not in broad_search_verbs]
    if not query_core:
        return False

    # Broad search verbs are intentionally conservative: "find project notes" can
    # mean a topic, while "tell me project notes" is much more likely to mean the
    # named file/result.
    if any(part in broad_search_verbs for part in query_parts):
        return False

    stem_compact = re.sub(r"[\s\\/_\-.]+", "", cand_stem)
    query_core_compact = "".join(query_core)
    if query_core_compact and stem_compact and query_core_compact == stem_compact:
        return True

    return all(part in stem_terms for part in query_core)


def _post_answer_filename_focus_sources(
    sources: List[Dict[str, Any]],
    *,
    query_text: str,
    answer_text: str,
    max_input_sources: int = 12,
) -> List[Dict[str, Any]]:
    """Narrow displayed files after answering when one exact filename is obvious.

    The search/answer path should keep enough candidates for recall quality.
    After the model has answered, however, the UI can be updated to the single
    precise filename hit when the evidence is unambiguous.  This keeps broad
    searches broad while fixing cases where the answer clearly focuses on one
    file but the relevant-files panel still shows neighboring candidates.
    """
    if not sources or len(sources) <= 1 or len(sources) > max_input_sources:
        return []

    query = str(query_text or "").strip()
    answer_key = compact_filename_key(answer_text or "")
    ranked: List[Dict[str, Any]] = []
    for src in list(sources or []):
        if not isinstance(src, dict) or src.get("is_matched_folder") or src.get("is_folder_chain_match"):
            continue
        file_path = str(src.get("file_path") or "").strip()
        file_name = str(src.get("file_name") or os.path.basename(file_path) or "").strip()
        if not file_name:
            continue
        direct_score = int(src.get("_direct_score", 0) or 0)
        filename_hit = bool(
            src.get("_lexical_filename_exact")
            or src.get("_lookup_match_exact")
            or (
                src.get("_is_lexical_hit")
                and direct_score >= 90
                and _query_contains_filename_needle(query, file_name)
            )
        )
        if not filename_hit:
            continue

        unambiguous_query = _query_is_unambiguous_filename_stem_reference(query, file_name)
        if not unambiguous_query and not has_plausible_filename_extension(query):
            continue

        name_key = compact_filename_key(file_name)
        stem_key = compact_filename_key(os.path.splitext(file_name)[0])
        answer_mentions_file = bool(
            (name_key and name_key in answer_key)
            or (stem_key and len(stem_key) >= 6 and stem_key in answer_key)
        )
        if not answer_mentions_file and not unambiguous_query:
            continue
        ranked.append(src)

    focused = _collapse_exact_filename_focus_hits(ranked)
    if len(focused) != 1:
        return []
    return focused


def _query_stream_intent_dispatch(
    self,
    question: str,
    *,
    active_paths: Optional[List[str]],
    session_id: Optional[str],
    emit_status: bool,
    prompt_language: Optional[str] = None,
    opened_file_path: Optional[str] = None,
):
    q = (question or "").strip()
    if not q:
        yield {"type": "done", "ok": True, "query_type": "chat", "sources": [], "trace": []}
        return
    _request_t0 = time.time()
    _first_text_emitted = False
    _first_files_emitted = False
    _search_ack_emitted = False

    def _request_elapsed_ms() -> int:
        return int((time.time() - _request_t0) * 1000)

    def _emit_timing_trace(stage: str, **payload: Any) -> Dict[str, Any]:
        item = {
            "stage": stage,
            "type": "timing",
            "elapsed_ms": _request_elapsed_ms(),
            **payload,
        }
        try:
            logger.info(
                "[timing] session=%s stage=%s payload=%s",
                session_id,
                stage,
                json.dumps(item, ensure_ascii=False, default=str),
            )
        except Exception:
            logger.info("[timing] session=%s stage=%s", session_id, stage)
        return {"type": "trace_append", "item": item}

    def _mark_first_text(source: str, *, chars: int = 0) -> Optional[Dict[str, Any]]:
        nonlocal _first_text_emitted
        if _first_text_emitted:
            return None
        _first_text_emitted = True
        return _emit_timing_trace("first_text", source=source, chars=int(chars or 0))

    def _mark_first_files(source: str, *, total_matches: int = 0, shown_count: int = 0) -> Optional[Dict[str, Any]]:
        nonlocal _first_files_emitted
        if _first_files_emitted:
            return None
        _first_files_emitted = True
        return _emit_timing_trace(
            "first_files",
            source=source,
            total_matches=int(total_matches or 0),
            shown_count=int(shown_count or 0),
        )
    user_lang = self._resolve_prompt_language(None, question=q, session_id=session_id)
    internal_lang = "en"
    response_language_label = "Simplified Chinese" if user_lang == "zh" else "English"
    self._remember_prompt_language(session_id, user_lang)

    if opened_file_path:
        q_test = q.lower()
        if re.search(r"(这个|该|本|当前|选中|this)\s*(文件|文档|图片|照片|pdf|docx|txt|file|image|document|pic)", q_test) or "总结" in q_test or "summarize" in q_test:
            file_name = os.path.basename(opened_file_path)
            q = f"{q} (System Context: The user is referring to the currently opened file: '{file_name}', path: '{opened_file_path}')"

    hist_ref = self._get_history_ref(session_id)

    sid = (session_id or "").strip()
    if sid:
        scope_changed = self._sync_session_active_paths(session_id, active_paths)
        if scope_changed:
            import logging as _log
            _log.getLogger(__name__).info(
                f"[session={sid}] Sources changed"
                " -> auto-clearing history & last_results"
            )
            self._clear_session_runtime_state(session_id, clear_history=True, reason="active_paths_changed")
            hist_ref = self._get_history_ref(session_id)

    hist_ref.append({"q": q, "a": ""})
    if len(hist_ref) > 10:
        del hist_ref[: max(0, len(hist_ref) - 10)]

    q_norm2 = q.replace("？", "?").strip().lower()
    short_all = q_norm2 in {"所有的", "全部", "都要", "都行", "都可以", "全都要", "全部都要", "all"}
    if short_all:
        recent_text = ""
        try:
            for h in hist_ref[-2:]:
                recent_text += f"{h.get('q','')}\n{h.get('a','')}\n"
        except Exception:
            recent_text = ""
        if any(k in recent_text for k in ["音频", "音乐风格", "音乐", "听", "分析音频", "分析音乐", "风格"]):
            yield {"type": "thinking", "delta": "Detected context confirmation request; generating response...\n"}
            if user_lang == "zh":
                msg = (
                    "我明白你想“把全部音频都分析一遍”。不过目前我没法直接“听音频内容”来判断曲风，\n"
                    "我能做的是基于已索引的信息（文件名/路径/标签/描述/摘要）来归纳你的音乐类型分布，或帮你把音频文件按风格线索快速筛出来。\n\n"
                    "你想要哪一种结果？\n"
                    "1) 先列出所有音乐/音频文件（按文件名/路径）\n"
                    "2) 基于文件名/标签做“风格关键词”归类（如 Lo-fi / 摇滚 / 电子 / 古典 等）\n"
                    "3) 你选 3-5 个代表音频文件（或给我文件名关键词），我再更细地总结“你的音乐风格倾向”"
                )
            else:
                msg = (
                    "Got it, you want to analyze all audio files. I cannot directly listen to audio content yet,\n"
                    "but I can infer style patterns from indexed info (file names/paths/tags/descriptions/summaries)\n"
                    "and help you quickly group audio files by style cues.\n\n"
                    "Which result do you want?\n"
                    "1) List all music/audio files first (by file name/path)\n"
                    "2) Group by style keywords from names/tags (e.g., Lo-fi / Rock / Electronic / Classical)\n"
                    "3) Pick 3-5 representative files (or give keywords), and I will summarize your style preference in more detail"
                )
            yield {"type": "text", "delta": msg}
            try:
                hist_ref[-1]["a"] = msg
            except Exception:
                pass
            yield {"type": "done", "ok": True, "query_type": "clarify", "sources": [], "trace": []}
            return

    _confirm_clear_kws = ["确认清空", "确认删除数据库", "确认清除索引", "确认重置", "confirm clear", "confirm reset"]
    if any(ck in q for ck in _confirm_clear_kws):
        recent_a = ""
        try:
            for h in hist_ref[-3:]:
                recent_a += h.get("a", "")
        except Exception:
            recent_a = ""
        if "清空数据库" in recent_a or "索引数据" in recent_a or "操作不可逆" in recent_a:
            yield {"type": "thinking", "delta": "Executing database clear operation...\n"}
            clearing_msg = "正在清空数据库…\n" if user_lang == "zh" else "Clearing indexed database...\n"
            yield {"type": "text", "delta": clearing_msg}
            result = self.kb.clear_all()
            if result.get("ok"):
                if user_lang == "zh":
                    msg = f"✅ 已清空所有索引数据，共删除 {result.get('deleted_count', 0)} 条记录。\n原始文件不受影响。如需重新索引，请添加文件夹并重新建立索引。"
                else:
                    msg = f"✅ All indexed data has been cleared, removed {result.get('deleted_count', 0)} records.\nOriginal files are not affected. To re-index, add folders and build the index again."
            else:
                msg = (
                    f"❌ 清空失败：{result.get('error', '未知错误')}"
                    if user_lang == "zh"
                    else f"❌ Clear failed: {result.get('error', 'unknown error')}"
                )
            yield {"type": "text", "delta": msg}
            try:
                hist_ref[-1]["a"] = msg
            except Exception:
                pass
            yield {"type": "done", "ok": True, "query_type": "db_clear", "sources": [], "trace": []}
            return

    q_norm = q.replace("？", "?").strip().lower()
    from core.intent.preprocessor import QueryPreprocessor
    if QueryPreprocessor.is_capability_query(q):
        yield {"type": "thinking", "delta": "Detected feature-introduction query; generating response...\n"}
        llm = self._get_llm_service(detailed=False, session_id=session_id, prompt_language=user_lang)
        try:
            recent = ""
            for h in hist_ref[-3:-1]:
                if h.get("q"):
                    recent += f"用户：{h.get('q')}\n"
                if h.get("a"):
                    recent += f"助手：{h.get('a')}\n"
        except Exception:
            recent = ""

        hint = (
            f"IMPORTANT: Your entire response MUST be written in {response_language_label}. Do NOT mix languages.\n"
            "- Keep the information moderate: 6-10 points are enough; 1 line per point.\n"
            "- Only describe what you can do in this app: indexing, search Q&A, finding files, statistical overview, opening files, lightweight file operations.\n"
            "- If there are no data sources/indexes currently, remind the user to add folders and build indexes first.\n"
            "- Use a natural tone, do not sound like a product manual.\n"
            "- At the end, provide 2-3 sample questions. These MUST be realistic questions the user can actually try right now. Use these exact examples:\n"
            f"  {'Chinese' if user_lang == 'zh' else 'English'} examples to use verbatim:\n"
            + (
                "  1. \"查看我有哪些文件\"\n"
                "  2. \"帮我找一下关于XX的文件\" (XX = a concrete topic like 项目报告, 会议纪要, etc.)\n"
                "  3. \"这些文件里有没有提到XX的内容\" (XX = a concrete keyword)\n"
                if user_lang == "zh" else
                "  1. \"Show me all my files\"\n"
                "  2. \"Find files related to XX\" (XX = a concrete topic like project reports, meeting notes, etc.)\n"
                "  3. \"Do any of these files mention XX?\" (XX = a concrete keyword)\n"
            )
            + "- Do NOT use vague placeholders like 'some project', 'some file type', 'some company'. Use specific, realistic examples instead.\n"
        )
        recent_block_text = recent if recent else ""
        prompt = self._prompt("CAPABILITY_QUERY_PROMPT", internal_lang).format(
            instruction=hint,
            recent_block=recent_block_text,
            question=q
        )

        resp_text = ""
        for chunk in llm.generate_stream(prompt):
            if self.is_aborted(session_id):
                logger.info(f"检测到中断标志，停止生成 (session={session_id})")
                yield {"type": "done", "ok": False, "query_type": "interrupted", "message": "生成已中断"}
                return
            if not chunk:
                continue
            yield {"type": "text", "delta": chunk}
            resp_text += chunk
        if len(resp_text.strip()) < 40:
            extra = (
                "\n你可以先告诉我：你想“找文件”、还是“问内容”、还是“整理文件”？"
                if user_lang == "zh"
                else "\nYou can tell me first: do you want to find files, ask about content, or organize files?"
            )
            yield {"type": "text", "delta": extra}
            resp_text += extra
        try:
            hist_ref[-1]["a"] = resp_text
        except Exception:
            pass
        yield {"type": "done", "ok": True, "query_type": "chat", "sources": [], "trace": []}
        return

    def _emit_status(phase: str, message: str):
        if emit_status:
            yield {"type": "status", "phase": phase, "message": message}

    def _to_user_text(text: str) -> str:
        return str(text or "")

    def _collect_or_emit_stream(llm_service, prompt_text: str, link_map: Optional[dict] = None) -> Optional[str]:

        def _iter_display_chunks(text: str):
            raw = str(text or "")
            if not raw:
                return
            if len(raw) <= 120 and "\n\n" not in raw:
                yield raw
                return

            preferred = 120
            hard_limit = 160
            split_chars = " \n\t，。！？；：,.!?;:)]}\"'」』）】"
            start = 0
            total = len(raw)
            while start < total:
                remain = total - start
                if remain <= hard_limit:
                    yield raw[start:]
                    break

                window_end = min(total, start + hard_limit)
                preferred_end = min(total, start + preferred)
                candidate = raw[start:window_end]

                split_at = -1
                for i in range(min(len(candidate), preferred_end - start), 0, -1):
                    if candidate[i - 1] in split_chars:
                        split_at = i
                        break
                if split_at <= 0:
                    for i in range(min(len(candidate), hard_limit), min(len(candidate), preferred_end - start), -1):
                        if candidate[i - 1] in split_chars:
                            split_at = i
                            break
                if split_at <= 0:
                    split_at = min(len(candidate), preferred)

                piece = candidate[:split_at]
                if not piece:
                    piece = candidate[: min(len(candidate), preferred)]
                yield piece
                start += len(piece)

        def _looks_tail_incomplete(text: str) -> bool:
            t = str(text or "").rstrip()
            if len(t) < 20:
                return False
            if t.endswith(("。", "！", "？", ".", "!", "?", "”", "\"", "’", "」", "』", "）", ")", "】", "]", "`")):
                return False
            tail = t[-24:]
            if any(k in tail for k in ["例如", "比如", "包括", "其中", "以及", "和", "或", "等", "如", "四舍", "备注", "：", ":"]):
                return True
            if re.search(r"[A-Za-z]{1,3}$", tail):
                return True
            return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]$", t))

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

        out_resp = ""
        def _raw_gen():
            for chunk in llm_service.generate_stream(prompt_text):
                if chunk: yield chunk
                
        stream_gen = stream_replace_markdown_links(_raw_gen(), link_map) if link_map else _raw_gen()

        for chunk in stream_gen:
            if self.is_aborted(session_id):
                logger.info(f"[Agent] 检测到中断标志，停止生成 (session={session_id})")
                yield {"type": "done", "ok": False, "query_type": "interrupted", "message": "生成已中断"}
                return None
            if not chunk:
                continue
            for piece in _iter_display_chunks(chunk):
                if not piece:
                    continue
                _first_text_trace = _mark_first_text("llm_stream", chars=len(piece))
                if _first_text_trace:
                    yield _first_text_trace
                yield {"type": "text", "delta": piece}
                out_resp += piece

        last_finish_reason = None
        try:
            getter = getattr(llm_service, "get_last_finish_reason", None)
            if callable(getter):
                last_finish_reason = getter()
        except Exception:
            last_finish_reason = None
            
        _finish_reason_str = str(last_finish_reason or "").strip().lower()
        was_len_truncated = _finish_reason_str in {"length", "max_tokens"}
        # finish_reason="stop" means llama.cpp emitted the EOS token — generation is complete.
        # Only fall back to heuristic _looks_tail_incomplete when finish_reason is unknown
        # (e.g. older llama.cpp build, network error, None). Never trigger continuation when
        # the model explicitly signalled it finished normally.
        _finish_reason_known = bool(_finish_reason_str) and _finish_reason_str not in {"unknown", "none"}
        _heuristic_incomplete = (not _finish_reason_known) and _looks_tail_incomplete(out_resp)

        auto_continue = str(os.getenv("FILEAGENT_AUTO_CONTINUE_ON_TRUNCATION", "true")).strip().lower() in {"1", "true", "yes", "on"}
        if auto_continue and (was_len_truncated or _heuristic_incomplete):
            max_continue_rounds = max(1, min(4, int(os.getenv("FILEAGENT_CONTINUE_ROUNDS", "2") or 2)))
            max_continue_chars = max(120, int(os.getenv("FILEAGENT_CONTINUE_MAX_CHARS", "1200") or 1200))
            appended = 0
            need_continue = True
            for _ in range(max_continue_rounds):
                if not need_continue or appended >= max_continue_chars:
                    break
                tail = out_resp[-1200:]
                if user_lang == "zh":
                    cont_prompt = (
                        "你上一段回答在中途结束了。请从中断处继续补全，保证语义完整。\n"
                        "- 不要重复已写内容；只续写缺失部分。\n"
                        "- 严禁输出任何开场白或寒暄语（绝对不要输出“好的”、“没问题”、“这就为您补全”等），必须直接给出连贯的续写文本。\n"
                        "- 继续使用当前语言。\n\n"
                        "<已输出内容末尾>\n"
                        f"{tail}\n"
                        "</已输出内容末尾>"
                    )
                else:
                    cont_prompt = (
                        "Your previous answer appears to end mid-sentence. Continue from where it stopped.\n"
                        "- Do not repeat previous content; only provide the missing continuation.\n"
                        "- ABSOLUTELY NO conversational filler or greetings (e.g. do not say 'Sure', 'Understood', 'Here is the continuation'). Start outputting the missing text immediately.\n"
                        "- Keep the same language.\n\n"
                        "<Tail of emitted answer>\n"
                        f"{tail}\n"
                        "</Tail of emitted answer>"
                    )

                cont = ""
                def _cont_raw_gen():
                    for chunk in llm_service.generate_stream(cont_prompt):
                        if chunk: yield chunk
                        
                cont_stream_gen = stream_replace_markdown_links(_cont_raw_gen(), link_map) if link_map else _cont_raw_gen()

                for chunk in cont_stream_gen:
                    if self.is_aborted(session_id):
                        logger.info(f"[Agent] 检测到中断标志，停止续写 (session={session_id})")
                        yield {"type": "done", "ok": False, "query_type": "interrupted", "message": "生成已中断"}
                        return None
                    if not chunk:
                        continue
                    if len(cont) >= max_continue_chars:
                        break
                    remain = max_continue_chars - appended - len(cont)
                    if remain <= 0:
                        break
                    emitted_this_chunk = 0
                    for piece in _iter_display_chunks(chunk):
                        if not piece:
                            continue
                        piece_remain = remain - emitted_this_chunk
                        if piece_remain <= 0:
                            break
                        safe_piece = piece[:piece_remain]
                        if not safe_piece:
                            continue
                        yield {"type": "text", "delta": safe_piece}
                        cont += safe_piece
                        emitted_this_chunk += len(safe_piece)
                        if emitted_this_chunk >= remain:
                            break

                if not cont:
                    break
                out_resp = _merge_with_overlap(out_resp, cont)
                appended += len(cont)

                cont_finish_reason = None
                try:
                    getter = getattr(llm_service, "get_last_finish_reason", None)
                    if callable(getter):
                        cont_finish_reason = getter()
                except Exception:
                    cont_finish_reason = None
                need_continue = (
                    str(cont_finish_reason or "").strip().lower() in {"length", "max_tokens"}
                    or _looks_tail_incomplete(out_resp)
                )
            if _looks_tail_incomplete(out_resp) and appended < max_continue_chars:
                try:
                    final_tail = out_resp[-1200:]
                    if user_lang == "zh":
                        final_prompt = (
                            "你上一段回答仍然没有结束。请只补全最后一句，让结尾完整自然。\n"
                            "- 不要重复前文。\n"
                            "- 只补必要的缺失部分。\n\n"
                            "<已输出内容末尾>\n"
                            f"{final_tail}\n"
                            "</已输出内容末尾>"
                        )
                    else:
                        final_prompt = (
                            "The previous answer still ends abruptly. Complete only the unfinished final sentence.\n"
                            "- Do not repeat prior text.\n"
                            "- Add only the missing ending.\n\n"
                            "<Tail of emitted answer>\n"
                            f"{final_tail}\n"
                            "</Tail of emitted answer>"
                        )
                    final_fix = str(llm_service.generate(final_prompt) or "").strip()
                    if final_fix:
                        merged_fix = _merge_with_overlap(out_resp, final_fix)
                        addition = merged_fix[len(out_resp):]
                        for piece in _iter_display_chunks(addition):
                            if not piece:
                                continue
                            _first_text_trace = _mark_first_text("llm_stream_continue", chars=len(piece))
                            if _first_text_trace:
                                yield _first_text_trace
                            yield {"type": "text", "delta": piece}
                        out_resp = merged_fix
                except Exception:
                    pass
        return out_resp

    def _stream_natural_count_reply(
        *,
        user_question: str,
        structured_count_text: str,
        file_preview: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        raw = str(structured_count_text or "").strip()
        if not raw:
            return ""

        def _extract_total(text: str) -> Optional[int]:
            patterns = [
                r"共有\s*\*?(\d+)\*?\s*份文档",
                r"there\s+are\s+(\d+)\s+files?",
                r"\btotal\b[^0-9]{0,8}(\d+)\b",
                r"\b(\d+)\s+documents?\b",
            ]
            for p in patterns:
                m = re.search(p, text, flags=re.IGNORECASE)
                if m:
                    try:
                        return int(m.group(1))
                    except Exception:
                        continue
            return None

        def _extract_categories(text: str) -> List[Tuple[str, int]]:
            out: List[Tuple[str, int]] = []
            seen = set()
            for ln in text.splitlines():
                s = ln.strip()
                if not s:
                    continue
                m = re.match(r"^-?\s*([^:：\n]{1,48})\s*[:：]\s*(\d+)\s*份?\s*$", s)
                if not m:
                    m = re.match(r"^\|\s*([^|\n]{1,48})\s*\|\s*(\d+)\s*份?\s*\|?$", s)
                if not m:
                    continue
                cat = str(m.group(1) or "").strip().strip("*")
                if cat in {"分类", "category", "---", "----"}:
                    continue
                try:
                    cnt = int(m.group(2))
                except Exception:
                    continue
                if not cat:
                    continue
                key = cat.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append((cat, cnt))
            return out

        def _looks_like_meta_reasoning(text: str) -> bool:
            low = str(text or "").lower()
            markers = [
                "the user wants me",
                "let me ",
                "first,",
                "source is provided",
                "translate the assistant reply",
                "output translation only",
                "do not echo instructions",
                "i need to",
                "looking at the",
                "wait,",
                "用户想让我",
                "让我先",
                "翻译引擎",
                "根据指令",
                "<source>",
            ]
            return any(m in low for m in markers)

        def _fallback_explanation(total_num: Optional[int], cats: List[Tuple[str, int]]) -> str:
            top = cats[:3]
            if user_lang == "zh":
                if total_num and top:
                    cat_txt = "、".join([f"{c}{n}份" for c, n in top])
                    return f"从分布看，当前文件以 {cat_txt} 为主。你可以继续让我按任一分类细分，或按关键词筛选具体文件。"
                if total_num:
                    return f"当前共 {total_num} 份文件。你可以继续让我按分类细分，或按关键词筛选具体文件。"
                return "你可以继续让我按分类细分，或按关键词筛选具体文件。"
            if total_num and top:
                cat_txt = ", ".join([f"{c} ({n})" for c, n in top])
                return f"From this distribution, the largest groups are {cat_txt}. I can further drill down by category or filter by keywords."
            if total_num:
                return f"You currently have {total_num} files. I can further drill down by category or filter by keywords."
            return "I can further drill down by category or filter by keywords."

        total = _extract_total(raw)
        categories = _extract_categories(raw)
        fallback = _fallback_explanation(total, categories)

        # If the handler already provided a conversational follow-up, do not append anything.
        if "需要我详细介绍某一份文件吗" in raw or "explain one specific file in detail" in raw:
            return ""

        # Directly return the fast local fallback rule, completely bypassing LLM to avoid latency and hallucinations.
        return fallback

    def _icon_type_for_path(p: str) -> str:

        ext = os.path.splitext((p or "").lower())[1].lstrip(".")
        if ext == "pdf":
            return "pdf"
        if ext in {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic"}:
            return "image"
        if ext in {"xls", "xlsx", "csv"}:
            return "sheet"
        if ext in {"mp4", "mov", "avi", "mkv", "webm", "flv", "wmv"}:
            return "video"
        if ext in {"mp3", "wav", "m4a", "flac", "aac", "ogg"}:
            return "audio"
        return "doc"

    def _emit_files_from_sources(
        srcs: List[Dict[str, Any]],
        *,
        total_matches: Optional[Any] = None,
        shown_count: Optional[Any] = None,
    ):

        raw_sources = _dedupe_sources_by_file_identity(list(srcs or []))
        emit_cap = max(1, int(os.getenv("FILES_EMIT_MAX_K", "50")))
        _sum_cap_raw = str(os.getenv("FILES_EMIT_DOC_SUMMARY_MAX_CHARS", "") or "").strip()
        emit_summary_max: Optional[int] = None
        if _sum_cap_raw and _sum_cap_raw != "0":
            try:
                emit_summary_max = max(256, int(_sum_cap_raw))
            except ValueError:
                emit_summary_max = None
        effective_total = len(raw_sources)
        try:
            if total_matches is not None:
                parsed_total = int(total_matches)
                if parsed_total >= 0:
                    effective_total = max(effective_total, parsed_total)
        except Exception:
            pass

        if not raw_sources:
            yield {
                "type": "files",
                "total": effective_total,
                "total_matches": effective_total,
                "shown_count": 0,
                "preview": [],
                "all": [],
            }
            return

        def _virtual_tree_path_for_folder_chain(s: Dict[str, Any], fp: str) -> str:

            """Build a virtual path rooted at the matched folder for frontend tree rendering."""
            if not s.get("is_folder_chain_match"):
                return str(fp or "")
            fc_root = str(s.get("folder_chain_root") or "").strip()
            fc_rel = str(s.get("folder_chain_relative_path") or "").replace("\\", "/").strip()
            if not fc_root:
                return str(fp or "")
            try:
                rbn = os.path.basename(os.path.normpath(os.path.expanduser(fc_root))) or ""
            except Exception:
                rbn = os.path.basename(fc_root) or ""
            if not rbn:
                return str(fp or "")
            if fc_rel:
                return f"{rbn}/{fc_rel}".replace("//", "/")
            return rbn

        files = []
        for s in raw_sources[:emit_cap]:
            fp = s.get("file_path") or ""
            fn = _best_display_name_for_source(s) or os.path.basename(fp) or ""
            icon = s.get("iconType") or s.get("type") or _icon_type_for_path(fp)
            ds = str(s.get("doc_summary", "") or "")
            if emit_summary_max is not None and len(ds) > emit_summary_max:
                ds = ds[: max(0, emit_summary_max - 1)].rstrip() + "…"
            tp = _virtual_tree_path_for_folder_chain(s, fp)
            row: Dict[str, Any] = {
                "file_name": fn,
                "file_path": fp,
                "doc_category": s.get("doc_category", "") or s.get("category", ""),
                "doc_summary": ds,
                "type": icon,
                "iconType": icon,
            }
            if tp and tp != fp:
                row["tree_path"] = tp
            if s.get("is_folder_chain_match"):
                row["from_folder_chain"] = True
                fc_root = str(s.get("folder_chain_root") or "").strip()
                if fc_root:
                    row["folder_chain_root"] = fc_root
            # folder_cards: pass through folder-specific metadata for frontend rendering
            if s.get("is_matched_folder"):
                row["is_matched_folder"] = True
                row["child_file_count"] = s.get("child_file_count", 0)
            files.append(row)
        
        preview = files[: max(1, int(os.getenv("FILES_PREVIEW_K", "20")))]
        effective_shown = len(files)
        try:
            if shown_count is not None:
                parsed_shown = int(shown_count)
                if parsed_shown >= 0:
                    effective_shown = min(max(effective_shown, parsed_shown), effective_total)
        except Exception:
            pass
        yield {
            "type": "files",
            "total": effective_total,  # backward-compatible alias
            "total_matches": effective_total,
            "shown_count": effective_shown,
            "preview": preview,
            "all": files,
        }
        _first_files_trace = _mark_first_files(
            "files_event",
            total_matches=effective_total,
            shown_count=effective_shown,
        )
        if _first_files_trace:
            yield _first_files_trace

    def _extract_json_block_loose(text: str) -> Optional[str]:
        if not text:
            return None
        s = str(text).strip()
        if not s:
            return None
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            return s
        fenced = re.search(r"```json\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return s[start : end + 1]
        return None

    def _ellipsis_at_word_boundary(text: str, max_len: int) -> str:
        t = (text or "").strip()
        if max_len < 12:
            max_len = 12
        suffix = "..."
        if len(t) <= max_len:
            return t
        budget = max(1, max_len - len(suffix))
        chunk = t[:budget]
        if budget < len(t):
            nxt = t[budget : budget + 1]
            if nxt and (nxt.isalnum() or nxt in "-'"):
                sp = chunk.rfind(" ")
                nl = chunk.rfind("\n")
                cut = max(sp, nl)
                min_keep = max(8, budget // 5)
                if cut >= min_keep:
                    chunk = chunk[:cut]
        chunk = chunk.rstrip()
        while chunk.endswith("-") and len(chunk) > 1:
            chunk = chunk[:-1].rstrip()
        chunk = chunk.rstrip(" ,;:")
        if not chunk:
            chunk = t[: max(1, budget - 1)].rstrip()
        return chunk + suffix

    _INDEXER_DOC_SUMMARY_CHAR_CAP = 600

    def _softer_indexer_summary_hard_cap(text: str) -> str:
        t = (text or "").strip()
        if len(t) != _INDEXER_DOC_SUMMARY_CHAR_CAP:
            return t
        return _ellipsis_at_word_boundary(t, _INDEXER_DOC_SUMMARY_CHAR_CAP - len("..."))

    def _clip_one_line(text: Any, limit: int = 220) -> str:
        s = re.sub(r"\s+", " ", str(text or "")).strip()
        s = _softer_indexer_summary_hard_cap(s)
        return _ellipsis_at_word_boundary(s, limit)

    def _normalize_summary_text(text: Any) -> str:
        s = str(text or "").strip()
        s = re.sub(r"\n+", " ", s)
        return re.sub(r"\s{2,}", " ", s).strip()

    def _display_summary_cap() -> int:

        raw = str(os.getenv("SEARCH_DISPLAY_SUMMARY_MAX_CHARS", "400") or "").strip()
        if raw in ("0", "unlimited", "none"):
            return 10**9
        try:
            return max(120, int(raw))
        except ValueError:
            return 400

    def _fallback_brief_text_chunk_cap() -> int:

        raw = str(os.getenv("SEARCH_FALLBACK_TEXT_CHUNK_CHARS", "24000") or "").strip()
        try:
            return max(2000, int(raw))
        except ValueError:
            return 24000

    def _build_clickable_file_link(file_name: str, file_path: str) -> str:

        from urllib.parse import quote

        name = str(file_name or os.path.basename(file_path) or "file")
        name = name.replace("[", r"\[").replace("]", r"\]")
        path = str(file_path or "").strip()
        if not path:
            return f"`{name}`"
        return f"[{name}](unfoldly://open?path={quote(path, safe='')})"

    def _best_summary_for_clickable_line(src: Dict[str, Any], kb_opt: Any) -> str:

        """Use the longest available text when composing a clickable link plus summary."""
        fp = str(src.get("file_path") or "").strip()
        chunks: List[str] = [
            _normalize_summary_text(src.get("doc_summary") or ""),
            _normalize_summary_text(src.get("llm_refine_brief") or ""),
        ]
        prefetch = str(os.getenv("SEARCH_CLICKABLE_PREFETCH_SUMMARY", "1") or "").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        if prefetch and kb_opt is not None and fp:
            try:
                cr = kb_opt.collection.get(
                    where={"file_path": fp},
                    limit=1,
                    include=["metadatas"],
                )
                metas = cr.get("metadatas") or []
                if metas:
                    chunks.append(_normalize_summary_text((metas[0] or {}).get("doc_summary") or ""))
            except Exception:
                pass
        best = ""
        for c in chunks:
            if len(c) > len(best):
                best = c
        if not best:
            best = _normalize_summary_text(_fallback_brief_for_source(src))
        return best

    def _render_clickable_file_lines(refined_sources: List[Dict[str, Any]]) -> str:

        if not refined_sources:
            return ""

        cap = _display_summary_cap()
        kb_line = None
        try:
            kb_line = get_kb_instance()
        except Exception:
            kb_line = None
        lines = []
        for src in refined_sources:
            if src.get("is_folder_chain_match"):
                continue
            fp = str(src.get("file_path") or "")
            fn = str(src.get("file_name") or os.path.basename(fp) or "file")
            full = _best_summary_for_clickable_line(src, kb_line)
            full = _softer_indexer_summary_hard_cap(full)
            if cap < 10**8 and len(full) > cap:
                full = _ellipsis_at_word_boundary(full, cap)
            link = _build_clickable_file_link(fn, fp)
            lines.append(f"- {link} - {full}" if full else f"- {link}")
        return "\n".join(lines) + "\n\n"

    def _fallback_brief_for_source(src: Dict[str, Any]) -> str:
        raw_sum = str(src.get("doc_summary") or "")
        summary = ""
        if raw_sum:
            lines = raw_sum.split("\n")
            first_meaningful_line = next((line for line in lines if line.strip() and not line.strip().startswith("```") and not any("\u4e00" <= ch <= "\u9fff" for ch in line)), "")
            if first_meaningful_line:
                summary = _normalize_summary_text(first_meaningful_line)
            else:
                summary = _normalize_summary_text(raw_sum)
            if len(summary) > 1500:
                summary = summary[:1500].rstrip() + "..."
                
        if summary:
            return summary
        tcap = _fallback_brief_text_chunk_cap()
        text = _normalize_summary_text(src.get("text") or "")
        if text:
            if len(text) > tcap:
                return text[: max(0, tcap - 1)].rstrip() + "…"
            return text
        if user_lang == "zh":
            return "该文件与当前问题可能相关，但索引摘要较少。"
        return "This file may be relevant, but the indexed summary is limited."

    def _sort_sources_by_lookup_overlap(cands: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        if not query or not cands:
            return cands
        return sort_candidates_by_topic_overlap(cands, query)

    def _refine_sources_with_llm(
        user_question: str,
        raw_sources: List[Dict[str, Any]],
        *,
        effective_category: str = "",
        retrieval_query: str = "",
        keyword_hint: str = "",
        is_lexical_fallback: bool = False,
    ) -> List[Dict[str, Any]]:
        # Bug fix: only set a category filter when intent actually resolved a meaningful
        # category. Previously we did `_normalize_category_name(effective_category or "")`,
        # but the underlying normalizer returns the default "other" for empty input. That
        # caused `_filter_sources_for_refine` to silently drop every candidate whose stored
        # doc_category was not literally "other" (e.g. "resume", "report", "paper"), turning
        # "no category constraint" into "force category=other". For queries like a person
        # name (no category detected) this dropped the truly relevant resume/report files
        # and kept only stray "other" files.
        target_cat = ""
        if effective_category:
            _cat_norm = self._normalize_category_name(str(effective_category))
            if _cat_norm and _cat_norm not in {"all", "unknown"}:
                target_cat = _cat_norm
        if str((params or {}).get("_scope_disambiguation") or "").strip() in {"previous_choice", "selected_choice"}:
            pending_query = str((params or {}).get("query") or "").strip()
            if pending_query:
                user_question = pending_query

        compatible_target_cats: set[str] = set()
        if target_cat:
            try:
                from core.retrieval.category_engine import get_compatible_categories

                compatible_target_cats = get_compatible_categories(target_cat) or {target_cat}
            except Exception:
                compatible_target_cats = {target_cat}

        def _filter_sources_for_refine(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Narrow candidates only by parsed intent category; let LLM rerank judge relevance."""
            filtered = list(cands or [])
            if target_cat:
                cat_only = [
                    src for src in filtered
                    if self._normalize_category_name(
                        str(
                            src.get("doc_category_family")
                            or src.get("doc_category")
                            or "other"
                        )
                    ) in compatible_target_cats
                ]
                if cat_only:
                    filtered = cat_only
                exact_family = [
                    src
                    for src in filtered
                    if self._normalize_category_name(
                        str(
                            src.get("doc_category_family")
                            or src.get("doc_category")
                            or "other"
                        )
                    ) == target_cat
                ]
                query_for_primary = " ".join(
                    [
                        str(user_question or ""),
                        str(retrieval_query or ""),
                        str(keyword_hint or ""),
                    ]
                ).lower()
                derivative_query = bool(
                    re.search(
                        r"\b(summary|summaries|summarize|overview|explainer|explain|analysis|review|compare|comparison)\b"
                        r"|总结|概述|解读|解释|分析|评估|对比",
                        query_for_primary,
                        re.IGNORECASE,
                    )
                )
                if target_cat == "paper" and len(exact_family) >= 3 and not derivative_query:
                    filtered = exact_family
            return filtered

        def _candidate_family(src: Dict[str, Any]) -> str:
            return self._normalize_category_name(
                str(src.get("doc_category_family") or src.get("doc_category") or "other")
            )

        def _candidate_leaf(src: Dict[str, Any]) -> str:
            return str(
                src.get("doc_category_leaf")
                or src.get("doc_category_raw")
                or src.get("doc_category")
                or "other"
            ).strip().lower()

        def _candidate_role(src: Dict[str, Any]) -> str:
            raw_role = str(src.get("doc_role") or "").strip().lower()
            role_map = {
                "primary": "primary_source",
                "primary_source": "primary_source",
                "source": "primary_source",
                "summary": "summary",
                "summarization": "summary",
                "explainer": "explainer",
                "explanation": "explainer",
                "analysis": "analysis",
                "analytical": "analysis",
                "generated": "generated_doc",
                "generated_doc": "generated_doc",
                "transcript": "transcript",
                "chat_transcript": "transcript",
                "ocr": "ocr_result",
                "ocr_result": "ocr_result",
                "reference": "reference",
                "other": "other",
            }
            normalized = role_map.get(raw_role, raw_role)
            if normalized:
                return normalized

            file_name = str(src.get("file_name") or src.get("file_path") or "").lower()
            leaf = _candidate_leaf(src)
            if any(tok in file_name for tok in ("summary", "摘要", "总结", "概述")) or leaf.endswith("_summary"):
                return "summary"
            if any(tok in file_name for tok in ("explainer", "解释", "解读", "含义")) or leaf.endswith("_explainer"):
                return "explainer"
            if any(tok in file_name for tok in ("analysis", "分析", "评估", "review", "comparison", "对比")) or leaf.endswith("_analysis"):
                return "analysis"
            return "primary_source"

        def _taxonomy_priority(src: Dict[str, Any]) -> int:
            if not target_cat:
                return 0
            family = _candidate_family(src)
            leaf = _candidate_leaf(src)
            role = _candidate_role(src)
            file_ext = os.path.splitext(
                str(src.get("file_name") or src.get("file_path") or "")
            )[1].lower()
            score = 0

            if target_cat == "paper":
                if family == "paper":
                    score += 3
                if role == "primary_source":
                    score += 5
                elif role == "reference":
                    score += 2
                elif role in {"summary", "explainer", "analysis", "generated_doc"}:
                    score -= 4
                elif role in {"transcript", "ocr_result"}:
                    score -= 2
                if leaf.endswith(("_summary", "_explainer", "_analysis")):
                    score -= 3
                if file_ext == ".pdf":
                    score += 1
            elif target_cat == "resume":
                if family == "resume":
                    score += 4
                if role == "primary_source":
                    score += 1
                elif role in {"summary", "explainer", "analysis", "generated_doc"}:
                    score -= 3
            elif target_cat in {"report", "document"}:
                if role == "primary_source":
                    score += 2
                elif role in {"summary", "explainer"}:
                    score -= 1

            return score

        def _sort_sources_for_refine(cands: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
            if not cands:
                return cands

            def _match_blob(src: Dict[str, Any]) -> str:
                fp = str(src.get("file_path") or "").lower()
                fn = str(src.get("file_name") or "").lower()
                ds = str(src.get("doc_summary") or "")
                aliases = str(src.get("lookup_aliases") or "")
                schema = str(src.get("table_schema_hint") or "")
                leaf = _candidate_leaf(src)
                role = _candidate_role(src)
                return f"{fn} {fp} {ds} {aliases} {schema} {leaf} {role}"

            def _sort_key(item: tuple[int, Dict[str, Any]]) -> tuple:
                idx, src = item
                overlap = compute_lookup_overlap_score(query, _match_blob(src)) if query else 0
                direct_score = int(src.get("_direct_score", 0) or 0)
                lexical_exact = 1 if src.get("_lexical_filename_exact") else 0
                filename_subject_anchor = 1 if src.get("_filename_subject_anchor") else 0
                bm25_score = float(src.get("_bm25_score", 0.0) or 0.0)
                rerank_score = float(src.get("rerank_score", 0.0) or 0.0)
                role_priority = _taxonomy_priority(src)
                return (
                    role_priority,
                    filename_subject_anchor,
                    lexical_exact,
                    overlap,
                    direct_score,
                    bm25_score,
                    rerank_score,
                    -idx,
                )

            indexed = list(enumerate(cands))
            indexed.sort(key=_sort_key, reverse=True)
            return [src for _, src in indexed]

        candidates = list(raw_sources or [])
        if not candidates:
            return []

        explicit_ref = (params or {}).get("_explicit_file_ref") if isinstance(params, dict) else None
        basename_query_key = ""
        if isinstance(explicit_ref, dict):
            raw_name = str(explicit_ref.get("raw_name") or "").strip()
            search_term = str(explicit_ref.get("search_term") or "").strip()
            basename_query = search_term or raw_name
            if basename_query and not has_plausible_filename_extension(basename_query):
                basename_query_key = compact_filename_key(os.path.splitext(os.path.basename(basename_query))[0])

        if basename_query_key:
            sibling_front: List[Dict[str, Any]] = []
            sibling_seen: set[str] = set()
            remainder: List[Dict[str, Any]] = []
            for src in candidates:
                file_name = str(src.get("file_name") or src.get("file_path") or "")
                file_stem_key = compact_filename_key(os.path.splitext(os.path.basename(file_name))[0])
                fp = str(src.get("file_path") or "")
                if file_stem_key == basename_query_key and fp not in sibling_seen:
                    sibling_front.append(src)
                    if fp:
                        sibling_seen.add(fp)
                else:
                    remainder.append(src)
            candidates = sibling_front + remainder

        _q_for_sort = (user_question or retrieval_query or keyword_hint or "").strip()

        selected_cap = max(1, int(os.getenv("SEARCH_LLM_REFINE_MAX_FILES", "6") or 6))
        candidate_cap_env = max(1, int(os.getenv("SEARCH_LLM_REFINE_CANDIDATES", "18") or 18))
        candidate_cap = max(candidate_cap_env, selected_cap * 3)
        if target_cat:
            # Broad category+topic searches often have many same-type candidates
            # with similar semantic scores. If we keep the refine window too small,
            # near-duplicate head results crowd out other representative files
            # before the LLM can compare them. Expand only for an already-scoped
            # category to improve recall diversity without turning every search
            # into a large refine pass.
            category_candidate_cap = max(
                1,
                int(os.getenv("SEARCH_LLM_REFINE_CANDIDATES_CATEGORY", "36") or 36),
            )
            candidate_cap = max(candidate_cap, category_candidate_cap)

        def _filename_anchor_sort_key(src: Dict[str, Any]) -> Optional[tuple]:
            if not _q_for_sort:
                return None
            file_label = str(src.get("file_name") or os.path.basename(str(src.get("file_path") or "")) or "")
            alias_blob = str(src.get("lookup_aliases") or "")
            if not file_label and not alias_blob:
                return None
            exact, overlap = lookup_match_quality(_q_for_sort, f"{file_label} {alias_blob}".strip())
            lexical_exact = bool(src.get("_lexical_filename_exact") or src.get("_lookup_match_exact"))
            direct_score = int(src.get("_direct_score", 0) or 0)
            bm25_score = float(src.get("_bm25_score", 0.0) or 0.0)
            lexical_hit = bool(src.get("_is_lexical_hit"))
            if not (
                exact
                or lexical_exact
                or overlap >= 3
                or (overlap >= 2 and (lexical_hit or direct_score > 0 or bm25_score > 0.0))
            ):
                return None
            return (
                1 if exact else 0,
                1 if lexical_exact else 0,
                int(overlap),
                direct_score,
                bm25_score,
                float(src.get("rerank_score", 0.0) or 0.0),
            )

        has_indexed_lookup_signal = bool(
            _q_for_sort
            and any(
                src.get("_is_lexical_hit")
                or src.get("_lexical_filename_exact")
                or src.get("_lookup_match_exact")
                or int(src.get("_direct_score", 0) or 0) > 0
                or float(src.get("_bm25_score", 0.0) or 0.0) > 0.0
                for src in candidates
            )
        )
        filename_subject_generic_terms = {
            "find", "search", "show", "list", "display", "retrieve", "locate",
            "file", "files", "folder", "folders", "document", "documents", "doc", "docs",
            "report", "reports", "paper", "papers", "image", "images", "photo", "photos",
            "video", "videos", "audio", "data", "table", "tables", "all", "my", "the",
            "a", "an", "of", "for", "with", "about", "related", "brand", "topic", "topics",
            "找", "搜索", "显示", "文件", "文档", "资料", "报告", "图片", "照片",
            "视频", "音频", "文件夹", "目录", "全部", "所有", "我的", "相关", "关于",
        }
        filename_subject_terms = [
            str(term or "").strip().lower()
            for term in extract_lookup_terms(_q_for_sort, max_terms=32)
            if len(str(term or "").strip()) >= 2
            and str(term or "").strip().lower() not in filename_subject_generic_terms
        ]

        def _filename_subject_anchor_sort_key(src: Dict[str, Any]) -> Optional[tuple]:
            """Prefer candidates whose own filename/title carries the topic anchor.

            Folder-path overlap is useful recall evidence, but for broad topical
            searches it can crowd title hits out of the small LLM-refine window.
            This generic pre-window sort only fires when a distinctive query term
            appears in the candidate's filename/aliases, not merely in its parent
            folder.
            """
            if not filename_subject_terms:
                return None
            file_label = str(src.get("file_name") or os.path.basename(str(src.get("file_path") or "")) or "")
            alias_blob = str(src.get("lookup_aliases") or "")
            label_blob = f"{file_label} {alias_blob}".strip()
            if not label_blob:
                return None
            label_low = label_blob.lower()
            if not any(term in label_low for term in filename_subject_terms):
                return None
            exact, overlap = lookup_match_quality(_q_for_sort, label_blob)
            direct_score = int(src.get("_direct_score", 0) or 0)
            bm25_score = float(src.get("_bm25_score", 0.0) or 0.0)
            lexical_signal = bool(
                src.get("_is_lexical_hit")
                or src.get("_lexical_filename_exact")
                or src.get("_lookup_match_exact")
                or direct_score > 0
                or bm25_score > 0.0
            )
            if not lexical_signal and not exact and overlap <= 0:
                return None
            return (
                1 if exact else 0,
                int(overlap),
                1 if src.get("_lexical_filename_exact") or src.get("_lookup_match_exact") else 0,
                direct_score,
                bm25_score,
                float(src.get("rerank_score", 0.0) or 0.0),
            )

        if has_indexed_lookup_signal:
            # Make the refine window lexical-aware before applying candidate_cap.
            # Otherwise exact title/filename/path hits can be sliced out before
            # the LLM ever sees them, especially when rerank scores are tied.
            candidates = _sort_sources_by_lookup_overlap(candidates, _q_for_sort)
            filename_front: List[Tuple[tuple, Dict[str, Any]]] = []
            seen_front_paths: set[str] = set()
            for src in candidates:
                score_key = _filename_anchor_sort_key(src)
                if score_key is None:
                    continue
                fp = str(src.get("file_path") or src.get("file_name") or "")
                if fp and fp in seen_front_paths:
                    continue
                filename_front.append((score_key, src))
                if fp:
                    seen_front_paths.add(fp)
            if filename_front:
                filename_front.sort(key=lambda item: item[0], reverse=True)
                front_paths = {
                    str(src.get("file_path") or src.get("file_name") or "")
                    for _, src in filename_front
                }
                candidates = [src for _, src in filename_front] + [
                    src
                    for src in candidates
                    if str(src.get("file_path") or src.get("file_name") or "") not in front_paths
                ]
        filename_subject_front: List[Tuple[tuple, Dict[str, Any]]] = []
        seen_subject_paths: set[str] = set()
        for src in candidates:
            score_key = _filename_subject_anchor_sort_key(src)
            if score_key is None:
                continue
            fp = str(src.get("file_path") or src.get("file_name") or "")
            if fp and fp in seen_subject_paths:
                continue
            src["_filename_subject_anchor"] = True
            filename_subject_front.append((score_key, src))
            if fp:
                seen_subject_paths.add(fp)
        if filename_subject_front:
            filename_subject_front.sort(key=lambda item: item[0], reverse=True)
            subject_paths = {
                str(src.get("file_path") or src.get("file_name") or "")
                for _, src in filename_subject_front
            }
            candidates = [src for _, src in filename_subject_front] + [
                src
                for src in candidates
                if str(src.get("file_path") or src.get("file_name") or "") not in subject_paths
            ]
        candidates = _filter_sources_for_refine(candidates[:candidate_cap])
        candidates = _sort_sources_by_lookup_overlap(candidates, _q_for_sort)
        if not candidates:
            return []

        personal_attribute_route = str((params or {}).get("_expert_route") or "").strip() == "personal_attribute"
        resolved_entity_hint = str((params or {}).get("_resolved_entity") or "").strip()
        resolved_attribute_hint = str((params or {}).get("_resolved_attribute") or "").strip().lower()
        if personal_attribute_route:
            def _personal_attribute_sort_key(src: Dict[str, Any]) -> tuple[int, int, int, float]:
                blob = " ".join(
                    [
                        str(src.get("file_name") or ""),
                        str(src.get("file_path") or ""),
                        str(src.get("doc_summary") or ""),
                        str(src.get("text") or "")[:600],
                    ]
                )
                exact, overlap = lookup_match_quality(resolved_entity_hint or user_question, blob)
                entity_bonus = 0
                if resolved_entity_hint and resolved_entity_hint.lower() in blob.lower():
                    entity_bonus = 3
                elif exact:
                    entity_bonus = 2
                elif overlap >= 2:
                    entity_bonus = 1

                attr_bonus = 0
                if resolved_attribute_hint and resolved_attribute_hint in blob.lower():
                    attr_bonus = 1

                return (
                    entity_bonus,
                    int(overlap),
                    attr_bonus,
                    float(src.get("rerank_score", 0.0) or 0.0),
                )

            candidates = sorted(candidates, key=_personal_attribute_sort_key, reverse=True)
        else:
            candidates = _sort_sources_for_refine(candidates, _q_for_sort)

        basename_sibling_indices: List[int] = []
        basename_sibling_cap = 0
        if basename_query_key:
            for idx, src in enumerate(candidates, 1):
                file_name = str(src.get("file_name") or src.get("file_path") or "")
                file_stem_key = compact_filename_key(os.path.splitext(os.path.basename(file_name))[0])
                if not file_stem_key or file_stem_key != basename_query_key:
                    continue
                if not (
                    src.get("_lexical_filename_exact")
                    or bool(src.get("_lookup_match_exact"))
                    or int(src.get("_direct_score", 0) or 0) >= 90
                ):
                    continue
                basename_sibling_indices.append(idx)
            basename_sibling_cap = min(max(2, len(basename_sibling_indices)), 6) if basename_sibling_indices else 0

        lookup_heavy_query = is_lookup_heavy_query(_q_for_sort)
        strong_lookup_indices: List[int] = []
        filename_anchor_indices: List[int] = []
        for idx, src in enumerate(candidates, 1):
            match_exact, overlap_score = lookup_match_quality(
                _q_for_sort or keyword_hint,
                (
                    f"{src.get('file_name', '')} {src.get('file_path', '')} "
                    f"{src.get('lookup_aliases', '')} {src.get('doc_summary', '')}"
                ),
            )
            src["_lookup_match_exact"] = bool(match_exact)
            src["_lookup_overlap_score"] = int(overlap_score)
            if match_exact or overlap_score >= 3:
                strong_lookup_indices.append(idx)
            if src.get("_filename_subject_anchor") or _filename_anchor_sort_key(src) is not None:
                filename_anchor_indices.append(idx)

        cat_scope = str(effective_category or "").strip()
        if cat_scope:
            cat_max = max(12, int(os.getenv("SEARCH_LLM_REFINE_MAX_FILES_CATEGORY", "24") or 24))
            selected_cap = max(selected_cap, min(len(candidates), cat_max))
        if basename_sibling_cap:
            selected_cap = max(selected_cap, min(len(candidates), basename_sibling_cap))

        if (
            is_lexical_fallback
            and not cat_scope
            and not basename_query_key
            and not explicit_ref
            and len(candidates) > selected_cap
            and (len(filename_anchor_indices) > selected_cap or len(strong_lookup_indices) > selected_cap)
        ):
            lexical_multi_cap = max(
                selected_cap,
                int(os.getenv("SEARCH_LLM_REFINE_MAX_FILES_LEXICAL", "12") or 12),
            )
            selected_cap = max(selected_cap, min(len(candidates), lexical_multi_cap))

        fallback_briefs: Dict[str, str] = {}
        for src in candidates:
            fp = str(src.get("file_path") or "")
            if fp:
                fallback_briefs[fp] = _fallback_brief_for_source(src)

        try:
            _ref_prompt_sum = max(
                200,
                int(str(os.getenv("SEARCH_LLM_REFINE_PROMPT_SUMMARY_CHARS", "200") or "200").strip() or "200"),
            )
        except ValueError:
            _ref_prompt_sum = 200
        llm = self._get_llm_service(detailed=False, session_id=session_id, prompt_language=user_lang)
        blocks = []
        for idx, src in enumerate(candidates, 1):
            raw_summary = src.get("doc_summary") or src.get("text") or ""
            summary_text = _clip_one_line(raw_summary, _ref_prompt_sum)

            fp0 = str(src.get("file_path") or "")
            fn0 = str(src.get("file_name") or os.path.basename(fp0) or "")
            folder_path = os.path.dirname(fp0) if fp0 else ""
            dir_info = f" [folder: {folder_path}]" if folder_path else ""
            taxonomy_bits = [f"family={_candidate_family(src)}"]
            leaf0 = _candidate_leaf(src)
            role0 = _candidate_role(src)
            if leaf0 and leaf0 != taxonomy_bits[0].split("=", 1)[1]:
                taxonomy_bits.append(f"leaf={leaf0}")
            if role0:
                taxonomy_bits.append(f"role={role0}")
            taxonomy_info = f" [taxonomy: {', '.join(taxonomy_bits)}]"
            blocks.append(
                f"[{idx}] {fn0}{dir_info}{taxonomy_info}: {summary_text}"
            )

        _pn_user = (user_question or "").strip()
        _pn_retrieval = (retrieval_query or "").strip()
        _pn_kw = (keyword_hint or "").strip()
        if _pn_user and _pn_retrieval and _pn_user != _pn_retrieval:
            primary_need = f"{_pn_retrieval} ({_pn_user})"
        else:
            primary_need = _pn_user or _pn_retrieval or _pn_kw
            
        extra_rules = ""
        if is_lexical_fallback:
            extra_rules = (
                "5. The user is searching with filename-like or title-like anchors. "
                "Keep candidates whose filename, folder path, aliases, or summary match the distinctive subject words, even across Chinese/English translation.\n"
                "6. Generic type words such as file, document, doc, plan, report, image, photo, video, and data are NOT enough by themselves. "
                "Discard candidates that only match those generic type words but miss the specific subject.\n"
                "7. If the user need resembles a title, prefer a filename/title match over a related derivative, outline, or generic topical note. Do not drop exact title-like candidates just because another candidate has a richer summary.\n"
            )
        elif target_cat == "paper":
            extra_rules = (
                "5. This is a source-paper retrieval task. When original paper files and derived summaries/explainers/analysis notes are both present, prefer the original source papers.\n"
                "6. Treat taxonomy role metadata as meaningful: role=primary_source should rank ahead of role=summary, role=explainer, role=analysis, or role=generated_doc unless the source file is clearly irrelevant.\n"
                "7. If multiple different papers clearly match the topic, keep a broad set of those papers. Do not collapse the result to only the first few strongest-scored files when other distinct papers are also clearly relevant.\n"
            )
        elif target_cat == "resume":
            extra_rules = (
                "5. This is a file-type retrieval task. If several independent files clearly match the requested subject, keep all clearly relevant files within the visible window rather than collapsing to one result.\n"
                "6. Prefer a candidate's own filename, aliases, metadata, and summary over folder-only context. A shared parent folder is supporting evidence, not sufficient evidence by itself.\n"
                "7. If the user asks for a specific person or entity, keep candidates that are directly about that same subject ahead of loosely related documents.\n"
            )
        elif target_cat in {"report", "document"}:
            extra_rules = (
                "5. Use taxonomy role metadata to separate primary documents from derived notes. Prefer role=primary_source when it directly satisfies the user need.\n"
                "6. If the user's wording looks like a document title, a candidate whose filename/title contains the distinctive words is a direct match and MUST be included even if its short summary is sparse.\n"
            )
        elif personal_attribute_route:
            entity_hint_line = (
                f"6. Target person hint: '{resolved_entity_hint}'. Any candidate explicitly about this same person MUST be kept ahead of ambiguous same-surname or loosely related documents.\n"
                if resolved_entity_hint
                else ""
            )
            extra_rules = (
                "5. This is a personal-attribute lookup (email/phone/address/contact). Prefer documents whose PRIMARY subject is the same person asked about.\n"
                "6. Do NOT discard a candidate just because the requested attribute is missing from the short summary. Keep the best profile/resume/card/contact documents for that person so the final answer can verify whether the attribute exists.\n"
                + entity_hint_line
            )
        elif lookup_heavy_query and strong_lookup_indices:
            extra_rules = (
                "5. The user query is identifier/file-name heavy. If a candidate's filename or path strongly overlaps with the query tokens, numbers, or mixed-language name, you MUST keep it unless it is clearly unrelated.\n"
                "6. Prefer exact or near-exact filename/path overlap over generic semantic similarity when choosing between candidates.\n"
            )

        prompt = (
            f"You are a relevance filter evaluating search candidates against the USER NEED: '{primary_need}'.\n"
            f"Your task is to identify files that are relevant to this need.\n"
            f"Each candidate may include taxonomy hints in the form family=..., leaf=..., role=.... Use them to distinguish original sources from derived summaries or analysis notes.\n\n"
            f"RULES:\n"
            f"1. DIRECT MATCH: If a candidate's file name, folder path, or summary directly matches the user need, MUST include it.\n"
        )
        
        if keyword_hint and len(keyword_hint.strip()) >= 2:
            prompt += f"2. KEYWORD PRIORITY: The user search implies the specific keyword '{keyword_hint}'. ANY candidate whose file name contains '{keyword_hint}' MUST be considered highly relevant and MUST NOT be discarded.\n"
        else:
            prompt += f"2. PATH/FOLDER INTENT: If the user explicitly asks for files in a specific folder or topic (e.g. 'homework' or 'reports'), prioritize files whose folder path or content matches this topic. But do NOT blindly reject files just because they are in a different folder if their content is highly relevant.\n"

        prompt += (
            f"3. RELEVANCE THRESHOLD: Do not blindly include documents just because they share a generic keyword (e.g. 'email', 'report'). The document must be genuinely relevant to the specific subject requested by the user.\n"
            f"4. FAULT TOLERANCE: If a candidate seems highly relevant to the subject but is missing an exact secondary keyword, STILL include it. Rely on semantic relevance rather than strict word matching.\n"
            f"5. SPECIFIC ANCHORS: Proper names, exact title words, product/model codes, invoice/order numbers, amounts, filenames, and bilingual name/title terms outweigh generic file-type words such as guide, pdf, report, document, image, or data.\n"
            f"6. BREADTH FOR BROAD SEARCHES: When the user need naturally matches multiple independent files (for example several resumes, several papers, several recordings, or several photos), include all clearly relevant candidates within the visible window instead of collapsing to only one cluster or one near-duplicate group.\n"
            f"7. PRIORITY ORDER: Return selected_indices in descending relevance order. Put candidates matching the distinctive anchors before candidates that only match generic type words.\n"
            f"8. ZERO RESULT: If NO candidates are relevant, you MUST return exactly: {{\"selected_indices\": [0]}}\n"
        )
        
        if extra_rules:
            prompt += f"\n[SPECIAL LEXICAL RULES]\n{extra_rules}\n"
            
        prompt += (
            f"\nOutput ONLY a valid JSON object with key `selected_indices` (array of 1-based indices, or [0] if none match). Example: {{\"selected_indices\": [1, 2]}}\n"
            f"DO NOT include any explanation, <think> tags, or markdown formatting, only the JSON.\n\n"
            f"Candidates:\n"
            + "[0] None of the above candidates are relevant.\n"
            + "\n".join(blocks)
        )

        if hasattr(llm, "force_text_model"):
            llm.force_text_model = True

        try:
            raw = str(llm.generate(prompt) or "").strip()
        except Exception:
            raw = ""

        if self.is_aborted(session_id):
            return []

        payload = _extract_json_block_loose(raw)
        result: Any = None
        if payload:
            try:
                result = json.loads(payload)
            except Exception:
                result = None

        def _indices_from_refine_json(obj: Any) -> List[int]:
            """Accept several local-model JSON shapes for selected candidate indices."""
            acc: List[int] = []
            if obj is None:
                return acc
            if isinstance(obj, dict):
                for key in ("selected_indices", "indices", "indexes", "selected"):
                    raw_list = obj.get(key)
                    if not isinstance(raw_list, list):
                        continue
                    for x in raw_list:
                        try:
                            acc.append(int(x))
                        except (TypeError, ValueError):
                            pass
                    if acc:
                        return acc
                return acc
            if isinstance(obj, list):
                if not obj:
                    return acc
                if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
                    for x in obj:
                        try:
                            acc.append(int(x))
                        except (TypeError, ValueError):
                            pass
                    return acc
                if all(isinstance(x, dict) for x in obj):
                    for item in obj:
                        if not isinstance(item, dict):
                            continue
                        for k in ("index", "idx", "i", "n", "rank", "candidate_index"):
                            if k not in item:
                                continue
                            try:
                                acc.append(int(item[k]))
                            except (TypeError, ValueError):
                                pass
                            break
            return acc

        ordered_indices: List[int] = []
        for idx in _indices_from_refine_json(result):
            if idx < 0 or idx > len(candidates):
                continue
            if idx not in ordered_indices:
                ordered_indices.append(idx)

        pre_fallback_indices = list(ordered_indices)

        def _explicit_empty_refine_selection(parsed: Any, raw_text: str = "") -> bool:
            """Return true when the LLM explicitly says no candidate should be selected."""
            if isinstance(parsed, list):
                if len(parsed) == 0 or (len(parsed) == 1 and parsed[0] == 0):
                    return True
            if isinstance(parsed, dict):
                for key in ("selected_indices", "indices", "indexes", "selected"):
                    if key not in parsed:
                        continue
                    val = parsed.get(key)
                    if isinstance(val, list):
                        if len(val) == 0 or (len(val) == 1 and val[0] == 0):
                            return True
            
            # Fallback check on raw text if JSON parsing failed or wasn't clean
            if parsed is None and raw_text:
                s_lower = str(raw_text).lower()
                if "[0]" in s_lower or "none of the above" in s_lower or "no candidate" in s_lower or "not relevant" in s_lower:
                    return True
                
            return False

        explicit_empty = _explicit_empty_refine_selection(result, raw)
        if personal_attribute_route and explicit_empty:
            entity_seed = resolved_entity_hint or user_question
            entity_kept: List[int] = []
            for idx, src in enumerate(candidates, 1):
                blob = " ".join(
                    [
                        str(src.get("file_name") or ""),
                        str(src.get("file_path") or ""),
                        str(src.get("doc_summary") or ""),
                        str(src.get("text") or "")[:600],
                    ]
                )
                exact, overlap = lookup_match_quality(entity_seed, blob)
                if exact or overlap >= 2:
                    entity_kept.append(idx)
                if len(entity_kept) >= selected_cap:
                    break
            if entity_kept:
                ordered_indices = entity_kept
                explicit_empty = False
                logger.info(
                    "[search_llm_refine] personal_attribute safeguard kept entity-matching candidates: %s",
                    ordered_indices,
                )

        if is_lexical_fallback and explicit_empty:
            lexical_empty_kept: List[tuple[tuple, int]] = []
            lexical_rescue_floor = max(float(settings.RELEVANCE_THRESHOLD), 2.0)
            rescue_query = (_q_for_sort or keyword_hint or retrieval_query or user_question).strip()
            for idx, src in enumerate(candidates, 1):
                affinity_blob = " ".join(
                    [
                        str(src.get("file_name") or ""),
                        str(src.get("file_path") or ""),
                        str(src.get("lookup_aliases") or ""),
                        str(src.get("doc_summary") or ""),
                        str(src.get("table_schema_hint") or ""),
                        str(src.get("text") or "")[:800],
                    ]
                )
                match_exact, lookup_overlap = lookup_match_quality(rescue_query, affinity_blob)
                topic_overlap = compute_lookup_overlap_score(rescue_query, affinity_blob)
                strongest_overlap = max(int(lookup_overlap), int(topic_overlap))
                rerank_score = float(src.get("rerank_score", 0.0) or 0.0)
                bm25_score = float(src.get("_bm25_score", 0.0) or 0.0)
                lexical_exact = bool(src.get("_lexical_filename_exact") or src.get("_lookup_match_exact"))
                if not (
                    match_exact
                    or lexical_exact
                    or strongest_overlap >= 4
                    or (strongest_overlap >= 2 and rerank_score >= lexical_rescue_floor)
                    or rerank_score >= lexical_rescue_floor + 1.0
                ):
                    continue
                lexical_empty_kept.append(
                    (
                        (
                            1 if match_exact else 0,
                            1 if lexical_exact else 0,
                            strongest_overlap,
                            rerank_score,
                            bm25_score,
                            int(src.get("_direct_score", 0) or 0),
                        ),
                        idx,
                    )
                )
            if lexical_empty_kept:
                lexical_empty_kept.sort(key=lambda item: item[0], reverse=True)
                ordered_indices = [idx for _, idx in lexical_empty_kept[:selected_cap]]
                explicit_empty = False
                logger.info(
                    "[search_llm_refine] lexical explicit-empty safeguard kept high-signal candidates: %s",
                    ordered_indices,
                )

        #
        #
        def _is_must_rescue_lexical_hit(doc: Dict[str, Any]) -> bool:
            if not doc.get("_is_lexical_hit"):
                return False
            if doc.get("_lexical_filename_exact") or doc.get("_lookup_match_exact"):
                return True
            if int(doc.get("_direct_score", 0) or 0) >= 95:
                file_name = str(doc.get("file_name") or doc.get("file_path") or "")
                exact_match, overlap = lookup_match_quality(
                    user_question or retrieval_query or keyword_hint,
                    file_name,
                )
                if exact_match or overlap >= 2:
                    return True
            return False

        rescue_lexical_indices: List[int] = []
        for i, doc in enumerate(candidates):
            if _is_must_rescue_lexical_hit(doc):
                idx = i + 1
                if idx not in ordered_indices and idx not in rescue_lexical_indices:
                    rescue_lexical_indices.append(idx)

        if rescue_lexical_indices:
            if not ordered_indices:
                ordered_indices = list(rescue_lexical_indices)
                explicit_empty = False
            else:
                for idx in rescue_lexical_indices:
                    if idx not in ordered_indices:
                        ordered_indices.append(idx)

        force_fallback_on_llm_empty = str(
            os.getenv("SEARCH_LLM_REFINE_FORCE_FALLBACK_ON_LLM_EMPTY", "") or ""
        ).strip().lower() in ("1", "true", "yes")
        skip_retention_fallback = explicit_empty and not force_fallback_on_llm_empty

        try:
            raw_log = raw if len(raw) <= 12000 else (raw[:12000] + "…[truncated]")
            parsed_repr = (
                json.dumps(result, ensure_ascii=False)
                if result is not None
                else "null"
            )
            logger.info(
                "[search_llm_refine] candidates=%s selected_cap=%s raw_response=%r",
                len(candidates),
                selected_cap,
                raw_log,
            )
            logger.info(
                "[search_llm_refine] extracted_payload=%r parsed_json=%s llm_selected_indices=%s explicit_empty=%s",
                payload,
                parsed_repr,
                pre_fallback_indices,
                explicit_empty,
            )
        except Exception:
            pass

        try:
            logger.info(
                f"[search_llm_refine] final_indices={pre_fallback_indices} fallback_used={False}"
            )
            # ==========================================================
        except Exception:
            pass

        no_refine_fallback = str(os.getenv("SEARCH_LLM_REFINE_NO_FALLBACK", "") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        if not ordered_indices and candidates and not no_refine_fallback and not skip_retention_fallback:
            for i, doc in enumerate(candidates):
                if float(doc.get("rerank_score", 0.0) or 0.0) >= 4.0 or doc.get("is_folder_match", False):
                    if (i + 1) not in ordered_indices:
                        ordered_indices.append(i + 1)

            if not ordered_indices:
                ordered_indices = [i + 1 for i in range(min(3, len(candidates)))]

            logger.info(
                f"[search_llm_refine] LLM empty/unparsed; applied retention fallback. final_indices={ordered_indices}"
            )
        elif skip_retention_fallback:
            logger.info(
                "[search_llm_refine] LLM 明确返回空选择，跳过捞回/关键词兜底（SEARCH_LLM_REFINE_FORCE_FALLBACK_ON_LLM_EMPTY=1 可恢复旧行为）"
            )

        if not ordered_indices and candidates and not no_refine_fallback and not skip_retention_fallback:
            _fb_noise = frozenset(
                {
                    "the",
                    "and",
                    "for",
                    "with",
                    "from",
                    "this",
                    "that",
                    "about",
                    "into",
                    "through",
                    "your",
                    "their",
                    "what",
                    "when",
                    "where",
                    "which",
                    "have",
                    "has",
                    "been",
                    "will",
                    "would",
                    "could",
                    "should",
                    "details",
                    "detail",
                    "information",
                    "informations",
                    "content",
                    "contents",
                    "document",
                    "documents",
                    "file",
                    "files",
                    "guide",
                    "guides",
                    "overview",
                    "summary",
                    "summaries",
                    "description",
                    "descriptions",
                    "related",
                    "based",
                    "using",
                    "model",
                    "models",
                    "system",
                    "systems",
                    "data",
                    "analysis",
                    "method",
                    "methods",
                    "paper",
                    "papers",
                    "technical",
                    "training",
                    "learning",
                    "understanding",
                    "cross",
                    "modal",
                    "multimodal",
                }
            )
            qcomb = f"{retrieval_query or ''} {keyword_hint or ''} {user_question or ''}"
            q_lower = qcomb.lower()
            tokens = [t for t in re.split(r"[\s,/_.-]+", q_lower) if len(t) >= 3 and t not in _fb_noise]
            if not tokens:
                tokens = [
                    t
                    for t in re.split(r"\W+", q_lower)
                    if t.isalnum() and len(t) >= 3 and t not in _fb_noise
                ]
            for m in re.findall(r"[\u4e00-\u9fff]{2,}", qcomb):
                if m not in tokens:
                    tokens.append(m)
            kept_idx: List[int] = []
            if tokens:
                for i, src in enumerate(candidates, 1):
                    blob = (
                        f"{src.get('file_name', '')} {src.get('file_path', '')} "
                        f"{src.get('doc_summary', '')} {str(src.get('text', '') or '')[:800]}"
                    ).lower()
                    if any(tok.lower() in blob for tok in tokens):
                        kept_idx.append(i)
                    if len(kept_idx) >= selected_cap:
                        break
            if kept_idx:
                ordered_indices = kept_idx
            elif str(os.getenv("SEARCH_LLM_REFINE_FALLBACK_TOPK", "") or "").strip().lower() in (
                "1",
                "true",
                "yes",
            ):
                ordered_indices = list(range(1, min(len(candidates), selected_cap) + 1))

        try:
            refine_fallback_used = (
                not pre_fallback_indices
                and bool(ordered_indices)
                and candidates
                and not no_refine_fallback
            )
            logger.info(
                "[search_llm_refine] final_indices=%s fallback_used=%s",
                ordered_indices,
                refine_fallback_used,
            )
        except Exception:
            pass

        ordered_indices = [idx for idx in ordered_indices if idx != 0]

        if strong_lookup_indices and (is_lexical_fallback or lookup_heavy_query):
            selected_has_strong_lookup = any(
                1 <= idx <= len(candidates)
                and (
                    candidates[idx - 1].get("_lookup_match_exact")
                    or int(candidates[idx - 1].get("_lookup_overlap_score", 0) or 0) >= 3
                )
                for idx in ordered_indices
            )
            if not ordered_indices or not selected_has_strong_lookup:
                merged_lookup = []
                for idx in strong_lookup_indices[: min(3, selected_cap)]:
                    if idx not in merged_lookup:
                        merged_lookup.append(idx)
                for idx in ordered_indices:
                    if idx not in merged_lookup:
                        merged_lookup.append(idx)
                ordered_indices = merged_lookup[:selected_cap]

        if basename_sibling_indices:
            sibling_selected = any(idx in basename_sibling_indices for idx in ordered_indices)
            if sibling_selected or not ordered_indices:
                merged_siblings: List[int] = []
                for idx in basename_sibling_indices:
                    if idx not in merged_siblings:
                        merged_siblings.append(idx)
                for idx in ordered_indices:
                    if idx not in merged_siblings:
                        merged_siblings.append(idx)
                ordered_indices = merged_siblings[:selected_cap]

        # ===================================================================
        #
        # basename_sibling_indices / retention fallback / keyword fallback)
        #
        # ===================================================================
        llm_picked_indices = [
            idx for idx in pre_fallback_indices
            if 1 <= idx <= len(candidates) and idx != 0
        ]
        if (
            llm_picked_indices
            and is_lexical_fallback
            and not target_cat
            and not basename_query_key
            and not explicit_ref
            and len(llm_picked_indices) > selected_cap
        ):
            lexical_llm_cap = max(
                selected_cap,
                int(os.getenv("SEARCH_LLM_REFINE_MAX_FILES_LEXICAL", "12") or 12),
            )
            selected_cap = max(
                selected_cap,
                min(len(candidates), len(llm_picked_indices), lexical_llm_cap),
            )
        if llm_picked_indices:
            anchored: List[int] = []
            for idx in llm_picked_indices:
                if idx not in anchored:
                    anchored.append(idx)
            for idx in ordered_indices:
                if idx not in anchored:
                    anchored.append(idx)
            ordered_indices = anchored

        if filename_anchor_indices:
            anchored_filename: List[int] = []
            if llm_picked_indices:
                # LLM refine is the relevance judge; filename anchors may fill
                # spare slots, but must not displace explicit LLM selections.
                for idx in ordered_indices:
                    if idx not in anchored_filename:
                        anchored_filename.append(idx)
                for idx in filename_anchor_indices[: min(selected_cap, 6)]:
                    if idx not in anchored_filename:
                        anchored_filename.append(idx)
            else:
                for idx in filename_anchor_indices[: min(selected_cap, 6)]:
                    if idx not in anchored_filename:
                        anchored_filename.append(idx)
                for idx in ordered_indices:
                    if idx not in anchored_filename:
                        anchored_filename.append(idx)
            ordered_indices = anchored_filename

        def _should_apply_broad_diversity_backfill() -> bool:
            if target_cat not in {"resume", "paper"}:
                return False
            if personal_attribute_route or basename_query_key:
                return False
            if explicit_ref or has_plausible_filename_extension(_q_for_sort):
                return False
            if _extract_identifier_filename_anchors(_q_for_sort):
                return False
            return len(candidates) >= 8 and len(ordered_indices) < min(len(candidates), selected_cap)

        def _candidate_backfill_signal(src: Dict[str, Any]) -> bool:
            overlap = int(src.get("_lookup_overlap_score", 0) or 0)
            direct_score = int(src.get("_direct_score", 0) or 0)
            bm25_score = float(src.get("_bm25_score", 0.0) or 0.0)
            rerank_score = float(src.get("rerank_score", 0.0) or 0.0)
            taxonomy_priority = _taxonomy_priority(src)
            return bool(
                src.get("_lookup_match_exact")
                or src.get("_lexical_filename_exact")
                or overlap >= 2
                or direct_score >= 60
                or bm25_score >= 8.0
                or rerank_score >= max(float(settings.RELEVANCE_THRESHOLD), 1.0)
                or taxonomy_priority > 0
            )

        def _candidate_diversity_key(src: Dict[str, Any]) -> str:
            fp = str(src.get("file_path") or "").strip()
            fn = str(src.get("file_name") or os.path.basename(fp) or "").strip()
            stem_key = compact_filename_key(os.path.splitext(fn)[0])
            if stem_key and len(stem_key) >= 6:
                return stem_key
            return fp.lower()

        if _should_apply_broad_diversity_backfill():
            diversified = list(ordered_indices)
            seen_keys = {
                _candidate_diversity_key(candidates[idx - 1])
                for idx in diversified
                if 1 <= idx <= len(candidates)
            }
            appended = 0
            for idx, src in enumerate(candidates, 1):
                if idx in diversified or len(diversified) >= selected_cap:
                    continue
                if not _candidate_backfill_signal(src):
                    continue
                diversity_key = _candidate_diversity_key(src)
                if diversity_key and diversity_key in seen_keys:
                    continue
                diversified.append(idx)
                if diversity_key:
                    seen_keys.add(diversity_key)
                appended += 1
            if appended:
                ordered_indices = diversified
                logger.info(
                    "[search_llm_refine] broad diversity backfill appended=%d target_cat=%s final_indices=%s",
                    appended,
                    target_cat,
                    ordered_indices,
                )

        if not ordered_indices:
            return []

        refined = []
        for idx in ordered_indices[:selected_cap]:
            src = dict(candidates[idx - 1])
            refined.append(src)

        if llm_picked_indices:
            picked_paths = {
                str(candidates[idx - 1].get("file_path") or "")
                for idx in llm_picked_indices[:selected_cap]
            }
            picked_paths.discard("")
            refined_paths = {str(s.get("file_path") or "") for s in refined}
            missing = picked_paths - refined_paths
            if missing:
                logger.warning(
                    "[search_llm_refine] INVARIANT VIOLATION: LLM-picked files "
                    "dropped from refined result. picked=%s refined=%s missing=%s",
                    sorted(picked_paths),
                    sorted(refined_paths),
                    sorted(missing),
                )

        return refined

    def _retain_personal_attribute_sources(
        cands: List[Dict[str, Any]],
        *,
        query_text: str,
        entity_hint: str = "",
        attribute_hint: str = "",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        if not cands:
            return []

        seed = entity_hint.strip() or query_text.strip()
        has_entity_hint = bool(entity_hint.strip())
        attr = attribute_hint.strip().lower()

        def _score(src: Dict[str, Any]) -> tuple[int, int, int, int, float]:
            blob = " ".join(
                [
                    str(src.get("file_name") or ""),
                    str(src.get("file_path") or ""),
                    str(src.get("doc_summary") or ""),
                    str(src.get("text") or "")[:800],
                ]
            )
            exact, overlap = lookup_match_quality(seed, blob)
            entity_bonus = 0
            if entity_hint and entity_hint.lower() in blob.lower():
                entity_bonus = 3
            elif exact:
                entity_bonus = 2
            elif overlap >= 2:
                entity_bonus = 1

            attr_bonus = 0
            if attr and attr in blob.lower():
                attr_bonus = 1

            cat = self._normalize_category_name(str(src.get("doc_category") or "other"))
            profile_bonus = 1 if cat in {"resume", "document", "report", "note"} else 0

            return (
                entity_bonus,
                int(overlap),
                attr_bonus,
                profile_bonus,
                float(src.get("rerank_score", 0.0) or 0.0),
            )

        ranked = sorted(cands, key=_score, reverse=True)
        def _needs_llm_same_person_check() -> bool:
            if len(ranked) < 2:
                return False
            hint = str(entity_hint or "").strip()
            if not hint:
                return False
            ascii_letters = [ch for ch in hint if ch.isalpha()]
            if not ascii_letters:
                return False
            if not all(ord(ch) < 128 for ch in ascii_letters):
                return False
            top_score = _score(ranked[0])
            second_score = _score(ranked[1])
            return top_score[0] <= 1 and second_score[0] <= 1

        if _needs_llm_same_person_check():
            prompt_blocks = []
            for idx, src in enumerate(ranked[:8], 1):
                prompt_blocks.append(
                    f"[{idx}] {str(src.get('file_name') or '')}: "
                    f"{str(src.get('doc_summary') or src.get('text') or '')[:260]}"
                )
            try:
                llm = self._get_llm_service(detailed=False, session_id=session_id, prompt_language="en")
                raw = str(
                    llm.generate(
                        "You are selecting files about the SAME person for a personal attribute lookup.\n"
                        f"User need: {query_text}\n"
                        f"Target person hint: {entity_hint}\n"
                        f"Target attribute: {attribute_hint or 'contact info'}\n\n"
                        "Select only the candidate files whose primary subject is the same person the user asked about.\n"
                        "Prefer full profiles/resumes/cards for that exact person, and reject same-surname or loosely related people.\n"
                        "Output ONLY JSON: {\"selected_indices\": [1,2]} or {\"selected_indices\": [0]} if none.\n\n"
                        + "\n".join(prompt_blocks)
                    )
                    or ""
                ).strip()
            except Exception:
                raw = ""

            payload = _extract_json_block_loose(raw)
            selected_indices: List[int] = []
            if payload:
                try:
                    parsed = json.loads(payload)
                    for idx in list(parsed.get("selected_indices") or []):
                        try:
                            idx_int = int(idx)
                        except (TypeError, ValueError):
                            continue
                        if 1 <= idx_int <= min(8, len(ranked)) and idx_int not in selected_indices:
                            selected_indices.append(idx_int)
                except Exception:
                    selected_indices = []
            if selected_indices and selected_indices != [0]:
                chosen = [ranked[idx - 1] for idx in selected_indices]
                ranked = sorted(chosen, key=_score, reverse=True) + [src for src in ranked if src not in chosen]
                logger.info(
                    "[dispatch] personal_attribute same-person disambiguation: entity=%r selected=%s raw=%r",
                    entity_hint,
                    selected_indices,
                    raw[:800],
                )

        kept = [src for src in ranked if _score(src)[0] > 0 or _score(src)[1] >= 2]
        if not kept:
            if has_entity_hint:
                logger.info(
                    "[dispatch] personal_attribute strict filter found no entity-matching candidates: entity=%r attr=%r candidates=%d",
                    entity_hint,
                    attribute_hint,
                    len(cands),
                )
                return []
            kept = ranked[: min(limit, len(ranked))]
        return kept[: min(limit, len(kept))]

    yield from _emit_status("thinking", "Understanding your question...")
    yield {"type": "thinking", "delta": "Analyzing user intent...\n"}
    
    # 1.1) ✅ QueryPreprocessor Agent: greeting / capability / translate / zero-scope
    from core.intent.preprocessor import QueryPreprocessor
    _prep_result = QueryPreprocessor.preprocess(
        q,
        has_history=bool(hist_ref and len(hist_ref) > 1),
        total_searchable=-1,  # will be set below after count
    )
    if _prep_result.intercepted and _prep_result.action == "greeting":
        yield {"type": "thinking", "delta": "Detected casual greeting; replying directly.\n"}
        llm = self._get_llm_service(detailed=False, session_id=session_id, prompt_language=user_lang)
        greet_prompt = (
            f"Reply naturally and briefly in {response_language_label}.\n\n"
            f"<User Message>\n{q}\n</User Message>\n"
        )
        resp_text = yield from _collect_or_emit_stream(llm, greet_prompt)
        if resp_text is None:
            return
        try:
            hist_ref[-1]["a"] = resp_text
        except Exception:
            pass
        yield {"type": "done", "ok": True, "query_type": "chat", "sources": [], "trace": []}
        return
    try:
        cat_stats = self.kb.count_all_categories(allowed_paths=active_paths)
        total_searchable = sum(cat_stats.values())
        if total_searchable == 0 and active_paths:
            keyword_ready = False
            try:
                keyword_ready = bool(getattr(self.kb, "is_keyword_index_ready", lambda: False)())
            except Exception:
                keyword_ready = False
            if not keyword_ready:
                logger.info(
                    "[dispatch] category inventory not ready for active scope; "
                    "treating searchable count as unknown instead of zero"
                )
                total_searchable = -1
    except Exception:
        total_searchable = -1

    # Query-time media answers are DB-only. If a selected/opened media path is
    # not indexed, do not inspect local files to keep latency bounded.

    if total_searchable == 0:
        _prep_zero = QueryPreprocessor.preprocess(
            q,
            has_history=bool(hist_ref and len(hist_ref) > 1),
            total_searchable=total_searchable,
        )
        is_translation = not _prep_zero.intercepted  # if NOT intercepted, it's a translate request
        
        if not is_translation:
            has_explicit_file_scope = bool(active_paths or opened_file_path)
            file_scoped_question = bool(
                has_explicit_file_scope
                and re.search(
                    r"\b(this|selected|opened|current)\s+"
                    r"(file|document|doc|pdf|image|photo|picture|video|audio|recording|media)\b"
                    r"|\b(file|document|doc|pdf|image|photo|picture|video|audio|recording|media)\b"
                    r"|这个|当前|选中|文件|文档|图片|照片|视频|音频|录音",
                    q,
                    re.IGNORECASE,
                )
            )
            if file_scoped_question:
                msg = (
                    "当前选中/打开的文件在索引数据库中没有可用内容，因此我不能在对话阶段直接读取本地文件来回答。请先重新索引该文件或选择已完成索引的文件。"
                    if user_lang == "zh"
                    else "The selected/opened file has no usable evidence in the index database, so I cannot read the local file directly during chat. Please index that file first or select a file that has already been indexed."
                )
                yield {"type": "thinking", "delta": "0 indexed files in selected scope; DB-only file QA cannot continue.\n"}
                yield {"type": "text", "content": msg}
                try:
                    hist_ref[-1]["a"] = msg
                except Exception:
                    pass
                yield {"type": "done", "ok": True, "query_type": "no_indexed_scope", "sources": [], "trace": []}
                return
            yield {"type": "thinking", "delta": "0 indexed files in selected scope; pivoting to fast chat bypass...\n"}
            llm = self._get_llm_service(detailed=False, session_id=session_id, prompt_language=user_lang)
            if user_lang == "zh":
                sys_msg = "由于当前系统没有任何索引文件或未选中有效文件范围，我无法进行文件检索。请明确提示用户：如果你需要进行文件相关的提问检索，请先点击界面侧边的“Add Source”添加你的文件夹数据源。这段提示之后，请无视文件检索失败的事实，直接转变身份，利用你自带的百科知识去全心全意解答用户的此问题："
            else:
                sys_msg = "Since there are 0 indexed files in the current scope or no sources have been added, I cannot perform file retrieval. Please remind the user: if you want to ask questions about your files, please click 'Add Source' on the menu to add your folders first. Right after this reminder, drop the file retrieval topic and immediately use your general AI knowledge to answer the user's question directly and fully:"
            chat_prompt = f"Reply in {response_language_label}. Do NOT output JSON.\n\n<System>\n{sys_msg}\n</System>\n\n<User Message>\n{q}\n</User Message>"
            resp_text = yield from _collect_or_emit_stream(llm, chat_prompt)
            if resp_text is None:
                return
            try:
                hist_ref[-1]["a"] = resp_text
            except Exception:
                pass
            yield {"type": "done", "ok": True, "query_type": "chat", "sources": [], "trace": []}
            return


    yield {"type": "thinking", "delta": "Analyzing intent...\n"}
    orchestrator = QueryOrchestrator(self)
    total_searchable_for_context = total_searchable if "total_searchable" in locals() else None
    query_context = orchestrator.build_query_context(
        question=q,
        normalized_question=q,
        session_id=session_id,
        prompt_language=internal_lang,
        user_language=user_lang,
        active_paths=active_paths,
        opened_file_path=opened_file_path,
        total_searchable=total_searchable_for_context,
        metadata={"response_language_label": response_language_label},
    )
    _intent_t0 = time.time()
    try:
        action_request, intent = orchestrator.analyze(query_context)
        logger.info(f"[Intent LLM] raw parsed intent: {intent}")
    except Exception as e:
        logger.error(f"Intent analysis error: {e}")
        import traceback
        traceback.print_exc()
        fallback_intent = {"action": "count", "params": {"category": "all"}} if not active_paths else {"action": "search", "params": {"query": q}}
        from core.domain import ActionRequest, IntentDecision

        fallback_decision = IntentDecision.from_legacy(
            self._normalize_intent_to_internal_en(q, fallback_intent, session_id=session_id),
            source="query_orchestrator_fallback",
        )
        action_request = ActionRequest.from_intent(
            fallback_decision,
            query_context=query_context,
            execution_mode=orchestrator.execution_mode_for(fallback_decision.action),
            query_type=orchestrator.query_type_for(fallback_decision.action),
            metadata={"fallback": True},
        )
        intent = fallback_intent
    _intent_total_ms = int((time.time() - _intent_t0) * 1000)

    action = action_request.action
    params = dict(action_request.params or {})
    intent_normalized_en = bool(action_request.metadata.get("normalized_internal_en"))
    orchestrator_timing = dict(action_request.metadata.get("timing") or {})
    if not isinstance(params, dict):
        params = {}
    routing_trace = {
        "stage": "intent_routing",
        "action": action,
        "query_type": action_request.query_type,
        "execution_mode": action_request.execution_mode.value,
        "arbiter_source": str(action_request.metadata.get("arbiter_source") or ""),
        "intent_source": str(action_request.metadata.get("intent_source") or ""),
        "expert_route": str(action_request.metadata.get("expert_route") or str((params or {}).get("_expert_route") or "")),
        "used_legacy_fallback": bool(action_request.metadata.get("used_legacy_fallback")),
        "normalized_internal_en": bool(action_request.metadata.get("normalized_internal_en")),
        "timing": orchestrator_timing,
    }
    yield {"type": "trace_append", "item": routing_trace}
    yield _emit_timing_trace(
        "intent_analysis",
        duration_ms=_intent_total_ms,
        arbiter_ms=int(orchestrator_timing.get("arbiter_ms") or 0),
        legacy_fallback_ms=int(orchestrator_timing.get("legacy_fallback_ms") or 0),
        normalize_ms=int(orchestrator_timing.get("normalize_ms") or 0),
        execution_mode=action_request.execution_mode.value,
        action=action,
    )

    yield {"type": "thinking", "delta": f"Intent detected: {action}\n"}
    yield {"type": "thinking", "delta": f"Execution mode: {action_request.execution_mode.value}, query_type: {action_request.query_type}\n"}
    yield {
        "type": "thinking",
        "delta": (
            "Intent routing: "
            f"arbiter_source={routing_trace['arbiter_source'] or 'n/a'}, "
            f"expert_route={routing_trace['expert_route'] or 'n/a'}, "
            f"legacy_fallback={routing_trace['used_legacy_fallback']}\n"
        ),
    }
    if any("\u4e00" <= ch <= "\u9fff" for ch in q):
        norm_q = str(params.get("query") or "").strip()
        if norm_q:
            yield {"type": "thinking", "delta": f"Normalized intent query (EN): {norm_q}\n"}
        norm_kw = str(params.get("keywords") or "").strip()
        if norm_kw:
            yield {"type": "thinking", "delta": f"Normalized intent keywords (EN): {norm_kw}\n"}
    action_executor = ActionExecutor(self)
    handled = yield from action_executor.execute_structured_action(
        action=action,
        params=params,
        question=q,
        active_paths=active_paths,
        opened_file_path=opened_file_path,
        session_id=session_id,
        hist_ref=hist_ref,
        user_lang=user_lang,
        response_language_label=response_language_label,
        emit_status_enabled=emit_status,
        collect_or_emit_stream=_collect_or_emit_stream,
        emit_files_from_sources=_emit_files_from_sources,
        emit_status_fn=_emit_status,
        to_user_text=_to_user_text,
        stream_natural_count_reply=_stream_natural_count_reply,
        icon_type_for_path=_icon_type_for_path,
        request_started_at=_request_t0,
    )
    if handled:
        yield _emit_timing_trace("request_done", action=action, handled=True)
        return
    if action == "count":
        action = "search"
        params = {"query": q}
    elif action == "open_file":
        fname = str(params.get("file_name") or params.get("file") or "")
        params = {"query": fname or q}

    if (params or {}).get("_scope") == "previous":
        last_results = self._get_last_search_results_ref(session_id) or []
        prev_paths = []
        for doc in last_results:
            p = doc.get("file_path")
            if p and p not in prev_paths:
                prev_paths.append(p)
        if prev_paths:
            yield {"type": "thinking", "delta": f"Limiting search scope to {len(prev_paths)} previous results...\n"}
            active_paths = prev_paths

    query_for_search = str(params.get("query") or q).strip() or q
    scope_disambiguation_reason = str((params or {}).get("_scope_disambiguation") or "").strip()
    # Multi-turn personal-attribute queries such as "and his email" or
    # Reuse the lightweight pronoun resolver instead of adding more routing rules.
    if action == "search":
        try:
            resolver_query = query_for_search
            if str((params or {}).get("_expert_route") or "").strip() == "personal_attribute":
                original_for_resolution = str(q or "").strip()
                if original_for_resolution and original_for_resolution != query_for_search:
                    resolver_query = original_for_resolution
            resolved_attr = self._resolve_pronoun_query(
                resolver_query,
                session_id=session_id,
                prompt_language=user_lang,
            )
        except Exception as _e:
            logger.warning(f"[dispatch] pronoun resolution failed: {_e}")
            resolved_attr = None
        if resolved_attr and str(resolved_attr.get("resolved_query") or "").strip():
            resolved_query = str(resolved_attr.get("resolved_query") or "").strip()
            resolved_entity = str(resolved_attr.get("entity") or "").strip()
            resolved_attribute = str(resolved_attr.get("attribute") or "").strip()
            if resolved_query and resolved_query != query_for_search:
                yield {
                    "type": "thinking",
                    "delta": f"Anchored personal attribute query to prior entity: {resolved_query}\n",
                }
            query_for_search = resolved_query or query_for_search
            if resolved_entity:
                params["_resolved_entity"] = resolved_entity
            if resolved_attribute:
                params["_resolved_attribute"] = resolved_attribute
            if str((params or {}).get("_expert_route") or "").strip() == "personal_attribute":
                logger.info(
                    "[dispatch] personal_attribute resolve: original=%r resolved_query=%r entity=%r attribute=%r",
                    q,
                    query_for_search,
                    params.get("_resolved_entity", ""),
                    params.get("_resolved_attribute", ""),
                )
    explicit_file_ref = (params or {}).get("_explicit_file_ref") if isinstance(params, dict) else None
    if not isinstance(explicit_file_ref, dict):
        explicit_file_ref = None
    if self._looks_like_file_content_analysis_query(q, prompt_language=user_lang) or \
       self._looks_like_scoped_file_search_query(q, prompt_language=user_lang):
        focused_query = self._extract_file_analysis_focus_query(q, prompt_language=user_lang) or ""
        if focused_query:
            query_for_search = focused_query
    scope_choice_only = bool(
        scope_disambiguation_reason in {"previous_choice", "selected_choice"}
        and str((params or {}).get("query") or "").strip()
        and str((params or {}).get("query") or "").strip() != q.strip()
    )
    search_need_text = query_for_search if scope_choice_only else q
    lexical_query_text = _build_lexical_query_text(search_need_text, query_for_search)
    if intent_normalized_en:
        retrieval_query = query_for_search
        if any("\u4e00" <= ch <= "\u9fff" for ch in retrieval_query):
            retrieval_query = self._augment_query_for_retrieval(
                query_for_search,
                prompt_language=user_lang,
                session_id=session_id,
            )
    else:
        retrieval_query = self._augment_query_for_retrieval(
            query_for_search,
            prompt_language=user_lang,
            session_id=session_id,
        )
    retrieval_query = self._blend_retrieval_query_with_original_cjk(retrieval_query, search_need_text)
    retrieval_query = self._anchor_retrieval_query_with_last_search(retrieval_query, search_need_text, session_id)
    category = str(params.get("category") or "").strip()
    if category and self._normalize_category_name(category) == "document":
        invoice_type_query = bool(
            re.search(
                r"\b(?:invoice|invoices|receipt|receipts|bill|bills)\b|发票|收据|账单",
                " ".join([str(search_need_text or ""), str(query_for_search or ""), str(retrieval_query or "")]),
                re.IGNORECASE,
            )
        )
        inventory_or_file_type_context = bool(
            re.search(
                r"\b(?:find|search|show|list|display|all|which|what|have|files?|documents?|docs?|pdfs?)\b"
                r"|找|搜索|查找|显示|列出|全部|所有|有哪些|文件|文档|pdf|PDF",
                str(search_need_text or q or ""),
                re.IGNORECASE,
            )
        )
        if invoice_type_query and inventory_or_file_type_context:
            category = "invoice"
            params["category"] = "invoice"
            yield {
                "type": "thinking",
                "delta": "Detected invoice file-type intent inside generic document category.\n",
            }
    _cat_norm_chk = self._normalize_category_name(category) if category else ""
    if category and self._paper_category_likely_wrong_for_query(search_need_text, _cat_norm_chk):
        yield {
            "type": "thinking",
            "delta": "Ignoring category=paper: query looks like school/math problems, not academic papers.\n",
        }
        category = ""
    if category:
        _cat_norm_report = self._normalize_category_name(category)
        if self._report_category_likely_wrong_for_query(search_need_text, _cat_norm_report):
            yield {
                "type": "thinking",
                "delta": "Ignoring category=report: query looks like homework/schoolwork, not business reports.\n",
            }
            category = ""
    if category:
        _cat_norm_anchor = self._normalize_category_name(category)
        anchor_like_query = bool(
            _extract_identifier_filename_anchors(search_need_text)
            or is_lookup_heavy_query(search_need_text)
        )
        explicit_file_type_category = False
        if _cat_norm_anchor in {"manual", "document", "report", "paper"}:
            try:
                from core.intent.entity_experts import CategoryListExpert

                cat_intent = CategoryListExpert.analyze(search_need_text, has_content_qualifier=True)
                cat_params = dict((cat_intent or {}).get("params") or {})
                explicit_file_type_category = (
                    str((cat_intent or {}).get("action") or "").strip() == "search"
                    and self._normalize_category_name(str(cat_params.get("category") or "")) == _cat_norm_anchor
                )
            except Exception:
                explicit_file_type_category = False
        if (
            _cat_norm_anchor in {"manual", "document", "report", "paper"}
            and anchor_like_query
            and not explicit_file_type_category
        ):
            yield {
                "type": "thinking",
                "delta": (
                    f"Ignoring category={_cat_norm_anchor}: strong lookup/title anchor should not be over-narrowed "
                    "to a generic document bucket.\n"
                ),
            }
            category = ""
    keywords = str(params.get("keywords") or "").strip()
    folder = str(params.get("folder") or "").strip()
    folder_retrieval_hint = folder
    if folder and any("\u4e00" <= ch <= "\u9fff" for ch in folder):
        folder_en = self._augment_query_for_retrieval(folder, prompt_language=user_lang, session_id=session_id)
        if folder_en and folder_en != folder:
            yield {"type": "thinking", "delta": f"Translated folder filter '{folder}' -> '{folder_en}'\n"}
            folder_retrieval_hint = _build_lexical_query_text(folder, folder_en)
    if folder_retrieval_hint and str(query_for_search or "").strip() == str(folder or "").strip():
        retrieval_query = _build_lexical_query_text(retrieval_query, folder_retrieval_hint)

    folder_filter: Optional[str] = folder if folder else None
    if folder_filter and active_paths:
        f = folder_filter.strip().lower()
        if f and "/" not in f and "\\" not in f:
            try:
                has_folder_match = any(
                    (f in os.path.basename(str(p or "")).lower()) or (f in str(p or "").lower())
                    for p in (active_paths or [])
                )
            except Exception:
                has_folder_match = False
            if not has_folder_match:
                normalized_folder = str(self._normalize_category_name(folder_filter) or "").strip().lower()
                generic_bucket_like = normalized_folder not in {"", "all", "other", "unknown"}
                if generic_bucket_like:
                    yield {"type": "thinking", "delta": f"Ignoring folder filter (no path matched): {folder_filter}\n"}
                    folder_filter = None
    document_retrieval_media_topic = False
    try:
        from core.intent.media_query_expert import MediaQueryExpert

        document_retrieval_media_topic = MediaQueryExpert._looks_like_document_retrieval_with_media_topic(
            _build_lexical_query_text(search_need_text, query_for_search, retrieval_query, q)
        )
    except Exception:
        document_retrieval_media_topic = False
    if category and self._normalize_category_name(category) in {"audio", "video", "audio/video", "image"} and document_retrieval_media_topic:
        yield {
            "type": "thinking",
            "delta": "Treating media/visual term as a document topic, not a file category filter.\n",
        }
        category = ""
        params.pop("category", None)
        params.pop("media_type", None)
    effective_category = ""
    if category:
        c0 = self._normalize_category_name(category)
        if c0 not in {"all", "unknown"}:
            effective_category = c0
    if not effective_category:
        inferred = self._normalize_category_name(query_for_search)
        token_count = len([t for t in query_for_search.replace("/", " ").split() if t.strip()])
        query_short = len(query_for_search.strip()) <= 40 and token_count <= 8
        if inferred == "manual":
            specific_terms = [
                t for t in extract_lookup_terms(query_for_search, max_terms=16)
                if len(str(t or "").strip()) >= 3
            ]
            if len(specific_terms) >= 3:
                inferred = ""
        if query_short and inferred not in {"", "all", "other", "unknown", "document"}:
            if inferred in {"audio", "video", "audio/video", "image"} and document_retrieval_media_topic:
                inferred = ""
            try:
                cat_counts = self._collect_category_counts()
                if int(cat_counts.get(inferred, 0)) > 0:
                    effective_category = inferred
                    yield {"type": "thinking", "delta": f"Detected category intent: {effective_category}\n"}
            except Exception:
                pass

    params_category_inventory_mode = (
        str((params or {}).get("_inventory_mode") or "").strip().lower() == "category"
    )
    folder_category_listing_query = bool(
        effective_category
        and getattr(self, "_looks_like_folder_listing_query", None)
        and self._looks_like_folder_listing_query(search_need_text)
    )
    generic_inventory_query = bool(
        getattr(self, "_looks_like_generic_inventory_query", None)
        and self._looks_like_generic_inventory_query(search_need_text)
    )
    category_inventory_query = bool(
        effective_category
        and not explicit_file_ref
        and not self._looks_like_file_content_analysis_query(search_need_text, prompt_language=user_lang)
        and (
            generic_inventory_query
            or params_category_inventory_mode
            or folder_category_listing_query
        )
    )

    explicit_selected_scope = bool(
        scope_disambiguation_reason in {"explicit_selected_scope", "selected_choice"}
        or str((params or {}).get("_scope") or "").strip().lower() in {"selected", "selected_items", "selected_folder"}
        or str((params or {}).get("scope") or "").strip().lower() in {"selected", "selected_items", "selected_folder"}
        or str((params or {}).get("_scope_kind") or "").strip().lower() in {"selected", "selected_items", "selected_folder"}
        or str((params or {}).get("_preserve_selected_scope") or "").strip().lower() in {"1", "true", "yes"}
    )
    if (
        active_paths
        and effective_category
        and not folder_filter
        and not explicit_selected_scope
        and _active_scope_lacks_category_files(active_paths, effective_category)
    ):
        yield {
            "type": "thinking",
            "delta": (
                f"Current active source scope has no {effective_category}-compatible files; "
                "broadening to all indexed files.\n"
            ),
        }
        params["_active_scope_relaxed"] = "category_incompatible"
        active_paths = None

    yield {"type": "thinking", "delta": f"Mode: search, query: {query_for_search}\n"}
    if folder_filter:
        yield {"type": "thinking", "delta": f"Folder filter: {folder_filter}\n"}

    yield from _emit_status("running", "Retrieving...")
    kb = get_kb_instance()
    filename_route_results: List[Dict[str, Any]] = []

    # ── explicit_category must be computed before the direct-filename short-circuit
    # guard at line ~2640, which references it. (Moved from the Lexical Sub-Agent
    # section below to avoid UnboundLocalError when direct_candidates exist.)
    explicit_category = ""
    if category:
        _c_exp = self._normalize_category_name(category)
        if _c_exp not in {"", "all", "unknown"}:
            explicit_category = _c_exp

    # Explicit inventory requests should use the indexed metadata sidecar
    # directly.  Sending "all indexed files in folder / filename contains X"
    # through semantic rerank + LLM refine drops valid siblings by design.
    inventory_text = str(search_need_text or q or "").strip()
    inventory_folder = ""
    inventory_contains = ""
    inventory_exact_name = ""
    inventory_extensions: List[str] = []
    inventory_media_kind = ""
    negative_inventory_category = _extract_negative_category_inventory(inventory_text)
    m_folder_inventory = re.search(
        r'\bfind\s+all\s+indexed\s+files\s+in\s+(?:the\s+)?(?:folder|directory)\s+["“](.+?)["”]\s*$',
        inventory_text,
        re.IGNORECASE,
    )
    if m_folder_inventory:
        inventory_folder = str(m_folder_inventory.group(1) or "").strip()
    else:
        m_contains_inventory = re.search(
            r'\bfind\s+all\s+indexed\s+files\s+whose\s+file\s*name\s+contains\s+["“]?(.+?)["”]?\s*$'
            r'|\bfind\s+all\s+indexed\s+files\s+whose\s+filename\s+contains\s+["“]?(.+?)["”]?\s*$',
            inventory_text,
            re.IGNORECASE,
        )
        if m_contains_inventory:
            inventory_contains = str(m_contains_inventory.group(1) or m_contains_inventory.group(2) or "").strip()
        else:
            m_exact_inventory = re.search(
                r'\bfind\s+all\s+indexed\s+files\s+named\s+["“](.+?)["”]\s*$',
                inventory_text,
                re.IGNORECASE,
            )
            if m_exact_inventory:
                inventory_exact_name = str(m_exact_inventory.group(1) or "").strip()
            else:
                m_media_inventory = re.search(
                    r'\bfind\s+all\s+indexed\s+'
                    r'(?P<kind>videos?|movies?|clips?|audio|audios|recordings?|songs?|music)\s+files\s*$',
                    inventory_text,
                    re.IGNORECASE,
                )
                if m_media_inventory:
                    inventory_media_kind = str(m_media_inventory.group("kind") or "").strip().lower()
                    if re.search(r"\b(video|videos|movie|movies|clip|clips)\b", inventory_media_kind):
                        inventory_extensions = [".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv", ".wmv", ".ts"]
                    else:
                        inventory_extensions = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape"]
                m_ext_inventory = re.search(
                    r'\bfind\s+all\s+indexed\s+\.([a-z0-9]{1,12})\s+files\s*$',
                    inventory_text,
                    re.IGNORECASE,
                )
                if m_ext_inventory and not inventory_extensions:
                    inventory_extensions = [f".{str(m_ext_inventory.group(1) or '').strip().lower()}"]

    if inventory_folder or inventory_contains or inventory_exact_name or inventory_extensions or negative_inventory_category:
        inventory_sources: List[Dict[str, Any]] = []
        inventory_label = ""
        try:
            if negative_inventory_category:
                inventory_label = f"files excluding {negative_inventory_category}"
                pack = kb.indexed_file_inventory(
                    allowed_paths=active_paths,
                    limit=0,
                    hydrate=True,
                    include_documents=False,
                )
                excluded_exts = set(_CATEGORY_COMPATIBLE_EXTS.get(negative_inventory_category, set()) or set())
                for item in list(pack.get("files") or []):
                    fp = str(item.get("file_path") or (item.get("metadata") or {}).get("file_path") or "").strip()
                    if not fp:
                        continue
                    ext = os.path.splitext(fp)[1].lower()
                    normalized_cat = self._normalize_category_name(
                        str(item.get("doc_category") or (item.get("metadata") or {}).get("doc_category") or "other")
                    )
                    if normalized_cat == negative_inventory_category:
                        continue
                    if excluded_exts and ext in excluded_exts:
                        continue
                    inventory_sources.append(item)
            elif inventory_folder:
                inventory_label = f'folder "{inventory_folder}"'
                pack = kb.indexed_file_inventory(
                    allowed_paths=[inventory_folder],
                    limit=0,
                    hydrate=True,
                    include_documents=False,
                )
                inventory_sources = list(pack.get("files") or [])
            elif inventory_extensions:
                if inventory_media_kind:
                    inventory_label = "video files" if inventory_extensions[0] in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv", ".wmv", ".ts"} else "audio files"
                else:
                    inventory_label = "extension " + ", ".join(inventory_extensions)
                pack = kb.indexed_file_inventory(
                    allowed_paths=active_paths,
                    file_extensions=inventory_extensions,
                    limit=0,
                    hydrate=True,
                    include_documents=False,
                )
                inventory_sources = list(pack.get("files") or [])
            else:
                needle = (inventory_contains or inventory_exact_name).strip().strip("\"'“”")
                inventory_label = (
                    f'filename contains "{needle}"'
                    if inventory_contains
                    else f'filename named "{needle}"'
                )
                # Exact inventory/listing queries must be exhaustive.  BM25 is
                # useful for ranked recall, but can drop valid sibling files for
                # "filename contains X" because only positive-scoring records
                # are returned.  Use the indexed file sidecar as an inventory
                # table here, then apply the deterministic filename predicate.
                pack = kb.indexed_file_inventory(
                    allowed_paths=active_paths,
                    limit=0,
                    hydrate=True,
                    include_documents=False,
                )
                needle_l = needle.lower()
                for item in list(pack.get("files") or []):
                    file_name = os.path.basename(str(item.get("file_name") or item.get("file_path") or ""))
                    file_name_l = file_name.lower()
                    if inventory_contains and needle_l in file_name_l:
                        inventory_sources.append(item)
                    elif inventory_exact_name and file_name_l == needle_l:
                        inventory_sources.append(item)
        except Exception as inventory_exc:
            logger.warning("[dispatch] exact indexed inventory fast-path failed: %s", inventory_exc)
            inventory_sources = []

        if inventory_sources:
            seen_inventory_paths: set[str] = set()
            deduped_inventory_sources: List[Dict[str, Any]] = []
            for item in sorted(
                inventory_sources,
                key=lambda src: (
                    str(src.get("file_name") or "").lower(),
                    str(src.get("file_path") or "").lower(),
                ),
            ):
                fp = str(item.get("file_path") or "").strip()
                if not fp or fp in seen_inventory_paths:
                    continue
                seen_inventory_paths.add(fp)
                deduped_inventory_sources.append(item)
            inventory_sources = deduped_inventory_sources

            self._clear_count_scope_context(session_id, reason="search_results_updated")
            self._set_last_search_results(session_id, inventory_sources[:50])
            self._set_followup_hint(
                session_id,
                action="process_previous",
                params={},
                ttl_turns=2,
                uses=2,
            )
            yield {
                "type": "trace_append",
                "item": {
                    "stage": "exact_indexed_inventory",
                    "type": "retrieval",
                    "mode": (
                        "negative_category"
                        if negative_inventory_category
                        else "folder"
                        if inventory_folder
                        else "extension"
                        if inventory_extensions
                        else "filename_contains"
                        if inventory_contains
                        else "filename_exact"
                    ),
                    "sources_count": len(inventory_sources),
                },
            }
            yield from _emit_files_from_sources(inventory_sources)

            if user_lang == "zh":
                inventory_answer = f"已找到 {len(inventory_sources)} 个匹配 {inventory_label} 的已索引文件。"
            else:
                inventory_answer = f"Found {len(inventory_sources)} indexed file(s) matching {inventory_label}."
            _first_text_trace = _mark_first_text("exact_indexed_inventory", chars=len(inventory_answer))
            if _first_text_trace:
                yield _first_text_trace
            yield {"type": "text", "content": inventory_answer}
            try:
                hist_ref[-1]["a"] = inventory_answer
            except Exception:
                pass
            yield _emit_timing_trace(
                "request_done",
                action=action,
                query_type="search",
                sources_count=len(inventory_sources),
            )
            yield {
                "type": "done",
                "ok": True,
                "query_type": "search",
                "sources": inventory_sources,
                "trace": [],
            }
            return

    if explicit_file_ref:
        yield {"type": "thinking", "delta": f"Direct filename lookup: {explicit_file_ref.get('raw_name','')}\n"}
        direct_candidates: List[Dict[str, Any]] = []
        direct_folder_candidates: List[Dict[str, Any]] = []
        direct_folder_seed_sources: List[Dict[str, Any]] = []
        search_terms = []
        raw_name = str(explicit_file_ref.get("raw_name") or "").strip()
        search_term = str(explicit_file_ref.get("search_term") or "").strip()
        explicit_all_names = [
            str(item or "").strip()
            for item in (explicit_file_ref.get("all_names") or [])
            if str(item or "").strip()
        ] if isinstance(explicit_file_ref.get("all_names"), list) else []
        explicit_match_mode_meta = classify_explicit_filename_match_mode(q, explicit_file_ref)
        explicit_match_mode = str(explicit_match_mode_meta.get("mode") or "broad").strip().lower()
        explicit_match_stem_key = compact_filename_key(str(explicit_match_mode_meta.get("stem_key") or ""))
        strict_exact_filename_lookup = explicit_match_mode == "exact_filename"
        strict_exact_stem_lookup = explicit_match_mode == "exact_stem"
        strict_direct_filename_lookup = strict_exact_filename_lookup or strict_exact_stem_lookup
        direct_scope_matcher = ensure_path_scope_matcher(active_paths)
        direct_lookup_expert_route = str((params or {}).get("_expert_route") or "").strip().lower()
        ql_direct_lookup = str(q or "").strip().lower()
        direct_lookup_content_query = bool(
            direct_lookup_expert_route == "semantic_file_op"
            or self._looks_like_file_content_analysis_query(q, prompt_language=user_lang)
            or re.search(
                r"\b(tell\s+me\s+about|about|summarize|summary|describe|description|"
                r"explain|details?|content|contents|inside|what(?:'s|\s+is).{0,24}\babout)\b",
                ql_direct_lookup,
                re.IGNORECASE,
            )
            or any(tok in q for tok in ("关于", "总结", "概括", "内容", "讲了什么", "说了什么", "介绍", "说明", "详细"))
        )
        basename_query = raw_name if has_plausible_filename_extension(raw_name) else (search_term or raw_name)
        basename_query_key = ""
        full_query_name_key = ""
        full_query_name_keys: set[str] = set()
        basename_query_keys: set[str] = set()
        explicit_query_ext = ""
        for explicit_name in explicit_all_names:
            if explicit_name and explicit_name not in search_terms:
                search_terms.append(explicit_name)
        if raw_name and raw_name not in search_terms:
            search_terms.append(raw_name)
        if search_term and search_term not in search_terms:
            search_terms.append(search_term)
        if basename_query:
            basename_name = os.path.basename(basename_query)
            full_query_name_key = compact_filename_key(basename_name)
            if full_query_name_key:
                full_query_name_keys.add(full_query_name_key)
            explicit_query_ext = os.path.splitext(basename_name)[1].lower()
            if not explicit_query_ext:
                basename_query_key = compact_filename_key(os.path.splitext(basename_name)[0])
                if basename_query_key:
                    basename_query_keys.add(basename_query_key)
        for explicit_name in explicit_all_names:
            explicit_base = os.path.basename(explicit_name)
            explicit_key = compact_filename_key(explicit_base)
            if explicit_key:
                full_query_name_keys.add(explicit_key)
            explicit_stem_key = compact_filename_key(os.path.splitext(explicit_base)[0])
            if explicit_stem_key:
                basename_query_keys.add(explicit_stem_key)
        if strict_exact_stem_lookup and explicit_match_stem_key:
            basename_query_keys.add(explicit_match_stem_key)
            if not basename_query_key:
                basename_query_key = explicit_match_stem_key

        def _compact_name(text: str) -> str:
            return re.sub(r"[\s\\/_\-.]+", "", str(text or "").strip().lower())

        def _score_folder_hit(folder_row: Dict[str, Any]) -> int:
            folder_name = str(folder_row.get("file_name") or folder_row.get("folder_path") or "").strip()
            folder_key = compact_filename_key(folder_name)
            if basename_query_key and folder_key == basename_query_key:
                return 100
            if full_query_name_key and folder_key == full_query_name_key:
                return 100

            best_score = 0
            for term in search_terms:
                _, surface_score = score_filename_surface_match(term, folder_name, "")
                best_score = max(best_score, surface_score)
            return best_score

        def _direct_category_bonus(meta: Dict[str, Any]) -> int:
            if not effective_category:
                return 0
            fp = str(meta.get("file_path") or "")
            ext = os.path.splitext(fp)[1].lower()
            cat = self._normalize_category_name(str(meta.get("doc_category") or "other"))
            compatible = {
                "image": cat == "image" or ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"},
                "data": cat == "data" or ext in {".csv", ".tsv", ".xlsx", ".xls", ".numbers"},
                "audio": cat == "audio" or ext in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"},
                "video": cat == "video" or ext in {".mp4", ".mov", ".mkv", ".avi", ".webm"},
                "document": cat in {"document", "resume", "report", "contract", "note", "manual", "book", "paper"} or ext in {".md", ".txt", ".html", ".pdf", ".doc", ".docx", ".ppt", ".pptx"},
            }
            return 8 if compatible.get(effective_category, False) else 0

        merged_hits: Dict[str, Dict[str, Any]] = {}

        if basename_query and not explicit_query_ext and not strict_exact_stem_lookup and not direct_lookup_content_query:
            try:
                folder_rows = kb.collect_folder_index_candidates(
                    basename_query,
                    original_query=search_need_text,
                    allowed_paths=active_paths,
                )
                exact_folder_rows: List[Dict[str, Any]] = []
                for row in folder_rows:
                    score = _score_folder_hit(row)
                    if score < 90:
                        continue
                    folder_path = str(row.get("folder_path") or row.get("file_path") or "").strip()
                    folder_name = str(row.get("file_name") or os.path.basename(folder_path) or "").strip()
                    if not folder_path or not folder_name:
                        continue
                    exact_folder_rows.append(
                        {
                            "file_name": folder_name,
                            "file_path": folder_path,
                            "doc_summary": "",
                            "doc_category": "folder",
                            "type": "folder",
                            "iconType": "folder",
                            "is_matched_folder": True,
                            "_direct_score": score,
                            "_is_folder_hit": True,
                        }
                    )
                if exact_folder_rows:
                    exact_folder_rows.sort(
                        key=lambda item: (
                            -int(item.get("_direct_score", 0) or 0),
                            str(item.get("file_path") or "").count(os.sep),
                            str(item.get("file_name") or "").lower(),
                        )
                    )
                    direct_folder_candidates = exact_folder_rows[:6]
                    yield {
                        "type": "thinking",
                        "delta": f"Exact folder match: {len(direct_folder_candidates)}\n",
                    }
            except Exception as exc:
                logger.warning("[direct_filename_lookup] exact folder candidate scan failed: %s", exc)

        def _record_matches_explicit_mode(record: Dict[str, Any]) -> bool:
            if not strict_direct_filename_lookup:
                return True
            file_name = str(record.get("file_name") or record.get("file_path") or "").strip()
            if not file_name:
                return False
            file_base = os.path.basename(file_name)
            file_name_key = compact_filename_key(file_base)
            file_stem_key = compact_filename_key(os.path.splitext(file_base)[0])
            if strict_exact_filename_lookup:
                return bool(file_name_key and file_name_key in full_query_name_keys)
            return bool(
                file_stem_key
                and any(filename_stem_key_matches_query(file_stem_key, query_key) for query_key in basename_query_keys)
            )

        variant_terms: List[str] = []
        for base in search_terms:
            candidate = str(base or "").strip()
            if not candidate:
                continue
            variant_terms.append(candidate)
            lowered = candidate.lower()
            spaced = re.sub(r"[_\-]+", " ", lowered).strip()
            underscored = re.sub(r"\s+", "_", lowered).strip()
            hyphenated = re.sub(r"\s+", "-", lowered).strip()
            compact = _compact_name(candidate)
            for alt in (spaced, underscored, hyphenated, compact):
                if alt and alt not in variant_terms:
                    variant_terms.append(alt)

        # Direct filename lookup should stay database-only.
        # File/folder names are already indexed into Chroma metadata / folder index
        # during ingestion, so avoid redundant tool indirection or filesystem scans
        # here; they only add latency and can stall the product on large scopes.

        _exact_index_lookup_current = False
        exact_lookup_names = explicit_all_names or ([basename_query] if basename_query else [])
        if strict_direct_filename_lookup and exact_lookup_names and hasattr(kb, "indexed_exact_filename_lookup"):
            try:
                exact_added = 0
                for lookup_name in exact_lookup_names:
                    lookup_base = os.path.basename(str(lookup_name or "").strip())
                    if not lookup_base:
                        continue
                    lookup_ext = os.path.splitext(lookup_base)[1].lower()
                    extension_filter = [lookup_ext] if strict_exact_filename_lookup and lookup_ext else None
                    exact_pack = kb.indexed_exact_filename_lookup(
                        lookup_base,
                        allowed_paths=direct_scope_matcher,
                        file_extensions=extension_filter,
                        match_mode="exact_filename" if strict_exact_filename_lookup else "exact_stem",
                        limit=12,
                        hydrate=True,
                        include_documents=True,
                    )
                    _exact_index_lookup_current = (
                        _exact_index_lookup_current
                        or (bool((exact_pack or {}).get("ready")) and not bool((exact_pack or {}).get("stale")))
                    )
                    exact_rows = list((exact_pack or {}).get("files") or [])
                    for row in exact_rows:
                        if not _record_matches_explicit_mode(row):
                            continue
                        meta = dict(row.get("metadata") or {})
                        fp = str(row.get("file_path") or meta.get("file_path") or "").strip()
                        if not fp:
                            continue
                        fname = str(row.get("file_name") or meta.get("file_name") or os.path.basename(fp))
                        if not meta:
                            meta = {
                                "file_name": fname,
                                "file_path": fp,
                                "doc_summary": row.get("doc_summary", ""),
                                "doc_category": row.get("doc_category", "other"),
                            }
                        aliases = str(row.get("lookup_aliases") or meta.get("lookup_aliases") or "")
                        scored = {
                            "text": str(row.get("text") or row.get("doc_summary") or meta.get("doc_summary") or ""),
                            "metadata": meta,
                            "file_name": fname,
                            "file_path": fp,
                            "doc_summary": meta.get("doc_summary", row.get("doc_summary", "")),
                            "doc_category": meta.get("doc_category", row.get("doc_category", "other")),
                            "lookup_aliases": aliases,
                            "_direct_score": 120 + _direct_category_bonus(meta),
                            "_bm25_score": 0.0,
                            "_lexical_filename_exact": True,
                            "_keyword_index_exact_name": True,
                            "hit_chunks": int(row.get("hit_chunks", 1) or 1),
                        }
                        prev = merged_hits.get(fp)
                        if (prev is None) or (int(scored.get("_direct_score", 0)) > int(prev.get("_direct_score", 0))):
                            merged_hits[fp] = scored
                            exact_added += 1
                if exact_added:
                    yield {"type": "thinking", "delta": f"Exact indexed filename match: {exact_added}\n"}
            except Exception as exc:
                logger.warning("[direct_filename_lookup] exact indexed filename lookup failed: %s", exc)

        keyword_index_ready = bool(getattr(kb, "is_keyword_index_ready", lambda: False)())
        # Track whether BM25 actually returned ANY documents (vs returning zero results).
        # If BM25 returns zero, the file might simply not be in the keyword index yet
        # (e.g. index incomplete, version mismatch) — we should NOT exit early in that case.
        _indexed_search_returned_any = False
        try:
            indexed_lookup_query = " ".join(dict.fromkeys(variant_terms or search_terms)).strip()
            indexed_extension_filter = [explicit_query_ext] if strict_exact_filename_lookup and explicit_query_ext else None
            indexed_hits = kb.indexed_keyword_search(
                indexed_lookup_query,
                allowed_paths=direct_scope_matcher,
                category_filter="",
                file_extensions=indexed_extension_filter,
                limit=50,
            ) if indexed_lookup_query and hasattr(kb, "indexed_keyword_search") else []
            _indexed_search_returned_any = bool(indexed_hits)
            for hit in indexed_hits:
                meta = dict(hit.get("metadata") or {})
                fp = str(hit.get("file_path") or meta.get("file_path") or "")
                if not fp:
                    continue
                fname = str(hit.get("file_name") or meta.get("file_name") or os.path.basename(fp))
                fstem = os.path.splitext(fname)[0]
                aliases = str(hit.get("lookup_aliases") or meta.get("lookup_aliases") or "")
                blob = " ".join([fname, fstem, fp, aliases])
                best_score = 0
                for term in variant_terms:
                    exact_match, surface_score = score_filename_surface_match(term, fname, aliases)
                    if exact_match:
                        best_score = max(best_score, 100)
                        continue
                    if surface_score > 0:
                        best_score = max(best_score, surface_score)
                        continue
                    exact, overlap = lookup_match_quality(term, blob)
                    if exact:
                        best_score = max(best_score, 96)
                        continue
                    if overlap > 0:
                        best_score = max(best_score, min(80, int(overlap * 12)))

                if best_score <= 0:
                    continue
                best_score += _direct_category_bonus(meta)
                prev = merged_hits.get(fp)
                scored = {
                    "text": str(hit.get("text") or meta.get("doc_summary") or ""),
                    "metadata": meta,
                    "file_name": fname,
                    "file_path": fp,
                    "doc_summary": meta.get("doc_summary", hit.get("doc_summary", "")),
                    "doc_category": meta.get("doc_category", hit.get("doc_category", "other")),
                    "lookup_aliases": aliases,
                    "_direct_score": best_score,
                    "_bm25_score": float(hit.get("_bm25_score", 0.0) or 0.0),
                    "hit_chunks": 1,
                }
                if (prev is None) or (int(scored.get("_direct_score", 0)) > int(prev.get("_direct_score", 0))):
                    merged_hits[fp] = scored
        except Exception as exc:
            logger.warning("[direct_filename_lookup] indexed lookup failed: %s", exc)

        ranked_hits = sorted(
            merged_hits.values(),
            key=lambda x: (int(x.get("_direct_score", 0)), int(x.get("hit_chunks", 0) or 0)),
            reverse=True,
        )
        if strict_direct_filename_lookup:
            ranked_hits = [hit for hit in ranked_hits if _record_matches_explicit_mode(hit)]

        for hit in ranked_hits[:5]:
            fp = str(hit.get("file_path") or "")
            doc = str(hit.get("text") or hit.get("doc_summary") or "")
            meta = dict(hit.get("metadata") or {})
            if not meta:
                meta = {
                    "file_name": hit.get("file_name", ""),
                    "file_path": fp,
                    "doc_summary": hit.get("doc_summary", ""),
                    "doc_category": hit.get("doc_category", "other"),
                }
            direct_candidates.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": 0.0,
                    "file_name": meta.get("file_name", hit.get("file_name", "")),
                    "file_path": meta.get("file_path", fp),
                    "doc_summary": meta.get("doc_summary", hit.get("doc_summary", "")),
                    "doc_category": self._normalize_category_name(meta.get("doc_category", hit.get("doc_category", "other"))),
                    "score": max(0.90, float(settings.RELEVANCE_THRESHOLD) + 0.2),
                    "rerank_score": max(0.90, float(settings.RELEVANCE_THRESHOLD) + 0.2),
                    "_direct_score": int(hit.get("_direct_score", 0)),
                    "_is_lexical_hit": True,
                }
            )

        # Only exit early when a current filename index scan, or a positive BM25
        # lookup over the filename index, found no exact match. If the index did
        # not answer confidently, fall through to semantic retrieval.
        hard_filename_miss = bool(
            strict_exact_filename_lookup
            and explicit_query_ext
            and not direct_lookup_content_query
        )
        if (
            strict_direct_filename_lookup
            and hard_filename_miss
            and keyword_index_ready
            and (_exact_index_lookup_current or _indexed_search_returned_any)
            and not direct_candidates
            and not direct_folder_candidates
        ):
            focus_name = str(
                explicit_match_mode_meta.get("target")
                or explicit_file_ref.get("raw_name")
                or explicit_file_ref.get("search_term")
                or ""
            ).strip()
            if user_lang == "zh":
                final_answer = (
                    f"没有找到名称精确匹配“{focus_name}”的文件。"
                    if focus_name else
                    "没有找到名称精确匹配的文件。"
                )
            else:
                final_answer = (
                    f"No files with the exact requested name were found for \"{focus_name}\"."
                    if focus_name else
                    "No files with the exact requested name were found."
                )
            _first_text_trace = _mark_first_text("filename_lookup_direct_miss", chars=len(final_answer))
            if _first_text_trace:
                yield _first_text_trace
            yield {"type": "text", "content": final_answer}
            try:
                hist_ref[-1]["a"] = final_answer
            except Exception:
                pass
            yield {"type": "done", "ok": True, "query_type": "search", "sources": [], "trace": []}
            return
        if direct_folder_candidates:
            folder_only_sources = [{k: v for k, v in item.items() if k != "_direct_score"} for item in direct_folder_candidates]
            expanded_sources: List[Dict[str, Any]] = []
            try:
                direct_folder_paths = [
                    str(item.get("file_path") or "").strip()
                    for item in folder_only_sources
                    if str(item.get("file_path") or "").strip()
                ]
                if direct_folder_paths:
                    try:
                        _direct_folder_expand = int(os.getenv("FOLDER_CHAIN_EXPAND_PER_FOLDER", "0") or 0)
                    except ValueError:
                        _direct_folder_expand = 0
                    yield {
                        "type": "thinking",
                        "delta": "Expanding directly matched folders into indexed files...\n",
                    }
                    expanded_sources = kb.expand_folder_paths_to_chain_sources(
                        direct_folder_paths,
                        max_per_folder=_direct_folder_expand,
                        allow_skip_basename_roots=True,
                        relax_ignore_rules=True,
                    )
            except Exception as exc:
                logger.warning("[direct_folder_lookup] folder expansion failed: %s", exc)
                expanded_sources = []

            if folder_only_sources or expanded_sources:
                direct_folder_seed_sources = list(folder_only_sources) + [
                    item for item in expanded_sources
                    if str(item.get("file_path") or "").strip()
                    not in {
                        str(folder_item.get("file_path") or "").strip()
                        for folder_item in folder_only_sources
                        if str(folder_item.get("file_path") or "").strip()
                    }
                ]

            matched_count = len(expanded_sources)
            matched_folder_count = len(folder_only_sources)
            focus_name = str(explicit_file_ref.get("raw_name") or explicit_file_ref.get("search_term") or "").strip()
            if user_lang == "zh":
                if expanded_sources:
                    yield {
                        "type": "thinking",
                        "delta": (
                            f"已命中 {matched_folder_count} 个文件夹，并展开其中 {matched_count} 个已索引文件；继续结合语义召回补充相关发票文件...\n"
                            if focus_name else
                            f"已命中 {matched_folder_count} 个文件夹，并展开其中 {matched_count} 个已索引文件；继续结合语义召回补充相关文件...\n"
                        ),
                    }
                else:
                    yield {
                        "type": "thinking",
                        "delta": (
                            f"已命中 {matched_folder_count} 个文件夹；继续结合语义召回补充相关文件...\n"
                            if focus_name else
                            f"已命中 {matched_folder_count} 个文件夹；继续结合语义召回补充相关文件...\n"
                        ),
                    }
            else:
                if expanded_sources:
                    yield {
                        "type": "thinking",
                        "delta": (
                            f"Found {matched_folder_count} directly matched folder(s) and expanded {matched_count} indexed file(s); continuing semantic retrieval for related files...\n"
                            if focus_name else
                            f"Found {matched_folder_count} directly matched folder(s) and expanded {matched_count} indexed file(s); continuing semantic retrieval for related files...\n"
                        ),
                    }
                else:
                    yield {
                        "type": "thinking",
                        "delta": (
                            f"Found {matched_folder_count} directly matched folder(s); continuing semantic retrieval for related files...\n"
                            if focus_name else
                            f"Found {matched_folder_count} directly matched folder(s); continuing semantic retrieval for related files...\n"
                        ),
                    }

        if direct_candidates:
            direct_candidates.sort(
                key=lambda x: (int(x.get("_direct_score", 0)), float(x.get("rerank_score", 0.0) or 0.0)),
                reverse=True,
            )
            best_direct_score = int(direct_candidates[0].get("_direct_score", 0))
            direct_source_records = [dict(item) for item in direct_candidates]

            def _select_direct_sources(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                selected: List[Dict[str, Any]] = []
                seen_paths: set[str] = set()
                if not records:
                    return selected

                if strict_exact_filename_lookup:
                    selected = [
                        record for record in records
                        if _record_matches_explicit_mode(record)
                    ][:6]
                elif basename_query_key:
                    exact_stem_matches: List[Dict[str, Any]] = []
                    fuzzy_stem_matches: List[Dict[str, Any]] = []
                    best_surface_score = 0
                    surface_scored: List[tuple[int, Dict[str, Any]]] = []
                    query_surface = basename_query or raw_name or search_term
                    for record in records:
                        file_name = str(record.get("file_name") or record.get("file_path") or "")
                        file_stem_key = compact_filename_key(os.path.splitext(os.path.basename(file_name))[0])
                        if any(filename_stem_key_matches_query(file_stem_key, query_key) for query_key in basename_query_keys):
                            exact_stem_matches.append(record)
                            continue
                        _, surface_score = score_filename_surface_match(
                            query_surface,
                            file_name,
                            str(record.get("lookup_aliases") or ""),
                        )
                        if surface_score > 0:
                            best_surface_score = max(best_surface_score, surface_score)
                            surface_scored.append((surface_score, record))
                    if exact_stem_matches:
                        selected = exact_stem_matches
                    elif strict_exact_stem_lookup:
                        selected = []
                    elif surface_scored:
                        floor = max(78, best_surface_score - 8)
                        surface_scored.sort(
                            key=lambda item: (
                                item[0],
                                int(item[1].get("_direct_score", 0) or 0),
                                -(str(item[1].get("file_path") or "").count(os.sep)),
                            ),
                            reverse=True,
                        )
                        fuzzy_stem_matches = [record for score, record in surface_scored if score >= floor]
                        selected = fuzzy_stem_matches[:6]

                if not selected:
                    if strict_direct_filename_lookup:
                        selected = []
                    elif basename_query_key:
                        floor = max(70, best_direct_score - 10)
                        selected = [
                            record for record in records
                            if int(record.get("_direct_score", 0) or 0) >= floor
                        ][:6]
                    else:
                        selected = records[:1]

                ordered_sources = sorted(
                    selected,
                    key=lambda src: (
                        -int(src.get("_direct_score", 0) or 0),
                        str(src.get("file_path") or "").count(os.sep),
                        str(src.get("file_name") or "").lower(),
                    ),
                )
                extension_head: List[Dict[str, Any]] = []
                extension_tail: List[Dict[str, Any]] = []
                seen_exts: set[str] = set()
                for src in ordered_sources:
                    fp = str(src.get("file_path") or "")
                    if fp and fp in seen_paths:
                        continue
                    if fp:
                        seen_paths.add(fp)
                    file_name = str(src.get("file_name") or src.get("file_path") or "")
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext not in seen_exts:
                        extension_head.append(src)
                        seen_exts.add(ext)
                    else:
                        extension_tail.append(src)
                return extension_head + extension_tail

            _is_folder_listing_q = bool(
                getattr(self, "_looks_like_folder_listing_query", None)
                and self._looks_like_folder_listing_query(q)
            )
            if strict_direct_filename_lookup:
                should_short_circuit_direct = True
            elif folder_filter or explicit_category or _is_folder_listing_q:
                should_short_circuit_direct = False
            else:
                should_short_circuit_direct = best_direct_score >= 90
                if basename_query_key and len(direct_source_records) >= 2 and best_direct_score >= 78:
                    should_short_circuit_direct = True

            sources = [{k: v for k, v in item.items() if k != "_direct_score"} for item in direct_source_records]
            if should_short_circuit_direct:
                if strict_direct_filename_lookup or basename_query_key:
                    refined_sources = [
                        {k: v for k, v in item.items() if k != "_direct_score"}
                        for item in _select_direct_sources(direct_source_records)
                    ]
                else:
                    refined_sources = sources[:1]
                for src in refined_sources:
                    fp = str(src.get("file_path") or "")
                    brief = _fallback_brief_for_source(src)
                    if brief:
                        src["llm_refine_brief"] = brief

                sources = refined_sources
                self._clear_count_scope_context(session_id, reason="search_results_updated")
                self._set_last_search_results(session_id, sources[:50])
                self._set_followup_hint(
                    session_id,
                    action="process_previous",
                    params={},
                    ttl_turns=2,
                    uses=2,
                )

                yield from _emit_files_from_sources(sources)

                def _looks_like_direct_filename_content_request() -> bool:
                    if self._looks_like_file_content_analysis_query(q, prompt_language=user_lang):
                        return True
                    ql_direct = str(q or "").strip().lower()
                    has_content_signal = bool(
                        re.search(
                            r"\b(tell\s+me\s+about|about|summarize|summary|describe|description|"
                            r"explain|details?|content|contents|inside|what(?:'s|\s+is).{0,24}\babout)\b",
                            ql_direct,
                            re.IGNORECASE,
                        )
                        or any(tok in q for tok in ("关于", "总结", "概括", "内容", "讲了什么", "说了什么", "介绍", "说明", "详细"))
                    )
                    if not has_content_signal:
                        return False
                    focus_query = str(
                        self._extract_file_analysis_focus_query(q, prompt_language=user_lang) or ""
                    ).strip()
                    focus_name = str(
                        explicit_file_ref.get("search_term") or explicit_file_ref.get("raw_name") or ""
                    ).strip()
                    if not focus_query or not focus_name:
                        return False
                    focus_key = compact_filename_key(focus_query)
                    focus_name_key = compact_filename_key(focus_name)
                    return bool(
                        focus_key
                        and focus_name_key
                        and (
                            focus_key == focus_name_key
                            or focus_key in focus_name_key
                            or focus_name_key in focus_key
                        )
                    )

                plain_lookup_query = not _looks_like_direct_filename_content_request()
                if plain_lookup_query:
                    matched_count = len(sources)
                    focus_name = str(explicit_file_ref.get("raw_name") or explicit_file_ref.get("search_term") or "").strip()
                    if user_lang == "zh":
                        final_answer = (
                            f"已找到 {matched_count} 个和“{focus_name}”直接匹配的文件。"
                            if focus_name else
                            f"已找到 {matched_count} 个直接匹配的文件。"
                        )
                    else:
                        final_answer = (
                            f"Found {matched_count} directly matched files for \"{focus_name}\"."
                            if focus_name else
                            f"Found {matched_count} directly matched files."
                        )
                    _first_text_trace = _mark_first_text("filename_lookup_direct", chars=len(final_answer))
                    if _first_text_trace:
                        yield _first_text_trace
                    yield {"type": "text", "content": final_answer}
                else:
                    yield from _emit_status("thinking", "Generating answer...")
                    yield {"type": "thinking", "delta": "Resolved explicit file reference; generating answer from matched file...\n"}
                    ctx = self._build_context(session_id)
                    snippets = []
                    for i, d in enumerate(sources[:3], 1):
                        snippets.append(
                            f"[{i}] {d.get('file_name','')}\n"
                            f"Summary: {d.get('doc_summary','')}\n"
                            f"Content: {(d.get('text','') or '')[:500]}\n"
                        )
                    prompt_str = (
                        "You are a file retrieval assistant.\n"
                        f"- Reply in {response_language_label}.\n"
                        "- The user is asking about a directly matched file or a small set of filename-matched files.\n"
                        "- Answer strictly based on the provided indexed summary/content.\n"
                        "- If the file is an image and indexed visual summary exists, describe it directly from that summary.\n"
                        "- Do NOT say you cannot view or describe the file if usable indexed summary/content is available.\n"
                        "- If the indexed information is limited, say so briefly instead of inventing details.\n\n"
                        + f"{ctx}\n"
                        + "<Matched File Snippets>\n" + "\n---\n".join(snippets) + "\n</Matched File Snippets>\n\n"
                        + f"<User Question>\n{q}\n</User Question>\n"
                    )
                    llm = self._get_llm_service(detailed=True, session_id=session_id, prompt_language=user_lang)
                    final_answer = yield from _collect_or_emit_stream(llm, prompt_str)
                    if final_answer is None:
                        return
                try:
                    hist_ref[-1]["a"] = final_answer
                except Exception:
                    pass
                yield _emit_timing_trace("request_done", action=action, query_type="search", sources_count=len(sources))
                
                yield {"type": "done", "ok": True, "query_type": "search", "sources": sources, "trace": []}
                return
            else:
                filename_route_results = sources[:5]
                yield {"type": "thinking", "delta": "Precise filename hit not confident enough for short-circuit, or constraints exist; supplementing with semantic retrieval...\n"}
        else:
            yield {"type": "thinking", "delta": "No direct filename hit; trying semantic retrieval...\n"}
    else:
        direct_folder_seed_sources = []

    # ===== Lexical Sub-Agent Pipeline =====
    lexical_filenames = []
    lexical_extensions = []
    identifier_filename_anchors: List[str] = []
    explicit_filename_lookup = False
    extension_inventory_fast_path = False

    # explicit_category is already computed above (before the explicit_file_ref block).
    # No re-assignment needed here; variable is available from the earlier initialization.

    try:
        lexical = self._extract_lexical_features(lexical_query_text or search_need_text, session_id=session_id)
        lexical_filenames = lexical.get("filenames", [])
        lexical_extensions = lexical.get("extensions", [])
        param_extensions_raw = str((params or {}).get("file_extensions") or "").strip()
        if param_extensions_raw:
            for ext in re.split(r"[\s,;|]+", param_extensions_raw):
                cleaned_ext = str(ext or "").strip().lower()
                if not cleaned_ext:
                    continue
                normalized_ext = cleaned_ext if cleaned_ext.startswith(".") else f".{cleaned_ext}"
                if normalized_ext not in lexical_extensions:
                    lexical_extensions.append(normalized_ext)
        if lexical_extensions:
            try:
                from core.intent.entity_experts import CategoryListExpert

                negated_exts: set[str] = set()
                source_text_for_negation = search_need_text or q
                for cat_name, cat_exts in _CATEGORY_COMPATIBLE_EXTS.items():
                    aliases = CategoryListExpert._category_aliases(cat_name)
                    if any(CategoryListExpert._category_token_is_negated(source_text_for_negation, alias) for alias in aliases):
                        negated_exts.update(str(ext).lower() for ext in cat_exts)
                if negated_exts:
                    lexical_extensions = [
                        ext for ext in lexical_extensions
                        if (ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}") not in negated_exts
                    ]
            except Exception:
                pass

        def _is_strong_filename_candidate(name: str) -> bool:
            s = str(name or "").strip()
            if not s:
                return False
            sl = s.lower()
            token_parts = [t for t in re.split(r"[\s_\-]+", sl) if t]
            has_sep = any(ch in s for ch in (".", "/", "\\", "_", "-"))
            has_digit = any(ch.isdigit() for ch in s)
            has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in s)
            # Real filenames / stems / mixed identifiers are allowed.
            if "." in s or "/" in s or "\\" in s:
                return True
            if has_digit:
                return True
            if has_cjk and any(ch.isascii() and ch.isalpha() for ch in s):
                return len(s) >= 2
            if has_cjk:
                if len(s) >= 16 and not has_sep and not has_digit:
                    return False
                return len(s) >= 2
            if len(token_parts) >= 4 and not has_sep and not has_digit:
                return False
            # Single-token short/generic English terms are topic words, not filenames.
            weak_singletons = {
                "ai", "chip", "chips", "file", "files", "document", "documents",
                "photo", "photos", "image", "images", "picture", "pictures",
                "video", "videos", "audio", "music", "resume", "resumes",
                "invoice", "invoices", "report", "reports", "paper", "papers",
                "data", "csv", "excel", "pdf", "doc", "docx", "ppt", "pptx",
            }
            if len(token_parts) == 1 and (token_parts[0] in weak_singletons or len(token_parts[0]) < 4):
                return False
            return True

        lexical_filenames = [
            fn
            for fn in dict.fromkeys(
                _collapse_repeated_lookup_phrase(fn)
                for fn in lexical_filenames
                if _collapse_repeated_lookup_phrase(fn)
            )
            if _is_strong_filename_candidate(fn)
        ]
        identifier_filename_anchors = _extract_identifier_filename_anchors(
            lexical_query_text or search_need_text
        )
        if identifier_filename_anchors:
            existing_filename_keys = {
                compact_filename_key(fn)
                for fn in lexical_filenames
                if str(fn or "").strip()
            }
            for anchor in identifier_filename_anchors:
                anchor_key = compact_filename_key(anchor)
                if anchor_key and anchor_key not in existing_filename_keys:
                    lexical_filenames.append(anchor)
                    existing_filename_keys.add(anchor_key)
        if effective_category and not explicit_file_ref:
            ql_inventory = (search_need_text or q).lower()
            looks_like_category_inventory = bool(
                re.search(r"\b(?:what|which|show|list|display|find)\b", ql_inventory)
                or any(tok in search_need_text for tok in ("哪些", "有什么", "列出", "看看", "给我看", "查看"))
            )
            if looks_like_category_inventory and not lexical_extensions:
                generic_filename_terms = {
                    "file", "files", "document", "documents", "doc", "docs",
                    "image", "images", "photo", "photos", "picture", "pictures",
                    "video", "videos", "movie", "movies", "clip", "clips",
                    "audio", "audios", "music", "song", "songs", "recording", "recordings",
                    "图片", "照片", "视频", "影片", "录像", "音频", "录音", "音乐", "文件", "文档", "资料",
                }
                stripped_generic = [
                    fn for fn in lexical_filenames
                    if str(fn or "").strip().lower() in generic_filename_terms
                ]
                if stripped_generic and len(stripped_generic) == len(lexical_filenames):
                    yield {
                        "type": "thinking",
                        "delta": "Ignoring generic lexical filename candidates for category inventory query.\n",
                    }
                    lexical_filenames = []
        explicit_filename_lookup = any(
            _query_contains_filename_needle(lexical_query_text or search_need_text, fn)
            for fn in lexical_filenames
        )

        extension_inventory_fast_path = bool(
            lexical_extensions
            and not lexical_filenames
            and not explicit_file_ref
            and generic_inventory_query
        )

        if lexical_filenames or lexical_extensions:
            yield {"type": "thinking", "delta": f"Lexical extraction: filenames={lexical_filenames}, extensions={lexical_extensions}\n"}

            # ===== Extension-only inventory route =====
            if lexical_extensions and not lexical_filenames:
                try:
                    _ext_set = set(
                        (e.lower() if e.startswith(".") else f".{e.lower()}")
                        for e in lexical_extensions
                    )
                    inventory_pack = kb.indexed_file_inventory(
                        allowed_paths=active_paths,
                        file_extensions=sorted(_ext_set),
                        limit=0,
                        hydrate=True,
                        include_documents=False,
                    )
                    if not inventory_pack.get("ready"):
                        yield {"type": "thinking", "delta": "Indexed extension inventory is warming; no metadata scan fallback used.\n"}
                    _seen_ext_paths: set = set()
                    for item in inventory_pack.get("files") or []:
                        meta = dict(item.get("metadata") or {})
                        fp = str(item.get("file_path") or meta.get("file_path") or "").strip()
                        if not fp:
                            continue
                        if fp in _seen_ext_paths:
                            continue
                        _seen_ext_paths.add(fp)
                        doc = item.get("text") or item.get("doc_summary") or meta.get("doc_summary", "")
                        filename_route_results.append({
                            "text": doc,
                            "metadata": meta,
                            "distance": 0.0,
                            "file_name": item.get("file_name") or meta.get("file_name", os.path.basename(fp)),
                            "file_path": fp,
                            "doc_summary": item.get("doc_summary") or meta.get("doc_summary", ""),
                            "doc_category": item.get("doc_category") or meta.get("doc_category", "other"),
                            "score": 1.0,
                            "_is_lexical_hit": True,
                            "_lexical_filename_exact": False,
                            "_direct_score": 60,
                            "_bm25_score": 5.0,
                        })
                    yield {"type": "thinking", "delta": f"Extension inventory: found {len(_seen_ext_paths)} unique files for {list(_ext_set)}\n"}
                except Exception as _ext_err:
                    yield {"type": "thinking", "delta": f"Extension inventory failed: {_ext_err}\n"}

                if extension_inventory_fast_path and filename_route_results:
                    inventory_sources: List[Dict[str, Any]] = []
                    seen_inventory_paths: set[str] = set()
                    for item in sorted(
                        filename_route_results,
                        key=lambda row: (
                            os.path.splitext(str(row.get("file_name") or row.get("file_path") or ""))[1].lower(),
                            str(row.get("file_name") or "").lower(),
                            str(row.get("file_path") or "").lower(),
                        ),
                    ):
                        fp = str(item.get("file_path") or "").strip()
                        if not fp or fp in seen_inventory_paths:
                            continue
                        inventory_sources.append(dict(item))
                        seen_inventory_paths.add(fp)

                    self._clear_count_scope_context(session_id, reason="search_results_updated")
                    self._set_last_search_results(session_id, inventory_sources[:50])
                    self._set_followup_hint(
                        session_id,
                        action="process_previous",
                        params={},
                        ttl_turns=2,
                        uses=2,
                    )
                    yield from _emit_files_from_sources(inventory_sources)
                    shown_count = len(inventory_sources)
                    ext_label = ", ".join(
                        sorted(
                            {
                                f".{str(ext).lower().lstrip('.')}"
                                for ext in (lexical_extensions or [])
                                if str(ext or "").strip()
                            }
                        )
                    ) or "requested extension"
                    inventory_answer = (
                        f"已找到 {shown_count} 个 {ext_label} 文件。"
                        if user_lang == "zh"
                        else f"Found {shown_count} {ext_label} file(s)."
                    )
                    _first_text_trace = _mark_first_text("extension_inventory_fast_path", chars=len(inventory_answer))
                    if _first_text_trace:
                        yield _first_text_trace
                    yield {"type": "text", "content": inventory_answer}
                    try:
                        hist_ref[-1]["a"] = inventory_answer
                    except Exception:
                        pass
                    yield _emit_timing_trace("request_done", action=action, query_type="search", sources_count=shown_count)
                    yield {"type": "done", "ok": True, "query_type": "search", "sources": inventory_sources, "trace": []}
                    return
            # =============================================

            if lexical_filenames:
                yield {
                    "type": "thinking",
                    "delta": (
                        "Filename lexical hints routed to indexed BM25/semantic recall: "
                        f"{lexical_filenames}\n"
                    ),
                }
    except Exception:
        pass
    # =======================================

    # Keep the extracted anchors as retrieval hints only. Query-time full metadata
    # scans are intentionally avoided; filename/alias surfaces are already in the
    # keyword index used by kb.vector_search().
    strong_lookup_anchors: List[str] = list(identifier_filename_anchors or [])

    cleaned_kw = None
    if keywords:
        import ast
        try:
            parsed_kw = ast.literal_eval(keywords)
            if isinstance(parsed_kw, list):
                cleaned_kw = " ".join(str(k) for k in parsed_kw)
            else:
                cleaned_kw = str(keywords)
        except Exception:
            cleaned_kw = keywords.replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace(",", " ")
        if cleaned_kw.strip() == query_for_search.strip():
            cleaned_kw = None
        if cleaned_kw:
            if not intent_normalized_en:
                cleaned_kw = self._augment_query_for_retrieval(
                    cleaned_kw,
                    prompt_language=user_lang,
                    session_id=session_id,
                )
            kw_has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in cleaned_kw)
            if kw_has_cjk:
                rq_has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in retrieval_query)
                if not rq_has_cjk:
                    cleaned_kw = retrieval_query
            yield {"type": "thinking", "delta": f"Retrieval keywords: {cleaned_kw}\n"}
    if explicit_file_ref and not cleaned_kw:
        cleaned_kw = str(explicit_file_ref.get("search_term") or "").strip()
        if cleaned_kw:
            yield {"type": "thinking", "delta": f"Filename recall keyword: {cleaned_kw}\n"}
    if lexical_filenames and not cleaned_kw:
        cleaned_kw = " ".join(str(fn or "").strip() for fn in lexical_filenames[:4] if str(fn or "").strip()) or None
        if cleaned_kw:
            yield {"type": "thinking", "delta": f"Indexed lexical recall keyword: {cleaned_kw}\n"}
    if (
        not cleaned_kw
        and str((params or {}).get("_expert_route") or "").strip() == "personal_attribute"
        and str((params or {}).get("_resolved_entity") or "").strip()
    ):
        cleaned_kw = str((params or {}).get("_resolved_entity") or "").strip()
        yield {"type": "thinking", "delta": f"Personal attribute keyword anchor: {cleaned_kw}\n"}
    if not cleaned_kw:
        fallback_kw_source = str(search_need_text or retrieval_query or query_for_search or q or "").strip()
        fallback_kw = self._strip_meta_for_rerank(fallback_kw_source)
        fallback_terms = [
            str(part or "").strip()
            for part in re.split(r"[\s,，/\\]+", fallback_kw)
            if len(str(part or "").strip()) >= 2
        ]
        if len(fallback_terms) <= 1 and any("\u4e00" <= ch <= "\u9fff" for ch in fallback_kw):
            fallback_terms = [
                str(term or "").strip()
                for term in extract_lookup_terms(fallback_kw, max_terms=12)
                if len(str(term or "").strip()) >= 2
            ]
        generic_only_terms = {
            "file", "files", "document", "documents", "doc", "docs",
            "image", "images", "photo", "photos", "picture", "pictures",
            "video", "videos", "audio", "audios", "recording", "recordings",
            "music", "song", "songs", "report", "reports", "resume", "resumes",
            "paper", "papers", "invoice", "invoices", "data", "table", "tables",
            "文件", "文档", "图片", "照片", "视频", "音频", "录音", "资料", "表格",
            "报告", "简历", "论文", "发票", "数据",
        }
        if fallback_terms and not all(str(term).lower() in generic_only_terms for term in fallback_terms):
            cleaned_kw = " ".join(list(dict.fromkeys(fallback_terms))[:8])
            yield {"type": "thinking", "delta": f"Fallback keyword anchor: {cleaned_kw}\n"}

    _search_t0 = time.time()
    def _elapsed_ms() -> int:
        return int((time.time() - _search_t0) * 1000)

    def _log_dispatch_stage(stage: str, start_ts: float, **extra: Any) -> None:
        payload = {
            "stage": stage,
            "duration_ms": int((time.time() - start_ts) * 1000),
            **extra,
        }
        try:
            logger.info("[dispatch_stage] %s", json.dumps(payload, ensure_ascii=False, default=str))
        except Exception:
            logger.info("[dispatch_stage] %s duration_ms=%s", stage, payload.get("duration_ms"))
            
    _initial_recall_t0 = time.time()

    final_extensions = list(set([e.lower() if e.startswith(".") else f".{e.lower()}" for e in lexical_extensions])) if lexical_extensions else None
    final_extension_set = set(final_extensions or [])
    if final_extension_set & _DATA_LIKE_EXTS:
        data_exts = sorted(final_extension_set & _DATA_LIKE_EXTS)
        if explicit_category in {"", "other", "document", "spreadsheet", "table", "data"}:
            if explicit_category != "data":
                logger.info(
                    "[dispatch] remapping explicit_category=%r to data for data extensions=%s",
                    explicit_category,
                    data_exts,
                )
            explicit_category = "data"
        if effective_category in {"", "other", "document", "spreadsheet", "table", "data"}:
            effective_category = "data"

    if lexical_filenames:
        try:
            existing_filename_paths = {
                str(src.get("file_path") or "").strip()
                for src in filename_route_results
                if str(src.get("file_path") or "").strip()
            }
            lexical_filename_hits: List[Dict[str, Any]] = []
            for needle in list(dict.fromkeys(str(fn or "").strip() for fn in lexical_filenames if str(fn or "").strip()))[:8]:
                needle_base = os.path.basename(needle)
                needle_has_ext = has_plausible_filename_extension(needle_base)
                if not needle_has_ext and needle not in strong_lookup_anchors:
                    continue
                needle_exts = None
                needle_ext = os.path.splitext(needle_base)[1].lower()
                if needle_ext:
                    needle_exts = [needle_ext]
                indexed_hits = kb.indexed_keyword_search(
                    needle,
                    allowed_paths=active_paths,
                    category_filter=explicit_category or None,
                    file_extensions=needle_exts or final_extensions,
                    limit=24,
                ) if hasattr(kb, "indexed_keyword_search") else []
                for hit in indexed_hits or []:
                    meta = dict(hit.get("metadata") or {})
                    fp = str(hit.get("file_path") or meta.get("file_path") or "").strip()
                    if not fp:
                        continue
                    fname = str(hit.get("file_name") or meta.get("file_name") or os.path.basename(fp))
                    aliases = str(hit.get("lookup_aliases") or meta.get("lookup_aliases") or "")
                    exact_match, surface_score = score_filename_surface_match(needle, fname, aliases)
                    lookup_exact, overlap = lookup_match_quality(needle, " ".join([fname, os.path.splitext(fname)[0], fp, aliases]))
                    bm25_score = float(hit.get("_bm25_score", 0.0) or 0.0)
                    if not exact_match and not lookup_exact and surface_score < 84 and overlap <= 0 and bm25_score <= 0:
                        continue
                    if needle_has_ext and os.path.basename(fname).lower() == needle_base.lower():
                        direct_score = 100
                        exact_match = True
                    elif exact_match:
                        direct_score = 100
                    elif lookup_exact:
                        direct_score = 96
                    else:
                        direct_score = max(90 if needle_has_ext and surface_score >= 84 else 70, surface_score, min(94, 78 + int(overlap * 6)))
                    if direct_score < 90 and needle_has_ext:
                        continue
                    candidate = {
                        "text": str(hit.get("text") or meta.get("doc_summary") or ""),
                        "metadata": meta,
                        "distance": 0.0,
                        "file_name": fname,
                        "file_path": fp,
                        "doc_summary": hit.get("doc_summary") or meta.get("doc_summary", ""),
                        "doc_category": self._normalize_category_name(
                            str(meta.get("doc_category_family") or meta.get("doc_category") or hit.get("doc_category") or "other")
                        ),
                        "lookup_aliases": aliases,
                        "score": max(0.90, float(settings.RELEVANCE_THRESHOLD) + 0.2),
                        "rerank_score": max(0.90, float(settings.RELEVANCE_THRESHOLD) + 0.2),
                        "_is_lexical_hit": True,
                        "_lexical_filename_exact": bool(exact_match or lookup_exact),
                        "_direct_score": direct_score,
                        "_bm25_score": max(bm25_score, 20.0 if direct_score >= 90 else 5.0),
                        "_filename_anchor": needle,
                    }
                    lexical_filename_hits.append(candidate)

            if lexical_filename_hits:
                lexical_filename_hits.sort(
                    key=lambda src: (
                        -int(src.get("_direct_score", 0) or 0),
                        -float(src.get("_bm25_score", 0.0) or 0.0),
                        str(src.get("file_name") or "").lower(),
                        str(src.get("file_path") or "").lower(),
                    )
                )
                added = 0
                for src in lexical_filename_hits[:12]:
                    fp = str(src.get("file_path") or "").strip()
                    if not fp or fp in existing_filename_paths:
                        continue
                    filename_route_results.append(src)
                    existing_filename_paths.add(fp)
                    added += 1
                if added:
                    unique_hit_count = len(_dedupe_sources_by_file_identity(lexical_filename_hits))
                    yield {
                        "type": "thinking",
                        "delta": f"Exact filename lexical route: {unique_hit_count} indexed file(s)\n",
                    }
        except Exception as exc:
            logger.warning("[dispatch] lexical filename indexed lookup failed: %s", exc)

    if strong_lookup_anchors:
        try:
            hits_by_path: Dict[str, Dict[str, Any]] = {}
            for anchor in strong_lookup_anchors[:4]:
                anchor = str(anchor or "").strip()
                if not anchor:
                    continue
                indexed_hits = kb.indexed_keyword_search(
                    anchor,
                    allowed_paths=active_paths,
                    category_filter=explicit_category or None,
                    file_extensions=final_extensions,
                    limit=24,
                ) if hasattr(kb, "indexed_keyword_search") else []
                anchor_key = compact_filename_key(anchor)
                for hit in indexed_hits or []:
                    meta = dict(hit.get("metadata") or {})
                    fp = str(hit.get("file_path") or meta.get("file_path") or "").strip()
                    if not fp:
                        continue
                    fname = str(hit.get("file_name") or meta.get("file_name") or os.path.basename(fp))
                    aliases = str(hit.get("lookup_aliases") or meta.get("lookup_aliases") or "")
                    name_blob = " ".join([fname, os.path.splitext(fname)[0], fp, aliases])
                    name_blob_key = compact_filename_key(name_blob)
                    exact_surface = bool(anchor_key and anchor_key in name_blob_key)
                    exact_lookup, overlap = lookup_match_quality(anchor, name_blob)
                    bm25_score = float(hit.get("_bm25_score", 0.0) or 0.0)
                    if not exact_surface and not exact_lookup and overlap <= 0 and bm25_score <= 0:
                        continue
                    direct_score = 100 if exact_surface else (96 if exact_lookup else max(90, min(94, 78 + int(overlap * 6))))
                    candidate = {
                        "text": str(hit.get("text") or meta.get("doc_summary") or ""),
                        "metadata": meta,
                        "distance": 0.0,
                        "file_name": fname,
                        "file_path": fp,
                        "doc_summary": hit.get("doc_summary") or meta.get("doc_summary", ""),
                        "doc_category": self._normalize_category_name(
                            str(meta.get("doc_category_family") or meta.get("doc_category") or hit.get("doc_category") or "other")
                        ),
                        "lookup_aliases": aliases,
                        "score": max(0.90, float(settings.RELEVANCE_THRESHOLD) + 0.2),
                        "rerank_score": max(0.90, float(settings.RELEVANCE_THRESHOLD) + 0.2),
                        "_is_lexical_hit": True,
                        "_lexical_filename_exact": bool(exact_surface or exact_lookup),
                        "_direct_score": direct_score,
                        "_bm25_score": max(bm25_score, 20.0 if exact_surface else 5.0),
                        "_identifier_anchor": anchor,
                    }
                    prev = hits_by_path.get(fp)
                    if prev is None or int(candidate.get("_direct_score", 0) or 0) > int(prev.get("_direct_score", 0) or 0):
                        hits_by_path[fp] = candidate

            if hits_by_path:
                identifier_hits = sorted(
                    hits_by_path.values(),
                    key=lambda src: (
                        -int(src.get("_direct_score", 0) or 0),
                        str(src.get("file_name") or "").lower(),
                        str(src.get("file_path") or "").lower(),
                    ),
                )
                existing_paths = {
                    str(src.get("file_path") or "").strip()
                    for src in filename_route_results
                    if str(src.get("file_path") or "").strip()
                }
                for src in identifier_hits[:12]:
                    fp = str(src.get("file_path") or "").strip()
                    if fp and fp not in existing_paths:
                        filename_route_results.append(src)
                        existing_paths.add(fp)
                yield {
                    "type": "thinking",
                    "delta": f"Identifier filename anchors: {strong_lookup_anchors[:4]} -> {len(identifier_hits)} indexed hit(s)\n",
                }
        except Exception as exc:
            logger.warning("[dispatch] identifier filename anchor lookup failed: %s", exc)

    media_route = action in {"media_export", "media_content_search"} or str((params or {}).get("_expert_route") or "").strip() == "media"
    media_target_hint = str((params or {}).get("file_hint") or "").strip()
    media_target_type = str((params or {}).get("target_type") or "").strip().lower()
    category_inventory_mode = str((params or {}).get("_inventory_mode") or "").strip().lower() == "category"
    folder_listing_route = str((params or {}).get("_expert_route") or "").strip().lower() == "folder_listing"
    folder_listing_query = bool(
        getattr(self, "_looks_like_folder_listing_query", None)
        and self._looks_like_folder_listing_query(q)
    )
    plain_folder_name_recall = not bool(
        direct_folder_seed_sources
        or folder_listing_route
        or folder_listing_query
        or folder_filter
        or category_inventory_mode
    )
    folder_chain_recall_enabled = _should_run_folder_chain_recall(
        direct_folder_seed_sources=direct_folder_seed_sources,
        folder_listing_route=folder_listing_route,
        folder_listing_query=folder_listing_query,
        folder_filter=folder_filter,
        category_inventory_mode=category_inventory_mode,
    )
    requested_media_type = str((params or {}).get("media_type") or "").strip().lower()
    if requested_media_type not in {"audio", "video"}:
        requested_media_type = ""
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv", ".wmv", ".ts"}
    audio_exts = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape"}

    def _infer_media_scope_kind() -> str:
        hint_ext = os.path.splitext(media_target_hint)[1].lower()
        if hint_ext in video_exts:
            return "video"
        if hint_ext in audio_exts:
            return "audio"
        if media_target_type == "video_visual":
            return "video"
        if media_target_type == "video_audio":
            return "video"
        if media_target_type == "audio_content":
            return "audio"
        return ""

    media_scope_kind = _infer_media_scope_kind() if media_route else ""

    def _inventory_file_matches_requested_media(file_info: Dict[str, Any]) -> bool:
        if not requested_media_type:
            return True
        raw_category = self._normalize_category_name(str(file_info.get("doc_category") or "other"))
        if raw_category == requested_media_type:
            return True
        file_path = str(file_info.get("file_path") or "").strip().lower()
        file_ext = os.path.splitext(file_path)[1].lower()
        if requested_media_type == "audio":
            return file_ext in audio_exts
        return file_ext in video_exts

    # ── Early personal_attribute flags (needed before parallel retrieval) ──
    _early_expert_route   = str((params or {}).get("_expert_route") or "").strip()
    _early_entity_hint    = str((params or {}).get("_resolved_entity") or "").strip()
    _early_attribute_hint = str((params or {}).get("_resolved_attribute") or "").strip()
    _early_personal_attr  = _early_expert_route == "personal_attribute" and bool(_early_entity_hint)

    # ── personal_attribute follow-up shortcut ──────────────────────────────
    # skip expensive parallel semantic retrieval and reuse prior session results.
    # Hash-embedding fallback makes semantic search return random results anyway.
    _personal_attr_used_prior = False
    _prior_entity_files: List[Dict[str, Any]] = []
    if _early_personal_attr:
        try:
            _prior_results = self._get_last_search_results_ref(session_id) or []
            _entity_lower  = _early_entity_hint.lower()
            _threshold_floor = max(float(settings.RELEVANCE_THRESHOLD) + 0.05, 0.55)
            for _pr in _prior_results:
                _pr_blob = " ".join([
                    str(_pr.get("file_name") or ""),
                    str(_pr.get("doc_summary") or ""),
                    str(_pr.get("file_path") or ""),
                    str(_pr.get("lookup_aliases") or ""),
                ]).lower()
                if _entity_lower not in _pr_blob:
                    continue
                _enriched = dict(_pr)
                # Guarantee these files pass the threshold filter.
                # The actual cross-encoder rerank will overwrite if it runs.
                if not _enriched.get("rerank_score"):
                    _enriched["rerank_score"] = _threshold_floor
                if not _enriched.get("score"):
                    _enriched["score"] = _threshold_floor
                # Ensure text is populated for LLM answer generation
                if not _enriched.get("text"):
                    _enriched["text"] = str(_enriched.get("doc_summary") or "")
                _enriched["_is_prior_entity_file"] = True
                _prior_entity_files.append(_enriched)
            if _prior_entity_files:
                logger.info(
                    "[personal_attr_shortcut] Reusing %d prior entity files for entity=%r attr=%r; skipping parallel retrieval.",
                    len(_prior_entity_files), _early_entity_hint, _early_attribute_hint,
                )
                _personal_attr_used_prior = True
        except Exception as _pae:
            logger.debug("[personal_attr_shortcut] prior lookup failed: %s", _pae)
    # ──────────────────────────────────────────────────────────────────────

    import concurrent.futures


    def _run_semantic():
        _stage_t0 = time.time()
        logger.info(
            "[dispatch_stage] semantic_start query=%r category=%r keyword=%r folder=%r extensions=%s active_paths=%s",
            retrieval_query,
            explicit_category,
            cleaned_kw,
            folder_filter,
            final_extensions,
            len(active_paths or []),
        )
        try:
            out = kb.vector_search(
                retrieval_query,
                n_results=settings.VECTOR_SEARCH_TOP_K,
                allowed_paths=active_paths,
                category_filter=explicit_category,
                keyword=cleaned_kw,
                folder=folder_filter,
                original_query=search_need_text,
                file_extensions=final_extensions,
            )
            _log_dispatch_stage("semantic_done", _stage_t0, candidates=len(out or []))
            return out
        except Exception as e:
            _log_dispatch_stage("semantic_failed", _stage_t0, error=str(e))
            logger.warning(f"[_run_semantic] failed: {e}")
            return []

    def _run_category():
        _stage_t0 = time.time()
        logger.info(
            "[dispatch_stage] category_start effective_category=%r keyword=%r folder=%r active_paths=%s",
            effective_category,
            cleaned_kw if cleaned_kw else None,
            folder_filter,
            len(active_paths or []),
        )
        cat_res: List[Dict[str, Any]] = []
        if not effective_category:
            _log_dispatch_stage("category_skipped", _stage_t0, reason="no_effective_category")
            return cat_res
        try:
            query_blob = _build_lexical_query_text(
                retrieval_query,
                cleaned_kw or "",
                search_need_text,
            )
            category_keyword_query = query_blob if cleaned_kw else None
            cat_pack = kb.count_by_category(
                    category=effective_category,
                    keyword=category_keyword_query,
                    allowed_paths=active_paths,
                    folder=folder_filter,
                    original_query=search_need_text,
                    compatible_category_match=True,
                )
            cat_files_primary = cat_pack.get("files") or []
            cat_files_supplemental: List[Dict[str, Any]] = []
            if cleaned_kw:
                cat_pack_full = kb.count_by_category(
                    category=effective_category,
                    allowed_paths=active_paths,
                    folder=folder_filter,
                    original_query=search_need_text,
                    compatible_category_match=True,
                )
                cat_files_supplemental = cat_pack_full.get("files") or []

            category_candidate_cap = max(30, int(settings.VECTOR_SEARCH_TOP_K) * 3)
            if cleaned_kw:
                category_candidate_cap = max(category_candidate_cap, int(settings.VECTOR_SEARCH_TOP_K) * 4, 48)
            if self._normalize_category_name(str(effective_category or "")) in {"resume", "paper"}:
                category_candidate_cap = max(category_candidate_cap, 60)
            if self._normalize_category_name(str(effective_category or "")) == "resume":
                try:
                    resume_cat_cap = int(os.getenv("SEARCH_CATEGORY_CANDIDATES_RESUME", "96") or 96)
                except ValueError:
                    resume_cat_cap = 96
                category_candidate_cap = max(category_candidate_cap, resume_cat_cap)

            cat_files = _merge_category_inventory_candidates(
                cat_files_primary,
                cat_files_supplemental,
                query_text=query_blob,
                max_files=category_candidate_cap,
            )
            if not cat_files:
                return cat_res

            # Only fetch the first chunk for the small category candidate set.
            # Rebuilding a full include_documents cache here is expensive on large
            # libraries and does not improve retrieval quality.
            _chunk_by_id: Dict[str, tuple[str, Dict[str, Any]]] = {}
            try:
                _cat_chunk_ids = [
                    str(cf.get("_first_chunk_id") or "").strip()
                    for cf in cat_files
                    if str(cf.get("_first_chunk_id") or "").strip()
                ]
                if _cat_chunk_ids:
                    _chunk_batch = kb.collection.get(
                        ids=list(dict.fromkeys(_cat_chunk_ids)),
                        include=["documents", "metadatas"],
                    )
                    for _cid, _cdoc, _cmeta in zip(
                        _chunk_batch.get("ids") or [],
                        _chunk_batch.get("documents") or [],
                        _chunk_batch.get("metadatas") or [],
                    ):
                        _chunk_by_id[str(_cid or "")] = (
                            str(_cdoc or ""),
                            dict(_cmeta or {}),
                        )
            except Exception:
                _chunk_by_id = {}

            for cf in cat_files:
                fp = str(cf.get("file_path") or "")
                if not fp:
                    continue

                _chunk_id = str(cf.get("_first_chunk_id") or "").strip()
                if _chunk_id and _chunk_id in _chunk_by_id:
                    doc, meta = _chunk_by_id[_chunk_id]
                else:
                    try:
                        chunk_res = kb.collection.get(
                            where={"file_path": fp},
                            limit=1,
                            include=["documents", "metadatas"],
                        )
                        if chunk_res and chunk_res.get("documents"):
                            doc = str((chunk_res.get("documents") or [""])[0] or "")
                            meta = (chunk_res.get("metadatas") or [{}])[0] or {}
                        else:
                            raise ValueError("empty")
                    except Exception:
                        doc = str(cf.get("doc_summary") or "")
                        meta = {
                            "file_name": cf.get("file_name", ""),
                            "file_path": fp,
                            "doc_summary": cf.get("doc_summary", ""),
                            "doc_category": cf.get("doc_category", effective_category),
                            "doc_category_family": cf.get("doc_category_family", cf.get("doc_category", effective_category)),
                            "doc_category_leaf": cf.get("doc_category_leaf", cf.get("doc_category", effective_category)),
                            "doc_role": cf.get("doc_role", "primary_source"),
                        }

                cat_res.append(
                    {
                        "text": doc,
                        "metadata": meta,
                        "distance": 0.0,
                        "file_name": meta.get("file_name", cf.get("file_name", "")),
                        "file_path": meta.get("file_path", fp),
                        "file_name_en": meta.get("file_name_en", cf.get("file_name_en", "")),
                        "folder_name_en": meta.get("folder_name_en", cf.get("folder_name_en", "")),
                        "doc_summary": meta.get("doc_summary", cf.get("doc_summary", "")),
                        "doc_category": self._normalize_category_name(
                            meta.get("doc_category_family") or meta.get("doc_category", effective_category)
                        ),
                        "doc_category_family": self._normalize_category_name(
                            meta.get("doc_category_family") or meta.get("doc_category", effective_category)
                        ),
                        "doc_category_leaf": str(
                            meta.get("doc_category_leaf")
                            or meta.get("doc_category_family")
                            or meta.get("doc_category")
                            or effective_category
                        ),
                        "doc_role": str(meta.get("doc_role") or "primary_source"),
                        "lookup_aliases": meta.get("lookup_aliases", cf.get("lookup_aliases", "")),
                        "_inventory_keyword_hit": bool(cf.get("_inventory_keyword_hit")),
                        "score": max(0.70, float(settings.RELEVANCE_THRESHOLD) + 0.05),
                    }
                )
        except Exception as e:
            _log_dispatch_stage("category_failed", _stage_t0, error=str(e))
            logger.warning(f"[_run_category] failed: {e}")
        else:
            _log_dispatch_stage("category_done", _stage_t0, candidates=len(cat_res))
        return cat_res

    def _run_indexed_keyword():
        _stage_t0 = time.time()
        keyword_query = str(
            _build_lexical_query_text(
                retrieval_query,
                cleaned_kw or "",
                search_need_text,
            )
            if cleaned_kw
            else ""
        ).strip()
        if not keyword_query:
            _log_dispatch_stage("indexed_keyword_skipped", _stage_t0, reason="no_keyword_anchor")
            return []
        try:
            logger.info(
                "[dispatch_stage] indexed_keyword_start query=%r category=%r folder=%r extensions=%s active_paths=%s",
                keyword_query,
                explicit_category or effective_category or "",
                folder_filter,
                final_extensions,
                len(active_paths or []),
            )
            out = kb.indexed_keyword_search(
                keyword_query,
                allowed_paths=active_paths,
                category_filter=explicit_category or effective_category or "",
                file_extensions=final_extensions,
                limit=max(24, int(settings.VECTOR_SEARCH_TOP_K) * 2),
            ) if hasattr(kb, "indexed_keyword_search") else []
            _log_dispatch_stage("indexed_keyword_done", _stage_t0, candidates=len(out or []))
            return out
        except Exception as exc:
            _log_dispatch_stage("indexed_keyword_failed", _stage_t0, error=str(exc))
            logger.warning("[dispatch] indexed keyword recall failed: %s", exc)
            return []


    def _run_folder_cands():
        """Collect folder candidates only; rerank them later on the main thread."""
        _stage_t0 = time.time()
        logger.info(
            "[dispatch_stage] folder_candidates_start query=%r active_paths=%s",
            retrieval_query,
            len(active_paths or []),
        )
        try:
            cands = kb.collect_folder_index_candidates(
                retrieval_query,
                original_query=search_need_text,
                allowed_paths=active_paths,
            )
            _log_dispatch_stage("folder_candidates_done", _stage_t0, candidates=len(cands or []))
            return cands
        except Exception as e:
            _log_dispatch_stage("folder_candidates_failed", _stage_t0, error=str(e))
            logger.warning(f"[_run_folder_cands] failed: {e}")
            return []

    _folder_chain_t0 = time.time()
    if _personal_attr_used_prior:
        # ── Shortcut: reuse prior session entity files, skip parallel retrieval ──
        semantic_results        = list(_prior_entity_files)
        category_route_results  = []
        indexed_keyword_results = []
        folder_index_candidates = []
        folder_reranked         = []
        logger.info(
            "[personal_attr_shortcut] Skipped parallel retrieval; using %d prior entity files directly.",
            len(semantic_results),
        )
    else:
        yield {
            "type": "thinking",
            "delta": "Starting broad semantic recall from indexed database...\n",
        }
        _serialize_recall = str(os.getenv("FILEAGENT_SERIALIZE_RECALL", "1") or "").strip().lower() not in {
            "0", "false", "off", "no"
        }
        if _serialize_recall:
            semantic_results = _run_semantic()
            category_route_results = _run_category()
            indexed_keyword_results = _run_indexed_keyword()
            if folder_chain_recall_enabled:
                folder_index_candidates = _run_folder_cands()
            else:
                folder_index_candidates = []
                logger.info("[folder_chain] skipped for broad content search")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                f_sem = executor.submit(_run_semantic)
                f_cat = executor.submit(_run_category)
                f_kw = executor.submit(_run_indexed_keyword)
                f_fol = executor.submit(_run_folder_cands) if folder_chain_recall_enabled else None

                semantic_results        = f_sem.result()
                category_route_results  = f_cat.result()
                indexed_keyword_results = f_kw.result()
                folder_index_candidates = f_fol.result() if f_fol is not None else []
                if f_fol is None:
                    logger.info("[folder_chain] skipped for broad content search")

        try:
            _f_rk = max(20, int(os.getenv("FOLDER_CHAIN_RERANK_TOP_K", "40") or 40))
        except ValueError:
            _f_rk = 40
        _folder_rerank_t0 = time.time()
        folder_reranked = kb.rerank(retrieval_query, folder_index_candidates, top_k=_f_rk) if folder_index_candidates else []
        _log_dispatch_stage(
            "folder_rerank_done",
            _folder_rerank_t0,
            input_candidates=len(folder_index_candidates or []),
            output_candidates=len(folder_reranked or []),
        )

    if not folder_index_candidates and semantic_results:
        _inferred_folders: Dict[str, Dict[str, Any]] = {}
        _inf_hints = [str(h or "").strip().lower() for h in (lexical_filenames or [])]
        _inf_q_words = [
            w for w in re.split(r"[\s,，/\\]+", str(search_need_text or "").lower())
            if len(w) > 1 and w not in {"find", "找", "搜索", "search", "show", "显示", "get", "tell", "about"}
        ]
        _inf_hints_all = list(dict.fromkeys(_inf_hints + _inf_q_words))
        for _sr in semantic_results:
            _sr_fp = str(_sr.get("file_path") or "").strip()
            if not _sr_fp:
                continue
            _par = os.path.dirname(_sr_fp)
            if not _par or _par == _sr_fp:
                continue
            _par_bn = os.path.basename(os.path.normpath(_par)).lower()
            if not _par_bn or _par in _inferred_folders:
                continue
            for _h in _inf_hints_all:
                if not _h:
                    continue
                if _h in _par_bn or _par_bn in _h or compute_lookup_overlap_score(_h, _par_bn) > 0:
                    _inferred_folders[_par] = {
                        "file_path": _par,
                        "file_name": os.path.basename(os.path.normpath(_par)),
                        "folder_path": _par,
                        "rerank_score": float(settings.RELEVANCE_THRESHOLD) + 0.35,
                        "_direct_score": 95,
                        "_folder_literal_hit": True,
                        "_inferred_folder_match": True,
                    }
                    logger.info(
                        "[dispatch] inferred folder match from semantic parent: %r (hint=%r)",
                        _par, _h,
                    )
                    break
        if _inferred_folders:
            folder_reranked = list(_inferred_folders.values()) + list(folder_reranked)
            logger.info(
                "[dispatch] added %d inferred folder(s) to folder_reranked",
                len(_inferred_folders),
            )

    _parallel_recall_ms = int((time.time() - _folder_chain_t0) * 1000)
    logger.info(
        "[parallel_recall] semantic=%d category=%d indexed_keyword=%d folder_cands=%d elapsed_ms=%d shortcut=%s",
        len(semantic_results or []),
        len(category_route_results),
        len(indexed_keyword_results or []),
        len(folder_index_candidates),
        _parallel_recall_ms,
        _personal_attr_used_prior,
    )


    # folder_reranked is set inside the if/else block above
    # (shortcut: [] | full path: kb.rerank(...))

    merged_candidates: List[Dict[str, Any]] = []
    seen_merge_keys = set()
    for d in (filename_route_results or []) + (indexed_keyword_results or []) + category_route_results + (semantic_results or []):
        fp = str(d.get("file_path") or "")
        tx = str(d.get("text") or "")[:120]
        key = f"{fp}::{tx}"
        if key in seen_merge_keys:
            continue
        seen_merge_keys.add(key)
        merged_candidates.append(d)
        
    results = merged_candidates
    _initial_recall_ms = int((time.time() - _initial_recall_t0) * 1000)

    try:
        _f_exp = int(os.getenv("FOLDER_CHAIN_EXPAND_PER_FOLDER", "0") or 0)
    except ValueError:
        _f_exp = 0
    try:
        _f_roots = max(1, int(os.getenv("FOLDER_CHAIN_MAX_ROOTS", "8") or 8))
    except ValueError:
        _f_roots = 8

    _q_for_folder_gate = str(search_need_text or retrieval_query or "").strip()
    folder_after_thr: List[Dict[str, Any]] = []
    for _fd in folder_reranked:
        if float(_fd.get("rerank_score", 0.0) or 0.0) < float(settings.RELEVANCE_THRESHOLD):
            continue
        if plain_folder_name_recall and not (
            bool(_fd.get("_folder_literal_hit"))
            or int(_fd.get("_direct_score", 0) or 0) >= 90
        ):
            continue
        if _q_for_folder_gate:
            _fd_blob = " ".join([
                str(_fd.get("file_name") or ""),
                str(_fd.get("folder_name") or os.path.basename(_fd.get("file_path") or "") or ""),
                str(_fd.get("lookup_aliases") or ""),
                str(_fd.get("doc_summary") or ""),
            ])
            _fd_overlap = compute_lookup_overlap_score(_q_for_folder_gate, _fd_blob)
            _fd_direct = int(_fd.get("_direct_score", 0) or 0)
            _folder_high_thr = max(float(settings.RELEVANCE_THRESHOLD) + 0.25, 0.55)
            if _fd_overlap == 0 and _fd_direct < 60 and float(_fd.get("rerank_score", 0.0) or 0.0) < _folder_high_thr:
                logger.debug(
                    "[folder_gate] filtered noise folder %r (overlap=0, direct=%d, rerank=%.3f)",
                    _fd.get("file_name") or _fd.get("file_path"),
                    _fd_direct,
                    float(_fd.get("rerank_score", 0.0) or 0.0),
                )
                continue
        folder_after_thr.append(_fd)

    folder_after_thr_sorted = sorted(
        folder_after_thr,
        key=lambda _fd: (
            0 if bool(_fd.get("_folder_exact_name_hit")) else 1,
            0 if bool(_fd.get("_folder_literal_hit")) else 1,
            -int(_fd.get("_direct_score", 0) or 0),
            -float(_fd.get("rerank_score", 0.0) or 0.0),
            len(str(_fd.get("folder_path") or _fd.get("file_path") or "")),
            str(_fd.get("folder_path") or _fd.get("file_path") or ""),
        ),
    )

    yield {
        "type": "thinking",
        "delta": (
            f"Folder chain: index_candidates={len(folder_index_candidates)}, "
            f"after_rerank_threshold={len(folder_after_thr_sorted)}\n"
        ),
    }
    yield {
        "type": "trace_append",
        "item": {
            "stage": "folder_chain",
            "type": "retrieval",
            "elapsed_ms": _elapsed_ms(),
            "duration_ms": int((time.time() - _folder_chain_t0) * 1000),
            "index_candidates": len(folder_index_candidates),
            "after_rerank_threshold": len(folder_after_thr_sorted),
            "max_roots": _f_roots,
            "expand_per_folder": _f_exp,
        },
    }
    sources_folder_chain: List[Dict[str, Any]] = list(direct_folder_seed_sources or [])
    if folder_after_thr_sorted:
        yield from _emit_status("thinking", "Expanding matched folders (all indexed files under each root)...")
        _fp_kept = [
            str(r.get("folder_path") or r.get("file_path") or "").strip()
            for r in folder_after_thr_sorted[:_f_roots]
            if str(r.get("folder_path") or r.get("file_path") or "").strip()
        ]
        _fp_needles = kb._folder_path_literal_needles(search_need_text, retrieval_query)
        if lexical_filenames:
            _fp_needles.extend(lexical_filenames)
        _exact_literal_roots = kb.collect_exact_folder_literal_roots(
            _fp_needles,
            active_paths,
            max_roots=max(_f_roots, 12),
        )
        _disk_exact_roots: List[str] = []
        if (folder_listing_route or folder_filter) and _fp_needles and not _exact_literal_roots:
            _disk_exact_roots = kb.discover_exact_folder_literal_roots_from_disk(
                _fp_needles,
                active_paths,
                max_roots=max(_f_roots, 12),
            )
        _fp_narrowed = kb._finalize_folder_expand_roots_for_cjk_query(_fp_kept, _fp_needles)
        if not _fp_narrowed:
            # English aliases such as folder_name_en="project materials" can
            # case the path itself will not contain the English needle, so keep
            # literal/direct folder hits as a DB-indexed supplemental recall.
            _direct_literal_roots: List[str] = []
            for _fc_r in folder_after_thr_sorted[:_f_roots]:
                _root = str(_fc_r.get("folder_path") or _fc_r.get("file_path") or "").strip()
                if not _root:
                    continue
                if not (
                    bool(_fc_r.get("_folder_literal_hit"))
                    or int(_fc_r.get("_direct_score", 0) or 0) >= 90
                ):
                    continue
                try:
                    _root_norm = os.path.normpath(os.path.expanduser(_root))
                except Exception:
                    _root_norm = _root
                if _root_norm and _root_norm not in _direct_literal_roots:
                    _direct_literal_roots.append(_root_norm)

            _direct_literal_roots.sort(key=lambda p: (p.count(os.sep), len(p), p))
            _deduped_direct_roots: List[str] = []
            for _root in _direct_literal_roots:
                is_nested = False
                for _kept_root in _deduped_direct_roots:
                    try:
                        is_nested = os.path.commonpath([_root, _kept_root]) == _kept_root
                    except Exception:
                        is_nested = _root.startswith(_kept_root.rstrip(os.sep) + os.sep)
                    if is_nested:
                        break
                if is_nested:
                    continue
                _deduped_direct_roots.append(_root)
                if len(_deduped_direct_roots) >= _f_roots:
                    break
            _fp_kept = _deduped_direct_roots
        else:
            _fp_kept = _fp_narrowed
        if _exact_literal_roots or _disk_exact_roots:
            _combined_roots = list(_disk_exact_roots) + list(_exact_literal_roots) + list(_fp_kept)
            _deduped_exact_first: List[str] = []
            for _root in _combined_roots:
                if not _root or _root in _deduped_exact_first:
                    continue
                _deduped_exact_first.append(_root)
                if len(_deduped_exact_first) >= max(_f_roots, len(_exact_literal_roots), len(_disk_exact_roots)):
                    break
            _fp_kept = _deduped_exact_first
        sources_folder_chain = kb.expand_folder_paths_to_chain_sources(
            _fp_kept,
            max_per_folder=_f_exp,
            allow_skip_basename_roots=bool(folder_listing_route or folder_filter),
            relax_ignore_rules=bool(folder_listing_route or folder_filter),
        )
        if sources_folder_chain:
            def _matches_folder_type_scope(src: Dict[str, Any]) -> bool:
                fp = str(src.get("file_path") or "").strip()
                ext = os.path.splitext(fp)[1].lower()
                doc_cat = self._normalize_category_name(str(src.get("doc_category") or "other"))

                if media_scope_kind == "video":
                    return ext in video_exts
                if media_scope_kind == "audio":
                    return ext in audio_exts

                if final_extensions:
                    return ext in set(final_extensions)

                if explicit_category and explicit_category != "other":
                    from core.retrieval.category_engine import get_compatible_categories

                    compatible_categories = get_compatible_categories(explicit_category) or {explicit_category}
                    if explicit_category == "audio/video":
                        ql_local = (q or retrieval_query or "").lower()
                        asks_video = bool(re.search(r"\b(video|videos|movie|movies|clip|clips)\b|视频|影片|录像", ql_local))
                        asks_audio = bool(re.search(r"\b(audio|audios|recording|recordings|song|songs|music)\b|音频|录音|歌曲|音乐", ql_local))
                        if asks_video and not asks_audio:
                            return ext in video_exts
                        if asks_audio and not asks_video:
                            return ext in audio_exts
                    return doc_cat in compatible_categories

                return True

            filtered_folder_chain = [s for s in sources_folder_chain if _matches_folder_type_scope(s)]
            if len(filtered_folder_chain) != len(sources_folder_chain):
                logger.info(
                    "[dispatch] type-filtered folder chain: %d -> %d",
                    len(sources_folder_chain),
                    len(filtered_folder_chain),
                )
            sources_folder_chain = filtered_folder_chain

    if direct_folder_seed_sources and folder_after_thr:
        _seed_paths = {
            str(item.get("file_path") or "").strip()
            for item in direct_folder_seed_sources
            if str(item.get("file_path") or "").strip()
        }
        if _seed_paths:
            sources_folder_chain = list(direct_folder_seed_sources) + [
                item for item in sources_folder_chain
                if str(item.get("file_path") or "").strip() not in _seed_paths
            ]

    if sources_folder_chain and active_paths is not None:
        scoped_folder_chain = filter_sources_to_scope(
            sources_folder_chain,
            active_paths,
            keep_matching_folders=False,
        )
        if len(scoped_folder_chain) != len(sources_folder_chain):
            logger.info(
                "[dispatch] scope-filtered folder chain: %d -> %d (active_paths=%d)",
                len(sources_folder_chain),
                len(scoped_folder_chain),
                len(active_paths or []),
            )
            sources_folder_chain = scoped_folder_chain

    # ── Build folder cards (one pill per matched root) ───────────────────────
    folder_cards: List[Dict[str, Any]] = []
    if folder_after_thr_sorted and sources_folder_chain:
        _files_per_fc_root: Dict[str, int] = {}
        for _fc_item in sources_folder_chain:
            _fc_root = str(_fc_item.get("folder_chain_root") or "").strip()
            if _fc_root:
                _files_per_fc_root[_fc_root] = _files_per_fc_root.get(_fc_root, 0) + 1
        _seen_fc_roots: set = set()
        for _fc_r in folder_after_thr_sorted[:_f_roots]:
            _fc_root_path = str(_fc_r.get("folder_path") or _fc_r.get("file_path") or "").strip()
            if not _fc_root_path or _fc_root_path in _seen_fc_roots:
                continue
            _seen_fc_roots.add(_fc_root_path)
            try:
                _fc_bn = os.path.basename(os.path.normpath(os.path.expanduser(_fc_root_path))) or _fc_root_path
            except Exception:
                _fc_bn = os.path.basename(_fc_root_path) or _fc_root_path
            _fc_cnt = _files_per_fc_root.get(_fc_root_path, 0)
            if _fc_cnt == 0:
                continue
            _fc_summary = str(_fc_r.get("doc_summary") or "").strip()
            if not _fc_summary:
                if user_lang == "zh":
                    _fc_summary = f"文件夹，包含 {_fc_cnt} 个已索引文件" if _fc_cnt else "已索引文件夹"
                else:
                    _fc_summary = f"Folder containing {_fc_cnt} indexed file(s)" if _fc_cnt else "Indexed folder"
            folder_cards.append({
                "file_name": _fc_bn,
                "file_path": _fc_root_path,
                "doc_category": "folder",
                "doc_summary": _fc_summary,
                "type": "folder",
                "iconType": "folder",
                "is_matched_folder": True,
                "child_file_count": _fc_cnt,
                "rerank_score": float(_fc_r.get("rerank_score", 0.0) or 0.0),
                "_direct_score": int(_fc_r.get("_direct_score", 0) or 0),
                "_folder_literal_hit": bool(_fc_r.get("_folder_literal_hit")),
                "_folder_exact_name_hit": bool(_fc_r.get("_folder_exact_name_hit")),
                "_inferred_folder_match": bool(_fc_r.get("_inferred_folder_match")),
            })
        logger.info(f"[dispatch] folder_cards: {len(folder_cards)} root(s) from {len(folder_after_thr_sorted)} candidates")

    yield {
        "type": "thinking",
        "delta": (
            f"Recall routes: semantic={len(semantic_results or [])}, indexed_keyword={len(indexed_keyword_results or [])}, category={len(category_route_results)}, "
            f"folder_chain_files={len(sources_folder_chain)}\n"
        ),
    }
    yield {
        "type": "trace_append",
        "item": {
            "stage": "recall_routes",
            "type": "retrieval",
            "elapsed_ms": _elapsed_ms(),
            "duration_ms": _initial_recall_ms,
            "semantic_candidates": len(semantic_results or []),
            "indexed_keyword_candidates": len(indexed_keyword_results or []),
            "category_candidates": len(category_route_results),
            "filename_candidates": len(filename_route_results or []),
            "merged_candidates": len(results or []),
            "folder_chain_files": len(sources_folder_chain),
            "effective_category": effective_category,
            "keyword_hint": cleaned_kw or "",
        },
    }

    if category_inventory_mode and (explicit_category or effective_category) and not explicit_file_ref:
        inventory_category = str(explicit_category or effective_category or "").strip()
        # Inventory/listing requests should be precise for named leaf buckets
        # such as resume or invoice. Compatibility expansion is still useful
        # for broad document/report/paper buckets and extension-driven lists.
        inventory_compatible_category_match = bool(final_extensions) or inventory_category in {
            "document",
            "report",
            "paper",
        }
        inventory_files: List[Dict[str, Any]] = []
        try:
            inventory_pack = kb.count_by_category(
                category=inventory_category,
                file_extensions=final_extensions,
                allowed_paths=active_paths,
                folder=folder_filter,
                compatible_category_match=inventory_compatible_category_match,
            )
            inventory_files = list(inventory_pack.get("files") or [])
        except Exception as inventory_exc:
            logger.warning(
                "[dispatch] category inventory fast-path failed for category=%r: %s",
                inventory_category,
                inventory_exc,
            )
            inventory_files = []

        if inventory_files:
            inventory_files = [item for item in inventory_files if _inventory_file_matches_requested_media(item)]

        if inventory_files:
            inventory_sources: List[Dict[str, Any]] = []
            seen_inventory_paths: set[str] = set()
            threshold_floor = max(float(settings.RELEVANCE_THRESHOLD) + 0.05, 0.55)
            for item in sorted(
                inventory_files,
                key=lambda row: (
                    str(row.get("file_name") or "").lower(),
                    str(row.get("file_path") or "").lower(),
                ),
            ):
                file_path = str(item.get("file_path") or "").strip()
                if not file_path or file_path in seen_inventory_paths:
                    continue
                seen_inventory_paths.add(file_path)
                file_name = str(item.get("file_name") or os.path.basename(file_path))
                doc_summary = str(item.get("doc_summary") or "")
                normalized_category = self._normalize_category_name(
                    str(item.get("doc_category") or inventory_category or "other")
                )
                if requested_media_type in {"audio", "video"}:
                    normalized_category = requested_media_type
                source_media_type = requested_media_type or (
                    normalized_category if normalized_category in {"audio", "video"} else ""
                )
                meta = {
                    "file_name": file_name,
                    "file_path": file_path,
                    "doc_summary": doc_summary,
                    "doc_category": normalized_category,
                    "doc_category_family": normalized_category,
                    "doc_category_leaf": normalized_category,
                    "doc_role": "primary_source",
                    "media_type": source_media_type,
                }
                inventory_sources.append(
                    {
                        "text": doc_summary,
                        "metadata": meta,
                        "distance": 0.0,
                        "file_name": file_name,
                        "file_path": file_path,
                        "doc_summary": doc_summary,
                        "doc_category": normalized_category,
                        "doc_category_family": normalized_category,
                        "doc_category_leaf": normalized_category,
                        "doc_role": "primary_source",
                        "media_type": source_media_type,
                        "score": threshold_floor,
                        "rerank_score": threshold_floor,
                    }
                )

            if inventory_sources:
                self._clear_count_scope_context(session_id, reason="search_results_updated")
                self._set_last_search_results(session_id, inventory_sources[:50])
                self._set_followup_hint(
                    session_id,
                    action="process_previous",
                    params={},
                    ttl_turns=2,
                    uses=2,
                )

                yield {
                    "type": "thinking",
                    "delta": (
                        f"Category inventory fast-path: category={inventory_category}, "
                        f"media_type={requested_media_type or 'all'}, files={len(inventory_sources)}\n"
                    ),
                }
                yield from _emit_files_from_sources(inventory_sources)

                shown_count = len(inventory_sources)
                if user_lang == "zh":
                    if requested_media_type == "audio":
                        inventory_answer = f"已找到 {shown_count} 个音频文件。"
                    elif requested_media_type == "video":
                        inventory_answer = f"已找到 {shown_count} 个视频文件。"
                    else:
                        inventory_answer = f"已找到 {shown_count} 个该类别文件。"
                else:
                    if requested_media_type == "audio":
                        inventory_answer = f"Found {shown_count} audio file(s)."
                    elif requested_media_type == "video":
                        inventory_answer = f"Found {shown_count} video file(s)."
                    else:
                        inventory_answer = f"Found {shown_count} file(s) in the requested category."

                _first_text_trace = _mark_first_text("category_inventory_fast_path", chars=len(inventory_answer))
                if _first_text_trace:
                    yield _first_text_trace
                yield {"type": "text", "content": inventory_answer}
                try:
                    hist_ref[-1]["a"] = inventory_answer
                except Exception:
                    pass
                yield _emit_timing_trace(
                    "request_done",
                    action=action,
                    query_type="search",
                    sources_count=shown_count,
                )
                yield {
                    "type": "done",
                    "ok": True,
                    "query_type": "search",
                    "sources": inventory_sources,
                    "trace": [],
                }
                return

    if not results and not sources_folder_chain:
        self._set_followup_hint(
            session_id,
            action="process_previous",
            params={
                "allow_without_results": True,
                "anchor": "search_topic",
                "prior_search_query": search_need_text,
            },
            ttl_turns=1,
            uses=1,
        )
        yield from _emit_status("thinking", "No relevant indexed content found; answering directly...")
        llm = self._get_llm_service(detailed=False, session_id=session_id, prompt_language=user_lang)
        scope_note = ""
        if active_paths:
            scope_note = (
                f"\nCurrent retrieval scope is limited to {len(active_paths)} active source file(s). "
                "If the query appears otherwise clear, mention that the user may need to broaden the source scope."
            )
        chat_prompt = (
            f"No indexed content matched. Reply naturally in {response_language_label}, "
            "clearly stating that no matching indexed files were found. "
            "Do not ask what kind of file/content the user wants if they already supplied a topic or file type; "
            "suggest broader source scope, different keywords, or checking indexing instead."
            f"{scope_note}\n\n"
            f"<User Question>\n{q}\n</User Question>\n"
        )
        resp_text = yield from _collect_or_emit_stream(llm, chat_prompt)
        if resp_text is None:
            return
        try:
            self._get_history_ref(session_id).append({"q": q, "a": resp_text})
        except Exception:
            pass
        yield _emit_timing_trace(
            "request_done",
            action=action,
            query_type="search",
            sources_count=0,
        )
        yield {"type": "done", "ok": True, "query_type": "search", "sources": [], "trace": []}
        return

    data_like_exts = _DATA_LIKE_EXTS
    document_evidence_categories = {
        "invoice", "contract", "quotation", "resume", "manual",
        "report", "paper", "book", "document",
    }
    document_evidence_exts = {
        ".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".pages",
    }
    query_anchor = (retrieval_query or search_need_text or keywords or "").strip()

    def _is_document_evidence_source(src: Dict[str, Any]) -> bool:
        fp = str(src.get("file_path") or "").lower()
        ext = os.path.splitext(fp)[1].lower()
        cat = self._normalize_category_name(
            str(src.get("doc_category_family") or src.get("doc_category") or "other")
        )
        leaf = self._normalize_category_name(
            str(src.get("doc_category_leaf") or src.get("doc_category_raw") or src.get("doc_category") or "other")
        )
        if cat in document_evidence_categories or leaf in document_evidence_categories:
            return True
        return bool(ext in document_evidence_exts and cat in {"document", "other"})

    dynamic_rerank_top_k = settings.RERANK_TOP_K
    if effective_category:
        dynamic_rerank_top_k = max(settings.RERANK_TOP_K, 50)
        if self._normalize_category_name(str(effective_category or "")) in {"resume", "paper"}:
            dynamic_rerank_top_k = max(dynamic_rerank_top_k, 80)
    if query_anchor and (
        is_lookup_heavy_query(query_anchor)
        or bool(_extract_identifier_filename_anchors(query_anchor))
    ):
        dynamic_rerank_top_k = max(dynamic_rerank_top_k, 50)
        
    _file_rerank_t0 = time.time()
    logger.info(
        "[dispatch_stage] file_rerank_start query=%r merged_candidates=%s top_k=%s",
        retrieval_query,
        len(results or []),
        dynamic_rerank_top_k,
    )
    reranked_normal = kb.rerank(retrieval_query, results, top_k=dynamic_rerank_top_k)
    _log_dispatch_stage(
        "file_rerank_done",
        _file_rerank_t0,
        input_candidates=len(results or []),
        reranked_candidates=len(reranked_normal or []),
    )

    best_by_file: Dict[str, Dict[str, Any]] = {}
    
    for d in results:
        is_lex_hit = d.get("_is_lexical_hit", False)
        is_personal_attr_entity_hit = d.get("_is_personal_attr_entity_hit", False)
        direct_confidence = int(d.get("_direct_score", 0))
        if (is_lex_hit and direct_confidence >= 90) or (is_personal_attr_entity_hit and direct_confidence >= 70):
            fp = d.get("file_path") or ""
            if fp and fp not in best_by_file:
                d["rerank_score"] = max(d.get("rerank_score", 0), float(settings.RELEVANCE_THRESHOLD) + 1.0)
                best_by_file[fp] = d

    explicit_data_extension_query = bool(
        lexical_extensions
        and any(
            (str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}")
            in data_like_exts
            for ext in (lexical_extensions or [])
        )
        and self._normalize_category_name(str(explicit_category or effective_category or "")) == "data"
    )
    if explicit_data_extension_query:
        preserved = 0
        for d in list(filename_route_results or []) + list(results or []):
            fp = str(d.get("file_path") or "")
            if not fp:
                continue
            if os.path.splitext(fp.lower())[1] not in data_like_exts:
                continue
            boosted = dict(d)
            boosted["_is_lexical_hit"] = True
            boosted["_direct_score"] = max(int(boosted.get("_direct_score", 0) or 0), 70)
            boosted["rerank_score"] = max(
                float(boosted.get("rerank_score", 0.0) or 0.0),
                float(settings.RELEVANCE_THRESHOLD) + 0.05,
            )
            boosted["_data_extension_inventory_rescue"] = True
            prev = best_by_file.get(fp)
            if prev is None or float(boosted.get("rerank_score", 0.0) or 0.0) > float(prev.get("rerank_score", 0.0) or 0.0):
                best_by_file[fp] = boosted
                preserved += 1
        if preserved:
            logger.info(
                "[dispatch] preserved %d data extension inventory candidate(s) through rerank for query=%r",
                preserved,
                retrieval_query,
            )

    for d in reranked_normal:
        is_lex_hit = d.get("_is_lexical_hit", False)
        is_personal_attr_entity_hit = d.get("_is_personal_attr_entity_hit", False)
        direct_confidence = int(d.get("_direct_score", 0))
        
        if (is_lex_hit and direct_confidence >= 90) or (is_personal_attr_entity_hit and direct_confidence >= 70):
            pass
        elif is_lex_hit and float(d.get("_bm25_score", 0) or 0) > 1.0:
            if float(d.get("rerank_score", 0.0) or 0.0) <= -2.0:
                continue
        elif float(d.get("rerank_score", 0.0) or 0.0) < float(settings.RELEVANCE_THRESHOLD):
            _rescue_q = str(retrieval_query or search_need_text or "").strip()
            if _rescue_q:
                _meta_blob = " ".join([
                    str(d.get("lookup_aliases") or ""),
                    str(d.get("file_name_en") or ""),
                    str(d.get("folder_name_en") or ""),
                ])
                _meta_overlap = compute_lookup_overlap_score(_rescue_q, _meta_blob)
                _broad_overlap = 0
                _broad_exact = False
                if _meta_overlap <= 0 and (
                    is_lookup_heavy_query(_rescue_q)
                    or bool(_extract_identifier_filename_anchors(_rescue_q))
                ):
                    _broad_blob = " ".join(
                        [
                            str(d.get("file_name") or ""),
                            str(d.get("file_path") or ""),
                            str(d.get("lookup_aliases") or ""),
                            str(d.get("file_name_en") or ""),
                            str(d.get("folder_name_en") or ""),
                            str(d.get("doc_summary") or ""),
                            str(d.get("table_schema_hint") or ""),
                            str(d.get("text") or "")[:1200],
                        ]
                    )
                    _broad_exact, _broad_overlap = lookup_match_quality(_rescue_q, _broad_blob)
                if _meta_overlap > 0 or _broad_exact or _broad_overlap > 0:
                    d["rerank_score"] = max(
                        float(d.get("rerank_score", 0.0) or 0.0),
                        float(settings.RELEVANCE_THRESHOLD) + 0.01,
                    )
                    if _meta_overlap > 0:
                        d["_metadata_rescue"] = True
                    else:
                        d["_structured_anchor_rescue"] = True
                else:
                    continue
            else:
                continue

        fp = d.get("file_path") or ""
        if not fp:
            continue
        prev = best_by_file.get(fp)
        if (prev is None) or (d.get("rerank_score", 0) > prev.get("rerank_score", 0)):
            best_by_file[fp] = d
    scoped_category = self._normalize_category_name(
        str(explicit_category or effective_category or "other")
    )
    if query_anchor and reranked_normal and scoped_category in {"image", "audio", "video", "audio/video"}:
        topic_ranked = sort_candidates_by_topic_overlap(reranked_normal, query_anchor)
        topic_positive = [
            src
            for src in topic_ranked
            if bool(src.get("_topic_lookup_exact")) or int(src.get("_topic_lookup_overlap", 0) or 0) > 0
        ]
        if topic_positive:
            rescued = 0
            rescue_cap = max(8, min(len(topic_positive), dynamic_rerank_top_k))
            for src in topic_positive[:rescue_cap]:
                fp = str(src.get("file_path") or "")
                if not fp or fp in best_by_file:
                    continue
                boosted = dict(src)
                boosted["rerank_score"] = max(
                    float(boosted.get("rerank_score", 0.0) or 0.0),
                    float(settings.RELEVANCE_THRESHOLD) + 0.01,
                )
                boosted["_topic_rescue"] = True
                best_by_file[fp] = boosted
                rescued += 1
            if rescued:
                logger.info(
                    "[dispatch] rescued %d %s candidate(s) via topic overlap: query=%r",
                    rescued,
                    scoped_category,
                    query_anchor,
                )

    if query_anchor and reranked_normal and scoped_category in document_evidence_categories:
        topic_ranked = sort_candidates_by_topic_overlap(reranked_normal, query_anchor)
        doc_rescued = 0
        doc_rescue_cap = max(8, min(len(topic_ranked), dynamic_rerank_top_k))
        doc_bm25_max = max(
            (
                float(src.get("_bm25_score", 0.0) or 0.0)
                for src in topic_ranked[:doc_rescue_cap]
                if _is_document_evidence_source(src)
            ),
            default=0.0,
        )
        doc_bm25_floor = max(20.0, min(75.0, doc_bm25_max * 0.25))
        for src in topic_ranked[:doc_rescue_cap]:
            fp = str(src.get("file_path") or "")
            if not fp or fp in best_by_file or not _is_document_evidence_source(src):
                continue
            affinity_blob = " ".join(
                [
                    str(src.get("file_name") or ""),
                    fp,
                    str(src.get("file_name_en") or ""),
                    str(src.get("folder_name_en") or ""),
                    str(src.get("lookup_aliases") or ""),
                    str(src.get("doc_summary") or ""),
                    str(src.get("table_schema_hint") or ""),
                    str(src.get("text") or "")[:800],
                ]
            )
            match_exact, lookup_overlap = lookup_match_quality(query_anchor, affinity_blob)
            topic_overlap = compute_lookup_overlap_score(query_anchor, affinity_blob)
            bm25_score = float(src.get("_bm25_score", 0.0) or 0.0)
            direct_score = int(src.get("_direct_score", 0) or 0)
            topic_exact = bool(src.get("_topic_lookup_exact"))
            focus_hits = int(src.get("_topic_lookup_focus_hits", 0) or 0)
            strong_doc_signal = bool(
                match_exact
                or topic_exact
                or lookup_overlap >= 2
                or topic_overlap >= 2
                or focus_hits > 0
                or (bm25_score >= doc_bm25_floor and topic_overlap >= 1)
                or (direct_score >= 80 and topic_overlap >= 1)
            )
            if not strong_doc_signal:
                continue
            boosted = dict(src)
            boosted["rerank_score"] = max(
                float(boosted.get("rerank_score", 0.0) or 0.0),
                float(settings.RELEVANCE_THRESHOLD) + 0.01,
            )
            boosted["_document_evidence_rescue"] = True
            best_by_file[fp] = boosted
            doc_rescued += 1
        if doc_rescued:
            logger.info(
                "[dispatch] rescued %d high-evidence %s document candidate(s) through rerank for query=%r",
                doc_rescued,
                scoped_category,
                query_anchor,
            )

    sources_file = sorted(best_by_file.values(), key=lambda x: x.get("rerank_score", 0), reverse=True)
    yield {
        "type": "trace_append",
        "item": {
            "stage": "file_rerank",
            "type": "retrieval",
            "elapsed_ms": _elapsed_ms(),
            "duration_ms": int((time.time() - _file_rerank_t0) * 1000),
            "reranked_candidates": len(reranked_normal or []),
            "threshold_pass_files": len(sources_file),
            "threshold": float(settings.RELEVANCE_THRESHOLD),
        },
    }

    if explicit_category and explicit_category != "other":
        from core.retrieval.category_engine import get_compatible_categories

        compatible_categories = get_compatible_categories(explicit_category) or {explicit_category}
        strict_category_sources = [
            d for d in sources_file
            if (
                self._normalize_category_name(str(d.get("doc_category") or "other")) in compatible_categories
            )
        ]
        if strict_category_sources:
            sources_file = strict_category_sources
        else:
            bm25_hits_in_sources = [
                d for d in sources_file
                if d.get("_is_lexical_hit") or float(d.get("_bm25_score", 0) or 0) > 0
            ]
            if bm25_hits_in_sources:
                logger.info(
                    f"[strict_category] explicit_category={explicit_category} 无匹配，"
                    f"但发现 {len(bm25_hits_in_sources)} 条 BM25 命中，自动放宽类别约束"
                )
                sources_file = bm25_hits_in_sources
            else:
                logger.info(
                    f"[strict_category] explicit_category={explicit_category} 但无匹配文件，"
                    f"清空 sources_file（原 {len(sources_file)} 条全为其他类别）"
                )
                sources_file = []

    if sources_file:
        constrained_sources, constrained = apply_media_query_constraints(
            sources_file,
            (retrieval_query or search_need_text or keywords or "").strip(),
        ) if not document_retrieval_media_topic else (sources_file, False)
        if constrained:
            logger.info(
                "[dispatch] applied media query constraints: before=%s after=%s query=%r",
                len(sources_file),
                len(constrained_sources),
                (retrieval_query or search_need_text or keywords or "").strip(),
            )
            sources_file = constrained_sources

        sources_file = _sort_sources_by_lookup_overlap(
            sources_file,
            (retrieval_query or search_need_text or keywords or "").strip(),
        )
        if query_anchor and not category_inventory_query and scoped_category not in {"", "other"}:
            narrowed_sources, narrowed = narrow_candidates_by_topic_overlap(
                sources_file,
                query_anchor,
                require_topic=True,
            )
            if narrowed:
                logger.info(
                    "[dispatch] narrowed category-scoped candidates by topic overlap: before=%s after=%s category=%r query=%r",
                    len(sources_file),
                    len(narrowed_sources),
                    scoped_category,
                    query_anchor,
                )
                sources_file = narrowed_sources

    logger.info(
        f"==========> [DEBUG] file-chain after threshold filter: {len(sources_file)} | "
        f"folder_chain_files={len(sources_folder_chain)}"
    )
    if sources_file:
        logger.info(
            f"==========> [DEBUG] file-chain threshold sample: "
            f"{[(s.get('rerank_score'), s.get('file_path')) for s in sources_file[:5]]}"
        )

    if str(os.getenv("FILEAGENT_SEARCH_SOURCES_DEBUG", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        logger.info(f"[search-sources-debug] RELEVANCE_THRESHOLD={settings.RELEVANCE_THRESHOLD}")
        for i, d in enumerate(sources_file[:20]):
            logger.info(
                f"[search-sources-debug] file-chain #{i} rerank_score={d.get('rerank_score')} "
                f"file={os.path.basename(str(d.get('file_path') or d.get('file_name') or ''))!r}"
            )
        logger.info(f"[search-sources-debug] file_chain_count={len(sources_file)}")

    if not sources_file and not sources_folder_chain:
        logger.info("[DEBUG] No sources found after applying threshold filtering.")
        self._set_followup_hint(
            session_id,
            action="process_previous",
            params={
                "allow_without_results": True,
                "anchor": "search_topic",
                "prior_search_query": search_need_text,
            },
            ttl_turns=1,
            uses=1,
        )
        self._clear_count_scope_context(session_id, reason="search_no_sources")
        yield from _emit_status("thinking", "No highly relevant indexed content found; answering directly...")
        yield from _emit_files_from_sources([])
        llm = self._get_llm_service(detailed=False, session_id=session_id, prompt_language=user_lang)
        scope_note = ""
        if active_paths:
            scope_note = (
                f"\nCurrent retrieval scope is limited to {len(active_paths)} active source file(s). "
                "If the request already names a topic or file type, mention broadening the source scope instead of asking another generic clarification."
            )
        
        prompt_str = (
            f"You are a helpful file retrieval assistant.\n"
            f"The user searched for: '{search_need_text}'\n"
            f"However, no highly relevant content was found in the indexed files.\n"
            f"Please reply clearly and nicely in {response_language_label}, stating that no relevant files were found. "
            "If the user already supplied a topic or file type, do not ask what kind they mean; suggest broader source scope, different keywords, or checking indexing."
            f"{scope_note}"
        )
        
        resp_text = yield from _collect_or_emit_stream(llm, prompt_str)
        if resp_text is None:
            return
        try:
            self._get_history_ref(session_id).append({"q": q, "a": resp_text})
        except Exception:
            pass
        yield _emit_timing_trace("request_done", action=action, query_type="search", sources_count=0)
        yield {"type": "done", "ok": True, "query_type": "search", "sources": [], "trace": []}
        return

    if self.is_aborted(session_id):
        yield {"type": "done", "ok": False, "query_type": "interrupted", "message": "生成已中断"}
        return
        
    is_data_query = effective_category == "data" or (
        sources_file and str(sources_file[0].get("file_path", "")).lower().endswith((".csv", ".tsv", ".xlsx", ".xls", ".numbers"))
    )
    is_lexical_hit = sources_file and any(d.get("_is_lexical_hit") for d in sources_file)
    personal_attribute_route = str((params or {}).get("_expert_route") or "").strip() == "personal_attribute"
    resolved_entity_hint = str((params or {}).get("_resolved_entity") or "").strip()
    resolved_attribute_hint = str((params or {}).get("_resolved_attribute") or "").strip()
    if personal_attribute_route:
        logger.info(
            "[dispatch] personal_attribute refine entry: candidates=%s entity=%r attribute=%r",
            len(sources_file or []),
            resolved_entity_hint,
            resolved_attribute_hint,
        )
    lexical_lookup_query = is_lookup_heavy_query((retrieval_query or search_need_text or "").strip())
    lexical_filename_style_query = bool(
        explicit_filename_lookup
        or lexical_lookup_query
        or lexical_filenames
        or lexical_extensions
    )
    strict_lexical_refine_query = bool(
        explicit_file_ref
        or explicit_filename_lookup
        or has_plausible_filename_extension(
            (retrieval_query or lexical_query_text or search_need_text or q).strip()
        )
        or bool(
            _extract_identifier_filename_anchors(
                (retrieval_query or lexical_query_text or search_need_text or q).strip()
            )
        )
        or (
            lexical_lookup_query
            and not effective_category
            and not category_inventory_query
        )
    )
    def _is_data_like_source(src: Dict[str, Any]) -> bool:
        fp = str(src.get("file_path") or "").lower()
        ext = os.path.splitext(fp)[1].lower()
        if ext in data_like_exts:
            return True
        cat = self._normalize_category_name(str(src.get("doc_category") or "other"))
        return cat == "data"

    def _is_media_like_source(src: Dict[str, Any]) -> bool:
        fp = str(src.get("file_path") or "").lower()
        ext = os.path.splitext(fp)[1].lower()
        if ext in {
            ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg",
            ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
        }:
            return True
        cat = self._normalize_category_name(str(src.get("doc_category") or "other"))
        media_type = self._normalize_category_name(
            str((src.get("metadata") or {}).get("media_type") or "")
        )
        return cat in {"audio", "video", "audio/video"} or media_type in {"audio", "video", "audio/video"}

    def _inject_data_lookup_candidates(
        query_text: str,
        current_sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not query_text:
            return current_sources

        by_path: Dict[str, Dict[str, Any]] = {}
        for src in current_sources:
            fp = str(src.get("file_path") or "")
            if fp:
                by_path[fp] = src

        try:
            indexed_hits = kb.indexed_keyword_search(
                query_text,
                allowed_paths=active_paths,
                file_extensions=sorted(data_like_exts),
                limit=24,
            ) if hasattr(kb, "indexed_keyword_search") else []
        except Exception as exc:
            logger.warning("[dispatch] indexed data lookup failed: %s", exc)
            indexed_hits = []

        for hit in indexed_hits:
            meta = dict(hit.get("metadata") or {})
            fp = str(hit.get("file_path") or meta.get("file_path") or "")
            if not fp:
                continue
            ext = os.path.splitext(fp)[1].lower()
            if ext not in data_like_exts:
                continue
            match_exact, overlap = lookup_match_quality(
                query_text,
                " ".join(
                    [
                        str(meta.get("file_name") or ""),
                        str(meta.get("file_name_en") or ""),
                        str(meta.get("parent_folder") or ""),
                        str(meta.get("folder_name_en") or ""),
                        str(meta.get("lookup_aliases") or ""),
                        str(meta.get("table_schema_hint") or ""),
                        str(hit.get("doc_summary") or ""),
                        fp,
                    ]
                ),
            )
            bm25_score = float(hit.get("_bm25_score", 0.0) or 0.0)
            if not match_exact and overlap <= 0 and bm25_score <= 0:
                continue
            prev = by_path.get(fp)
            candidate = {
                "text": str(hit.get("text") or meta.get("doc_summary") or ""),
                "metadata": meta,
                "distance": 0.0,
                "file_name": hit.get("file_name") or meta.get("file_name", os.path.basename(fp)),
                "file_path": fp,
                "doc_summary": hit.get("doc_summary") or meta.get("doc_summary", ""),
                "doc_category": hit.get("doc_category") or meta.get("doc_category", "data"),
                "lookup_aliases": hit.get("lookup_aliases") or meta.get("lookup_aliases", ""),
                "score": 1.0,
                "_is_lexical_hit": True,
                "_lexical_filename_exact": bool(match_exact),
                "_direct_score": 100 if match_exact else max(70, int(overlap * 10)),
                "_bm25_score": max(bm25_score, 12.0, float(overlap * 5)),
            }
            if prev is None or int(candidate.get("_direct_score", 0)) > int(prev.get("_direct_score", 0) or 0):
                by_path[fp] = candidate

        merged = list(by_path.values())
        merged.sort(
            key=lambda x: (
                int(x.get("_direct_score", 0) or 0),
                compute_lookup_overlap_score(query_text, " ".join([
                    str(x.get("file_name") or ""),
                    str(x.get("file_path") or ""),
                    str(x.get("lookup_aliases") or ""),
                    str(x.get("doc_summary") or ""),
                    str(x.get("table_schema_hint") or ""),
                ])),
                float(x.get("_bm25_score", 0.0) or 0.0),
                float(x.get("rerank_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return merged

    def _select_indexed_data_lookup_sources(
        query_text: str,
        current_sources: List[Dict[str, Any]],
        *,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        if not current_sources:
            return []

        ranked = _sort_sources_by_lookup_overlap(current_sources, query_text)
        anchored: List[Dict[str, Any]] = []
        for src in ranked:
            overlap = int(src.get("_topic_lookup_overlap", 0) or 0)
            focus_hits = int(src.get("_topic_lookup_focus_hits", 0) or 0)
            direct_score = int(src.get("_direct_score", 0) or 0)
            bm25_score = float(src.get("_bm25_score", 0.0) or 0.0)
            if (
                bool(src.get("_topic_lookup_exact"))
                or bool(src.get("_lexical_filename_exact"))
                or focus_hits > 0
                or overlap > 0
                or direct_score >= 70
                or bm25_score >= 25.0
            ):
                anchored.append(src)

        return (anchored or ranked)[: max(1, limit)]

    def _collect_strong_lookup_anchor_sources(
        query_text: str,
        current_sources: List[Dict[str, Any]],
        *,
        limit: int = 6,
    ) -> List[Dict[str, Any]]:
        """Keep title/filename anchors that are stronger than an LLM miss.

        The LLM refine step is allowed to narrow broad semantic recall, but it
        should not drop a candidate whose filename/path/aliases/summary strongly
        match a title-like or filename-like user query. This is deliberately
        evidence-based instead of phrase-specific: require a high lexical signal
        plus distinctive-term overlap, not just generic words such as document or
        plan.
        """
        anchor = str(query_text or "").strip()
        if not anchor or not current_sources:
            return []
        generic_terms = {
            "find", "search", "show", "list", "display", "retrieve", "locate",
            "file", "files", "document", "documents", "doc", "docs",
            "report", "reports", "plan", "plans", "paper", "papers",
            "image", "images", "photo", "photos", "video", "videos",
            "audio", "data", "table", "tables", "all", "my", "the",
            "a", "an", "of", "for", "with", "about", "instead",
            "start", "over", "ignore", "previous",
        }
        distinctive_terms = [
            term
            for term in extract_lookup_terms(anchor, max_terms=64)
            if len(str(term or "")) >= 2 and str(term or "").lower() not in generic_terms
        ]
        simple_distinctive_terms: List[str] = []
        for term in distinctive_terms:
            term_s = str(term or "").strip().lower()
            if not term_s or term_s in generic_terms:
                continue
            # Multi-token phrases are useful for lookup overlap, but single
            # anchors give a more stable signal for title/name preservation.
            if " " in term_s and not any("\u4e00" <= ch <= "\u9fff" for ch in term_s):
                continue
            if term_s not in simple_distinctive_terms:
                simple_distinctive_terms.append(term_s)
        min_term_hits = max(1, min(2, len(simple_distinctive_terms)))
        min_overlap = 2 if len(distinctive_terms) <= 2 else min(4, len(distinctive_terms))
        rows: List[Tuple[Dict[str, Any], str, str]] = []
        term_df: Dict[str, int] = {}
        for src in current_sources:
            fp = str(src.get("file_path") or "").strip()
            file_blob = " ".join(
                [
                    str(src.get("file_name") or ""),
                    os.path.basename(fp) if fp else "",
                    fp,
                    str(src.get("file_name_en") or ""),
                    str((src.get("metadata") or {}).get("file_name_en") or ""),
                    str(src.get("folder_name_en") or ""),
                    str((src.get("metadata") or {}).get("folder_name_en") or ""),
                    str(src.get("lookup_aliases") or ""),
                    str((src.get("metadata") or {}).get("lookup_aliases") or ""),
                    str(src.get("en_tags") or ""),
                    str((src.get("metadata") or {}).get("en_tags") or ""),
                ]
            )
            affinity_blob = " ".join(
                [
                    file_blob,
                    str(src.get("doc_summary") or ""),
                    str(src.get("table_schema_hint") or ""),
                    str(src.get("text") or "")[:600],
                ]
            )
            affinity_lower = affinity_blob.lower()
            file_lower = file_blob.lower()
            rows.append((src, affinity_lower, file_lower))
            for term in simple_distinctive_terms:
                if term and term in affinity_lower:
                    term_df[term] = term_df.get(term, 0) + 1

        rare_df_cutoff = max(2, min(4, max(1, len(rows) // 5)))
        ranked: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        seen_paths: set[str] = set()
        for src, affinity_lower, file_lower in rows:
            fp = str(src.get("file_path") or "").strip()
            if fp and fp in seen_paths:
                continue
            file_blob = file_lower
            affinity_blob = affinity_lower
            exact_match, lookup_overlap = lookup_match_quality(anchor, affinity_blob)
            filename_overlap = compute_lookup_overlap_score(anchor, file_blob)
            affinity_overlap = compute_lookup_overlap_score(anchor, affinity_blob)
            strongest_overlap = max(int(lookup_overlap), int(filename_overlap), int(affinity_overlap))
            direct_score = int(src.get("_direct_score", 0) or 0)
            bm25_score = float(src.get("_bm25_score", 0.0) or 0.0)
            rerank_score = float(src.get("rerank_score", 0.0) or 0.0)
            lexical_exact = bool(src.get("_lexical_filename_exact") or src.get("_lookup_match_exact"))
            term_hits = 0
            filename_term_hits = 0
            rare_term_hits = 0
            rare_filename_term_hits = 0
            for term in simple_distinctive_terms:
                if not term:
                    continue
                if term in affinity_lower:
                    term_hits += 1
                    if term_df.get(term, 0) <= rare_df_cutoff:
                        rare_term_hits += 1
                if term in file_lower:
                    filename_term_hits += 1
                    if term_df.get(term, 0) <= rare_df_cutoff:
                        rare_filename_term_hits += 1

            strong_anchor = bool(
                lexical_exact
                or (exact_match and strongest_overlap >= 2)
                or (rare_filename_term_hits > 0 and (bm25_score >= 12.0 or direct_score >= 70 or rerank_score >= 1.0))
                or (rare_term_hits > 0 and bm25_score >= 20.0)
                or (direct_score >= 95 and strongest_overlap >= min_overlap)
                or (bm25_score >= 80.0 and strongest_overlap >= min_overlap)
                or (bm25_score >= 20.0 and term_hits >= min_term_hits)
                or (direct_score >= 90 and term_hits >= min_term_hits)
                or (rerank_score >= 4.0 and strongest_overlap >= min_overlap)
            )
            if not strong_anchor:
                continue
            if fp:
                seen_paths.add(fp)
            ranked.append(
                (
                    (
                        rare_filename_term_hits,
                        rare_term_hits,
                        term_hits,
                        filename_term_hits,
                        1 if lexical_exact else 0,
                        1 if exact_match else 0,
                        strongest_overlap,
                        direct_score,
                        bm25_score,
                        rerank_score,
                    ),
                    src,
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [src for _, src in ranked[: max(1, limit)]]

    def _merge_preserved_lookup_anchors(
        refined_sources: List[Dict[str, Any]],
        current_sources: List[Dict[str, Any]],
        query_text: str,
    ) -> List[Dict[str, Any]]:
        query_anchor = str(query_text or "").strip()
        if not query_anchor:
            return refined_sources
        if not (
            is_lookup_heavy_query(query_anchor)
            or bool(_extract_identifier_filename_anchors(query_anchor))
            or has_plausible_filename_extension(query_anchor)
        ):
            return refined_sources
        preserved = _collect_strong_lookup_anchor_sources(query_text, current_sources)
        if not preserved:
            return refined_sources
        by_path: Dict[str, Dict[str, Any]] = {}
        anonymous: List[Dict[str, Any]] = []
        for src in list(refined_sources or []):
            fp = str(src.get("file_path") or "").strip()
            if fp:
                by_path[fp] = src
            else:
                anonymous.append(src)
        added = 0
        for src in preserved:
            fp = str(src.get("file_path") or "").strip()
            if fp:
                if fp in by_path:
                    continue
                by_path[fp] = src
            else:
                anonymous.append(src)
            added += 1
        if added:
            logger.info(
                "[dispatch] preserved %d strong lookup/title anchor candidate(s) after LLM refine for query=%r",
                added,
                query_text,
            )
        return list(by_path.values()) + anonymous

    def _merge_preserved_resume_profile_sources(
        refined_sources: List[Dict[str, Any]],
        current_sources: List[Dict[str, Any]],
        query_text: str,
        *,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        query_anchor = str(query_text or "").strip()
        if not query_anchor or not current_sources:
            return refined_sources
        resume_query = bool(
            scoped_category == "resume"
            or re.search(r"\b(?:resume|resumes|cv|cvs|candidate|candidates|profile|profiles)\b", query_anchor, re.IGNORECASE)
            or any(token in query_anchor for token in ("简历", "履历", "候选人"))
        )
        if not resume_query:
            return refined_sources
        broad_resume_inventory_query = bool(
            re.search(r"\b(?:resumes|candidates|profiles|cvs)\b", query_anchor, re.IGNORECASE)
            or any(token in query_anchor for token in ("简历", "候选人"))
        )
        preserve_limit = max(limit, 18 if broad_resume_inventory_query else 10)
        profile_like_sources = _filter_resume_profile_noise(current_sources, query_text=query_anchor)
        preserved = _collect_resume_profile_anchor_sources(
            profile_like_sources or current_sources,
            query_anchor,
            limit=preserve_limit,
        )
        if not preserved:
            return refined_sources
        by_path: Dict[str, Dict[str, Any]] = {}
        anonymous: List[Dict[str, Any]] = []
        for src in list(refined_sources or []):
            fp = str(src.get("file_path") or "").strip()
            if fp:
                by_path[fp] = src
            else:
                anonymous.append(src)
        added = 0
        for src in preserved:
            fp = str(src.get("file_path") or "").strip()
            if fp:
                if fp in by_path:
                    continue
                by_path[fp] = src
            else:
                anonymous.append(src)
            added += 1
        if added:
            logger.info(
                "[dispatch] preserved %d resume/profile anchor candidate(s) after LLM refine for query=%r",
                added,
                query_text,
            )
        merged_sources = list(by_path.values()) + anonymous
        if broad_resume_inventory_query:
            merged_sources = _sort_sources_by_lookup_overlap(merged_sources, query_anchor)
        return merged_sources

    def _merge_preserved_data_evidence_sources(
        refined_sources: List[Dict[str, Any]],
        current_sources: List[Dict[str, Any]],
        query_text: str,
        *,
        limit: int = 6,
    ) -> List[Dict[str, Any]]:
        """Keep high-evidence table/CSV candidates that LLM refine can under-select.

        This is intentionally evidence-based: it only fires for data/metric/review
        style questions or when the refined set already contains data files, and it
        requires strong lexical/BM25/topic overlap from the indexed metadata/chunks.
        """
        query_anchor = str(query_text or "").strip()
        if not query_anchor or not current_sources:
            return refined_sources

        data_question = bool(
            re.search(
                r"\b(?:data|dataset|datasets|table|tables|spreadsheet|csv|excel|benchmark|metric|metrics|"
                r"performance|rating|ratings|review|reviews|feedback|score|scores|comparison|compare|"
                r"spec|specs|specification|statistics|stats)\b"
                r"|数据|数据集|表格|数据表|指标|性能|算力|评分|评价|评论|反馈|对比|评测|参数|规格|统计",
                query_anchor,
                re.IGNORECASE,
            )
        )
        refined_has_data = any(_is_data_like_source(src) for src in refined_sources or [])
        if not data_question and not refined_has_data:
            return refined_sources

        existing_paths = {
            str(src.get("file_path") or "").strip()
            for src in refined_sources or []
            if str(src.get("file_path") or "").strip()
        }
        existing_stems = {
            compact_filename_key(
                os.path.splitext(
                    os.path.basename(str(src.get("file_name") or src.get("file_path") or ""))
                )[0]
            )
            for src in refined_sources or []
        }
        existing_stems.discard("")
        data_candidates = [src for src in current_sources or [] if _is_data_like_source(src)]
        if not data_candidates:
            return refined_sources

        max_bm25 = max((float(src.get("_bm25_score", 0.0) or 0.0) for src in data_candidates), default=0.0)
        bm25_floor = max(25.0, min(80.0, max_bm25 * 0.30))
        ranked: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        seen_paths: set[str] = set()
        seen_stems: set[str] = set()

        for src in data_candidates:
            fp = str(src.get("file_path") or "").strip()
            if not fp or fp in existing_paths or fp in seen_paths:
                continue
            stem_key = compact_filename_key(
                os.path.splitext(
                    os.path.basename(str(src.get("file_name") or fp))
                )[0]
            )
            if stem_key and (stem_key in existing_stems or stem_key in seen_stems):
                continue
            affinity_blob = " ".join(
                [
                    str(src.get("file_name") or ""),
                    fp,
                    str(src.get("lookup_aliases") or ""),
                    str(src.get("doc_summary") or ""),
                    str(src.get("table_schema_hint") or ""),
                    str(src.get("text") or "")[:1200],
                ]
            )
            match_exact, lookup_overlap = lookup_match_quality(query_anchor, affinity_blob)
            topic_overlap = compute_lookup_overlap_score(query_anchor, affinity_blob)
            bm25_score = float(src.get("_bm25_score", 0.0) or 0.0)
            direct_score = int(src.get("_direct_score", 0) or 0)
            rerank_score = float(src.get("rerank_score", 0.0) or 0.0)
            topic_exact = bool(src.get("_topic_lookup_exact"))
            focus_hits = int(src.get("_topic_lookup_focus_hits", 0) or 0)
            strong_signal = bool(
                match_exact
                or topic_exact
                or lookup_overlap >= 2
                or topic_overlap >= 2
                or focus_hits > 0
                or (bm25_score >= bm25_floor and topic_overlap >= 1)
                or (data_question and bm25_score >= max(80.0, bm25_floor))
                or (refined_has_data and bm25_score >= bm25_floor and direct_score >= 70)
            )
            if not strong_signal:
                continue
            seen_paths.add(fp)
            if stem_key:
                seen_stems.add(stem_key)
            ranked.append(
                (
                    (
                        1 if match_exact else 0,
                        1 if topic_exact else 0,
                        int(lookup_overlap),
                        int(topic_overlap),
                        focus_hits,
                        bm25_score,
                        direct_score,
                        rerank_score,
                    ),
                    src,
                )
            )

        if not ranked:
            return refined_sources

        ranked.sort(key=lambda item: item[0], reverse=True)
        additions = [src for _, src in ranked[: max(1, limit)]]
        logger.info(
            "[dispatch] preserved %d high-evidence data candidate(s) after LLM refine for query=%r",
            len(additions),
            query_text,
        )
        return list(refined_sources or []) + additions

    def _merge_preserved_document_evidence_sources(
        refined_sources: List[Dict[str, Any]],
        current_sources: List[Dict[str, Any]],
        query_text: str,
        *,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Keep high-evidence document candidates that LLM refine under-selects."""
        query_anchor = str(query_text or "").strip()
        if not query_anchor or not current_sources:
            return refined_sources

        scoped_document_query = bool(scoped_category in document_evidence_categories)
        document_question = bool(
            re.search(
                r"\b(?:invoice|invoices|receipt|receipts|bill|bills|contract|contracts|quotation|quotations|"
                r"quote|quotes|resume|resumes|cv|manual|manuals|report|reports|paper|papers|book|books|"
                r"document|documents|doc|docs|pdf|pdfs)\b"
                r"|发票|收据|账单|合同|报价单|报价|简历|履历|手册|说明书|报告|论文|书籍|文档|文件",
                query_anchor,
                re.IGNORECASE,
            )
        )
        refined_has_document = any(_is_document_evidence_source(src) for src in refined_sources or [])
        if not scoped_document_query and not document_question:
            return refined_sources

        existing_paths = {
            str(src.get("file_path") or "").strip()
            for src in refined_sources or []
            if str(src.get("file_path") or "").strip()
        }
        document_candidates = [
            src for src in current_sources or []
            if _is_document_evidence_source(src)
        ]
        if not document_candidates:
            return refined_sources

        max_bm25 = max((float(src.get("_bm25_score", 0.0) or 0.0) for src in document_candidates), default=0.0)
        bm25_floor = max(20.0, min(80.0, max_bm25 * 0.25))
        ranked: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        seen_paths: set[str] = set()

        for src in document_candidates:
            fp = str(src.get("file_path") or "").strip()
            if not fp or fp in existing_paths or fp in seen_paths:
                continue
            meta = dict(src.get("metadata") or {})
            affinity_blob = " ".join(
                str(part or "")
                for part in [
                    src.get("file_name"),
                    fp,
                    src.get("file_name_en"),
                    meta.get("file_name_en"),
                    src.get("folder_name_en"),
                    meta.get("folder_name_en"),
                    src.get("lookup_aliases"),
                    meta.get("lookup_aliases"),
                    src.get("doc_summary"),
                    meta.get("doc_summary"),
                    src.get("table_schema_hint"),
                    src.get("text"),
                ]
                if str(part or "").strip()
            )
            if not affinity_blob:
                continue
            match_exact, lookup_overlap = lookup_match_quality(query_anchor, affinity_blob)
            topic_overlap = compute_lookup_overlap_score(query_anchor, affinity_blob)
            bm25_score = float(src.get("_bm25_score", 0.0) or 0.0)
            direct_score = int(src.get("_direct_score", 0) or 0)
            rerank_score = float(src.get("rerank_score", 0.0) or 0.0)
            topic_exact = bool(src.get("_topic_lookup_exact"))
            focus_hits = int(src.get("_topic_lookup_focus_hits", 0) or 0)
            lexical_exact = bool(src.get("_lexical_filename_exact") or src.get("_lookup_match_exact"))
            same_scope_category = bool(
                scoped_category in document_evidence_categories
                and self._normalize_category_name(
                    str(src.get("doc_category_family") or src.get("doc_category") or "other")
                ) == scoped_category
            )
            strong_signal = bool(
                match_exact
                or lexical_exact
                or topic_exact
                or lookup_overlap >= 2
                or topic_overlap >= 2
                or focus_hits > 0
                or (bm25_score >= bm25_floor and topic_overlap >= 1)
                or (same_scope_category and bm25_score >= max(45.0, bm25_floor))
                or (refined_has_document and direct_score >= 80 and topic_overlap >= 1)
            )
            if not strong_signal:
                continue
            seen_paths.add(fp)
            preserved = dict(src)
            preserved["_document_evidence_preserved"] = True
            ranked.append(
                (
                    (
                        1 if match_exact else 0,
                        1 if lexical_exact else 0,
                        1 if topic_exact else 0,
                        int(lookup_overlap),
                        int(topic_overlap),
                        focus_hits,
                        1 if same_scope_category else 0,
                        bm25_score,
                        direct_score,
                        rerank_score,
                    ),
                    preserved,
                )
            )

        if not ranked:
            return refined_sources

        ranked.sort(key=lambda item: item[0], reverse=True)
        additions = [src for _, src in ranked[: max(1, limit)]]
        logger.info(
            "[dispatch] preserved %d high-evidence document candidate(s) after LLM refine for query=%r",
            len(additions),
            query_text,
        )
        return list(refined_sources or []) + additions

    def _merge_preserved_media_evidence_sources(
        refined_sources: List[Dict[str, Any]],
        current_sources: List[Dict[str, Any]],
        query_text: str,
        *,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Keep high-evidence media candidates that LLM refine under-selects.

        This is intentionally topic-evidence based: it only applies to media
        candidates and requires the query's media topic terms to appear in
        filename/path/summary/text evidence.
        """
        query_anchor = str(query_text or "").strip()
        if not query_anchor or not current_sources:
            return refined_sources

        try:
            from core.kb.knowledge_base import FileKnowledgeBase, _tokenize_for_bm25

            query_tokens = _tokenize_for_bm25(query_anchor)
            media_terms, expanded_terms, wants_media = FileKnowledgeBase._media_content_query_terms(
                query_anchor,
                query_tokens,
                category_filter="",
                file_extensions=None,
            )
        except Exception:
            media_terms = [
                term for term in extract_lookup_terms(query_anchor, max_terms=32)
                if len(str(term or "")) >= 3
            ]
            expanded_terms = []
            wants_media = bool(re.search(r"\b(?:audio|video|recording|sound|clip)\b|音频|视频|录音|声音", query_anchor, re.IGNORECASE))

        if not wants_media and not any(_is_media_like_source(src) for src in current_sources[:12]):
            return refined_sources

        focus_terms = [
            str(term or "").strip().lower()
            for term in list(media_terms or [])
            if len(str(term or "").strip()) >= 2
        ]
        expansion_terms = [
            str(term or "").strip().lower()
            for term in list(expanded_terms or [])
            if len(str(term or "").strip()) >= 3
        ]
        if not focus_terms and not expansion_terms:
            return refined_sources

        def _count_media_term_hits(terms: List[str], text_lower: str) -> int:
            if not terms or not text_lower:
                return 0
            latin_tokens = set(re.findall(r"[a-z0-9]+", text_lower))
            hits = 0
            for raw_term in terms:
                term = str(raw_term or "").strip().lower()
                if not term:
                    continue
                if any("\u4e00" <= ch <= "\u9fff" for ch in term):
                    if term in text_lower:
                        hits += 1
                    continue
                # Short Latin terms such as "sea" and "eel" must match as
                # tokens; substring matching would otherwise hit unrelated
                # words like "seated", "several", or "search".
                if len(term) <= 3:
                    if term in latin_tokens:
                        hits += 1
                elif term in latin_tokens or term in text_lower:
                    hits += 1
            return hits

        existing_paths = {
            str(src.get("file_path") or "").strip()
            for src in refined_sources or []
            if str(src.get("file_path") or "").strip()
        }
        media_candidates = [src for src in current_sources or [] if _is_media_like_source(src)]
        if not media_candidates:
            return refined_sources

        ranked: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        seen_paths: set[str] = set()
        for src in media_candidates:
            fp = str(src.get("file_path") or "").strip()
            if not fp or fp in existing_paths or fp in seen_paths:
                continue
            meta = dict(src.get("metadata") or {})
            label_blob = " ".join(
                str(part or "")
                for part in [
                    src.get("file_name"),
                    fp,
                    src.get("file_name_en"),
                    meta.get("file_name_en"),
                    src.get("folder_name_en"),
                    meta.get("folder_name_en"),
                    src.get("lookup_aliases"),
                    meta.get("lookup_aliases"),
                    src.get("doc_summary"),
                    meta.get("doc_summary"),
                ]
                if str(part or "").strip()
            ).lower()
            evidence_blob = " ".join(
                str(part or "")
                for part in [
                    label_blob,
                    src.get("text"),
                ]
                if str(part or "").strip()
            ).lower()
            if not evidence_blob:
                continue

            label_focus_hits = _count_media_term_hits(focus_terms, label_blob)
            label_expanded_hits = _count_media_term_hits(expansion_terms, label_blob)
            focus_hits = _count_media_term_hits(focus_terms, evidence_blob)
            expanded_hits = _count_media_term_hits(expansion_terms, evidence_blob)
            direct_score = int(src.get("_direct_score", 0) or 0)
            bm25_score = float(src.get("_bm25_score", 0.0) or 0.0)
            rerank_score = float(src.get("rerank_score", 0.0) or 0.0)
            topic_overlap = int(src.get("_topic_lookup_overlap", 0) or 0)

            strong_signal = bool(
                focus_hits >= 1
                or (expanded_hits >= 1 and (direct_score >= 80 or bm25_score >= 10.0 or rerank_score >= 0.5))
                or (expanded_hits >= 2 and (direct_score >= 70 or bm25_score >= 5.0))
                or (topic_overlap >= 2 and (direct_score >= 80 or bm25_score >= 10.0))
            )
            if not strong_signal:
                continue
            seen_paths.add(fp)
            preserved = dict(src)
            preserved["_media_evidence_preserved"] = True
            ranked.append(
                (
                    (
                        label_focus_hits,
                        label_expanded_hits,
                        focus_hits,
                        expanded_hits,
                        topic_overlap,
                        direct_score,
                        bm25_score,
                        rerank_score,
                    ),
                    preserved,
                )
            )

        if not ranked:
            return refined_sources

        ranked.sort(key=lambda item: item[0], reverse=True)
        additions = [src for _, src in ranked[: max(1, limit)]]
        logger.info(
            "[dispatch] preserved %d high-evidence media candidate(s) after LLM refine for query=%r",
            len(additions),
            query_text,
        )
        return list(refined_sources or []) + additions

    _llm_refine_t0 = time.time()
    refined_normal: List[Dict[str, Any]] = []
    sources_file_chain: List[Dict[str, Any]] = []
    exact_filename_hits: List[Dict[str, Any]] = []
    if sources_file:
        exact_media_file_hits: List[Dict[str, Any]] = []
        if personal_attribute_route:
            yield from _emit_status("thinking", "Keeping best matching profile/contact files...")
            sources_file_chain = _retain_personal_attribute_sources(
                sources_file,
                query_text=(retrieval_query or search_need_text or "").strip(),
                entity_hint=resolved_entity_hint,
                attribute_hint=resolved_attribute_hint,
                limit=5,
            )
        elif media_route and media_target_hint:
            hint_name = os.path.basename(media_target_hint).strip().lower()
            hint_stem = os.path.splitext(hint_name)[0]
            for d in sources_file:
                cand_name = str(d.get("file_name") or os.path.basename(str(d.get("file_path") or ""))).strip().lower()
                cand_stem = os.path.splitext(cand_name)[0]
                if cand_name and (cand_name == hint_name or cand_stem == hint_stem):
                    exact_media_file_hits.append(d)
        else:
            if category_inventory_query:
                yield from _emit_status("thinking", "Keeping category inventory results without extra LLM narrowing...")
                try:
                    category_inventory_limit = max(
                        1,
                        int(os.getenv("SEARCH_CATEGORY_INVENTORY_MAX_FILES", "50") or 50),
                    )
                except ValueError:
                    category_inventory_limit = 50
                refined_normal = list(sources_file[:category_inventory_limit])
                sources_file_chain = list(refined_normal or [])
            else:
                query_for_data = _build_lexical_query_text(
                    cleaned_kw or "",
                    keywords or "",
                    retrieval_query or "",
                    search_need_text or "",
                )
                if is_data_query:
                    data_like_sources = [src for src in sources_file if _is_data_like_source(src)]
                    if data_like_sources:
                        sources_file = data_like_sources
                    lexical_filename_reinjection_guard = bool(
                        explicit_file_ref
                        or has_plausible_filename_extension(query_for_data)
                        or lexical_extensions
                    )
                    if is_lexical_hit and lexical_filename_reinjection_guard:
                        logger.info(
                            "[dispatch] skip data metadata reinjection for lexical data lookup: query=%r candidates=%s",
                            query_for_data,
                            len(sources_file),
                        )
                        sources_file = _sort_sources_by_lookup_overlap(sources_file, query_for_data)
                    else:
                        logger.info(
                            "[dispatch] inject indexed data candidates: query=%r base_candidates=%s",
                            query_for_data,
                            len(sources_file),
                        )
                        sources_file = _inject_data_lookup_candidates(query_for_data, sources_file)
                        sources_file = _sort_sources_by_lookup_overlap(sources_file, query_for_data)

                exact_filename_hits = [
                    d for d in sources_file
                    if d.get("_is_lexical_hit")
                    and int(d.get("_direct_score", 0)) >= 90
                    and _query_contains_filename_needle(
                        (retrieval_query or lexical_query_text or search_need_text or "").strip(),
                        str(d.get("file_name") or d.get("file_path") or ""),
                    )
                ]
                focused_exact_hits: List[Dict[str, Any]] = []
                
                if exact_media_file_hits:
                    logger.info(
                        f"[DEBUG] Exact media target match found ({len(exact_media_file_hits)} files), "
                        "preserving matched media file while refining supporting context."
                    )
                    yield from _emit_status(
                        "thinking",
                        "Exact media file matched, preserving target file while refining context...",
                    )
                    refined_semantic = _refine_sources_with_llm(
                        q,
                        sources_file,
                        retrieval_query=retrieval_query,
                        effective_category=effective_category,
                        keyword_hint=keywords,
                    )
                    merged_by_path: Dict[str, Dict[str, Any]] = {}
                    for _src in exact_media_file_hits:
                        _fp = str(_src.get("file_path") or "")
                        if _fp:
                            merged_by_path[_fp] = _src
                    for _src in list(refined_semantic or []):
                        _fp = str(_src.get("file_path") or "")
                        if not _fp or _fp in merged_by_path:
                            continue
                        merged_by_path[_fp] = _src
                    refined_normal = list(merged_by_path.values())
                else:
                    focused_exact_hits = _collapse_exact_filename_focus_hits(exact_filename_hits)
                filename_prefilter_query = (retrieval_query or lexical_query_text or search_need_text or "").strip()
                exact_filename_prefilter = bool(
                    explicit_filename_lookup
                    and lexical_lookup_query
                    and has_plausible_filename_extension(filename_prefilter_query)
                )
                if exact_filename_hits and exact_filename_prefilter:
                    logger.info(
                        "[DEBUG] Exact filename match found (%d chunk/source hits -> %d focused files), "
                        "using focused filename matches without widening from semantic context.",
                        len(exact_filename_hits),
                        len(focused_exact_hits),
                    )
                    yield from _emit_status(
                        "thinking",
                        "Exact filename match found, using matched file(s) only...",
                    )
                    refined_normal = focused_exact_hits or exact_filename_hits
                elif is_lexical_hit and strict_lexical_refine_query:
                    lexical_query_anchor = (retrieval_query or search_need_text or query_for_data or q).strip()
                    lexical_candidates = [
                        d for d in sources_file 
                        if d.get("_is_lexical_hit")
                    ]
                    if lexical_query_anchor and (
                        is_lookup_heavy_query(lexical_query_anchor)
                        or bool(_extract_identifier_filename_anchors(lexical_query_anchor))
                    ):
                        lexical_by_path: Dict[str, Dict[str, Any]] = {}
                        lexical_anon: List[Dict[str, Any]] = []

                        def _register_lexical_candidate(src: Dict[str, Any]) -> None:
                            fp = str(src.get("file_path") or "").strip()
                            if fp:
                                lexical_by_path.setdefault(fp, src)
                            else:
                                lexical_anon.append(src)

                        for src in lexical_candidates:
                            _register_lexical_candidate(src)

                        anchored_added = 0
                        for src in sources_file:
                            fp = str(src.get("file_path") or "").strip()
                            if fp and fp in lexical_by_path:
                                continue
                            affinity_blob = " ".join(
                                [
                                    str(src.get("file_name") or ""),
                                    str(src.get("file_path") or ""),
                                    str(src.get("lookup_aliases") or ""),
                                    str(src.get("doc_summary") or ""),
                                    str(src.get("table_schema_hint") or ""),
                                    str(src.get("text") or "")[:1200],
                                ]
                            )
                            exact, overlap = lookup_match_quality(lexical_query_anchor, affinity_blob)
                            topic_overlap = compute_lookup_overlap_score(lexical_query_anchor, affinity_blob)
                            if not (exact or overlap > 0 or topic_overlap > 0 or src.get("_structured_anchor_rescue")):
                                continue
                            _register_lexical_candidate(src)
                            anchored_added += 1

                        if anchored_added:
                            logger.info(
                                "[dispatch] expanded lexical refine window with %d anchored candidate(s) for lookup-heavy query=%r",
                                anchored_added,
                                lexical_query_anchor,
                            )
                        lexical_candidates = list(lexical_by_path.values()) + lexical_anon
                    if lexical_candidates:
                        if is_data_query:
                            logger.info(
                                "[dispatch] lexical data lookup fast path: keeping anchored indexed table files. query=%r candidates=%s",
                                query_for_data,
                                len(lexical_candidates),
                            )
                            yield from _emit_status(
                                "thinking",
                                "Data file query detected, keeping strongest indexed table files...",
                            )
                            refined_normal = _select_indexed_data_lookup_sources(
                                query_for_data,
                                lexical_candidates,
                                limit=24,
                            )
                        else:
                            logger.info(f"[DEBUG] Lexical hit fallback, refining {len(lexical_candidates)} files with LLM...")
                            yield from _emit_status("thinking", "No exact match, refining similar files with AI...")
                            def _lexical_refine_sort_key(src: Dict[str, Any]) -> tuple:
                                affinity_blob = " ".join(
                                    [
                                        str(src.get("file_name") or ""),
                                        str(src.get("file_path") or ""),
                                        str(src.get("lookup_aliases") or ""),
                                        str(src.get("doc_summary") or ""),
                                        str(src.get("table_schema_hint") or ""),
                                    ]
                                )
                                exact, overlap = lookup_match_quality(lexical_query_anchor, affinity_blob)
                                return (
                                    1 if exact else 0,
                                    int(overlap),
                                    compute_lookup_overlap_score(lexical_query_anchor, affinity_blob),
                                    1 if src.get("_lexical_filename_exact") else 0,
                                    int(src.get("_direct_score", 0) or 0),
                                    float(src.get("_bm25_score", 0.0) or 0.0),
                                    float(src.get("rerank_score", 0.0) or 0.0),
                                )
                            lexical_candidates.sort(
                                key=_lexical_refine_sort_key,
                                reverse=True
                            )
                            refined_normal = _refine_sources_with_llm(
                                q,
                                lexical_candidates,
                                retrieval_query=retrieval_query,
                                effective_category=effective_category,
                                keyword_hint=keywords,
                                is_lexical_fallback=True,
                            )
                    else:
                        yield from _emit_status("thinking", "No highly relevant exact match, refining others...")
                        refined_normal = _refine_sources_with_llm(
                            q,
                            sources_file,
                            retrieval_query=retrieval_query,
                            effective_category=effective_category,
                            keyword_hint=keywords,
                        )
                elif is_lexical_hit:
                    logger.info(
                        "[DEBUG] Broad semantic query has lexical-scored hits; "
                        "keeping full reranked candidate set for LLM refine."
                    )
                    yield from _emit_status("thinking", "Refining semantic search results with lexical support...")
                    refined_normal = _refine_sources_with_llm(
                        q,
                        sources_file,
                        retrieval_query=retrieval_query,
                        effective_category=effective_category,
                        keyword_hint=keywords,
                        is_lexical_fallback=True,
                    )
                else:
                    if is_data_query:
                        query_for_data = (retrieval_query or search_need_text or "").strip()
                        strong_data_lookup = any(
                            compute_lookup_overlap_score(
                                query_for_data,
                                " ".join(
                                    [
                                        str(s.get("file_name") or ""),
                                        str(s.get("file_path") or ""),
                                        str(s.get("lookup_aliases") or ""),
                                        str(s.get("doc_summary") or ""),
                                    ]
                                ),
                            ) >= 2
                            for s in sources_file[:20]
                        )
                        if strong_data_lookup:
                            logger.info(f"[DEBUG] Bypassing LLM refinement with strong data lookup anchors. is_data_query={is_data_query}")
                            yield from _emit_status("thinking", "Data file query detected, keeping strongest matching table files...")
                            anchored_data_sources = []
                            for src in sources_file:
                                affinity_blob = " ".join(
                                    [
                                        str(src.get("file_name") or ""),
                                        str(src.get("file_path") or ""),
                                        str(src.get("lookup_aliases") or ""),
                                        str(src.get("doc_summary") or ""),
                                        str(src.get("table_schema_hint") or ""),
                                    ]
                                )
                                overlap = compute_lookup_overlap_score(query_for_data, affinity_blob)
                                bm25_score = float(src.get("_bm25_score", 0.0) or 0.0)
                                direct_score = int(src.get("_direct_score", 0) or 0)
                                if overlap > 0 or direct_score >= 70 or bm25_score >= 25.0:
                                    anchored_data_sources.append(src)
                            refined_normal = (anchored_data_sources or sources_file)[:10]
                        else:
                            yield from _emit_status("thinking", "Refining data file matches...")
                            refined_normal = _refine_sources_with_llm(
                                q,
                                sources_file,
                                retrieval_query=retrieval_query,
                                effective_category=effective_category,
                                keyword_hint=keywords,
                            )
                    else:
                        yield from _emit_status("thinking", "Refining file search results (LLM)...")
                        refined_normal = _refine_sources_with_llm(
                            q,
                            sources_file,
                            retrieval_query=retrieval_query,
                            effective_category=effective_category,
                            keyword_hint=keywords,
                        )
                preservation_query_text = (retrieval_query or lexical_query_text or search_need_text or q).strip()
                if document_retrieval_media_topic:
                    preservation_query_text = _build_lexical_query_text(
                        search_need_text,
                        q,
                        lexical_query_text,
                        retrieval_query,
                    )
                if not category_inventory_query and not personal_attribute_route:
                    refined_normal = _merge_preserved_lookup_anchors(
                        refined_normal,
                        sources_file,
                        preservation_query_text,
                    )
                    refined_normal = _merge_preserved_data_evidence_sources(
                        refined_normal,
                        sources_file,
                        preservation_query_text,
                    )
                    refined_normal = _merge_preserved_document_evidence_sources(
                        refined_normal,
                        sources_file,
                        preservation_query_text,
                    )
                    refined_normal = _merge_preserved_resume_profile_sources(
                        refined_normal,
                        sources_file,
                        preservation_query_text,
                    )
                    refined_normal = _merge_preserved_media_evidence_sources(
                        refined_normal,
                        sources_file,
                        preservation_query_text,
                    )
                sources_file_chain = list(refined_normal or [])
            if not sources_file_chain and sources_file:
                query_anchor = (retrieval_query or search_need_text or "").strip()
                anchored_rescue: List[Dict[str, Any]] = []
                for src in sources_file:
                    affinity_blob = " ".join(
                        [
                            str(src.get("file_name") or ""),
                            str(src.get("file_path") or ""),
                            str(src.get("lookup_aliases") or ""),
                            str(src.get("doc_summary") or ""),
                            str(src.get("table_schema_hint") or ""),
                        ]
                    )
                    overlap = compute_lookup_overlap_score(query_anchor, affinity_blob)
                    match_exact, lookup_overlap = lookup_match_quality(query_anchor, affinity_blob)
                    bm25_score = float(src.get("_bm25_score", 0.0) or 0.0)
                    direct_score = int(src.get("_direct_score", 0) or 0)
                    if lexical_filename_style_query or lexical_lookup_query:
                        should_rescue = bool(
                            match_exact
                            or lookup_overlap >= 2
                            or overlap >= 2
                            or src.get("_lexical_filename_exact")
                        )
                    else:
                        should_rescue = bool(
                            overlap >= 2
                            or bm25_score >= 35.0
                            or direct_score >= 70
                        )
                    if should_rescue:
                        rescued = dict(src)
                        rescued["_lookup_overlap"] = max(int(overlap), int(lookup_overlap))
                        anchored_rescue.append(rescued)
                if anchored_rescue:
                    anchored_rescue.sort(
                        key=lambda x: (
                            int(x.get("_lookup_overlap", 0) or 0),
                            float(x.get("_bm25_score", 0.0) or 0.0),
                            int(x.get("_direct_score", 0) or 0),
                            float(x.get("rerank_score", 0.0) or 0.0),
                        ),
                        reverse=True,
                    )
                    sources_file_chain = anchored_rescue[:5]
                    logger.info(
                        "[dispatch] LLM refine returned empty; rescued %d anchored candidates",
                        len(sources_file_chain),
                    )
    yield {
        "type": "trace_append",
        "item": {
            "stage": "llm_refine",
            "type": "retrieval",
            "elapsed_ms": _elapsed_ms(),
            "duration_ms": int((time.time() - _llm_refine_t0) * 1000),
            "input_file_candidates": len(sources_file),
            "output_file_candidates": len(sources_file_chain),
            "used_data_query_bypass": bool(is_data_query and not is_lexical_hit and sources_file),
            "used_lexical_path": bool(is_lexical_hit),
        },
    }
    _paths_in_folder_chain = {
        str(d.get("file_path") or "") for d in sources_folder_chain if d.get("file_path")
    }
    exact_filename_route_hit = any(
        int(s.get("_direct_score", 0) or 0) >= 90
        or bool(s.get("_lexical_filename_exact"))
        for s in (filename_route_results or [])
    )
    filename_like_query = bool(
        explicit_file_ref
        or lexical_filenames
        or has_plausible_filename_extension(str(query_for_search or q or "").strip())
    )
    strong_filename_hit = bool(
        active_paths and filename_like_query and exact_filename_route_hit
    )
    if (
        strong_filename_hit
        and sources_folder_chain
        and not direct_folder_seed_sources
        and not (folder_listing_route or folder_filter)
    ):
        logger.info(
            "[dispatch] suppressing %d folder-chain entries due to strong filename hit",
            len(sources_folder_chain),
        )
        sources_folder_chain = []
        folder_cards = []
        _paths_in_folder_chain = set()
    if _should_suppress_folder_chain_for_refined_file_chain(
        sources_file_chain=sources_file_chain,
        sources_folder_chain=sources_folder_chain,
        direct_folder_seed_sources=direct_folder_seed_sources,
        folder_listing_route=folder_listing_route,
        folder_filter=folder_filter,
        category_inventory_mode=category_inventory_mode,
        query_text=search_need_text or retrieval_query or query_for_search or q or "",
        folder_cards=folder_cards,
    ):
        logger.info(
            "[dispatch] suppressing %d folder-chain entries because LLM refine selected %d file-chain entries",
            len(sources_folder_chain),
            len({str(s.get("file_path") or "") for s in sources_file_chain if str(s.get("file_path") or "").strip()}),
        )
        sources_folder_chain = []
        folder_cards = []
        _paths_in_folder_chain = set()
    sources_file_chain_dedup = [
        s
        for s in sources_file_chain
        if str(s.get("file_path") or "") not in _paths_in_folder_chain
    ]
    explicit_filename_mode = bool(
        explicit_file_ref
        or strong_filename_hit
        or lexical_filenames
    )
    sources: List[Dict[str, Any]] = _compose_search_sources_for_display(
        folder_cards=folder_cards,
        sources_folder_chain=sources_folder_chain,
        sources_file_chain=sources_file_chain_dedup,
        direct_folder_seed_sources=direct_folder_seed_sources,
        folder_listing_route=folder_listing_route,
        folder_filter=folder_filter,
        explicit_filename_mode=explicit_filename_mode,
        query_text=search_need_text or retrieval_query or query_for_search or q or "",
    )
    try:
        from core.intent.entity_experts import CategoryListExpert

        negated_categories: set[str] = set()
        source_text_for_negation = search_need_text or retrieval_query or query_for_search or q or ""
        for cat_name in _CATEGORY_COMPATIBLE_EXTS:
            aliases = CategoryListExpert._category_aliases(cat_name)
            if any(CategoryListExpert._category_token_is_negated(source_text_for_negation, alias) for alias in aliases):
                negated_categories.add(cat_name)
        if negated_categories:
            negated_exts = {
                str(ext).lower()
                for cat_name in negated_categories
                for ext in (_CATEGORY_COMPATIBLE_EXTS.get(cat_name, set()) or set())
            }

            def _is_negated_source(src: Dict[str, Any]) -> bool:
                fp = str(src.get("file_path") or src.get("file_name") or "").strip()
                ext = os.path.splitext(fp)[1].lower()
                cat = self._normalize_category_name(str(src.get("doc_category") or "other"))
                return bool(cat in negated_categories or (ext and ext in negated_exts))

            before_negation_filter = len(sources)
            sources = [src for src in sources if not _is_negated_source(src)]
            if len(sources) != before_negation_filter:
                logger.info(
                    "[dispatch] filtered %d source(s) by negated category constraint: %s",
                    before_negation_filter - len(sources),
                    sorted(negated_categories),
                )
    except Exception:
        pass
    logger.info(
        "[dispatch] merge: folder_cards=%d, folder_chain=%d, file_chain=%d, total=%d",
        len(folder_cards), len(sources_folder_chain), len(sources_file_chain_dedup), len(sources)
    )
    yield {
        "type": "trace_append",
        "item": {
            "stage": "final_sources",
            "type": "retrieval",
            "elapsed_ms": _elapsed_ms(),
            "duration_ms": 0,
            "folder_chain_files": len(sources_folder_chain),
            "file_chain_files": len(sources_file_chain_dedup),
            "display_total": len(sources),
            "top_files": [
                os.path.basename(str(s.get("file_path") or s.get("file_name") or ""))
                for s in sources[:5]
            ],
        },
    }

    logger.info(
        f"==========> [DEBUG] after llm refine: folder_chain_files={len(sources_folder_chain)}, "
        f"file_chain={len(sources_file_chain_dedup)} (raw_file_refine={len(sources_file_chain)}), "
        f"display_total={len(sources)}"
    )
    if sources:
        logger.info(
            f"==========> [DEBUG] display sample: {[(s.get('rerank_score'), s.get('file_path')) for s in sources[:5]]}"
        )

    if not sources:
        logger.info("[DEBUG] No sources found after LLM refinement step.")
        self._set_followup_hint(
            session_id,
            action="process_previous",
            params={
                "allow_without_results": True,
                "anchor": "search_topic",
                "prior_search_query": search_need_text,
            },
            ttl_turns=1,
            uses=1,
        )
        self._clear_count_scope_context(session_id, reason="search_llm_refine_no_sources")
        
        yield from _emit_status("thinking", "No directly relevant files remained after refinement.")
        yield from _emit_files_from_sources([])
        llm = self._get_llm_service(detailed=False, session_id=session_id, prompt_language=user_lang)
        scope_note = ""
        if active_paths:
            scope_note = (
                f"\nCurrent retrieval scope is limited to {len(active_paths)} active source file(s). "
                "If the request already names a topic or file type, mention broadening the source scope instead of asking another generic clarification."
            )

        prompt_str = (
            f"You are a helpful file retrieval assistant.\n"
            f"The user searched for: '{search_need_text}'\n"
            f"The system tried to find related files, but after filtering, no highly relevant files matched the question.\n"
            f"In 2-3 short sentences in {response_language_label}, please clearly state that no indexed files matched the query, and suggest trying different keywords or checking the indexing scope. "
            "If the user already supplied a topic or file type, do not ask what kind they mean. Do not use bullet points or mention internal steps."
            f"{scope_note}"
        )

        resp_text = yield from _collect_or_emit_stream(llm, prompt_str)
        if resp_text is None:
            return
        try:
            self._get_history_ref(session_id).append({"q": q, "a": resp_text})
        except Exception:
            pass
        yield _emit_timing_trace("request_done", action=action, query_type="search", sources_count=0)
        yield {"type": "done", "ok": True, "query_type": "search", "sources": [], "trace": []}
        return

    self._clear_count_scope_context(session_id, reason="search_results_updated")
    self._set_last_search_results(session_id, sources[:50])
    self._set_followup_hint(
        session_id,
        action="process_previous",
        params={},
        ttl_turns=2,
        uses=2,
    )

    try:
        _media_ctx_cap = max(
            1,
            int(str(os.getenv("FILEAGENT_SEARCH_MEDIA_CONTEXT_MAX_FILES", "3") or "3").strip() or "3"),
        )
    except ValueError:
        _media_ctx_cap = 3

    try:
        _media_ctx_chars = max(
            1200,
            int(str(os.getenv("FILEAGENT_SEARCH_MEDIA_CONTEXT_MAX_CHARS", "6000") or "6000").strip() or "6000"),
        )
    except ValueError:
        _media_ctx_chars = 6000

    _search_media_exts = {
        ".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".ts",
        ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
    }

    def _build_media_search_context(file_path: str) -> str:
        try:
            all_chunks = kb.collection.get(
                where={"file_path": file_path},
                include=["documents", "metadatas"],
            )
            meta_lines: List[str] = []
            summary_line = ""
            keyframe_entries: List[Tuple[float, str]] = []
            asr_entries: List[str] = []
            for chunk_doc, chunk_meta in zip(
                all_chunks.get("documents") or [],
                all_chunks.get("metadatas") or [],
            ):
                ctype = str((chunk_meta or {}).get("chunk_type") or "")
                chunk_doc = str(chunk_doc or "")
                if ctype == "media_summary":
                    for line in chunk_doc.splitlines():
                        if line.startswith("内容摘要:"):
                            summary_line = line[len("内容摘要:"):].strip()
                        elif line.lower().startswith("content summary:"):
                            summary_line = line.split(":", 1)[-1].strip()
                        elif line.strip():
                            meta_lines.append(line.strip())
                elif ctype == "keyframe":
                    try:
                        t = float((chunk_meta or {}).get("keyframe_time_sec", 0) or 0)
                    except Exception:
                        t = 0.0
                    keyframe_entries.append((t, chunk_doc.strip()))
                elif ctype == "asr_transcript":
                    asr_entries.append(chunk_doc.strip())

            keyframe_entries.sort(key=lambda item: item[0])
            parts: List[str] = []
            if summary_line:
                parts.append(f"[Media Summary]\\n{summary_line}")
            if meta_lines:
                parts.append("[Media Metadata]\\n" + "\\n".join(meta_lines))
            if keyframe_entries:
                parts.append(
                    "[Visual Timeline]\\n" + "\\n\\n".join(
                        f"[{t:.0f}s] {desc}" for t, desc in keyframe_entries if desc
                    )
                )
            if asr_entries:
                parts.append("[Speech Transcript Segments]\\n" + "\\n".join(seg for seg in asr_entries if seg))

            merged = "\\n\\n".join(part for part in parts if part.strip())
            return merged[:_media_ctx_chars].strip()
        except Exception as exc:
            logger.warning("[search_media_context] failed for %s: %s", os.path.basename(file_path), exc)
            return ""

    _hydrated_media = 0
    for _src in sources:
        if _hydrated_media >= _media_ctx_cap:
            break
        if _src.get("is_folder_chain_match"):
            continue
        _fp = str(_src.get("file_path") or "").strip()
        _ext = os.path.splitext(_fp)[1].lower()
        _doc_cat = str(_src.get("doc_category") or "").strip().lower()
        _media_type = str(_src.get("media_type") or "").strip().lower()
        if _ext not in _search_media_exts and _doc_cat != "audio/video" and _media_type not in {"audio", "video"}:
            continue
        _media_ctx = _build_media_search_context(_fp)
        if not _media_ctx:
            continue
        _orig_summary = str(_src.get("doc_summary") or "").strip()
        if _orig_summary and _orig_summary not in _media_ctx:
            _src["text"] = f"{_media_ctx}\\n\\n[Indexed Summary]\\n{_orig_summary}"
        else:
            _src["text"] = _media_ctx
        _hydrated_media += 1
    if _hydrated_media:
        logger.info("[search_media_context] hydrated %d media source(s) for answer generation", _hydrated_media)

    yield from _emit_files_from_sources(sources)
    _q_lower_for_listing = str(search_need_text or q or "").strip().lower()
    _listing_only_search_query = bool(
        action == "search"
        and not personal_attribute_route
        and not media_route
        and not self._looks_like_file_content_analysis_query(search_need_text, prompt_language=user_lang)
        and (
            generic_inventory_query
            or
            re.search(r"\b(?:find|search|show|list|display|browse|look\s+for)\b", _q_lower_for_listing)
            or any(tok in search_need_text for tok in ("找", "查", "搜索", "查找", "显示", "列出", "看看", "给我看", "查看"))
        )
    )

    if category_inventory_query or _listing_only_search_query:
        inventory_answer = _format_listing_found_answer(sources, user_lang=user_lang)
        yield {"type": "text", "delta": inventory_answer}
        try:
            self._get_history_ref(session_id)[-1]["a"] = inventory_answer.strip()
        except Exception:
            pass
        yield _emit_timing_trace("request_done", action=action, query_type="search", sources_count=len(sources))
        yield {"type": "done", "ok": True, "query_type": "search", "sources": sources, "trace": []}
        return

    if not _search_ack_emitted:
        _search_ack_emitted = True
        yield from _emit_status(
            "thinking",
            "Files found. Preparing a single final answer..."
            if user_lang != "zh"
            else "已找到相关文件，正在整理最终答案...",
        )

    if str(os.getenv("FILEAGENT_TEST_FAST_RESPONSE", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        _answer_t0 = time.time()
        quick_answer = f"Found {len(sources)} relevant files."
        try:
            self._get_history_ref(session_id)[-1]["a"] = quick_answer
        except Exception:
            pass
        yield {
            "type": "trace_append",
            "item": {
                "stage": "answer_generation",
                "type": "response",
                "elapsed_ms": _elapsed_ms(),
                "duration_ms": int((time.time() - _answer_t0) * 1000),
                "mode": "fast_response",
                "answer_length": len(quick_answer),
            },
        }
        yield {"type": "text", "delta": quick_answer}
        post_answer_focus_sources = _post_answer_filename_focus_sources(
            sources,
            query_text=(q or retrieval_query or lexical_query_text or search_need_text).strip(),
            answer_text=quick_answer,
        )
        if post_answer_focus_sources:
            logger.info(
                "[dispatch] post-answer relevant-files focus: %d -> %d for query=%r",
                len(sources),
                len(post_answer_focus_sources),
                q,
            )
            sources = post_answer_focus_sources
            self._set_last_search_results(session_id, sources[:50])
            yield {
                "type": "trace_append",
                "item": {
                    "stage": "post_answer_relevant_files_focus",
                    "type": "files",
                    "display_total": len(sources),
                },
            }
            yield from _emit_files_from_sources(
                sources,
                total_matches=len(sources),
                shown_count=len(sources),
            )
        yield _emit_timing_trace("request_done", action=action, query_type="search", sources_count=len(sources))
        yield _emit_timing_trace("request_done", action=action, query_type="search", sources_count=len(sources))
        yield {"type": "done", "ok": True, "query_type": "search", "sources": sources, "trace": []}
        return

    yield from _emit_status("thinking", "Generating answer...")
    _answer_t0 = time.time()
    
    _is_fallback_hit = False
    if not exact_filename_hits and is_lexical_hit and sources_file and len(sources_file) > 0:
        _is_fallback_hit = True

    yield {"type": "thinking", "delta": f"Retained {len(sources)} refined files; generating answer from content...\n"}
    ctx = self._build_context(session_id)
    clickable_file_lines = _render_clickable_file_lines(sources)
    try:
        _snippet_content_cap = max(
            800,
            int(str(os.getenv("SEARCH_ANSWER_CONTENT_SNIPPET_CHARS", "32000") or "32000").strip() or "32000"),
        )
    except ValueError:
        _snippet_content_cap = 32000

    if personal_attribute_route:
        # For contact/profile lookups, summaries alone are often too lossy.
        # Hydrate a few top files with raw content so the answer model can extract
        # explicit values such as email / phone / school from the original text.
        hydrated = 0
        for _src in sources:
            if hydrated >= 3:
                break
            if _src.get("is_folder_chain_match"):
                continue
            file_path = str(_src.get("file_path") or "").strip()
            if not file_path:
                continue
            try:
                raw_content = str(self._read_file_content(file_path) or "").strip()
            except Exception:
                raw_content = ""
            if not raw_content or raw_content.startswith("无法读取") or raw_content.startswith("读取失败"):
                continue
            _src["text"] = raw_content[:5000]
            if not str(_src.get("doc_summary") or "").strip():
                _src["doc_summary"] = raw_content[:800]
            hydrated += 1

    folder_snippets: List[str] = []
    file_snippets: List[str] = []

    folder_chain_for_tree = [d for d in sources[:50] if d.get("is_folder_chain_match")]
    if folder_chain_for_tree:
        by_root: Dict[str, List[str]] = {}
        for d in folder_chain_for_tree:
            r = str(d.get("folder_chain_root") or "").strip()
            rel = str(d.get("folder_chain_relative_path") or "").replace("\\", "/").strip()
            if not rel:
                rel = os.path.basename(str(d.get("file_path") or ""))
            if r:
                by_root.setdefault(r, []).append(rel)
        tree_lines = ["[Indexed directory layout — relative paths under matched folder roots]"]
        for r in sorted(by_root.keys()):
            tree_lines.append(f"▸ {r}")
            for rel in sorted(set(by_root[r])):
                tree_lines.append(f"    · {rel}")
        folder_snippets.append("\n".join(tree_lines))

    file_chain_for_snippets = sources
    for i, d in enumerate(file_chain_for_snippets[:30], 1):
        raw_sum = str(d.get("doc_summary", ""))
        clean_sum = raw_sum.replace("```", "") if raw_sum else ""
        if len(clean_sum) > _snippet_content_cap:
            truncated_sum = clean_sum[: max(0, _snippet_content_cap - 1)]
            match_sum = re.search(r'^(.*[.,;!?\s\u4e00-\u9fa5])[^\s\u4e00-\u9fa5]*$', truncated_sum)
            if match_sum:
                clean_sum = match_sum.group(1).rstrip() + "..."
            else:
                clean_sum = truncated_sum.rstrip() + "..."

        _txt = str(d.get("text", "") or "")
        if len(_txt) > _snippet_content_cap:
            truncated = _txt[: max(0, _snippet_content_cap - 1)]
            match = re.search(r'^(.*[.,;!?\s\u4e00-\u9fa5])[^\s\u4e00-\u9fa5]*$', truncated)
            if match:
                _txt = match.group(1).rstrip() + "..."
            else:
                _txt = truncated.rstrip() + "..."

        file_snippets.append(
            f"[{i}] File: {d.get('file_name','')} (Path: {d.get('file_path', '')} )\n"
            f"Summary: {clean_sum}\n"
            f"Content: {_txt}\n"
        )

    llm = self._get_llm_service(detailed=True, session_id=session_id, prompt_language=user_lang)
    combined_answer = ""

    def _extract_explicit_personal_attribute() -> Optional[Dict[str, Any]]:
        attr_key = str(resolved_attribute_hint or "").lower().strip()
        if not attr_key:
            return None
        attr_patterns = {
            "email": [r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"],
            "phone": [
                r"(?<!\d)(1[3-9]\d{9})(?!\d)",
                r"(?<!\d)(?:\+?\d[\d\-\s()]{7,}\d)(?!\d)",
            ],
            "wechat": [
                r"(?:wechat|weixin|微信(?:号)?|vx)\s*[:：]?\s*([A-Za-z][A-Za-z0-9_-]{4,}|[A-Za-z0-9_-]{5,})"
            ],
            "linkedin": [r"https?://(?:www\.)?linkedin\.com/[^\s)>\"]+"],
            "address": [r"(?:address|location|地址|所在地)\s*[:：]?\s*([^\n]{4,120})"],
            "location": [r"(?:address|location|地址|所在地)\s*[:：]?\s*([^\n]{4,120})"],
        }
        patterns = attr_patterns.get(attr_key) or []
        if not patterns:
            return None

        for idx, src in enumerate(sources, 1):
            blob = "\n".join(
                [
                    str(src.get("text") or ""),
                    str(src.get("doc_summary") or ""),
                ]
            ).strip()
            if not blob:
                continue
            for pattern in patterns:
                m = re.search(pattern, blob, flags=re.IGNORECASE)
                if not m:
                    continue
                if m.lastindex:
                    value = str(m.group(m.lastindex) or "").strip()
                else:
                    value = str(m.group(0) or "").strip()
                if attr_key == "phone":
                    value = re.sub(r"\s+", "", value)
                value = value.strip(" \t\r\n,;:：")
                if value:
                    return {
                        "found": True,
                        "value": value,
                        "evidence_indices": [idx],
                    }
        return None

    if file_snippets:
        yield from _emit_status("thinking", "Summarizing files...")

        fallback_instruction = ""
        if _is_fallback_hit:
            fallback_instruction = (
                f"- The user was searching for a specific file (e.g. {q}), but an exact filename match could not be found.\n"
                "- Instead, the system found some partially matching or related files.\n"
                "- Briefly tell the user that you found the closest relevant files, then answer from those files only.\n"
            )

        evidence_sections: List[str] = []
        if folder_snippets:
            evidence_sections.append(
                "<Folder Retrieval Evidence>\n"
                + "\n---\n".join(folder_snippets)
                + "\n</Folder Retrieval Evidence>"
            )
        evidence_sections.append(
            "<Refined Retrieved Snippets>\n"
            + "\n---\n".join(file_snippets)
            + "\n</Refined Retrieved Snippets>"
        )
        evidence_block = "\n\n".join(evidence_sections)

        if personal_attribute_route:
            entity_label = resolved_entity_hint or "the person in the retrieved files"
            attribute_label = resolved_attribute_hint or "the requested contact/profile detail"
            direct_extract_enabled = str(
                os.getenv("FILEAGENT_PERSONAL_ATTR_DIRECT_EXTRACT", "0") or "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
            direct_hit = _extract_explicit_personal_attribute() if direct_extract_enabled else None
            if isinstance(direct_hit, dict) and direct_hit.get("found") and direct_hit.get("value"):
                attribute_value = str(direct_hit.get("value") or "").strip()
                evidence_indices = [
                    idx
                    for idx in list(direct_hit.get("evidence_indices") or [])
                    if isinstance(idx, int) and 1 <= idx <= len(sources)
                ]
                evidence_sources = [
                    sources[idx - 1]
                    for idx in evidence_indices
                    if 1 <= idx <= len(sources)
                ]
                evidence_name = str((evidence_sources[0].get("file_name") if evidence_sources else "") or "").strip()
                if response_language_label.lower().startswith("chinese") or response_language_label == "中文":
                    attr_label_out = {
                        "email": "邮箱",
                        "phone": "电话号码",
                        "address": "地址",
                        "location": "地址",
                        "school": "毕业院校",
                        "university": "毕业院校",
                        "wechat": "微信",
                        "linkedin": "LinkedIn",
                    }.get(attribute_label.lower(), attribute_label)
                    direct_answer = (
                        f"根据《{evidence_name}》，{entity_label}的{attr_label_out}是 `{attribute_value}`。"
                        if evidence_name
                        else f"{entity_label}的{attr_label_out}是 `{attribute_value}`。"
                    )
                else:
                    attr_label_out = {
                        "email": "email address",
                        "phone": "phone number",
                        "address": "address",
                        "location": "location",
                        "school": "school",
                        "university": "university",
                        "wechat": "WeChat ID",
                        "linkedin": "LinkedIn profile",
                    }.get(attribute_label.lower(), attribute_label)
                    direct_answer = (
                        f"According to {evidence_name}, the {attr_label_out} for {entity_label} is `{attribute_value}`."
                        if evidence_name
                        else f"The {attr_label_out} for {entity_label} is `{attribute_value}`."
                    )
                combined_answer += direct_answer
                try:
                    self._get_history_ref(session_id)[-1]["a"] = combined_answer
                except Exception:
                    pass
                yield {
                            "type": "trace_append",
                            "item": {
                                "stage": "answer_generation",
                                "type": "response",
                                "elapsed_ms": _elapsed_ms(),
                                "duration_ms": int((time.time() - _answer_t0) * 1000),
                                "mode": "personal_attribute_direct",
                                "answer_length": len(combined_answer or ""),
                                "folder_snippet_blocks": len(folder_snippets),
                                "file_snippet_blocks": len(file_snippets),
                            },
                        }
                yield {"type": "text", "delta": combined_answer}
                yield _emit_timing_trace("request_done", action=action, query_type="search", sources_count=len(sources))
                yield {"type": "done", "ok": True, "query_type": "search", "sources": sources, "trace": []}
                return

            file_prompt = (
                "You are a file retrieval assistant performing a targeted personal-attribute lookup.\n"
                f"- Reply in {response_language_label}.\n"
                f"- The target person is: {entity_label}\n"
                f"- The requested attribute is: {attribute_label}\n"
                "- Produce exactly one direct final answer for the user.\n"
                "- Read the retrieved evidence carefully and state the requested value in the first sentence if it is explicitly present.\n"
                "- Use the raw Content blocks as primary evidence. Summaries may help identify a file but must not be treated as the requested value.\n"
                "- Do not treat metadata, field-name lists, tags, or bracketed attribute lists as a value.\n"
                "- For address/location/home requests, extract the concrete place or address from the evidence; if no concrete place is present, say it is not found.\n"
                "- If the value is not explicitly present, say that clearly and briefly, then mention the most relevant supporting file or two.\n"
                "- Do not invent details not present in the evidence.\n"
                "- Do not repeat the full matched-file list because the UI already shows it.\n\n"
                + f"{ctx}\n"
                + evidence_block
                + "\n\n"
                + f"<User Question>\n{q}\n</User Question>\n\n"
                + f"【IMPORTANT】You MUST strictly answer in {response_language_label}. Even if the evidence contains Chinese or other languages, your response must be entirely in {response_language_label}."
            )
        else:
            file_prompt = (
                "You are a file retrieval assistant.\n"
                f"- Reply in {response_language_label}.\n"
                "- Produce exactly one final answer for the user from the retrieved evidence below.\n"
                "- Keep it concise and direct.\n"
                "- Mention only the 1-3 most relevant files when that helps answer the question.\n"
                "- Do not repeat the full matched-file list or directory tree because the UI already shows it.\n"
                "- If the evidence does not really answer the question, say that briefly and clearly.\n"
                "- Do not invent details not present in the evidence.\n"
                + fallback_instruction + "\n"
                + f"{ctx}\n"
                + evidence_block
                + "\n\n"
                + f"<User Question>\n{q}\n</User Question>\n\n"
                + f"【IMPORTANT】You MUST strictly answer in {response_language_label}. Even if the evidence contains Chinese or other languages, your response must be entirely in {response_language_label}."
            )
        file_answer = yield from _collect_or_emit_stream(llm, file_prompt)
        if file_answer:
            combined_answer += file_answer

    post_answer_focus_sources = _post_answer_filename_focus_sources(
        sources,
        query_text=(q or retrieval_query or lexical_query_text or search_need_text).strip(),
        answer_text=combined_answer,
    )
    if post_answer_focus_sources:
        logger.info(
            "[dispatch] post-answer relevant-files focus: %d -> %d for query=%r",
            len(sources),
            len(post_answer_focus_sources),
            q,
        )
        sources = post_answer_focus_sources
        self._set_last_search_results(session_id, sources[:50])
        yield {
            "type": "trace_append",
            "item": {
                "stage": "post_answer_relevant_files_focus",
                "type": "files",
                "display_total": len(sources),
            },
        }
        yield from _emit_files_from_sources(
            sources,
            total_matches=len(sources),
            shown_count=len(sources),
        )

    try:
        self._get_history_ref(session_id)[-1]["a"] = combined_answer
    except Exception:
        pass
    yield {
        "type": "trace_append",
        "item": {
            "stage": "answer_generation",
            "type": "response",
            "elapsed_ms": _elapsed_ms(),
            "duration_ms": int((time.time() - _answer_t0) * 1000),
            "mode": "full_response",
            "answer_length": len(combined_answer or ""),
            "folder_snippet_blocks": len(folder_snippets),
            "file_snippet_blocks": len(file_snippets),
        },
    }
    yield _emit_timing_trace("request_done", action=action, query_type="search", sources_count=len(sources))
    yield {"type": "done", "ok": True, "query_type": "search", "sources": sources, "trace": []}
