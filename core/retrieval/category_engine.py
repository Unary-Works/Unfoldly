"""
Category Engine — file category inference, normalization, and validation.

Extracted from langgraph_agent.py. This module provides:
  - normalize_category_name: Normalize raw category strings to canonical English
  - paper_category_likely_wrong: Check if "paper" category is misapplied (e.g. homework)
  - report_category_likely_wrong: Check if "report" category is misapplied (e.g. homework)
  - is_generic_file_scope_category: Check if category is a generic scope word
  - localize_category_label: Convert English category to localized display name

All functions are stateless and require no LLM dependency.
"""
from __future__ import annotations

import json
import os
import re
import logging
import time
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

MEDIA_CATEGORY_FAMILY = "audio/video"  # legacy/all-media compatibility bucket
MEDIA_CATEGORY_LEAVES = frozenset({"audio", "video"})
_CATEGORY_REGISTRY_FILENAME = "category_registry_cache.json"
_CATEGORY_REGISTRY_CACHE_TTL_SEC = 300.0
_REGISTRY_SKIP_CATEGORIES = frozenset({"", "all", "other", "unknown"})


def normalize_category_name(category: str) -> str:
    """Normalize raw category string to canonical English form."""
    # _normalize_category_en is defined as a module-level function in knowledge_base.py
    from core.kb.knowledge_base import _normalize_category_en
    return _normalize_category_en(category, default="other")


def is_media_category_value(category: str) -> bool:
    """Return True when a stored/raw category value represents audio or video."""
    value = str(category or "").strip().lower()
    return value == MEDIA_CATEGORY_FAMILY or value in MEDIA_CATEGORY_LEAVES


def normalize_stored_category_name(category: str, *, media_type: str = "") -> str:
    """
    Normalize a stored category for display/counting while preserving media leaves.

    New indexed rows store audio and video as separate leaf categories. The
    legacy "audio/video" family is only kept for old data and all-media queries.
    """
    raw = str(category or "").strip().lower()
    inferred_media_type = str(media_type or "").strip().lower()

    if raw in MEDIA_CATEGORY_LEAVES:
        return raw
    if raw == MEDIA_CATEGORY_FAMILY and inferred_media_type in MEDIA_CATEGORY_LEAVES:
        return inferred_media_type

    normalized = normalize_category_name(raw)
    if normalized == MEDIA_CATEGORY_FAMILY and raw in MEDIA_CATEGORY_LEAVES:
        return raw
    return normalized


def normalize_meta_category_name(meta: Dict[str, Any]) -> str:
    """Resolve the effective stored category bucket for a metadata row."""
    if not isinstance(meta, dict):
        return "other"

    media_type = str(meta.get("media_type") or "").strip().lower()
    for key in ("doc_category", "doc_category_family", "doc_category_raw", "doc_category_leaf"):
        value = str(meta.get(key) or "").strip()
        if value:
            return normalize_stored_category_name(value, media_type=media_type)
    return "other"


def _resolve_category_registry_dir() -> str:
    data_dir = (os.getenv("FILEAGENT_DATA_DIR") or "").strip()
    if data_dir.startswith("~"):
        data_dir = os.path.expanduser(data_dir)
    if data_dir:
        base_dir = os.path.abspath(data_dir)
    else:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    return os.path.join(base_dir, "cache")


def get_category_registry_cache_path() -> str:
    return os.path.join(_resolve_category_registry_dir(), _CATEGORY_REGISTRY_FILENAME)


def _normalize_alias_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _english_word_variants(word: str) -> set[str]:
    token = _normalize_alias_text(word)
    if not token or not re.fullmatch(r"[a-z0-9]+", token):
        return {token} if token else set()

    variants = {token}
    if token.endswith("ies") and len(token) > 3:
        variants.add(token[:-3] + "y")
    elif token.endswith("es") and len(token) > 2:
        variants.add(token[:-2])
    elif token.endswith("s") and not token.endswith("ss") and len(token) > 1:
        variants.add(token[:-1])

    if token.endswith("y") and len(token) > 1 and token[-2] not in "aeiou":
        variants.add(token[:-1] + "ies")
    elif token.endswith(("s", "x", "z", "ch", "sh")):
        variants.add(token + "es")
    else:
        variants.add(token + "s")
    return {v for v in variants if v}


def build_dynamic_category_aliases(category: str) -> set[str]:
    raw = _normalize_alias_text(category)
    if not raw:
        return set()

    aliases = {
        raw,
        _normalize_alias_text(raw.replace("_", " ")),
        _normalize_alias_text(raw.replace("-", " ")),
        _normalize_alias_text(raw.replace("/", " ")),
        _normalize_alias_text(raw.replace("/", "")),
    }

    expanded = set()
    for alias in aliases:
        if not alias:
            continue
        expanded.add(alias)
        if re.fullmatch(r"[a-z0-9 ]+", alias):
            if " " in alias:
                head, tail = alias.rsplit(" ", 1)
                for variant in _english_word_variants(tail):
                    expanded.add(f"{head} {variant}".strip())
            else:
                expanded.update(_english_word_variants(alias))
    return {a for a in expanded if a and a not in _REGISTRY_SKIP_CATEGORIES}


def persist_category_registry(category_counts: Dict[str, int], *, source: str = "count_all_categories") -> Dict[str, Any]:
    payload_categories: List[Dict[str, Any]] = []
    for category, count in sorted((category_counts or {}).items(), key=lambda item: (-int(item[1] or 0), str(item[0] or ""))):
        normalized = normalize_stored_category_name(str(category or ""))
        if normalized in _REGISTRY_SKIP_CATEGORIES:
            continue
        payload_categories.append(
            {
                "category": normalized,
                "count": int(count or 0),
                "aliases": sorted(build_dynamic_category_aliases(normalized)),
            }
        )

    payload = {
        "generated_at": time.time(),
        "source": source,
        "categories": payload_categories,
    }

    path = get_category_registry_cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.debug("[category_registry] persist failed: %s", exc)
    return payload


def load_category_registry(*, refresh_if_missing: bool = False, max_age_sec: Optional[float] = None) -> Dict[str, Any]:
    path = get_category_registry_cache_path()
    payload: Dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle) or {}
        except Exception as exc:
            logger.debug("[category_registry] load failed: %s", exc)
            payload = {}

    try:
        ttl = float(max_age_sec if max_age_sec is not None else _CATEGORY_REGISTRY_CACHE_TTL_SEC)
    except Exception:
        ttl = _CATEGORY_REGISTRY_CACHE_TTL_SEC
    ttl = max(10.0, ttl)

    generated_at = float(payload.get("generated_at") or 0.0) if isinstance(payload, dict) else 0.0
    is_fresh = bool(payload) and generated_at > 0 and (time.time() - generated_at) <= ttl
    if payload and (is_fresh or not refresh_if_missing):
        return payload

    if refresh_if_missing:
        try:
            from tools.document_tools import get_kb_instance

            kb = get_kb_instance()
            counts = kb.count_all_categories() or {}
            return persist_category_registry(counts, source="kb_refresh")
        except Exception as exc:
            logger.debug("[category_registry] refresh failed: %s", exc)
    return payload if isinstance(payload, dict) else {}


def match_dynamic_category_from_query(query: str, *, refresh_if_missing: bool = True) -> str:
    ql = _normalize_alias_text(query)
    if not ql:
        return ""

    payload = load_category_registry(refresh_if_missing=refresh_if_missing)
    categories = payload.get("categories") or []
    if not isinstance(categories, list):
        return ""

    en_tokens = set(re.sub(r"[^a-z0-9]+", " ", ql).split())
    best_match = ""
    best_score = -1

    for item in categories:
        if not isinstance(item, dict):
            continue
        category = _normalize_alias_text(item.get("category") or "")
        if not category or category in _REGISTRY_SKIP_CATEGORIES:
            continue
        aliases = item.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        for alias in aliases or [category]:
            alias_norm = _normalize_alias_text(alias)
            if not alias_norm or alias_norm in _REGISTRY_SKIP_CATEGORIES:
                continue
            matched = False
            if re.search(r"[\u4e00-\u9fff]", alias_norm):
                matched = alias_norm in ql
            elif " " in alias_norm:
                matched = alias_norm in ql
            else:
                matched = alias_norm in en_tokens
            if not matched:
                continue
            score = len(alias_norm)
            if score > best_score:
                best_match = category
                best_score = score

    return best_match


def get_compatible_categories(category: str) -> set[str]:
    """
    Return retrieval-time compatible categories for a normalized category.

    Index-time document labeling is sometimes coarse (for example, a paper may
    be stored as report/document). Search recall should treat nearby document
    classes as compatible, while keeping count semantics explicit elsewhere.
    """
    normalized = normalize_category_name(category)
    if normalized in {"", "all", "unknown"}:
        return set()
    if normalized == MEDIA_CATEGORY_FAMILY:
        return {MEDIA_CATEGORY_FAMILY, *MEDIA_CATEGORY_LEAVES}
    if normalized in MEDIA_CATEGORY_LEAVES:
        return {normalized}
    compatibility = {
        "document": {
            "document", "resume", "report", "contract", "note", "manual",
            "book", "paper", "quotation", "invoice", "presentation",
        },
        "resume": {"resume", "document", "report", "note"},
        "manual": {"manual", "document", "report"},
        "paper": {"paper", "report", "document"},
        "report": {"report", "paper", "document", "manual"},
    }
    return set(compatibility.get(normalized, {normalized}))


def paper_category_likely_wrong_for_query(user_question: str, normalized_category: str) -> bool:
    """
    Check if "paper" category is misapplied.
    
    The intent model often maps homework/math-related queries to category=paper
    because "paper" is ambiguous in English. This function detects such cases.
    """
    if normalized_category != "paper":
        return False
    uq = str(user_question or "").strip()
    if not uq:
        return False
    uql = uq.lower()

    math_school_cn = (
        "数学", "数学题", "几何", "代数", "算术", "应用题",
        "口算", "习题", "练习册", "作业", "测验", "月考", "期中", "期末",
        "试卷", "卷子", "考卷", "考试卷", "真题", "模拟卷",
    )
    has_math_school = any(x in uq for x in math_school_cn)
    has_paper_intent_cn = "论文" in uq or "期刊" in uq
    has_paper_intent_en = any(
        p in uql for p in (
            " journal", "journal ", "arxiv", "preprint",
            " research paper", " academic paper",
        )
    ) or (
        ("paper" in uql or "papers" in uql)
        and ("research" in uql or "academic" in uql or "journal" in uql)
    )
    if has_math_school and not has_paper_intent_cn and not has_paper_intent_en:
        return True

    en_math = (
        "math problem", "math problems", "math homework",
        "math exercise", "school math", "word problem",
        "mathematics problems", "mathematics problem",
        "math test paper", "math test papers",
        "math exam paper", "math exam papers",
        "test paper", "test papers",
        "exam paper", "exam papers",
    )
    if any(x in uql for x in en_math):
        if not has_paper_intent_cn and not has_paper_intent_en:
            return True
    return False


def report_category_likely_wrong_for_query(user_question: str, normalized_category: str) -> bool:
    """
    Check if "report" category is misapplied.
    
    The intent model often maps homework-related queries to category=report.
    This function detects such cases.
    """
    if normalized_category != "report":
        return False
    uq = str(user_question or "").strip()
    if not uq:
        return False
    uql = uq.lower()

    homework_markers_cn = ("作业", "习题", "练习册", "功课", "试卷", "手抄报", "暑假", "寒假")
    homework_markers_en = ("homework", "schoolwork", "assignment", "assignments", "worksheet", "exercise")
    has_hw = any(x in uq for x in homework_markers_cn) or any(x in uql for x in homework_markers_en)
    if not has_hw:
        return False

    report_intent_cn = any(
        x in uq for x in (
            "报告", "研报", "总结报告", "分析报告",
            "述职报告", "开题报告", "调研报告", "汇报材料",
        )
    )
    report_intent_en = any(
        p in uql for p in (
            " business report", " annual report", " research report",
            " analysis report", " progress report", " lab report",
        )
    )
    if report_intent_cn or report_intent_en:
        return False
    return True


def is_generic_file_scope_category(category: str) -> bool:
    c = (category or "").strip().lower()
    if not c:
        return False
    generic = {
        "all", "全部", "所有", "文件", "文档", "资料", "数据源",
        "所有文件", "全部文件", "所有文档", "全部文档",
        "所有资料", "全部资料",
        "file", "files", "document", "documents", "doc", "docs",
    }
    return c in generic
