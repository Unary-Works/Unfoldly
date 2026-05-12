"""
QueryNormalizer — intent normalization pipeline.

Extracted from langgraph_agent.py's 220-line _normalize_intent_to_internal_en() method.
This is a pure-function pipeline that normalizes LLM intent output to internal format:
  - Unifies action names to English lowercase
  - Normalizes parameter keys
  - Cleans search query boilerplate
  - Handles selected-override (clear category for selection queries)
  - Handles extension-listing override (count + ext → search)
  - English category inference (fallback when LLM emits no category)
  - Anti-trigger correction (fix LLM mis-categorization, e.g. "manual" for "manager")

No LLM dependency. Deterministic, sub-millisecond.
"""
from __future__ import annotations

import re
import logging
from typing import Optional, List, Dict, Any, Callable, Tuple

logger = logging.getLogger(__name__)


# ── English Category Inference ────────────────────────────────────────────────
# Rule-based safety net: when LLM intent pass doesn't emit a category for
# English queries, these trigger-word lists map query content to the correct
# file-type filter.  Ordered from most-specific to least-specific.
_EN_CATEGORY_TRIGGERS: List[Tuple[str, List[str]]] = [
    ("video", [
        "video", "videos", "movie", "movies", "film", "clip", "clips",
        "footage", "reel", "mp4", "mov", "mkv", "avi", "webm", "m4v",
    ]),
    ("audio", [
        "audio", "sound", "recording", "music", "song", "track", "mp3",
        "wav", "flac", "aac", "m4a", "ogg", "wma", "aiff",
        "podcast", "voice memo", "noise", "melody", "beat", "rhythm",
    ]),
    ("image", [
        "image", "photo", "picture", "screenshot", "diagram", "chart",
        "graphic", "png", "jpg", "jpeg", "gif", "bmp", "tiff", "heic",
        "illustration", "figure", "drawing", "sketch", "mockup", "thumbnail",
        "wiring diagram", "schematic", "blueprint",
    ]),
    # 'document' covers reports, analysis, resumes, invoices, strategy docs, papers
    ("document", [
        "resume", " cv ", "curriculum vitae", "invoice", "receipt", "bill",
        "contract", "agreement", "thesis", "presentation", "slides", "spreadsheet",
        # strategy / business docs — fix for "GTM", "strategy", "analysis" misclassified as manual
        "strategy", "gtm", "roadmap", "plan", "research", "findings", "insights",
        "analysis", "report", "paper",
    ]),
    # 'manual' only for explicit product/technical manuals — NEVER triggered by "manager" or "strategy"
    ("manual", [
        "product manual", "user guide", "data sheet", "datasheet",
        "installation guide", "technical specification", "spec sheet",
    ]),
]

# Words that look like triggers but should NOT infer category (anti-patterns)
_EN_CATEGORY_ANTITRIGGERS: frozenset = frozenset({
    "manager", "management", "managed", "managing",  # "manager" sounds like "manual" to LLMs
})

# Real manual keywords for anti-trigger correction guard
_REAL_MANUAL_KWS = frozenset({
    "product manual", "user guide", "datasheet", "data sheet",
    "installation guide", "technical specification", "spec sheet",
})

# Report keywords that indicate the "manual" category is wrong
_REPORT_TRIGGER_KWS = frozenset({
    "strategy", "gtm", "report", "analysis", "roadmap", "plan",
    "research", "findings", "relationship", "executive",
})
_EXPLICIT_COUNT_CATEGORIES = frozenset({
    "resume",
    "report",
    "contract",
    "note",
    "manual",
    "paper",
    "presentation",
    "data",
    "email",
    "image",
    "audio",
    "video",
    "audio/video",
    "book",
    "code",
    "invoice",
    "quotation",
})


def _infer_category_from_english_query(query: str) -> str:
    """
    Infer a file-type category from English query signal words.

    Returns the canonical category string (e.g. "video", "audio", "image",
    "document") or an empty string if no signal is detected.

    Pure Python, zero LLM calls – executes in microseconds.
    """
    if not query:
        return ""
    q_lower = query.lower()
    q = f" {q_lower} "          # pad so " cv " matches "my cv"
    for category, triggers in _EN_CATEGORY_TRIGGERS:
        for trigger in triggers:
            # Use word-boundary-aware matching via space padding
            t = trigger if " " in trigger else f" {trigger} "
            if t in q:
                # Anti-trigger guard: don't fire if an anti-pattern overwrites
                # e.g. "manual" trigger blocked when "manager" present
                blocked = any(f" {anti} " in q for anti in _EN_CATEGORY_ANTITRIGGERS)
                if blocked and category == "manual":
                    continue
                return category
    return ""


def _infer_specific_count_category(query: str) -> str:
    """
    Infer an explicit count category from the raw user query.

    Count requests often arrive as count(all) from the LLM even when the user
    explicitly says "invoice files", "resumes", or similar. Keep this narrow:
    only return non-generic categories, otherwise let count(all) stand.
    """
    question = str(query or "").strip()
    if not question:
        return ""

    try:
        from core.kb.knowledge_base import _normalize_category_en

        normalized = str(_normalize_category_en(question, default="") or "").strip().lower()
    except Exception:
        normalized = ""

    if normalized in _EXPLICIT_COUNT_CATEGORIES:
        return normalized

    try:
        from core.retrieval.category_engine import match_dynamic_category_from_query

        dynamic = str(match_dynamic_category_from_query(question, refresh_if_missing=True) or "").strip().lower()
    except Exception:
        dynamic = ""

    if dynamic in _EXPLICIT_COUNT_CATEGORIES:
        return dynamic
    return ""


class QueryNormalizer:
    """
    Pure normalization pipeline for intent outputs.
    
    Takes raw LLM intent output and produces a standardized internal format.
    """

    # ── Boilerplate words commonly appended by LLMs ───────────────────────
    _BOILERPLATE_WORDS = frozenset({
        "document", "documents", "file", "files", "record", "records",
        "item", "items", "info", "information", "data", "material",
        "materials", "paper", "papers", "related", "relevant",
        "content", "contents",
    })

    # ── Selected-scope keywords ───────────────────────────────────────────
    _SELECTED_KWS = frozenset({
        "selected", "chosen", "checked", "current file", "current document",
        "选中", "勾选", "当前文件", "这个文件", "这个文档",
    })

    # ── Generic categories that should be cleared on selection queries ────
    _GENERIC_CATS = frozenset({
        "document", "documents", "doc", "docs", "file", "files", "all",
    })

    # ── Extension aliases for listing override ────────────────────────────
    _EXT_ALIASES: Dict[str, str] = {
        "pdf": ".pdf", "pdfs": ".pdf",
        "docx": ".docx", "doc": ".doc", "word": ".docx",
        "xlsx": ".xlsx", "xls": ".xls", "excel": ".xlsx",
        "pptx": ".pptx", "ppt": ".ppt", "powerpoint": ".pptx",
        "txt": ".txt", "csv": ".csv", "json": ".json",
        "mp3": ".mp3", "mp4": ".mp4", "wav": ".wav",
        "jpg": ".jpg", "jpeg": ".jpeg", "png": ".png",
    }

    @classmethod
    def normalize(
        cls,
        question: str,
        result: dict,
        *,
        normalize_category_fn: Optional[Callable] = None,
    ) -> dict:
        """
        Normalize intent result to internal English format.
        
        Args:
            question: Original user query
            result: Raw intent {"action": ..., "params": ...}
            normalize_category_fn: Function to normalize category names
            
        Returns:
            Normalized intent dict with _normalized_internal_en=True
        """
        if not isinstance(result, dict):
            result = {}

        # Fast path: already normalized
        if bool(result.get("_normalized_internal_en")):
            action = str(result.get("action") or "").strip().lower() or "search"
            params = result.get("params") or {}
            if not isinstance(params, dict):
                params = {}
            return {"action": action, "params": params, "_normalized_internal_en": True}

        action = str(result.get("action") or "").strip().lower() or "search"

        # Passthrough: media_export carries specialized params (time_sec, target_type, file_hint,
        # time_end_sec) that must not be lost during normalization.  Return as-is with the flag set.
        # media_content_search / media_count / media_summarize also carry their own params.
        _MEDIA_PASSTHROUGH = {"media_export", "media_content_search", "media_count", "media_summarize"}
        if action in _MEDIA_PASSTHROUGH:
            raw_params = result.get("params") or {}
            if not isinstance(raw_params, dict):
                raw_params = {}
            out: Dict[str, Any] = {"action": action, "params": dict(raw_params), "_normalized_internal_en": True}
            if "confidence" in result:
                out["confidence"] = result["confidence"]
            return out
        raw_params = result.get("params") or {}
        if not isinstance(raw_params, dict):
            raw_params = {}

        # ── Extract and normalize parameters ──────────────────────────────
        query_raw = cls._pick_param(raw_params, "query", "question", "搜索词", "检索词", "问题")
        keywords_raw = cls._pick_param(raw_params, "keywords", "keyword", "关键词", "关键字")
        category_raw = cls._pick_param(raw_params, "category", "分类", "类别")
        folder_raw = cls._pick_param(raw_params, "folder", "path", "目录", "文件夹")
        file_name_raw = cls._pick_param(raw_params, "file_name", "file", "文件名")
        file_hint_raw = cls._pick_param(raw_params, "file_hint", "fileHint", "focused_file")
        media_type_raw = cls._pick_param(raw_params, "media_type", "mediaType")
        target_type_raw = cls._pick_param(raw_params, "target_type", "targetType")
        sub_intent_raw = cls._pick_param(raw_params, "sub_intent", "subIntent")
        file_extensions_raw = cls._pick_param(
            raw_params,
            "file_extensions",
            "file_extension",
            "extensions",
            "extension",
            "focus_extension",
        )
        index_raw = raw_params.get("index") or raw_params.get("序号")

        # Default query for search
        if action == "search" and not query_raw:
            query_raw = (question or "").strip()

        # Normalize text values
        query_en = cls._cheap_en_alias(query_raw) if query_raw else ""
        query_en = cls._trim_boilerplate(question, query_en, action)
        
        if keywords_raw:
            if query_raw and keywords_raw.strip().lower() == query_raw.strip().lower():
                keywords_en = query_en
            else:
                keywords_en = cls._cheap_en_alias(keywords_raw)
        else:
            keywords_en = ""

        # ── Build normalized params ───────────────────────────────────────
        params: Dict[str, Any] = {}
        if action == "clarify" and query_raw:
            params["question"] = query_raw
        if query_en:
            params["query"] = query_en
        if keywords_en:
            params["keywords"] = keywords_en
        if category_raw:
            cat = normalize_category_fn(category_raw) if normalize_category_fn else category_raw
            params["category"] = cat
        if folder_raw:
            params["folder"] = folder_raw
        if file_name_raw:
            params["file_name"] = file_name_raw
        if file_hint_raw:
            params["file_hint"] = file_hint_raw
        if media_type_raw:
            params["media_type"] = media_type_raw
        if target_type_raw:
            params["target_type"] = target_type_raw
        if sub_intent_raw:
            params["sub_intent"] = sub_intent_raw
        if file_extensions_raw:
            found_exts: List[str] = []
            for token in re.split(r"[\s,;|]+", file_extensions_raw):
                cleaned = str(token or "").strip().lower().lstrip(".")
                if not cleaned:
                    continue
                ext = cls._EXT_ALIASES.get(cleaned, f".{cleaned}")
                if ext not in found_exts:
                    found_exts.append(ext)
            if found_exts:
                params["file_extensions"] = ",".join(found_exts)
        if index_raw is not None:
            try:
                params["index"] = int(index_raw)
            except (ValueError, TypeError):
                pass

        # Preserve internal routing / metadata fields produced by upstream
        # expert arbitration so downstream validators can make safe decisions
        # without re-inferring deterministic routes from scratch.
        for k, v in raw_params.items():
            if isinstance(k, str) and k.startswith("_") and k not in params:
                params[k] = v

        # Preserve structured skill routing fields. These are not user-facing
        # search terms, but execution needs them to keep follow-ups scoped to
        # selected items or previous results instead of falling back to corpus-wide
        # behavior during normalization.
        for k in (
            "scope",
            "operation",
            "focused_file",
            "rewrite_mode",
            "focus_extension",
            "time_sec",
            "time_end_sec",
        ):
            if k in raw_params and k not in params:
                params[k] = raw_params[k]

        # ── Selected-scope override ───────────────────────────────────────
        q_lower = (question or "").lower()
        is_selected = any(kw in q_lower for kw in cls._SELECTED_KWS)
        if is_selected and params.get("category") in cls._GENERIC_CATS:
            logger.info(
                f"[normalizer] clearing category='{params['category']}' "
                f"for selected-scope query"
            )
            params.pop("category", None)

        # ── Count default / specific category recovery ────────────────────
        if action == "count" and "category" not in params:
            params["category"] = "all"
        if action == "count":
            current_category = str(params.get("category") or "").strip().lower()
            if current_category in {"", "all", "document", "documents", "other", "unknown"}:
                inferred_count_category = _infer_specific_count_category(question)
                if inferred_count_category:
                    params["category"] = inferred_count_category
                    logger.info(
                        f"[CountCategoryInference] explicit count scope detected → "
                        f"category='{inferred_count_category}' for query='{question[:60]}'"
                    )

        # ── Extension-listing override ────────────────────────────────────
        if action == "count":
            action, params = cls._handle_ext_listing_override(question, action, params)

        # ── search: keywords fallback ─────────────────────────────────────
        if action == "search" and ("query" not in params) and ("keywords" in params):
            params["query"] = params["keywords"]

        # ── English Category Inference (Python-side safety net) ───────────
        # Two modes:
        # 1. Correction: LLM emits WRONG category → override with correct one.
        # 2. Fallback: LLM emits no category → infer from English trigger words.
        if action == "search":
            params = cls._apply_category_inference(question, params)

        # Preserve confidence if present
        out: Dict[str, Any] = {"action": action, "params": params, "_normalized_internal_en": True}
        if "confidence" in result:
            out["confidence"] = result["confidence"]
        return out

    # ════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _pick_param(params: dict, *keys: str) -> str:
        """Pick the first non-empty value from multiple possible param keys."""
        for k in keys:
            v = params.get(k)
            if v is not None and str(v).strip():
                if isinstance(v, list):
                    parts = [str(x).strip() for x in v if str(x).strip()]
                    return " ".join(parts).strip()
                return str(v).strip()
        return ""

    @staticmethod
    def _cheap_en_alias(text: str) -> str:
        """Lightweight text cleanup without translation."""
        s = str(text or "").strip()
        if not s:
            return s
        try:
            # Add spacing between Latin and CJK boundaries
            s = re.sub(r"(?<=[A-Za-z0-9])(?=[\u4e00-\u9fff])", " ", s)
            s = re.sub(r"(?<=[\u4e00-\u9fff])(?=[A-Za-z0-9])", " ", s)
            # Chinese punctuation normalization
            s = s.replace("，", " ").replace("。", " ").replace("？", "?").replace("！", "!")
            s = re.sub(r"\s+", " ", s).strip()
        except Exception:
            pass
        return " ".join(s.split())

    @classmethod
    def _trim_boilerplate(cls, user_q: str, model_q: str, action: str) -> str:
        """Remove LLM-appended boilerplate from search queries."""
        u = str(user_q or "").strip()
        q = str(model_q or "").strip()
        if not u or not q or action != "search":
            return q

        # Skip for CJK-dominant queries
        if any("\u4e00" <= ch <= "\u9fff" for ch in u):
            return q

        uw = u.split()
        if len(uw) == 1 and 2 <= len(uw[0]) <= 64:
            w = uw[0]
            if q.casefold() != w.casefold() and q.casefold().startswith(w.casefold() + " "):
                return w

        if len(u) > 56 or len(u.split()) > 5:
            return q
        ul, ql2 = u.casefold(), q.casefold()
        if ql2 == ul:
            return u
        if not ql2.startswith(ul):
            return q
        rest = ql2[len(ul):].strip()
        if not rest:
            return u
        kept = [t for t in rest.split() if t not in cls._BOILERPLATE_WORDS]
        if not kept:
            return u
        return f"{u} {' '.join(kept)}".strip()

    @classmethod
    def _handle_ext_listing_override(cls, question: str, action: str, params: dict) -> tuple:
        """Convert count + extension query → search when it's a listing request."""
        q_low = (question or "").lower()

        _is_listing = any(kw in q_low for kw in [
            "find", "show", "list", "get", "display", "where",
            "有哪些", "列出", "找", "查找", "显示", "给我", "看看",
        ])
        _is_count = any(kw in q_low for kw in [
            "how many", "count", "total", "number of",
            "多少", "几个", "数量", "共有", "一共",
        ])

        if _is_listing and not _is_count:
            found_exts = []
            for tok in re.findall(r'\b[a-z0-9]+\b', q_low):
                if tok in cls._EXT_ALIASES and cls._EXT_ALIASES[tok] not in found_exts:
                    found_exts.append(cls._EXT_ALIASES[tok])
            if found_exts:
                logger.info(f"[normalizer] count → search (extensions={found_exts})")
                return "search", {"query": question, "file_extensions": ",".join(found_exts)}

        return action, params

    @classmethod
    def _apply_category_inference(cls, question: str, params: dict) -> dict:
        """
        Apply English category inference and anti-trigger correction.
        
        Two modes:
        1. Correction: LLM said "manual" but query is NOT about a product manual → fix it.
        2. Fallback: No category set by LLM → infer from English trigger words.
        
        Migrated from langgraph_agent._normalize_intent_to_internal_en().
        """
        q_lower = str(question or "").lower()
        llm_category = params.get("category", "")

        # ── Anti-trigger correction: LLM said "manual" but query is NOT about a product manual ──
        if llm_category == "manual":
            is_real_manual = any(kw in q_lower for kw in _REAL_MANUAL_KWS)
            has_anti_trigger = any(
                f" {anti} " in f" {q_lower} "
                for anti in _EN_CATEGORY_ANTITRIGGERS
            )
            has_report_trigger = any(kw in q_lower for kw in _REPORT_TRIGGER_KWS)
            
            if not is_real_manual and (has_anti_trigger or has_report_trigger):
                corrected = _infer_category_from_english_query(str(question or ""))
                if corrected and corrected != "manual":
                    params["category"] = corrected
                else:
                    # Remove the wrong manual category — better no filter than a wrong one
                    params.pop("category", None)
                logger.info(
                    f"[CategoryCorrection] LLM said 'manual' for '{question[:50]}', "
                    f"corrected to '{params.get('category', '(none)')}'"
                )

        # ── Fallback: no category set by LLM, infer from Python rules ──
        if "category" not in params:
            inferred = _infer_category_from_english_query(str(question or ""))
            if inferred:
                params["category"] = inferred
                logger.info(
                    f"[CategoryInference] English signal detected → "
                    f"category='{inferred}' for query='{question[:60]}'"
                )

        return params
