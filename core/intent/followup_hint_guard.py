"""
FollowupHintGuard — determines whether a followup_hint should be honored.

Extracted from langgraph_agent._analyze_intent_with_context() to reduce
coupling and improve testability. Reuses existing micro-agent classifiers
(FilenameExpert, ContextFollowupExpert) instead of inline checks.

Pure rule-based, no LLM dependency.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Known file extensions for bare-filename detection
_HINT_BYPASS_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tiff",
    ".pdf", ".txt", ".md", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".mp3", ".mp4", ".wav", ".m4a", ".mov", ".avi", ".zip", ".rar",
    ".py", ".js", ".ts", ".json", ".csv", ".xml", ".html",
})

_EN_CONTENT_QUESTION_RE = re.compile(
    r"^(what|how|which|why|when|where|does|do|did|is|are|can|could|would|will)\b",
    re.IGNORECASE,
)
_ZH_CONTENT_QUESTION_RE = re.compile(
    r"^(这|那|它|该|这个|那个|这份|那份|怎么|如何|为什么|为何|是否|有没有|能否|会不会|讲了什么|说了什么|内容是啥|内容是什么)"
)
_EXPLICIT_FILEOP_VERB_PAT = re.compile(
    r'(?i)\b(find|search|look\s+for|show\s+me|retrieve|locate|get\s+me|list|do\s+i\s+have)\b'
)
_EXPLICIT_FILEOP_NOUN_PAT = re.compile(
    r'(?i)\b(files?|documents?|docs?|resumes?|papers?|reports?|invoices?|recordings?|'
    r'slides?|presentations?|datasheets?|configs?|images?|photos?|videos?|audios?|csv|pdf|wav|mp3)\b'
)


def should_honor_followup_hint(
    question: str,
    followup_hint: Dict[str, Any],
    *,
    has_last_results: bool,
    prompt_language: Optional[str] = None,
) -> bool:
    """
    Determine whether to honor a followup_hint or let the query fall through
    to the full IntentAnalyzer pipeline.

    Returns True if the hint should be applied (caller should use hinted action),
    False if the hint should be bypassed (caller should run IntentAnalyzer).
    """
    hinted_action = str(followup_hint.get("action") or "").strip()
    hint_params = followup_hint.get("params") or {}
    allow_without_results = bool(hint_params.get("allow_without_results"))

    if hinted_action not in {"process_previous", "view_detail"}:
        return False

    if not has_last_results:
        # Default behavior stays conservative: follow-up hints usually depend on
        # concrete prior files. The only exception is a short-lived search-topic
        # anchor emitted after a weak/empty search, which may still route a
        # clear detail question back to process_previous.
        if hinted_action != "process_previous" or not allow_without_results:
            return False

    # ── Check 1: Bare filename bypass ─────────────────────────────────────
    qn_strip = question.strip()
    qn_words = qn_strip.split()
    has_ext = any(qn_strip.lower().endswith(ext) for ext in _HINT_BYPASS_EXTS)
    is_bare_fn = has_ext and len(qn_words) <= 2

    if is_bare_fn:
        logger.info(
            f"🔧 followup_hint bypassed (bare filename): '{question}'"
        )
        return False

    # ── Check 2: New search detection (process_previous only) ─────────────
    if hinted_action == "process_previous":
        from core.intent_analyzer import IntentAnalyzer, IntentKeywords
        from core.intent.context_followup_expert import ContextFollowupExpert

        if ContextFollowupExpert._ATTR_LOOKUP_RE.search(question.lower()):
            logger.info(
                "🔧 followup_hint bypassed (personal attribute lookup): '%s'",
                question,
            )
            return False

        has_prev_ref = IntentAnalyzer._is_kw_match(
            question.lower(), IntentKeywords.PREV_REF_KWS, 'all'
        )
        is_meta_followup = IntentAnalyzer.looks_like_meta_followup_on_last_results(
            question, prompt_language
        )
        is_content_followup = IntentAnalyzer.looks_like_content_followup_on_prior_results(
            question
        )
        has_pronoun = bool(ContextFollowupExpert._PRONOUN_FOLLOWUP_RE.search(question.lower()))

        # ── Check 2.5: Recount guard ─────────────────────────────────────
        # "how many of them are PDFs" has a pronoun ("them") but is a recount,
        # NOT a content followup.  Recount should bypass the hint → count intent.
        is_recount = bool(ContextFollowupExpert._RECOUNT_RE.search(question.lower()))
        if is_recount:
            logger.info(
                f"🔧 followup_hint bypassed (recount detected): '{question}'"
            )
            return False

        is_new_scope = bool(ContextFollowupExpert._NEW_SCOPE_RE.search(question.lower()))
        is_search_entity = bool(ContextFollowupExpert._SEARCH_VERB_ENTITY_RE.search(question.lower()))
        is_explicit_fileop_search = bool(
            _EXPLICIT_FILEOP_VERB_PAT.search(question)
            and _EXPLICIT_FILEOP_NOUN_PAT.search(question)
            and not has_prev_ref
        )
        if is_new_scope or is_search_entity or is_explicit_fileop_search:
            logger.info(
                f"🔧 followup_hint bypassed (explicit new scope/search): '{question}'"
            )
            return False

        q_strip = question.strip()
        implicit_content_question = bool(
            _EN_CONTENT_QUESTION_RE.search(q_strip)
            or _ZH_CONTENT_QUESTION_RE.search(q_strip)
        )

        if not (
            has_prev_ref
            or is_meta_followup
            or is_content_followup
            or has_pronoun
            or implicit_content_question
        ):
            logger.info(
                f"🔧 followup_hint bypassed (new search detected): '{question}'"
            )
            return False

    # ── All checks passed: honor the hint ─────────────────────────────────
    logger.info(
        "🔧 followup_hint honored: %s%s",
        hinted_action,
        " (topic-anchor)" if (allow_without_results and not has_last_results) else "",
    )
    return True
