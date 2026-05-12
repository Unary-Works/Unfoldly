"""
IntentClassifier — top-level LLM router for v2 architecture.

Design:
  - Replaces the 7-layer sequential rule fast-path in IntentAnalyzer.analyze()
    as the PRIMARY routing decision maker.
  - Has cheap rule-based fast-paths for high-confidence cases (saves LLM call).
  - Falls back to one LLM call (~130 tok, max_tokens=20) for ambiguous cases.
  - Returns (group, confidence) where group ∈:
      "continuation" | "selection" | "media" | "file_op" | "chat"

Each group is handled by a dedicated Group class (ContinuationGroup, SelectionGroup, etc.)
that contains the original rules + an optional group-level LLM for sub-decisions.
"""
from __future__ import annotations

import re
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

_VALID_GROUPS = frozenset({"continuation", "selection", "media", "file_op", "chat"})

# ── Fast-path patterns (rule-based, zero LLM) ─────────────────────────────

# Explicit new-search starters — almost certainly file_op, never continuation
_STRONG_FILE_OP_START = re.compile(
    r'^(find|search|look\s+for|show\s+me\s+(all|my)|list|count|give\s+me'
    r'|open|launch|summarize\s+all|how\s+many'
    r'|\u627e|\u641c|\u67e5\u627e|\u641c\u7d22|\u641c\u4e00\u4e0b|\u67e5\u4e00\u4e0b|\u5217\u51fa|\u6253\u5f00'
    r'|(\u6211)?\u6709\u54ea\u4e9b|(\u6211)?\u6709\u591a\u5c11|\u7edf\u8ba1)\b',
    re.IGNORECASE,
)

# Explicit time references + media words → media (high confidence)
_MEDIA_TIME_RE = re.compile(
    r'(?:\d+\s*(?:s(?:ec(?:onds?)?)?|m(?:in(?:utes?)?)?)'
    r'|\d{1,3}:\d{2}'
    r'|\u7b2c\s*\d+\s*(?:\u79d2|\u5206))',
    re.IGNORECASE,
)
_MEDIA_WORD_RE = re.compile(
    r'\b(?:videos?|audios?|recordings?|podcasts?|movies?|clips?)\b'
    r'|\u89c6\u9891|\u97f3\u9891|\u5f55\u97f3|\u5f55\u5c4f',
    re.IGNORECASE,
)

# Media content-search (no timestamp needed)
_MEDIA_CONTENT_RE = re.compile(
    r'(?:\u97f3\u9891|\u89c6\u9891|\u5f55\u97f3|\u64ad\u5ba2).*?(?:\u63d0\u5230|\u8bb2\u5230|\u8bf4\u5230|\u5173\u4e8e|\u5185\u5bb9|\u8bb2\u7684|\u8bf4\u7684|\u63d0\u53ca)'
    r'|(?:\u63d0\u5230|\u8bb2\u5230|\u8bf4\u5230|\u5173\u4e8e|\u5185\u5bb9).*?(?:\u97f3\u9891|\u89c6\u9891|\u5f55\u97f3)'
    r'|\b(?:audios?|videos?|recordings?|podcasts?)\b.*?\b(?:mention|discuss|say|talk|about|content|contain)\b'
    r'|\b(?:mention|discuss|say|talk|about|content|contain)\b.*?\b(?:audios?|videos?|recordings?|podcasts?)\b'
    r'|\u627e\u97f3\u9891|\u627e\u89c6\u9891|\u97f3\u9891.*\u627e|\u89c6\u9891.*\u641c',
    re.IGNORECASE,
)

# Explicit chat/greeting signals
_CHAT_RE = re.compile(
    r'^(hello|hi|hey|who\s+are\s+you|what\s+(time|day)|can\s+you\s+help|write\s+code|write\s+a\s+program'
    r'|\u4f60\u597d|\u60a8\u597d|\u55e8|\u4f60\u662f\u8c01|\u73b0\u5728\u51e0\u70b9|\u5199\u4e2a\u7a0b\u5e8f)\b',
    re.IGNORECASE,
)

# Selection trigger words
_SELECTION_TRIGGER_RE = re.compile(
    r'\b(selected|seleted|selectd|slected|chosen|chosn|picked)\b'
    r'|\u9009\u4e2d|\u5df2\u9009|\u5df2\u52fe\u9009|\u5f53\u524d\u9009\u4e2d|\u52fe\u9009'
    r'|these\s+files|these\s+docs|this\s+file|\u8fd9\u4e9b\u6587\u4ef6|\u8fd9\u4e9b\u6587\u6863',
    re.IGNORECASE,
)

# Explicit continuation: very short with pronoun reference
_PRONOUN_RE = re.compile(
    r'\b(it|them|they|these|those|this|above|the\s+(first|second|third|last)\s+one)\b'
    r'|\u5b83\u4eec|\u8fd9\u4e9b|\u90a3\u4e9b|\u4e0a\u9762|\u4e0a\u8ff0|\u7b2c[一二两三1-3]\u4e2a',
)


class IntentClassifier:
    """
    Top-level intent group classifier for v2 architecture.

    Usage:
        group, conf = IntentClassifier.classify(ctx)
        # group ∈ {"continuation", "selection", "media", "file_op", "chat"}
    """

    @classmethod
    def classify(cls, ctx: Any) -> Tuple[str, float]:
        """
        Return (group, confidence).  Fast-paths first, LLM fallback.
        ctx: IntentContext with .question, .history, .last_results, .active_paths,
             .prompt_language, .llm_service, .prior_intent_action
        """
        qn = (ctx.question or "").strip()
        ql = qn.lower()
        has_history = bool(ctx.history)
        last_results = ctx.last_results or []
        active_paths = ctx.active_paths or []
        n_selected = len(active_paths)

        if not qn:
            return "chat", 0.5

        # ── Fast-path 1: explicit chat ─────────────────────────────────────
        if _CHAT_RE.match(ql):
            logger.info("[IntentClassifier] fast-path: chat")
            return "chat", 0.95

        # ── Fast-path 2: selection trigger (only when files are selected) ──
        if n_selected > 0 and _SELECTION_TRIGGER_RE.search(ql):
            logger.info("[IntentClassifier] fast-path: selection (trigger + active_paths)")
            return "selection", 0.92

        # ── Fast-path 3: media time+content query ─────────────────────────
        if _MEDIA_TIME_RE.search(ql) and _MEDIA_WORD_RE.search(ql):
            logger.info("[IntentClassifier] fast-path: media (time + media word)")
            return "media", 0.93

        if _MEDIA_CONTENT_RE.search(qn):
            logger.info("[IntentClassifier] fast-path: media (content search)")
            return "media", 0.88

        # ── Fast-path 4: strong file_op starter ────────────────────────────
        if _STRONG_FILE_OP_START.match(ql):
            logger.info("[IntentClassifier] fast-path: file_op (strong verb)")
            return "file_op", 0.90

        # ── Fast-path 5: short pronoun + has prior results → continuation ─
        word_count = len([w for w in ql.split() if w])
        if has_history and last_results and word_count <= 5 and _PRONOUN_RE.search(ql):
            logger.info("[IntentClassifier] fast-path: continuation (short pronoun + results)")
            return "continuation", 0.87

        # ── LLM fallback ───────────────────────────────────────────────────
        if not ctx.llm_service:
            logger.info("[IntentClassifier] no llm_service → file_op default")
            return "file_op", 0.60

        return cls._classify_via_llm(ctx, qn, n_selected)

    @classmethod
    def _classify_via_llm(cls, ctx: Any, qn: str, n_selected: int) -> Tuple[str, float]:
        """One small LLM call to classify the group."""
        last_results = ctx.last_results or []
        history = ctx.history or []
        lang = ctx.prompt_language or "en"

        # Build compact context strings
        prior_action = str(getattr(ctx, "prior_intent_action", "") or "none")
        n_results = len(last_results)
        prior_topic = ""
        if history:
            last = history[-1]
            prior_topic = str(last.get("a") or last.get("content") or "")[:80]

        try:
            from config.prompts import get_prompt
            prompt = get_prompt("INTENT_CLASSIFIER_PROMPT", lang).format(
                prior_action=prior_action,
                n_selected=n_selected,
                n_results=n_results,
                prior_topic=prior_topic,
                query=qn,
            )
        except Exception:
            # Fallback inline prompt if key not registered yet
            prompt = cls._build_inline_prompt(
                qn=qn, lang=lang,
                prior_action=prior_action,
                n_selected=n_selected,
                n_results=n_results,
                prior_topic=prior_topic,
            )

        try:
            raw = (ctx.llm_service.generate(prompt, history=[], system_prompt=None) or "").strip()
            group = cls._parse_group(raw, n_selected)
            logger.info(f"[IntentClassifier] LLM → group={group!r} (raw={raw[:30]!r})")
            return group, 0.82
        except Exception as e:
            logger.warning(f"[IntentClassifier] LLM failed: {e} → file_op fallback")
            return "file_op", 0.55

    @classmethod
    def _parse_group(cls, raw: str, n_selected: int) -> str:
        """Extract group word from LLM output."""
        word = raw.strip().lower().split()[0] if raw.strip() else ""
        # Normalize variations
        _MAP = {
            "continuation": "continuation", "continue": "continuation", "followup": "continuation",
            "selection": "selection", "selected": "selection",
            "media": "media",
            "file_op": "file_op", "fileop": "file_op", "file": "file_op", "op": "file_op",
            "chat": "chat",
        }
        group = _MAP.get(word, "")
        # Safety: selection requires active_paths
        if group == "selection" and n_selected == 0:
            logger.info("[IntentClassifier] selection rejected (n_selected=0) → file_op")
            group = "file_op"
        return group if group in _VALID_GROUPS else "file_op"

    @staticmethod
    def _build_inline_prompt(*, qn, lang, prior_action, n_selected, n_results, prior_topic) -> str:
        """Minimal inline prompt when the prompt key is not registered."""
        if lang == "zh":
            return (
                f"[\u4e0a\u4e00\u8f6e\u52a8\u4f5c]: {prior_action}\n"
                f"[\u5df2\u9009\u6587\u4ef6\u6570]: {n_selected}\n"
                f"[\u4e0a\u8f6e\u7ed3\u679c\u6570]: {n_results}\n"
                f"[\u4e0a\u8f6e\u4e3b\u9898]: {prior_topic}\n"
                f"[\u7528\u6237\u8f93\u5165]: {qn}\n\n"
                "\u5206\u7ec4\uff08\u9009\u4e00\uff09: continuation / selection / media / file_op / chat\n"
                "\u89c4\u5219\uff1a\u82e5\u5df2\u9009\u6587\u4ef6\u6570=0\uff0c\u4e0d\u5f97\u8f93\u51fa selection\n"
                "\u53ea\u8f93\u51fa\u4e00\u4e2a\u8bcd\uff1a"
            )
        return (
            f"[Prior action]: {prior_action}\n"
            f"[Selected files]: {n_selected}\n"
            f"[Prior results]: {n_results}\n"
            f"[Prior topic]: {prior_topic}\n"
            f"[User]: {qn}\n\n"
            "Group (one of): continuation / selection / media / file_op / chat\n"
            "Rule: if selected files=0, never output selection\n"
            "Output ONE word only:"
        )
