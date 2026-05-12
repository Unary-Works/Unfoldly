"""
ContinuationGroup — Expert Group for continuation / followup intents (v2 arch).

Design (v2 strengthened):
  - ContextFollowupExpert rules only intercept TWO high-certainty specialized cases:
      1. media_content_search followup (user asks about media content → media_content_search)
      2. explicit recount / stats re-query (confirmed recount signal)
  - ALL other cases (pronoun followup, topic followup, ambiguous) → FollowupClassifier LLM
  - FollowupClassifier (binary A/B, ~60 tok) decides: followup or new request?
  - If followup → ContinuationAgent LLM resolves the sub-intent
  - If new request → fallback_to_file_op (re-routed to FileOpGroup)

This gives LLM ~70% more authority vs previous "rules-first" approach.
"""
from __future__ import annotations

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Only intercept queries that are OBVIOUSLY new searches (strong verb at start)
_HARD_NEW_SCOPE_RE = re.compile(
    r'^(find\b|search\b|look\s+for\b|list\s+all\b|count\s+my\b|how\s+many\b'
    r'|show\s+me\s+all\b|give\s+me\s+all\b|open\b|launch\b'
    r'|\u627e\b|\u641c\b|\u67e5\u627e\b|\u641c\u7d22\b|\u6709\u591a\u5c11\b|\u5217\u51fa\b|\u6253\u5f00\b)',
    re.IGNORECASE,
)

# Explicit media-content followup signal (high-confidence rule KEPT)
_MEDIA_CONTENT_FOLLOWUP_RE = re.compile(
    r'(?:\u97f3\u9891|\u89c6\u9891|\u5f55\u97f3|\u64ad\u5ba2).*?(?:\u63d0\u5230|\u8bb2\u5230|\u8bf4\u5230|\u5173\u4e8e|\u5185\u5bb9)'
    r'|(?:\u63d0\u5230|\u8bb2\u5230|\u8bf4\u5230|\u5173\u4e8e|\u5185\u5bb9).*?(?:\u97f3\u9891|\u89c6\u9891|\u5f55\u97f3)'
    r'|\b(?:audios?|videos?|recordings?|podcasts?)\b.*?\b(?:mention|discuss|say|talk|about|content|contain)\b'
    r'|\b(?:mention|discuss|say|talk|about|content|contain)\b.*?\b(?:audios?|videos?|recordings?|podcasts?)\b'
    r'|\u627e\u97f3\u9891|\u627e\u89c6\u9891',
    re.IGNORECASE,
)

# Explicit recount signal (high-confidence rule KEPT)
_RECOUNT_SIGNAL_RE = re.compile(
    r'\b(recount|count\s+again|re-?count)\b'
    r'|\u518d\u6570|\u91cd\u65b0\u7edf\u8ba1|\u518d\u7edf\u8ba1\u4e00\u6b21',
    re.IGNORECASE,
)

# ── Refine-after-summarize fast-path ──────────────────────────────────────────
# When prior action was summarize/summarize_all/summarize_selected AND user now
# says "focus only on X", "now look at just Y", "summarize it more briefly" etc.
# → these are unambiguous process_previous refinements, skip LLM entirely.
_REFINE_AFTER_SUMMARIZE_RE = re.compile(
    r'^(focus\s+(only\s+)?on'
    r'|now\s+(focus|look\s+at|only\s+show)'
    r'|only\s+(show|consider|the\s+)'
    r'|just\s+(the|show|focus)'
    r'|narrow\s+(down|it)'
    r'|filter\s+(to|by|down)'
    r'|which\s+(files?|ones?|documents?)\s+(in|from|support|back|back\s+that)'
    r'|what\s+(files?|ones?|documents?)\s+support\s+(that|the\s+summary)'
    r'|summarize\s+(it|this|that|them)?\s*(more|again|briefly|shorter|better|concisely)'
    r'|make\s+(it|that)\s+(shorter|briefer|more\s+concise)'
    r'|can\s+you\s+(be\s+)?(more\s+)?(brief|concise|short)'
    r'|give\s+me\s+(a\s+)?(shorter|briefer|more\s+concise)'
    r'|\u53ea\u770b|\u53ea\u8003\u8651|\u66f4\u7b80\u6d01|\u7b2c\u4e00|\u91cd\u70b9|\u6982\u62ec|\u7b80\u5316)'
    r'(?!.*\b(find|search|look\s+for|retrieve|all\s+my|\u641c|\u627e|\u67e5)\b)',
    re.IGNORECASE,
)

# Prior actions where user is in a "summarize" conversation state
_SUMMARIZE_PRIOR_ACTIONS = frozenset({
    "summarize_all", "summarize", "summarize_selected", "process_previous",
})


class ContinuationGroup:
    """
    Routes continuation intents in v2 architecture.
    LLM (FollowupClassifier) is now the PRIMARY decision maker for ambiguous cases.
    Rules only handle 2 deterministic specialized cases.
    """

    @classmethod
    def route(cls, query: str, ctx: Any) -> dict:
        ql = (query or "").lower().strip()
        last_results = ctx.last_results or []
        active_paths = getattr(ctx, "active_paths", None) or []
        history = ctx.history or []
        lang = ctx.prompt_language or "en"
        prior_action = str(getattr(ctx, "prior_intent_action", "") or "")

        # ── Hard guard: obvious new search → bail immediately ────────────
        if _HARD_NEW_SCOPE_RE.match(ql):
            logger.info("[ContinuationGroup] hard new-scope → fallback_to_file_op")
            return {"action": "fallback_to_file_op", "params": {}}

        # ── Rule 1 (kept): media content followup → media_content_search ─
        if last_results and _MEDIA_CONTENT_FOLLOWUP_RE.search(ql):
            logger.info("[ContinuationGroup] rule: media-content followup → media_content_search")
            return {
                "action": "media_content_search",
                "params": {"query": query, "media_type": "all"},
                "confidence": 0.92,
            }

        # ── Rule 2 (kept): explicit recount → process_previous ───────────
        if last_results and _RECOUNT_SIGNAL_RE.search(ql):
            logger.info("[ContinuationGroup] rule: recount signal → process_previous")
            return {"action": "process_previous", "params": {}, "confidence": 0.93}

        # ── Rule 3 (NEW): refine-after-summarize fast-path ────────────────
        # When the prior turn was a summarize of selected files/folder AND the
        # current query is a refinement ("focus only on X", "more briefly" etc.),
        # route directly to process_previous without any LLM call.
        if (prior_action in _SUMMARIZE_PRIOR_ACTIONS
                and (last_results or active_paths)
                and _REFINE_AFTER_SUMMARIZE_RE.match(ql)):
            logger.info(
                f"[ContinuationGroup] refine-after-summarize fast-path "
                f"(prior={prior_action!r}) → process_previous"
            )
            return {"action": "process_previous", "params": {}, "confidence": 0.94}

        # ── All other cases → FollowupClassifier LLM ────────────────────
        # (pronoun followup, topic followup, short ambiguous queries — all go to LLM)
        from core.intent.followup_classifier import FollowupClassifier

        if not last_results and not history:
            logger.info("[ContinuationGroup] no context → fallback_to_file_op")
            return {"action": "fallback_to_file_op", "params": {}}

        if FollowupClassifier.should_activate(query, last_results or [{}]) and ctx.llm_service:
            is_followup = FollowupClassifier.is_followup(
                query, last_results or [{}], ctx.llm_service,
                prior_action=prior_action, lang=lang,
            )
            if not is_followup:
                logger.info("[ContinuationGroup] FollowupClassifier → new_request → fallback_to_file_op")
                return {"action": "fallback_to_file_op", "params": {}}
            logger.info("[ContinuationGroup] FollowupClassifier → followup → ContinuationAgent")
        elif not ctx.llm_service:
            # No LLM available: fall back to ContextFollowupExpert rules as safety net
            logger.info("[ContinuationGroup] no llm_service → ContextFollowupExpert fallback")
            try:
                from core.intent.context_followup_expert import ContextFollowupExpert
                rule_r = ContextFollowupExpert.analyze(ctx)
                if rule_r is not None:
                    return rule_r
            except Exception:
                pass

        # ── ContinuationAgent: resolve sub-intent ────────────────────────
        try:
            from core.intent.continuation_agent import ContinuationAgent
            result = ContinuationAgent.analyze(ctx)
            logger.info(f"[ContinuationGroup] ContinuationAgent → {result.get('action')!r}")
            return result
        except Exception as e:
            logger.error(f"[ContinuationGroup] ContinuationAgent error: {e}")
            return {"action": "process_previous", "params": {}, "confidence": 0.70}
