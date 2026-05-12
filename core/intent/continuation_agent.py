"""
Layer 2A: ContinuationAgent
─────────────────────────────
Called when Layer 1 Router returns "continuation".
Focused LLM call (~180 token prompt, max_tokens=30).

Determines the specific continuation action:
  - translate_response: user wants the previous response translated
  - process_previous:   user wants more detail / expansion on the previous topic
  - chat:               general non-file follow-up (fallback)

Key guard preserved from b2a16e3:
  If query starts with find/search/show/list → NEVER process_previous → chat
"""
import re
import json
import logging
from typing import Any, List, Dict, Optional

logger = logging.getLogger(__name__)

# Detect target language from query
_LANG_EN_PAT = re.compile(
    r'\b(english|en|英文|英语)\b', re.IGNORECASE
)
_LANG_ZH_PAT = re.compile(
    r'\b(chinese|zh|中文|中文版|普通话)\b', re.IGNORECASE
)
_ORDINAL_CONTENT_FOLLOWUP_PAT = re.compile(
    r'(tell\s+me\s+about|describe|explain|summar|overview|what\s+is\s+in|what\s+does|'
    r'讲讲|介绍|说说|总结|概述|内容|讲了什么).{0,24}'
    r'(first|second|third|one|file|document|第[一二两三1-3]|这个|那个)',
    re.IGNORECASE,
)
# Guard: strong new-action verbs prevent process_previous (preserved from b2a16e3)
# NOTES:
#   - bare `show` excluded: 'show me what it says' is a valid continuation
#   - bare `what` excluded: 'what are the key points?' / 'what does it mean?' are continuations
#   - `what files/documents/items/are my` are explicit file-listing requests → block
#   - CJK alternates have NO \b suffix (Python \b is ASCII-only; \b fails before CJK chars)
_STRONG_ACTION_PAT = re.compile(
    # ASCII part with proper \b
    r'^(find|search|list|get|display|retrieve|count|'
    r'what\s+files\b|what\s+documents?\b|what\s+docs\b|what\s+items\b|what\s+are\s+my\b|'
    r'how\s+many|which\s+files?\b|which\s+documents?\b|do\s+i\s+have)\b'
    # CJK part — NO outer \b (CJK chars have no ASCII word boundaries)
    r'|^(找|搜|查找|搜索|显示|列出|查一下|搜一下|(我)?有哪些|(我)?有(什么|哪些)|(我)?(一共|总计)?有多少)',
    re.IGNORECASE
)


def _get_prev_response_preview(history: List[Dict], max_chars: int = 1200) -> str:
    """Get the last assistant response, truncated."""
    if not history:
        return ""
    last = history[-1]
    answer = str(last.get("a") or last.get("content") or "").strip()
    return answer[:max_chars] if answer else ""


def _get_prev_user_query(history: List[Dict], max_chars: int = 120) -> str:
    if not history:
        return ""
    for msg in reversed(history):
        text = str(msg.get("q") or msg.get("content") or "").strip()
        role = str(msg.get("role") or "")
        if text and (msg.get("q") or role == "user"):
            return text[:max_chars]
    return ""


def _build_last_results_preview(last_results: Optional[List[Dict]], max_items: int = 4) -> str:
    rows = list(last_results or [])
    if not rows:
        return "(none)"
    items: List[str] = []
    for row in rows[:max_items]:
        name = str(row.get("file_name") or row.get("name") or "").strip()
        if name:
            items.append(name)
    if not items:
        return "(none)"
    suffix = f" ...+{len(rows) - max_items} more" if len(rows) > max_items else ""
    return ", ".join(items) + suffix


def _extract_focused_file(last_results: Optional[List[Dict]]) -> str:
    rows = list(last_results or [])
    unique_names: List[str] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("file_path") or row.get("file_name") or row.get("name") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        name = str(row.get("file_name") or row.get("name") or key).strip()
        if name:
            unique_names.append(name)
    if len(unique_names) == 1:
        return unique_names[0]
    return "(none)"


class ExplicitContinuationExpert:

    _NO_FILE_OP_PAT = re.compile(
        # ASCII part — \b works correctly
        r'\b(find|search|show|list|count|display|retrieve|how\s+many|summarize)\b'
        r'|'
        # CJK part — NO outer \b (Python \b is ASCII-only)
        r'(总结|搜索|查找|统计|显示|列出)',
        re.IGNORECASE
    )
    _TRANSLATE_PAT = re.compile(
        r'^(please\s+|can you\s+)?'
        r'(in\s+(english|chinese|zh|en|japanese|french|german)'
        r'|translate(\s+(it|this|that|above|below|them))?(\s+to\s+(english|chinese|zh|en))?'
        r'|用(英文|中文|日文|法文)(\s*?(说|回答|写|回复|再说一遍))?'
        r'|翻译(成|为)?(英文|中文)?)'
        r'[\s.!？。]*(above|below|it|this|that|these|those|again)?[\s.!？。]*$',
        re.IGNORECASE
    )
    _CONTINUE_PAT = re.compile(
        r'^(ok[,\s]*(go\s+on|continue|next|proceed)'
        r'|go\s+on|continue|keep\s+going|next\s+one|more\s+details?'
        r'|(好|继续|下一个|更多|详细|展开))'
        r'[\s.!？。]*$',
        re.IGNORECASE
    )

    @classmethod
    def analyze(cls, query: str, history: List[Dict]) -> Optional[dict]:
        """Check if query is a pure continuation command."""
        if not history:
            return None
            
        ql = (query or "").lower().strip()
        if cls._NO_FILE_OP_PAT.search(ql):
            return None

        if cls._TRANSLATE_PAT.match(ql):
            _tgt = "zh" if re.search(r'(chinese|zh|中文)', ql, re.IGNORECASE) else "en"
            logger.info(f"[explicit_continuation] translate → translate_response(lang={_tgt})")
            return {"action": "translate_response", "params": {"lang": _tgt}, "confidence": 1.0}

        if cls._CONTINUE_PAT.match(ql):
            logger.debug(f"[explicit_continuation] continue/expand → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 1.0}

        return None

class ContinuationAgent:
    """
    Layer 2A: handles continuation intents.
    Returns an intent dict like the main analyzer.
    """

    @classmethod
    def analyze(cls, ctx: Any) -> dict:  # ctx: IntentContext
        """
        Returns one of:
          {"action": "translate_response", "params": {"lang": "en"/"zh"}}
          {"action": "view_detail",        "params": {"index": N}}
          {"action": "process_previous",   "params": {}}
          {"action": "chat",               "params": {}}
        """
        qn = (ctx.question or "").strip()
        ql = qn.lower()
        lang = ctx.prompt_language or "en"
        history = ctx.history or []
        last_results = ctx.last_results or []

        # ── Guard: strong action verb → not a continuation (from b2a16e3) ──────
        if _STRONG_ACTION_PAT.match(ql):
            logger.info("[ContinuationAgent] strong action verb guard → fallback_to_file_op")
            return {"action": "fallback_to_file_op", "params": {}}

        from core.intent_analyzer import IntentAnalyzer

        if last_results and (
            IntentAnalyzer.looks_like_meta_followup_on_last_results(qn, lang)
            or IntentAnalyzer.looks_like_content_followup_on_prior_results(qn)
        ):
            logger.info("[ContinuationAgent] deterministic prior-results followup → process_previous")
            return {"action": "process_previous", "params": {}, "confidence": 0.92}

        prev_preview = _get_prev_response_preview(history)
        prev_user_query = _get_prev_user_query(history)
        last_results_preview = _build_last_results_preview(last_results)
        focused_file = _extract_focused_file(last_results)

        # No previous response → can't do meaningful continuation
        if not prev_preview and not last_results:
            logger.info("[ContinuationAgent] no prior response → fallback_to_file_op")
            return {"action": "fallback_to_file_op", "params": {}}

        # ── LLM call ────────────────────────────────────────────────────────────
        from config.prompts import get_prompt
        prompt = get_prompt("CONTINUATION_AGENT_PROMPT", lang).format(
            prev_response_preview=prev_preview,
            prev_user_query=prev_user_query or "(none)",
            last_results_preview=last_results_preview,
            focused_file=focused_file,
            query=qn,
        )
        logger.debug(f"[ContinuationAgent] prompt_chars={len(prompt)} query_chars={len(qn)}")

        try:
            response = ctx.llm_service.generate(
                prompt,
                history=[],
                system_prompt=None,
            )
            raw = (response or "").strip()
            # Extract JSON from response
            start = raw.find("{")
            if start >= 0:
                decoder = json.JSONDecoder()
                result, _ = decoder.raw_decode(raw, start)
                action = str(result.get("action") or "").strip()
                if action == "translate_response":
                    # Detect target language — prefer explicit lang field, fallback to regex
                    lang_param = str(result.get("lang") or "").strip().lower()
                    if not lang_param:
                        lang_param = "en" if _LANG_EN_PAT.search(ql) else (
                            "zh" if _LANG_ZH_PAT.search(ql) else "en"
                        )
                    logger.info(f"[ContinuationAgent] → translate_response(lang={lang_param})")
                    return {"action": "translate_response", "params": {"lang": lang_param}}
                elif action == "view_detail":
                    # User referenced a specific result by number
                    try:
                        idx = int((result.get("params") or {}).get("index") or 1)
                    except (TypeError, ValueError):
                        idx = 1
                    if _ORDINAL_CONTENT_FOLLOWUP_PAT.search(ql):
                        logger.info(
                            f"[ContinuationAgent] view_detail(index={idx}) reinterpreted as process_previous for ordinal content follow-up"
                        )
                        return {"action": "process_previous", "params": {}}
                    logger.info(f"[ContinuationAgent] → view_detail(index={idx})")
                    return {"action": "view_detail", "params": {"index": idx}}
                elif action == "process_previous":
                    logger.info("[ContinuationAgent] → process_previous")
                    return {"action": "process_previous", "params": {}}
                elif action == "fallback_to_file_op":
                    logger.info("[ContinuationAgent] → fallback_to_file_op (unrelated new request)")
                    return {"action": "fallback_to_file_op", "params": {}}
                else:
                    logger.info(f"[ContinuationAgent] → chat (action={action!r})")
                    return {"action": "chat", "params": {}}
            logger.warning(f"[ContinuationAgent] no JSON in response: raw_chars={len(raw or '')} → fallback_to_file_op")
            return {"action": "fallback_to_file_op", "params": {}}
        except Exception as e:
            logger.error(f"[ContinuationAgent] LLM failed: {e}", exc_info=True)
            return {"action": "fallback_to_file_op", "params": {}}
