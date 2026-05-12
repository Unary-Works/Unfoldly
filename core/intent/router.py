"""
Layer 1: ConversationRouter
────────────────────────────
Ultra-lightweight LLM call (~120 token prompt, max_tokens=5).
Decides between four routing categories:
  - "continuation": user is following up on the previous assistant response
  - "file_op": user wants to find / search / count / summarize files
  - "media": user wants to search / query / count audio/video by *content*
  - "chat": general conversation, not file-related

Only reached when Layer 0 fast paths (fp_continuation, fp0, fp1, fp_filename,
fp_find_category) did NOT match.
"""
import re
import logging
from typing import Any, List, Dict, Optional

logger = logging.getLogger(__name__)

# Valid router tokens (LLM may output surrounding noise — strip it)
_VALID_ROUTES = {"continuation", "file_op", "media", "chat"}

# Prior topic extraction: take the first sentence / 60 chars of last assistant answer
def _extract_prior_topic(history: List[Dict]) -> str:
    if not history:
        return ""
        
    # Guard against frontend sending initial UI greeting as "history"
    # If there are no past 'user' turns, this is undeniably a new chat.
    has_user_msg = any(m.get("role") == "user" or "q" in m for m in history)
    if not has_user_msg:
        return ""

    last = history[-1]
    answer = str(last.get("a") or last.get("content") or "").strip()
    if not answer:
        return ""

    # Take first sentence
    first_sent = re.split(r'[。！？\n.!?]', answer)[0].strip()
    return first_sent[:80] if first_sent else answer[:80]


def _extract_prior_user_query(history: List[Dict], max_chars: int = 120) -> str:
    if not history:
        return ""
    for msg in reversed(history):
        text = str(msg.get("q") or msg.get("content") or "").strip()
        role = str(msg.get("role") or "")
        if text and (msg.get("q") or role == "user"):
            return text[:max_chars]
    return ""


def _build_last_results_preview(last_results: Optional[List[Dict]], max_items: int = 4) -> str:
    if not last_results:
        return "(none)"
    items: List[str] = []
    for row in last_results[:max_items]:
        name = str(row.get("file_name") or row.get("name") or "").strip()
        if name:
            items.append(name)
    if not items:
        return "(none)"
    suffix = f" ...+{len(last_results) - max_items} more" if len(last_results) > max_items else ""
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


class ConversationRouter:
    """
    Layer 1 Router: decides routing category with a tiny LLM call.
    Prompt ≈ 120 tokens.  max_tokens = 5 (one word answer).
    """

    # Media signal: query explicitly mentions video/audio content
    _MEDIA_NOUN_PAT = re.compile(
        r'(?i)\b(videos?|audios?|recordings?|clips?|movies?)\b|视频|音频|录屏|录音|录像',
    )
    _MEDIA_CONTENT_PAT = re.compile(
        r'(?i)'
        r'\b(about|contain|mention|show|appear|happening|scene|content|inside|within)\b|'
        r'关于|涉及|出现|包含|提到|有关|内容|画面|场景|里面|里有|哪些.*有',
    )

    @classmethod
    def route(cls, ctx: Any) -> str:  # ctx: IntentContext
        """
        Returns one of: "continuation" | "file_op" | "media" | "chat"
        Falls back to "file_op" on LLM failure (safe default).
        """
        lang = ctx.prompt_language or "en"
        history = ctx.history or []
        prior_topic = _extract_prior_topic(history)
        prior_user_query = _extract_prior_user_query(history)
        has_prior = bool(prior_topic)
        last_results = ctx.last_results or []
        last_results_preview = _build_last_results_preview(last_results)
        focused_file = _extract_focused_file(last_results)
        from core.intent_analyzer import IntentAnalyzer

        # ── Fast-path Guard: Explicit count/search ──
        if re.search(r'\b(how many|count|有多少|多少个|多少份|统计)\b', ctx.question, re.IGNORECASE):
            logger.info(f"[Router L1] explicit count guard → file_op")
            return "file_op"
            
        # ── Fast-path Guard: Explicit global reset (start over) ──
        if re.search(r'重新(查找|寻找|搜索|搜)|帮我重新|全局(查找|搜索|搜)|start\s+over|ignore\s+previous|全局.*所有', ctx.question, re.IGNORECASE):
            logger.info(f"[Router L1] explicit reset/global guard → file_op")
            return "file_op"

        q = ctx.question or ""

        # ── Fast-path Guard: Prior-results follow-up beats fresh media routing ──
        # to the current result set instead of opening a brand-new media search path.
        if last_results and (
            IntentAnalyzer.looks_like_meta_followup_on_last_results(q, lang)
            or IntentAnalyzer.looks_like_content_followup_on_prior_results(q)
        ):
            logger.info("[Router L1] prior-results followup guard → continuation")
            return "continuation"

        # ── Fast-path Guard: Explicit search/find verbs ──
        _FILEOP_VERB_PAT = re.compile(
            r'(?i)'
            r'\b(find|search|look\s+for|show(?:\s+me)?|retrieve|locate|get(?:\s+me)?|list|display|browse)\b|'
            r'\b(do\s+i\s+have)\b|'
            r'(查找|搜索|搜一下|查一下|帮我找|找一下|找到|调出|列出)'
        )
        _FILEOP_NOUN_PAT = re.compile(
            r'(?i)'
            r'\b(files?|documents?|docs?|resumes?|papers?|reports?|invoices?|recordings?|'
            r'slides?|presentations?|datasheet|datasheets|config|configs|images?|photos?|pictures?|'
            r'screenshots?|videos?|audios?|csvs?|pdfs?|tsvs?|txts?|json|xml|html|md|'
            r'xlsx?|xls|docx?|pptx?|wavs?|mp3s?|m4as?|mp4s?|movs?|'
            r'excel|word|powerpoint|spreadsheets?|worksheets?|tables?)\b|'
            r'(文件|文档|简历|论文|报告|发票|录音|幻灯片|配置|数据|图片)'
        )
        if _FILEOP_VERB_PAT.search(q) and _FILEOP_NOUN_PAT.search(q):
            logger.info(f"[Router L1] file-op verb+noun guard → file_op")
            return "file_op"
        if re.match(
            r'(?i)^(?:all\s+my|my)\s+'
            r'(?:files?|documents?|docs?|images?|photos?|pictures?|screenshots?|videos?|audios?|recordings?|'
            r'pdfs?|csvs?|tsvs?|docx?|xlsx?|xls|pptx?|txts?|json|xml|html|md|'
            r'wavs?|mp3s?|m4as?|mp4s?|movs?|excel|word|powerpoint|spreadsheets?|worksheets?|tables?)\s*$',
            q.strip(),
        ):
            logger.info(f"[Router L1] possessive inventory guard → file_op")
            return "file_op"
        # Also: queries starting with explicit file-op patterns → always file_op
        # Bare 'what' excluded: 'what are the key points?' is a valid continuation
        # 'show me' has negative lookahead: 'show me what it says' is continuation, not file_op
        if re.match(
            r'(?i)^(find|search|show(?:\s+me)?(?!\s+(what\s+(it|they|he|she|this|that)\b|the\s+content|more\s+about\s+(it|them|this|that)))|look\s+for|retrieve|locate|get(?:\s+me)?|list|display|browse|count|'
            r'what\s+files\b|what\s+documents?\b|what\s+docs\b|what\s+items\b|what\s+reports?\b|what\s+invoices?\b|'
            r'what\s+(images?|photos?|pictures?|screenshots?|videos?|audios?|recordings?|pdfs?|csvs?|tsvs?|docx?|xlsx?|xls|pptx?|wavs?|mp3s?|m4as?|mp4s?|movs?)\b|what\s+are\s+my\b|'
            r'how\s+many|which\s+files?\b|which\s+documents?\b|which\s+(reports?|invoices?|images?|photos?|videos?|audios?|recordings?)\b|do\s+i\s+have)'
            r'|^(我)?(有(哪些|什么)|(一共|总共)?有多少|找|搜|查找|搜索|显示|列出|查一下|搜一下|帮我找'
            r'|重新(查找|寻找|搜索|搜)|你重新查找|全局(查找|搜)|start\s+over|ignore\s+previous|全局.*所有)',
            q.strip(),
        ):
            logger.info(f"[Router L1] leading file-op verb guard → file_op")
            return "file_op"

        # ── Fast-path Guard: Media content query ──
        # Explicit file-search phrasing above should win. Queries that reach here
        # are the content-oriented ones, e.g. "what does the video say about X".
        if cls._MEDIA_NOUN_PAT.search(q) and cls._MEDIA_CONTENT_PAT.search(q):
            logger.info(f"[Router L1] media content guard → media")
            return "media"

        # Build prompt
        from config.prompts import get_prompt
        prompt = get_prompt("ROUTER_PROMPT", lang).format(
            has_prior="Yes" if has_prior else "No",
            prior_topic=prior_topic or "(none)",
            prior_user_query=prior_user_query or "(none)",
            recent_result_files=last_results_preview,
            focused_file=focused_file,
            query=ctx.question,
        )

        logger.debug(f"[Router L1] prompt_chars={len(prompt)} prior={has_prior} query_chars={len(ctx.question or '')}")

        try:
            response = ctx.llm_service.generate(
                prompt,
                history=[],        # Router needs no conversation history
                system_prompt=None,
            )
            raw = (response or "").strip().lower()
            # Extract first word that matches a valid route
            for token in re.split(r'[\s\n,;.]+', raw):
                token = token.strip('"\'')
                if token in _VALID_ROUTES:
                    logger.info(f"[Router L1] → {token}")
                    return token
            # Heuristic fallback: if answer contains any route keyword
            for route in ("continuation", "media", "file_op", "chat"):
                if route in raw:
                    logger.info(f"[Router L1] heuristic → {route}")
                    return route
            logger.warning(f"[Router L1] unrecognized output: raw_chars={len(raw or '')} → defaulting to file_op")
            return "file_op"
        except Exception as e:
            logger.error(f"[Router L1] LLM call failed: {e}", exc_info=True)
            return "file_op"
