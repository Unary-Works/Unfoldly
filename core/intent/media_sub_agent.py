"""
Layer 2C: MediaSubAgent
────────────────────────
Called when Layer 1 Router returns "media".
Focused LLM call (~200 token prompt, max_tokens=80).

Determines the specific media operation:
  - media_content_search: semantic search across video/audio keyframe content
  - media_count:          count video/audio files
  - media_summarize:      summarize video/audio files (redirects to summarize)
"""
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"media_content_search", "media_count", "media_summarize"}


class MediaSubAgent:
    """
    Layer 2C: handles media content intents.
    Returns an intent dict accepted by the main dispatch pipeline.
    """

    @classmethod
    def analyze(cls, ctx: Any) -> dict:  # ctx: IntentContext
        qn = (ctx.question or "").strip()
        lang = ctx.prompt_language or "en"

        from config.prompts import get_prompt
        prompt = get_prompt("MEDIA_SUB_AGENT_PROMPT", lang).format(query=qn)
        logger.debug(f"[MediaSubAgent] prompt_chars={len(prompt)} query_chars={len(qn)}")

        try:
            response = ctx.llm_service.generate(
                prompt,
                history=[],
                system_prompt=None,
            )
            raw = (response or "").strip()
            start = raw.find("{")
            if start >= 0:
                decoder = json.JSONDecoder()
                result, _ = decoder.raw_decode(raw, start)
                return cls._postprocess(result, ctx)
            logger.warning(f"[MediaSubAgent] no JSON: raw_chars={len(raw or '')} → media_content_search fallback")
            return {"action": "media_content_search", "params": {"query": qn, "media_type": "all"}}
        except Exception as e:
            logger.error(f"[MediaSubAgent] LLM failed: {e}", exc_info=True)
            return {"action": "media_content_search", "params": {"query": qn, "media_type": "all"}}

    @classmethod
    def _postprocess(cls, result: dict, ctx: Any) -> dict:
        action = str(result.get("action") or "media_content_search").strip()
        params = result.get("params") or {}

        if action not in _VALID_ACTIONS:
            logger.warning(f"[MediaSubAgent] invalid action {action!r} → media_content_search")
            action = "media_content_search"

        if not isinstance(params, dict):
            params = {}

        media_type = str(params.get("media_type") or "all").strip().lower()
        if media_type not in ("video", "audio", "all"):
            media_type = "all"

        if action == "media_content_search":
            query = str(params.get("query") or ctx.question or "").strip()
            if not query:
                query = ctx.question or ""
            logger.debug(f"[MediaSubAgent] → media_content_search(query_chars={len(query)}, media_type={media_type})")
            return {"action": "media_content_search", "params": {"query": query, "media_type": media_type}}

        if action == "media_count":
            logger.info(f"[MediaSubAgent] → media_count(media_type={media_type})")
            return {"action": "media_count", "params": {"media_type": media_type}}

        if action == "media_summarize":
            logger.info(f"[MediaSubAgent] → media_summarize(media_type={media_type})")
            return {"action": "media_summarize", "params": {"media_type": media_type}}

        return {"action": "media_content_search", "params": {"query": ctx.question or "", "media_type": "all"}}
