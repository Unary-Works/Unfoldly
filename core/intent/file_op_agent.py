"""
Layer 2B: FileOpAgent
──────────────────────
Called when Layer 1 Router returns "file_op".
Focused LLM call (~220 token prompt, max_tokens=50).

Determines the specific file operation:
  - search:       semantic search with mandatory English query
  - count:        list files by category
  - summarize:    topic summary for a category
  - summarize_all: global overview

Correctness rules preserved from previous bug fixes:
  - 48c1b19: global summarize does NOT trigger process_previous
  - b2a16e3: query MUST be translated to English for search
  - correct_llm_intent guards are applied on top of LLM output
"""
import re
import json
import logging
from typing import Any, List, Dict, Optional

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"search", "count", "summarize", "summarize_all", "view_detail", "open_file"}

# Only honor open_file when the user explicitly uses one of these keywords.
_EXPLICIT_OPEN_KWS = frozenset({
    "打开", "开启", "启动", "open", "launch",
})

# Category canonical mapping for count/summarize
_CAT_ALIASES = {
    "docs": "document", "doc": "document", "documents": "document",
    "pdfs": "document", "pdf": "document",
    "photos": "image", "pictures": "image", "imgs": "image",
    "videos": "video", "video": "video", "movies": "video", "clips": "video",
    "audios": "audio", "audio": "audio", "recordings": "audio", "songs": "audio", "music": "audio",
    "media": "audio/video",
    "sheets": "data", "spreadsheets": "data", "csvs": "data", "excels": "data",
    "slides": "presentation", "presentations": "presentation", "ppts": "presentation",
    "resumes": "resume", "cvs": "resume",
    "reports": "report", "papers": "paper",
    "books": "book", "manuals": "manual",
    "invoices": "invoice", "contracts": "contract",
}


def _normalize_cat(raw: str) -> str:
    r = (raw or "").strip().lower()
    return _CAT_ALIASES.get(r, r)


def _build_selection_preview(active_paths: Optional[list], max_items: int = 5) -> str:
    if not active_paths:
        return "(none)"
    import os
    names = [os.path.basename(p) for p in active_paths[:max_items]]
    suffix = f" ...+{len(active_paths) - max_items} more" if len(active_paths) > max_items else ""
    return ", ".join(names) + suffix


def _build_last_results_preview(last_results: Optional[list], max_items: int = 5) -> str:
    if not last_results:
        return "(none)"
    lines = []
    for r in last_results[:max_items]:
        fn = r.get("file_name") or ""
        lines.append(fn)
    suffix = f" ...+{len(last_results) - max_items} more" if len(last_results) > max_items else ""
    return ", ".join(lines) + suffix


class FileOpAgent:
    """
    Layer 2B: handles file operation intents.
    Returns an intent dict accepted by the main dispatch pipeline.
    """

    @classmethod
    def analyze(cls, ctx: Any) -> dict:  # ctx: IntentContext
        qn = (ctx.question or "").strip()
        lang = ctx.prompt_language or "en"
        active_paths = ctx.active_paths or []
        last_results = ctx.last_results or []

        selection_preview = _build_selection_preview(active_paths)
        last_results_preview = _build_last_results_preview(last_results)
        n_sel = len(active_paths)
        n_last = len(last_results)

        from config.prompts import get_prompt
        prompt = get_prompt("FILE_OP_AGENT_PROMPT", lang).format(
            n_sel=n_sel,
            selection_preview=selection_preview,
            n_last=n_last,
            last_results_preview=last_results_preview,
            query=qn,
        )
        logger.debug(f"[FileOpAgent] prompt_chars={len(prompt)} query_chars={len(qn)}")

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
            logger.warning(f"[FileOpAgent] no JSON: raw_chars={len(raw or '')} → search fallback")
            return {"action": "search", "params": {"query": qn}}
        except Exception as e:
            logger.error(f"[FileOpAgent] LLM failed: {e}", exc_info=True)
            return {"action": "search", "params": {"query": qn}}

    @classmethod
    def _postprocess(cls, result: dict, ctx: Any) -> dict:
        """Apply correctness guards on top of LLM output (mirrors correct_llm_intent logic)."""
        action = str(result.get("action") or "search").strip()
        params = result.get("params") or {}

        if action not in _VALID_ACTIONS:
            logger.warning(f"[FileOpAgent] invalid action {action!r} → search")
            action = "search"

        if action == "search":
            query = str(params.get("query") or ctx.question or "").strip()
            if not query:
                query = ctx.question or ""
            logger.debug(f"[FileOpAgent] → search(query_chars={len(query)})")
            return {"action": "search", "params": {"query": query}}

        if action in ("count", "summarize"):
            cat = _normalize_cat(str(params.get("category") or "all"))
            logger.info(f"[FileOpAgent] → {action}(category={cat!r})")
            return {"action": action, "params": {"category": cat}}

        if action == "summarize_all":
            logger.info("[FileOpAgent] → summarize_all")
            return {"action": "summarize_all", "params": {}}

        if action == "view_detail":
            try:
                idx = int((params or {}).get("index") or 1)
            except (TypeError, ValueError):
                idx = 1
            logger.info(f"[FileOpAgent] → view_detail(index={idx})")
            return {"action": "view_detail", "params": {"index": idx}}

        if action == "open_file":
            file_name = str((params or {}).get("file_name") or "").strip()
            # Otherwise redirect to search → the file will be found and summarized.
            q_low = (ctx.question or "").lower()
            has_explicit_open = any(kw in q_low for kw in _EXPLICIT_OPEN_KWS)
            if has_explicit_open:
                logger.info(f"[FileOpAgent] → open_file(file_name_chars={len(file_name or '')})")
                return {"action": "open_file", "params": {"file_name": file_name}}
            else:
                # Not an explicit open → search for the file and summarize it
                search_query = file_name or ctx.question or ""
                logger.info(
                    f"[FileOpAgent] open_file rejected (no explicit open keyword), "
                    f"redirecting to search query_chars={len(search_query or '')}"
                )
                return {"action": "search", "params": {"query": search_query}}

        return {"action": "search", "params": {"query": ctx.question or ""}}
