"""
CountExpert — micro-agent for detecting "how many files" queries.

Simple binary check: "how many" + file term → count(all).
Replaces fp1 regex in intent_analyzer.py.
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


class CountExpert:
    """Detects explicit count/stats queries."""

    @classmethod
    def is_how_many_files(cls, query: str) -> bool:
        """Check if query is 'how many files/documents'."""
        ql = (query or "").lower()
        direct_match = (
            bool(re.search(r"\bhow\s+many\b", ql))
            and bool(re.search(
                r"\b(files?|documents?|docs?|sources?|"
                r"resumes?|cvs?|invoices?|receipts?|reports?|papers?|"
                r"photos?|images?|pictures?|screenshots?|"
                r"recordings?|videos?|audios?|"
                r"contracts?|books?|manuals?|slides?|presentations?|"
                r"spreadsheets?|datasets?|tables?|items?)\b", ql))
        )
        if direct_match:
            return True

        if not re.search(r"\bhow\s+many\b", ql):
            return False

        try:
            from core.retrieval.category_engine import match_dynamic_category_from_query

            return bool(match_dynamic_category_from_query(query, refresh_if_missing=True))
        except Exception:
            return False

    @classmethod
    def to_intent(cls) -> dict:
        logger.info("[count_expert] how-many-files → count(all)")
        return {"action": "count", "params": {"category": "all"}, "confidence": 0.95}
