"""
FollowupClassifier — binary LLM classifier: "is this a followup?"

Used by ContinuationGroup when rules produce uncertain signal.
~60 token prompt, max_tokens=5, output A (followup) or B (new request).
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class FollowupClassifier:
    """Binary LLM: followup (A) or new request (B)."""

    _MIN_WORDS = 3
    _MAX_WORDS = 18

    @classmethod
    def should_activate(cls, query: str, last_results: Optional[List[dict]]) -> bool:
        if not last_results:
            return False
        wc = len([w for w in (query or "").split() if w])
        return cls._MIN_WORDS <= wc <= cls._MAX_WORDS

    @classmethod
    def is_followup(
        cls,
        query: str,
        last_results: List[dict],
        llm_service: Any,
        *,
        prior_action: str = "",
        lang: str = "en",
    ) -> bool:
        """Return True = followup, False = new request. Defaults to False on failure."""
        if not llm_service:
            return False

        file_names = ", ".join(
            os.path.basename(str(r.get("file_name") or r.get("name") or ""))
            for r in last_results[:4]
        )
        n = len(last_results)
        prompt = cls._build_prompt(query, lang=lang, prior_action=prior_action or "search",
                                   n_results=n, file_names=file_names)
        try:
            raw = (llm_service.generate(prompt, history=[], system_prompt=None) or "").strip()
            result = raw[:5].upper().startswith("A")
            logger.info(f"[FollowupClassifier] {'followup' if result else 'new_request'} raw={raw[:10]!r}")
            return result
        except Exception as e:
            logger.warning(f"[FollowupClassifier] LLM error: {e} -> new_request")
            return False

    @staticmethod
    def _build_prompt(query, *, lang, prior_action, n_results, file_names) -> str:
        if lang.startswith("zh"):
            return (
                f"判断是跟进(A)还是新请求(B)。\n"
                f"[上次]: {prior_action}（{n_results}个文件: {file_names}）\n"
                f"[用户]: \"{query}\"\n"
                "A=追问上次结果  B=新搜索\n只输出A或B:"
            )
        return (
            f"Followup(A) or new request(B)?\n"
            f"[Prior]: {prior_action} ({n_results} files: {file_names})\n"
            f"[User]: \"{query}\"\n"
            "A=followup on prior  B=new search\nOutput A or B:"
        )

