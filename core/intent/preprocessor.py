from __future__ import annotations

import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class PreprocessorResult:
    """Result from the QueryPreprocessor."""
    __slots__ = ("intercepted", "action", "params", "reason")

    def __init__(self, intercepted: bool, action: str = "", params: dict = None, reason: str = ""):
        self.intercepted = intercepted
        self.action = action
        self.params = params or {}
        self.reason = reason

    def __repr__(self):
        if self.intercepted:
            return f"PreprocessorResult(action={self.action!r}, reason={self.reason!r})"
        return "PreprocessorResult(pass_through)"


class QueryPreprocessor:
    """
    Layer 0 agent: intercepts queries that can be fully resolved
    without intent analysis or LLM calls.
    """

    # ── Greeting detection ────────────────────────────────────────────────
    _GREETINGS = frozenset({
        "你好", "您好", "hi", "hello", "嗨", "在吗", "在不在", "你好啊", "你好哈",
        "你好呀", "哈喽", "嗨喽", "hey", "hello啊",
        "谢谢", "感谢", "多谢", "thank you", "thanks", "thx", "太感谢了", "谢谢你",
        "再见", "拜拜", "bye", "goodbye", "byebye", "see you",
        "好的", "好", "ok", "收到", "明白了", "知道了", "行", "嗯", "恩", "ok了", "好的呢",
        "no need", "never mind", "nevermind", "no thanks", "all set", "cancel that",
        "不用了", "先不用", "算了", "不需要了",
    })

    # ── Capability / help detection ───────────────────────────────────────
    _CAPABILITY_PATTERNS = [
        re.compile(r"^(what|tell me what)\s+(can|do)\s+you\s+(do|help)", re.IGNORECASE),
        re.compile(r"^(你|您)(能做什么|可以做什么|有什么功能|能干嘛|能帮我什么)", re.IGNORECASE),
        re.compile(r"^(你的|您的)(功能|能力|特长)", re.IGNORECASE),
        re.compile(r"^(help|帮助|功能介绍|介绍一下你自己)$", re.IGNORECASE),
        re.compile(r"^(help\s+me|i\s+need\s+help|can\s+you\s+help\s+me)$", re.IGNORECASE),
    ]

    _FILE_OP_HELP_GUARD = re.compile(
        r"\b(find|search|show|list|display|browse|locate|open|get|folder|directory|dir|"
        r"file|files|document|documents|docs|pdf|csv|xlsx|xls|docx|txt|image|images|"
        r"photo|photos|video|videos|audio|recordings?)\b|"
        r"(找|查找|搜索|显示|列出|文件夹|目录|文件|文档|图片|视频|音频)",
        re.IGNORECASE,
    )

    @classmethod
    def is_capability_query(cls, question: str) -> bool:
        q = (question or "").strip()
        if not q:
            return False

        q_stripped = re.sub(r'[！？。，!?.,\s]+$', '', q).strip()
        q_lower = q_stripped.lower()

        for pat in cls._CAPABILITY_PATTERNS:
            if pat.search(q_stripped):
                return True

        if "help" in q_lower or "帮助" in q_stripped:
            # Keep bare help-like prompts as capability, but never steal an actual
            # file/folder operation such as "help me find folder test_csv".
            if cls._FILE_OP_HELP_GUARD.search(q_stripped):
                return False
            helpish = re.fullmatch(
                r"(help|help me|i need help|can you help me|please help|need help)",
                q_lower,
                re.IGNORECASE,
            )
            return bool(helpish)

        return False

    # ── Translate detection ───────────────────────────────────────────────
    _TRANSLATE_PAT = re.compile(
        r'^(please\s+|can you\s+)?'
        r'(in\s+(english|chinese|zh|en|japanese|french|german)'
        r'|translate(\s+(it|this|that|above|below|them))?(\s+to\s+(english|chinese|zh|en))?'
        r'|用(英文|中文|日文|法文)(\s*?(说|回答|写|回复|再说一遍))?'
        r'|翻译(成|为)?(英文|中文)?)'
        r'[\s.!？。]*(above|below|it|this|that|these|those|again)?[\s.!？。]*$',
        re.IGNORECASE,
    )

    # ── DB clear / confirm clear ──────────────────────────────────────────
    _CLEAR_PATTERNS = [
        re.compile(r"^(clear|reset|delete)\s+(all\s+)?(data|files|history|索引|数据)", re.IGNORECASE),
        re.compile(r"^(清空|重置|删除)(所有|全部)?(数据|文件|索引|历史)", re.IGNORECASE),
    ]

    @classmethod
    def preprocess(
        cls,
        question: str,
        *,
        has_history: bool = False,
        total_searchable: int = -1,
    ) -> PreprocessorResult:
        """
        Run all pre-routing checks.

        Returns PreprocessorResult with intercepted=True if the query is fully handled.
        Otherwise returns intercepted=False → continue to IntentAnalyzer.
        """
        q = (question or "").strip()
        if not q:
            return PreprocessorResult(intercepted=False)

        # Normalize for matching
        q_stripped = re.sub(r'[！？。，!?.,\s]+$', '', q).strip().lower()

        # 1. Pure greeting
        if q_stripped in cls._GREETINGS:
            logger.debug(f"[preprocessor] greeting detected query_chars={len(q or '')}")
            return PreprocessorResult(
                intercepted=True,
                action="greeting",
                reason="pure_greeting",
            )

        # 2. Capability query
        if cls.is_capability_query(q):
            logger.debug(f"[preprocessor] capability query_chars={len(q or '')}")
            return PreprocessorResult(
                intercepted=True,
                action="capability",
                reason="capability_query",
            )

        # 3. Translate request (requires conversation history)
        if has_history:
            # Only trigger translate if there's no file-op keyword in the query
            _NO_FILE_OP_PAT = re.compile(
                r'\b(find|search|show|list|count|how\s+many|summarize|总结|搜索|查找|统计|显示|列出)\b',
                re.IGNORECASE,
            )
            if not _NO_FILE_OP_PAT.search(q_stripped):
                m = cls._TRANSLATE_PAT.match(q_stripped)
                if m:
                    tgt = "zh" if re.search(r'(chinese|zh|中文)', q_stripped, re.IGNORECASE) else "en"
                    logger.debug(f"[preprocessor] translate request → {tgt}: query_chars={len(q or '')}")
                    return PreprocessorResult(
                        intercepted=True,
                        action="translate_response",
                        params={"lang": tgt},
                        reason="translate_request",
                    )

        # 4. Zero-scope check (no indexed files, not a translation)
        if total_searchable == 0:
            # Check if it's a translate request even with 0 files (don't intercept)
            if has_history:
                _TRANSLATE_PAT2 = re.compile(
                    r'(translate|翻译|in english|in chinese|用英文|用中文)',
                    re.IGNORECASE,
                )
                if _TRANSLATE_PAT2.search(q):
                    return PreprocessorResult(intercepted=False)
            
            logger.debug(f"[preprocessor] zero-scope (0 searchable files) query_chars={len(q or '')}")
            return PreprocessorResult(
                intercepted=True,
                action="zero_scope_chat",
                reason="no_indexed_files",
            )

        # Not intercepted → pass through to IntentAnalyzer
        return PreprocessorResult(intercepted=False)
