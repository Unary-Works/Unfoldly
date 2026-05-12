"""
MediaFollowupSkill — structured planner for audio/video follow-up work.

The goal is to let the model decide whether the user wants:
  - a topic search inside current media
  - a timestamp lookup
  - a range summary
  - an overall media summary / rewrite

Time parsing remains deterministic when available, but the semantic choice of
operation is model-driven.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.intent.media_query_expert import MediaQueryExpert

_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape"}
_VIDEO_EXTS = {".mp4", ".m4v", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".ts"}
_MEDIA_EXTS = _AUDIO_EXTS | _VIDEO_EXTS
_GENERIC_MEDIA_SUMMARY_PREFIX_RE = re.compile(
    r"\b(?:tell\s+(?:me\s+)?about|describe|summarize|summary\s+of|overview\s+of|"
    r"(?:give\s+me\s+)?(?:an?\s+)?(?:(?:overall|detailed|concise|brief|short)\s+)?(?:explanation|description|summary|overview)\s+of|"
    r"what\s+can\s+be\s+heard(?:\s+in)?|what\s+is\s+heard(?:\s+in)?|"
    r"what\s+can\s+i\s+hear(?:\s+in)?|what\s+is\s+the\s+most\s+important\s+information\s+in|"
    r"what\s+(?:is|are)\s+(?:the\s+)?(?:main\s+)?content\s+of|"
    r"what\s+is\s+(?:in|inside)|what\s+is\s+this\s+.+?\s+about|what\s+does\s+this\s+.+?\s+contain|"
    r"what\s+happens(?:\s+in)?|what\s+is\s+the\s+content\s+of|content\s+of)\b"
    r"|介绍一下|讲讲|概括|总结|概述|主要内容|最重要信息|听到什么|能听到什么"
    r"|内容是什么|是什么内容|有什么内容|里面是什么|里面有什么|讲的是什么|说的是什么|在讲什么|在说什么",
    re.IGNORECASE,
)
_GENERIC_MEDIA_REF_RE = re.compile(
    r"\b(?:the|a|an|selected|this|that|current|audio|video|recording|clip|media|file|files)\b"
    r"|这个音频|这个视频|所选音频|所选视频|当前音频|当前视频|音频文件|视频文件",
    re.IGNORECASE,
)
_GENERIC_MEDIA_OVERVIEW_STOPWORDS = {
    "about",
    "file",
    "files",
    "media",
    "audio",
    "video",
    "recording",
    "recordings",
    "clip",
    "clips",
    "content",
    "contents",
    "main",
    "detailed",
    "detail",
    "explanation",
    "description",
    "overview",
    "overall",
    "first",
    "please",
    "just",
    "give",
    "me",
    "s",
    "selected",
    "current",
    "this",
    "that",
    "the",
    "a",
    "an",
    "in",
    "of",
}


@dataclass(frozen=True)
class MediaFollowupState:
    active_file_names: List[str] = field(default_factory=list)
    last_result_names: List[str] = field(default_factory=list)
    focused_file: str = ""
    media_scope_size: int = 0
    has_selected_media: bool = False
    has_prior_media: bool = False


@dataclass(frozen=True)
class MediaFollowupPlan:
    operation: str
    query: str
    media_type: str = "all"
    file_hint: str = ""
    target_type: str = "audio_content"
    time_sec: Optional[float] = None
    time_end_sec: Optional[float] = None
    reason: str = ""


class MediaFollowupSkill:
    """State and execution helpers for context-bound media follow-ups."""

    @staticmethod
    def _is_media_path(path: str) -> bool:
        return os.path.splitext(str(path or "").strip())[1].lower() in _MEDIA_EXTS

    @classmethod
    def _result_is_media(cls, item: Dict[str, Any]) -> bool:
        category = str(item.get("doc_category") or item.get("doc_category_family") or "").strip().lower()
        file_name = str(item.get("file_name") or item.get("file_path") or "").strip()
        return category in {"audio", "video", "audio/video"} or cls._is_media_path(file_name)

    @classmethod
    def supports_ctx(cls, ctx: Any) -> bool:
        active_paths = list(getattr(ctx, "active_paths", None) or [])
        if any(cls._is_media_path(path) for path in active_paths):
            return True
        last_results = list(getattr(ctx, "last_results", None) or [])
        sample = list(last_results[:12])
        if not sample:
            return False
        media_hits = sum(1 for item in sample if cls._result_is_media(item))
        return media_hits >= max(1, (len(sample) + 1) // 2)

    @classmethod
    def build_state(cls, ctx: Any) -> MediaFollowupState:
        active_paths = list(getattr(ctx, "active_paths", None) or [])
        last_results = list(getattr(ctx, "last_results", None) or [])

        active_media: List[str] = []
        seen_active: set[str] = set()
        for path in active_paths:
            if not cls._is_media_path(path):
                continue
            name = os.path.basename(str(path))
            key = str(path or name).strip()
            if key in seen_active:
                continue
            seen_active.add(key)
            active_media.append(name)
        prior_media: List[str] = []
        seen_prior: set[str] = set()
        for item in last_results[:8]:
            if not cls._result_is_media(item):
                continue
            raw = str(item.get("file_path") or item.get("file_name") or "").strip()
            name = str(item.get("file_name") or os.path.basename(raw) or "").strip()
            key = raw or name
            if key in seen_prior:
                continue
            seen_prior.add(key)
            prior_media.append(name)
        focused_file = ""
        if len(active_media) == 1:
            focused_file = active_media[0]
        elif len(prior_media) == 1:
            focused_file = prior_media[0]

        return MediaFollowupState(
            active_file_names=active_media[:4],
            last_result_names=prior_media[:4],
            focused_file=focused_file,
            media_scope_size=max(len(active_media), len(prior_media)),
            has_selected_media=bool(active_media),
            has_prior_media=bool(prior_media),
        )

    @classmethod
    def render_prompt_block(cls, ctx: Any) -> str:
        if not cls.supports_ctx(ctx):
            return ""
        state = cls.build_state(ctx)
        lines = ["[Media Context]"]
        if state.active_file_names:
            lines.append(f"Selected media: {', '.join(state.active_file_names[:4])}")
        if state.last_result_names:
            lines.append(f"Prior media results: {', '.join(state.last_result_names[:4])}")
        if state.focused_file:
            lines.append(f"Focused media file: {state.focused_file}")
        return "\n".join(lines)

    @classmethod
    def looks_like_generic_overview_query(cls, query: str, *, file_hint: str = "") -> bool:
        raw = str(query or "").strip()
        if not raw:
            return False

        normalized = MediaQueryExpert._normalize_time_query(raw)
        ql = normalized.lower()
        if MediaQueryExpert._extract_time(ql) is not None or MediaQueryExpert._extract_time_range_end(ql) is not None:
            return False
        if MediaQueryExpert.looks_like_explicit_media_file_search(normalized):
            return False
        if not _GENERIC_MEDIA_SUMMARY_PREFIX_RE.search(ql):
            return False

        reduced = ql
        if file_hint:
            reduced = re.sub(re.escape(str(file_hint).lower()), " ", reduced, flags=re.IGNORECASE)
        reduced = _GENERIC_MEDIA_SUMMARY_PREFIX_RE.sub(" ", reduced)
        reduced = _GENERIC_MEDIA_REF_RE.sub(" ", reduced)
        reduced = re.sub(r"[^\w\u4e00-\u9fff]+", " ", reduced)

        tokens = [
            tok
            for tok in reduced.split()
            if tok and tok not in _GENERIC_MEDIA_OVERVIEW_STOPWORDS
        ]
        return not tokens

    @classmethod
    def plan_from_params(
        cls,
        question: str,
        params: Optional[Dict[str, Any]],
        *,
        last_results: Optional[Sequence[Dict[str, Any]]] = None,
        active_paths: Optional[Sequence[str]] = None,
    ) -> MediaFollowupPlan:
        payload = dict(params or {})
        operation = str(payload.get("operation") or "").strip().lower()
        query = str(payload.get("query") or question or "").strip()
        media_type = str(payload.get("media_type") or "all").strip().lower() or "all"
        file_hint = str(payload.get("file_hint") or "").strip()
        target_type = str(payload.get("target_type") or "audio_content").strip() or "audio_content"
        time_sec = payload.get("time_sec")
        time_end_sec = payload.get("time_end_sec")
        reason = str(payload.get("_dispatch_reason") or payload.get("reason") or "").strip()

        if (time_sec is None or (operation in {"time_lookup", "range_summary"} and time_end_sec is None)) and query:
            detected = MediaQueryExpert.analyze(
                query,
                last_results=list(last_results or []),
                llm_service=None,
            ) or {}
            detected_params = dict((detected or {}).get("params") or {})
            if time_sec is None and detected_params.get("time_sec") is not None:
                time_sec = float(detected_params.get("time_sec"))
            if time_end_sec is None and detected_params.get("time_end_sec") is not None:
                time_end_sec = float(detected_params.get("time_end_sec"))
            if not file_hint:
                file_hint = str(detected_params.get("file_hint") or "").strip()
            if detected_params.get("target_type"):
                target_type = str(detected_params.get("target_type"))

        if time_sec is not None:
            time_sec = float(time_sec)
        if time_end_sec is not None:
            time_end_sec = float(time_end_sec)

        if (
            time_sec is None
            and time_end_sec is None
            and cls.looks_like_generic_overview_query(query, file_hint=file_hint)
            and operation not in {"time_lookup", "range_summary"}
        ):
            operation = "summary"

        if not operation:
            if time_sec is not None or time_end_sec is not None:
                operation = "range_summary" if time_end_sec is not None else "time_lookup"
            else:
                ql = query.lower()
                if any(token in ql for token in {"summary", "summarize", "overview", "总结", "概括", "主要内容"}):
                    operation = "summary"
                else:
                    operation = "topic_search"

        if not file_hint:
            active_media = [os.path.basename(str(path)) for path in list(active_paths or []) if cls._is_media_path(path)]
            if len(active_media) == 1:
                file_hint = active_media[0]
            else:
                prior_media = [
                    str(item.get("file_name") or item.get("file_path") or "").strip()
                    for item in list(last_results or [])[:4]
                    if cls._result_is_media(item)
                ]
                if len(prior_media) == 1:
                    file_hint = prior_media[0]

        return MediaFollowupPlan(
            operation=operation,
            query=query,
            media_type=media_type if media_type in {"audio", "video", "all"} else "all",
            file_hint=file_hint,
            target_type=target_type,
            time_sec=time_sec,
            time_end_sec=time_end_sec,
            reason=reason,
        )
