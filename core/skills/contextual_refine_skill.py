"""
ContextualRefineSkill — shared scope/state helpers for follow-ups that stay
inside the current conversational file context.

This skill is intentionally broader than the old split between:
  - selected files
  - selected folders
  - previous result-set follow-ups

The model should pick the semantic operation, while this module exposes a
stable scope vocabulary and prompt context so execution does not fall back to
regex-only routing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .folder_summarize_skill import extract_files, extract_folders
from .selected_summarize_skill import parse_focus_extensions, parse_focus_filter

_MEDIA_EXTS = {
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv",
}

_TEXTISH_EXTS = {
    ".txt", ".md", ".pdf", ".doc", ".docx", ".rtf", ".ppt", ".pptx",
    ".xls", ".xlsx", ".csv", ".tsv", ".json", ".yaml", ".yml",
}


@dataclass(frozen=True)
class CandidateScope:
    scope_id: str
    kind: str
    label: str
    size: int
    modality: str = "mixed"


@dataclass(frozen=True)
class ContextualRefineState:
    prior_user_query: str = ""
    prior_answer_excerpt: str = ""
    candidate_scopes: List[CandidateScope] = field(default_factory=list)
    selected_paths: List[str] = field(default_factory=list)
    prior_result_count: int = 0
    focused_file: str = ""


@dataclass(frozen=True)
class ContextualRefinePlan:
    scope: str
    operation: str
    query: str
    focus_extension: Optional[str] = None
    focus_extensions: List[str] = field(default_factory=list)
    rewrite_mode: str = ""
    file_hint: str = ""
    reason: str = ""


class ContextualRefineSkill:
    """Build state and execution hints for context-bound follow-up work."""

    @staticmethod
    def supports_ctx(ctx: Any) -> bool:
        return bool(getattr(ctx, "active_paths", None) or getattr(ctx, "last_results", None))

    @staticmethod
    def _normalize_extension(raw_ext: str) -> str:
        ext = str(raw_ext or "").strip().lower()
        if not ext:
            return ""
        return ext if ext.startswith(".") else f".{ext}"

    @staticmethod
    def _infer_path_modality(paths: Sequence[str]) -> str:
        if not paths:
            return "mixed"
        ext_hits = {
            ContextualRefineSkill._normalize_extension(os.path.splitext(str(p))[1])
            for p in paths
            if str(p or "").strip()
        }
        has_media = any(ext in _MEDIA_EXTS for ext in ext_hits)
        has_text = any(ext in _TEXTISH_EXTS for ext in ext_hits)
        if has_media and not has_text:
            return "media"
        if has_text and not has_media:
            return "document"
        return "mixed"

    @staticmethod
    def _infer_result_modality(results: Sequence[Dict[str, Any]]) -> str:
        if not results:
            return "mixed"
        media_hits = 0
        text_hits = 0
        sample = list(results[:12])
        for item in sample:
            category = str(item.get("doc_category") or item.get("doc_category_family") or "").strip().lower()
            file_name = str(item.get("file_name") or item.get("file_path") or "").strip()
            ext = os.path.splitext(file_name)[1].lower()
            if ext in _MEDIA_EXTS or category in {"audio", "video", "audio/video"}:
                media_hits += 1
            elif ext in _TEXTISH_EXTS or category in {"document", "data", "pdf", "report", "paper"}:
                text_hits += 1
        if media_hits and not text_hits:
            return "media"
        if text_hits and not media_hits:
            return "document"
        return "mixed"

    @classmethod
    def _normalize_extensions(cls, raw_extensions: Sequence[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for raw_ext in raw_extensions or []:
            ext = cls._normalize_extension(str(raw_ext or ""))
            if ext and ext not in seen:
                seen.add(ext)
                normalized.append(ext)
        return normalized

    @staticmethod
    def _looks_like_focus_subset_request(question: str) -> bool:
        ql = str(question or "").lower()
        if not ql:
            return False
        return bool(
            "focus only on" in ql
            or "only on" in ql
            or "just the" in ql
            or "only show" in ql
            or "只看" in ql
            or "只聚焦" in ql
            or "只关注" in ql
        )

    @staticmethod
    def _last_user_query(history: Sequence[Dict[str, Any]]) -> str:
        for msg in reversed(list(history or [])):
            q = str(msg.get("q") or "").strip()
            if q:
                return q[:120]
            if msg.get("role") == "user":
                content = str(msg.get("content") or "").strip()
                if content:
                    return content[:120]
        return ""

    @staticmethod
    def _last_answer_excerpt(history: Sequence[Dict[str, Any]]) -> str:
        for msg in reversed(list(history or [])):
            answer = str(msg.get("a") or msg.get("content") or "").strip()
            role = str(msg.get("role") or "").strip().lower()
            if answer and role in {"assistant", ""}:
                return " ".join(answer.split())[:220]
        return ""

    @classmethod
    def build_state(cls, ctx: Any, *, include_active_scope: bool = True) -> ContextualRefineState:
        history = list(getattr(ctx, "history", None) or [])
        active_paths = list(getattr(ctx, "active_paths", None) or []) if include_active_scope else []
        last_results = list(getattr(ctx, "last_results", None) or [])

        candidate_scopes: List[CandidateScope] = []
        selected_files = extract_files(active_paths)
        selected_folders = extract_folders(active_paths)

        if selected_files:
            candidate_scopes.append(
                CandidateScope(
                    scope_id="selected_items",
                    kind="selected_items",
                    label=f"{len(selected_files)} selected file(s)",
                    size=len(selected_files),
                    modality=cls._infer_path_modality(selected_files),
                )
            )

        if selected_folders:
            primary_folder = os.path.basename(str(selected_folders[0]).rstrip("/\\")) or str(selected_folders[0])
            extra = len(selected_folders) - 1
            label = f"folder '{primary_folder}'" + (f" (+{extra} more folders)" if extra > 0 else "")
            candidate_scopes.append(
                CandidateScope(
                    scope_id="selected_folder",
                    kind="selected_folder",
                    label=label,
                    size=len(selected_folders),
                    modality=cls._infer_path_modality(selected_folders),
                )
            )

        if last_results:
            candidate_scopes.append(
                CandidateScope(
                    scope_id="last_results",
                    kind="last_results",
                    label=f"{len(last_results)} previous result file(s)",
                    size=len(last_results),
                    modality=cls._infer_result_modality(last_results),
                )
            )

        focused_file = ""
        if len(last_results) == 1:
            focused_file = str(last_results[0].get("file_name") or last_results[0].get("file_path") or "").strip()
        elif len(active_paths) == 1:
            focused_file = os.path.basename(str(active_paths[0]).rstrip("/\\"))

        return ContextualRefineState(
            prior_user_query=cls._last_user_query(history),
            prior_answer_excerpt=cls._last_answer_excerpt(history),
            candidate_scopes=candidate_scopes,
            selected_paths=active_paths,
            prior_result_count=len(last_results),
            focused_file=focused_file,
        )

    @classmethod
    def render_prompt_block(cls, ctx: Any, *, include_active_scope: bool = True) -> str:
        state = cls.build_state(ctx, include_active_scope=include_active_scope)
        if not state.candidate_scopes:
            return ""

        lines = ["[Contextual Scope]"]
        if state.prior_user_query:
            lines.append(f"Prior user query: {state.prior_user_query}")
        if state.prior_answer_excerpt:
            lines.append(f"Prior answer excerpt: {state.prior_answer_excerpt}")
        if state.focused_file:
            lines.append(f"Focused file: {state.focused_file}")
        lines.append("Candidate scopes:")
        for scope in state.candidate_scopes:
            lines.append(
                f"- {scope.scope_id}: {scope.label} (size={scope.size}, modality={scope.modality})"
            )
        return "\n".join(lines)

    @classmethod
    def plan_from_params(
        cls,
        question: str,
        params: Optional[Dict[str, Any]],
        *,
        active_paths: Optional[Sequence[str]] = None,
        last_results: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> ContextualRefinePlan:
        payload = dict(params or {})
        scope = str(payload.get("scope") or payload.get("_scope_kind") or "").strip()
        if not scope:
            if extract_folders(active_paths or []):
                scope = "selected_folder"
            elif active_paths:
                scope = "selected_items"
            elif last_results:
                scope = "last_results"
            else:
                scope = "last_results"

        operation = str(payload.get("operation") or payload.get("_context_operation") or "").strip().lower()
        rewrite_mode = str(payload.get("rewrite_mode") or "").strip().lower()
        if not operation:
            if rewrite_mode:
                operation = "rewrite"
            elif payload.get("query"):
                operation = "qa"
            else:
                operation = "summary"

        focus_extensions = cls._normalize_extensions(payload.get("focus_extensions") or [])
        focus_extension = cls._normalize_extension(str(payload.get("focus_extension") or ""))
        if focus_extension and focus_extension not in focus_extensions:
            focus_extensions = [focus_extension, *focus_extensions]
        if not focus_extensions:
            ext_csv = str(payload.get("file_extensions") or "").strip()
            if ext_csv:
                focus_extensions = cls._normalize_extensions(ext_csv.split(","))
        if not focus_extensions:
            focus_extensions = cls._normalize_extensions(parse_focus_extensions(question))
        if not focus_extensions:
            focus_extension = cls._normalize_extension(parse_focus_filter(question) or "")
            if focus_extension:
                focus_extensions = [focus_extension]
        focus_extension = focus_extensions[0] if focus_extensions else (focus_extension or "")

        if not rewrite_mode:
            ql = str(question or "").lower()
            if any(token in ql for token in {"shorter", "brief", "简短", "更短"}):
                rewrite_mode = "shorter"
            elif any(token in ql for token in {"more detail", "detailed", "更详细", "详细"}):
                rewrite_mode = "more_detail"
            elif any(token in ql for token in {"support", "evidence", "依据", "证据"}):
                rewrite_mode = "supporting_files"
            elif cls._looks_like_focus_subset_request(question):
                rewrite_mode = "focus"

        if rewrite_mode in {"shorter", "more_detail", "supporting_files", "focus"} and operation in {"summary", "qa", "list"}:
            operation = "rewrite"
        elif focus_extensions and operation == "list" and cls._looks_like_focus_subset_request(question):
            operation = "rewrite"
        elif operation == "qa" and any(token in str(question or "").lower() for token in {"most important", "most relevant", "most detailed", "best", "top"}):
            operation = "rewrite"

        query = str(payload.get("query") or question or "").strip()
        file_hint = str(payload.get("file_hint") or payload.get("focused_file") or "").strip()
        reason = str(payload.get("_dispatch_reason") or payload.get("reason") or "").strip()

        return ContextualRefinePlan(
            scope=scope,
            operation=operation,
            query=query,
            focus_extension=focus_extension or None,
            focus_extensions=focus_extensions,
            rewrite_mode=rewrite_mode,
            file_hint=file_hint,
            reason=reason,
        )

    @staticmethod
    def filter_results(
        results: Sequence[Dict[str, Any]],
        *,
        focus_extension: Optional[str],
        focus_extensions: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        extensions = ContextualRefineSkill._normalize_extensions(
            list(focus_extensions or []) + ([focus_extension] if focus_extension else [])
        )
        if not extensions:
            return list(results or [])
        allowed = set(extensions)
        filtered: List[Dict[str, Any]] = []
        for item in list(results or []):
            file_name = str(item.get("file_name") or item.get("file_path") or "").strip()
            if os.path.splitext(file_name)[1].lower() in allowed:
                filtered.append(item)
        return filtered
