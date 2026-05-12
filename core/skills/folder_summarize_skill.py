"""
FolderSummarizeSkill — scope resolver for "summarize this folder" requests.

Responsibilities:
  1. Detect when active_paths contains a folder (not just individual files).
  2. Build a FolderSummarizePlan that carries the folder path and optional focus filter.
  3. Handle T2 refinements ("focus only on text files in this folder") by parsing
     focus keywords and propagating them to the retrieval scope.

This is a *planner* class — execution is handled by the summarize_all /
process_previous handlers in query_orchestrator.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .selected_summarize_skill import parse_focus_filter, _MOST_IMPORTANT_RE


# ---------------------------------------------------------------------------
# Folder detection
# ---------------------------------------------------------------------------


def extract_folders(paths: Sequence[str]) -> List[str]:
    """Return paths that are directories."""
    return [str(p) for p in (paths or []) if os.path.isdir(str(p))]


def extract_files(paths: Sequence[str]) -> List[str]:
    """Return paths that are regular files."""
    return [str(p) for p in (paths or []) if not os.path.isdir(str(p))]


# ---------------------------------------------------------------------------
# Plan dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FolderSummarizePlan:
    """Resolved plan for a folder summarize request."""

    # Original user question
    question: str

    # Primary folder being summarized
    folder_path: str

    # Any additional individual files selected alongside the folder
    extra_files: List[str] = field(default_factory=list)

    # Optional file-type filter for T2 refinements
    focus_extension: Optional[str] = None

    # True if this is a T2 refinement (prior_action was summarize_all/folder)
    is_refinement: bool = False

    # Language for response generation
    language: str = "en"

    @property
    def folder_name(self) -> str:
        return os.path.basename(self.folder_path.rstrip("/\\")) or self.folder_path

    @property
    def scope_label(self) -> str:
        """Human-readable scope descriptor for prompts."""
        label = f"folder '{self.folder_name}'"
        if self.focus_extension:
            label += f" (filtered to {self.focus_extension} files)"
        return label


# ---------------------------------------------------------------------------
# Skill class
# ---------------------------------------------------------------------------


class FolderSummarizeSkill:
    """
    Planner for folder-level summarize requests.

    Usage:
        if FolderSummarizeSkill.supports(active_paths):
            plan = FolderSummarizeSkill.build_plan(query, active_paths, ...)
            # pass plan to summarize_all handler
    """

    @staticmethod
    def supports(active_paths: Optional[Sequence[str]]) -> bool:
        """True when at least one selected path is a directory."""
        return bool(extract_folders(active_paths or []))

    @classmethod
    def build_plan(
        cls,
        query: str,
        active_paths: Sequence[str],
        *,
        prior_action: str = "",
        language: str = "en",
    ) -> FolderSummarizePlan:
        """
        Build a folder summarize plan.

        Selects the first detected folder from active_paths and collects any
        co-selected individual files. For T2 refinements, parses focus filters.
        """
        _SUMMARIZE_PRIORS = {
            "summarize_all", "summarize_selected", "summarize",
            "process_previous", "folder_summarize",
        }
        is_refinement = prior_action in _SUMMARIZE_PRIORS

        folders = extract_folders(active_paths)
        extra_files = extract_files(active_paths)

        # Use the first detected folder as primary scope
        folder_path = folders[0] if folders else ""

        focus_ext = parse_focus_filter(query) if is_refinement else None

        return FolderSummarizePlan(
            question=str(query or ""),
            folder_path=folder_path,
            extra_files=extra_files,
            focus_extension=focus_ext,
            is_refinement=is_refinement,
            language=str(language or "en"),
        )

    @staticmethod
    def is_support_query(query: str) -> bool:
        """True if the user asks which files in the folder support a conclusion."""
        _SUPPORT_RE = re.compile(
            r"\b(?:which|what)\s+files?\s+(?:in\s+(?:the\s+)?folder\s+)?(?:support|back(?:\s+up)?|evidence|prove)\b",
            re.IGNORECASE,
        )
        return bool(_SUPPORT_RE.search(str(query or "")))

    @staticmethod
    def is_most_important_request(query: str) -> bool:
        return bool(_MOST_IMPORTANT_RE.search(str(query or "")))
