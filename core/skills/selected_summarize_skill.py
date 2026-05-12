"""
SelectedSummarizeSkill — scope resolver for "summarize my selected files".

Responsibilities:
  1. Confirm that active_paths has selected files (expose_condition guard).
  2. Build a SummarizePlan that carries the exact file list and optional focus filter.
  3. Parse T2 refinement queries ("focus only on text files", "only show PDFs") into
     a filter so process_previous handlers can narrow the file set without a new search.

This is a *planner* class only — execution is handled by the summarize_all / process_previous
handlers in query_orchestrator. The plan is passed as structured context.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# Focus-filter extraction
# ---------------------------------------------------------------------------

_EXT_ALIASES: dict[str, str] = {
    "text": ".txt",
    "txt": ".txt",
    "pdf": ".pdf",
    "word": ".docx",
    "doc": ".docx",
    "docx": ".docx",
    "image": ".png",
    "images": ".png",
    "photo": ".jpg",
    "photos": ".jpg",
    "png": ".png",
    "jpg": ".jpg",
    "jpeg": ".jpg",
    "csv": ".csv",
    "excel": ".xlsx",
    "xlsx": ".xlsx",
    "audio": ".mp3",
    "mp3": ".mp3",
    "video": ".mp4",
    "mp4": ".mp4",
    "ppt": ".pptx",
    "pptx": ".pptx",
    "slides": ".pptx",
}

_EXT_GROUPS: dict[str, list[str]] = {
    "text": [".txt", ".md", ".pdf", ".doc", ".docx", ".rtf"],
    "document": [".txt", ".md", ".pdf", ".doc", ".docx", ".rtf"],
    "documents": [".txt", ".md", ".pdf", ".doc", ".docx", ".rtf"],
    "image": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"],
    "images": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"],
    "picture": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"],
    "pictures": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"],
    "photo": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"],
    "photos": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"],
    "audio": [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"],
    "audios": [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"],
    "video": [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".ts"],
    "videos": [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".ts"],
    "spreadsheet": [".csv", ".tsv", ".xls", ".xlsx", ".numbers"],
    "spreadsheets": [".csv", ".tsv", ".xls", ".xlsx", ".numbers"],
    "excel": [".csv", ".tsv", ".xls", ".xlsx"],
    "slides": [".ppt", ".pptx", ".key"],
    "presentation": [".ppt", ".pptx", ".key"],
    "presentations": [".ppt", ".pptx", ".key"],
}

_FOCUS_TRIGGER = re.compile(
    r"\b(?:focus(?:\s+only)?(?:\s+on)?|look\s+at(?:\s+just)?|only(?:\s+show(?:\s+me)?)?|"
    r"just|filter(?:\s+(?:to|by))?|narrow(?:\s+(?:to|down))?|show\s+(?:only|me)|"
    r"switch\s+to|change\s+to)\b",
    re.IGNORECASE,
)

_STOP_WORDS = {
    "a", "an", "the", "my", "me", "only", "just", "please", "now", "to", "by",
    "of", "in", "at", "on", "for", "with", "and", "or", "that", "those",
    "these", "this", "it", "all", "some", "any", "i", "you",
}

_MOST_IMPORTANT_RE = re.compile(
    r"\b(?:most\s+important|most\s+relevant|most\s+detailed|strongest|best|top)\b",
    re.IGNORECASE,
)


def parse_focus_filter(query: str) -> Optional[str]:
    """
    Extract a file-type focus keyword from a refinement query.

    Returns a canonical extension string like '.pdf', or None if not found.
    Strategy:
      1. Check if the query contains a focus-trigger phrase.
      2. Scan ALL tokens (not just after trigger) for a known file-type alias.
    Examples:
      "focus only on text files" -> ".txt"
      "now only images"          -> ".png"
      "now look at just PDFs"    -> ".pdf"
      "only show me word docs"   -> ".docx"
      "filter by excel"          -> ".xlsx"
    """
    if not _FOCUS_TRIGGER.search(query):
        return None
    extensions = parse_focus_extensions(query)
    if extensions:
        return extensions[0]
    return None


def parse_focus_extensions(query: str) -> List[str]:
    """Return canonical extension filters for a refinement query."""
    if not _FOCUS_TRIGGER.search(query):
        return []
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    collected: List[str] = []
    seen = set()
    for tok in tokens:
        if tok in _STOP_WORDS:
            continue
        candidates: List[str] = []
        # try exact token
        if tok in _EXT_GROUPS:
            candidates = list(_EXT_GROUPS[tok])
        elif tok in _EXT_ALIASES:
            candidates = [_EXT_ALIASES[tok]]
        else:
            singular = tok.rstrip("s")
            if singular and singular in _EXT_GROUPS:
                candidates = list(_EXT_GROUPS[singular])
            elif singular and singular in _EXT_ALIASES:
                candidates = [_EXT_ALIASES[singular]]
        for ext in candidates:
            if ext not in seen:
                seen.add(ext)
                collected.append(ext)
    return collected


def filter_paths_by_extension(
    paths: Sequence[str], extension: str
) -> List[str]:
    """Return only paths whose extension matches (case-insensitive)."""
    ext_lower = extension.lower()
    return [p for p in paths if os.path.splitext(str(p))[1].lower() == ext_lower]


# ---------------------------------------------------------------------------
# Plan dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectedSummarizePlan:
    """Resolved plan for a selected-files summarize request."""

    # Original user question
    question: str

    # The files to summarize (post-filter)
    resolved_paths: List[str] = field(default_factory=list)

    # If the user asked to narrow by file type, the extension used
    focus_extension: Optional[str] = None

    # True if this is a T2 refinement (not the initial summarize request)
    is_refinement: bool = False

    # Language for response generation
    language: str = "en"

    @property
    def has_files(self) -> bool:
        return bool(self.resolved_paths)

    @property
    def file_count(self) -> int:
        return len(self.resolved_paths)


# ---------------------------------------------------------------------------
# Skill class
# ---------------------------------------------------------------------------


class SelectedSummarizeSkill:
    """
    Planner for summarize-selected-files requests.

    Usage:
        if SelectedSummarizeSkill.supports(active_paths):
            plan = SelectedSummarizeSkill.build_plan(query, active_paths, ...)
            # pass plan to summarize_all handler
    """

    @staticmethod
    def supports(active_paths: Optional[Sequence[str]]) -> bool:
        """True when there are user-selected files."""
        return bool(active_paths)

    @classmethod
    def build_plan(
        cls,
        query: str,
        active_paths: Sequence[str],
        *,
        prior_action: str = "",
        language: str = "en",
    ) -> SelectedSummarizePlan:
        """
        Build a summarize plan for the currently selected files.

        For T2 refinements (prior_action == 'summarize_all' or 'summarize_selected'),
        also parse focus filters to narrow the file set.
        """
        _SUMMARIZE_PRIORS = {"summarize_all", "summarize_selected", "summarize", "process_previous"}
        is_refinement = prior_action in _SUMMARIZE_PRIORS

        focus_ext = parse_focus_filter(query) if is_refinement else None
        if focus_ext:
            resolved = filter_paths_by_extension(active_paths, focus_ext)
            if not resolved:
                # Filter produced empty set — fall back to all selected files
                resolved = list(active_paths)
                focus_ext = None
        else:
            resolved = list(active_paths)

        return SelectedSummarizePlan(
            question=str(query or ""),
            resolved_paths=resolved,
            focus_extension=focus_ext,
            is_refinement=is_refinement,
            language=str(language or "en"),
        )

    @staticmethod
    def is_most_important_request(query: str) -> bool:
        """True if user asks for the single most important / strongest file."""
        return bool(_MOST_IMPORTANT_RE.search(str(query or "")))
