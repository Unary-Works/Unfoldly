"""
Response Formatter — utilities for rendering search results into user-facing output.

Extracted from dispatch.py for reusability. Contains:
  - icon_type_for_path: Map file extensions to display icon types
  - ellipsis_at_word_boundary: Smart text truncation
  - clip_one_line: Single-line truncation
  - normalize_summary_text: Clean up summary text for display
  - display_summary_cap: Max chars for displayed summaries
  - build_clickable_file_link: Create clickable file reference links

These are pure utility functions — no LLM dependency.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Any


def icon_type_for_path(p: str) -> str:
    """Map file extension to a display icon type."""
    ext = os.path.splitext(str(p or ""))[-1].lower()
    _MAP = {
        ".pdf": "pdf",
        ".doc": "word", ".docx": "word",
        ".xls": "excel", ".xlsx": "excel",
        ".ppt": "ppt", ".pptx": "ppt",
        ".txt": "text", ".md": "text", ".rtf": "text",
        ".jpg": "image", ".jpeg": "image", ".png": "image",
        ".gif": "image", ".bmp": "image", ".webp": "image",
        ".heic": "image", ".tiff": "image", ".svg": "image",
        ".mp3": "audio", ".wav": "audio", ".flac": "audio",
        ".aac": "audio", ".m4a": "audio", ".ogg": "audio",
        ".wma": "audio", ".aiff": "audio",
        ".mp4": "video", ".mov": "video", ".avi": "video",
        ".mkv": "video", ".wmv": "video", ".flv": "video",
        ".zip": "archive", ".rar": "archive", ".7z": "archive",
        ".tar": "archive", ".gz": "archive",
        ".py": "code", ".js": "code", ".ts": "code",
        ".java": "code", ".cpp": "code", ".c": "code",
        ".html": "code", ".css": "code", ".json": "code",
        ".xml": "code", ".yaml": "code", ".yml": "code",
        ".csv": "data", ".tsv": "data",
        ".db": "data", ".sqlite": "data",
    }
    return _MAP.get(ext, "file")


def ellipsis_at_word_boundary(text: str, max_len: int) -> str:
    """Truncate text at word boundary with ellipsis."""
    text = str(text or "").strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rfind(" ")
    if cut < max_len * 0.4:
        cut = max_len
    truncated = text[:cut].rstrip(" .,;:!?-")
    return truncated + "..." if truncated != text else truncated


def clip_one_line(text: Any, limit: int = 220) -> str:
    """Clip text to one line with a character limit."""
    s = str(text or "").replace("\n", " ").replace("\r", " ").strip()
    if len(s) <= limit:
        return s
    return s[:limit].rstrip() + "..."


def normalize_summary_text(text: Any) -> str:
    """Clean up summary text — remove markdown artifacts, normalize whitespace."""
    s = str(text or "").strip()
    # Remove 'Summary:' prefix
    if s.lower().startswith("summary:"):
        s = s[8:].strip()
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def display_summary_cap() -> int:
    """Max characters for displayed summaries (configurable via env)."""
    try:
        return max(60, int(os.getenv("DISPLAY_SUMMARY_CAP", "300")))
    except (ValueError, TypeError):
        return 300


def fallback_brief_text_chunk_cap() -> int:
    """Max characters for fallback brief text chunks."""
    try:
        return max(50, int(os.getenv("FALLBACK_BRIEF_CAP", "180")))
    except (ValueError, TypeError):
        return 180


def build_clickable_file_link(file_name: str, file_path: str) -> str:
    """Build a clickable file reference link for chat UI."""
    name = str(file_name or "").strip()
    path = str(file_path or "").strip()
    if not name and path:
        name = os.path.basename(path)
    if not name:
        name = "Unknown File"
    # Use markdown-style clickable link
    if path:
        return f"[{name}]({path})"
    return name


def softer_indexer_summary_hard_cap(text: str) -> str:
    """Apply a softer hard cap on indexer summaries for display."""
    s = str(text or "").strip()
    cap = display_summary_cap()
    if len(s) <= cap:
        return s
    return ellipsis_at_word_boundary(s, cap)
