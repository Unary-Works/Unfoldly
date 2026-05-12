from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence


@dataclass(frozen=True)
class PathScopeMatcher:
    allow_all: bool
    exact_paths: frozenset[str]
    roots: tuple[str, ...]
    root_prefixes: tuple[str, ...]

    @staticmethod
    def normalize_path(raw_path: Any) -> str:
        try:
            return os.path.normcase(
                os.path.normpath(
                    os.path.abspath(os.path.expanduser(str(raw_path or "")))
                )
            )
        except Exception:
            return ""

    @classmethod
    def from_allowed_paths(
        cls,
        allowed_paths: Optional[Sequence[Any]],
    ) -> "PathScopeMatcher":
        if allowed_paths is None:
            return cls(
                allow_all=True,
                exact_paths=frozenset(),
                roots=(),
                root_prefixes=(),
            )

        normalized: list[str] = []
        for raw in allowed_paths:
            norm = cls.normalize_path(raw)
            if norm:
                normalized.append(norm)

        if not normalized:
            return cls(
                allow_all=False,
                exact_paths=frozenset(),
                roots=(),
                root_prefixes=(),
            )

        compressed_roots: list[str] = []
        for norm in sorted(set(normalized), key=lambda p: (p.count(os.sep), len(p), p)):
            if any(norm == root or norm.startswith(root.rstrip(os.sep) + os.sep) for root in compressed_roots):
                continue
            compressed_roots.append(norm)

        return cls(
            allow_all=False,
            exact_paths=frozenset(normalized),
            roots=tuple(compressed_roots),
            root_prefixes=tuple(root.rstrip(os.sep) + os.sep for root in compressed_roots),
        )

    def allows_file(self, file_path: str) -> bool:
        if self.allow_all:
            return True
        if not self.exact_paths:
            return False
        norm_fp = self.normalize_path(file_path)
        if not norm_fp:
            return False
        if norm_fp in self.exact_paths:
            return True
        return any(norm_fp.startswith(prefix) for prefix in self.root_prefixes)

    def allows_folder(self, folder_path: str) -> bool:
        if self.allow_all:
            return True
        if not self.roots:
            return False
        norm_folder = self.normalize_path(folder_path).rstrip(os.sep)
        if not norm_folder:
            return False
        folder_prefix = norm_folder + os.sep
        for root in self.roots:
            if root == norm_folder:
                return True
            if root.startswith(folder_prefix):
                return True
            if norm_folder.startswith(root.rstrip(os.sep) + os.sep):
                return True
        return False


def ensure_path_scope_matcher(
    allowed_paths: Optional[Sequence[Any] | PathScopeMatcher],
) -> PathScopeMatcher:
    if isinstance(allowed_paths, PathScopeMatcher):
        return allowed_paths
    return PathScopeMatcher.from_allowed_paths(allowed_paths)


def filter_sources_to_scope(
    sources: Optional[Iterable[dict[str, Any]]],
    allowed_paths: Optional[Sequence[Any] | PathScopeMatcher],
    *,
    keep_matching_folders: bool = False,
) -> list[dict[str, Any]]:
    matcher = ensure_path_scope_matcher(allowed_paths)
    raw_sources = list(sources or [])
    if matcher.allow_all:
        return raw_sources

    filtered: list[dict[str, Any]] = []
    for source in raw_sources:
        if not isinstance(source, dict):
            continue
        file_path = str(source.get("file_path") or "").strip()
        if not file_path:
            continue

        icon_type = str(source.get("iconType") or source.get("type") or "").strip().lower()
        doc_category = str(source.get("doc_category") or source.get("category") or "").strip().lower()
        is_folder_like = icon_type == "folder" or doc_category == "folder"

        if is_folder_like:
            if keep_matching_folders and matcher.allows_folder(file_path):
                filtered.append(source)
            continue

        if matcher.allows_file(file_path):
            filtered.append(source)

    return filtered
