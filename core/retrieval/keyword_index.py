"""
Keyword index manager and backend abstraction for lexical retrieval.

TODO(scale guidance, heuristic bands; revisit with production benchmarks):
- If future file count reaches hundreds of thousands or millions, do not pursue GPU BM25.
- Recommended directions by file-count band:
  - In-memory BM25 + sidecar: up to roughly 100k files.
  - SQLite FTS5: roughly 50k-300k files when you want zero extra runtime and tight SQL/filter integration.
  - Tantivy: roughly 100k-5M files for local/desktop embedded search with stronger lexical scalability.
  - Lucene: roughly 500k+ files when a JVM/runtime dependency is acceptable and advanced search features matter.
  - Custom inverted index: multi-million files or highly specialized ranking/storage constraints.

This module keeps the orchestration layer backend-agnostic so we can switch from BM25
to Tantivy / Lucene / SQLite FTS5 later without rewriting query routing.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from utils.logger import get_logger
from .path_scope import ensure_path_scope_matcher

logger = get_logger()

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi

    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False
    _BM25Okapi = None  # type: ignore


_KEYWORD_INDEX_SCHEMA_VERSION = 5


@dataclass
class KeywordIndexRecord:
    chroma_id: str
    file_path: str
    normalized_path: str = ""
    file_name: str = ""
    category: str = ""
    file_extension: str = ""
    tokens: List[str] = field(default_factory=list)
    content_preview: str = ""


class KeywordBackend(Protocol):
    backend_name: str

    def score(
        self,
        query_tokens: Sequence[str],
        candidate_indices: Optional[Sequence[int]] = None,
    ) -> List[Tuple[int, float]]:
        ...


class BM25KeywordBackend:
    backend_name = "bm25"

    def __init__(
        self,
        records: List[KeywordIndexRecord],
        local_scope_max_docs: int = 400,
        local_scope_min_total_docs: int = 10000,
    ):
        if not _HAS_BM25:
            raise RuntimeError("rank_bm25 unavailable")
        self._records = list(records or [])
        self._local_scope_max_docs = max(1, int(local_scope_max_docs or 1))
        self._local_scope_min_total_docs = max(1, int(local_scope_min_total_docs or 1))
        self._tokenized_corpus: List[List[str]] = [
            list(rec.tokens or [""]) or [""] for rec in self._records
        ]
        self._bm25 = _BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    def _score_subset(
        self,
        query_tokens: Sequence[str],
        candidate_indices: Sequence[int],
    ) -> List[Tuple[int, float]]:
        local_corpus = [self._tokenized_corpus[idx] for idx in candidate_indices]
        local_bm25 = _BM25Okapi(local_corpus)
        local_scores = local_bm25.get_scores(list(query_tokens))
        results: List[Tuple[int, float]] = []
        for local_idx, score in enumerate(local_scores):
            if score > 0:
                results.append((int(candidate_indices[local_idx]), float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def score(
        self,
        query_tokens: Sequence[str],
        candidate_indices: Optional[Sequence[int]] = None,
    ) -> List[Tuple[int, float]]:
        if not query_tokens or self._bm25 is None:
            return []

        candidate_list = list(candidate_indices) if candidate_indices is not None else None
        if candidate_list is not None and not candidate_list:
            return []

        if (
            candidate_list is not None
            and len(candidate_list) < len(self._records)
            and len(candidate_list) <= self._local_scope_max_docs
            and len(self._records) >= self._local_scope_min_total_docs
        ):
            return self._score_subset(query_tokens, candidate_list)

        scores = self._bm25.get_scores(list(query_tokens))
        iter_indices = candidate_list if candidate_list is not None else range(len(scores))
        results: List[Tuple[int, float]] = []
        for idx in iter_indices:
            score = scores[idx]
            if score > 0:
                results.append((int(idx), float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class KeywordIndexManager:
    def __init__(
        self,
        *,
        backend_name: str,
        sidecar_path: str,
        build_records_fn: Callable[[], List[KeywordIndexRecord]],
        current_chunk_count_fn: Callable[[], int],
        path_allow_fn: Callable[[str, Optional[List[str]]], bool],
        local_scope_max_docs: int = 400,
        rebuild_delay_sec: float = 2.0,
    ):
        self.backend_name = str(backend_name or "bm25").strip().lower()
        self.sidecar_path = os.path.abspath(os.path.expanduser(sidecar_path))
        self._build_records_fn = build_records_fn
        self._current_chunk_count_fn = current_chunk_count_fn
        self._path_allow_fn = path_allow_fn
        self._local_scope_max_docs = max(1, int(local_scope_max_docs or 1))
        self._local_scope_min_total_docs = max(
            1,
            int(os.getenv("FILEAGENT_KEYWORD_LOCAL_SCOPE_MIN_TOTAL_DOCS", "10000") or 10000),
        )
        self._rebuild_delay_sec = max(0.0, float(rebuild_delay_sec or 0.0))

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._backend: Optional[KeywordBackend] = None
        self._records: List[KeywordIndexRecord] = []
        self._records_chunk_count: int = -1
        self._dirty_generation: int = 0
        self._ready_generation: int = -1
        self._building_generation: int = -1
        self._rebuild_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    @staticmethod
    def _normalize_path(path: str) -> str:
        try:
            return os.path.normcase(
                os.path.normpath(
                    os.path.abspath(os.path.expanduser(str(path or "")))
                )
            )
        except Exception:
            return ""

    def _get_backend(self, records: List[KeywordIndexRecord]) -> KeywordBackend:
        if self.backend_name == "bm25":
            return BM25KeywordBackend(
                records,
                local_scope_max_docs=self._local_scope_max_docs,
                local_scope_min_total_docs=self._local_scope_min_total_docs,
            )
        raise NotImplementedError(f"Unsupported keyword backend: {self.backend_name}")

    def _serialize_payload(
        self,
        records: List[KeywordIndexRecord],
        chunk_count: int,
    ) -> Dict[str, Any]:
        return {
            "schema_version": _KEYWORD_INDEX_SCHEMA_VERSION,
            "backend_name": self.backend_name,
            "chunk_count": int(chunk_count),
            "record_count": len(records),
            "created_at": time.time(),
            "records": [asdict(rec) for rec in records],
        }

    def _persist_sidecar(
        self,
        records: List[KeywordIndexRecord],
        chunk_count: int,
    ) -> None:
        try:
            os.makedirs(os.path.dirname(self.sidecar_path), exist_ok=True)
            payload = self._serialize_payload(records, chunk_count)
            fd, tmp_path = tempfile.mkstemp(
                prefix="keyword_index_",
                suffix=".tmp",
                dir=os.path.dirname(self.sidecar_path),
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, self.sidecar_path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"[KeywordIndex] persist sidecar failed: {e}")

    def _load_sidecar(
        self,
        current_chunk_count: int,
    ) -> bool:
        """Load sidecar only if chunk_count matches exactly (strict, used post-rebuild)."""
        return self._load_sidecar_internal(
            expected_chunk_count=current_chunk_count,
            tolerate_stale=False,
        )

    def _load_sidecar_tolerant(self) -> bool:
        """Load whatever sidecar exists regardless of chunk_count ('stale-ok' warm start).

        This allows serving the previous session's BM25 index immediately at startup
        while a background rebuild catches up to the current DB state.
        Returns True and installs the backend if a valid sidecar is found.
        """
        return self._load_sidecar_internal(
            expected_chunk_count=None,  # skip chunk_count check
            tolerate_stale=True,
        )

    def _load_sidecar_internal(
        self,
        expected_chunk_count: Optional[int],
        tolerate_stale: bool,
    ) -> bool:
        if not os.path.exists(self.sidecar_path):
            return False
        try:
            with open(self.sidecar_path, "rb") as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict):
                return False
            if int(payload.get("schema_version") or 0) != _KEYWORD_INDEX_SCHEMA_VERSION:
                return False
            if str(payload.get("backend_name") or "").strip().lower() != self.backend_name:
                return False
            saved_chunk_count = int(payload.get("chunk_count") or -1)
            if not tolerate_stale and expected_chunk_count is not None:
                if saved_chunk_count != int(expected_chunk_count):
                    return False
            raw_records = payload.get("records") or []
            records = [KeywordIndexRecord(**dict(item or {})) for item in raw_records]
            backend = self._get_backend(records)
            with self._lock:
                self._backend = backend
                self._records = records
                self._records_chunk_count = saved_chunk_count
                # If tolerate_stale, do NOT update ready_generation so a rebuild
                # is still triggered when the prewarm thread calls ensure_ready_sync.
                if not tolerate_stale:
                    self._ready_generation = self._dirty_generation
                self._cond.notify_all()
            stale_tag = " (stale-ok)" if tolerate_stale else ""
            logger.info(
                f"[KeywordIndex] sidecar load{stale_tag}: backend={self.backend_name}, "
                f"records={len(records)}, chunk_count={saved_chunk_count}"
            )
            return True
        except Exception as e:
            logger.warning(f"[KeywordIndex] sidecar load failed: {e}")
            return False

    def _build_and_install(
        self,
        current_chunk_count: int,
        *,
        expected_generation: Optional[int] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> bool:
        t0 = time.time()
        if cancel_check and cancel_check():
            return False
        records = self._build_records_fn()
        if cancel_check and cancel_check():
            return False
        backend = self._get_backend(records)
        with self._lock:
            if expected_generation is not None and expected_generation != self._dirty_generation:
                return False
            if cancel_check and cancel_check():
                return False
            self._backend = backend
            self._records = records
            self._records_chunk_count = int(current_chunk_count)
            self._ready_generation = self._dirty_generation
            self._cond.notify_all()
        self._persist_sidecar(records, current_chunk_count)
        elapsed = time.time() - t0
        logger.info(
            f"[KeywordIndex] rebuild ready: backend={self.backend_name}, "
            f"records={len(records)}, chunk_count={current_chunk_count}, took={elapsed:.3f}s"
        )
        return True

    def _ensure_ready_sync(
        self,
        force_rebuild: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        current_chunk_count = int(self._current_chunk_count_fn() or 0)
        build_generation = -1
        while True:
            if cancel_check and cancel_check():
                return
            with self._lock:
                ready = (
                    self._backend is not None
                    and self._records_chunk_count == current_chunk_count
                    and self._ready_generation == self._dirty_generation
                )
                if ready and not force_rebuild:
                    return
                if self._building_generation == self._dirty_generation:
                    self._cond.wait(timeout=0.1)
                    continue
                self._building_generation = self._dirty_generation
                build_generation = self._building_generation
                break
        try:
            if not force_rebuild and not (cancel_check and cancel_check()) and self._load_sidecar(current_chunk_count):
                return
            self._build_and_install(
                current_chunk_count,
                expected_generation=build_generation,
                cancel_check=cancel_check,
            )
        finally:
            with self._lock:
                if self._building_generation == build_generation:
                    self._building_generation = -1
                self._cond.notify_all()

    def warm_start(
        self,
        background: bool = True,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Warm up the keyword index at startup.

        Strategy (stale-ok fast path):
        1. Try to load the previous session's sidecar ignoring chunk_count
           so the index is available immediately (sub-millisecond).
        2. If the loaded sidecar is stale OR no sidecar exists, schedule a
           rebuild so the index catches up to the current DB state.
        """
        current_chunk_count = int(self._current_chunk_count_fn() or 0)
        if current_chunk_count <= 0:
            return
        if cancel_check and cancel_check():
            return

        # --- fast path: serve previous session's index immediately ---
        stale_loaded = False
        with self._lock:
            already_ready = (
                self._backend is not None
                and self._records_chunk_count == current_chunk_count
                and self._ready_generation == self._dirty_generation
            )
        if not already_ready:
            stale_loaded = self._load_sidecar_tolerant()

        # Check if the sidecar was an exact match (ready_generation updated)
        with self._lock:
            exact_match = (
                self._backend is not None
                and self._records_chunk_count == current_chunk_count
                and self._ready_generation == self._dirty_generation
            )

        if exact_match:
            # Perfect — sidecar is current, no rebuild needed.
            return

        # Sidecar is stale or missing → rebuild.
        if background:
            self.schedule_rebuild()
            return
        # Synchronous path (called from prewarm thread): full build.
        self._ensure_ready_sync(force_rebuild=False, cancel_check=cancel_check)

    def invalidate(self, reason: str = "") -> None:
        with self._lock:
            self._dirty_generation += 1
            self._backend = None
            self._records = []
            self._records_chunk_count = -1
            self._cond.notify_all()
        logger.info(f"[KeywordIndex] invalidated: generation={self._dirty_generation} reason={reason or 'n/a'}")

    def schedule_rebuild(self) -> None:
        with self._lock:
            worker = self._rebuild_thread
            if worker is not None and worker.is_alive():
                return
            worker = threading.Thread(
                target=self._background_rebuild_loop,
                name="fileagent-keyword-index-rebuild",
                daemon=True,
            )
            self._rebuild_thread = worker
            worker.start()

    def _background_rebuild_loop(self) -> None:
        while not self._shutdown.is_set():
            with self._lock:
                target_generation = self._dirty_generation
            if self._rebuild_delay_sec > 0:
                time.sleep(self._rebuild_delay_sec)
            with self._lock:
                if target_generation != self._dirty_generation:
                    continue
            try:
                current_chunk_count = int(self._current_chunk_count_fn() or 0)
                if current_chunk_count > 0:
                    self._ensure_ready_sync(force_rebuild=False)
            except Exception as e:
                logger.warning(f"[KeywordIndex] background rebuild failed: {e}")
            with self._lock:
                if target_generation == self._dirty_generation:
                    return

    def close(self) -> None:
        self._shutdown.set()

    def is_ready(self) -> bool:
        with self._lock:
            return self._backend is not None

    def records_snapshot(self, *, require_current: bool = True) -> List[KeywordIndexRecord]:
        """Return indexed file records without rebuilding from Chroma metadata.

        If the in-memory index is empty, this may load an exact-match sidecar using
        only collection.count(); it never calls the build_records_fn.
        """
        current_chunk_count = int(self._current_chunk_count_fn() or 0)
        with self._lock:
            ready = self._backend is not None and bool(self._records)
            if ready and (
                not require_current
                or (
                    self._records_chunk_count == current_chunk_count
                    and self._ready_generation == self._dirty_generation
                )
            ):
                return list(self._records)

        if require_current:
            self._load_sidecar(current_chunk_count)
        else:
            self._load_sidecar_tolerant()

        with self._lock:
            ready = self._backend is not None and bool(self._records)
            if not ready:
                return []
            if require_current and not (
                self._records_chunk_count == current_chunk_count
                and self._ready_generation == self._dirty_generation
            ):
                return []
            return list(self._records)

    def list_records(
        self,
        *,
        allowed_paths: Optional[List[str]] = None,
        category_filter: str = "",
        file_extensions: Optional[List[str]] = None,
        require_current: bool = True,
    ) -> List[KeywordIndexRecord]:
        """List indexed file records via sidecar/in-memory data only."""
        records = self.records_snapshot(require_current=require_current)
        if not records:
            return []
        candidate_indices = self._collect_candidate_indices(
            records,
            allowed_paths=allowed_paths,
            category_filter=category_filter,
            file_extensions=file_extensions,
        )
        if candidate_indices is None:
            return list(records)
        return [records[idx] for idx in candidate_indices]

    def _collect_candidate_indices(
        self,
        records: List[KeywordIndexRecord],
        *,
        allowed_paths: Optional[List[str]] = None,
        category_filter: str = "",
        file_extensions: Optional[List[str]] = None,
    ) -> Optional[List[int]]:
        need_filter = (
            allowed_paths is not None
            or bool(category_filter)
            or bool(file_extensions)
        )
        if not need_filter:
            return None

        target_exts = {str(ext).lower() for ext in (file_extensions or []) if str(ext).strip()}
        target_category = str(category_filter or "").strip().lower()
        compatible_categories = {target_category} if target_category else set()
        if target_category:
            try:
                from core.retrieval.category_engine import get_compatible_categories

                compatible_categories = {
                    str(cat or "").strip().lower()
                    for cat in (get_compatible_categories(target_category) or {target_category})
                    if str(cat or "").strip()
                } or {target_category}
            except Exception:
                compatible_categories = {target_category}
        scope_matcher = ensure_path_scope_matcher(allowed_paths)
        indices: List[int] = []
        for idx, rec in enumerate(records):
            if not scope_matcher.allows_file(rec.normalized_path or rec.file_path):
                continue
            if target_exts and str(rec.file_extension or "").lower() not in target_exts:
                continue
            rec_category = str(rec.category or "").strip().lower()
            if target_category and rec_category not in compatible_categories:
                continue
            indices.append(idx)
        return indices

    def score(
        self,
        query_tokens: Sequence[str],
        *,
        allowed_paths: Optional[List[str]] = None,
        category_filter: str = "",
        file_extensions: Optional[List[str]] = None,
    ) -> List[Tuple[str, str, float]]:
        """Score files against query tokens.

        Non-blocking: if the index is not yet ready, schedules a background
        rebuild and returns [] (graceful degraded mode — vector search still
        runs). This prevents the 9-second startup stall that occurred when
        the first query arrived before the index was warm.
        """
        if not query_tokens:
            return []

        with self._lock:
            backend = self._backend
            records = list(self._records)

        if backend is None:
            # Index not ready — return [] and degrade to vector-only search.
            # Do NOT call schedule_rebuild() here: a background rebuild thread
            # building the full 1776-doc BM25 corpus would hold large amounts of
            # memory concurrently with GGML Metal GPU embedding during video
            # indexing, causing Metal abort() on the final keyframe.
            # Rebuilds are triggered exclusively by request_query_cache_prewarm()
            # which fires after every indexing job completes.
            logger.debug("[KeywordIndex] score: index not ready, returning [] (prewarm will rebuild)")
            return []

        candidate_indices = self._collect_candidate_indices(
            records,
            allowed_paths=allowed_paths,
            category_filter=category_filter,
            file_extensions=file_extensions,
        )
        scored = backend.score(query_tokens, candidate_indices=candidate_indices)
        return [
            (records[idx].chroma_id, records[idx].file_path, score)
            for idx, score in scored
        ]
