
from __future__ import annotations

import os
import sys
import time
import uuid
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, Generator, Iterable, List, Optional, Set

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

_DATA_DIR = (os.getenv("FILEAGENT_DATA_DIR") or "").strip()
if _DATA_DIR:
    try:
        _DATA_DIR = os.path.abspath(os.path.expanduser(_DATA_DIR))
        os.makedirs(_DATA_DIR, exist_ok=True)
        os.environ.setdefault("DB_PATH", os.path.join(_DATA_DIR, "chroma_db"))
        os.environ.setdefault("FILEAGENT_LOCAL_MODELS_DIR", os.path.join(_DATA_DIR, "local_models"))
        os.environ.setdefault("FILEAGENT_PREFERENCES_PATH", os.path.join(_DATA_DIR, "user_preferences.json"))
    except Exception:
        pass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.langgraph_agent import FileAgent

from services.model_manager import ModelManager, DEFAULT_UI_MODEL_IDS
from services.local_llm import LocalLLMManager, get_local_llm_manager
from services.preference_manager import PreferenceManager
from services.download_utils import ensure_model_downloaded, ensure_gguf_downloaded
from tools.document_tools import set_active_paths, set_active_session_id, cache_opened_file
from config import settings
from utils.logger import get_logger

logger = get_logger()



from utils.file_explorer import (
    _stable_id, _icon_type_for_path, _is_indexable_file_for_sources,
    _build_file_node, _source_path_key, _is_folder_fully_indexed,
    _collect_indexable_file_paths, _folder_has_relevant_indexable_file,
    _build_folder_node
)
from services.storage.source_store import SourceStore
from services.storage.history_manager import HistoryManager
from services.indexing.index_job import IndexJobState



_GGUF_VALID_CACHE: Dict[str, Dict[str, Any]] = {}
_GGUF_VALID_CACHE_TTL_SEC = 15.0


def _gguf_file_valid(path: str) -> bool:
    """Check if a GGUF file is complete by checking magic number and size."""
    try:
        import os
        if not os.path.isfile(path):
            return False
        st = os.stat(path)
        now = time.time()
        cached = _GGUF_VALID_CACHE.get(path)
        if cached:
            same_file = (
                int(cached.get("size", -1)) == int(st.st_size)
                and float(cached.get("mtime", -1.0)) == float(st.st_mtime)
            )
            if same_file:
                was_ok = bool(cached.get("ok"))
                checked_at = float(cached.get("ts", 0.0))
                if was_ok or (now - checked_at) < _GGUF_VALID_CACHE_TTL_SEC:
                    return was_ok

        from services.download_utils import (
            _is_gguf_complete,
            _query_gguf_file_size,
            _read_gguf_size_hint,
            _write_gguf_size_hint,
            _is_gguf_loadable,
        )
        expected = 0
        try:
            filename = os.path.basename(path)
            # Find which model config matches this filename
            repo_id = ""
            if filename == settings.RERANKER_GGUF_FILE:
                repo_id = settings.RERANKER_MODEL
            elif filename == os.path.basename(settings.LOCAL_EMBEDDING_MODEL_PATH):
                repo_id = getattr(settings, "EMBEDDING_REPO_ID", settings.EMBEDDING_MODEL)

            expected = _read_gguf_size_hint(path)
            
            # Fast path: if file size perfectly matches the hint, bypass remote API calls
            if expected > 0 and expected == st.st_size:
                if _is_gguf_complete(path, expected):
                    _GGUF_VALID_CACHE[path] = {"size": int(st.st_size), "mtime": float(st.st_mtime), "ok": True, "ts": now}
                    return True
                    
            if repo_id:
                expected = max(expected, _query_gguf_file_size(repo_id, filename))
        except Exception as e:
            from utils.logger import get_logger
            get_logger().error(f"Error querying size for {path}: {e}")
            pass

        if expected > 0:
            ok = _is_gguf_complete(path, expected)
            if not ok:
                try:
                    ok = _is_gguf_loadable(path)
                except Exception:
                    ok = False
            if ok:
                try:
                    _write_gguf_size_hint(path, os.path.getsize(path))
                except Exception:
                    pass
            _GGUF_VALID_CACHE[path] = {"size": int(st.st_size), "mtime": float(st.st_mtime), "ok": bool(ok), "ts": now}
            return ok

        ok2 = _is_gguf_loadable(path)
        if ok2:
            try:
                _write_gguf_size_hint(path, os.path.getsize(path))
            except Exception:
                pass
        _GGUF_VALID_CACHE[path] = {"size": int(st.st_size), "mtime": float(st.st_mtime), "ok": bool(ok2), "ts": now}
        return ok2
    except Exception:
        return False


def _model_dir_installed(local_dir: str) -> bool:
    """Check if a model exists locally. Supports both HF directories and single GGUF files."""
    try:
        if local_dir.endswith(".gguf"):
            return _gguf_file_valid(local_dir)
        has_config = os.path.exists(os.path.join(local_dir, "config.json"))
        has_safetensors = os.path.exists(os.path.join(local_dir, "model.safetensors"))
        has_bin = os.path.exists(os.path.join(local_dir, "pytorch_model.bin"))
        return bool(has_config and (has_safetensors or has_bin))
    except Exception:
        return False


class Backend:

    def __init__(self, data_dir: str = ""):
        if data_dir:
            os.environ["FILEAGENT_DATA_DIR"] = data_dir

        self._state_dir = _DATA_DIR or os.path.join(BASE_DIR, "data")
        try:
            os.makedirs(self._state_dir, exist_ok=True)
        except Exception:
            pass

        self.model_manager = ModelManager(BASE_DIR)
        self.llm_manager: LocalLLMManager = get_local_llm_manager(BASE_DIR)
        self.pref_manager = PreferenceManager(BASE_DIR)
        self.sources = SourceStore(self._state_dir)
        self.history = HistoryManager(self._state_dir)

        self._agent: Optional["FileAgent"] = None
        self._agent_lock = threading.Lock()
        self._jobs: Dict[str, IndexJobState] = {}
        self._jobs_lock = threading.Lock()
        self._indexing_lock = threading.Lock()
        self._embedding_runtime_lock = threading.Lock()
        self._model_switch_lock = threading.Lock()
        self._active_index_job_id: Optional[str] = None
        self._index_cancel_event = threading.Event()
        self._index_append_queue: list = []
        self._model_before_indexing: Optional[str] = None
        self._reranker_suspended_for_indexing = False
        self._list_sources_cache_log = {"reason": "", "count": -1, "at": 0.0}
        self._indexed_paths_cache_lock = threading.Lock()
        self._indexed_paths_cache_file = os.path.join(self._state_dir, ".indexed_paths_cache.json")
        self.ACTIVE_INDEX_STATE_PATH = os.path.join(self._state_dir, "active_index_job.json")
        self._active_index_state_last_write_ts = 0.0

        self._file_id_map: Dict[str, str] = {}   # stable_id → absolute_path
        self._file_id_map_lock = threading.Lock()
        self._file_id_map_dirty = True

        self._core_models_lock = threading.Lock()
        self._core_models_state: Dict[str, Any] = {
            "embedding": {"status": "idle", "error": None},
            "reranker": {"status": "idle", "error": None},
            "is_downloading": False,
            "progress": 0,
            "run_id": 0,
        }
        self._asr_model_lock = threading.Lock()
        self._asr_model_state: Dict[str, Any] = {
            "asr": {"status": "idle", "error": None},
            "is_downloading": False,
            "progress": 0,
            "run_id": 0,
        }

        self._ui_ready = threading.Event()
        self._startup_sync_done = threading.Event()

        self._warmup_thread = threading.Thread(target=self._delayed_warmup, daemon=True)
        self._warmup_thread.start()
        
        threading.Thread(target=self._delayed_resume_index_job, daemon=True).start()

    def notify_ui_ready(self) -> Dict[str, Any]:
        """Mark the UI as ready so the backend can start heavier background work."""
        self._ui_ready.set()
        return {"ok": True}

    def _delayed_warmup(self):
        self._ui_ready.wait(timeout=30)
        import time
        time.sleep(1.0)  # Give the freshly ready UI a short buffer before taking the GIL.
        pending_resume = self._has_pending_resume_index_job()
        logger.info(f"[Backend] delayed_warmup start: pending_resume={pending_resume}")
        try:
            self._sync_sources_with_db(allow_startup_prewarm=(not pending_resume))
        finally:
            self._startup_sync_done.set()

        if pending_resume:
            logger.info("[Backend] Pending resume index job detected; skipping startup warmup and restoring the index first.")
            try:
                self._update_startup_index_prefill_status(
                    state="skipped",
                    reason="pending_resume_index_job",
                )
            except Exception:
                pass
            return

        self._warmup_models()

    def _delayed_resume_index_job(self):
        if os.environ.get("FILEAGENT_SKIP_RESUME_INDEX_JOB", "0") == "1":
            logger.info("[Backend] Skipping automatic index resume because FILEAGENT_SKIP_RESUME_INDEX_JOB=1")
            return
        self._ui_ready.wait(timeout=30)
        import time
        self._startup_sync_done.wait(timeout=60)
        logger.info("[Backend] delayed_resume_index_job gate opened after startup sync")
        time.sleep(0.5)  # Add a short buffer after startup sync before resuming indexing.
        self._resume_active_index_job_if_needed()

    def _sync_sources_with_db(self, *, allow_startup_prewarm: bool = True):
        """Synchronize ChromaDB state with frontend source configuration at startup."""
        try:
            logger.info("[Sync] Synchronizing backend index state with frontend sources...")
            if self._should_avoid_live_kb_reads():
                logger.info("[Sync] Active or pending resume index job detected; skipping startup live DB sync.")
                return
            agent = self._ensure_agent()
            if agent is None:
                return
            kb = getattr(agent, "kb", None)
            if not kb or not hasattr(kb, "get_indexed_file_paths"):
                return
            try:
                if allow_startup_prewarm and hasattr(kb, "request_query_cache_prewarm"):
                    kb.request_query_cache_prewarm(background=True, reason="startup_sync")
                elif not allow_startup_prewarm:
                    logger.info("[Sync] Skipping startup prewarm while a resume index job is pending.")
            except Exception as e:
                logger.warning(f"[Sync] Startup prewarm request failed: {e}")
            
            db_paths = kb.get_indexed_file_paths()
            
            try:
                self._replace_indexed_paths_cache(db_paths)
                logger.info(f"[Sync] Refreshed .indexed_paths_cache.json with {len(db_paths)} indexed files.")
            except Exception as e:
                logger.warning(f"[Sync] Failed to refresh indexed paths cache: {e}")
            
            self.sources.load()
            configured_folders = self.sources.folders or []
            configured_files = self.sources.files or []
            
            missing_paths = []
            for dp in db_paths:
                covered = False
                if dp in configured_files:
                    covered = True
                else:
                    for cf in configured_folders:
                        cfa = os.path.abspath(os.path.expanduser(cf))
                        if dp.startswith(cfa) and (len(dp) == len(cfa) or dp[len(cfa)] == os.sep):
                            covered = True
                            break
                if not covered:
                    missing_paths.append(dp)
            
            if missing_paths:
                added_count = 0
                for mp in missing_paths:
                    if os.path.exists(mp):
                        self.sources.add_file(mp)
                        added_count += 1
                logger.info(f"[Sync] Repaired {added_count} indexed file references missing from frontend config.")
                
        except Exception as e:
            logger.error(f"[Sync] Failed to synchronize index state: {e}")

    def _has_pending_resume_index_job(self) -> bool:
        """Return whether startup should resume an interrupted index job."""
        try:
            st = self._read_active_index_state()
            if self._should_discard_persisted_index_state(st):
                self._clear_active_index_state()
                return False
            if not st or not st.get("is_indexing"):
                return False
            if st.get("error") == "cancelled" or st.get("error") is not None:
                return False
            total_files = max(0, int(st.get("total_files") or 0))
            completed_files = max(0, int(st.get("completed_files") or 0))
            if total_files > 0 and completed_files >= total_files:
                return False
            kind = str(st.get("kind") or "folder")
            if kind == "files":
                raw_files = st.get("files") or []
                return bool(isinstance(raw_files, list) and any(str(fp).strip() for fp in raw_files))
            folder = str(st.get("folder") or "").strip()
            return bool(folder)
        except Exception:
            return False

    def _should_discard_persisted_index_state(self, st: Optional[Dict[str, Any]]) -> bool:
        """Detect invalid persisted index state that could cause a resume crash loop."""
        try:
            if not st or not st.get("is_indexing"):
                return False
            error = str(st.get("error") or "").strip()
            if error:
                logger.warning(
                    "[Backend] Persisted index state is terminal with error=%r; discarding stale indexing marker.",
                    error,
                )
                return True
            total_files = max(0, int(st.get("total_files") or 0))
            completed_files = max(0, int(st.get("completed_files") or 0))
            failed_files = max(0, int(st.get("failed_files") or 0))
            if total_files > 0 and (completed_files + failed_files) >= total_files:
                logger.warning(
                    "[Backend] Persisted index state is already complete "
                    "(completed=%s failed=%s total=%s); discarding stale indexing marker.",
                    completed_files,
                    failed_files,
                    total_files,
                )
                return True
            kind = str(st.get("kind") or "folder").strip().lower()
            if kind != "files":
                return False
            stage = str(st.get("stage") or "").strip().lower()
            current_frame = max(0, int(st.get("current_frame") or 0))
            total_frames = max(0, int(st.get("total_frames") or 0))
            if stage == "analyzing_frames" and total_frames > 0 and current_frame >= total_frames:
                logger.warning(
                    "[Backend] Persisted file-indexing state is stuck on the final frame; "
                    "discarding it to avoid another resume crash."
                )
                return True
        except Exception:
            return False
        return False

    def _should_avoid_live_kb_reads(self) -> bool:
        """Avoid live Chroma reads while indexing is active or pending resume."""
        try:
            with self._jobs_lock:
                for job in self._jobs.values():
                    if job.is_indexing:
                        return True
        except Exception:
            pass
        try:
            st = self._read_active_index_state()
            if self._should_discard_persisted_index_state(st):
                self._clear_active_index_state()
                return False
            return bool(st and st.get("is_indexing"))
        except Exception:
            return False

    # ─── Lifecycle ───

    def shutdown(self):
        logger.info("Shutting down...")
        try:
            self._index_cancel_event.set()
            with self._jobs_lock:
                for j in self._jobs.values():
                    if j and j.is_indexing and not j.error:
                        j.error = "cancelled"
        except Exception:
            pass

        try:
            deadline = time.time() + 3.0
            while time.time() < deadline:
                with self._jobs_lock:
                    has_running_job = any(j and j.is_indexing for j in self._jobs.values())
                if not has_running_job:
                    break
                time.sleep(0.05)
        except Exception:
            pass

        try:
            if self._agent is not None and hasattr(self._agent, "close"):
                self._agent.close()
        except Exception:
            pass
        try:
            self.llm_manager.stop_server()
        except Exception:
            pass

        try:
            import gc
            gc.collect()
            gc.collect()
            time.sleep(0.3)
        except Exception:
            pass

    # ─── Health ───

    def health_check(self) -> Dict[str, Any]:
        return {"ok": True, "ts": time.time(), "version": "1.0"}

    def get_runtime_paths(self) -> Dict[str, Any]:
        try:
            local_models_dir = os.getenv("FILEAGENT_LOCAL_MODELS_DIR") or ""
        except Exception:
            local_models_dir = ""
        return {
            "base_dir": BASE_DIR,
            "data_dir": _DATA_DIR or "",
            "state_dir": self._state_dir,
            "indexed_folders_path": getattr(self.sources, "new_path", "") or getattr(self.sources, "legacy_path", ""),
            "db_path": getattr(settings, "DB_PATH", ""),
            "local_models_dir": local_models_dir,
            "embedding_dir": getattr(settings, "LOCAL_EMBEDDING_MODEL_PATH", ""),
            "reranker_dir": getattr(settings, "LOCAL_RERANKER_MODEL_PATH", ""),
        }

    # ─── Indexed path cache ───

    def _get_indexed_paths_cache_lock(self) -> threading.Lock:
        lock = getattr(self, "_indexed_paths_cache_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._indexed_paths_cache_lock = lock
        return lock

    def _get_indexed_paths_cache_file(self) -> str:
        path = getattr(self, "_indexed_paths_cache_file", "")
        if path:
            return path
        state_dir = getattr(self, "_state_dir", _DATA_DIR or os.path.join(BASE_DIR, "data"))
        path = os.path.join(state_dir, ".indexed_paths_cache.json")
        self._indexed_paths_cache_file = path
        return path

    def _clean_indexed_cache_paths(self, paths: Iterable[Any]) -> Dict[str, str]:
        cleaned: Dict[str, str] = {}
        for p in paths or []:
            raw = str(p or "").strip()
            if not raw:
                continue
            try:
                abs_path = os.path.abspath(os.path.expanduser(raw))
            except Exception:
                abs_path = raw
            cleaned[_source_path_key(abs_path)] = abs_path
        return cleaned

    def _read_indexed_paths_cache_unlocked(self) -> Set[str]:
        try:
            import json as _json

            cache_path = self._get_indexed_paths_cache_file()
            if not os.path.exists(cache_path):
                return set()
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = _json.load(f)
            if isinstance(raw, list):
                return set(self._clean_indexed_cache_paths(raw).values())
        except Exception:
            pass
        return set()

    def _read_indexed_paths_cache(self) -> Set[str]:
        with self._get_indexed_paths_cache_lock():
            return self._read_indexed_paths_cache_unlocked()

    def _write_indexed_paths_cache_unlocked(self, paths: Iterable[Any]) -> Set[str]:
        cleaned = self._clean_indexed_cache_paths(paths)
        cache_path = self._get_indexed_paths_cache_file()
        try:
            import json as _json

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_path = f"{cache_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                _json.dump(sorted(cleaned.values()), f, ensure_ascii=False)
            os.replace(tmp_path, cache_path)
        except Exception:
            pass
        return set(cleaned.values())

    def _replace_indexed_paths_cache(self, paths: Iterable[Any]) -> Set[str]:
        with self._get_indexed_paths_cache_lock():
            return self._write_indexed_paths_cache_unlocked(paths)

    def _merge_indexed_paths_cache(self, paths: Iterable[Any]) -> Set[str]:
        with self._get_indexed_paths_cache_lock():
            existing = self._clean_indexed_cache_paths(self._read_indexed_paths_cache_unlocked())
            changed = False
            for key, path in self._clean_indexed_cache_paths(paths).items():
                if existing.get(key) != path:
                    existing[key] = path
                    changed = True
            if not changed:
                return set(existing.values())
            return self._write_indexed_paths_cache_unlocked(existing.values())

    def _remove_indexed_paths_cache(self, paths: Iterable[Any]) -> Set[str]:
        remove_keys = set(self._clean_indexed_cache_paths(paths).keys())
        if not remove_keys:
            return self._read_indexed_paths_cache()
        with self._get_indexed_paths_cache_lock():
            existing = self._clean_indexed_cache_paths(self._read_indexed_paths_cache_unlocked())
            for key in remove_keys:
                existing.pop(key, None)
            return self._write_indexed_paths_cache_unlocked(existing.values())

    def _remove_indexed_paths_cache_under(self, roots: Iterable[Any]) -> Set[str]:
        root_keys = set(self._clean_indexed_cache_paths(roots).keys())
        if not root_keys:
            return self._read_indexed_paths_cache()
        with self._get_indexed_paths_cache_lock():
            existing = self._clean_indexed_cache_paths(self._read_indexed_paths_cache_unlocked())
            kept: Dict[str, str] = {}
            for key, path in existing.items():
                under_removed_root = any(key == rk or key.startswith(rk.rstrip(os.sep) + os.sep) for rk in root_keys)
                if not under_removed_root:
                    kept[key] = path
            return self._write_indexed_paths_cache_unlocked(kept.values())

    def _refresh_indexed_paths_cache_from_kb(self, kb: Any) -> None:
        try:
            if kb is not None and hasattr(kb, "get_indexed_file_paths"):
                self._replace_indexed_paths_cache(kb.get_indexed_file_paths())
        except Exception as e:
            logger.warning(f"[Backend] 刷新已索引路径缓存失败: {e}")

    # ─── Sources ───

    def list_sources(self) -> Dict[str, Any]:
        self.sources.load()
        indexing_paths = set()
        indexing_file_paths = set()
        active_indexed_paths: Set[str] = set()
        avoid_live_db_query = False
        with self._jobs_lock:
            for j in self._jobs.values():
                if j.is_indexing and j.folder:
                    indexing_paths.add(j.folder)
                if j.is_indexing:
                    active_indexed_paths.update(
                        str(p) for p in (getattr(j, "indexed_paths", []) or []) if str(p or "").strip()
                    )
                    current_path = os.path.abspath(os.path.expanduser(str(getattr(j, "current_path", "") or "")))
                    if current_path and os.path.isfile(current_path):
                        indexing_file_paths.add(current_path)
        try:
            st = self._read_active_index_state()
            if self._should_discard_persisted_index_state(st):
                self._clear_active_index_state()
                st = None
            if st and st.get("is_indexing") and st.get("folder"):
                indexing_paths.add(os.path.abspath(os.path.expanduser(str(st.get("folder")))))
            if st and st.get("is_indexing") and st.get("kind") == "files":
                raw_files = st.get("files") or []
                if isinstance(raw_files, list):
                    for fp in raw_files:
                        fp_abs = os.path.abspath(os.path.expanduser(str(fp)))
                        if fp_abs and os.path.isfile(fp_abs):
                            indexing_file_paths.add(fp_abs)
            raw_indexed = st.get("indexed_paths") if st else None
            if isinstance(raw_indexed, list):
                active_indexed_paths.update(str(p) for p in raw_indexed if str(p or "").strip())
            avoid_live_db_query = bool(st and st.get("is_indexing"))
        except Exception:
            pass
        avoid_live_db_query = avoid_live_db_query or bool(indexing_paths) or bool(indexing_file_paths)

        indexed_paths = None
        try:
            if self._agent is not None and not avoid_live_db_query:
                from core.langgraph_agent import get_kb_instance
                kb = get_kb_instance()
                indexed_paths = kb.get_indexed_file_paths()
        except Exception:
            pass

        if indexed_paths is not None:
            try:
                self._replace_indexed_paths_cache(indexed_paths)
            except Exception:
                pass
        else:
            try:
                cached_paths = self._read_indexed_paths_cache()
                if cached_paths or os.path.exists(self._get_indexed_paths_cache_file()):
                    indexed_paths = set(cached_paths)
                    reason = "indexing" if avoid_live_db_query else "db_unavailable"
                    now = time.time()
                    log_state = self._list_sources_cache_log
                    should_log = (
                        reason != str(log_state.get("reason") or "")
                        or len(indexed_paths) != int(log_state.get("count") or -1)
                        or (now - float(log_state.get("at") or 0.0)) >= 10.0
                    )
                    if should_log:
                        if reason == "indexing":
                            logger.info(f"[Backend] list_sources: 索引中跳过 live DB 查询，使用缓存 ({len(indexed_paths)} paths)")
                        else:
                            logger.info(f"[Backend] list_sources: DB 未就绪，使用缓存 ({len(indexed_paths)} paths)")
                        self._list_sources_cache_log = {"reason": reason, "count": len(indexed_paths), "at": now}
            except Exception:
                pass

        if active_indexed_paths:
            if indexed_paths is None:
                indexed_paths = set()
            indexed_paths.update(active_indexed_paths)
            if avoid_live_db_query:
                try:
                    self._merge_indexed_paths_cache(active_indexed_paths)
                except Exception:
                    pass
        elif indexed_paths is None and avoid_live_db_query:
            # During indexing, unknown is safer as pending than as indexed.
            indexed_paths = set()

        indexed_path_keys: Optional[Set[str]] = None
        if indexed_paths is not None:
            indexed_path_keys = {_source_path_key(x) for x in indexed_paths}

        skipped_keys: Set[str] = set()
        with self._jobs_lock:
            for j in self._jobs.values():
                if j.is_indexing and j.skipped_files:
                    skipped_keys.update(j.skipped_files)

        sources = []
        indexing_abs = {os.path.abspath(os.path.expanduser(str(x))) for x in indexing_paths}
        displayed_file_keys: Set[str] = set()
        for f in (self.sources.folders or []):
            fa = os.path.abspath(os.path.expanduser(f))
            is_indexing = f in indexing_paths or fa in indexing_abs
            st = "indexing" if is_indexing else "indexed" # Folder is marked indexed generally if not indexing
            
            all_idx = _collect_indexable_file_paths(fa)
            if skipped_keys:
                all_idx = all_idx - skipped_keys
            if not all_idx:
                continue
                
            if not is_indexing:
                if indexed_path_keys is not None:
                    relevant = all_idx.intersection(indexed_path_keys)
                    if not relevant:
                        continue
                    prune = True
                else:
                    continue
            else:
                relevant = set(all_idx)
                prune = True
                
            sources.append(
                _build_folder_node(
                    fa,
                    status=st,
                    indexed_path_keys=indexed_path_keys,
                    depth=0,
                    relevant_indexable_paths=relevant,
                    prune_empty_subfolders=prune,
                )
            )
        for fp in (self.sources.files or []):
            if not _is_indexable_file_for_sources(fp, treat_parent_as_explicit_source=True):
                continue
            fpk = _source_path_key(fp)
            if fpk in skipped_keys:
                continue
            fp_abs = os.path.abspath(os.path.expanduser(fp))
            if fp_abs in indexing_file_paths:
                status = "indexing"
            elif fp in indexing_paths or any(fp.startswith(p) for p in indexing_paths):
                status = "indexing"
            elif indexed_path_keys is not None and fpk in indexed_path_keys:
                status = "indexed"
            else:
                continue
            displayed_file_keys.add(fpk)
            sources.append(_build_file_node(fp, status=status))

        for fp_abs in sorted(indexing_file_paths):
            if not _is_indexable_file_for_sources(fp_abs, treat_parent_as_explicit_source=True):
                continue
            fpk = _source_path_key(fp_abs)
            if fpk in skipped_keys or fpk in displayed_file_keys:
                continue
            sources.append(_build_file_node(fp_abs, status="indexing"))
        return {"sources": sources}

    def add_source(self, folder: str) -> Dict[str, Any]:
        folder = os.path.abspath(os.path.expanduser(folder))
        self.sources.add_folder(folder)
        node = _build_folder_node(folder, status="indexing")
        return {"folder": folder, "node": node}

    def remove_source(self, folder: str) -> Dict[str, Any]:
        folder = os.path.abspath(os.path.expanduser(folder))
        self.sources.remove(folder)
        logger.info(f"已从 indexed_sources.json 移除: {folder}")
        try:
            agent = self._ensure_agent()
            from core.langgraph_agent import get_kb_instance
            kb = get_kb_instance()
            result = kb.delete_by_folder(folder)
            if result.get("ok"):
                deleted_count = result.get("deleted_count", 0)
                self._remove_indexed_paths_cache_under([folder])
                self._invalidate_file_id_map()
                return {"ok": True, "folder": folder, "deleted_count": deleted_count,
                        "message": f"成功移除文件夹及其 {deleted_count} 条索引"}
            else:
                error = result.get("error", "未知错误")
                return {"ok": False, "folder": folder, "error": error,
                        "message": "从配置中移除成功，但删除索引失败"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"ok": False, "folder": folder, "error": str(e),
                    "message": "从配置中移除成功，但删除索引失败"}

    def remove_sources_batch(self, folders: List[str]) -> Dict[str, Any]:
        """Remove multiple sources in one database scan."""
        folders_abs = [os.path.abspath(os.path.expanduser(f)) for f in folders]
        for folder in folders_abs:
            self.sources.remove(folder)
        logger.info(f"Removed {len(folders_abs)} source configurations in batch")
        try:
            self._ensure_agent()
            from core.langgraph_agent import get_kb_instance
            kb = get_kb_instance()
            result = kb.delete_by_folders(folders_abs)
            if result.get("ok"):
                deleted_count = result.get("deleted_count", 0)
                self._remove_indexed_paths_cache_under(folders_abs)
                self._invalidate_file_id_map()
                return {"ok": True, "folders": folders_abs, "deleted_count": deleted_count}
            else:
                return {"ok": False, "folders": folders_abs, "error": result.get("error", "未知错误")}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"ok": False, "folders": folders_abs, "error": str(e)}

    def refresh_source(self, folder: str = None, target: str = None) -> Dict[str, Any]:
        target = target or folder
        if not target:
            return {"ok": False, "error": "No target/folder provided"}

        if target == "__ALL__":
            self.sources.load()
            dirs = (self.sources.folders or []) + (self.sources.files or [])
        else:
            dirs = [os.path.abspath(os.path.expanduser(target))]
            
        with self._jobs_lock:
            if (self._active_index_job_id
                    and self._active_index_job_id in self._jobs
                    and self._jobs[self._active_index_job_id].is_indexing):
                running = self._jobs[self._active_index_job_id]
                return {"job_id": running.job_id, "target": target, "already_running": True}

        job_id = str(uuid.uuid4())
        # We store target string for logging but we need list for job payload
        job = IndexJobState(job_id=job_id, folder=target)
        # Hack to attach list to job momentarily, will be read by _run_refresh_job
        job._refresh_dirs = dirs

        with self._jobs_lock:
            job.is_indexing = True
            self._jobs[job_id] = job
            self._active_index_job_id = job_id

        self._write_active_index_state(job, kind="folder")
        t = threading.Thread(target=self._run_refresh_job, args=(job,), daemon=True)
        t.start()
        return {"ok": True, "job_id": job_id, "target": target}

    # ─── History ───


    def get_history(self) -> Dict[str, Any]:
        data = self.history.load_all()
        sessions = []
        for sid, sdata in data.items():
            if isinstance(sdata, dict):
                if "id" not in sdata:
                    sdata["id"] = sid
                if "messages" not in sdata:
                    sdata["messages"] = []
                if "title" not in sdata:
                    sdata["title"] = "New Chat"
                if "lastActive" not in sdata:
                    sdata["lastActive"] = 0
                sessions.append(sdata)
        sessions.sort(key=lambda x: x.get("lastActive", 0), reverse=True)
        return {"sessions": sessions}

    def sync_history(self, session: Dict[str, Any]) -> Dict[str, Any]:
        if session and session.get("id"):
            self.history.save_session(session["id"], session)
        return {"ok": True}

    def delete_history(self, id: str) -> Dict[str, Any]:
        if id:
            self.history.delete_session(id)
        return {"ok": True}

    # ─── Models ───

    def _apply_runtime_model_hint(self, model_id: Optional[str]) -> None:
        """Apply per-request model hint from frontend to backend preference/cache."""
        mid = str(model_id or "").strip()
        if not mid:
            return
        # Protect indexing flow: do not override preference while indexing
        # or while waiting to restore pre-index model.
        if self._indexing_in_progress() is not None:
            return
        if self._model_before_indexing:
            return
        try:
            current = self.pref_manager.get_selected_model_id()
        except Exception:
            current = None
        if current == mid:
            return
        try:
            self.pref_manager.set_selected_model_id(mid)
        except Exception:
            return
        # Clear cached in-proc client / agent llm wrappers so next token generation
        # uses the hinted model immediately instead of stale preference snapshots.
        try:
            from services.inproc_openai_client import clear_inproc_openai_client
            clear_inproc_openai_client()
        except Exception:
            pass
        if self._agent is not None:
            try:
                if hasattr(self._agent, "_llm_service"):
                    delattr(self._agent, "_llm_service")
                if hasattr(self._agent, "_llm_service_detailed"):
                    delattr(self._agent, "_llm_service_detailed")
            except Exception:
                pass

    def list_models(self) -> Dict[str, Any]:
        raw_ui_ids = (os.getenv("FILEAGENT_UI_MODEL_IDS") or "").strip()
        ui_ids = [x.strip() for x in raw_ui_ids.split(",") if x.strip()] if raw_ui_ids else []
        if not ui_ids:
            ui_ids = list(DEFAULT_UI_MODEL_IDS)
        all_models = self.model_manager.get_supported_models()
        ui_set = set(ui_ids)
        models = [m for m in all_models if (m.get("id") in ui_set)]
        if not models:
            models = all_models
        selected_id = self.pref_manager.get_selected_model_id()
        model_ids = [str(m.get("id")) for m in models if m.get("id")]
        installed_ids = [str(m.get("id")) for m in models if m.get("id") and (m.get("status") == "installed" or bool(m.get("installed")))]
        selected_installed = any(str(m.get("id")) == str(selected_id) and (m.get("status") == "installed" or bool(m.get("installed"))) for m in models)
        need_fallback = bool(model_ids) and (
            (not selected_id) or
            (selected_id not in model_ids) or
            (installed_ids and not selected_installed)
        )
        if need_fallback:
            # Prefer installed models first to avoid selecting an unavailable default
            # like qwen3-4b-gguf when only VL/other models are installed.
            preferred_default = "qwen3-4b-gguf"
            if installed_ids:
                fallback_id = preferred_default if preferred_default in installed_ids else installed_ids[0]
            else:
                fallback_id = preferred_default if preferred_default in model_ids else model_ids[0]
            try:
                self.pref_manager.set_selected_model_id(fallback_id)
                selected_id = fallback_id
            except Exception:
                selected_id = fallback_id
        qmap = self.pref_manager.get_model_quantization_map() if hasattr(self.pref_manager, "get_model_quantization_map") else {}
        models_dir = str(getattr(self.llm_manager, "models_dir", "") or "")
        for m in models:
            m["selected"] = (m["id"] == selected_id)
            mid = m.get("id")
            if not mid:
                continue

            # Expose local storage folder for UI actions like "Open Location".
            model_dir = os.path.join(models_dir, str(mid)) if models_dir else ""
            if model_dir:
                m["model_dir"] = model_dir

            selected_qf = None
            if isinstance(qmap, dict) and mid in qmap:
                selected_qf = qmap.get(mid)
                m["selected_quantization"] = selected_qf

            # Resolve actual gguf path (and mmproj path for VL) so UI can reveal file location.
            try:
                resolved = self.llm_manager.resolve_target_model(
                    str(mid),
                    preferred_quantization_file=selected_qf,
                )
                if resolved:
                    _cfg, model_path, mmproj_path = resolved
                    if model_path:
                        m["selected_model_path"] = model_path
                    if mmproj_path:
                        m["selected_mmproj_path"] = mmproj_path
            except Exception:
                pass
        return {"models": models}

    def select_model(self, model_id: str) -> Dict[str, Any]:
        with self._model_switch_lock:
            j = self._indexing_in_progress()
            if j:
                return {
                    "ok": False, "error": "indexing_in_progress",
                    "job_id": j.job_id, "folder": j.folder,
                    "message": "索引进行中：禁止切换模型",
                    "current_model_id": getattr(self.llm_manager, "current_model_id", None),
                }
            prev_id = self.pref_manager.get_selected_model_id()
            if prev_id == model_id and getattr(self.llm_manager, "current_model_id", None) == model_id:
                return {"ok": True, "model_id": model_id, "already_selected": True}
            self.pref_manager.set_selected_model_id(model_id)
            
            try:
                from services.inproc_openai_client import clear_inproc_openai_client
                clear_inproc_openai_client()
            except Exception:
                pass
            if self._agent is not None:
                try:
                    if hasattr(self._agent, "_llm_service"):
                        delattr(self._agent, "_llm_service")
                    if hasattr(self._agent, "_llm_service_detailed"):
                        delattr(self._agent, "_llm_service_detailed")
                except Exception:
                    pass
                    
            try:
                qf = None
                try:
                    qf = self.pref_manager.get_selected_quantization_file(model_id)
                except Exception:
                    pass
                resolved = self.llm_manager.resolve_target_model(model_id, preferred_quantization_file=qf)
                if not resolved:
                    model_dir = os.path.join(
                        getattr(self.llm_manager, "models_dir", ""),
                        model_id,
                    )
                    return {
                        "ok": False,
                        "model_id": model_id,
                        "error": f"模型文件未找到，请确认已在「管理模型」中完成该模型下载。查找路径: {model_dir}",
                    }
                else:
                    _cfg, model_path, _mmproj = resolved
                    logger.info(f"[Backend] Switching model: {model_id}, path={model_path}")
                
                if getattr(self.llm_manager, "current_model_id", None) == model_id:
                    return {"ok": True, "model_id": model_id, "already_running": True}
                
                self.llm_manager.stop_server()
                time.sleep(0.5)
                started = self.llm_manager.start_server(
                    preferred_model_id=model_id,
                    preferred_quantization_file=qf,
                )
                current_after = getattr(self.llm_manager, "current_model_id", None)
                if (not started) or (current_after != model_id):
                    try:
                        if prev_id:
                            self.pref_manager.set_selected_model_id(prev_id)
                    except Exception:
                        pass
                    return {
                        "ok": False,
                        "model_id": model_id,
                        "current_model_id": current_after,
                        "error": f"模型切换失败：目标={model_id}，当前={current_after or '<none>'}",
                    }
                logger.info(
                    f"[Backend] Model switch success: target={model_id}, current={current_after}, "
                    f"quantization={qf or '<default>'}"
                )
            except Exception as e:
                logger.error(f"Failed to switch model: {e}")
                try:
                    if prev_id:
                        self.pref_manager.set_selected_model_id(prev_id)
                except Exception:
                    pass
                return {"ok": False, "model_id": model_id, "error": str(e)}
            return {"ok": True, "model_id": model_id}

    def select_quantization(self, model_id: str, quantization_file: str) -> Dict[str, Any]:
        self.pref_manager.set_selected_quantization_file(model_id, quantization_file)
        selected_id = self.pref_manager.get_selected_model_id()
        if selected_id == model_id:
            try:
                resolved = self.llm_manager.resolve_target_model(model_id, preferred_quantization_file=quantization_file)
                if not resolved:
                    return {
                        "ok": False,
                        "model_id": model_id,
                        "quantization_file": quantization_file,
                        "error": "量化文件未找到，请先完成该量化下载。",
                    }
                self.llm_manager.stop_server()
                time.sleep(0.5)
                started = self.llm_manager.start_server(
                    preferred_model_id=model_id,
                    preferred_quantization_file=quantization_file,
                )
                current_after = getattr(self.llm_manager, "current_model_id", None)
                if (not started) or (current_after != model_id):
                    return {
                        "ok": False,
                        "model_id": model_id,
                        "quantization_file": quantization_file,
                        "current_model_id": current_after,
                        "error": f"量化切换失败：目标={model_id}，当前={current_after or '<none>'}",
                    }
                logger.info(
                    f"[Backend] Quantization switch success: model={model_id}, "
                    f"quantization={quantization_file}, current={current_after}"
                )
                try:
                    from services.inproc_openai_client import clear_inproc_openai_client
                    clear_inproc_openai_client()
                    if self._agent is not None:
                        if hasattr(self._agent, "_llm_service"):
                            delattr(self._agent, "_llm_service")
                        if hasattr(self._agent, "_llm_service_detailed"):
                            delattr(self._agent, "_llm_service_detailed")
                except Exception:
                    pass
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return {"ok": True, "model_id": model_id, "quantization_file": quantization_file}

    def download_model(self, model_id: str, source: str = "auto",
                       quantization_file: Optional[str] = None) -> Dict[str, Any]:
        return self.model_manager.download_model(model_id, source, quantization_file=quantization_file)

    def cancel_download(self, model_id: str) -> Dict[str, Any]:
        return self.model_manager.cancel_download(model_id)

    def delete_model(self, model_id: str, quantization_file: Optional[str] = None) -> Dict[str, Any]:
        return self.model_manager.delete_model(model_id, quantization_file=quantization_file)

    # ─── Core Models (Embedding / Reranker) ───

    def core_models_status(self) -> Dict[str, Any]:
        return self._core_models_payload()

    def download_core_models(self) -> Dict[str, Any]:
        payload = self._core_models_payload()
        embedding_ready = (
            payload.get("embedding", {}).get("installed")
            and payload.get("embedding", {}).get("status") == "installed"
        )
        reranker_ready = (
            payload.get("reranker", {}).get("installed")
            and payload.get("reranker", {}).get("status") == "installed"
        )
        if embedding_ready and reranker_ready:
            return {"ok": True, "already_installed": True, "status": payload}
        run_id = 0
        with self._core_models_lock:
            if self._core_models_state.get("is_downloading"):
                return {"ok": True, "already_running": True, "status": payload}
            run_id = int(self._core_models_state.get("run_id") or 0) + 1
            self._core_models_state["run_id"] = run_id
            self._core_models_state["is_downloading"] = True
            self._core_models_state["progress"] = int(payload.get("progress") or 0)
        t = threading.Thread(target=self._download_core_models_worker, args=(run_id,), daemon=True)
        t.start()
        return {"ok": True, "started": True, "status": self._core_models_payload()}

    def cancel_core_models_download(self) -> Dict[str, Any]:
        with self._core_models_lock:
            running = bool(self._core_models_state.get("is_downloading"))
            self._core_models_state["run_id"] = int(self._core_models_state.get("run_id") or 0) + 1
            self._core_models_state["is_downloading"] = False
            for key in ("embedding", "reranker"):
                item = dict(self._core_models_state.get(key) or {})
                if str(item.get("status") or "") == "downloading":
                    item["status"] = "idle"
                    item["error"] = None
                    item["speed"] = 0
                    item["eta"] = None
                self._core_models_state[key] = item
        return {"ok": True, "cancelled": running, "status": self._core_models_payload()}

    def asr_model_status(self) -> Dict[str, Any]:
        return self._asr_model_payload()

    def download_asr_model(self) -> Dict[str, Any]:
        payload = self._asr_model_payload()
        if payload.get("asr", {}).get("installed"):
            return {"ok": True, "already_installed": True, "status": payload}
        with self._asr_model_lock:
            if self._asr_model_state.get("is_downloading"):
                return {"ok": True, "already_running": True, "status": payload}
            run_id = int(self._asr_model_state.get("run_id") or 0) + 1
            self._asr_model_state["run_id"] = run_id
            self._asr_model_state["is_downloading"] = True
            self._asr_model_state["progress"] = int(payload.get("progress") or 0)
            self._asr_model_state["asr"] = {
                **dict(self._asr_model_state.get("asr") or {}),
                "status": "downloading",
                "error": None,
            }
        t = threading.Thread(target=self._download_asr_model_worker, args=(run_id,), daemon=True)
        t.start()
        return {"ok": True, "started": True, "status": self._asr_model_payload()}

    def cancel_asr_model_download(self) -> Dict[str, Any]:
        with self._asr_model_lock:
            running = bool(self._asr_model_state.get("is_downloading"))
            self._asr_model_state["run_id"] = int(self._asr_model_state.get("run_id") or 0) + 1
            self._asr_model_state["is_downloading"] = False
            item = dict(self._asr_model_state.get("asr") or {})
            if str(item.get("status") or "") == "downloading":
                item["status"] = "idle"
                item["error"] = None
                item["speed"] = 0
                item["eta"] = None
            self._asr_model_state["asr"] = item
        return {"ok": True, "cancelled": running, "status": self._asr_model_payload()}

    # ─── Indexing ───

    def get_active_job(self) -> Dict[str, Any]:
        j = self._indexing_in_progress()
        if j:
            return {"active": True, "job_id": j.job_id, "job": j.to_payload()}
            
        st = self._read_active_index_state()
        if st:
            if self._should_discard_persisted_index_state(st):
                self._clear_active_index_state()
                return {"active": False, "job_id": None}
            job_id = str(st.get("job_id") or "")
            job_payload = None
            if job_id:
                with self._jobs_lock:
                    jj = self._jobs.get(job_id)
                if jj:
                    job_payload = jj.to_payload()
            return {"active": False, "job_id": job_id or None, "job": job_payload, "persisted": st}
            
        return {"active": False, "job_id": None}

    def _resume_active_index_job_if_needed(self) -> None:
        """Best-effort resume of persisted indexing job after process restart."""
        try:
            st = self._read_active_index_state()
            if self._should_discard_persisted_index_state(st):
                self._clear_active_index_state()
                return
            if not st or not st.get("is_indexing"):
                return
            
            if st.get("error") == "cancelled" or st.get("error") is not None:
                self._clear_active_index_state()
                return

            kind = str(st.get("kind") or "folder")
            job_id = str(st.get("job_id") or "").strip() or str(uuid.uuid4())
            folder = os.path.abspath(os.path.expanduser(str(st.get("folder") or "").strip())) if st.get("folder") else ""
            total_files = max(0, int(st.get("total_files") or 0))
            completed_files = max(0, int(st.get("completed_files") or 0))
            persisted_indexed_paths = st.get("indexed_paths") if isinstance(st.get("indexed_paths"), list) else []
            if persisted_indexed_paths:
                try:
                    self._merge_indexed_paths_cache(persisted_indexed_paths)
                except Exception:
                    pass
            
            # Check if the job actually has work to do. If it's already complete or close to complete, 
            # or if the app was closed during indexing, we shouldn't get stuck in a bad state.
            # But we DO want to resume it.
            # We will assume that if total_files > 0 and completed_files >= total_files, we are done.
            if total_files > 0 and completed_files >= total_files:
                self._clear_active_index_state()
                return

            with self._jobs_lock:
                active = self._jobs.get(self._active_index_job_id) if self._active_index_job_id else None
                if active and active.is_indexing:
                    return
                existing = self._jobs.get(job_id)
                if existing and existing.is_indexing:
                    self._active_index_job_id = job_id
                    return

            if kind == "files":
                raw_files = st.get("files") or []
                if not isinstance(raw_files, list) or not raw_files:
                    return
                file_paths = [os.path.abspath(os.path.expanduser(str(fp))) for fp in raw_files if str(fp).strip()]
                if not file_paths:
                    return
                    
                initial_total_files = max(int(st.get("total_files") or 0), len(file_paths))
                initial_completed_files = max(0, int(st.get("completed_files") or 0))
                
                job = IndexJobState(
                    job_id=job_id,
                    folder="",
                    is_indexing=True,
                    total_files=initial_total_files,
                    completed_files=initial_completed_files,
                    eta_seconds=max(0, int(st.get("eta_seconds") or 0)),
                    current_file=str(st.get("current_file") or ""),
                    started_at=time.time(),
                )
                
                # Attach original state so callback can add to it instead of overwriting
                setattr(job, "initial_total_files", initial_total_files)
                setattr(job, "initial_completed_files", initial_completed_files)
                
                job.is_indexing = True
                job.error = None
                with self._jobs_lock:
                    self._jobs[job_id] = job
                    self._active_index_job_id = job_id
                t = threading.Thread(target=self._run_file_index_job, args=(job, file_paths), daemon=True)
                t.start()
                logger.info(f"[Backend] Resumed file indexing job: {job_id} ({len(file_paths)} files)")
                print(f"[Backend] Resumed file indexing job: {job_id} ({len(file_paths)} files)")
                return

            if kind == "folder":
                if not folder or not os.path.isdir(folder):
                    return
                # We save initial state so we don't start from 0 and lose progress visually
                initial_total_files = max(0, int(st.get("total_files") or 0))
                initial_completed_files = max(0, int(st.get("completed_files") or 0))
                
                job = IndexJobState(
                    job_id=job_id,
                    folder=folder,
                    is_indexing=True,
                    total_files=initial_total_files,
                    completed_files=initial_completed_files,
                    eta_seconds=max(0, int(st.get("eta_seconds") or 0)),
                    current_file=str(st.get("current_file") or ""),
                    started_at=time.time(),
                )
                
                # Attach original state so callback can add to it instead of overwriting
                setattr(job, "initial_total_files", initial_total_files)
                setattr(job, "initial_completed_files", initial_completed_files)
                
                job.is_indexing = True
                job.error = None
                with self._jobs_lock:
                    self._jobs[job_id] = job
                    self._active_index_job_id = job_id
                
                # We start a new thread to resume indexing, but we need to ensure the worker starts correctly 
                # after everything is ready.
                t = threading.Thread(target=self._run_index_job, args=(job,), daemon=True)
                t.start()
                folder_name = os.path.basename(folder.rstrip(os.sep)) or "<root>"
                logger.info(f"[Backend] Resumed folder indexing job: {job_id} ({folder_name})")
                print(f"[Backend] Resumed folder indexing job: {job_id} ({folder_name})")
        except Exception as e:
            print(f"[Backend] Failed to resume active indexing job: {e}")

    def start_index(self, folder: str) -> Dict[str, Any]:
        folder = os.path.abspath(os.path.expanduser(folder))
        emb_ok = _model_dir_installed(settings.LOCAL_EMBEDDING_MODEL_PATH)
        rr_ok = _model_dir_installed(settings.LOCAL_RERANKER_MODEL_PATH) or bool(getattr(settings, "RERANKER_OPTIONAL", False))
        if not (emb_ok and rr_ok):
            return {
                "ok": False,
                "error": "core_models_not_ready",
                "message": "Embedding/Reranker 模型未就绪，请先在 onboarding 完成核心模型下载。",
                "embedding_installed": bool(emb_ok),
                "reranker_installed": bool(rr_ok),
            }
        self.sources.add_folder(folder)
        with self._jobs_lock:
            if (self._active_index_job_id
                    and self._active_index_job_id in self._jobs
                    and self._jobs[self._active_index_job_id].is_indexing):
                running = self._jobs[self._active_index_job_id]
                self._index_append_queue.append({"kind": "folder", "folder": folder})
                logger.info(f"[Backend] 索引进行中，文件夹已追加到队列: {folder}")
                return {"ok": True, "job_id": running.job_id, "folder": folder, "appended": True}
        job_id = str(uuid.uuid4())
        job = IndexJobState(job_id=job_id, folder=folder)
        with self._jobs_lock:
            job.is_indexing = True
            self._jobs[job_id] = job
            self._active_index_job_id = job_id
        self._write_active_index_state(job, kind="folder")
        t = threading.Thread(target=self._run_index_job, args=(job,), daemon=True)
        t.start()
        return {"ok": True, "job_id": job_id, "folder": folder}

    def index_files(self, files: List[str]) -> Dict[str, Any]:
        if not files:
            return {"ok": False, "error": "files 参数必须是非空数组"}
        emb_ok = _model_dir_installed(settings.LOCAL_EMBEDDING_MODEL_PATH)
        rr_ok = _model_dir_installed(settings.LOCAL_RERANKER_MODEL_PATH) or bool(getattr(settings, "RERANKER_OPTIONAL", False))
        if not (emb_ok and rr_ok):
            return {
                "ok": False,
                "error": "core_models_not_ready",
                "message": "Embedding/Reranker 模型未就绪，请先在 onboarding 完成核心模型下载。",
                "embedding_installed": bool(emb_ok),
                "reranker_installed": bool(rr_ok),
            }
        file_paths = [os.path.abspath(os.path.expanduser(fp)) for fp in files]
        
        with self._jobs_lock:
            if (self._active_index_job_id
                    and self._active_index_job_id in self._jobs
                    and self._jobs[self._active_index_job_id].is_indexing):
                running = self._jobs[self._active_index_job_id]
                
                unskipped = []
                for fp in file_paths:
                    fp_key = _source_path_key(fp)
                    if fp_key in running.skipped_files:
                        running.skipped_files.discard(fp_key)
                        running.total_files += 1
                        unskipped.append(fp)
                if unskipped:
                    logger.info(f"[Backend] 恢复 {len(unskipped)} 个之前跳过的文件")
                
                existing: Set[str] = set()
                try:
                    st = self._read_active_index_state()
                    if st and st.get("files"):
                        existing.update(os.path.abspath(os.path.expanduser(f)) for f in st["files"])
                except Exception:
                    pass
                for item in self._index_append_queue:
                    if item.get("kind") == "files":
                        existing.update(item.get("files", []))
                
                new_files = [fp for fp in file_paths if fp not in existing and fp not in unskipped]
                
                if new_files:
                    self._index_append_queue.append({"kind": "files", "files": new_files})
                    logger.info(f"[Backend] 索引进行中，{len(new_files)} 个新文件已追加到队列（去重后）")
                
                total_restored = len(unskipped) + len(new_files)
                if total_restored == 0:
                    logger.info(f"[Backend] 所有 {len(file_paths)} 个文件已在索引列表中，跳过")
                    return {"ok": True, "job_id": running.job_id, "files": file_paths, "appended": False, "message": "all duplicates"}
                
                return {"ok": True, "job_id": running.job_id, "files": file_paths, "appended": True,
                        "unskipped": len(unskipped), "queued": len(new_files)}
        job_id = str(uuid.uuid4())
        job = IndexJobState(job_id=job_id, folder="", total_files=len(file_paths), completed_files=0)
        with self._jobs_lock:
            job.is_indexing = True
            self._jobs[job_id] = job
            self._active_index_job_id = job_id
        self._write_active_index_state(job, kind="files", files=file_paths)
        t = threading.Thread(target=self._run_file_index_job, args=(job, file_paths), daemon=True)
        t.start()
        return {"ok": True, "job_id": job_id, "files": file_paths}

    def get_index_status(self, job_id: str) -> Dict[str, Any]:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if not job:
            st = self._read_active_index_state()
            if st and st.get("job_id") == job_id:
                return {"job": st}
            return {"error": "job_not_found", "job_id": job_id}
        return {"job": job.to_payload()}

    def cancel_index(self, job_id: str = "") -> Dict[str, Any]:
        with self._jobs_lock:
            active_id = self._active_index_job_id
            active_job = self._jobs.get(active_id) if active_id else None
        
        if not active_id or not active_job or not active_job.is_indexing:
            return {"ok": False, "error": "no_active_job"}
            
        if job_id and job_id != active_id:
            return {"ok": False, "error": "job_id_mismatch", "active_job_id": active_id}
            
        try:
            print(f"[Backend] 收到取消索引请求，正在停止任务 {active_id}...")
            self._index_cancel_event.set()
            
            with self._jobs_lock:
                if self._active_index_job_id in self._jobs:
                    self._jobs[self._active_index_job_id].error = "cancelled"
            
            self._clear_active_index_state()
            self._index_append_queue.clear()
            print("[Backend] Cancel signal sent; waiting for the current file to finish cleanly")
            
        except Exception as e:
            print(f"[Backend] Failed to cancel indexing: {e}")
            
        return {"ok": True, "job_id": active_id}

    def skip_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Mark files to be skipped by the active indexing thread."""
        norm_paths = [_source_path_key(fp) for fp in file_paths]
        with self._jobs_lock:
            active_id = self._active_index_job_id
            job = self._jobs.get(active_id) if active_id else None
        if job and job.is_indexing:
            new_skipped = [p for p in norm_paths if p not in job.skipped_files]
            job.skipped_files.update(norm_paths)
            if new_skipped:
                job.total_files = max(0, job.total_files - len(new_skipped))
                job.skipped_count = getattr(job, 'skipped_count', 0) + len(new_skipped)
                if job.completed_files > job.total_files:
                    job.completed_files = job.total_files
            logger.info(f"[Backend] Marked {len(new_skipped)} files as skipped (total_files={job.total_files})")
            return {"ok": True, "skipped": norm_paths, "skipped_count": len(new_skipped), "job_id": active_id}
        removed = 0
        for item in list(self._index_append_queue):
            if item.get("kind") == "files":
                before = len(item.get("files", []))
                item["files"] = [f for f in item.get("files", []) if _source_path_key(f) not in set(norm_paths)]
                removed += before - len(item.get("files", []))
        return {"ok": True, "skipped": norm_paths, "skipped_count": removed, "job_id": None}

    def _process_append_queue(self) -> None:
        """Process queued index additions after the current job completes."""
        if not self._index_append_queue:
            return
        folders = []
        files = []
        while self._index_append_queue:
            item = self._index_append_queue.pop(0)
            if item.get("kind") == "folder":
                folders.append(item["folder"])
            elif item.get("kind") == "files":
                files.extend(item.get("files", []))
        if folders:
            job_id = str(uuid.uuid4())
            job = IndexJobState(job_id=job_id, folder=folders[0])
            with self._jobs_lock:
                job.is_indexing = True
                self._jobs[job_id] = job
                self._active_index_job_id = job_id
            for f in folders[1:]:
                self._index_append_queue.append({"kind": "folder", "folder": f})
            if files:
                self._index_append_queue.append({"kind": "files", "files": files})
            self._write_active_index_state(job, kind="folder")
            t = threading.Thread(target=self._run_index_job, args=(job,), daemon=True)
            t.start()
            logger.info(f"[Backend] 自动开始队列中的文件夹索引: {folders[0]}")
        elif files:
            job_id = str(uuid.uuid4())
            job = IndexJobState(job_id=job_id, folder="", total_files=len(files), completed_files=0)
            with self._jobs_lock:
                job.is_indexing = True
                self._jobs[job_id] = job
                self._active_index_job_id = job_id
            self._write_active_index_state(job, kind="files", files=files)
            t = threading.Thread(target=self._run_file_index_job, args=(job, files), daemon=True)
            t.start()
            logger.info(f"[Backend] 自动开始队列中的 {len(files)} 个文件索引")

    # ─── Query (sync) ───

    def query(self, message: str, active_source_ids: Optional[List[str]] = None,
              model_id: Optional[str] = None, session_id: Optional[str] = None, language: Optional[str] = None, opened_file_path: Optional[str] = None) -> Dict[str, Any]:
        running = self._indexing_in_progress()
        if running is not None:
            return {
                "ok": False, "answer": "正在索引中，请等待索引完成后再提问。",
                "sources": [], "trace": [], "query_type": "busy",
                "need_clarify": False, "relevantFiles": [],
                "error": "indexing_in_progress", "job_id": running.job_id,
            }
        try:
            # Honor frontend-selected chat model for this request.
            self._apply_runtime_model_hint(model_id)
            agent = self._ensure_agent()
            active_paths = self._get_active_paths(active_source_ids) if active_source_ids is not None else None
            set_active_paths(active_paths)
            set_active_session_id(session_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "ok": False, "answer": f"请求处理失败: {e}",
                "sources": [], "trace": [], "query_type": "error",
                "need_clarify": False, "relevantFiles": [],
            }
        answer_parts: List[str] = []
        trace: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        query_type = "agent"
        ok = True
        try:
            for ev in agent._query_stream_intent_dispatch(
                message,
                active_paths=active_paths,
                session_id=session_id,
                emit_status=True,
                prompt_language=language,
                opened_file_path=opened_file_path,
            ):
                ev_type = ev.get("type")
                if ev_type == "text":
                    if "delta" in ev:
                        answer_parts.append(ev.get("delta", ""))
                    else:
                        answer_parts.append(ev.get("content", ""))
                elif ev_type == "trace_append":
                    item = ev.get("item")
                    if item:
                        trace.append(item)
                elif ev_type == "done":
                    ok = bool(ev.get("ok", True))
                    query_type = ev.get("query_type", query_type)
                    sources = ev.get("sources", []) or []
                    trace = ev.get("trace", trace) or trace
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok = False
            answer_parts.append(f"请求处理失败: {e}")
            query_type = "error"
        relevant_files = []
        for s in (sources or [])[:20]:
            p = s.get("file_path") or ""
            n = s.get("file_name") or os.path.basename(p) or ""
            relevant_files.append({
                "id": _stable_id(f"file:{p or n}"),
                "name": n,
                "type": _icon_type_for_path(p) if p else "doc",
                "path": p,
            })
        return {
            "ok": ok, "answer": "".join(answer_parts).strip(),
            "sources": sources, "trace": trace,
            "query_type": query_type, "need_clarify": query_type == "clarify",
            "relevantFiles": relevant_files,
        }

    # ─── Query (streaming generator) ───

    def query_stream(self, message: str, active_source_ids: Optional[List[str]] = None,
                     model_id: Optional[str] = None,
                     session_id: Optional[str] = None,
                     language: Optional[str] = None,
                     opened_file_path: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        import json
        import uuid
        import time

        def _debug(hyp_id, loc, msg, data=None):
            # Dev-time debug helper — standard logger used in production
            logger.debug(f"[debug] {loc}: {msg} {data or {}}")

        _debug("A,B,C,D", "backend_core.py:query_stream_start", "query_stream started", {"message": message, "active_source_ids": active_source_ids})
        def _preview_list(values: Optional[List[Any]], max_items: int = 8) -> str:
            arr = list(values or [])
            if len(arr) <= max_items:
                return str(arr)
            return f"{arr[:max_items]} ... (+{len(arr) - max_items} more)"

        logger.info(
            f"[query_stream] start: active_source_ids_count={len(active_source_ids or [])} "
            f"preview={_preview_list(active_source_ids, 8)}"
        )
        yield {"event": "status", "data": {"type": "status", "phase": "running", "message": "Processing..."}}

        logger.info("[query_stream] yielded first status")
        running = self._indexing_in_progress()
        logger.info(f"[query_stream] indexing_in_progress = {running}")
        if running is not None:
            logger.info("[query_stream] indexing in progress, aborting")
            yield {"event": "done", "data": {
                "type": "done", "ok": False, "query_type": "busy",
                "message": "正在索引中，请等待索引完成后再提问。",
                "error": "indexing_in_progress", "job_id": running.job_id,
            }}
            return

        try:
            logger.info("[query_stream] ensuring agent")
            # Honor frontend-selected chat model for this request.
            self._apply_runtime_model_hint(model_id)
            agent = self._ensure_agent()
            agent.clear_abort_flag(session_id)
            active_paths = self._get_active_paths(active_source_ids) if active_source_ids is not None else None
            logger.info(
                f"[query_stream] computed active_paths_count={len(active_paths or [])} "
                f"preview={_preview_list(active_paths, 5)}"
            )
            set_active_paths(active_paths)
            set_active_session_id(session_id)
        except Exception as e:
            try:
                import traceback
                tb = traceback.format_exc()
                logger.error(f"[query_stream] error computing paths: {e}\n{tb}")
            except Exception:
                pass
            yield {"event": "done", "data": {"type": "done", "ok": False, "query_type": "error", "error": str(e)}}
            return

        try:
            logger.info("[query_stream] starting _query_stream_intent_dispatch")
            files_emitted = False
            text_event_count = 0
            for ev in agent._query_stream_intent_dispatch(
                message,
                active_paths=active_paths,
                session_id=session_id,
                emit_status=True,
                prompt_language=language,
                opened_file_path=opened_file_path,
            ):
                ev_type = ev.get("type")
                if ev_type == "text":
                    text_event_count += 1
                    if text_event_count % 200 == 1:
                        logger.info(f"[query_stream] yielding text chunk #{text_event_count}")
                elif ev_type in ["status", "done", "files"]:
                    logger.info(f"[query_stream] yielding {ev_type}")
                
                if ev_type == "status":
                    yield {"event": "status", "data": ev}
                elif ev_type == "trace_append":
                    yield {"event": "trace_append", "data": ev}
                elif ev_type == "files":
                    files_emitted = True
                    yield {"event": "files", "data": ev}
                elif ev_type == "sources":
                    preview = []
                    for s in (ev.get("content") or [])[:20]:
                        fp = s.get("file_path") or ""
                        fn = s.get("file_name") or os.path.basename(fp) or ""
                        ftype = s.get("type") or (_icon_type_for_path(fp) if fp else "doc")
                        preview.append({
                            "id": _stable_id(f"file:{fp or fn}"),
                            "file_name": fn,
                            "file_path": fp,
                            "type": ftype,
                            "iconType": ftype,
                            "doc_category": s.get("doc_category", ""),
                            "doc_summary": s.get("doc_summary", ""),
                        })
                    files_emitted = True
                    yield {
                        "event": "files",
                        "data": {
                            "type": "files",
                            "total": int(ev.get("total_matches") or len(preview)),
                            "total_matches": int(ev.get("total_matches") or len(preview)),
                            "shown_count": int(ev.get("shown_count") or len(preview)),
                            "preview": preview,
                            "all": preview,
                        },
                    }
                elif ev_type == "opened_file":
                    try:
                        f = (ev.get("file") or {}) if isinstance(ev, dict) else {}
                        fp = (f.get("file_path") or "").strip()
                        content = ev.get("content") or ""
                        is_image = (f.get("type") == "image" or f.get("iconType") == "image")
                        if not is_image and not (isinstance(content, str) and content.startswith("data:image/")):
                            cache_opened_file(session_id, fp, content)
                    except Exception:
                        pass
                    yield {"event": "opened_file", "data": ev}
                elif ev_type == "thinking":
                    yield {"event": "thinking", "data": ev}
                elif ev_type == "text":
                    yield {"event": "text", "data": ev}
                elif ev_type == "done":
                    logger.info(f"[query_stream] total text chunks={text_event_count}")
                    srcs = ev.get("sources") or []
                    if srcs and not files_emitted:
                        preview = []
                        for s in srcs[:20]:
                            fp = s.get("file_path") or ""
                            fn = s.get("file_name") or os.path.basename(fp) or ""
                            ftype = s.get("type") or _icon_type_for_path(fp) if fp else "doc"
                            preview.append({
                                "id": _stable_id(f"file:{fp or fn}"),
                                "file_name": fn, "file_path": fp,
                                "type": ftype, "iconType": ftype,
                                "doc_category": s.get("doc_category", ""),
                                "doc_summary": s.get("doc_summary", ""),
                            })
                        total_matches = max(len(srcs), int(ev.get("total_matches") or 0))
                        yield {
                            "event": "files",
                            "data": {
                                "type": "files",
                                "total": total_matches,
                                "total_matches": total_matches,
                                "shown_count": len(preview),
                                "preview": preview,
                                "all": preview,
                            },
                        }
                    yield {"event": "done", "data": ev}
            logger.info("[query_stream] finished _query_stream_intent_dispatch stream loop")
        except Exception as e:
            try:
                import traceback
                traceback.print_exc()
                logger.error(f"[query_stream] exception in stream: {e}")
            except Exception:
                pass
            yield {"event": "done", "data": {"type": "done", "ok": False, "query_type": "error", "error": str(e)}}
        finally:
            try:
                time.sleep(0.15)
                agent = self._ensure_agent()
                agent.clear_abort_flag(session_id)
            except Exception:
                pass

    def abort_query(self, session_id: str) -> Dict[str, Any]:
        """Abort an ongoing query stream."""
        try:
            agent = self._ensure_agent()
            agent.set_abort_flag(session_id)
            return {"status": "success", "session_id": session_id}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ─── Settings ───

    def get_settings(self) -> Dict[str, Any]:
        return dict(self.pref_manager.preferences)

    def update_settings(self, key: str, value: Any) -> Dict[str, Any]:
        self.pref_manager.set(key, value)
        return {"ok": True}

    # ─── Internal helpers ───

    def _ensure_agent(self):
        a = self._agent
        if a is not None:
            return a
        with self._agent_lock:
            if self._agent is None:
                from core.langgraph_agent import FileAgent

                print("[Backend] 初始化 FileAgent (工具调用版)...")
                self._agent = FileAgent(llm_manager=self.llm_manager)
        return self._agent

    def _set_core_embedding_state(
        self,
        *,
        status: str,
        error: Optional[str] = None,
        percent: Optional[float] = None,
    ) -> None:
        with self._core_models_lock:
            emb = dict(self._core_models_state.get("embedding") or {})
            emb["status"] = status
            emb["error"] = error
            if percent is not None:
                emb["percent"] = float(percent)
            self._core_models_state["embedding"] = emb

    def _embedding_runtime_smoke_test(self, kb: Any) -> bool:
        emb = getattr(kb, "embedding_model", None)
        if emb is None:
            return False
        try:
            vec = emb.get_text_embedding("unfoldly embedding readiness check")
            if not isinstance(vec, list) or not vec:
                return False
            return any(abs(float(x)) > 1e-12 for x in vec[: min(len(vec), 64)])
        except Exception as e:
            logger.warning(f"[EmbeddingReady] smoke test failed: {e}")
            return False

    def _drop_embedding_runtime(self, kb: Any) -> None:
        emb = getattr(kb, "embedding_model", None)
        raw = getattr(emb, "_model", emb)
        try:
            if raw is not None and hasattr(raw, "close"):
                raw.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            kb.embedding_model = None
        except Exception:
            pass

    def _ensure_embedding_runtime_active(
        self,
        kb: Optional[Any] = None,
        *,
        update_core_state: bool = False,
    ) -> bool:
        """
        Ensure the embedding model is not only downloaded, but actually loaded and
        able to produce a non-empty vector. This is the hard gate before indexing.
        """
        with self._embedding_runtime_lock:
            if kb is None:
                agent = self._ensure_agent()
                kb = getattr(agent, "kb", None)
            if kb is None:
                logger.error("[EmbeddingReady] knowledge base unavailable")
                if update_core_state:
                    self._set_core_embedding_state(status="error", error="knowledge base unavailable")
                return False

            if self._embedding_runtime_smoke_test(kb):
                if update_core_state:
                    self._set_core_embedding_state(status="installed", error=None, percent=100.0)
                return True

            if not _model_dir_installed(settings.LOCAL_EMBEDDING_MODEL_PATH):
                msg = f"embedding model file is not installed: {settings.LOCAL_EMBEDDING_MODEL_PATH}"
                logger.error(f"[EmbeddingReady] {msg}")
                if update_core_state:
                    self._set_core_embedding_state(status="error", error=msg)
                return False

            if update_core_state:
                self._set_core_embedding_state(status="downloading", error=None, percent=99.0)

            logger.warning("[EmbeddingReady] embedding runtime missing; initialising now.")
            self._drop_embedding_runtime(kb)
            try:
                kb._init_embedding()
            except Exception as e:
                logger.exception(f"[EmbeddingReady] embedding initialisation failed: {e}")

            ok = self._embedding_runtime_smoke_test(kb)
            if ok:
                logger.info("[EmbeddingReady] embedding runtime ready.")
                if update_core_state:
                    self._set_core_embedding_state(status="installed", error=None, percent=100.0)
                try:
                    if _model_dir_installed(settings.LOCAL_RERANKER_MODEL_PATH) and getattr(kb, "reranker", None) is None:
                        kb._init_reranker()
                except Exception as rr_err:
                    logger.warning(f"[EmbeddingReady] reranker reinit skipped/failed: {rr_err}")
                return True

            msg = "embedding runtime unavailable after initialisation"
            logger.error(f"[EmbeddingReady] {msg}")
            if update_core_state:
                self._set_core_embedding_state(status="error", error=msg)
            return False

    def _reinit_embedding_after_download(self) -> bool:
        return self._ensure_embedding_runtime_active(update_core_state=True)

    def _indexing_in_progress(self) -> Optional[IndexJobState]:
        if self._index_cancel_event.is_set():
            return None
        with self._jobs_lock:
            if self._active_index_job_id and self._active_index_job_id in self._jobs:
                j = self._jobs[self._active_index_job_id]
                if j.is_indexing:
                    return j
            for j in self._jobs.values():
                if j and j.is_indexing:
                    return j
        return None

    def _suspend_reranker_for_indexing(self, kb: Optional[Any] = None) -> None:
        if self._reranker_suspended_for_indexing:
            return
        try:
            target_kb = kb
            if target_kb is None:
                agent = self._agent
                target_kb = getattr(agent, "kb", None) if agent is not None else None
            if target_kb is None or not hasattr(target_kb, "unload_reranker"):
                return
            unloaded = bool(target_kb.unload_reranker(reason="backend_indexing"))
            self._reranker_suspended_for_indexing = unloaded
            if unloaded:
                logger.info("[Indexing] reranker unloaded for indexing to reduce Metal/GGML pressure.")
        except Exception as e:
            logger.warning(f"[Indexing] failed to unload reranker before indexing: {e}")

    def _restore_reranker_after_indexing(self) -> None:
        if not self._reranker_suspended_for_indexing:
            return
        try:
            agent = self._agent
            kb = getattr(agent, "kb", None) if agent is not None else None
            if kb is None or not hasattr(kb, "ensure_reranker_ready"):
                return
            ready = bool(kb.ensure_reranker_ready(reason="post_indexing_restore"))
            if ready:
                logger.info("[Indexing] reranker restored after indexing.")
            else:
                logger.warning("[Indexing] reranker restore after indexing returned not-ready; query rerank will stay degraded until next reload.")
        except Exception as e:
            logger.warning(f"[Indexing] failed to restore reranker after indexing: {e}")
        finally:
            self._reranker_suspended_for_indexing = False

    def _invalidate_file_id_map(self) -> None:
        """Mark the id-to-path cache dirty so the next lookup rebuilds it."""
        with self._file_id_map_lock:
            self._file_id_map_dirty = True
        get_logger().info("[file_id_map] cache invalidated (indexing completed)")

    def _ensure_file_id_map(self) -> Dict[str, str]:
        """Return the id-to-path cache, rebuilding it once when dirty."""
        with self._file_id_map_lock:
            if not self._file_id_map_dirty:
                return self._file_id_map

        new_map: Dict[str, str] = {}

        self.sources.load()
        for folder in self.sources.folders:
            new_map[_stable_id(f"folder:{folder}")] = folder
        for fp in self.sources.files:
            new_map[_stable_id(f"file:{fp}")] = fp

        try:
            agent = self._agent
            if agent is not None and not self._should_avoid_live_kb_reads():
                kb = getattr(agent, "kb", None)
                if kb is not None and hasattr(kb, "get_indexed_file_paths"):
                    for fp in kb.get_indexed_file_paths():
                        new_map[_stable_id(f"file:{fp}")] = fp
        except Exception as e:
            get_logger().warning(f"[file_id_map] DB read failed, map may be incomplete: {e}")

        with self._file_id_map_lock:
            self._file_id_map = new_map
            self._file_id_map_dirty = False

        get_logger().info(f"[file_id_map] cache rebuilt: {len(new_map)} entries")
        return new_map

    def _get_active_paths(self, active_ids: List[str]) -> List[str]:
        if not active_ids:
            return []

        id_map = self._ensure_file_id_map()
        active_paths = [id_map[aid] for aid in active_ids if aid in id_map]

        get_logger().info(f"[_get_active_paths] active_ids={len(active_ids)}, map_size={len(id_map)}, mapped={len(active_paths)}")

        if not active_paths and active_ids:
            get_logger().warning("[_get_active_paths] no valid paths found for active_ids! Fallback to empty restrict.")
            return ["/dev/null/no_match"]

        return active_paths

    def _core_models_payload(self) -> Dict[str, Any]:
        emb_installed = _model_dir_installed(settings.LOCAL_EMBEDDING_MODEL_PATH)
        rr_installed = _model_dir_installed(settings.LOCAL_RERANKER_MODEL_PATH)
        rr_optional = getattr(settings, "RERANKER_OPTIONAL", False)
        if rr_optional and not rr_installed:
            rr_installed = True
        with self._core_models_lock:
            st = dict(self._core_models_state)
            emb_state = dict(st.get("embedding") or {})
            rr_state = dict(st.get("reranker") or {})
            is_downloading = bool(st.get("is_downloading"))
            progress = int(st.get("progress") or 0)

        def _normalize(item: Dict[str, Any], installed: bool) -> Dict[str, Any]:
            status = (item.get("status") or "idle")
            err = item.get("error")
            if installed and is_downloading and status == "downloading":
                status = "downloading"
            elif installed and status not in {"error"}:
                status = "installed"
            elif not installed and status == "installed":
                # Local file was removed but in-memory status may still be "installed".
                # Downgrade it so frontend progress does not jump to 100% incorrectly.
                status = "downloading" if is_downloading else "idle"
            result: Dict[str, Any] = {"installed": bool(installed), "status": status, "error": err}
            if status == "installed":
                result["percent"] = 100.0
            elif status == "downloading":
                for key in ("percent", "speed", "eta", "downloaded_bytes", "total_bytes"):
                    if key in item:
                        result[key] = item[key]
            else:
                result["percent"] = 0.0
                if "total_bytes" in item:
                    result["total_bytes"] = item.get("total_bytes", 0)
                if "downloaded_bytes" in item:
                    result["downloaded_bytes"] = item.get("downloaded_bytes", 0)
            return result

        embedding_payload = _normalize(emb_state, emb_installed)
        reranker_payload = _normalize(rr_state, rr_installed)
        if emb_installed and embedding_payload.get("status") != "installed":
            embedding_payload["installed"] = False

        if bool(embedding_payload.get("installed")) and bool(reranker_payload.get("installed")):
            progress2 = 100
        elif emb_installed or rr_installed:
            progress2 = max(progress, 50)
        else:
            progress2 = progress
        return {
            "embedding": embedding_payload,
            "reranker": reranker_payload,
            "progress": int(max(0, min(100, progress2))),
            "is_downloading": is_downloading,
        }

    def _asr_model_payload(self) -> Dict[str, Any]:
        try:
            from core.media.media_expert import MediaExpert
            disk = MediaExpert.get_ggml_model_status()
        except Exception as e:
            disk = {
                "installed": False,
                "status": "idle",
                "error": str(e),
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "percent": 0.0,
            }

        installed = bool(disk.get("installed"))
        with self._asr_model_lock:
            st = dict(self._asr_model_state)
            item = dict(st.get("asr") or {})
            is_downloading = bool(st.get("is_downloading"))
            progress = int(st.get("progress") or 0)

        status = str(item.get("status") or disk.get("status") or "idle")
        err = item.get("error") or disk.get("error")
        if installed:
            status = "installed"
            err = None
        elif not installed and status == "installed":
            status = "downloading" if is_downloading else "idle"

        result: Dict[str, Any] = {
            "installed": installed,
            "status": status,
            "error": err,
            "model_name": disk.get("model_name"),
            "filename": disk.get("filename"),
            "path": disk.get("path"),
            "tmp_path": disk.get("tmp_path"),
        }

        for key in ("downloaded_bytes", "total_bytes"):
            if key in item and status == "downloading":
                result[key] = item.get(key)
            else:
                result[key] = disk.get(key, 0)

        if status == "installed":
            result["percent"] = 100.0
        elif status == "downloading":
            result["percent"] = item.get("percent", disk.get("percent", 0.0))
            for key in ("speed", "eta"):
                if key in item:
                    result[key] = item[key]
        else:
            result["percent"] = 0.0 if status == "error" else disk.get("percent", 0.0)

        if installed:
            progress2 = 100
        elif status == "downloading":
            progress2 = max(progress, int(float(result.get("percent") or 0)))
        else:
            progress2 = int(float(disk.get("percent") or 0))

        if installed:
            with self._asr_model_lock:
                self._asr_model_state["is_downloading"] = False
                self._asr_model_state["progress"] = 100
                self._asr_model_state["asr"] = {
                    **dict(self._asr_model_state.get("asr") or {}),
                    "status": "installed",
                    "error": None,
                    "percent": 100.0,
                    "downloaded_bytes": result.get("downloaded_bytes", 0),
                    "total_bytes": result.get("total_bytes", 0),
                    "speed": 0,
                    "eta": 0,
                }

        return {
            "asr": result,
            "progress": int(max(0, min(100, progress2))),
            "is_downloading": is_downloading,
        }

    def _download_core_models_worker(self, run_id: int):
        def _is_run_active() -> bool:
            with self._core_models_lock:
                return (
                    int(self._core_models_state.get("run_id") or 0) == int(run_id)
                    and bool(self._core_models_state.get("is_downloading"))
                )

        def _mutate_state(mutator) -> bool:
            with self._core_models_lock:
                if int(self._core_models_state.get("run_id") or 0) != int(run_id):
                    return False
                mutator(self._core_models_state)
                return True

        def _run_download_with_stall_guard(task_name: str, state_key: str, fn) -> None:
            try:
                stall_timeout = int(os.getenv("FILEAGENT_DOWNLOAD_STALL_TIMEOUT_SEC", "75") or 75)
            except Exception:
                stall_timeout = 75
            stall_timeout = max(20, min(stall_timeout, 1800))

            result: Dict[str, Any] = {"done": False, "ok": False, "err": None}

            def _target():
                try:
                    if not _is_run_active():
                        return
                    ok = bool(fn())
                    if not _is_run_active():
                        return
                    if not ok:
                        raise RuntimeError(f"{task_name} download failed")
                    result["ok"] = True
                except Exception as e:
                    result["err"] = e
                finally:
                    result["done"] = True

            t = threading.Thread(target=_target, daemon=True)
            t.start()

            last_active_ts = time.time()
            last_bytes = -1

            while not bool(result.get("done")):
                if not _is_run_active():
                    raise RuntimeError("core_models_download_cancelled")
                time.sleep(1.0)
                with self._core_models_lock:
                    st = dict(self._core_models_state.get(state_key) or {})
                downloaded = int(st.get("downloaded_bytes") or 0)
                speed = float(st.get("speed") or 0)
                if downloaded > last_bytes or speed > 0:
                    last_bytes = downloaded
                    last_active_ts = time.time()
                    continue
                if (time.time() - last_active_ts) > stall_timeout:
                    raise TimeoutError(
                        f"{task_name} download stalled for {stall_timeout}s "
                        "(network offline or unstable)"
                    )

            if not _is_run_active():
                raise RuntimeError("core_models_download_cancelled")
            if result.get("err") is not None:
                raise result["err"]
            if not bool(result.get("ok")):
                raise RuntimeError(f"{task_name} download failed")

        try:
            if not _is_run_active():
                return

            emb_file_ready = False
            _mutate_state(lambda st: (
                st["embedding"].update({"status": "downloading", "error": None})
            ))
            try:
                def _emb_progress(info: dict):
                    def _apply(st: Dict[str, Any]):
                        emb = st["embedding"]
                        emb["percent"] = round(info.get("percent", 0), 1)
                        emb["speed"] = round(info.get("speed_bytes_per_sec", 0))
                        emb["eta"] = round(info.get("eta_seconds", 0))
                        emb["downloaded_bytes"] = info.get("downloaded_bytes", 0)
                        emb["total_bytes"] = info.get("total_bytes", 0)
                    _mutate_state(_apply)

                if settings.LOCAL_EMBEDDING_MODEL_PATH.endswith('.gguf'):
                    repo_id = getattr(settings, "EMBEDDING_REPO_ID", settings.EMBEDDING_MODEL)
                    filename = os.path.basename(settings.LOCAL_EMBEDDING_MODEL_PATH)
                    _run_download_with_stall_guard(
                        "embedding",
                        "embedding",
                        lambda: ensure_gguf_downloaded(
                            repo_id,
                            filename,
                            settings.LOCAL_EMBEDDING_MODEL_PATH,
                            on_progress=_emb_progress,
                            should_cancel=lambda: (not _is_run_active()),
                        ),
                    )
                else:
                    _run_download_with_stall_guard(
                        "embedding",
                        "embedding",
                        lambda: ensure_gguf_downloaded(
                            getattr(settings, "EMBEDDING_REPO_ID", settings.EMBEDDING_MODEL),
                            os.path.basename(settings.LOCAL_EMBEDDING_MODEL_PATH),
                            settings.LOCAL_EMBEDDING_MODEL_PATH,
                            on_progress=_emb_progress,
                            should_cancel=lambda: (not _is_run_active()),
                        ),
                    )
                emb_file_ready = _model_dir_installed(settings.LOCAL_EMBEDDING_MODEL_PATH)
                _mutate_state(lambda st: (
                    st["embedding"].update({
                        "status": "downloading" if emb_file_ready else "error",
                        "error": None if emb_file_ready else "embedding_incomplete_after_download",
                        "percent": 99.0 if emb_file_ready else 0.0,
                    }),
                    st.__setitem__("progress", 50 if emb_file_ready else 0),
                ))
            except Exception as e:
                if not _is_run_active():
                    return
                _mutate_state(lambda st: (
                    st["embedding"].update({"status": "error", "error": str(e)}),
                    st.__setitem__("progress", 0),
                ))
                emb_file_ready = False

            if not _is_run_active():
                return

            if not emb_file_ready:
                if not getattr(settings, "RERANKER_OPTIONAL", False):
                    _mutate_state(lambda st: st["reranker"].update({"status": "idle", "error": None}))
                return

            if not getattr(settings, "RERANKER_OPTIONAL", False):
                _mutate_state(lambda st: st["reranker"].update({"status": "downloading", "error": None}))

                def _rr_progress(info: dict):
                    def _apply(st: Dict[str, Any]):
                        rr = st["reranker"]
                        rr["percent"] = round(info.get("percent", 0), 1)
                        rr["speed"] = round(info.get("speed_bytes_per_sec", 0))
                        rr["eta"] = round(info.get("eta_seconds", 0))
                        rr["downloaded_bytes"] = info.get("downloaded_bytes", 0)
                        rr["total_bytes"] = info.get("total_bytes", 0)
                    _mutate_state(_apply)

                try:
                    _run_download_with_stall_guard(
                        "reranker",
                        "reranker",
                        lambda: ensure_gguf_downloaded(
                            settings.RERANKER_MODEL,
                            settings.RERANKER_GGUF_FILE,
                            settings.LOCAL_RERANKER_MODEL_PATH,
                            on_progress=_rr_progress,
                            should_cancel=lambda: (not _is_run_active()),
                        ),
                    )
                    rr_ready = _model_dir_installed(settings.LOCAL_RERANKER_MODEL_PATH)
                    _mutate_state(lambda st: (
                        st["reranker"].update({
                            "status": "installed" if rr_ready else "error",
                            "error": None if rr_ready else "reranker_incomplete_after_download",
                        }),
                        st.__setitem__("progress", 100 if rr_ready else int(st.get("progress") or 0)),
                    ))
                except Exception as e:
                    if not _is_run_active():
                        return
                    _mutate_state(lambda st: st["reranker"].update({"status": "error", "error": str(e)}))
            else:
                _mutate_state(lambda st: (
                    st["reranker"].update({"status": "installed", "error": None}),
                    st.__setitem__("progress", 100),
                ))

            if not _is_run_active():
                return

            # A complete file on disk is not enough: indexing needs the embedding
            # runtime loaded and able to produce vectors. Keep it as normal
            # download progress at 99% until the smoke test passes.
            _mutate_state(lambda st: (
                st["embedding"].update({"status": "downloading", "error": None, "percent": 99.0}),
                st.__setitem__("progress", max(int(st.get("progress") or 0), 99)),
            ))
            if not self._reinit_embedding_after_download():
                return
            _mutate_state(lambda st: (
                st["embedding"].update({"status": "installed", "error": None, "percent": 100.0}),
                st.__setitem__("progress", 100),
            ))

        finally:
            with self._core_models_lock:
                if int(self._core_models_state.get("run_id") or 0) != int(run_id):
                    return
                self._core_models_state["is_downloading"] = False

    def _download_asr_model_worker(self, run_id: int):
        def _is_run_active() -> bool:
            with self._asr_model_lock:
                return (
                    int(self._asr_model_state.get("run_id") or 0) == int(run_id)
                    and bool(self._asr_model_state.get("is_downloading"))
                )

        def _mutate_state(mutator) -> bool:
            with self._asr_model_lock:
                if int(self._asr_model_state.get("run_id") or 0) != int(run_id):
                    return False
                mutator(self._asr_model_state)
                return True

        try:
            if not _is_run_active():
                return
            _mutate_state(lambda st: (
                st["asr"].update({"status": "downloading", "error": None}),
                st.__setitem__("progress", int(st.get("progress") or 0)),
            ))

            from core.media.media_expert import MediaExpert

            def _asr_progress(info: dict):
                def _apply(st: Dict[str, Any]):
                    item = st["asr"]
                    item["percent"] = round(info.get("percent", 0), 1)
                    item["speed"] = round(info.get("speed_bytes_per_sec", 0))
                    item["eta"] = round(info.get("eta_seconds", 0))
                    item["downloaded_bytes"] = info.get("downloaded_bytes", 0)
                    item["total_bytes"] = info.get("total_bytes", 0)
                    st["progress"] = int(max(0, min(100, item["percent"])))
                _mutate_state(_apply)

            model_path = MediaExpert.ensure_ggml_model_downloaded(
                on_progress=_asr_progress,
                should_cancel=lambda: (not _is_run_active()),
            )
            if not _is_run_active():
                return
            status = MediaExpert.get_ggml_model_status()
            ready = bool(model_path) and bool(status.get("installed"))
            _mutate_state(lambda st: (
                st["asr"].update({
                    "status": "installed" if ready else "error",
                    "error": None if ready else "asr_model_incomplete_after_download",
                    "percent": 100.0 if ready else st["asr"].get("percent", 0),
                    "downloaded_bytes": status.get("downloaded_bytes", st["asr"].get("downloaded_bytes", 0)),
                    "total_bytes": status.get("total_bytes", st["asr"].get("total_bytes", 0)),
                }),
                st.__setitem__("progress", 100 if ready else int(st.get("progress") or 0)),
            ))
        except Exception as e:
            if not _is_run_active():
                return
            _mutate_state(lambda st: st["asr"].update({"status": "error", "error": str(e)}))
        finally:
            with self._asr_model_lock:
                if int(self._asr_model_state.get("run_id") or 0) != int(run_id):
                    return
                self._asr_model_state["is_downloading"] = False

    def _write_active_index_state(self, job: IndexJobState, *, kind: str,
                                  files: Optional[List[str]] = None) -> None:
        if self._index_cancel_event.is_set() or job.error == "cancelled":
            return
        try:
            import json
            data = job.to_payload()
            data["kind"] = kind
            if files:
                data["files"] = files
            with open(self.ACTIVE_INDEX_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._active_index_state_last_write_ts = time.time()
        except Exception:
            pass

    def _maybe_write_active_index_state_throttled(self, job: IndexJobState, *, kind: str,
                                                  files: Optional[List[str]] = None) -> None:
        if self._index_cancel_event.is_set() or job.error == "cancelled":
            return
        now = time.time()
        if now - self._active_index_state_last_write_ts < 2.0:
            return
        self._write_active_index_state(job, kind=kind, files=files)

    def _read_active_index_state(self) -> Optional[Dict[str, Any]]:
        try:
            import json
            if not os.path.exists(self.ACTIVE_INDEX_STATE_PATH):
                return None
            with open(self.ACTIVE_INDEX_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _clear_active_index_state(self) -> None:
        try:
            if os.path.exists(self.ACTIVE_INDEX_STATE_PATH):
                os.remove(self.ACTIVE_INDEX_STATE_PATH)
        except Exception:
            pass

    def _update_file_index_eta(self, job: "IndexJobState", files_done: int, total_files: int) -> None:
        """Estimate remaining seconds for batch file indexing."""
        total = max(0, int(total_files or 0))
        done = max(0, min(int(files_done or 0), total))
        stage_eta = self._estimate_current_file_stage_eta(job)

        if total <= 1:
            job.eta_seconds = stage_eta
            return

        remaining = max(0, total - done)
        if done <= 0 or remaining <= 0:
            job.eta_seconds = stage_eta
            return
        started = float(job.started_at or 0.0)
        if started <= 0:
            job.eta_seconds = stage_eta
            return
        elapsed = max(0.001, time.time() - started)
        avg = elapsed / done
        queue_eta = max(1, int(remaining * avg))

        if stage_eta > 0:
            remaining_after_current = max(0, total - done - 1)
            trailing_eta = max(0, int(remaining_after_current * avg)) if done > 0 else 0
            job.eta_seconds = max(stage_eta, stage_eta + trailing_eta)
            return

        job.eta_seconds = queue_eta

    def _estimate_current_file_stage_eta(self, job: "IndexJobState") -> int:
        stage = str(getattr(job, "stage", "") or "").strip().lower()
        if not stage:
            return 0

        if stage == "transcribing_audio":
            total_audio = float(getattr(job, "total_audio_sec", 0.0) or 0.0)
            current_audio = min(total_audio, float(getattr(job, "current_audio_sec", 0.0) or 0.0))
            rate = float(getattr(job, "stage_rate", 0.0) or 0.0)
            if total_audio <= 0:
                return 0
            if rate <= 0 and current_audio > 0:
                stage_started = float(getattr(job, "_stage_started_at", 0.0) or 0.0)
                if stage_started > 0:
                    elapsed = max(0.001, time.time() - stage_started)
                    rate = current_audio / elapsed
            remaining = max(0.0, total_audio - current_audio)
            if rate > 0 and remaining > 0:
                return max(1, int(remaining / rate))
            return 0

        if stage == "analyzing_frames":
            total_frames = int(getattr(job, "total_frames", 0) or 0)
            current_frame = min(total_frames, int(getattr(job, "current_frame", 0) or 0))
            rate = float(getattr(job, "stage_rate", 0.0) or 0.0)
            if total_frames <= 0:
                return 0
            if rate <= 0 and current_frame > 0:
                stage_started = float(getattr(job, "_stage_started_at", 0.0) or 0.0)
                if stage_started > 0:
                    elapsed = max(0.001, time.time() - stage_started)
                    rate = current_frame / elapsed
            remaining = max(0, total_frames - current_frame)
            if rate > 0 and remaining > 0:
                return max(1, int(remaining / rate))
            return 0

        return 0

    def _ensure_indexing_model_active(self) -> bool:
        current_id = getattr(self.llm_manager, "current_model_id", None)

        # Indexing must use the Add Sources model only. Do not fall back to chat.
        target_model_id = (self.pref_manager.get_selected_index_model_id() or "").strip()
            
        if not target_model_id:
            logger.warning("Indexing model not ready: no selected Add Sources index model")
            return False

        target_qf = None
        try:
            target_qf = self.pref_manager.get_selected_quantization_file(target_model_id)
        except Exception:
            target_qf = None
        resolved = self.llm_manager.resolve_target_model(
            target_model_id, preferred_quantization_file=target_qf
        )
        if not resolved:
            logger.warning(
                f"Indexing model not ready: target={target_model_id}, "
                f"current={current_id or 'none'}"
            )
            return False

        if current_id and current_id != target_model_id:
            self._model_before_indexing = current_id
            print(
                f"[Backend] [Indexing] 索引锁内切换模型: {current_id} -> {target_model_id}"
            )
            logger.info(
                f"[Indexing] model switch (lock): {current_id} -> {target_model_id}"
            )
        elif not current_id:
            print(
                f"[Backend] [Indexing] 索引锁内加载模型（此前无加载）: {target_model_id}"
            )
            logger.info(f"[Indexing] load model (lock, no prior): {target_model_id}")
        else:
            logger.info(
                f"[Indexing] ensure indexing-safe VL params for current model: {target_model_id}"
            )

        try:
            ok = self.llm_manager.load_model_for_indexing(
                preferred_model_id=target_model_id,
                preferred_quantization_file=target_qf,
            )
            if ok:
                print(
                    f"[Backend] [Indexing] 索引模型已加载: {getattr(self.llm_manager, 'current_model_id', None)}"
                )
            return bool(ok)
        except Exception as e:
            print(f"[Backend] Failed to prepare indexing model: {e}")
            return False

    def _restore_model_after_indexing(self) -> None:
        prev = self._model_before_indexing
        self._model_before_indexing = None
        current_id = getattr(self.llm_manager, "current_model_id", None)

        if prev and current_id != prev:
            try:
                print(
                    f"[Backend] [Indexing] 索引结束，恢复聊天模型: {current_id} -> {prev}"
                )
                logger.info(
                    f"[Indexing] restore chat model after index: {current_id} -> {prev}"
                )
                qf = None
                try:
                    qf = self.pref_manager.get_selected_quantization_file(prev)
                except Exception:
                    pass
                self.llm_manager.start_server(preferred_model_id=prev, preferred_quantization_file=qf)
            except Exception as e:
                print(f"[Backend] Failed to restore model after indexing: {e}")
        try:
            self._restore_reranker_after_indexing()
        except Exception:
            pass

    def _prepare_query_resources_after_indexing(self, kb: Any, *, reason: str) -> None:
        """Synchronously prepare retrieval resources before exposing indexing as done."""
        try:
            if hasattr(kb, "rebuild_folder_index_if_dirty"):
                kb.rebuild_folder_index_if_dirty()
        except Exception as e:
            logger.warning(f"[Backend] {reason} rebuild_folder_index_if_dirty failed: {e}")
        try:
            if hasattr(kb, "_maybe_persist"):
                kb._maybe_persist(force=True, reason=f"{reason}_job")
        except Exception:
            pass
        try:
            self._refresh_indexed_paths_cache_from_kb(kb)
        except Exception:
            pass
        try:
            if hasattr(kb, "ensure_query_resources_ready"):
                kb.ensure_query_resources_ready(reason=reason)
            elif hasattr(kb, "request_query_cache_prewarm"):
                kb.request_query_cache_prewarm(background=False, reason=reason)
        except Exception as e:
            logger.warning(f"[Backend] {reason} query resource preparation failed: {e}")

    def _run_index_job(self, job: IndexJobState) -> None:
        try:
            # Yield GIL to allow frontend API requests to complete during app startup
            import time
            time.sleep(2.0)
            
            agent = self._ensure_agent()
            from core.langgraph_agent import get_kb_instance
            kb = get_kb_instance()
            kb.enter_write_heavy_mode(reason="backend_folder_index_job")
            job.started_at = time.time()
            try:
                self._index_cancel_event.clear()
            except Exception:
                pass

            def _reset_media_stage(_job: IndexJobState, *, stage_started_at: Optional[float] = None) -> None:
                _job.current_frame = 0
                _job.total_frames = 0
                _job.current_audio_sec = 0.0
                _job.total_audio_sec = 0.0
                _job.stage_rate = 0.0
                _job.stage = ""
                setattr(_job, "_stage_started_at", stage_started_at or time.time())

            def cb(current: int, total: int, file_name: str, file_path: str = "") -> None:
                now = time.time()
                initial_total = getattr(job, "initial_total_files", 0)
                initial_completed = getattr(job, "initial_completed_files", 0)
                
                current_total = int(total or 0)
                skipped_n = getattr(job, 'skipped_count', 0)
                processed_before_current = max(0, int(current or 0) - 1)
                job.total_files = max(0, max(initial_total, current_total + initial_completed) - skipped_n)
                job.completed_files = min(job.total_files, initial_completed + processed_before_current)

                path_changed = (file_path or "") != (job.current_path or "")
                job.current_file = file_name or ""
                job.current_path = file_path or ""
                if path_changed:
                    _reset_media_stage(job, stage_started_at=now)
                self._update_file_index_eta(job, processed_before_current, job.total_files)
                
                # Check for cancellation before writing state to avoid overwriting after cancel
                if not self._index_cancel_event.is_set():
                    self._maybe_write_active_index_state_throttled(job, kind="folder")

            def on_indexed(file_path: str) -> None:
                """Authoritative callback from scan_directory after ingest_file succeeds."""
                if not any(_source_path_key(p) == _source_path_key(file_path) for p in job.indexed_paths):
                    job.indexed_paths.append(file_path)
                self._merge_indexed_paths_cache([file_path])

            def _on_frame_progress(cur: int, total: int, _job=job) -> None:
                if _job.stage != "analyzing_frames":
                    setattr(_job, "_stage_started_at", time.time())
                _job.current_frame = cur
                _job.total_frames = total
                _job.current_audio_sec = 0.0
                _job.total_audio_sec = 0.0
                _job.stage_rate = (
                    cur / max(0.001, time.time() - float(getattr(_job, "_stage_started_at", time.time()) or time.time()))
                    if cur > 0 else 0.0
                )
                _job.stage = "analyzing_frames"
                done = max(0, _job.completed_files - int(getattr(_job, "initial_completed_files", 0) or 0))
                self._update_file_index_eta(_job, done, _job.total_files)
                if not self._index_cancel_event.is_set():
                    self._maybe_write_active_index_state_throttled(_job, kind="folder")

            def _on_media_progress(progress: Dict[str, Any], _job=job) -> None:
                stage = str(progress.get("stage") or "").strip().lower()
                if stage and _job.stage != stage:
                    setattr(_job, "_stage_started_at", time.time())
                if stage:
                    _job.stage = stage
                if stage == "transcribing_audio":
                    _job.current_audio_sec = float(progress.get("current_audio_sec") or 0.0)
                    _job.total_audio_sec = float(progress.get("total_audio_sec") or 0.0)
                    _job.current_frame = 0
                    _job.total_frames = 0
                elif stage == "analyzing_frames":
                    _job.current_frame = int(progress.get("current_frame") or 0)
                    _job.total_frames = int(progress.get("total_frames") or 0)
                    _job.current_audio_sec = 0.0
                    _job.total_audio_sec = 0.0
                rate = float(progress.get("stage_rate") or 0.0)
                if rate <= 0.0 and stage == "transcribing_audio" and _job.current_audio_sec > 0:
                    elapsed_stage = max(0.001, time.time() - float(getattr(_job, "_stage_started_at", time.time()) or time.time()))
                    rate = _job.current_audio_sec / elapsed_stage
                elif rate <= 0.0 and stage == "analyzing_frames" and _job.current_frame > 0:
                    elapsed_stage = max(0.001, time.time() - float(getattr(_job, "_stage_started_at", time.time()) or time.time()))
                    rate = _job.current_frame / elapsed_stage
                _job.stage_rate = rate
                done = max(0, _job.completed_files - int(getattr(_job, "initial_completed_files", 0) or 0))
                self._update_file_index_eta(_job, done, _job.total_files)
                if not self._index_cancel_event.is_set():
                    self._maybe_write_active_index_state_throttled(_job, kind="folder")

            with self._indexing_lock:
                self._suspend_reranker_for_indexing(kb)
                if not self._ensure_embedding_runtime_active(kb):
                    raise RuntimeError("embedding runtime unavailable")
                self._refresh_indexed_paths_cache_from_kb(kb)
                if not self._ensure_indexing_model_active():
                    raise RuntimeError("indexing model unavailable")
                kb.scan_directory(
                    directories=[job.folder],
                    progress_callback=cb,
                    should_cancel=lambda: bool(self._index_cancel_event.is_set()),
                    should_skip_file=lambda fp: _source_path_key(fp) in job.skipped_files,
                    on_file_indexed=on_indexed,
                    on_frame_progress=_on_frame_progress,
                    on_media_progress=_on_media_progress,
                )
            if job.skipped_files:
                before = len(job.indexed_paths)
                rollback_paths = [p for p in job.indexed_paths if _source_path_key(p) in job.skipped_files]
                job.indexed_paths = [p for p in job.indexed_paths if _source_path_key(p) not in job.skipped_files]
                for rp in rollback_paths:
                    try:
                        kb.delete_file(rp)
                        logger.info(f"[Backend] ⏭️ 回滚文件夹索引（用户取消）: {rp}")
                    except Exception:
                        pass
                if rollback_paths:
                    self._remove_indexed_paths_cache(rollback_paths)
                if rollback_paths:
                    logger.info(f"[Backend] 文件夹索引回滚 {len(rollback_paths)} 个文件 ({before} -> {len(job.indexed_paths)})")
            job.eta_seconds = 0
            job.current_frame = 0
            job.total_frames = 0
            job.current_audio_sec = 0.0
            job.total_audio_sec = 0.0
            job.stage_rate = 0.0
            initial_completed = int(getattr(job, "initial_completed_files", 0) or 0)
            indexed_completed = initial_completed + len(job.indexed_paths)
            job.completed_files = min(job.total_files, max(job.completed_files, indexed_completed))
            job.message = "文件夹扫描完成"
            job.error = "cancelled" if self._index_cancel_event.is_set() else None
            if job.error == "cancelled":
                job.is_indexing = False
                job.finished_at = time.time()
                job.stage = ""
            else:
                job.stage = "preparing_search_resources"
                job.message = "正在准备检索资源"
        except Exception as e:
            job.is_indexing = False
            job.finished_at = time.time()
            job.current_frame = 0
            job.total_frames = 0
            job.current_audio_sec = 0.0
            job.total_audio_sec = 0.0
            job.stage_rate = 0.0
            job.stage = ""
            job.error = str(e)
        finally:
            try:
                kb.leave_write_heavy_mode(reason="backend_folder_index_job")
            except Exception:
                pass
            try:
                if job.error is None:
                    self._prepare_query_resources_after_indexing(kb, reason="post_folder_index")
                else:
                    try:
                        self._refresh_indexed_paths_cache_from_kb(kb)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[Backend] post-folder-index query resource gate failed: {e}")
            try:
                self._restore_model_after_indexing()
            except Exception:
                pass
            if job.error is None and job.is_indexing:
                job.is_indexing = False
                job.finished_at = time.time()
                job.stage = ""
                job.message = "文件夹扫描完成"
            with self._jobs_lock:
                if self._active_index_job_id == job.job_id:
                    self._active_index_job_id = None
            self._invalidate_file_id_map()
            try:
                self._write_active_index_state(job, kind="folder")
            except Exception:
                pass
            if (job.error is None) or (str(job.error).strip().lower() == "cancelled"):
                self._clear_active_index_state()
            if job.error != "cancelled" and self._index_append_queue:
                try:
                    self._process_append_queue()
                    return
                except Exception as e:
                    logger.warning(f"[Backend] 处理追加队列失败: {e}")

    def _run_file_index_job(self, job: IndexJobState, file_paths: List[str]) -> None:
        try:
            # Yield GIL to allow frontend API requests to complete during app startup
            import time
            time.sleep(2.0)
            
            agent = self._ensure_agent()
            from core.langgraph_agent import get_kb_instance
            kb = get_kb_instance()
            kb.enter_write_heavy_mode(reason="backend_files_index_job")
            
            success_files = []
            failed_files = []
            file_results = []

            def _sync_failed_files_count(*, include_unprocessed: bool = False) -> None:
                known_failed = len(failed_files)
                remaining = 0
                if include_unprocessed:
                    remaining = max(0, int(job.total_files or 0) - int(job.completed_files or 0))
                job.failed_files = max(0, known_failed + remaining)
            
            try:
                self._index_cancel_event.clear()
            except Exception:
                pass
            with self._indexing_lock:
                self._suspend_reranker_for_indexing(kb)
                if not self._ensure_embedding_runtime_active(kb):
                    raise RuntimeError("embedding runtime unavailable")
                self._refresh_indexed_paths_cache_from_kb(kb)
                if not self._ensure_indexing_model_active():
                    raise RuntimeError("indexing model unavailable")
                job.started_at = time.time()
                actual_done = 0
                for idx, fp in enumerate(file_paths, 1):
                    if self._index_cancel_event.is_set():
                        break
                    fp_key = _source_path_key(fp)
                    if fp_key in job.skipped_files:
                        logger.info(f"[Backend] ⏭️ 用户跳过: {fp}")
                        continue
                    file_started_at = time.time()
                    initial_completed = getattr(job, "initial_completed_files", 0)
                    if not os.path.exists(fp):
                        reason = "文件不存在"
                        failed_files.append((fp, reason))
                        file_results.append((fp, "failed", 0.0, reason))
                        print(f"[Backend] ❌ 建立索引失败: {fp} | 耗时: 0.00秒 | 原因: {reason}")
                        actual_done += 1
                        job.completed_files = initial_completed + actual_done
                        _sync_failed_files_count()
                        job.current_file = os.path.basename(fp)
                        job.current_path = fp
                        self._update_file_index_eta(job, actual_done, job.total_files)
                        if not self._index_cancel_event.is_set():
                            self._maybe_write_active_index_state_throttled(job, kind="files", files=file_paths)
                        continue
                    job.current_file = os.path.basename(fp)
                    job.current_path = fp
                    job.current_frame = 0
                    job.total_frames = 0
                    job.current_audio_sec = 0.0
                    job.total_audio_sec = 0.0
                    job.stage_rate = 0.0
                    job.stage = ""
                    setattr(job, "_stage_started_at", file_started_at)
                    
                    job.completed_files = initial_completed + actual_done
                    self._update_file_index_eta(job, actual_done, job.total_files)

                    def _on_frame_progress(cur: int, total: int, _job=job) -> None:
                        if _job.stage != "analyzing_frames":
                            setattr(_job, "_stage_started_at", time.time())
                        _job.current_frame = cur
                        _job.total_frames = total
                        _job.current_audio_sec = 0.0
                        _job.total_audio_sec = 0.0
                        _job.stage_rate = (
                            cur / max(0.001, time.time() - float(getattr(_job, "_stage_started_at", time.time()) or time.time()))
                            if cur > 0 else 0.0
                        )
                        _job.stage = "analyzing_frames"
                        self._update_file_index_eta(_job, actual_done, _job.total_files)

                    def _on_media_progress(progress: Dict[str, Any], _job=job) -> None:
                        stage = str(progress.get("stage") or "").strip().lower()
                        if stage and _job.stage != stage:
                            setattr(_job, "_stage_started_at", time.time())
                        if stage:
                            _job.stage = stage
                        if stage == "transcribing_audio":
                            _job.current_audio_sec = float(progress.get("current_audio_sec") or 0.0)
                            _job.total_audio_sec = float(progress.get("total_audio_sec") or 0.0)
                            _job.current_frame = 0
                            _job.total_frames = 0
                        elif stage == "analyzing_frames":
                            _job.current_frame = int(progress.get("current_frame") or 0)
                            _job.total_frames = int(progress.get("total_frames") or 0)
                            _job.current_audio_sec = 0.0
                            _job.total_audio_sec = 0.0
                        rate = float(progress.get("stage_rate") or 0.0)
                        if rate <= 0.0 and stage == "transcribing_audio" and _job.current_audio_sec > 0:
                            elapsed_stage = max(0.001, time.time() - float(getattr(_job, "_stage_started_at", time.time()) or time.time()))
                            rate = _job.current_audio_sec / elapsed_stage
                        elif rate <= 0.0 and stage == "analyzing_frames" and _job.current_frame > 0:
                            elapsed_stage = max(0.001, time.time() - float(getattr(_job, "_stage_started_at", time.time()) or time.time()))
                            rate = _job.current_frame / elapsed_stage
                        _job.stage_rate = rate
                        self._update_file_index_eta(_job, actual_done, _job.total_files)
                        if not self._index_cancel_event.is_set():
                            self._maybe_write_active_index_state_throttled(_job, kind="files", files=file_paths)

                    try:
                        print(f"[Backend] 正在索引文件: {fp}")
                        success = kb.index_file(
                            fp,
                            on_frame_progress=_on_frame_progress,
                            on_media_progress=_on_media_progress,
                        )
                        elapsed = time.time() - file_started_at
                        if success:
                            if fp_key in job.skipped_files:
                                logger.info(f"[Backend] ⏭️ 文件 {fp} 在索引期间被取消，回滚索引")
                                try:
                                    kb.delete_file(fp)
                                except Exception as del_err:
                                    logger.warning(f"[Backend] 回滚索引失败: {del_err}")
                                self._remove_indexed_paths_cache([fp])
                            else:
                                success_files.append(fp)
                                self.sources.add_file(fp)
                                if not any(_source_path_key(p) == fp_key for p in job.indexed_paths):
                                    job.indexed_paths.append(fp)
                                self._merge_indexed_paths_cache([fp])
                                file_results.append((fp, "success", elapsed, ""))
                                msg = f"[Backend] ✅ 成功建立索引: {fp} | 耗时: {elapsed:.2f}秒"
                                print(msg)
                                logger.info(msg)
                        else:
                            reason = "被忽略或无法读取"
                            failed_files.append((fp, reason))
                            file_results.append((fp, "failed", elapsed, reason))
                            msg = f"[Backend] ❌ 建立索引失败: {fp} | 耗时: {elapsed:.2f}秒 | 原因: {reason}"
                            print(msg)
                            logger.warning(msg)
                    except Exception as e:
                        elapsed = time.time() - file_started_at
                        reason = str(e)
                        print(f"[Backend] Failed to index file {fp}: {reason} | elapsed={elapsed:.2f}s")
                        failed_files.append((fp, reason))
                        file_results.append((fp, "failed", elapsed, reason))
                    actual_done += 1
                    job.completed_files = initial_completed + actual_done
                    _sync_failed_files_count()
                    self._update_file_index_eta(job, actual_done, job.total_files)
                    if not self._index_cancel_event.is_set():
                        self._maybe_write_active_index_state_throttled(job, kind="files", files=file_paths)
            job.eta_seconds = 0
            _sync_failed_files_count()
            
            from datetime import datetime
            log_msg = (
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"单文件批量索引任务完成 - 成功建立索引: {len(success_files)}, 失败: {len(failed_files)}\n"
            )
            if file_results:
                log_msg += "每个文档耗时:\n"
                for fp, status, elapsed, reason in file_results:
                    if status == "success":
                        log_msg += f"  - ✅ {fp} | {elapsed:.2f}秒\n"
                    else:
                        reason_text = reason or "未知原因"
                        log_msg += f"  - ❌ {fp} | {elapsed:.2f}秒 | 原因: {reason_text}\n"
            if failed_files:
                log_msg += "失败的文件列表:\n"
                for fp, reason in failed_files:
                    log_msg += f"  - {fp}: {reason}\n"
            logger.info(log_msg)
            
            try:
                base_dir = os.environ.get("FILEAGENT_DATA_DIR", "")
                if base_dir:
                    base_dir = os.path.abspath(os.path.expanduser(base_dir))
                else:
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
                
                logs_dir = os.path.join(base_dir, "logs")
                os.makedirs(logs_dir, exist_ok=True)
                with open(os.path.join(logs_dir, "index_details.log"), "a", encoding="utf-8") as f:
                    f.write(log_msg + "\n")
                    f.flush()
            except Exception as e:
                print(f"[Backend] 写入日志文件失败: {e}")
            
            initial_completed = int(getattr(job, "initial_completed_files", 0) or 0)
            indexed_completed = initial_completed + len(job.indexed_paths)
            job.completed_files = min(job.total_files, max(job.completed_files, indexed_completed))
            job.message = f"索引完成: 成功建立索引 {len(job.indexed_paths)} 个, 失败 {len(failed_files)} 个"
            
            if self._index_cancel_event.is_set():
                job.error = "cancelled"
                job.message = "索引已取消"
                job.is_indexing = False
                job.finished_at = time.time()
            else:
                job.stage = "preparing_search_resources"
                job.message = "正在准备检索资源"
        except Exception as e:
            job.is_indexing = False
            job.finished_at = time.time()
            _sync_failed_files_count(include_unprocessed=True)
            initial_completed = int(getattr(job, "initial_completed_files", 0) or 0)
            indexed_completed = initial_completed + len(job.indexed_paths)
            job.completed_files = min(job.total_files, max(job.completed_files, indexed_completed))
            job.error = str(e)
            job.message = f"索引失败: {e}"
            logger.exception(f"[Backend] file index job failed: job_id={job.job_id} error={e}")
        finally:
            try:
                kb.leave_write_heavy_mode(reason="backend_files_index_job")
            except Exception:
                pass
            try:
                if job.error is None:
                    self._prepare_query_resources_after_indexing(kb, reason="post_files_index")
                else:
                    try:
                        self._refresh_indexed_paths_cache_from_kb(kb)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[Backend] post-files-index query resource gate failed: {e}")
            try:
                self._restore_model_after_indexing()
            except Exception:
                pass
            if job.error is None and job.is_indexing:
                job.is_indexing = False
                job.finished_at = time.time()
                job.stage = ""
                job.message = f"索引完成: 成功建立索引 {len(job.indexed_paths)} 个, 失败 {int(job.failed_files or 0)} 个"
            with self._jobs_lock:
                if self._active_index_job_id == job.job_id:
                    self._active_index_job_id = None
            self._invalidate_file_id_map()
            if (job.error is None) or (str(job.error).strip().lower() == "cancelled"):
                self._clear_active_index_state()
            if job.error != "cancelled" and self._index_append_queue:
                try:
                    self._process_append_queue()
                    return
                except Exception as e:
                    logger.warning(f"[Backend] 处理追加队列失败: {e}")

    def _run_refresh_job(self, job: IndexJobState) -> None:
        try:
            import time
            time.sleep(1.0)
            
            logger.info(f"[Backend] refresh job started: job_id={job.job_id} folder={job.folder}")
            agent = self._ensure_agent()
            from core.langgraph_agent import get_kb_instance
            kb = get_kb_instance()
            job.started_at = time.time()
            job.message = "scanning"
            kb.enter_write_heavy_mode(reason="backend_refresh_index_job")
            
            try:
                self._index_cancel_event.clear()
            except Exception:
                pass

            def cb(current: int, total: int, file_name: str, file_path: str = "") -> None:
                job.total_files = int(total or 0)
                job.completed_files = int(current or 0)
                job.current_file = file_name or ""
                job.current_path = file_path or ""
                elapsed = max(0.001, time.time() - job.started_at)
                speed = job.completed_files / elapsed
                remaining = max(0, job.total_files - job.completed_files)
                job.eta_seconds = int(remaining / speed) if speed > 0 else 0
                
                if not self._index_cancel_event.is_set():
                    self._maybe_write_active_index_state_throttled(job, kind="folder")

            with self._indexing_lock:
                self._suspend_reranker_for_indexing(kb)
                if not self._ensure_embedding_runtime_active(kb):
                    raise RuntimeError("embedding runtime unavailable")
                if not self._ensure_indexing_model_active():
                    raise RuntimeError("indexing model unavailable")
                dirs_to_refresh = getattr(job, '_refresh_dirs', [job.folder])
                refresh_result = kb.refresh_source(
                    directories=dirs_to_refresh,
                    progress_callback=cb,
                    should_cancel=lambda: bool(self._index_cancel_event.is_set()),
                )

            job.eta_seconds = 0
            job.error = "cancelled" if self._index_cancel_event.is_set() else None
            # Store results in format expected by parsing logic handling refresh polling
            job.message = (
                f"refresh_done|added={refresh_result.get('added',0)}"
                f"|updated={refresh_result.get('updated',0)}"
                f"|deleted={refresh_result.get('deleted',0)}"
                f"|skipped={refresh_result.get('skipped',0)}"
                f"|errors={refresh_result.get('errors',0)}"
            )
            if job.error == "cancelled":
                job.is_indexing = False
                job.finished_at = time.time()
            else:
                job.stage = "preparing_search_resources"
            logger.info(f"[Backend] refresh job finished: {job.message}")
        except Exception as e:
            job.is_indexing = False
            job.finished_at = time.time()
            job.error = str(e)
            logger.error(f"[Backend] refresh job failed: job_id={job.job_id} error={e}")
        finally:
            try:
                kb.leave_write_heavy_mode(reason="backend_refresh_index_job")
            except Exception:
                pass
            try:
                if job.error is None:
                    self._prepare_query_resources_after_indexing(kb, reason="post_refresh_index")
                else:
                    try:
                        self._refresh_indexed_paths_cache_from_kb(kb)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[Backend] post-refresh-index query resource gate failed: {e}")
            try:
                self._restore_model_after_indexing()
            except Exception:
                pass
            if job.error is None and job.is_indexing:
                job.is_indexing = False
                job.finished_at = time.time()
                job.stage = ""
            with self._jobs_lock:
                if self._active_index_job_id == job.job_id:
                    self._active_index_job_id = None
            self._invalidate_file_id_map()
            try:
                self._write_active_index_state(job, kind="folder")
            except Exception:
                pass
            if (job.error is None) or (str(job.error).strip().lower() == "cancelled"):
                self._clear_active_index_state()


    def _warmup_models(self):
        raw_disable = (os.getenv("FILEAGENT_DISABLE_WARMUP") or "").strip().lower()
        if raw_disable in {"1", "true", "yes", "y", "on"}:
            print("[Backend] Warmup disabled.")
            self._update_startup_index_prefill_status(state="skipped", reason="warmup_disabled")
            return
        if settings.DEV_NO_MODEL_LOAD:
            print("[Backend] Dev mode: skipping model loading.")
            self._update_startup_index_prefill_status(state="skipped", reason="dev_no_model_load")
            return
        try:
            raw_onboarding = self.pref_manager.get("onboarding_complete", None)
            onboarding_complete = str(raw_onboarding).strip().lower() in {"1", "true", "yes", "y", "on"}
            raw_step = self.pref_manager.get("onboarding_step", None)
            onboarding_step = str(raw_step or "").strip().lower()
        except Exception:
            onboarding_complete = False
            onboarding_step = ""
        if (not onboarding_complete) or (onboarding_step and onboarding_step != "complete"):
            print("[Backend] Warmup skipped: onboarding not completed step.")
            self._update_startup_index_prefill_status(state="skipped", reason="onboarding_not_complete")
            return

        need_download = False
        run_id = 0
        with self._core_models_lock:
            if not self._core_models_state.get("is_downloading"):
                emb_ok = _model_dir_installed(settings.LOCAL_EMBEDDING_MODEL_PATH)
                rr_ok = _model_dir_installed(settings.LOCAL_RERANKER_MODEL_PATH) or getattr(settings, "RERANKER_OPTIONAL", False)
                if not (emb_ok and rr_ok):
                    need_download = True
                    run_id = int(self._core_models_state.get("run_id") or 0) + 1
                    self._core_models_state["run_id"] = run_id
                    self._core_models_state["is_downloading"] = True

        if need_download:
            print("[Backend] Warmup: core models missing, starting download with progress tracking...")
            self._download_core_models_worker(run_id)
        else:
            print("[Backend] Warmup: core models already installed or download in progress.")

        selected_id = self.pref_manager.get_selected_model_id()
        try:
            qf = None
            try:
                qf = self.pref_manager.get_selected_quantization_file(selected_id)
            except Exception:
                pass
            self.llm_manager.start_server(preferred_model_id=selected_id, preferred_quantization_file=qf)
            if getattr(self.llm_manager, "current_model_id", None):
                print(f"[Backend] LLM running: {self.llm_manager.current_model_id}")
                try:
                    self._maybe_run_index_prefill_warmup()
                except Exception as e:
                    logger.warning(f"[Backend] index prefill warmup failed: {e}")
        except Exception as e:
            self._update_startup_index_prefill_status(state="failed", reason=f"chat_warmup_failed:{e}")
            print(f"[Backend] LLM startup failed: {e}")

    def _update_startup_index_prefill_status(
        self,
        *,
        state: str,
        reason: str = "",
        target: Optional[Dict[str, str]] = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        elapsed_ms: Optional[int] = None,
    ) -> None:
        try:
            updater = getattr(self.llm_manager, "update_startup_index_prefill_status", None)
            if not callable(updater):
                return
            updater(
                state=state,
                reason=reason,
                target_model_id=(target or {}).get("model_id", ""),
                target_model_path=(target or {}).get("model_path", ""),
                started_at=started_at,
                completed_at=completed_at,
                elapsed_ms=elapsed_ms,
            )
        except Exception:
            pass

    def _resolve_selected_model_target_signature(self, model_id: str) -> Optional[Dict[str, str]]:
        mid = str(model_id or "").strip()
        if not mid:
            return None
        qf = None
        try:
            qf = self.pref_manager.get_selected_quantization_file(mid)
        except Exception:
            qf = None
        resolved = self.llm_manager.resolve_target_model(
            mid,
            preferred_quantization_file=qf,
        )
        if not resolved:
            return None
        cfg, model_path, mmproj_path = resolved
        return {
            "model_id": str(cfg.get("id") or mid).strip(),
            "model_path": str(model_path or ""),
            "mmproj_path": str(mmproj_path or ""),
        }

    def _should_run_index_prefill_warmup(self) -> tuple[bool, str, Optional[Dict[str, str]]]:
        selected_chat_id = str(self.pref_manager.get_selected_model_id() or "").strip()
        selected_index_id = str(self.pref_manager.get_selected_index_model_id() or "").strip()
        if not selected_chat_id or not selected_index_id:
            return False, "missing chat/index model selection", None

        chat_sig = self._resolve_selected_model_target_signature(selected_chat_id)
        index_sig = self._resolve_selected_model_target_signature(selected_index_id)
        if not chat_sig or not index_sig:
            return False, "unable to resolve chat/index model target", None

        if (
            chat_sig["model_id"] != index_sig["model_id"]
            or chat_sig["model_path"] != index_sig["model_path"]
            or chat_sig["mmproj_path"] != index_sig["mmproj_path"]
        ):
            return (
                False,
                f"chat/index target mismatch: chat={chat_sig['model_id']}::{os.path.basename(chat_sig['model_path'])} "
                f"index={index_sig['model_id']}::{os.path.basename(index_sig['model_path'])}",
                None,
            )

        current_sig = {
            "model_id": str(getattr(self.llm_manager, "current_model_id", None) or "").strip(),
            "model_path": str(getattr(self.llm_manager, "current_model_path", None) or ""),
            "mmproj_path": str(getattr(self.llm_manager, "current_mmproj_path", None) or ""),
        }
        if getattr(self.llm_manager, "_llama", None) is None:
            return False, "llm not loaded after chat warmup", None
        if current_sig != chat_sig:
            return (
                False,
                f"current loaded target differs: current={current_sig['model_id']}::{os.path.basename(current_sig['model_path'])} "
                f"expected={chat_sig['model_id']}::{os.path.basename(chat_sig['model_path'])}",
                None,
            )
        return True, "matched current chat/index target", chat_sig

    def _build_index_prefill_warmup_prompt(self) -> str:
        return (
            "Return compact JSON only with keys family, leaf_category, role, confidence, summary, "
            "file_name_en, extracted_key_info.\n"
            "file_name: index_prefill_warmup.md\n"
            "file_ext: .md\n"
            "page_count:\n"
            "content:\n"
            "This internal note discusses indexing speed, first-file cold start, GPU inference warmup, "
            "embedding throughput, reranker stability, multilingual retrieval, and robust parsing for "
            "PDF, DOCX, PPTX, spreadsheet, and markdown files. The goal is to preserve retrieval quality "
            "while reducing latency spikes during startup and the first indexing task."
        )

    def _maybe_run_index_prefill_warmup(self) -> None:
        raw_disable = (os.getenv("FILEAGENT_DISABLE_INDEX_PREFILL_WARMUP") or "").strip().lower()
        if raw_disable in {"1", "true", "yes", "y", "on"}:
            self._update_startup_index_prefill_status(state="skipped", reason="index_prefill_disabled")
            logger.info("[Backend] skip index prefill warmup: disabled by env")
            return
        if self._indexing_in_progress():
            self._update_startup_index_prefill_status(state="skipped", reason="indexing_already_started")
            logger.info("[Backend] skip index prefill warmup: indexing already started")
            return

        should_run, reason, target = self._should_run_index_prefill_warmup()
        if not should_run:
            self._update_startup_index_prefill_status(state="skipped", reason=reason, target=target)
            logger.info(f"[Backend] skip index prefill warmup: {reason}")
            return

        try:
            delay_sec = max(
                0.0,
                float(os.getenv("FILEAGENT_INDEX_PREFILL_WARMUP_DELAY_SEC", "0.0") or 0.0),
            )
        except Exception:
            delay_sec = 0.0
        if delay_sec > 0:
            time.sleep(delay_sec)
            if self._indexing_in_progress():
                self._update_startup_index_prefill_status(state="skipped", reason="indexing_started_during_prefill_delay", target=target)
                logger.info("[Backend] skip index prefill warmup after delay: indexing already started")
                return
            should_run, reason, target = self._should_run_index_prefill_warmup()
            if not should_run:
                self._update_startup_index_prefill_status(state="skipped", reason=reason, target=target)
                logger.info(f"[Backend] skip index prefill warmup after delay: {reason}")
                return

        try:
            max_tokens = max(
                16,
                int(os.getenv("FILEAGENT_INDEX_PREFILL_WARMUP_MAX_TOKENS", "64") or 64),
            )
        except Exception:
            max_tokens = 64

        started = time.time()
        self._update_startup_index_prefill_status(
            state="running",
            reason="startup_index_prefill_running",
            target=target,
            started_at=started,
            completed_at=0.0,
            elapsed_ms=0,
        )
        logger.info(
            f"[Backend] index prefill warmup start: model={target['model_id']} "
            f"target={os.path.basename(target['model_path'])} max_tokens={max_tokens}"
        )
        try:
            self.llm_manager.create_chat_completion(
                model_id=None,
                preferred_quantization_file=None,
                messages=[{"role": "user", "content": self._build_index_prefill_warmup_prompt()}],
                max_tokens=max_tokens,
                temperature=0.0,
                stream=False,
                needs_vision=False,
            )
            elapsed_ms = int((time.time() - started) * 1000)
            self._update_startup_index_prefill_status(
                state="done",
                reason="startup_index_prefill_done",
                target=target,
                started_at=started,
                completed_at=time.time(),
                elapsed_ms=elapsed_ms,
            )
            logger.info(
                f"[Backend] index prefill warmup done: model={target['model_id']} elapsed_ms={elapsed_ms}"
            )
        except Exception as e:
            self._update_startup_index_prefill_status(
                state="failed",
                reason=f"index_prefill_request_failed:{e}",
                target=target,
                started_at=started,
                completed_at=time.time(),
                elapsed_ms=int((time.time() - started) * 1000),
            )
            logger.warning(f"[Backend] index prefill warmup request failed: {e}")
