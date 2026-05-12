
from __future__ import annotations

import os
import time
import uuid
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# - chroma_db（DB_PATH）
# - indexed_folders.json / chat_history.json / user_preferences.json
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

# - indexed_folders.json
# - chroma_db/

def _maybe_migrate_legacy_state():
    try:
        data_dir = (_DATA_DIR or "").strip()
        if not data_dir:
            return
        data_dir = os.path.abspath(os.path.expanduser(data_dir))
        if os.path.abspath(data_dir) == os.path.abspath(BASE_DIR):
            return

        legacy_indexed = os.path.join(BASE_DIR, "indexed_folders.json")
        legacy_db = os.path.join(BASE_DIR, "chroma_db")

        new_indexed = os.path.join(data_dir, "indexed_folders.json")
        new_db = os.path.join(data_dir, "chroma_db")

        if os.path.exists(legacy_indexed):
            import json

            try:
                with open(legacy_indexed, "r", encoding="utf-8") as f:
                    old_list = json.load(f)
                if not isinstance(old_list, list):
                    old_list = []
            except Exception:
                old_list = []

            if os.path.exists(new_indexed):
                try:
                    with open(new_indexed, "r", encoding="utf-8") as f:
                        new_list = json.load(f)
                    if not isinstance(new_list, list):
                        new_list = []
                except Exception:
                    new_list = []
            else:
                new_list = []

            merged = []
            seen = set()
            for x in (new_list or []) + (old_list or []):
                try:
                    p = os.path.abspath(os.path.expanduser(str(x)))
                except Exception:
                    p = str(x)
                if p and p not in seen:
                    seen.add(p)
                    merged.append(p)

            if merged and (not os.path.exists(new_indexed) or merged != new_list):
                try:
                    os.makedirs(data_dir, exist_ok=True)
                    with open(new_indexed, "w", encoding="utf-8") as f:
                        json.dump(merged, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

        if os.path.exists(legacy_db) and not os.path.exists(new_db):
            try:
                os.makedirs(data_dir, exist_ok=True)
                os.rename(legacy_db, new_db)
            except Exception:
                pass
    except Exception:
        pass


if _DATA_DIR:
    try:
        threading.Thread(target=_maybe_migrate_legacy_state, daemon=True).start()
    except Exception:
        pass

from utils.logger import configure_logging, get_logger  # noqa: E402

configure_logging()
logger = get_logger()

from typing import TYPE_CHECKING  # noqa: E402
if TYPE_CHECKING:  # pragma: no cover
    from core.langgraph_agent import FileAgent  # type: ignore

MODEL_MANAGER = None
PREF_MANAGER = None
LLM_MANAGER = None
SOURCES = None

def _get_managers():
    """Lazily initialize all core managers."""
    global MODEL_MANAGER, PREF_MANAGER, LLM_MANAGER, SOURCES
    if MODEL_MANAGER is None:
        from services.model_manager import ModelManager
        from services.local_llm import LocalLLMManager
        from services.preference_manager import PreferenceManager
        from tools.document_tools import DocumentSources
        from config import settings
        
        PREF_MANAGER = PreferenceManager(settings.HOME_DIR)
        MODEL_MANAGER = ModelManager(settings.MODELS_DIR)
        LLM_MANAGER = LocalLLMManager()
        SOURCES = DocumentSources(os.path.join(settings.HOME_DIR, "indexed_folders.json"))
        
        def _init_models_bg():
            try:
                selected_model = PREF_MANAGER.get_selected_model_id()
                if selected_model:
                    qf = PREF_MANAGER.get_selected_quantization_file(selected_model)
                    LLM_MANAGER.start_server(selected_model, preferred_quantization_file=qf)
            except Exception as e:
                logger.error(f"[Server] Failed to auto-start model: {e}")
        
        threading.Thread(target=_init_models_bg, daemon=True).start()
        
    return MODEL_MANAGER, PREF_MANAGER, LLM_MANAGER, SOURCES

def _ensure_agent():
    global ACTIVE_AGENT
    if ACTIVE_AGENT:
        return ACTIVE_AGENT
    
    _, _, llm_mgr, _ = _get_managers()
    
    from core.langgraph_agent import FileAgent
    logger.info("[Backend] 初始化 FileAgent (工具调用版)...")
    ACTIVE_AGENT = FileAgent(llm_manager=llm_mgr)
    return ACTIVE_AGENT


from utils.file_explorer import (
    _stable_id, _icon_type_for_path, _is_indexable_file_for_sources,
    _build_file_node, _source_path_key, _is_folder_fully_indexed,
    _collect_indexable_file_paths, _folder_has_relevant_indexable_file,
    _build_folder_node
)
from services.storage.source_store import SourceStore
from services.storage.history_manager import HistoryManager
from services.indexing.index_job import IndexJobState
from tools.document_tools import set_active_paths, get_active_paths, set_active_session_id


# -------- FastAPI app --------

def _cors_config() -> Dict[str, Any]:
    raw_extra = os.getenv("FILEAGENT_CORS_ALLOW_ORIGINS", "")
    extra = [item.strip() for item in raw_extra.split(",") if item.strip()]
    unsafe = (os.getenv("FILEAGENT_UNSAFE_CORS", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if unsafe:
        return {
            "allow_origins": ["*"],
            "allow_credentials": False,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    return {
        "allow_origins": [
            "tauri://localhost",
            "http://tauri.localhost",
            "https://tauri.localhost",
            *extra,
        ],
        "allow_origin_regex": r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }


app = FastAPI(title="FileAgent Backend", version="0.1.0")
app.add_middleware(CORSMiddleware, **_cors_config())


class QueryRequest(BaseModel):
    message: str = Field(..., min_length=1)
    active_source_ids: Optional[List[str]] = None
    model_id: Optional[str] = None
    session_id: Optional[str] = None
    language: Optional[str] = None


def _apply_runtime_model_hint(model_id: Optional[str]) -> None:
    """Apply per-request model hint from frontend to backend preference/cache."""
    mid = str(model_id or "").strip()
    if not mid:
        return
    # Protect indexing flow: do not override preference while indexing
    # or while waiting to restore pre-index model.
    try:
        if _indexing_in_progress() is not None:
            return
    except Exception:
        pass
    if _MODEL_BEFORE_INDEXING:
        return
    try:
        current = PREF_MANAGER.get_selected_model_id()
    except Exception:
        current = None
    if current == mid:
        return
    try:
        PREF_MANAGER.set_selected_model_id(mid)
    except Exception:
        return
    try:
        from services.inproc_openai_client import clear_inproc_openai_client
        clear_inproc_openai_client()
    except Exception:
        pass
    global AGENT
    if AGENT is not None:
        try:
            if hasattr(AGENT, "_llm_service"):
                delattr(AGENT, "_llm_service")
            if hasattr(AGENT, "_llm_service_detailed"):
                delattr(AGENT, "_llm_service_detailed")
        except Exception:
            pass


class AddSourceRequest(BaseModel):
    folder: str = Field(..., min_length=1)


class RemoveSourceRequest(BaseModel):
    folder: str = Field(..., min_length=1)


class RemoveSourcesBatchRequest(BaseModel):
    folders: List[str] = Field(..., min_length=1)


AGENT: Optional["FileAgent"] = None

MODEL_MANAGER = None
PREF_MANAGER = None
LLM_MANAGER = None
SOURCES = None
HISTORY = None

def _get_managers():
    global MODEL_MANAGER, PREF_MANAGER, LLM_MANAGER, SOURCES, HISTORY
    if MODEL_MANAGER is None:
        from services.model_manager import ModelManager
        from services.local_llm import get_local_llm_manager
        from services.preference_manager import PreferenceManager
        from config import settings
        
        DEFAULT_STATE_DIR = os.path.join(BASE_DIR, "data")
        STATE_DIR = _DATA_DIR or DEFAULT_STATE_DIR
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
        except Exception:
            pass
            
        PREF_MANAGER = PreferenceManager(BASE_DIR)
        MODEL_MANAGER = ModelManager(BASE_DIR)
        LLM_MANAGER = get_local_llm_manager(BASE_DIR)
        SOURCES = SourceStore(STATE_DIR)
        HISTORY = HistoryManager(STATE_DIR)
        
        def _init_models_bg():
            try:
                selected_model = PREF_MANAGER.get_selected_model_id()
                if selected_model:
                    qf = PREF_MANAGER.get_selected_quantization_file(selected_model)
                    LLM_MANAGER.start_server(selected_model, preferred_quantization_file=qf)
            except Exception as e:
                logger.error(f"[Server] Failed to auto-start model: {e}")
        
        threading.Thread(target=_init_models_bg, daemon=True).start()
        
    return MODEL_MANAGER, PREF_MANAGER, LLM_MANAGER, SOURCES, HISTORY

DEFAULT_STATE_DIR = os.path.join(BASE_DIR, "data")
STATE_DIR = _DATA_DIR or DEFAULT_STATE_DIR
try:
    os.makedirs(STATE_DIR, exist_ok=True)
except Exception:
    pass

def _maybe_migrate_root_state_to_data_dir():
    try:
        if _DATA_DIR:
            return
        if os.path.abspath(STATE_DIR) == os.path.abspath(BASE_DIR):
            return
        os.makedirs(STATE_DIR, exist_ok=True)
        candidates = [
            "chat_history.json",
            "user_preferences.json",
            "indexed_sources.json",
            "indexed_folders.json",
            "active_index_job.json",
            "backend.port.json",
            "backend.singleton.lock",
        ]
        for name in candidates:
            src = os.path.join(BASE_DIR, name)
            dst = os.path.join(STATE_DIR, name)
            if os.path.exists(src) and (not os.path.exists(dst)):
                try:
                    os.rename(src, dst)
                except Exception:
                    try:
                        import shutil
                        shutil.copy2(src, dst)
                    except Exception:
                        pass
    except Exception:
        pass

_maybe_migrate_root_state_to_data_dir()
SOURCES = SourceStore(STATE_DIR)
HISTORY = HistoryManager(STATE_DIR)
JOBS: Dict[str, IndexJobState] = {}
JOBS_LOCK = threading.Lock()
INDEXING_LOCK = threading.Lock()
ACTIVE_INDEX_JOB_ID: Optional[str] = None
INDEX_CANCEL_EVENT = threading.Event()
_BACKEND_SINGLETON_LOCK_FH = None

_MODEL_BEFORE_INDEXING: Optional[str] = None

ACTIVE_INDEX_STATE_PATH = os.path.join(STATE_DIR, "active_index_job.json")
_ACTIVE_INDEX_STATE_LAST_WRITE_TS = 0.0

BACKEND_PORT_PATH = os.path.join(STATE_DIR, "backend.port.json")


def _write_backend_port_file(host: str, port: int) -> None:
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
    except Exception:
        pass
    try:
        import json

        with open(BACKEND_PORT_PATH, "w", encoding="utf-8") as f:
            json.dump({"host": str(host), "port": int(port), "ts": time.time()}, f, ensure_ascii=False)
    except Exception:
        pass


def _read_backend_port_file() -> Optional[Dict[str, Any]]:
    try:
        import json

        if not os.path.exists(BACKEND_PORT_PATH):
            return None
        with open(BACKEND_PORT_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:
        return None



def _update_file_index_eta(job: IndexJobState, files_done: int, total_files: int) -> None:
    """Estimate remaining seconds for batch file indexing."""
    total = max(0, int(total_files or 0))
    done = max(0, min(int(files_done or 0), total))
    remaining = max(0, total - done)
    if done <= 0 or remaining <= 0:
        job.eta_seconds = 0
        return
    started = float(job.started_at or 0.0)
    if started <= 0:
        job.eta_seconds = 0
        return
    elapsed = max(0.001, time.time() - started)
    avg = elapsed / done
    job.eta_seconds = max(1, int(remaining * avg))


def _write_active_index_state(job: IndexJobState, *, kind: str, files: Optional[List[str]] = None) -> None:
    global _ACTIVE_INDEX_STATE_LAST_WRITE_TS
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
    except Exception:
        pass
    try:
        _tp = job.to_payload()
        payload: Dict[str, Any] = {
            "job_id": job.job_id,
            "kind": kind,  # folder|files
            "folder": job.folder,
            "is_indexing": bool(job.is_indexing),
            "total_files": int(job.total_files or 0),
            "completed_files": int(job.completed_files or 0),
            "eta_seconds": int(job.eta_seconds or 0),
            "eta": str(_tp.get("eta") or "—"),
            "current_file": str(job.current_file or ""),
            "started_at": float(job.started_at or 0.0),
            "finished_at": float(job.finished_at or 0.0) if job.finished_at else None,
            "error": job.error,
            "updated_at": time.time(),
        }
        if files:
            payload["files"] = files
        try:
            payload["model_before_indexing"] = _MODEL_BEFORE_INDEXING
        except Exception:
            payload["model_before_indexing"] = None
        with open(ACTIVE_INDEX_STATE_PATH, "w", encoding="utf-8") as f:
            import json

            json.dump(payload, f, ensure_ascii=False, indent=2)
        _ACTIVE_INDEX_STATE_LAST_WRITE_TS = time.time()
    except Exception:
        pass


def _maybe_write_active_index_state_throttled(job: IndexJobState, *, kind: str, files: Optional[List[str]] = None) -> None:
    """Throttle active index state writes to roughly once every 2 seconds."""
    global _ACTIVE_INDEX_STATE_LAST_WRITE_TS
    try:
        now = time.time()
        if now - float(_ACTIVE_INDEX_STATE_LAST_WRITE_TS or 0.0) < 2.0:
            return
    except Exception:
        pass
    _write_active_index_state(job, kind=kind, files=files)


def _read_active_index_state() -> Optional[Dict[str, Any]]:
    try:
        if not os.path.exists(ACTIVE_INDEX_STATE_PATH):
            return None
        import json

        with open(ACTIVE_INDEX_STATE_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _clear_active_index_state() -> None:
    try:
        if os.path.exists(ACTIVE_INDEX_STATE_PATH):
            os.remove(ACTIVE_INDEX_STATE_PATH)
    except Exception:
        pass


def _persisted_index_state_is_terminal(st: Optional[Dict[str, Any]]) -> bool:
    try:
        if not st or st.get("is_indexing") is not True:
            return False
        if str(st.get("error") or "").strip():
            return True
        total_files = max(0, int(st.get("total_files") or 0))
        completed_files = max(0, int(st.get("completed_files") or 0))
        failed_files = max(0, int(st.get("failed_files") or 0))
        return bool(total_files > 0 and (completed_files + failed_files) >= total_files)
    except Exception:
        return False


def _resume_index_job_if_needed() -> None:
    st = _read_active_index_state()
    if _persisted_index_state_is_terminal(st):
        _clear_active_index_state()
        return
    if not st:
        return
    if st.get("is_indexing") is not True:
        return

    try:
        job_id = str(st.get("job_id") or "").strip()
        kind = str(st.get("kind") or "folder").strip().lower()
        folder = str(st.get("folder") or "").strip()
        files = st.get("files") if isinstance(st.get("files"), list) else None

        if not job_id:
            return

        with JOBS_LOCK:
            global ACTIVE_INDEX_JOB_ID
            if ACTIVE_INDEX_JOB_ID and ACTIVE_INDEX_JOB_ID in JOBS and JOBS[ACTIVE_INDEX_JOB_ID].is_indexing:
                return
            if job_id in JOBS and JOBS[job_id].is_indexing:
                ACTIVE_INDEX_JOB_ID = job_id
                return

        global _MODEL_BEFORE_INDEXING
        mb = st.get("model_before_indexing")
        if isinstance(mb, str) and mb.strip():
            _MODEL_BEFORE_INDEXING = mb.strip()

        if kind == "files" and files:
            job = IndexJobState(job_id=job_id, folder="")
            job.is_indexing = True
            with JOBS_LOCK:
                JOBS[job_id] = job
                ACTIVE_INDEX_JOB_ID = job_id

            def _runner():
                try:
                    from core.langgraph_agent import get_kb_instance

                    kb = get_kb_instance()
                    job.started_at = time.time()
                    kb.enter_write_heavy_mode(reason="resume_file_index_job")
                    with INDEXING_LOCK:
                        _ensure_indexing_model_active()
                        for idx, fp in enumerate([str(x) for x in files], 1):
                            fp_abs = os.path.abspath(os.path.expanduser(fp))
                            job.current_file = os.path.basename(fp_abs)
                            job.total_files = len(files)
                            job.completed_files = idx - 1
                            try:
                                kb.index_file(fp_abs)
                            except Exception:
                                pass
                            job.completed_files = idx
                            _maybe_write_active_index_state_throttled(job, kind="files", files=[str(x) for x in files])
                    job.is_indexing = False
                    job.finished_at = time.time()
                    job.eta_seconds = 0
                    job.error = None
                except Exception as e:
                    job.is_indexing = False
                    job.finished_at = time.time()
                    job.error = str(e)
                finally:
                    try:
                        kb.leave_write_heavy_mode(reason="resume_file_index_job")
                    except Exception:
                        pass
                    try:
                        kb.request_query_cache_prewarm(background=True, reason="post_resume_file_index")
                    except Exception:
                        pass
                    with JOBS_LOCK:
                        global ACTIVE_INDEX_JOB_ID
                        if ACTIVE_INDEX_JOB_ID == job.job_id:
                            ACTIVE_INDEX_JOB_ID = None
                    if not job.error:
                        _clear_active_index_state()
                    try:
                        _restore_model_after_indexing()
                    except Exception:
                        pass

            threading.Thread(target=_runner, daemon=True).start()
            logger.info(f"[Server] 🔄 Resumed file indexing job: job_id={job_id} files={len(files)}")
            return

        if not folder:
            return
        folder = os.path.abspath(os.path.expanduser(folder))
        job = IndexJobState(job_id=job_id, folder=folder)
        job.is_indexing = True
        with JOBS_LOCK:
            JOBS[job_id] = job
            ACTIVE_INDEX_JOB_ID = job_id
        _write_active_index_state(job, kind="folder")
        threading.Thread(target=_run_index_job, args=(job,), daemon=True).start()
        logger.info(f"[Server] 🔄 Resumed folder indexing job: job_id={job_id} folder={folder}")
    except Exception as e:
        logger.error(f"[Server] ⚠️ Resume indexing failed: {e}")


@app.get("/api/index/active")
def index_active() -> Dict[str, Any]:
    j = _indexing_in_progress()
    if j:
        return {"active": True, "job_id": j.job_id, "job": j.to_payload()}
    st = _read_active_index_state()
    if st:
        if _persisted_index_state_is_terminal(st):
            _clear_active_index_state()
            return {"active": False, "job_id": None}
        job_id = str(st.get("job_id") or "")
        job_payload = None
        if job_id:
            with JOBS_LOCK:
                jj = JOBS.get(job_id)
            if jj:
                job_payload = jj.to_payload()
        return {"active": False, "job_id": job_id or None, "job": job_payload, "persisted": st}
    return {"active": False, "job_id": None}


def _ensure_indexing_model_active() -> None:
    global _MODEL_BEFORE_INDEXING
    try:
        # Indexing must use the Add Sources model only. Do not fall back to chat.
        index_model_id = (PREF_MANAGER.get_selected_index_model_id() or "").strip()
            
        if not index_model_id:
            logger.warning("[Server] [Indexing] no selected Add Sources index model")
            return

        current_id = getattr(LLM_MANAGER, "current_model_id", None)
        
        if _MODEL_BEFORE_INDEXING is None and current_id and current_id != index_model_id:
            _MODEL_BEFORE_INDEXING = current_id

        if current_id != index_model_id:
            qf = None
            try:
                qf = PREF_MANAGER.get_selected_quantization_file(index_model_id)
            except Exception:
                pass
            logger.info(
                f"[Server] [Indexing] 索引锁内切换模型: {current_id or 'none'} -> {index_model_id}"
            )
            LLM_MANAGER.stop_server()
            time.sleep(0.3)
            LLM_MANAGER.start_server(preferred_model_id=index_model_id, preferred_quantization_file=qf)
            if getattr(LLM_MANAGER, "current_model_id", None):
                logger.info(
                    f"[Server] [Indexing] 索引模型已加载: {LLM_MANAGER.current_model_id}"
                )
            
            try:
                from services.inproc_openai_client import clear_inproc_openai_client
                clear_inproc_openai_client()
            except Exception as e:
                logger.error(f"[Server] [Indexing] Failed to clear inproc client cache: {e}")
    except Exception as e:
        logger.error(f"[Server] ⚠️ Failed to activate indexing model: {e}")


def _restore_model_after_indexing() -> None:
    global _MODEL_BEFORE_INDEXING
    prev = _MODEL_BEFORE_INDEXING
    _MODEL_BEFORE_INDEXING = None

    if not prev:
        return

    current_loaded = getattr(LLM_MANAGER, "current_model_id", None)
    if current_loaded == prev:
        return

    try:
        qf = None
        try:
            qf = PREF_MANAGER.get_selected_quantization_file(prev)
        except Exception:
            pass
        logger.info(
            f"[Server] [Indexing] 索引结束，恢复聊天模型: {current_loaded} -> {prev}"
        )
        LLM_MANAGER.stop_server()
        time.sleep(0.3)
        LLM_MANAGER.start_server(preferred_model_id=prev, preferred_quantization_file=qf)
        logger.info(
            f"[Server] [Indexing] 已恢复聊天模型: {getattr(LLM_MANAGER, 'current_model_id', None)}"
        )
        
        try:
            from services.inproc_openai_client import clear_inproc_openai_client
            clear_inproc_openai_client()
        except Exception as e:
            logger.error(f"[Server] [Indexing] Failed to clear inproc client cache: {e}")
            
    except Exception as e:
        logger.error(f"[Server] ⚠️ Failed to restore model {prev}: {e}")


def _acquire_backend_singleton_lock() -> None:
    global _BACKEND_SINGLETON_LOCK_FH
    try:
        raw = (os.getenv("FILEAGENT_DISABLE_BACKEND_SINGLETON_LOCK") or "").strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            print("[Server] ⚠️ Backend singleton lock disabled by env.")
            return
    except Exception:
        pass

    try:
        import fcntl  # type: ignore
    except Exception:
        return

    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        lock_path = os.path.join(STATE_DIR, "backend.singleton.lock")
        fh = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                info = _read_backend_port_file() or {}
                host = str(info.get("host") or os.getenv("FILEAGENT_BACKEND_HOST", "127.0.0.1"))
                port = int(info.get("port") or 0)
                if port > 0:
                    print(f"FILEAGENT_BACKEND_LISTENING host={host} port={port}")
                    get_logger().info(
                        f"FILEAGENT_BACKEND_LISTENING host={host} port={port}"
                    )
            except Exception:
                pass
            print(f"[Server] ❌ Another backend is already running (lock busy): {lock_path}", flush=True)
            os._exit(0)

        try:
            fh.seek(0)
            fh.truncate()
            fh.write(f"pid={os.getpid()} started_at={time.time()}\n")
            fh.flush()
        except Exception:
            pass

        _BACKEND_SINGLETON_LOCK_FH = fh
        print(f"[Server] ✅ Backend singleton lock acquired: {lock_path}")
    except Exception as e:
        print(f"[Server] ⚠️ Failed to acquire backend singleton lock: {e}")


def _indexing_in_progress() -> Optional[IndexJobState]:
    with JOBS_LOCK:
        if ACTIVE_INDEX_JOB_ID and ACTIVE_INDEX_JOB_ID in JOBS:
            j = JOBS[ACTIVE_INDEX_JOB_ID]
            if j.is_indexing:
                return j
        for j in JOBS.values():
            if j and j.is_indexing:
                return j
    return None

# ==================== Core Models (Embedding / Reranker) download state ====================

CORE_MODELS_LOCK = threading.Lock()
CORE_MODELS_STATE: Dict[str, Any] = {
    "embedding": {"status": "idle", "error": None},
    "reranker": {"status": "idle", "error": None},
    "is_downloading": False,
    "progress": 0,
}
_GGUF_VALID_CACHE: Dict[str, Dict[str, Any]] = {}
_GGUF_VALID_CACHE_TTL_SEC = 15.0


def _model_dir_installed(local_dir: str) -> bool:
    """Minimal local integrity check. Supports HF directories and single GGUF files."""
    try:
        if local_dir.endswith(".gguf"):
            from services.download_utils import (
                _is_gguf_complete,
                _query_gguf_file_size,
                _read_gguf_size_hint,
                _write_gguf_size_hint,
                _is_gguf_loadable,
            )
            if not os.path.isfile(local_dir):
                return False
            st = os.stat(local_dir)
            now = time.time()
            cached = _GGUF_VALID_CACHE.get(local_dir)
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
            filename = os.path.basename(local_dir)
            repo_id = ""
            if filename == settings.RERANKER_GGUF_FILE:
                repo_id = settings.RERANKER_MODEL
            elif filename == os.path.basename(settings.LOCAL_EMBEDDING_MODEL_PATH):
                repo_id = getattr(settings, "EMBEDDING_REPO_ID", settings.EMBEDDING_MODEL)
            expected = _read_gguf_size_hint(local_dir)
            
            # Fast path: if file size perfectly matches the hint, bypass remote API calls
            if expected > 0 and expected == st.st_size:
                if _is_gguf_complete(local_dir, expected):
                    _GGUF_VALID_CACHE[local_dir] = {"size": int(st.st_size), "mtime": float(st.st_mtime), "ok": True, "ts": now}
                    return True
                    
            if repo_id:
                expected = max(expected, _query_gguf_file_size(repo_id, filename))
            if expected > 0:
                ok = _is_gguf_complete(local_dir, expected)
                if not ok:
                    try:
                        ok = _is_gguf_loadable(local_dir)
                    except Exception:
                        ok = False
                if ok:
                    try:
                        _write_gguf_size_hint(local_dir, os.path.getsize(local_dir))
                    except Exception:
                        pass
                _GGUF_VALID_CACHE[local_dir] = {"size": int(st.st_size), "mtime": float(st.st_mtime), "ok": bool(ok), "ts": now}
                return ok
            ok2 = _is_gguf_loadable(local_dir)
            if ok2:
                try:
                    _write_gguf_size_hint(local_dir, os.path.getsize(local_dir))
                except Exception:
                    pass
            _GGUF_VALID_CACHE[local_dir] = {"size": int(st.st_size), "mtime": float(st.st_mtime), "ok": bool(ok2), "ts": now}
            return ok2
        has_config = os.path.exists(os.path.join(local_dir, "config.json"))
        has_safetensors = os.path.exists(os.path.join(local_dir, "model.safetensors"))
        has_bin = os.path.exists(os.path.join(local_dir, "pytorch_model.bin"))
        return bool(has_config and (has_safetensors or has_bin))
    except Exception:
        return False


def _core_models_payload() -> Dict[str, Any]:
    emb_installed = _model_dir_installed(settings.LOCAL_EMBEDDING_MODEL_PATH)
    rr_installed = _model_dir_installed(settings.LOCAL_RERANKER_MODEL_PATH)
    with CORE_MODELS_LOCK:
        st = dict(CORE_MODELS_STATE)
        emb_state = dict(st.get("embedding") or {})
        rr_state = dict(st.get("reranker") or {})
        is_downloading = bool(st.get("is_downloading"))
        progress = int(st.get("progress") or 0)

    def _normalize(item: Dict[str, Any], installed: bool) -> Dict[str, Any]:
        status = (item.get("status") or "idle")
        err = item.get("error")
        if installed:
            # If already installed locally, report as installed (unless explicit error)
            if status not in {"error"}:
                status = "installed"
        elif status == "installed":
            # Local file may be deleted while process is alive; avoid stale installed status.
            status = "downloading" if is_downloading else "idle"
        return {"installed": bool(installed), "status": status, "error": err}

    # Coarse progress: if both installed -> 100; else if one installed -> 50; else use state.progress
    if emb_installed and rr_installed:
        progress2 = 100
    elif emb_installed or rr_installed:
        progress2 = max(progress, 50)
    else:
        progress2 = progress

    return {
        "embedding": _normalize(emb_state, emb_installed),
        "reranker": _normalize(rr_state, rr_installed),
        "progress": int(max(0, min(100, progress2))),
        "is_downloading": is_downloading,
    }


def _download_core_models_worker():
    # Run sequentially, update coarse state.
    with CORE_MODELS_LOCK:
        CORE_MODELS_STATE["is_downloading"] = True
        CORE_MODELS_STATE["progress"] = 0
        CORE_MODELS_STATE["embedding"] = {"status": "downloading", "error": None}
        CORE_MODELS_STATE["reranker"] = {"status": "idle", "error": None}
    try:
        ok1 = False
        def _emb_progress(info: dict):
            with CORE_MODELS_LOCK:
                emb = CORE_MODELS_STATE["embedding"]
                emb["percent"] = round(info.get("percent", 0), 1)
                emb["speed"] = round(info.get("speed_bytes_per_sec", 0))
                emb["eta"] = round(info.get("eta_seconds", 0))
                emb["downloaded_bytes"] = info.get("downloaded_bytes", 0)
                emb["total_bytes"] = info.get("total_bytes", 0)

        ok1 = ensure_gguf_downloaded(
            getattr(settings, "EMBEDDING_REPO_ID", settings.EMBEDDING_MODEL), 
            os.path.basename(settings.LOCAL_EMBEDDING_MODEL_PATH), 
            settings.LOCAL_EMBEDDING_MODEL_PATH,
            on_progress=_emb_progress
        )
        with CORE_MODELS_LOCK:
            CORE_MODELS_STATE["embedding"]["status"] = "installed" if ok1 else "error"
            CORE_MODELS_STATE["embedding"]["error"] = None if ok1 else "embedding_download_failed"
            CORE_MODELS_STATE["progress"] = 50 if ok1 else 0

        if not ok1:
            with CORE_MODELS_LOCK:
                CORE_MODELS_STATE["reranker"]["status"] = "idle"
                CORE_MODELS_STATE["reranker"]["error"] = None
            return

        with CORE_MODELS_LOCK:
            CORE_MODELS_STATE["reranker"]["status"] = "downloading"
            CORE_MODELS_STATE["reranker"]["error"] = None
        ok2 = ensure_gguf_downloaded(settings.RERANKER_MODEL, settings.RERANKER_GGUF_FILE, settings.LOCAL_RERANKER_MODEL_PATH)
        with CORE_MODELS_LOCK:
            CORE_MODELS_STATE["reranker"]["status"] = "installed" if ok2 else "error"
            CORE_MODELS_STATE["reranker"]["error"] = None if ok2 else "reranker_download_failed"
            CORE_MODELS_STATE["progress"] = 100 if (ok1 and ok2) else CORE_MODELS_STATE.get("progress", 0)
    except Exception as e:
        with CORE_MODELS_LOCK:
            # don't clobber per-item errors if present
            if CORE_MODELS_STATE.get("embedding", {}).get("status") != "error":
                CORE_MODELS_STATE["embedding"]["status"] = "error"
                CORE_MODELS_STATE["embedding"]["error"] = str(e)
            if CORE_MODELS_STATE.get("reranker", {}).get("status") != "error":
                CORE_MODELS_STATE["reranker"]["status"] = "error"
                CORE_MODELS_STATE["reranker"]["error"] = str(e)
    finally:
        with CORE_MODELS_LOCK:
            CORE_MODELS_STATE["is_downloading"] = False
            # if installed locally, ensure final progress is 100
            if _model_dir_installed(settings.LOCAL_EMBEDDING_MODEL_PATH) and _model_dir_installed(settings.LOCAL_RERANKER_MODEL_PATH):
                CORE_MODELS_STATE["progress"] = 100


def _ensure_agent():
    global AGENT
    if AGENT is None:
        from core.langgraph_agent import FileAgent  # lazy import
        logger.info("[Agent] 初始化 FileAgent")
        AGENT = FileAgent()
    return AGENT


@app.on_event("shutdown")
def shutdown_event():
    logger.info("[Server] Shutting down...")
    try:
        global AGENT
        if AGENT is not None and hasattr(AGENT, "close"):
            AGENT.close()
    except Exception:
        pass
    try:
        if LLM_MANAGER is not None:
            LLM_MANAGER.stop_server()
    except Exception:
        pass

_ui_ready = threading.Event()

@app.post("/api/notify_ui_ready")
def notify_ui_ready() -> Dict[str, Any]:
    _ui_ready.set()
    return {"ok": True}

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ts": time.time()}


@app.post("/api/dev/reload_handlers")
def reload_handlers() -> Dict[str, Any]:
    """Dev-only: hot-reload handler modules so Python changes take effect without restarting Tauri."""
    import importlib, sys
    reloaded = []
    failed = []
    for mod_name in list(sys.modules.keys()):
        if "handlers" in mod_name or mod_name in (
            "core.handlers.summarize_all_handler",
            "core.handlers.summarize_handler",
            "core.handlers.view_detail_handler",
            "config.prompts",
        ):
            try:
                importlib.reload(sys.modules[mod_name])
                reloaded.append(mod_name)
            except Exception as e:
                failed.append(f"{mod_name}: {e}")
    # Force agent to re-bind handler methods on next call
    import core.agent.dispatch as _d
    try:
        importlib.reload(_d)
    except Exception as e:
        failed.append(f"dispatch: {e}")
    return {"ok": True, "reloaded": reloaded, "failed": failed}


@app.get("/api/runtime/paths")
def runtime_paths() -> Dict[str, Any]:
    try:
        local_models_dir = os.getenv("FILEAGENT_LOCAL_MODELS_DIR") or ""
    except Exception:
        local_models_dir = ""
    return {
        "base_dir": BASE_DIR,
        "data_dir": _DATA_DIR or "",
        "state_dir": STATE_DIR,
        "indexed_folders_path": getattr(SOURCES, "new_path", "") or getattr(SOURCES, "legacy_path", ""),
        "db_path": getattr(settings, "DB_PATH", ""),
        "local_models_dir": local_models_dir,
        "embedding_dir": getattr(settings, "LOCAL_EMBEDDING_MODEL_PATH", ""),
        "reranker_dir": getattr(settings, "LOCAL_RERANKER_MODEL_PATH", ""),
    }


@app.get("/api/sources")
def get_sources() -> Dict[str, Any]:
    SOURCES.load()
    
    indexing_paths = set()
    with JOBS_LOCK:
        for j in JOBS.values():
            if j.is_indexing and j.folder:
                indexing_paths.add(j.folder)

    try:
        st = _read_active_index_state()
        if _persisted_index_state_is_terminal(st):
            _clear_active_index_state()
            st = None
        if st and st.get("is_indexing") and st.get("folder"):
            indexing_paths.add(os.path.abspath(os.path.expanduser(str(st.get("folder")))))
    except Exception:
        pass
    
    sources = []
    indexed_paths = None
    try:
        global _agent
        if _agent is not None:
            from core.langgraph_agent import get_kb_instance
            kb = get_kb_instance()
            indexed_paths = kb.get_indexed_file_paths()
    except Exception:
        pass

    indexed_path_keys: Optional[Set[str]] = None
    if indexed_paths is not None and len(indexed_paths) > 0:
        indexed_path_keys = {_source_path_key(x) for x in indexed_paths}

    for f in (SOURCES.folders or []):
        fa = os.path.abspath(os.path.expanduser(f))
        is_indexing = f in indexing_paths or fa in {os.path.abspath(os.path.expanduser(x)) for x in indexing_paths}
        st = "indexing" if is_indexing else "indexed" # Folder defaults to indexed if not indexing
        
        all_idx = _collect_indexable_file_paths(fa)
        if not all_idx:
            continue
            
        if not is_indexing:
            if indexed_path_keys is not None:
                relevant = all_idx.intersection(indexed_path_keys)
                if not relevant:
                    continue
                prune = True
            else:
                relevant = set(all_idx)
                prune = True
        else:
            relevant = set(all_idx)
            prune = True
            
        sources.append(
            _build_folder_node(
                fa,
                status=st,
                indexed_path_keys=indexed_path_keys,
                relevant_indexable_paths=relevant,
                prune_empty_subfolders=prune,
            )
        )
    
    for fp in (SOURCES.files or []):
        if not _is_indexable_file_for_sources(fp, treat_parent_as_explicit_source=True):
            continue
        fpk = _source_path_key(fp)
        status = "pending" # Default to pending until DB is queried
        if fp in indexing_paths or any(fp.startswith(p) for p in indexing_paths):
            status = "indexing"
        elif indexed_path_keys is not None and fpk in indexed_path_keys:
            status = "indexed"
        elif indexed_path_keys is not None and fpk not in indexed_path_keys:
            continue
        sources.append(_build_file_node(fp, status=status))
    
    return {"sources": sources}


@app.get("/api/history")
def get_history() -> Dict[str, Any]:
    data = HISTORY.load_all()
    sessions = []
    for sid, sdata in data.items():
        if isinstance(sdata, dict):
             if "id" not in sdata: sdata["id"] = sid
             if "messages" not in sdata: sdata["messages"] = []
             if "title" not in sdata: sdata["title"] = "New Chat"
             if "lastActive" not in sdata: sdata["lastActive"] = 0
             sessions.append(sdata)
    
    sessions.sort(key=lambda x: x.get("lastActive", 0), reverse=True)
    return {"sessions": sessions}


@app.post("/api/history/sync")
def sync_history(session: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    if session and session.get("id"):
        HISTORY.save_session(session["id"], session)
    return {"ok": True}


@app.post("/api/history/delete")
def delete_history_item(payload: Dict[str, str] = Body(...)) -> Dict[str, Any]:
    session_id = payload.get("id")
    if session_id:
        HISTORY.delete_session(session_id)
    return {"ok": True}


@app.post("/api/sources/add")
def add_source(req: AddSourceRequest) -> Dict[str, Any]:
    folder = os.path.abspath(os.path.expanduser(req.folder))
    SOURCES.add_folder(folder)
    node = _build_folder_node(folder, status="indexing")
    return {"folder": folder, "node": node}


@app.post("/api/sources/remove")
def remove_source(req: RemoveSourceRequest) -> Dict[str, Any]:
    folder = os.path.abspath(os.path.expanduser(req.folder))
    
    SOURCES.remove(folder)
    logger.info(f"[Server] 已从 indexed_sources.json 移除: {folder}")
    
    try:
        agent = _ensure_agent()
        from core.langgraph_agent import get_kb_instance
        kb = get_kb_instance()
        result = kb.delete_by_folder(folder)
        
        if result.get("ok"):
            deleted_count = result.get("deleted_count", 0)
            logger.info(f"[Server] 已从数据库删除 {deleted_count} 条索引文档")
            return {
                "ok": True,
                "folder": folder,
                "deleted_count": deleted_count,
                "message": f"成功移除文件夹及其 {deleted_count} 条索引"
            }
        else:
            error = result.get("error", "未知错误")
            logger.error(f"[Server] 删除索引失败: {error}")
            return {
                "ok": False,
                "folder": folder,
                "error": error,
                "message": "从配置中移除成功，但删除索引失败"
            }
    except Exception as e:
        error_msg = f"删除索引时出错: {e}"
        logger.error(f"[Server] {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "ok": False,
            "folder": folder,
            "error": error_msg,
            "message": "从配置中移除成功，但删除索引失败"
        }


@app.post("/api/sources/remove_batch")
def remove_sources_batch(req: RemoveSourcesBatchRequest) -> Dict[str, Any]:
    folders_abs = [os.path.abspath(os.path.expanduser(f)) for f in req.folders]

    for folder in folders_abs:
        SOURCES.remove(folder)
    logger.info(f"[Server] 批量移除 {len(folders_abs)} 个数据源配置")

    try:
        _ensure_agent()
        from core.langgraph_agent import get_kb_instance
        kb = get_kb_instance()
        result = kb.delete_by_folders(folders_abs)

        if result.get("ok"):
            deleted_count = result.get("deleted_count", 0)
            logger.info(f"[Server] 批量删除完成，共删除 {deleted_count} 条索引文档")
            return {
                "ok": True,
                "folders": folders_abs,
                "deleted_count": deleted_count,
            }
        else:
            return {
                "ok": False,
                "folders": folders_abs,
                "error": result.get("error", "未知错误"),
            }
    except Exception as e:
        error_msg = f"批量删除索引时出错: {e}"
        logger.error(f"[Server] {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "ok": False,
            "folders": folders_abs,
            "error": error_msg,
        }


class RefreshSourceRequest(BaseModel):
    folder: str


@app.post("/api/sources/refresh")
def refresh_source_endpoint(req: RefreshSourceRequest) -> Dict[str, Any]:
    folder = os.path.abspath(os.path.expanduser(req.folder))
    logger.info(f"[Refresh] 收到刷新请求: folder={folder}")

    with JOBS_LOCK:
        global ACTIVE_INDEX_JOB_ID
        if ACTIVE_INDEX_JOB_ID and ACTIVE_INDEX_JOB_ID in JOBS and JOBS[ACTIVE_INDEX_JOB_ID].is_indexing:
            running = JOBS[ACTIVE_INDEX_JOB_ID]
            return {"job_id": running.job_id, "folder": folder, "already_running": True}

    job_id = str(uuid.uuid4())
    job = IndexJobState(job_id=job_id, folder=folder)

    with JOBS_LOCK:
        job.is_indexing = True
        JOBS[job_id] = job
        ACTIVE_INDEX_JOB_ID = job_id

    def _run_refresh_job():
        try:
            logger.info(f"[Refresh] job started: job_id={job.job_id} folder={folder}")
            _ensure_agent()
            from core.langgraph_agent import get_kb_instance
            kb = get_kb_instance()
            job.started_at = time.time()
            job.message = "scanning"
            kb.enter_write_heavy_mode(reason="refresh_index_job")

            try:
                INDEX_CANCEL_EVENT.clear()
            except Exception:
                pass

            def cb(current: int, total: int, file_name: str, *_args) -> None:
                job.total_files = int(total or 0)
                job.completed_files = int(current or 0)
                job.current_file = file_name or ""
                elapsed = max(0.001, time.time() - job.started_at)
                speed = job.completed_files / elapsed
                remaining = max(0, job.total_files - job.completed_files)
                job.eta_seconds = int(remaining / speed) if speed > 0 else 0

            with INDEXING_LOCK:
                _ensure_indexing_model_active()
                refresh_result = kb.refresh_source(
                    directories=[folder],
                    progress_callback=cb,
                    should_cancel=lambda: bool(INDEX_CANCEL_EVENT.is_set()),
                )

            job.is_indexing = False
            job.finished_at = time.time()
            job.eta_seconds = 0
            job.error = "cancelled" if INDEX_CANCEL_EVENT.is_set() else None
            job.message = (
                f"refresh_done|added={refresh_result.get('added',0)}"
                f"|updated={refresh_result.get('updated',0)}"
                f"|deleted={refresh_result.get('deleted',0)}"
                f"|skipped={refresh_result.get('skipped',0)}"
                f"|errors={refresh_result.get('errors',0)}"
            )
            logger.info(f"[Refresh] job finished: {job.message}")
        except Exception as e:
            job.is_indexing = False
            job.finished_at = time.time()
            job.error = str(e)
            logger.error(f"[Refresh] job failed: job_id={job.job_id} error={e}")
        finally:
            try:
                kb.leave_write_heavy_mode(reason="refresh_index_job")
            except Exception:
                pass
            try:
                kb.request_query_cache_prewarm(background=True, reason="post_refresh_index")
            except Exception:
                pass
            global ACTIVE_INDEX_JOB_ID
            with JOBS_LOCK:
                if ACTIVE_INDEX_JOB_ID == job.job_id:
                    ACTIVE_INDEX_JOB_ID = None
            try:
                _write_active_index_state(job, kind="folder")
            except Exception:
                pass
            if (job.error is None) or (str(job.error).strip().lower() == "cancelled"):
                _clear_active_index_state()
            try:
                _restore_model_after_indexing()
            except Exception:
                pass

    t = threading.Thread(target=_run_refresh_job, daemon=True)
    t.start()
    logger.info(f"[Refresh] job dispatched: job_id={job_id} folder={folder}")
    return {"job_id": job_id, "folder": folder}


def _run_index_job(job: IndexJobState) -> None:

    try:
        logger.info(f"[Index] folder job started: job_id={job.job_id} folder={job.folder}")
        agent = _ensure_agent()
        from core.langgraph_agent import get_kb_instance  # lazy import
        kb = get_kb_instance()
        job.started_at = time.time()
        kb.enter_write_heavy_mode(reason="folder_index_job")
        try:
            INDEX_CANCEL_EVENT.clear()
        except Exception:
            pass

        def cb(current: int, total: int, file_name: str) -> None:
            now = time.time()
            job.total_files = int(total or 0)
            job.completed_files = int(current or 0)
            job.current_file = file_name or ""
            elapsed = max(0.001, now - job.started_at)
            speed = job.completed_files / elapsed
            remaining = max(0, job.total_files - job.completed_files)
            job.eta_seconds = int(remaining / speed) if speed > 0 else 0
            _maybe_write_active_index_state_throttled(job, kind="folder")

        with INDEXING_LOCK:
            _ensure_indexing_model_active()
            kb.scan_directory(
                directories=[job.folder],
                progress_callback=cb,
                should_cancel=lambda: bool(INDEX_CANCEL_EVENT.is_set()),
            )
        job.is_indexing = False
        job.finished_at = time.time()
        job.eta_seconds = 0
        job.error = "cancelled" if INDEX_CANCEL_EVENT.is_set() else None
        logger.info(
            f"[Index] folder job finished: job_id={job.job_id} folder={job.folder} "
            f"completed={job.completed_files}/{job.total_files} cancelled={bool(job.error == 'cancelled')}"
        )
    except Exception as e:
        job.is_indexing = False
        job.finished_at = time.time()
        job.error = str(e)
        logger.error(f"[Index] folder job failed: job_id={job.job_id} folder={job.folder} error={e}")
    finally:
        try:
            kb.leave_write_heavy_mode(reason="folder_index_job")
        except Exception:
            pass
        try:
            kb.request_query_cache_prewarm(background=True, reason="post_folder_index")
        except Exception:
            pass
        global ACTIVE_INDEX_JOB_ID
        with JOBS_LOCK:
            if ACTIVE_INDEX_JOB_ID == job.job_id:
                ACTIVE_INDEX_JOB_ID = None
        try:
            _write_active_index_state(job, kind="folder")
        except Exception:
            pass
        if (job.error is None) or (str(job.error).strip().lower() == "cancelled"):
            _clear_active_index_state()
        try:
            _restore_model_after_indexing()
        except Exception:
            pass


@app.post("/api/index/start")
def start_index(req: AddSourceRequest) -> Dict[str, Any]:
    folder = os.path.abspath(os.path.expanduser(req.folder))
    logger.info(f"[Index] request start folder indexing: folder={folder}")
    SOURCES.add_folder(folder)

    with JOBS_LOCK:
        global ACTIVE_INDEX_JOB_ID
        if ACTIVE_INDEX_JOB_ID and ACTIVE_INDEX_JOB_ID in JOBS and JOBS[ACTIVE_INDEX_JOB_ID].is_indexing:
            running = JOBS[ACTIVE_INDEX_JOB_ID]
            logger.info(
                f"[Index] reject new folder job because another is running: "
                f"active_job_id={running.job_id} active_folder={running.folder}"
            )
            return {"job_id": running.job_id, "folder": running.folder, "already_running": True}

    job_id = str(uuid.uuid4())
    job = IndexJobState(job_id=job_id, folder=folder)
    with JOBS_LOCK:
        job.is_indexing = True
        JOBS[job_id] = job
        ACTIVE_INDEX_JOB_ID = job_id
    _write_active_index_state(job, kind="folder")
    t = threading.Thread(target=_run_index_job, args=(job,), daemon=True)
    t.start()
    logger.info(f"[Index] folder job dispatched: job_id={job_id} folder={folder}")
    return {"job_id": job_id, "folder": folder}


@app.post("/api/index/reindex_file")
def reindex_file(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    file_path = str(payload.get("file_path") or "").strip()
    if not file_path:
        return {"ok": False, "error": "file_path 参数不能为空"}
    file_path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.exists(file_path):
        return {"ok": False, "error": f"文件不存在: {file_path}"}

    try:
        from core.langgraph_agent import get_kb_instance
        kb = get_kb_instance()
        kb.enter_write_heavy_mode(reason="api_reindex_file")

        # Step 1: delete existing chunks
        deleted = kb.delete_file(file_path)
        logger.info(f"[reindex_file] delete_file_name_chars={len(os.path.basename(file_path))} deleted={deleted}")

        # Step 2: re-index with current code/prompts
        _ensure_indexing_model_active()
        ok = kb.ingest_file(file_path, use_smart_indexing=True)
        logger.info(f"[reindex_file] ingest_file_name_chars={len(os.path.basename(file_path))} ok={ok}")

        return {"ok": ok, "file_path": file_path, "deleted_first": deleted}
    except Exception as e:
        logger.error(f"[reindex_file] error: {e}")
        return {"ok": False, "error": str(e)}
    finally:
        try:
            kb.leave_write_heavy_mode(reason="api_reindex_file")
        except Exception:
            pass
        try:
            kb.request_query_cache_prewarm(background=True, reason="post_reindex_file")
        except Exception:
            pass


@app.post("/api/index/files")
def index_files(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    files = payload.get("files", [])
    if not files or not isinstance(files, list):
        return {"ok": False, "error": "files 参数必须是非空数组"}
    
    file_paths = [os.path.abspath(os.path.expanduser(fp)) for fp in files]
    logger.info(f"[Index] request start files indexing: total_files={len(file_paths)}")
    
    with JOBS_LOCK:
        global ACTIVE_INDEX_JOB_ID
        if ACTIVE_INDEX_JOB_ID and ACTIVE_INDEX_JOB_ID in JOBS and JOBS[ACTIVE_INDEX_JOB_ID].is_indexing:
            running = JOBS[ACTIVE_INDEX_JOB_ID]
            logger.info(
                f"[Index] reject file job because another is running: "
                f"active_job_id={running.job_id} active_folder={running.folder}"
            )
            return {"ok": False, "error": "indexing_in_progress", "job_id": running.job_id, "folder": running.folder}
    
    job_id = str(uuid.uuid4())
    job = IndexJobState(
        job_id=job_id,
        folder="",
        total_files=len(file_paths),
        completed_files=0
    )
    
    with JOBS_LOCK:
        job.is_indexing = True
        JOBS[job_id] = job
        ACTIVE_INDEX_JOB_ID = job_id
    _write_active_index_state(job, kind="files", files=file_paths)
    
    def _run_file_index_job():
        try:
            logger.info(f"[Index] file job started: job_id={job.job_id} total_files={len(file_paths)}")
            agent = _ensure_agent()
            from core.langgraph_agent import get_kb_instance
            kb = get_kb_instance()
            kb.enter_write_heavy_mode(reason="files_index_job")
            
            success_files = []
            failed_files = []
            
            try:
                INDEX_CANCEL_EVENT.clear()
            except Exception:
                pass
            
            with INDEXING_LOCK:
                _ensure_indexing_model_active()
                n_files = len(file_paths)
                job.started_at = time.time()
                for idx, fp in enumerate(file_paths, 1):
                    if INDEX_CANCEL_EVENT.is_set():
                        logger.warning("[Server] 收到取消信号：停止本次文件索引")
                        break
                    if not os.path.exists(fp):
                        logger.warning(f"[Server] 文件不存在，跳过: {fp}")
                        failed_files.append((fp, "文件不存在"))
                        job.current_file = os.path.basename(fp)
                        job.completed_files = idx
                        _update_file_index_eta(job, idx, n_files)
                        _maybe_write_active_index_state_throttled(job, kind="files", files=file_paths)
                        continue
                    
                    job.current_file = os.path.basename(fp)
                    job.completed_files = idx - 1
                    _update_file_index_eta(job, idx - 1, n_files)
                    
                    try:
                        success = kb.index_file(fp)
                        if success:
                            success_files.append(fp)
                            SOURCES.add_file(fp)
                            logger.info(f"[Server] ✓ 已索引文件: {fp}")
                        else:
                            failed_files.append((fp, "被忽略或无法读取"))
                            logger.warning(f"[Server] ✗ 文件被忽略或无法读取: {fp}")
                    except Exception as e:
                        failed_files.append((fp, str(e)))
                        logger.error(f"[Server] ✗ 索引文件失败 {fp}: {e}")
                    
                    job.completed_files = idx
                    _update_file_index_eta(job, idx, n_files)
                    _maybe_write_active_index_state_throttled(job, kind="files", files=file_paths)
        
            job.is_indexing = False
            job.finished_at = time.time()
            job.eta_seconds = 0
            
            from datetime import datetime

            log_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] File indexing job completed - success: {len(success_files)}, failed: {len(failed_files)}\n"
            if failed_files:
                log_msg += "Failed files:\n"
                for fp, reason in failed_files:
                    log_msg += f"  - {fp}: {reason}\n"
            get_logger().info(log_msg)
            
            try:
                base_dir = os.environ.get("FILEAGENT_DATA_DIR", "")
                if base_dir:
                    base_dir = os.path.abspath(os.path.expanduser(base_dir))
                else:
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
                
                logs_dir = os.path.join(base_dir, "logs")
                os.makedirs(logs_dir, exist_ok=True)
                with open(os.path.join(logs_dir, "index_details.log"), "a", encoding="utf-8") as f:
                    f.write(log_msg + "\n")
                    f.flush()
            except Exception as e:
                logger.error(f"[Server] 写入日志文件失败: {e}")
                
            job.message = f"索引完成: 成功 {len(success_files)} 个, 失败 {len(failed_files)} 个"
            
            if INDEX_CANCEL_EVENT.is_set():
                job.error = "cancelled"
                job.message = "索引已取消"
                logger.info(f"[Server] 文件索引已取消：已处理 {job.completed_files}/{len(file_paths)} 个文件")
            else:
                logger.info(f"[Server] 文件索引完成，共 {len(file_paths)} 个文件")
            
        except Exception as e:
            job.is_indexing = False
            job.finished_at = time.time()
            job.error = str(e)
            logger.error(f"[Server] 文件索引失败: {e}")
        finally:
            try:
                kb.leave_write_heavy_mode(reason="files_index_job")
            except Exception:
                pass
            try:
                kb.request_query_cache_prewarm(background=True, reason="post_files_index")
            except Exception:
                pass
            global ACTIVE_INDEX_JOB_ID
            with JOBS_LOCK:
                if ACTIVE_INDEX_JOB_ID == job.job_id:
                    ACTIVE_INDEX_JOB_ID = None
            if (job.error is None) or (str(job.error).strip().lower() == "cancelled"):
                _clear_active_index_state()
            try:
                _restore_model_after_indexing()
            except Exception:
                pass
    
    t = threading.Thread(target=_run_file_index_job, daemon=True)
    t.start()
    
    return {"ok": True, "job_id": job_id, "files": file_paths}


@app.get("/api/index/status")
def index_status(job_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        st = _read_active_index_state()
        if _persisted_index_state_is_terminal(st):
            _clear_active_index_state()
            st = None
        if st and st.get("job_id") == job_id:
            return {"job": st}
        return {"error": "job_not_found", "job_id": job_id}
    return {"job": job.to_payload()}


@app.post("/api/index/cancel")
def cancel_index(payload: Dict[str, Any] = Body(None)) -> Dict[str, Any]:
    job_id = ""
    try:
        if isinstance(payload, dict):
            job_id = str(payload.get("job_id") or "").strip()
    except Exception:
        job_id = ""

    with JOBS_LOCK:
        active_id = ACTIVE_INDEX_JOB_ID
        active_job = JOBS.get(active_id) if active_id else None

    if not active_id or not active_job or not active_job.is_indexing:
        return {"ok": False, "error": "no_active_job"}
    if job_id and job_id != active_id:
        return {"ok": False, "error": "job_id_mismatch", "active_job_id": active_id}

    try:
        logger.info(f"[Server] 收到取消索引请求，发送取消信号到任务 {active_id}...")
        INDEX_CANCEL_EVENT.set()
    except Exception as e:
        logger.error(f"[Server] 设置取消信号失败: {e}")
        
    return {"ok": True, "job_id": active_id}


@app.get("/api/models")
def list_models():
    raw_ui_ids = (os.getenv("FILEAGENT_UI_MODEL_IDS") or "").strip()
    ui_ids = [x.strip() for x in raw_ui_ids.split(",") if x.strip()] if raw_ui_ids else []
    if not ui_ids:
        from services.model_manager import DEFAULT_UI_MODEL_IDS

        ui_ids = list(DEFAULT_UI_MODEL_IDS)

    all_models = MODEL_MANAGER.get_supported_models()
    ui_set = set(ui_ids)
    models = [m for m in all_models if (m.get("id") in ui_set)]

    if not models:
        models = all_models

    selected_id = PREF_MANAGER.get_selected_model_id()
    model_ids = [str(m.get("id")) for m in models if m.get("id")]
    installed_ids = [str(m.get("id")) for m in models if m.get("id") and (m.get("status") == "installed" or bool(m.get("installed")))]
    selected_installed = any(str(m.get("id")) == str(selected_id) and (m.get("status") == "installed" or bool(m.get("installed"))) for m in models)
    need_fallback = bool(model_ids) and (
        (not selected_id) or
        (selected_id not in model_ids) or
        (installed_ids and not selected_installed)
    )
    if need_fallback:
        preferred_default = "qwen3-4b-gguf"
        if installed_ids:
            fallback_id = preferred_default if preferred_default in installed_ids else installed_ids[0]
        else:
            fallback_id = preferred_default if preferred_default in model_ids else model_ids[0]
        try:
            PREF_MANAGER.set_selected_model_id(fallback_id)
            selected_id = fallback_id
        except Exception:
            selected_id = fallback_id
    qmap = PREF_MANAGER.get_model_quantization_map() if hasattr(PREF_MANAGER, "get_model_quantization_map") else {}
    for m in models:
        m["selected"] = (m["id"] == selected_id)
        mid = m.get("id")
        if mid and isinstance(qmap, dict) and mid in qmap:
            m["selected_quantization"] = qmap.get(mid)
    return {"models": models}


@app.get("/api/core_models/status")
def core_models_status() -> Dict[str, Any]:
    return _core_models_payload()


@app.post("/api/core_models/download")
def core_models_download() -> Dict[str, Any]:
    payload = _core_models_payload()
    if payload.get("embedding", {}).get("installed") and payload.get("reranker", {}).get("installed"):
        return {"ok": True, "already_installed": True, "status": payload}

    with CORE_MODELS_LOCK:
        if CORE_MODELS_STATE.get("is_downloading"):
            return {"ok": True, "already_running": True, "status": payload}

        CORE_MODELS_STATE["is_downloading"] = True
        CORE_MODELS_STATE["progress"] = int(payload.get("progress") or 0)

    t = threading.Thread(target=_download_core_models_worker, daemon=True)
    t.start()
    return {"ok": True, "started": True, "status": _core_models_payload()}


class SelectModelRequest(BaseModel):
    model_id: str


@app.post("/api/models/select")
def select_model(req: SelectModelRequest):
    logger.info(f"[Model] select requested: target_model={req.model_id}")
    j = _indexing_in_progress()
    if j and req.model_id != getattr(LLM_MANAGER, "current_model_id", None):
        logger.warning(
            f"[Model] select rejected due to indexing: target_model={req.model_id} "
            f"active_job_id={j.job_id} folder={j.folder}"
        )
        return {
            "ok": False,
            "error": "indexing_in_progress",
            "job_id": j.job_id,
            "folder": j.folder,
            "message": "索引进行中：禁止切换模型",
            "current_model_id": getattr(LLM_MANAGER, "current_model_id", None),
        }

    prev_id = PREF_MANAGER.get_selected_model_id()
    if prev_id == req.model_id and getattr(LLM_MANAGER, "current_model_id", None) == req.model_id:
        logger.info(f"[Model] select noop: already_selected model_id={req.model_id}")
        return {"ok": True, "model_id": req.model_id, "already_selected": True}

    PREF_MANAGER.set_selected_model_id(req.model_id)
    logger.info(f"[Model] select apply preference: prev={prev_id} -> next={req.model_id}")
    try:
        qf = None
        try:
            qf = PREF_MANAGER.get_selected_quantization_file(req.model_id)
        except Exception:
            qf = None

        resolved = LLM_MANAGER.resolve_target_model(req.model_id, preferred_quantization_file=qf)
        if resolved:
            cfg, model_path, _mmproj = resolved
            logger.info(
                f"[Model] switching Local LLM: requested={req.model_id}, resolved={cfg.get('id')}, model_path={model_path}"
            )
        else:
            logger.error(
                f"[Model] switching Local LLM failed to resolve model config: requested={req.model_id}"
            )
            return {"ok": False, "model_id": req.model_id, "error": "模型文件未找到"}

        if getattr(LLM_MANAGER, "current_model_id", None) == req.model_id:
            logger.info(f"[Model] switch noop: already running model_id={req.model_id}")
            return {"ok": True, "model_id": req.model_id, "already_running": True}

        LLM_MANAGER.stop_server()
        
        import time
        time.sleep(0.5)
        
        started = LLM_MANAGER.start_server(
            preferred_model_id=req.model_id,
            preferred_quantization_file=qf,
        )
        current_after = getattr(LLM_MANAGER, "current_model_id", None)
        if (not started) or (current_after != req.model_id):
            try:
                if prev_id:
                    PREF_MANAGER.set_selected_model_id(prev_id)
            except Exception:
                pass
            err_msg = f"模型切换失败：目标={req.model_id}，当前={current_after or '<none>'}"
            logger.error(f"[Model] {err_msg}")
            return {"ok": False, "model_id": req.model_id, "current_model_id": current_after, "error": err_msg}
        if getattr(LLM_MANAGER, "current_model_id", None):
            logger.info(f"[Model] switch success: current_model_id={LLM_MANAGER.current_model_id}")
        
        try:
            from services.inproc_openai_client import clear_inproc_openai_client
            clear_inproc_openai_client()
            logger.info("[Model] ✅ Cleared InProcOpenAI client cache")
        except Exception as cache_err:
            logger.error(f"[Model] Failed to clear client cache: {cache_err}")
        
        global AGENT
        if AGENT is not None:
            try:
                if hasattr(AGENT, "_llm_service"):
                    delattr(AGENT, "_llm_service")
                if hasattr(AGENT, "_llm_service_detailed"):
                    delattr(AGENT, "_llm_service_detailed")
                logger.info("[Model] ✅ Cleared FileAgent LLM service cache")
            except Exception as agent_err:
                logger.error(f"[Model] Failed to clear agent cache: {agent_err}")
    except Exception as e:
        logger.error(f"[Model] Failed to restart LLM server: {e}")
        try:
            if prev_id:
                PREF_MANAGER.set_selected_model_id(prev_id)
        except Exception:
            pass
        return {"ok": False, "model_id": req.model_id, "error": str(e)}
        
    return {"ok": True, "model_id": req.model_id}


class SelectQuantizationRequest(BaseModel):
    model_id: str
    quantization_file: str


@app.post("/api/models/quantization/select")
def select_model_quantization(req: SelectQuantizationRequest):
    logger.info(
        f"[Model] quantization select requested: model_id={req.model_id} quantization_file={req.quantization_file}"
    )
    PREF_MANAGER.set_selected_quantization_file(req.model_id, req.quantization_file)
    selected_id = PREF_MANAGER.get_selected_model_id()

    if selected_id == req.model_id:
        try:
            resolved = LLM_MANAGER.resolve_target_model(req.model_id, preferred_quantization_file=req.quantization_file)
            if resolved:
                cfg, model_path, _mmproj = resolved
                logger.info(
                    f"[Model] switching quantization: model_id={req.model_id}, gguf={req.quantization_file}, "
                    f"resolved={cfg.get('id')}, model_path={model_path}"
                )
            else:
                logger.error(
                    f"[Model] switching quantization failed to resolve: model_id={req.model_id}, "
                    f"gguf={req.quantization_file}"
                )
                return {"ok": False, "model_id": req.model_id, "quantization_file": req.quantization_file, "error": "量化文件未找到"}
            LLM_MANAGER.stop_server()
            
            import time
            time.sleep(0.5)
            
            started = LLM_MANAGER.start_server(
                preferred_model_id=req.model_id,
                preferred_quantization_file=req.quantization_file,
            )
            current_after = getattr(LLM_MANAGER, "current_model_id", None)
            if (not started) or (current_after != req.model_id):
                err_msg = f"量化切换失败：目标={req.model_id}，当前={current_after or '<none>'}"
                logger.error(f"[Model] {err_msg}")
                return {
                    "ok": False,
                    "model_id": req.model_id,
                    "quantization_file": req.quantization_file,
                    "current_model_id": current_after,
                    "error": err_msg,
                }
            logger.info(
                f"[Model] quantization switch success: model_id={req.model_id}, "
                f"quantization_file={req.quantization_file}, current_model_id={current_after}"
            )
            
            try:
                from services.inproc_openai_client import clear_inproc_openai_client
                clear_inproc_openai_client()
                global AGENT
                if AGENT is not None:
                    if hasattr(AGENT, "_llm_service"):
                        delattr(AGENT, "_llm_service")
                    if hasattr(AGENT, "_llm_service_detailed"):
                        delattr(AGENT, "_llm_service_detailed")
                logger.info("[Model] ✅ Cleared LLM caches after quantization switch")
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[Model] quantization switch failed: {e}")
            return {"ok": False, "error": str(e)}

    return {"ok": True, "model_id": req.model_id, "quantization_file": req.quantization_file}


class DownloadModelRequest(BaseModel):
    model_id: str
    source: str = "auto"
    quantization_file: Optional[str] = None


@app.post("/api/models/download")
def download_model(req: DownloadModelRequest):
    return MODEL_MANAGER.download_model(req.model_id, req.source, quantization_file=req.quantization_file)


class CancelModelRequest(BaseModel):
    model_id: str


class DeleteModelRequest(BaseModel):
    model_id: str
    quantization_file: Optional[str] = None


@app.post("/api/models/cancel")
def cancel_download(req: CancelModelRequest):
    return MODEL_MANAGER.cancel_download(req.model_id)


@app.post("/api/models/delete")
def delete_model(req: DeleteModelRequest):
    return MODEL_MANAGER.delete_model(req.model_id, quantization_file=req.quantization_file)


def _get_active_paths(active_ids: List[str]) -> List[str]:
    """Resolve active source ids back to their real paths."""
    active_paths = []
    if not active_ids:
        return []
    
    SOURCES.load()
    
    id_map = {}
    
    def _add_node(node: Dict[str, Any]):
        nid = node.get("id")
        npath = node.get("path")
        if nid and npath:
            id_map[nid] = npath
        for child in node.get("children", []):
            _add_node(child)
            
    for f in SOURCES.folders:
        node = _build_folder_node(f)
        _add_node(node)
        
    for f in SOURCES.files:
        fid = _stable_id(f"file:{f}")
        id_map[fid] = f
        
    for aid in active_ids:
        if aid in id_map:
            active_paths.append(id_map[aid])
            
    return active_paths


@app.get("/api/personal_info/search")
def api_personal_info_search(q: str = "", types: str = "", limit: int = 50):
    if (os.getenv("FILEAGENT_ENABLE_PERSONAL_INFO_API", "") or "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return {"ok": False, "error": "Personal info HTTP API is disabled"}
    try:
        with JOBS_LOCK:
            # Reusing the existing pattern to avoid creating multiple KB instances unnecessarily
            from core.kb.knowledge_base import FileKnowledgeBase
            kb = FileKnowledgeBase()
            
        if not hasattr(kb, 'personal_info_db'):
            return {"ok": False, "error": "PersonalInfoDB not initialized"}
        
        type_list = [t.strip() for t in types.split(",") if t.strip()] if types else None
        results = kb.personal_info_db.search(query=q, types=type_list, limit=limit)
        return {"ok": True, "results": results, "total": len(results)}
    except Exception as e:
        logger.error(f"/api/personal_info/search error: {e}")
        return {"ok": False, "error": str(e)}


@app.get("/api/personal_info/stats")
def api_personal_info_stats():
    if (os.getenv("FILEAGENT_ENABLE_PERSONAL_INFO_API", "") or "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return {"ok": False, "error": "Personal info HTTP API is disabled"}
    try:
        with JOBS_LOCK:
            from core.kb.knowledge_base import FileKnowledgeBase
            kb = FileKnowledgeBase()
            
        if not hasattr(kb, 'personal_info_db'):
            return {"ok": False, "error": "PersonalInfoDB not initialized"}
            
        stats = kb.personal_info_db.get_stats()
        return {"ok": True, "stats": stats.get("stats", []), "total": stats.get("total", 0)}
    except Exception as e:
        logger.error(f"/api/personal_info/stats error: {e}")
        return {"ok": False, "error": str(e)}


@app.post("/api/query")
def query(req: QueryRequest) -> Dict[str, Any]:
    logger.info(
        f"[Chat] /api/query start session_id={req.session_id or '<none>'} "
        f"message_len={len(req.message or '')} active_source_ids={len(req.active_source_ids or [])}"
    )
    running = _indexing_in_progress()
    if running is not None:
        logger.info(f"[Chat] /api/query rejected due to indexing job_id={running.job_id}")
        return {
            "ok": False,
            "answer": "正在索引中，请等待索引完成后再提问。",
            "sources": [],
            "trace": [],
            "query_type": "busy",
            "need_clarify": False,
            "relevantFiles": [],
            "error": "indexing_in_progress",
            "job_id": running.job_id,
        }
    try:
        # Honor frontend-selected chat model for this request.
        _apply_runtime_model_hint(req.model_id)
        agent = _ensure_agent()
        if req.session_id:
            agent.set_abort_flag(req.session_id)
            import time as _t
            _t.sleep(0.05)
            agent.clear_abort_flag(req.session_id)
        active_paths = _get_active_paths(req.active_source_ids) if req.active_source_ids is not None else None
        set_active_paths(active_paths)
        set_active_session_id(req.session_id)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[query] preflight failed: {e}\n{tb}")
        return {
            "ok": False,
            "answer": f"请求处理失败: {e}",
            "sources": [],
            "trace": [],
            "query_type": "error",
            "need_clarify": False,
            "relevantFiles": [],
        }

    answer_parts: List[str] = []
    trace: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    query_type = "agent"
    ok = True

    try:
        for ev in agent.query_stream(
            req.message,
            session_id=req.session_id,
            prompt_language=req.language,
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
        tb = traceback.format_exc()
        logger.error(f"[query] run failed: {e}\n{tb}")
        ok = False
        answer_parts.append(f"请求处理失败: {e}")
        query_type = "error"

    relevant_files = []
    for s in (sources or [])[:20]:
        p = s.get("file_path") or ""
        n = s.get("file_name") or os.path.basename(p) or ""
        relevant_files.append(
            {
                "id": _stable_id(f"file:{p or n}"),
                "name": n,
                "type": _icon_type_for_path(p) if p else "doc",
                "path": p,
            }
        )

    answer_text = "".join(answer_parts).strip()
    logger.info(
        f"[Chat] /api/query done session_id={req.session_id or '<none>'} ok={ok} "
        f"query_type={query_type} answer_len={len(answer_text)} sources={len(sources or [])}"
    )
    return {
        "ok": ok,
        "answer": answer_text,
        "sources": sources,
        "trace": trace,
        "query_type": query_type,
        "need_clarify": query_type == "clarify",
        "relevantFiles": relevant_files,
    }


def _sse(event: str, data: Dict[str, Any]) -> str:
    import json

    return "event: " + event + "\n" + "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


class QueryAbortRequest(BaseModel):
    session_id: str

@app.post("/api/query/abort")
async def abort_query(req: QueryAbortRequest):
    """Abort an ongoing query stream."""
    try:
        agent = _ensure_agent()
        agent.set_abort_flag(req.session_id)
        return {"status": "success", "session_id": req.session_id}
    except Exception as e:
        logger.error(f"[abort_query] Failed to abort: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/api/query_stream")
async def query_stream(req: QueryRequest, request: Request):
    running = _indexing_in_progress()
    if running is not None:
        logger.info(f"[Chat] /api/query_stream rejected due to indexing job_id={running.job_id}")
        return StreamingResponse(
            iter([_sse("done", {
                "type": "done",
                "ok": False,
                "query_type": "busy",
                "message": "正在索引中，请等待索引完成后再提问。",
                "error": "indexing_in_progress",
                "job_id": running.job_id,
            })]),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    try:
        if req.session_id:
            agent = _ensure_agent()
            agent.set_abort_flag(req.session_id)
            import time
            time.sleep(0.05)
            agent.clear_abort_flag(req.session_id)
    except Exception as e:
        logger.warning(f"[query_stream] setup pre-check failed: {e}")

    async def gen():
        logger.info(
            f"[Chat] /api/query_stream start session_id={req.session_id or '<none>'} "
            f"message_len={len(req.message or '')} active_source_ids={len(req.active_source_ids or [])}"
        )
        import json
        def _debug(hyp_id, loc, msg, data=None):
            # Dev-time debug helper — standard logger used in production
            logger.debug(f"[debug] {loc}: {msg} {data or {}}")
            
        _debug("A,B,C,D", "api_server.py:query_stream_start", "query_stream started", {"message": req.message, "active_source_ids": req.active_source_ids})

        # 0) thinking
        yield _sse("status", {"type": "status", "phase": "thinking", "message": "正在思考中..."})

        try:
            # Honor frontend-selected chat model for this request.
            _apply_runtime_model_hint(req.model_id)
            agent = _ensure_agent()
            active_paths = _get_active_paths(req.active_source_ids) if req.active_source_ids is not None else None
            _debug("B", "api_server.py:query_stream_paths", "computed active_paths", {"active_paths": active_paths})
            set_active_paths(active_paths)
            set_active_session_id(req.session_id)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"[query_stream] preflight failed: {e}\n{tb}")
            _debug("B", "api_server.py:query_stream_paths_error", "error computing paths", {"error": str(e), "tb": tb})
            yield _sse("done", {"type": "done", "ok": False, "query_type": "error", "error": str(e)})
            return

        try:
            _debug("C", "api_server.py:query_stream_dispatch", "starting agent.query_stream")
            for ev in agent.query_stream(
                req.message,
                session_id=req.session_id,
                prompt_language=req.language,
            ):
                if await request.is_disconnected():
                    logger.warning("[query_stream] Client disconnected, setting abort flag")
                    agent.set_abort_flag(req.session_id)
                    yield _sse("done", {"type": "done", "ok": False, "query_type": "interrupted", "message": "生成已中断"})
                    return
                
                ev_type = ev.get("type")
                if ev_type in ["text", "done", "status"]:
                    _debug("C,D", "api_server.py:query_stream_yield", f"yielding {ev_type}", {"ev_type": ev_type})
                    
                if ev_type == "status":
                    yield _sse("status", ev)
                elif ev_type == "trace_append":
                    yield _sse("trace_append", ev)
                elif ev_type == "files":
                    yield _sse("files", ev)
                elif ev_type == "opened_file":
                    try:
                        f = (ev.get("file") or {}) if isinstance(ev, dict) else {}
                        fp = (f.get("file_path") or "").strip()
                        content = ev.get("content") or ""
                        is_image = (f.get("type") == "image" or f.get("iconType") == "image")
                        if not is_image and not (isinstance(content, str) and content.startswith("data:image/")):
                            cache_opened_file(req.session_id, fp, content)
                    except Exception:
                        pass
                    yield _sse("opened_file", ev)
                elif ev_type == "text":
                    if await request.is_disconnected():
                        logger.warning("[query_stream] Client disconnected during text streaming")
                        agent.set_abort_flag(req.session_id)
                        return
                    yield _sse("text", ev)
                elif ev_type == "done":
                    srcs = ev.get("sources") or []
                    if srcs:
                        preview = []
                        for s in srcs[:20]:
                            fp = s.get("file_path") or ""
                            fn = s.get("file_name") or os.path.basename(fp) or ""
                            ftype = s.get("type") or _icon_type_for_path(fp) if fp else "doc"
                            preview.append(
                                {
                                    "id": _stable_id(f"file:{fp or fn}"),
                                    "file_name": fn,
                                    "file_path": fp,
                                    "type": ftype,
                                    "iconType": ftype,
                                    "doc_category": s.get("doc_category", ""),
                                    "doc_summary": s.get("doc_summary", ""),
                                }
                            )
                        yield _sse("files", {"type": "files", "total": len(preview), "preview": preview, "all": preview})

                    yield _sse("done", ev)
            _debug("C", "api_server.py:query_stream_done", "finished agent.query_stream loop")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"[query_stream] stream failed: {e}\n{tb}")
            _debug("A", "api_server.py:query_stream_error", "exception during generation", {"error": str(e), "tb": tb})
            yield _sse("done", {"type": "done", "ok": False, "query_type": "error", "error": str(e)})
        finally:
            try:
                import time
                time.sleep(0.15)
                agent = _ensure_agent()
                agent.clear_abort_flag(req.session_id)
                logger.info(f"[query_stream] 已清除session中断标志: {req.session_id}")
            except Exception as e:
                logger.error(f"[query_stream] 清除中断标志失败: {e}")
            logger.info(f"[Chat] /api/query_stream finished session_id={req.session_id or '<none>'}")

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def main() -> None:
    import signal
    import sys
    import os

    _acquire_backend_singleton_lock()

    def delayed_resume():
        import time
        logger.info("[Server] Waiting 5 seconds before resuming index jobs to allow UI to render...")
        time.sleep(5)
        try:
            _resume_index_job_if_needed()
        except Exception as e:
            logger.error(f"[Server] Failed to resume index job: {e}")
            
    threading.Thread(target=delayed_resume, daemon=True).start()
    
    def signal_handler(sig, frame):
        logger.info(f"[Server] Received signal {sig}, shutting down immediately...")
        os._exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    def _warmup_models():
        import time
        time.sleep(2)
        raw_disable = (os.getenv("FILEAGENT_DISABLE_WARMUP") or "").strip().lower()
        if raw_disable in {"1", "true", "yes", "y", "on"}:
            logger.info("[Server] 🧊 Warmup disabled (FILEAGENT_DISABLE_WARMUP=true).")
            return

        # Check dev mode preference from settings (env var)
        from config import settings
        if settings.DEV_NO_MODEL_LOAD:
            logger.info("[Server] 🚧 Dev mode enabled (DEV_NO_MODEL_LOAD=true): Skipping local model loading.")
            return
        try:
            _, pref_manager, _, _, _ = _get_managers()
            raw_onboarding = pref_manager.get("onboarding_complete", None) if hasattr(pref_manager, "get") else None
            onboarding_complete = str(raw_onboarding).strip().lower() in {"1", "true", "yes", "y", "on"}
            raw_step = pref_manager.get("onboarding_step", None) if hasattr(pref_manager, "get") else None
            onboarding_step = str(raw_step or "").strip().lower()
        except Exception:
            onboarding_complete = False
            onboarding_step = ""
        if (not onboarding_complete) or (onboarding_step and onboarding_step != "complete"):
            logger.info("[Server] ⏸️ Warmup skipped: onboarding not completed step.")
            return

        logger.info("[Server] Checking embedding/reranker models in background...")
        try:
            from utils.download_utils import ensure_gguf_downloaded
            ensure_gguf_downloaded(
                getattr(settings, "EMBEDDING_REPO_ID", settings.EMBEDDING_MODEL),
                os.path.basename(settings.LOCAL_EMBEDDING_MODEL_PATH),
                settings.LOCAL_EMBEDDING_MODEL_PATH,
            )
            if not getattr(settings, "RERANKER_OPTIONAL", False):
                ensure_gguf_downloaded(settings.RERANKER_MODEL, settings.RERANKER_GGUF_FILE, settings.LOCAL_RERANKER_MODEL_PATH)
            logger.info("[Server] Embedding models check complete.")
        except Exception as e:
            logger.error(f"[Server] Model warmup failed: {e}")
        
        try:
            _, pref_manager, llm_manager, _, _ = _get_managers()
            selected_id = pref_manager.get_selected_model_id()
            if selected_id:
                logger.info(f"[Server] Found configured model '{selected_id}', checking and starting if needed...")
                qf = pref_manager.get_selected_quantization_file(selected_id)
                llm_manager.start_server(selected_id, preferred_quantization_file=qf)
            else:
                logger.info("[Server] No specific model configured by user. Doing nothing for LLM yet.")
        except Exception as e:
            logger.error(f"[Server] LLM warmup failed: {e}")

        raw_preload = (os.getenv("FILEAGENT_PRELOAD_AGENT") or "").strip().lower()
        if raw_preload in {"1", "true", "yes", "y", "on"}:
            logger.info("[Server] Preloading FileAgent...")
            try:
                agent = _ensure_agent()
                try:
                    kb = getattr(agent, "kb", None)
                    if kb is not None and hasattr(kb, "request_query_cache_prewarm"):
                        kb.request_query_cache_prewarm(background=True, reason="startup_preload")
                except Exception as e:
                    logger.warning(f"[Server] KB prewarm request failed during preload: {e}")
                logger.info("[Server] ✅ FileAgent preloaded successfully")
            except Exception as e:
                logger.error(f"[Server] FileAgent preload failed: {e}")

    threading.Thread(target=_warmup_models, daemon=True).start()

    import uvicorn

    _uvi_level = (os.getenv("FILEAGENT_UVICORN_LOG_LEVEL") or "info").strip().lower()
    if _uvi_level not in ("critical", "error", "warning", "info", "debug", "trace"):
        _uvi_level = "info"

    try:
        import argparse

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--host", type=str, default=None)
        parser.add_argument("--port", type=int, default=None)
        args, _unknown = parser.parse_known_args()
        cli_host = args.host
        cli_port = args.port
    except Exception:
        cli_host = None
        cli_port = None

    port = int(cli_port if cli_port is not None else os.getenv("FILEAGENT_BACKEND_PORT", "17831"))
    host = str(cli_host if cli_host is not None else os.getenv("FILEAGENT_BACKEND_HOST", "127.0.0.1"))

    if port == 0:
        import asyncio
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        sock.bind((host, 0))
        try:
            sock.listen(2048)
            sock.setblocking(False)
        except Exception:
            pass
        actual_port = int(sock.getsockname()[1])
        _write_backend_port_file(host, actual_port)
        get_logger().info(
            f"FILEAGENT_BACKEND_LISTENING host={host} port={actual_port}"
        )

        config = uvicorn.Config(
            app, host=host, port=actual_port, log_level=_uvi_level
        )
        server = uvicorn.Server(config)
        asyncio.run(server.serve(sockets=[sock]))
    else:
        _write_backend_port_file(host, port)
        print(f"FILEAGENT_BACKEND_LISTENING host={host} port={port}")
        get_logger().info(f"FILEAGENT_BACKEND_LISTENING host={host} port={port}")
        uvicorn.run(app, host=host, port=port, log_level=_uvi_level)


if __name__ == "__main__":
    main()
