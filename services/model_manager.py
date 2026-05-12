import os
import json
import multiprocessing
import threading
import time
from contextlib import contextmanager
from typing import List, Dict, Optional, Any

from utils.logger import get_child_logger

try:
    from modelscope.hub.file_download import model_file_download
    HAS_MODELSCOPE = True
except ImportError:
    HAS_MODELSCOPE = False

try:
    from huggingface_hub import hf_hub_download
    HAS_HF = True
except ImportError:
    HAS_HF = False

logger = get_child_logger(__name__)

DEFAULT_UI_MODEL_IDS: List[str] = [
    "qwen3-4b-gguf",
    "qwen3-vl-2b-instruct-gguf",
    "qwen3.5-4b-vl-gguf",
    "glm-4.6v-flash-gguf",
    "gemma-3-12b-it-gguf",
    "gemma-4-E2B-it-gguf",
    "gemma-4-e4b-it-gguf",
    "gemma-4-26B-A4B-it-gguf",
    "qwen3.5-4b-gguf",
    "llama-3.2-3b-instruct-gguf",
    "llama-3.1-8b-instruct-q4-km-gguf",
    "deepseek-r1-distill-qwen-7b-gguf",
    "gpt-oss-20b-gguf",
    "ministral-3-8b-instruct-2512-gguf",
    "ministral-3-14b-instruct-2512-gguf",
    "ministral-3-3b-instruct-2512-gguf",
    "qwen3.5-0.8b-gguf",
    "qwen3-1.7b-gguf",
    "qwen3.5-9b-gguf",
]


def _download_gguf_in_subprocess(
    source: str,
    repo_id: str,
    filename: str,
    local_dir: str,
    ms_domain: Optional[str],
    expected_size: int,
    result_queue: "multiprocessing.Queue[Dict[str, Any]]",
) -> None:
    try:
        try:
            from services.download_utils import (
                ResumableDownloadCancelled,
                ResumableDownloadInterrupted,
                ResumableDownloadUnavailable,
                download_gguf_with_resume,
            )
        except Exception:
            download_gguf_with_resume = None  # type: ignore[assignment]

        if download_gguf_with_resume is not None:
            try:
                target_path = os.path.join(local_dir, filename)
                ok = download_gguf_with_resume(
                    source,
                    repo_id,
                    filename,
                    target_path,
                    expected_size=int(expected_size or 0),
                    modelscope_domain=ms_domain,
                )
                if ok:
                    result_queue.put({"ok": True, "result_path": target_path})
                    return
            except ResumableDownloadUnavailable:
                pass
            except (ResumableDownloadCancelled, ResumableDownloadInterrupted) as e:
                result_queue.put({"ok": False, "error": str(e)})
                return

        if source == "modelscope":
            old_domain = os.environ.get("MODELSCOPE_DOMAIN")
            if ms_domain:
                os.environ["MODELSCOPE_DOMAIN"] = ms_domain
            try:
                result_path = model_file_download(model_id=repo_id, file_path=filename, local_dir=local_dir)
            finally:
                if ms_domain:
                    if old_domain is None:
                        os.environ.pop("MODELSCOPE_DOMAIN", None)
                    else:
                        os.environ["MODELSCOPE_DOMAIN"] = old_domain
        elif source == "hf":
            result_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        else:
            raise ValueError(f"Unknown source: {source}")

        result_queue.put({"ok": True, "result_path": str(result_path or "")})
    except Exception as e:
        result_queue.put({"ok": False, "error": str(e)})

class ModelManager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        cfg = (os.getenv("FILEAGENT_SUPPORTED_MODELS_PATH") or "").strip()
        if cfg:
            self.config_path = os.path.abspath(os.path.expanduser(cfg))
        else:
            candidate = os.path.join(base_dir, "config", "supported_models.json")
            legacy = os.path.join(base_dir, "supported_models.json")
            self.config_path = candidate if os.path.exists(candidate) else legacy
        data_dir = (os.getenv("FILEAGENT_DATA_DIR") or "").strip()
        data_dir = os.path.abspath(os.path.expanduser(data_dir)) if data_dir else ""
        models_dir = (os.getenv("FILEAGENT_LOCAL_MODELS_DIR") or "").strip()
        if not models_dir and data_dir:
            models_dir = os.path.join(data_dir, "local_models")
        self.models_dir = os.path.abspath(os.path.expanduser(models_dir)) if models_dir else os.path.join(base_dir, "local_models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.download_jobs: Dict[str, Dict[str, Any]] = {} # model_id -> status
        self._lock = threading.Lock()
        self._run_seq = 0

    @staticmethod
    def _is_transient_path(path: str) -> bool:
        p = str(path or "").replace("\\", "/").lower()
        if any(seg in p for seg in ("/._____temp/", "/.msc/", "/.mv/", "/.unfoldly_runs/", "/.unfoldly_downloads/")):
            return True
        if p.endswith((".downloading", ".tmp", ".part", ".incomplete")):
            return True
        return False

    @staticmethod
    def _expected_size_from_model_config(model_config: Dict[str, Any], filename: str, model_dir: str = "") -> int:
        """Best-effort expected file size in bytes from supported_models.json metadata."""
        try:
            for q in (model_config.get("quantizations") or []):
                if isinstance(q, dict) and str(q.get("file") or "") == str(filename):
                    sb = q.get("size_bytes")
                    if isinstance(sb, int) and sb > 0:
                        return int(sb)
        except Exception:
            pass
        try:
            fs = model_config.get("file_sizes") or {}
            sb = fs.get(filename) if isinstance(fs, dict) else None
            if isinstance(sb, int) and sb > 0:
                return int(sb)
        except Exception:
            pass
            
        if model_dir:
            try:
                meta_path = os.path.join(model_dir, ".unfoldly_meta.json")
                if os.path.isfile(meta_path):
                    import json as _json
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = _json.load(f)
                    sz = meta.get("file_sizes", {}).get(filename)
                    if isinstance(sz, int) and sz > 0:
                        return sz
            except Exception:
                pass
                
        return 0

    @staticmethod
    def _preferred_modelscope_domain(model_id: str) -> Optional[str]:
        """Per-model domain override for ModelScope downloads."""
        mid = (model_id or "").lower()
        if "ministral" in mid:
            return "www.modelscope.cn"
        if model_id in {
            "qwen3-vl-2b-instruct-gguf",
            "qwen3-4b-gguf",
            "qwen3.5-4b-gguf",
            "qwen3.5-4b-vl-gguf",
            "llama-3.2-3b-instruct-gguf",
            "llama-3.1-8b-instruct-q4-km-gguf",
            "deepseek-r1-distill-qwen-7b-gguf",
        }:
            return "www.modelscope.cn"
        return None

    @staticmethod
    @contextmanager
    def _temp_modelscope_domain(domain: Optional[str]):
        if not domain:
            yield
            return
        old = os.environ.get("MODELSCOPE_DOMAIN")
        os.environ["MODELSCOPE_DOMAIN"] = domain
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("MODELSCOPE_DOMAIN", None)
            else:
                os.environ["MODELSCOPE_DOMAIN"] = old

    @staticmethod
    def _find_gguf_in_tree(model_dir: str, filename: str, min_size: int = 1_000_000) -> Optional[str]:
        """Search model_dir recursively for `filename` with size >= min_size."""
        try:
            for root, _dirs, files in os.walk(model_dir):
                if filename in files:
                    p = os.path.join(root, filename)
                    try:
                        if ModelManager._is_transient_path(p):
                            continue
                        if os.path.getsize(p) >= min_size:
                            return p
                    except OSError:
                        pass
        except Exception:
            pass
        return None

    @staticmethod
    def _is_gguf_complete(
        path: str,
        expected_size: int = 0,
        *,
        strict_expected: bool = False,
        min_size: int = 1_000_000,
        allow_transient: bool = False,
    ) -> bool:
        """
        Validate GGUF completeness by header + size.
        - strict_expected=True: require expected_size > 0 and size >= 99% expected.
          Used for pre-download skip checks, to avoid mistaking partial files as complete.
        - strict_expected=False: if expected size is unknown, fall back to header + min size.
        """
        try:
            if not os.path.isfile(path):
                return False
            if (not allow_transient) and ModelManager._is_transient_path(path):
                return False
            sz = int(os.path.getsize(path))
            if sz < int(min_size):
                return False
            with open(path, "rb") as f:
                if f.read(4) != b"GGUF":
                    return False
            
            if expected_size > 0:
                if strict_expected:
                    return sz >= int(expected_size * 0.99)
                return sz >= int(expected_size * 0.99)
                
            return True
        except Exception:
            return False

    @staticmethod
    def _ensure_gguf_at_target(result_path: Optional[str], model_dir: str, filename: str, min_size: int = 1_000_000) -> bool:
        """Ensure downloaded GGUF file ends up at model_dir/filename."""
        import shutil
        target = os.path.join(model_dir, filename)
        if os.path.isfile(target) and os.path.getsize(target) >= int(min_size):
            return True
        sources = []
        if result_path and os.path.isfile(str(result_path)) and os.path.getsize(str(result_path)) >= int(min_size):
            sources.append(str(result_path))
        try:
            for root, _dirs, files in os.walk(model_dir):
                if filename in files:
                    p = os.path.join(root, filename)
                    if ModelManager._is_transient_path(p):
                        continue
                    if p not in sources and os.path.getsize(p) >= int(min_size):
                        sources.append(p)
        except Exception:
            pass
        for src in sources:
            try:
                if os.path.abspath(src) != os.path.abspath(target):
                    shutil.copy2(src, target)
                if os.path.isfile(target) and os.path.getsize(target) >= int(min_size):
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _cleanup_path(path: str) -> None:
        import shutil

        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    @staticmethod
    def _cleanup_completed_stage_artifacts(stage_dir: str, filename: str) -> None:
        """Reclaim completed staging files without deleting resumable .part files."""
        try:
            target = os.path.join(stage_dir, filename)
            if os.path.isfile(target):
                os.remove(target)
        except Exception:
            pass
        for subdir in ("._____temp", ".msc", ".mv"):
            try:
                ModelManager._cleanup_path(os.path.join(stage_dir, subdir))
            except Exception:
                pass

    def get_supported_models(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.config_path):
            return []
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                models = json.load(f)
            
            for model in models:
                model_id = model["id"]
                model_dir = os.path.join(self.models_dir, model_id)
                
                base_files = list(model.get("files", []) or [])
                quant_files: List[str] = []
                try:
                    for q in (model.get("quantizations") or []):
                        if isinstance(q, dict) and q.get("file"):
                            quant_files.append(str(q["file"]))
                except Exception:
                    pass
                if model.get("default_quantization"):
                    quant_files.append(str(model.get("default_quantization")))

                seen = set()
                quant_files = [x for x in quant_files if not (x in seen or seen.add(x))]

                is_installed = False
                installed_quantizations: List[str] = []
                if os.path.exists(model_dir):
                    has_base = all(os.path.exists(os.path.join(model_dir, f)) for f in base_files if f)
                    if has_base:
                        if quant_files:
                            for qf in quant_files:
                                if not qf:
                                    continue
                                target_qf = os.path.join(model_dir, qf)
                                # Determine the exact expected size using the most reliable
                                # source available, in priority order:
                                #   1. .size.json (written at download completion) — exact,
                                #      offline, most trustworthy.
                                #   2. supported_models.json size_bytes — static config,
                                #      offline, always available.
                                #   3. ModelScope API — authoritative remote source; cached
                                #      in-process so only one network call per model per run.
                                #      Only queried when the two offline sources disagree or
                                #      are both missing.
                                # No percentage tolerance: exact byte match required when we
                                # have an expected size, eliminating the 99%/97% window that
                                # caused the "99.9% → restart" loop.
                                size_hint = 0
                                try:
                                    from services.download_utils import _read_gguf_size_hint
                                    size_hint = _read_gguf_size_hint(target_qf)
                                except Exception:
                                    pass
                                config_sz = self._expected_size_from_model_config(model, qf, model_dir=model_dir)
                                expected_sz = max(size_hint, config_sz)

                                # If offline sources give us a confident answer, skip the
                                # network call entirely (size.json present = file was
                                # downloaded by us and the size is authoritative).
                                if expected_sz == 0:
                                    # Neither offline source has size info — try ModelScope API.
                                    # repo_id lives under sources.modelscope.repo_id or sources.hf.repo_id.
                                    # The result is cached so this is at most one HTTP call
                                    # per (model_id, qf) pair per process lifetime.
                                    try:
                                        from services.download_utils import _query_gguf_file_size
                                        _sources = model.get("sources") or {}
                                        _repo_id = (
                                            (_sources.get("modelscope") or {}).get("repo_id")
                                            or (_sources.get("hf") or {}).get("repo_id")
                                            or model.get("model_id")
                                            or model.get("repo_id")
                                            or model_id
                                        )
                                        remote_sz = _query_gguf_file_size(str(_repo_id), qf)
                                        if remote_sz > 0:
                                            expected_sz = remote_sz
                                            logger.info(f"[model_manager] remote size for {qf}: {remote_sz} bytes")
                                    except Exception:
                                        pass

                                if expected_sz > 0:
                                    min_required = expected_sz  # exact match required
                                else:
                                    min_required = 1_000_000    # no size info at all: >1 MB floor
                                if (
                                    os.path.isfile(target_qf)
                                    and (not self._is_transient_path(target_qf))
                                    and os.path.getsize(target_qf) >= min_required
                                ):
                                    installed_quantizations.append(qf)
                                elif self._find_gguf_in_tree(model_dir, qf, min_size=min_required):
                                    self._ensure_gguf_at_target(None, model_dir, qf, min_size=min_required)
                                    if (
                                        os.path.isfile(target_qf)
                                        and (not self._is_transient_path(target_qf))
                                        and os.path.getsize(target_qf) >= min_required
                                    ):
                                        installed_quantizations.append(qf)
                            is_installed = len(installed_quantizations) > 0
                        else:
                            is_installed = True
                
                model["installed"] = is_installed
                if installed_quantizations:
                    model["installed_quantizations"] = installed_quantizations
                
                with self._lock:
                    if model_id in self.download_jobs:
                        job = self.download_jobs[model_id]
                        model["status"] = job["status"] # downloading, error, installed
                        model["progress"] = job.get("progress", 0)
                        model["downloaded_bytes"] = job.get("downloaded_bytes", 0)
                        model["total_bytes"] = job.get("total_bytes", 0)
                        model["download_speed"] = job.get("download_speed", 0)
                        model["eta_seconds"] = job.get("eta_seconds")
                        model["error"] = job.get("error")
                        model["downloadingSource"] = job.get("source")
                        model["downloading_quantization_file"] = job.get("quantization_file")
                    else:
                        model["status"] = "installed" if is_installed else "available"
            
            return models
        except Exception as e:
            logger.error(f"Failed to load supported models: {e}")
            return []

    def download_model(self, model_id: str, source: str = "auto", quantization_file: Optional[str] = None) -> Dict[str, Any]:
        source = str(source or "auto").strip().lower() or "auto"
        models = self.get_supported_models()
        target_model = next((m for m in models if m["id"] == model_id), None)
        if not target_model:
            return {"ok": False, "error": "Model not found"}

        qf = (quantization_file or "").strip()
        if not qf:
            qf = (target_model.get("default_quantization") or "").strip()
        has_quantizations = bool(
            target_model.get("default_quantization")
            or (target_model.get("quantizations") and len(target_model.get("quantizations", [])) > 0)
        )
        if not qf and has_quantizations:
            return {"ok": False, "error": "No quantization file configured"}

        if target_model.get("installed"):
            if not qf or qf in target_model.get("installed_quantizations", []):
                logger.info(f"Model {model_id} (quant={qf}) is already fully installed locally. Skipping download.")
                return {"ok": True, "job_id": model_id, "status": "installed"}

        run_id = 0
        with self._lock:
            if model_id in self.download_jobs and self.download_jobs[model_id]["status"] == "downloading":
                return {"ok": False, "error": "Download already in progress"}
            self._run_seq += 1
            run_id = int(self._run_seq)
            
            self.download_jobs[model_id] = {
                "status": "downloading",
                "progress": 0,
                "source": source,
                "quantization_file": qf,
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "cancelled": False,
                "run_id": run_id,
            }

        thread = threading.Thread(target=self._download_worker, args=(target_model, source, qf, run_id))
        thread.start()
        
        return {"ok": True, "job_id": model_id}

    def cancel_download(self, model_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self.download_jobs.get(model_id)
            if job and job.get("status") == "downloading":
                self._run_seq += 1
                job["cancelled"] = True
                job["status"] = "cancelled"
                job["run_id"] = int(self._run_seq)
                job["download_speed"] = 0
                job["eta_seconds"] = None
                return {"ok": True}
        return {"ok": False, "error": "No active download found"}

    @staticmethod
    def _query_repo_total_size(repo_id: str, source: str) -> int:
        try:
            if source == "hf":
                from huggingface_hub import HfApi
                info = HfApi().model_info(repo_id, files_metadata=True)
                total = 0
                for sib in (info.siblings or []):
                    sz = getattr(sib, "size", None) or getattr(sib, "lfs", {})
                    if isinstance(sz, int):
                        total += sz
                    elif isinstance(sz, dict):
                        total += int(sz.get("size", 0))
                if total > 0:
                    return total
                used = getattr(info, "usedStorage", None)
                if isinstance(used, int) and used > 0:
                    return used
            else:
                import urllib.request
                import json as _json
                domain = os.environ.get("MODELSCOPE_DOMAIN", "www.modelscope.cn")
                url = f"https://{domain}/api/v1/models/{repo_id}/repo/files?Recursive=true"
                req = urllib.request.Request(url, headers={"User-Agent": "fileagent/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = _json.loads(resp.read().decode("utf-8"))
                total = 0
                for f in (data.get("Data", {}).get("Files", []) or []):
                    total += int(f.get("Size", 0))
                if total > 0:
                    return total
        except Exception as e:
            logger.debug(f"Failed to query repo size for {repo_id} ({source}): {e}")
        return 0

    @staticmethod
    def _query_file_size(repo_id: str, source: str, filename: str) -> int:
        try:
            if source == "hf":
                from huggingface_hub import HfApi
                info = HfApi().model_info(repo_id, files_metadata=True)
                for sib in (info.siblings or []):
                    if getattr(sib, "rfilename", "") == filename:
                        sz = getattr(sib, "size", None) or getattr(sib, "lfs", {})
                        if isinstance(sz, int) and sz > 0:
                            return sz
                        if isinstance(sz, dict):
                            return int(sz.get("size", 0))
            else:
                import urllib.request
                import json as _json
                domain = os.environ.get("MODELSCOPE_DOMAIN", "www.modelscope.cn")
                url = f"https://{domain}/api/v1/models/{repo_id}/repo/files?Recursive=true"
                req = urllib.request.Request(url, headers={"User-Agent": "fileagent/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = _json.loads(resp.read().decode("utf-8"))
                for f in (data.get("Data", {}).get("Files", []) or []):
                    if f.get("Name") == filename or f.get("Path") == filename:
                        sz = int(f.get("Size", 0))
                        if sz > 0:
                            return sz
        except Exception as e:
            logger.debug(f"Failed to query file size for {repo_id}/{filename} ({source}): {e}")
        return 0

    @staticmethod
    def _has_valid_gguf_header(path: str, min_size: int = 1_000_000, *, allow_transient: bool = False) -> bool:
        try:
            if not os.path.isfile(path):
                return False
            if (not allow_transient) and ModelManager._is_transient_path(path):
                return False
            if int(os.path.getsize(path)) < int(min_size):
                return False
            with open(path, "rb") as f:
                return f.read(4) == b"GGUF"
        except Exception:
            return False

    @staticmethod
    def _runtime_required_files(model_config: Dict[str, Any], quantization_file: Optional[str]) -> List[str]:
        """Files that must exist on a source for a GGUF download to be equivalent."""
        required: List[str] = []
        try:
            for f in (model_config.get("files") or []):
                item = str(f or "").strip()
                if item.lower().endswith(".gguf"):
                    required.append(item)
        except Exception:
            pass
        qf = str(quantization_file or "").strip()
        if qf:
            required.append(qf)

        seen = set()
        return [x for x in required if x and not (x in seen or seen.add(x))]

    @staticmethod
    def _source_item_exists(remote_files: set, item: str) -> bool:
        item = str(item or "").strip().strip("/")
        if not item:
            return False
        if item in remote_files:
            return True
        prefix = f"{item}/"
        return any(str(f).startswith(prefix) for f in remote_files)

    @staticmethod
    def _list_source_file_metadata(
        source: str,
        repo_id: str,
        ms_domain: Optional[str] = None,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        try:
            if source == "hf":
                from huggingface_hub import HfApi
                endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
                info = HfApi(endpoint=endpoint).model_info(repo_id, files_metadata=True)
                files: Dict[str, Dict[str, Any]] = {}
                for sib in (info.siblings or []):
                    path = str(getattr(sib, "rfilename", "") or "").strip().strip("/")
                    if not path:
                        continue
                    size = getattr(sib, "size", None)
                    sha256 = ""
                    lfs = getattr(sib, "lfs", None)
                    if lfs is not None:
                        size = getattr(lfs, "size", size)
                        sha256 = str(getattr(lfs, "sha256", "") or "").lower()
                    files[path] = {"size": int(size or 0), "sha256": sha256}
                return files

            if source == "modelscope":
                import urllib.request
                import json as _json
                candidates = [
                    ms_domain,
                    os.environ.get("MODELSCOPE_DOMAIN"),
                    "www.modelscope.cn",
                    "www.modelscope.ai",
                ]
                domains = []
                for d in candidates:
                    domain = str(d or "").strip()
                    if not domain:
                        continue
                    domain = domain.replace("https://", "").replace("http://", "").strip("/")
                    if domain and domain not in domains:
                        domains.append(domain)

                last_err = None
                for domain in domains:
                    try:
                        url = f"https://{domain}/api/v1/models/{repo_id}/repo/files?Recursive=true"
                        req = urllib.request.Request(url, headers={"User-Agent": "fileagent/1.0"})
                        with urllib.request.urlopen(req, timeout=12) as resp:
                            data = _json.loads(resp.read().decode("utf-8"))
                        files: Dict[str, Dict[str, Any]] = {}
                        for f in (data.get("Data", {}).get("Files", []) or []):
                            path = str(f.get("Path") or f.get("Name") or "").strip().strip("/")
                            if path:
                                files[path] = {
                                    "size": int(f.get("Size") or 0),
                                    "sha256": str(f.get("Sha256") or "").lower(),
                                }
                        if files:
                            return files
                    except Exception as e:
                        last_err = e
                if last_err:
                    raise last_err
        except Exception as e:
            logger.debug(f"Failed to list files for {source}:{repo_id}: {e}")
        return None

    @staticmethod
    def _list_source_files(source: str, repo_id: str, ms_domain: Optional[str] = None) -> Optional[set]:
        metadata = ModelManager._list_source_file_metadata(source, repo_id, ms_domain=ms_domain)
        return set(metadata.keys()) if metadata is not None else None

    @staticmethod
    def _source_matches_modelscope_runtime_files(
        model_config: Dict[str, Any],
        source: str,
        repo_id: str,
        quantization_file: Optional[str],
    ) -> bool:
        """Only allow an alternate source when runtime GGUF/mmproj bytes match ModelScope."""
        if source == "modelscope":
            return True

        required = ModelManager._runtime_required_files(model_config, quantization_file)
        if not required:
            return True

        sources = model_config.get("sources") or {}
        ms_repo = str((sources.get("modelscope") or {}).get("repo_id") or "").strip()
        if not ms_repo:
            return True

        model_id = str(model_config.get("id") or "")
        ms_domain = ModelManager._preferred_modelscope_domain(model_id)
        ms_meta = ModelManager._list_source_file_metadata("modelscope", ms_repo, ms_domain=ms_domain)
        alt_meta = ModelManager._list_source_file_metadata(source, repo_id)
        if ms_meta is None or alt_meta is None:
            logger.info(
                f"[Model] Skip source={source} repo={repo_id}; cannot verify metadata against ModelScope"
            )
            return False

        mismatches = []
        for filename in required:
            ms_file = ms_meta.get(filename)
            alt_file = alt_meta.get(filename)
            if not ms_file or not alt_file:
                mismatches.append({"file": filename, "reason": "missing"})
                continue
            same_size = int(ms_file.get("size") or 0) == int(alt_file.get("size") or 0)
            ms_sha = str(ms_file.get("sha256") or "").lower()
            alt_sha = str(alt_file.get("sha256") or "").lower()
            same_sha = bool(ms_sha and alt_sha and ms_sha == alt_sha)
            if not (same_size and same_sha):
                mismatches.append({
                    "file": filename,
                    "reason": "different_sha_or_size",
                    "modelscope_size": int(ms_file.get("size") or 0),
                    "source_size": int(alt_file.get("size") or 0),
                })

        if mismatches:
            logger.info(
                f"[Model] Skip source={source} repo={repo_id}; runtime files differ from ModelScope: {mismatches}"
            )
            return False

        return True

    @staticmethod
    def _source_has_runtime_files(
        model_config: Dict[str, Any],
        source: str,
        repo_id: str,
        quantization_file: Optional[str],
    ) -> bool:
        required = ModelManager._runtime_required_files(model_config, quantization_file)
        if not required:
            return True
        ms_domain = ModelManager._preferred_modelscope_domain(str(model_config.get("id") or "")) if source == "modelscope" else None
        remote_files = ModelManager._list_source_files(source, repo_id, ms_domain=ms_domain)
        if remote_files is None:
            return True
        missing = [f for f in required if not ModelManager._source_item_exists(remote_files, f)]
        if missing:
            logger.info(f"[Model] Skip source={source} repo={repo_id}; missing runtime files: {missing}")
            return False
        return True

    @staticmethod
    def _source_probe_url(source: str, repo_id: str, filename: Optional[str], ms_domain: Optional[str] = None) -> str:
        import urllib.parse

        if source == "hf":
            endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
            if filename:
                return f"{endpoint}/{repo_id}/resolve/main/{urllib.parse.quote(filename, safe='/')}"
            return f"{endpoint}/api/models/{repo_id}"

        if source == "modelscope":
            domain = (ms_domain or os.environ.get("MODELSCOPE_DOMAIN", "www.modelscope.cn")).strip()
            domain = domain.replace("https://", "").replace("http://", "").strip("/")
            if filename:
                return (
                    f"https://{domain}/api/v1/models/{repo_id}/repo?"
                    f"Revision=master&FilePath={urllib.parse.quote(filename, safe='')}"
                )
            return f"https://{domain}/api/v1/models/{repo_id}"

        return ""

    @staticmethod
    def _test_source_speed(
        source: str,
        repo_id: str,
        probe_filename: Optional[str] = None,
        *,
        ms_domain: Optional[str] = None,
    ) -> Dict[str, float]:
        """Probe source latency/throughput with a tiny ranged read."""
        import urllib.request
        import time
        from urllib.error import HTTPError

        if source not in {"modelscope", "hf"}:
            return {"ok": 0.0, "latency_ms": -1.0, "bytes_per_sec": 0.0}

        try:
            probe_bytes = int(os.getenv("FILEAGENT_SOURCE_PROBE_BYTES", "262144") or 262144)
        except Exception:
            probe_bytes = 262144
        probe_bytes = max(16 * 1024, min(probe_bytes, 1024 * 1024))

        url = ModelManager._source_probe_url(source, repo_id, probe_filename, ms_domain=ms_domain)
        headers = {"User-Agent": "fileagent/1.0"}
        if probe_filename:
            headers["Range"] = f"bytes=0-{probe_bytes - 1}"

        req = urllib.request.Request(url, headers=headers, method="GET")
        start_t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                read_bytes = 0
                while read_bytes < probe_bytes:
                    chunk = resp.read(min(64 * 1024, probe_bytes - read_bytes))
                    if not chunk:
                        break
                    read_bytes += len(chunk)
                    if not probe_filename:
                        break
            elapsed = max(time.time() - start_t, 0.001)
            latency_ms = elapsed * 1000
            bps = (read_bytes / elapsed) if read_bytes > 0 else 0.0
            return {"ok": 1.0, "latency_ms": latency_ms, "bytes_per_sec": bps}
        except HTTPError as e:
            if e.code in [401, 403]:
                return {"ok": 1.0, "latency_ms": (time.time() - start_t) * 1000, "bytes_per_sec": 0.0}
            return {"ok": 0.0, "latency_ms": -1.0, "bytes_per_sec": 0.0}
        except Exception:
            return {"ok": 0.0, "latency_ms": -1.0, "bytes_per_sec": 0.0}

    def _get_fallback_sources(self, model_config: Dict[str, Any], quantization_file: Optional[str] = None) -> List[str]:
        sources_dict = model_config.get("sources") or {}
        available_sources = [
            src for src in sources_dict.keys()
            if self._source_has_runtime_files(
                model_config,
                src,
                str((sources_dict.get(src) or {}).get("repo_id") or ""),
                quantization_file,
            )
            and self._source_matches_modelscope_runtime_files(
                model_config,
                src,
                str((sources_dict.get(src) or {}).get("repo_id") or ""),
                quantization_file,
            )
        ]
        if not available_sources:
            available_sources = list(sources_dict.keys()) or ["modelscope"]
        if len(available_sources) == 1:
            return available_sources
            
        import concurrent.futures

        probe_files = self._runtime_required_files(model_config, quantization_file)
        probe_filename = probe_files[-1] if probe_files else None
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(available_sources)) as executor:
            future_to_src = {
                executor.submit(
                    self._test_source_speed,
                    src,
                    str((sources_dict.get(src) or {}).get("repo_id") or ""),
                    probe_filename,
                    ms_domain=self._preferred_modelscope_domain(str(model_config.get("id") or "")) if src == "modelscope" else None,
                ): src
                for src in available_sources
            }
            for future in concurrent.futures.as_completed(future_to_src):
                src = future_to_src[future]
                try:
                    probe = future.result()
                    results.append({"source": src, **probe})
                    logger.info(
                        f"[Model] Speed test: source={src} latency={probe.get('latency_ms', -1):.2f}ms "
                        f"speed={probe.get('bytes_per_sec', 0):.0f}B/s"
                    )
                except Exception as e:
                    results.append({"source": src, "ok": 0.0, "latency_ms": -1.0, "bytes_per_sec": 0.0})
                    
        valid = [r for r in results if r.get("ok", 0.0) > 0 and r.get("latency_ms", -1.0) >= 0]
        valid.sort(key=lambda x: (0 if x.get("bytes_per_sec", 0) > 0 else 1, -x.get("bytes_per_sec", 0), x.get("latency_ms", 999999)))
        sorted_sources = [r["source"] for r in valid]
        sorted_sources.extend([src for src in available_sources if src not in sorted_sources])
        
        if not sorted_sources:
            sorted_sources = available_sources
            
        logger.info(f"[Model] Auto fallback order: {sorted_sources}")
        return sorted_sources

    def _download_worker(self, model_config: Dict[str, Any], source: str, quantization_file: str, run_id: int):
        model_id = model_config["id"]
        sources_to_try = [source]
        
        if source == "auto":
            sources_to_try = self._get_fallback_sources(model_config, quantization_file)
            
        last_error = None
        
        for src in sources_to_try:
            try:
                with self._lock:
                    job = self.download_jobs.get(model_id)
                    if job and not job.get("cancelled"):
                        job["source"] = src
                
                self._download_worker_impl(model_config, src, quantization_file, run_id)
                return
            except Exception as e:
                with self._lock:
                    job = self.download_jobs.get(model_id)
                    if job and job.get("cancelled"):
                        return
                last_error = e
                logger.warning(f"[Model] Download worker failed for source '{src}': {e}. Trying next...")
                import time
                time.sleep(1)
                
        with self._lock:
            job = self.download_jobs.get(model_id)
            if job and not job.get("cancelled"):
                job["status"] = "error"
                job["error"] = str(last_error) if last_error else "All download sources failed"
        logger.error(f"[Model] Download job {model_id} completely failed.")

    def _download_worker_impl(self, model_config: Dict[str, Any], source: str, quantization_file: str, run_id: int):
        model_id = model_config["id"]
        model_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        run_stage_root = os.path.join(model_dir, ".unfoldly_runs", f"run_{run_id}")
        os.makedirs(run_stage_root, exist_ok=True)
        ms_domain = self._preferred_modelscope_domain(model_id) if source == "modelscope" else None

        def _is_current_job(job: Optional[Dict[str, Any]]) -> bool:
            return bool(job) and int(job.get("run_id") or 0) == int(run_id)

        def _is_cancelled_or_stale() -> bool:
            with self._lock:
                job = self.download_jobs.get(model_id)
                if not _is_current_job(job):
                    return True
                return bool(job.get("cancelled"))
        
        files_to_download = list(model_config.get("files", []) or [])
        if quantization_file:
            files_to_download.append(quantization_file)

        seen = set()
        files_to_download = [x for x in files_to_download if x and not (x in seen or seen.add(x))]
            
        repo_id = ""
        try:
            if source == "modelscope":
                if not HAS_MODELSCOPE:
                    raise ImportError("modelscope package not installed")

                repo_id = (model_config.get("sources") or {}).get("modelscope", {}).get("repo_id", "") or ""
            elif source == "hf":
                if not HAS_HF:
                    raise ImportError("huggingface_hub package not installed")
                repo_id = (model_config.get("sources") or {}).get("hf", {}).get("repo_id", "") or ""
            else:
                raise ValueError(f"Unknown source: {source}")

            if not repo_id:
                raise ValueError(f"Model '{model_id}' does not support download source: {source}")


            gguf_files = [f for f in files_to_download if str(f).lower().endswith(".gguf")]
            other_items = [f for f in files_to_download if f not in gguf_files]
            has_quantizations = bool(
                model_config.get("default_quantization")
                or (model_config.get("quantizations") and len(model_config.get("quantizations", [])) > 0)
            )

            # Some ModelScope mirrors include helper files such as
            # configuration.json or params that do not exist in the matching
            # HF GGUF repos. Runtime only needs the main GGUF and optional
            # mmproj, so skip missing non-GGUF helpers on source-specific repos.
            if other_items and (gguf_files or has_quantizations):
                remote_files = self._list_source_files(source, repo_id, ms_domain=ms_domain)
                if remote_files is not None:
                    filtered_other_items = []
                    skipped_other_items = []
                    for it in other_items:
                        item = str(it or "").strip()
                        if not item:
                            continue
                        if self._source_item_exists(remote_files, item):
                            filtered_other_items.append(it)
                        else:
                            skipped_other_items.append(item)
                    if skipped_other_items:
                        logger.info(
                            f"Skipping source-missing auxiliary files for {source}:{repo_id}: {skipped_other_items}"
                        )
                    other_items = filtered_other_items

            gguf_expected_sizes: Dict[str, int] = {}
            for gf in gguf_files:
                local_sz = self._expected_size_from_model_config(model_config, gf, model_dir=model_dir)
                remote_sz = self._query_file_size(repo_id, source, gf) if repo_id else 0
                sz = int(remote_sz or local_sz or 0)
                if remote_sz and local_sz and int(remote_sz) != int(local_sz):
                    logger.warning(
                        f"Override configured size with remote size for {gf}: "
                        f"config={int(local_sz)} remote={int(remote_sz)}"
                    )
                logger.info(
                    f"GGUF expected size resolved: file={gf} source={source} "
                    f"config={int(local_sz or 0)} remote={int(remote_sz or 0)} selected={int(sz or 0)}"
                )
                gguf_expected_sizes[gf] = sz

            other_expected_total = 0
            for it in other_items:
                item = str(it or "").strip()
                if not item:
                    continue
                if item.endswith("/") or ("." not in os.path.basename(item)):
                    continue
                sz = self._expected_size_from_model_config(model_config, item, model_dir=model_dir)
                if (not sz) and repo_id:
                    sz = self._query_file_size(repo_id, source, item)
                if sz > 0:
                    other_expected_total += int(sz)

            aggregate_total_bytes = sum(v for v in gguf_expected_sizes.values() if v > 0) + int(other_expected_total or 0)
            aggregate_completed_bytes = 0

            with self._lock:
                job = self.download_jobs.get(model_id)
                if _is_current_job(job):
                    if aggregate_total_bytes > 0:
                        job["total_bytes"] = int(aggregate_total_bytes)
                    job["downloaded_bytes"] = 0
                    job["progress"] = 0

            total_steps = 1 + (1 if other_items else 0)
            step_idx = 0

            for gf in gguf_files:
                target_path = os.path.join(model_dir, gf)
                gf_total_bytes = int(gguf_expected_sizes.get(gf) or 0)
                stage_dir = os.path.join(model_dir, ".unfoldly_downloads", os.path.basename(gf))
                os.makedirs(stage_dir, exist_ok=True)

                if self._is_gguf_complete(target_path, gf_total_bytes, strict_expected=True):
                    logger.info(f"GGUF {gf} already at {target_path}, skipping.")
                    local_sz = int(os.path.getsize(target_path))
                    completed = min(local_sz, gf_total_bytes) if gf_total_bytes > 0 else local_sz
                    aggregate_completed_bytes += max(0, completed)
                    with self._lock:
                        job = self.download_jobs.get(model_id)
                        if _is_current_job(job):
                            if aggregate_total_bytes > 0:
                                job["total_bytes"] = int(aggregate_total_bytes)
                                job["downloaded_bytes"] = int(min(aggregate_completed_bytes, aggregate_total_bytes))
                                job["progress"] = int(min(99, max(0, (job["downloaded_bytes"] / aggregate_total_bytes) * 100)))
                            else:
                                job["downloaded_bytes"] = int(aggregate_completed_bytes)
                                job["progress"] = int((step_idx / max(1, total_steps)) * 100)
                    step_idx += 1
                    continue

                total_bytes = gf_total_bytes

                with self._lock:
                    job = self.download_jobs.get(model_id)
                    if _is_current_job(job):
                        if aggregate_total_bytes > 0:
                            job["total_bytes"] = int(aggregate_total_bytes)
                        else:
                            job["total_bytes"] = int(total_bytes or 0)
                        job["downloaded_bytes"] = int(min(aggregate_completed_bytes, aggregate_total_bytes)) if aggregate_total_bytes > 0 else int(aggregate_completed_bytes)

                def _probe_downloaded_size() -> int:
                    candidates = [
                        os.path.join(stage_dir, gf),
                        os.path.join(stage_dir, f"{gf}.part"),
                        os.path.join(stage_dir, "._____temp", gf),
                        os.path.join(stage_dir, ".msc", gf),
                    ]
                    best = 0
                    for p in candidates:
                        try:
                            if os.path.exists(p):
                                best = max(best, int(os.path.getsize(p)))
                        except Exception:
                            pass
                    if best > 0:
                        return best

                    try:
                        for root, _dirs, files in os.walk(stage_dir):
                            for fn in files:
                                if fn == gf or fn.startswith(gf) or gf in fn:
                                    p = os.path.join(root, fn)
                                    try:
                                        best = max(best, int(os.path.getsize(p)))
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    return best

                stop_flag = {"stop": False}
                download_queue: "multiprocessing.Queue[Dict[str, Any]]" = multiprocessing.Queue()
                proc = multiprocessing.Process(
                    target=_download_gguf_in_subprocess,
                    args=(source, repo_id, gf, stage_dir, ms_domain, total_bytes, download_queue),
                    daemon=True,
                )

                def _watch_size():
                    prev_sz = 0
                    prev_ts = time.time()
                    last_speed = 0
                    last_eta: int | None = None
                    stale_ticks = 0
                    while not stop_flag["stop"]:
                        if _is_cancelled_or_stale():
                            return
                        sz = _probe_downloaded_size()
                        now = time.time()
                        dt = now - prev_ts
                        if dt > 0.1 and sz > prev_sz:
                            last_speed = int((sz - prev_sz) / dt)
                            stale_ticks = 0
                        else:
                            stale_ticks += 1
                            if stale_ticks > 10:
                                last_speed = 0
                                last_eta = None
                        prev_sz, prev_ts = sz, now

                        with self._lock:
                            job = self.download_jobs.get(model_id)
                            if not _is_current_job(job):
                                return
                            tb = int(job.get("total_bytes") or 0)
                            agg_now = int(aggregate_completed_bytes + (min(int(sz), total_bytes) if total_bytes > 0 else int(sz)))
                            job["downloaded_bytes"] = min(agg_now, tb) if tb > 0 else int(agg_now)
                            job["download_speed"] = last_speed
                            if tb > 0:
                                pct = int(min(99, max(0, (job["downloaded_bytes"] / tb) * 100)))
                                job["progress"] = pct
                                remaining = max(tb - int(job["downloaded_bytes"]), 0)
                                last_eta = int(remaining / last_speed) if last_speed > 0 else last_eta
                            job["eta_seconds"] = last_eta
                        time.sleep(1.0)

                watcher = threading.Thread(target=_watch_size, daemon=True)
                watcher.start()
                proc.start()

                with self._lock:
                    job = self.download_jobs.get(model_id)
                    if (not _is_current_job(job)) or bool(job.get("cancelled")):
                        logger.info(f"Download cancelled for {model_id}")
                        stop_flag["stop"] = True
                        if proc.is_alive():
                            proc.terminate()
                            proc.join(timeout=2.0)
                        return
                    if not total_bytes:
                        job["progress"] = int((step_idx / max(1, total_steps)) * 100)

                logger.info(f"Downloading GGUF {gf} from {source} ({repo_id})...")
                result_payload: Dict[str, Any] = {}
                while proc.is_alive():
                    if _is_cancelled_or_stale():
                        logger.info(f"Download cancelled/stale during subprocess run for {model_id}:{gf}")
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        try:
                            proc.join(timeout=2.0)
                        except Exception:
                            pass
                        stop_flag["stop"] = True
                        try:
                            watcher.join(timeout=2.0)
                        except Exception:
                            pass
                        return
                    time.sleep(0.2)

                try:
                    proc.join(timeout=2.0)
                except Exception:
                    pass
                try:
                    if not download_queue.empty():
                        result_payload = dict(download_queue.get_nowait() or {})
                except Exception:
                    result_payload = {}

                stop_flag["stop"] = True
                try:
                    watcher.join(timeout=2.0)
                except Exception:
                    pass

                if _is_cancelled_or_stale():
                    return

                if not bool(result_payload.get("ok")):
                    raise RuntimeError(str(result_payload.get("error") or f"download_failed:{gf}"))

                stage_result_path = str(result_payload.get("result_path") or "")
                if (not stage_result_path) or (not os.path.isfile(stage_result_path)):
                    alt_path = self._find_gguf_in_tree(stage_dir, gf, min_size=1_000_000)
                    if alt_path:
                        stage_result_path = alt_path
                self._ensure_gguf_at_target(stage_result_path, stage_dir, gf)
                staged_target_path = os.path.join(stage_dir, gf)
                candidate_download_path = staged_target_path if os.path.isfile(staged_target_path) else stage_result_path
                final_sz = 0
                try:
                    if candidate_download_path and os.path.isfile(candidate_download_path):
                        final_sz = int(os.path.getsize(candidate_download_path))
                except Exception:
                    final_sz = _probe_downloaded_size()

                if source == "modelscope" and final_sz > 0:
                    if total_bytes <= 0 or abs(int(final_sz) - int(total_bytes)) > max(1_048_576, int(final_sz * 0.01)):
                        logger.warning(
                            f"Use actual downloaded size for ModelScope GGUF validation: "
                            f"file={gf} expected={int(total_bytes or 0)} actual={int(final_sz)}"
                        )
                        total_bytes = int(final_sz)

                staged_ok = self._is_gguf_complete(
                    staged_target_path,
                    total_bytes,
                    strict_expected=False,
                    allow_transient=True,
                )
                stage_header_ok = self._has_valid_gguf_header(candidate_download_path, allow_transient=True)
                if (not staged_ok) and source == "modelscope":
                    staged_ok = stage_header_ok
                logger.info(
                    f"GGUF staged validation: file={gf} result_path={stage_result_path!r} "
                    f"candidate={candidate_download_path!r} actual={int(final_sz or 0)} "
                    f"expected={int(total_bytes or 0)} staged_ok={bool(staged_ok)} "
                    f"header_ok={bool(stage_header_ok)}"
                )
                if not staged_ok:
                    raise RuntimeError(f"incomplete_download:{gf}")

                import shutil as _shutil
                copy_src = staged_target_path if os.path.isfile(staged_target_path) else candidate_download_path
                _shutil.copy2(copy_src, target_path)
                target_ok = self._is_gguf_complete(target_path, total_bytes, strict_expected=False)
                target_header_ok = self._has_valid_gguf_header(target_path)
                if (not target_ok) and source == "modelscope":
                    target_ok = target_header_ok
                logger.info(
                    f"GGUF target validation: file={gf} target={target_path!r} "
                    f"size={int(os.path.getsize(target_path)) if os.path.isfile(target_path) else 0} "
                    f"expected={int(total_bytes or 0)} target_ok={bool(target_ok)} "
                    f"header_ok={bool(target_header_ok)}"
                )
                if not target_ok:
                    self._cleanup_path(target_path)
                    raise RuntimeError(f"incomplete_download:{gf}")
                completed_now = min(int(final_sz), total_bytes) if total_bytes > 0 else int(final_sz)
                aggregate_completed_bytes += max(0, completed_now)
                with self._lock:
                    job = self.download_jobs.get(model_id)
                    if _is_current_job(job):
                        if aggregate_total_bytes > 0:
                            job["total_bytes"] = int(aggregate_total_bytes)
                            job["downloaded_bytes"] = int(min(aggregate_completed_bytes, aggregate_total_bytes))
                            job["progress"] = int(min(99, max(0, (job["downloaded_bytes"] / aggregate_total_bytes) * 100)))
                        else:
                            job["downloaded_bytes"] = int(aggregate_completed_bytes)
                            if int(job.get("total_bytes") or 0) == 0 and aggregate_completed_bytes > 0:
                                job["total_bytes"] = int(aggregate_completed_bytes)
                            job["progress"] = int(min(99, ((step_idx + 1) / max(1, total_steps)) * 100))

                if final_sz > 0:
                    try:
                        import json as _json
                        meta_path = os.path.join(model_dir, ".unfoldly_meta.json")
                        meta = {}
                        if os.path.isfile(meta_path):
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = _json.load(f)
                        if "file_sizes" not in meta:
                            meta["file_sizes"] = {}
                        meta["file_sizes"][gf] = int(final_sz)
                        with open(meta_path, "w", encoding="utf-8") as f:
                            _json.dump(meta, f)
                    except Exception:
                        pass
                self._cleanup_completed_stage_artifacts(stage_dir, gf)

            step_idx += 1

            do_full_snapshot = (not gguf_files and not has_quantizations) or other_items
            if do_full_snapshot:
                snapshot_total = self._query_repo_total_size(repo_id, source)

                with self._lock:
                    job = self.download_jobs.get(model_id)
                    if (not _is_current_job(job)) or bool(job.get("cancelled")):
                        logger.info(f"Download cancelled for {model_id}")
                        return
                    if not gguf_files:
                        job["progress"] = 1
                        if snapshot_total > 0:
                            job["total_bytes"] = snapshot_total
                    else:
                        job["progress"] = int(job.get("progress", 100) or 100)

                logger.info(f"Downloading extra files/dirs via snapshot from {source} ({repo_id})...")

                is_hf_format = model_config.get("format") == "hf"

                snap_stop = {"stop": False}

                def _watch_snapshot_size():
                    prev_sz = 0
                    prev_ts = time.time()
                    last_speed = 0
                    last_eta: int | None = None
                    stale_ticks = 0
                    while not snap_stop["stop"]:
                        if _is_cancelled_or_stale():
                            return
                        dir_size = 0
                        try:
                            for root, _ds, fs in os.walk(model_dir):
                                for fn in fs:
                                    try:
                                        dir_size += os.path.getsize(os.path.join(root, fn))
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        now = time.time()
                        dt = now - prev_ts
                        if dt > 0.1 and dir_size > prev_sz:
                            last_speed = int((dir_size - prev_sz) / dt)
                            stale_ticks = 0
                        else:
                            stale_ticks += 1
                            if stale_ticks > 10:
                                last_speed = 0
                                last_eta = None
                        prev_sz, prev_ts = dir_size, now

                        with self._lock:
                            job = self.download_jobs.get(model_id)
                            if not _is_current_job(job):
                                return
                            job["downloaded_bytes"] = int(dir_size)
                            job["download_speed"] = last_speed
                            tb = int(job.get("total_bytes") or 0)
                            if tb > 0:
                                job["progress"] = int(min(95, max(5, (dir_size / tb) * 100)))
                                remaining = tb - dir_size
                                last_eta = int(remaining / last_speed) if last_speed > 0 else last_eta
                            elif dir_size > 0:
                                job["progress"] = min(90, 5 + int(dir_size / (10 * 1024 * 1024)))
                            job["eta_seconds"] = last_eta
                        time.sleep(1.0)

                if not gguf_files:
                    snap_watcher = threading.Thread(target=_watch_snapshot_size, daemon=True)
                    snap_watcher.start()

                try:
                    if source == "modelscope":
                        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download  # type: ignore

                        if is_hf_format:
                            ignore = [r".*\.gguf$", r".*\.onnx$"]
                        else:
                            ignore = [r".*\.gguf$", r".*\.bin$", r".*\.h5$", r".*\.msgpack$", r".*\.onnx$"]
                        with self._temp_modelscope_domain(ms_domain):
                            ms_snapshot_download(
                                model_id=repo_id,
                                cache_dir=os.path.dirname(model_dir),
                                local_dir=model_dir,
                                ignore_file_pattern=ignore,
                            )
                    else:
                        from huggingface_hub import snapshot_download  # type: ignore

                        allow_patterns = None
                        if has_quantizations and other_items:
                            allow_patterns = []
                            for it in other_items:
                                it = str(it)
                                if it.endswith("/"):
                                    allow_patterns.append(it + "**")
                                elif "." not in os.path.basename(it):
                                    allow_patterns.append(it.rstrip("/") + "/**")
                                else:
                                    allow_patterns.append(it)

                        snapshot_download(
                            repo_id=repo_id,
                            local_dir=model_dir,
                            local_dir_use_symlinks=False,
                            resume_download=True,
                            allow_patterns=allow_patterns,
                            ignore_patterns=["*.gguf"],
                        )
                finally:
                    snap_stop["stop"] = True
                    if not gguf_files:
                        try:
                            snap_watcher.join(timeout=2.0)
                        except Exception:
                            pass

                if not gguf_files:
                    final_size = 0
                    try:
                        for root, _ds, fs in os.walk(model_dir):
                            for fn in fs:
                                try:
                                    final_size += os.path.getsize(os.path.join(root, fn))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    with self._lock:
                        job = self.download_jobs.get(model_id)
                        if _is_current_job(job):
                            job["downloaded_bytes"] = int(final_size)
                            if int(job.get("total_bytes") or 0) == 0 and final_size > 0:
                                job["total_bytes"] = int(final_size)
                            job["progress"] = 99
                elif other_expected_total > 0:
                    aggregate_completed_bytes += int(other_expected_total)
                    with self._lock:
                        job = self.download_jobs.get(model_id)
                        if _is_current_job(job) and aggregate_total_bytes > 0:
                            job["total_bytes"] = int(aggregate_total_bytes)
                            job["downloaded_bytes"] = int(min(aggregate_completed_bytes, aggregate_total_bytes))
                            job["progress"] = int(min(99, max(0, (job["downloaded_bytes"] / aggregate_total_bytes) * 100)))
            
            # Check cancellation again before finish
            with self._lock:
                job = self.download_jobs.get(model_id)
                if not _is_current_job(job):
                    return
                if bool(job.get("cancelled")):
                    return
                job["status"] = "installed"
                job["progress"] = 100
                tb = int(job.get("total_bytes") or 0)
                if tb > 0:
                    job["downloaded_bytes"] = tb

            def _delayed_cleanup(_mid: str, _rid: int) -> None:
                for _ in range(8):
                    time.sleep(1.0)
                    with self._lock:
                        j = self.download_jobs.get(_mid)
                        if not j or int(j.get("run_id") or 0) != _rid:
                            return
                        if j.get("status") != "installed":
                            return
                        j["status"] = "installed"
                        j["progress"] = 100
                with self._lock:
                    j = self.download_jobs.get(_mid)
                    if j and int(j.get("run_id") or 0) == _rid and j.get("status") == "installed":
                        del self.download_jobs[_mid]
            threading.Thread(target=_delayed_cleanup, args=(model_id, run_id), daemon=True).start()
                
        except Exception as e:
            logger.error(f"Download failed for {model_id}: {e}")
            with self._lock:
                job = self.download_jobs.get(model_id)
                # If cancelled/stale, don't overwrite with error
                if _is_current_job(job) and (not bool(job.get("cancelled"))):
                    job["status"] = "error"
                    job["error"] = str(e)
        finally:
            self._cleanup_path(run_stage_root)

    def delete_model(self, model_id: str, quantization_file: Optional[str] = None) -> Dict[str, Any]:
        model_dir = os.path.join(self.models_dir, model_id)
        qf = str(quantization_file or "").strip()
        try:
            import shutil

            with self._lock:
                job = self.download_jobs.get(model_id)
                if job and job.get("status") == "downloading":
                    should_cancel = (not qf) or (str(job.get("quantization_file") or "").strip() == qf)
                    if should_cancel:
                        self._run_seq += 1
                        job["cancelled"] = True
                        job["status"] = "cancelled"
                        job["run_id"] = int(self._run_seq)

            deadline = time.time() + 1.0
            while time.time() < deadline:
                with self._lock:
                    jj = self.download_jobs.get(model_id)
                    if not jj or jj.get("status") != "downloading":
                        break
                time.sleep(0.05)

            def _on_rm_error(func, path, _exc_info):
                try:
                    os.chmod(path, 0o700)
                    func(path)
                except Exception:
                    pass

            if qf:
                target_gguf = os.path.join(model_dir, qf)
                deleted = False
                if os.path.isfile(target_gguf):
                    try:
                        os.remove(target_gguf)
                        deleted = True
                        logger.info(f"[Model] Deleted quantization file: {target_gguf}")
                    except Exception as e:
                        return {"ok": False, "error": f"删除量化文件失败: {e}"}
                else:
                    logger.warning(f"[Model] Quantization file not found (already deleted?): {target_gguf}")
                    deleted = True

                remaining_ggufs = []
                if os.path.isdir(model_dir):
                    for fn in os.listdir(model_dir):
                        if fn.lower().endswith(".gguf") and os.path.isfile(os.path.join(model_dir, fn)):
                            remaining_ggufs.append(fn)

                if not remaining_ggufs and os.path.isdir(model_dir):
                    try:
                        shutil.rmtree(model_dir, onerror=_on_rm_error)
                        logger.info(f"[Model] No remaining GGUF files, removed entire model dir: {model_dir}")
                    except Exception as e:
                        logger.warning(f"[Model] Failed to remove model dir after last GGUF deleted: {e}")

                with self._lock:
                    if not remaining_ggufs and model_id in self.download_jobs:
                        del self.download_jobs[model_id]

                return {"ok": True, "deleted_quantization": qf, "remaining_quantizations": remaining_ggufs}
            else:
                if os.path.isdir(model_dir):
                    shutil.rmtree(model_dir, onerror=_on_rm_error)
                elif os.path.exists(model_dir):
                    try:
                        os.remove(model_dir)
                    except Exception:
                        pass

                residual_files: List[str] = []
                if os.path.exists(model_dir):
                    for root, _dirs, files in os.walk(model_dir):
                        for fn in files:
                            residual_files.append(os.path.join(root, fn))
                            if len(residual_files) >= 20:
                                break
                        if len(residual_files) >= 20:
                            break

                with self._lock:
                    if model_id in self.download_jobs:
                        del self.download_jobs[model_id]

                if residual_files:
                    return {
                        "ok": False,
                        "error": f"Model directory is not fully removed: {model_dir}",
                        "residual_files": residual_files,
                    }

                return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}
