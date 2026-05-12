import gc
import json
import os
import time
from utils.logger import get_child_logger

import threading
from typing import Any, Dict, Iterator, Optional, Tuple, Union

logger = get_child_logger(__name__)


def _parse_int_env(name: str, default: int) -> int:
    try:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return int(default)
        return int(raw)
    except Exception:
        return int(default)


class LocalLLMManager:

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

        data_dir = (os.getenv("FILEAGENT_DATA_DIR") or "").strip()
        data_dir = os.path.abspath(os.path.expanduser(data_dir)) if data_dir else ""
        models_dir = (os.getenv("FILEAGENT_LOCAL_MODELS_DIR") or "").strip()
        if not models_dir and data_dir:
            models_dir = os.path.join(data_dir, "local_models")
        self.models_dir = os.path.abspath(os.path.expanduser(models_dir)) if models_dir else os.path.join(base_dir, "local_models")

        cfg = (os.getenv("FILEAGENT_SUPPORTED_MODELS_PATH") or "").strip()
        if cfg:
            self.config_path = os.path.abspath(os.path.expanduser(cfg))
        else:
            candidate = os.path.join(base_dir, "config", "supported_models.json")
            legacy = os.path.join(base_dir, "supported_models.json")
            self.config_path = candidate if os.path.exists(candidate) else legacy

        self._lock = threading.RLock()
        self._llama = None
        self._chat_handler = None
        self._stream_abort_requested: threading.Event = threading.Event()

        self.current_model_id: Optional[str] = None
        self.current_model_path: Optional[str] = None
        self.current_mmproj_path: Optional[str] = None
        self.current_n_ctx: Optional[int] = None
        self.current_n_batch: Optional[int] = None
        self.current_n_ubatch: Optional[int] = None
        self.current_n_gpu_layers: Optional[int] = None

        self.default_n_ctx = _parse_int_env("FILEAGENT_LLM_N_CTX", 5120)
        self.default_n_batch = _parse_int_env("FILEAGENT_LLM_N_BATCH", 512)
        self.default_n_gpu_layers = _parse_int_env("FILEAGENT_LLM_N_GPU_LAYERS", -1)
        self.default_verbose = bool((os.getenv("FILEAGENT_LLM_VERBOSE") or "").strip().lower() in {"1", "true", "yes", "y", "on"})
        self._startup_index_prefill_lock = threading.Lock()
        self._startup_index_prefill_status: Dict[str, Any] = {
            "state": "idle",
            "reason": "",
            "target_model_id": "",
            "target_model_path": "",
            "started_at": 0.0,
            "completed_at": 0.0,
            "elapsed_ms": 0,
            "observations_total": 0,
            "hits_total": 0,
            "last_hit": None,
            "last_file_path": "",
            "last_observed_at": 0.0,
        }

    def update_startup_index_prefill_status(
        self,
        *,
        state: str,
        reason: str = "",
        target_model_id: str = "",
        target_model_path: str = "",
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        elapsed_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self._startup_index_prefill_lock:
            snapshot = dict(self._startup_index_prefill_status)
            snapshot["state"] = str(state or "unknown").strip() or "unknown"
            if reason or snapshot.get("reason", "") == "":
                snapshot["reason"] = str(reason or "")[:500]
            if target_model_id or snapshot.get("target_model_id", "") == "":
                snapshot["target_model_id"] = str(target_model_id or "")[:240]
            if target_model_path or snapshot.get("target_model_path", "") == "":
                snapshot["target_model_path"] = str(target_model_path or "")[:500]
            if started_at is not None:
                snapshot["started_at"] = float(started_at)
            if completed_at is not None:
                snapshot["completed_at"] = float(completed_at)
            if elapsed_ms is not None:
                snapshot["elapsed_ms"] = int(elapsed_ms)
            self._startup_index_prefill_status = snapshot
            return dict(snapshot)

    def get_startup_index_prefill_status(self) -> Dict[str, Any]:
        with self._startup_index_prefill_lock:
            return dict(self._startup_index_prefill_status)

    def record_startup_index_prefill_observation(
        self,
        *,
        hit: bool,
        file_path: str,
    ) -> Dict[str, Any]:
        with self._startup_index_prefill_lock:
            snapshot = dict(self._startup_index_prefill_status)
            observations_total = int(snapshot.get("observations_total", 0) or 0) + 1
            hits_total = int(snapshot.get("hits_total", 0) or 0) + (1 if hit else 0)
            snapshot["observations_total"] = observations_total
            snapshot["hits_total"] = hits_total
            snapshot["last_hit"] = bool(hit)
            snapshot["last_file_path"] = str(file_path or "")[:500]
            snapshot["last_observed_at"] = time.time()
            self._startup_index_prefill_status = snapshot
            result = dict(snapshot)
            result["hit_rate"] = (hits_total / observations_total) if observations_total > 0 else 0.0
            return result

    def get_target_model_config(self, preferred_model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Find the model config from supported_models.json"""
        if not os.path.exists(self.config_path):
            # Fallback to repo config path if app support path is missing
            repo_config = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "supported_models.json")
            if os.path.exists(repo_config):
                self.config_path = repo_config
            else:
                logger.warning(f"Model config not found at {self.config_path} or {repo_config}")
                return None
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                models = json.load(f)
            
            if preferred_model_id:
                for m in models:
                    if m.get("id", "").lower() == preferred_model_id.lower() or preferred_model_id.lower() in m.get("id", "").lower():
                        return m
            
            for m in models:
                if "qwen3-4b" in m["id"]:
                    return m

            for m in models:
                if "qwen3-vl-2b" in m["id"]:
                    return m

            for m in models:
                if "vl" in m["id"]:
                    return m
            
            return None
        except:
            return None

    def get_model_thinking_suppression(self, preferred_model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Read model-specific thinking suppression settings from supported_models.json.

        Design note:
        - Gemma 4 uses structured suppression here (`stop` / `logit_bias`) and
          then relies on downstream visible-text cleaning as a final guard.
        - We intentionally do NOT return direct chat-template kwargs such as
          `enable_thinking` / `add_generation_prompt`, because the current
          llama_cpp runtime used by the app does not support them reliably.
        - Qwen-family "no think" prompting is handled separately through
          `intent_prompt_suffix` in the indexing / intent prompt builders.

        Example schema:
        {
          "thinking_suppression": {
          "handler": "gemma4_channel",
          "logit_bias": {"100": -100.0},
            "stop": ["<|channel>thought", "<channel|>thought"]
          }
        }
        """
        cfg = self.get_target_model_config(preferred_model_id) or {}
        raw = cfg.get("thinking_suppression") or {}
        if not isinstance(raw, dict):
            return {}

        suppression: Dict[str, Any] = {}
        handler = str(raw.get("handler") or "").strip()
        if handler:
            suppression["handler"] = handler

        stop_items = []
        for item in list(raw.get("stop") or []):
            text = str(item or "").strip()
            if text:
                stop_items.append(text)
        if stop_items:
            suppression["stop"] = stop_items

        logit_bias_raw = raw.get("logit_bias") or {}
        logit_bias: Dict[int, float] = {}
        if isinstance(logit_bias_raw, dict):
            for key, value in logit_bias_raw.items():
                try:
                    logit_bias[int(key)] = float(value)
                except Exception:
                    continue
        if logit_bias:
            suppression["logit_bias"] = logit_bias

        return suppression

    def _apply_model_thinking_suppression(
        self,
        kwargs: Dict[str, Any],
        *,
        model_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Apply runtime-safe thinking suppression controls to llama_cpp kwargs.

        Current policy:
        - Gemma 4: only inject `stop` and `logit_bias`.
        - Qwen-family prompt-level no-think behavior is handled elsewhere via
          prompt suffixes, not by passing unsupported template kwargs here.
        """
        merged = dict(kwargs or {})
        suppression = self.get_model_thinking_suppression(model_id)
        if not suppression:
            return merged

        preset_stops = ["<|im_end|>", "</im_end>", "<|im_start|>", "<start_of_turn>", "<end_of_turn>"]
        existing_stop = merged.get("stop", [])
        if isinstance(existing_stop, str):
            existing_stop = [existing_stop]
        merged_stop = list(dict.fromkeys(list(existing_stop or []) + preset_stops + list(suppression.get("stop") or [])))
        merged["stop"] = merged_stop

        existing_logit_bias = merged.get("logit_bias") or {}
        if not isinstance(existing_logit_bias, dict):
            existing_logit_bias = {}
        merged_logit_bias = dict(existing_logit_bias)
        for token_id, bias_value in dict(suppression.get("logit_bias") or {}).items():
            merged_logit_bias[int(token_id)] = float(bias_value)
        if merged_logit_bias:
            merged["logit_bias"] = merged_logit_bias

        return merged

    def resolve_target_model(
        self, preferred_model_id: Optional[str] = None, preferred_quantization_file: Optional[str] = None
    ) -> Optional[Tuple[Dict[str, Any], str, Optional[str]]]:
        # If preferred model is passed in, check if it's actually valid before getting config
        # This prevents the local LLM manager from failing to find config when passed qwen3-vl-2b-instruct-gguf
        model_config = self.get_target_model_config(preferred_model_id)
        if not model_config:
            # Fallback for old configs that might not match exact ids
            if preferred_model_id and "vl" in preferred_model_id.lower():
                model_config = self.get_target_model_config("qwen3-vl-2b-instruct-gguf")
            if not model_config:
                return None

        model_id = model_config.get("id")
        if not model_id:
            return None

        model_dir = os.path.join(self.models_dir, model_id)

        def _is_stable_file(path: str, *, min_size: int = 1_000_000) -> bool:
            """Only treat complete/stable files as loadable model artifacts."""
            try:
                if not path or not os.path.isfile(path):
                    return False
                low = path.lower()
                if low.endswith((".downloading", ".tmp", ".part", ".incomplete")):
                    return False
                # Skip known transient cache/temp folders from download SDKs.
                for marker in ("/._____temp/", "/.msc/", "/.mv/", "/.cache/"):
                    if marker in low.replace("\\", "/"):
                        return False
                return os.path.getsize(path) >= int(min_size)
            except Exception:
                return False

        def _resolve_existing_file(root_dir: str, filename: str) -> Optional[str]:
            """Find stable file path in root_dir (prefer canonical target path)."""
            if not filename:
                return None
            direct = os.path.join(root_dir, filename)
            if _is_stable_file(direct):
                return direct
            try:
                for r, _ds, fs in os.walk(root_dir):
                    if filename in fs:
                        candidate = os.path.join(r, filename)
                        if _is_stable_file(candidate):
                            return candidate
                filename_lower = filename.lower()
                for r, _ds, fs in os.walk(root_dir):
                    for fn in fs:
                        if fn.lower() == filename_lower:
                            candidate = os.path.join(r, fn)
                            if _is_stable_file(candidate):
                                return candidate
            except Exception:
                pass
            return None

        gguf_candidates = []
        pq = (preferred_quantization_file or "").strip()
        if pq:
            gguf_candidates.append(pq)
        dq = (model_config.get("default_quantization") or "").strip()
        if dq:
            gguf_candidates.append(dq)
            
        # Fallback handle Qwen3-VL specifically where names might be mapped weirdly
        if model_id == "qwen3-vl-2b-instruct-gguf" and "Qwen3VL-2B-Instruct-Q4_K_M.gguf" not in gguf_candidates:
             gguf_candidates.append("Qwen3VL-2B-Instruct-Q4_K_M.gguf")
        try:
            for q in (model_config.get("quantizations") or []):
                if isinstance(q, dict) and q.get("file"):
                    gguf_candidates.append(str(q["file"]))
        except Exception:
            pass
        seen = set()
        gguf_candidates = [x for x in gguf_candidates if x and not (x in seen or seen.add(x))]

        resolved_model_path: Optional[str] = None
        for f in gguf_candidates:
            p = _resolve_existing_file(model_dir, f)
            if p:
                resolved_model_path = p
                break

        if not resolved_model_path:
            best_path = None
            best_size = -1
            best_path_any = None
            best_size_any = -1
            try:
                for r, _ds, fs in os.walk(model_dir):
                    for fn in fs:
                        if fn.lower().endswith(".gguf"):
                            fp = os.path.join(r, fn)
                            try:
                                sz = os.path.getsize(fp)
                            except Exception:
                                sz = 0
                            if not _is_stable_file(fp, min_size=1):
                                continue
                            if sz > best_size_any:
                                best_size_any = sz
                                best_path_any = fp
                            if "mmproj" in fn.lower():
                                continue
                            if sz > best_size:
                                best_size = sz
                                best_path = fp
            except Exception:
                best_path = None
            resolved_model_path = best_path or best_path_any

        if not resolved_model_path:
            return None

        mmproj_file = None
        try:
            for f in (model_config.get("files") or []):
                if isinstance(f, str) and "mmproj" in f:
                    mmproj_file = f
                    break
        except Exception:
            mmproj_file = None

        model_path = resolved_model_path
        mmproj_path = _resolve_existing_file(model_dir, mmproj_file) if mmproj_file else None
        return (model_config, model_path, mmproj_path)

    def _build_llama(
        self,
        model_path: str,
        mmproj_path: Optional[str],
        *,
        n_ctx: int,
        n_batch: int,
        n_ubatch: int,
        n_gpu_layers: int,
        verbose: bool,
    ):
        import multiprocessing
        n_threads = max(1, multiprocessing.cpu_count() - 2)
        import platform
        if platform.system() == "Darwin":
            n_threads = min(n_threads, 4)

        from llama_cpp import Llama  # type: ignore
        import os
        
        # [CRITICAL FIX] Disable Metal BF16 kernels for Gemma 4
        # Gemma 4 models (even Q4_0) contain a single `bf16` tensor for output embeddings.
        # Apple's Metal shader compiler has a bug compiling `kernel_mul_mm_bf16_f32` on some M-series chips,
        # which leads to a missing symbol and a hard segmentation fault during inference.
        # Disabling bf16 forces fallback to f32/CPU for just that one tensor, keeping GPU acceleration for the rest.
        if "gemma4" in model_path.lower() or "gemma-4" in model_path.lower():
            os.environ["GGML_METAL_BF16_DISABLE"] = "1"
            
        kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": int(n_ctx),
            "n_batch": int(n_batch),
            "n_ubatch": int(n_ubatch),
            "n_gpu_layers": int(n_gpu_layers),
            "n_threads": n_threads,
            "verbose": bool(verbose),
            "flash_attn": True,
        }

        chat_handler = None
        if mmproj_path and os.path.exists(mmproj_path):
            from llama_cpp import llama_chat_format  # type: ignore
            
            lower_model_path = model_path.lower()
            if "qwen3-vl" in lower_model_path or "qwen3" in lower_model_path:
                if hasattr(llama_chat_format, "Qwen3VLChatHandler"):
                    chat_handler = llama_chat_format.Qwen3VLChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
                elif hasattr(llama_chat_format, "Qwen25VLChatHandler"):
                    chat_handler = llama_chat_format.Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
                else:
                    chat_handler = llama_chat_format.Llava15ChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
            elif "qwen2.5-vl" in lower_model_path or "qwen25" in lower_model_path:
                chat_handler = llama_chat_format.Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
            elif "glm-4.6" in lower_model_path or "glm46" in lower_model_path or "glm-4.6v" in lower_model_path:
                if hasattr(llama_chat_format, "GLM46VChatHandler"):
                    chat_handler = llama_chat_format.GLM46VChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
                elif hasattr(llama_chat_format, "GLM41VChatHandler"):
                    chat_handler = llama_chat_format.GLM41VChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
                else:
                    chat_handler = llama_chat_format.Llava15ChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
            elif "glm-4.1" in lower_model_path or "glm41" in lower_model_path:
                if hasattr(llama_chat_format, "GLM41VChatHandler"):
                    chat_handler = llama_chat_format.GLM41VChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
                else:
                    chat_handler = llama_chat_format.Llava15ChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
            elif "ministral" in lower_model_path or "pixtral" in lower_model_path:
                if hasattr(llama_chat_format, "PixtralChatHandler"):
                    chat_handler = llama_chat_format.PixtralChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
                elif hasattr(llama_chat_format, "MTMDChatHandler"):
                    chat_handler = llama_chat_format.MTMDChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
                else:
                    # Fallback
                    chat_handler = llama_chat_format.Llava15ChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
            elif "gemma-4" in lower_model_path or "gemma4" in lower_model_path:
                if hasattr(llama_chat_format, "Gemma4ChatHandler"):
                    chat_handler = llama_chat_format.Gemma4ChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
                elif hasattr(llama_chat_format, "Gemma3ChatHandler"):
                    chat_handler = llama_chat_format.Gemma3ChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
                else:
                    chat_handler = llama_chat_format.Llava15ChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
            elif "gemma-3" in lower_model_path or "gemma3" in lower_model_path:
                if hasattr(llama_chat_format, "Gemma3ChatHandler"):
                    chat_handler = llama_chat_format.Gemma3ChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
                else:
                    chat_handler = llama_chat_format.Llava15ChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
            else:
                if hasattr(llama_chat_format, "Qwen25VLChatHandler"):
                    chat_handler = llama_chat_format.Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
                else:
                    chat_handler = llama_chat_format.Llava15ChatHandler(clip_model_path=mmproj_path, verbose=bool(verbose))
                
            kwargs["chat_handler"] = chat_handler

        llama = Llama(**kwargs)
        return llama, chat_handler

    def load_model_for_indexing(
        self,
        preferred_model_id: Optional[str] = None,
        *,
        preferred_quantization_file: Optional[str] = None,
    ) -> bool:
        logger.info("[LLM] Loading model for indexing with stable ctx/batch (GPU enabled)")
        return self.load_model(
            preferred_model_id=preferred_model_id,
            preferred_quantization_file=preferred_quantization_file,
            n_ctx=5120,
            n_batch=512,
            n_gpu_layers=self.default_n_gpu_layers,
            verbose=False,
            needs_vision=True,  # Always load the visual model component during indexing
        )

    def load_model(
        self,
        preferred_model_id: Optional[str] = None,
        *,
        preferred_quantization_file: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        verbose: Optional[bool] = None,
        needs_vision: Optional[bool] = None,
    ) -> bool:
        load_start = time.time()
        resolved = self.resolve_target_model(preferred_model_id, preferred_quantization_file=preferred_quantization_file)
        if not resolved:
            logger.warning("Target model config not found in supported_models.json")
            return False

        model_config, model_path, mmproj_path = resolved
        model_id = model_config.get("id", preferred_model_id) or "unknown"
        
        if needs_vision is False and mmproj_path:
            # If the model is already loaded with mmproj, retain it to avoid an expensive full-reload
            if self.current_model_id == model_id and self.current_mmproj_path:
                logger.debug("Chat/Text usage: retaining mmproj_path because it is already loaded in VRAM")
            else:
                logger.info("Chat/Text usage: disabling mmproj_path for memory/VRAM optimization")
                mmproj_path = None

        if not model_path or not os.path.exists(model_path):
            logger.warning(f"Model file not found/resolved for model_id={model_id}. Please download it via UI.")
            logger.info(f"Model not found, cannot load: model_id={model_id}, resolved_model_path={model_path}")
            return False

        vn_ctx = int(n_ctx if n_ctx is not None else self.default_n_ctx)
        vn_batch = int(n_batch if n_batch is not None else self.default_n_batch)
        vn_gpu_layers = int(n_gpu_layers if n_gpu_layers is not None else self.default_n_gpu_layers)
        vverbose = bool(verbose if verbose is not None else self.default_verbose)

        if mmproj_path and os.path.exists(mmproj_path):
            vl_n_ctx = _parse_int_env("FILEAGENT_VL_N_CTX", 5120)
            vl_n_batch = _parse_int_env("FILEAGENT_VL_N_BATCH", 512)
            vn_ctx = min(vn_ctx, int(vl_n_ctx))
            vn_batch = min(vn_batch, int(vl_n_batch))
        vn_ubatch = int(min(vn_batch, _parse_int_env("FILEAGENT_LLM_N_UBATCH", vn_batch)))

        with self._lock:
            # already loaded same target?
            # We must also consider if we need vision. If we need vision but didn't load mmproj, or vice versa, we must reload.
            already_loaded = (
                self._llama is not None 
                and self.current_model_id == model_id 
                and self.current_model_path == model_path
                and self.current_mmproj_path == mmproj_path
                and self.current_n_ctx == vn_ctx
                and self.current_n_batch == vn_batch
                and self.current_n_ubatch == vn_ubatch
                and self.current_n_gpu_layers == vn_gpu_layers
            )
            
            if already_loaded:
                logger.info(f"[LLM] model already loaded with correct mmproj state, skip reload: model_id={model_id}")
                return True

            # unload previous
            if self.current_model_id:
                logger.info(f"[LLM] switching model/modality: from={self.current_model_id} (mmproj: {bool(self.current_mmproj_path)}) to={model_id} (mmproj: {bool(mmproj_path)})")
            self.unload_model()

            logger.info(f"Loading model in-proc: model_id={model_id}")
            logger.info(f"model_path={model_path}")
            if mmproj_path:
                logger.info(f"mmproj_path={mmproj_path}")
            logger.info(f"llama params: n_ctx={vn_ctx}, n_batch={vn_batch}, n_ubatch={vn_ubatch}, n_gpu_layers={vn_gpu_layers}")

            try:
                llama, chat_handler = self._build_llama(
                    model_path,
                    mmproj_path,
                    n_ctx=vn_ctx,
                    n_batch=vn_batch,
                    n_ubatch=vn_ubatch,
                    n_gpu_layers=vn_gpu_layers,
                    verbose=vverbose,
                )
            except Exception as e:
                logger.error(f"[LLM] load failed: model_id={model_id} error={e}")
                raise

            self._llama = llama
            self._chat_handler = chat_handler
            self.current_model_id = str(model_id)
            self.current_model_path = model_path
            self.current_mmproj_path = mmproj_path
            self.current_n_ctx = vn_ctx
            self.current_n_batch = vn_batch
            self.current_n_ubatch = vn_ubatch
            self.current_n_gpu_layers = vn_gpu_layers
            elapsed_ms = int((time.time() - load_start) * 1000)
            logger.info(f"[LLM] load done: model_id={self.current_model_id} elapsed_ms={elapsed_ms}")
            return True

    def unload_model(self) -> None:
        unload_start = time.time()
        prev_model_id = self.current_model_id

        self._stream_abort_requested.set()

        _UNLOAD_LOCK_TIMEOUT = 15
        acquired = self._lock.acquire(timeout=_UNLOAD_LOCK_TIMEOUT)
        if not acquired:
            logger.warning(
                f"[LLM] unload_model: lock not acquired within {_UNLOAD_LOCK_TIMEOUT}s "
                "\u2014 Metal stream may still be running. Forcing Python state reset anyway."
            )
        try:
            llama = self._llama
            chat_handler = self._chat_handler
            self._llama = None
            self._chat_handler = None
            self.current_model_id = None
            self.current_model_path = None
            self.current_mmproj_path = None
            self.current_n_ctx = None
            self.current_n_batch = None
            self.current_n_ubatch = None
            self.current_n_gpu_layers = None

            try:
                if chat_handler is not None and hasattr(chat_handler, "close"):
                    chat_handler.close()  # type: ignore
            except Exception:
                pass

            try:
                if llama is not None and hasattr(llama, "close"):
                    llama.close()  # type: ignore
            except Exception:
                pass
        finally:
            if acquired:
                try:
                    self._lock.release()
                except Exception:
                    pass
            self._stream_abort_requested.clear()

        try:
            del chat_handler
        except Exception:
            pass
        try:
            del llama
        except Exception:
            pass
        try:
            gc.collect()
            gc.collect()
        except Exception:
            pass
        elapsed_ms = int((time.time() - unload_start) * 1000)
        if prev_model_id:
            logger.info(f"[LLM] unload done: model_id={prev_model_id} elapsed_ms={elapsed_ms}")

    def start_server(self, preferred_model_id: Optional[str] = None, preferred_quantization_file: Optional[str] = None):
        ok = self.load_model(
            preferred_model_id,
            preferred_quantization_file=preferred_quantization_file,
        )
        if ok and self.current_model_id:
            logger.info(f"Loaded successfully: current_model_id={self.current_model_id}")
        return ok

    def stop_server(self):
        if self.current_model_id:
            logger.info(f"Unloading model: model_id={self.current_model_id}")
        self.unload_model()

    def create_chat_completion(
        self,
        *,
        model_id: Optional[str],
        preferred_quantization_file: Optional[str],
        messages: Any,
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        #
        req_start = time.time()
        msg_count = len(messages) if isinstance(messages, list) else 0
        needs_vision = kwargs.pop("needs_vision", False)
        
        logger.info(
            f"[LLM] chat request: model_id={model_id or '<auto>'} stream={bool(stream)} "
            f"message_count={msg_count} max_tokens={max_tokens} needs_vision={needs_vision}"
        )
        if not stream:
            with self._lock:
                # model_id=None → reuse currently loaded model, no resolve/reload
                if model_id is None and self._llama is not None:
                    reload_needed = False
                else:
                    resolved = self.resolve_target_model(model_id, preferred_quantization_file)
                    target_mmproj = None
                    if resolved:
                        resolved_cfg, _, target_mmproj = resolved
                        res_id = resolved_cfg.get("id", model_id) if resolved_cfg else model_id
                        if needs_vision is False:
                            if self.current_model_id == res_id and self.current_mmproj_path and target_mmproj:
                                pass
                            else:
                                target_mmproj = None

                    reload_needed = (
                        self._llama is None
                        or (model_id and self.current_model_id != model_id)
                        or (self.current_mmproj_path != target_mmproj)
                    )

                if reload_needed:
                    logger.info(
                        f"[LLM] chat requires model load: current={self.current_model_id} target={model_id} needs_vision={needs_vision}"
                    )
                    ok = self.load_model(model_id, preferred_quantization_file=preferred_quantization_file, needs_vision=needs_vision)
                    if not ok:
                        raise ValueError(f"Failed to load local model: {model_id}")
                llama = self._llama
                if llama is None:
                    raise ValueError("Local LLM not loaded")
                kwargs = self._apply_model_thinking_suppression(
                    kwargs,
                    model_id=self.current_model_id or model_id,
                )
                result = llama.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=float(temperature),
                    stream=False,
                    **kwargs,
                )
                text_len = 0
                try:
                    text_len = len(
                        (((result.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
                    )
                except Exception:
                    text_len = 0
                elapsed_ms = int((time.time() - req_start) * 1000)
                logger.info(
                    f"[LLM] chat done: model_id={self.current_model_id} stream=False text_len={text_len} elapsed_ms={elapsed_ms}"
                )
                return result

        _STREAM_LOCK_TIMEOUT = 60
        lock_acquired = self._lock.acquire(timeout=_STREAM_LOCK_TIMEOUT)
        if not lock_acquired:
            raise TimeoutError(
                f"[LLM] Could not acquire lock within {_STREAM_LOCK_TIMEOUT}s — "
                "a previous streaming generation may have been abandoned without releasing the lock. "
                "This usually means the previous request's generator was garbage-collected. "
                "Reloading the model or restarting may be needed if this persists."
            )
        try:
            if model_id is None and self._llama is not None:
                reload_needed = False
            else:
                resolved = self.resolve_target_model(model_id, preferred_quantization_file)
                target_mmproj = None
                if resolved:
                    _, _, target_mmproj = resolved
                    if needs_vision is False:
                        target_mmproj = None

                reload_needed = (
                    self._llama is None
                    or (model_id and self.current_model_id != model_id)
                    or (self.current_mmproj_path != target_mmproj)
                )

            if reload_needed:
                logger.info(
                    f"[LLM] chat(stream) requires model load: current={self.current_model_id} target={model_id} needs_vision={needs_vision}"
                )
                ok = self.load_model(model_id, preferred_quantization_file=preferred_quantization_file, needs_vision=needs_vision)
                if not ok:
                    raise ValueError(f"Failed to load local model: {model_id}")
            llama = self._llama
            if llama is None:
                raise ValueError("Local LLM not loaded")
            kwargs = self._apply_model_thinking_suppression(
                kwargs,
                model_id=self.current_model_id or model_id,
            )
            it = llama.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=float(temperature),
                stream=True,
                **kwargs,
            )
        except Exception:
            self._lock.release()
            raise

        self._stream_abort_requested.clear()

        def _guarded_iter() -> Iterator[Dict[str, Any]]:
            chunk_count = 0
            emitted_chars = 0
            first_chunk_ms = 0
            released = False

            def _release_lock():
                nonlocal released
                if not released:
                    released = True
                    try:
                        self._lock.release()
                    except Exception:
                        pass
                    try:
                        if hasattr(it, "close"):
                            it.close()
                    except Exception:
                        pass
                    elapsed_ms = int((time.time() - req_start) * 1000)
                    logger.info(
                        f"[LLM] chat done: model_id={self.current_model_id} stream=True "
                        f"chunks={chunk_count} text_len={emitted_chars} elapsed_ms={elapsed_ms} "
                        f"first_chunk_ms={first_chunk_ms}"
                    )

            try:
                for ch in it:  # type: ignore
                    try:
                        delta = ((ch.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                        emitted_chars += len(str(delta))
                        if delta and not first_chunk_ms:
                            first_chunk_ms = int((time.time() - req_start) * 1000)
                            logger.info(
                                f"[LLM] first_chunk: model_id={self.current_model_id} stream=True first_chunk_ms={first_chunk_ms}"
                            )
                    except Exception:
                        pass
                    chunk_count += 1
                    yield ch
                    if self._stream_abort_requested.is_set():
                        logger.info("[LLM] _stream_abort_requested detected — stopping stream early")
                        break
            except GeneratorExit:
                logger.info("[LLM] Stream generator closed early (GeneratorExit) — releasing lock")
                _release_lock()
                return
            except Exception as exc:
                logger.warning(f"[LLM] Stream iteration error: {exc}")
                _release_lock()
                raise
            finally:
                _release_lock()

        return _guarded_iter()


_GLOBAL_LLM_MANAGER: Optional[LocalLLMManager] = None
_GLOBAL_LLM_MANAGER_LOCK = threading.Lock()


def get_local_llm_manager(base_dir: Optional[str] = None) -> LocalLLMManager:
    global _GLOBAL_LLM_MANAGER
    with _GLOBAL_LLM_MANAGER_LOCK:
        if _GLOBAL_LLM_MANAGER is None:
            if not base_dir:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _GLOBAL_LLM_MANAGER = LocalLLMManager(base_dir)
        return _GLOBAL_LLM_MANAGER
