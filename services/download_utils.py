import os
import threading
import time
import json
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_child_logger

logger = get_child_logger(__name__)

ProgressCallback = Callable[[dict], None]


def download_model_from_hf(repo_id: str, local_dir: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
        logger.info(f"Trying to download {repo_id} from Hugging Face to {local_dir}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logger.info(f"Successfully downloaded {repo_id} from Hugging Face.")
        return True
    except Exception as e:
        logger.warning(f"Failed to download {repo_id} from Hugging Face: {e}")
        return False


def download_model_from_modelscope(repo_id: str, local_dir: str) -> bool:
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
        logger.info(f"Trying to download {repo_id} from ModelScope to {local_dir}...")
        ms_snapshot_download(
            model_id=repo_id,
            cache_dir=os.path.dirname(local_dir),
            local_dir=local_dir,
            ignore_file_pattern=[r'.*\.bin$', r'.*\.h5$', r'.*\.msgpack$', r'.*\.onnx$']
        )
        logger.info(f"Successfully downloaded {repo_id} from ModelScope.")
        return True
    except Exception as e:
        logger.warning(f"Failed to download {repo_id} from ModelScope: {e}")
        return False


def ensure_model_downloaded(repo_id: str, local_dir: str) -> bool:
    has_config = os.path.exists(os.path.join(local_dir, "config.json"))
    has_safetensors = os.path.exists(os.path.join(local_dir, "model.safetensors"))
    has_bin = os.path.exists(os.path.join(local_dir, "pytorch_model.bin"))

    if has_config and (has_safetensors or has_bin):
        logger.info(f"Model {repo_id} seems already downloaded at {local_dir}. Skipping download.")
        return True

    logger.info(f"Model {repo_id} not found/incomplete at {local_dir}. Starting download...")
    os.makedirs(local_dir, exist_ok=True)

    logger.info("Trying ModelScope first...")
    if download_model_from_modelscope(repo_id, local_dir):
        return True

    logger.info("ModelScope failed/missing, falling back to Hugging Face...")
    if download_model_from_hf(repo_id, local_dir):
        return True

    return False


_gguf_size_cache: Dict[str, int] = {}
_gguf_metadata_cache: Dict[str, Dict[str, Any]] = {}


def _gguf_size_hint_path(path: str) -> str:
    return f"{path}.size.json"


def _read_gguf_size_hint(path: str) -> int:
    try:
        hp = _gguf_size_hint_path(path)
        if not os.path.isfile(hp):
            return 0
        with open(hp, "r", encoding="utf-8") as f:
            data = json.load(f)
        sz = int(data.get("size_bytes") or 0)
        return sz if sz > 0 else 0
    except Exception:
        return 0


def _write_gguf_size_hint(path: str, size_bytes: int) -> None:
    try:
        sz = int(size_bytes or 0)
        if sz <= 0:
            return
        hp = _gguf_size_hint_path(path)
        tmp = f"{hp}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"size_bytes": sz}, f, ensure_ascii=False)
        os.replace(tmp, hp)
    except Exception:
        pass


def _query_gguf_file_size(repo_id: str, filename: str) -> int:
    """Query single file size via ModelScope API (.cn then .ai), then HuggingFace API. Results are cached."""
    cache_key = f"{repo_id}/{filename}"
    if cache_key in _gguf_size_cache:
        return _gguf_size_cache[cache_key]

    import urllib.request
    import json as _json
    for domain in ("www.modelscope.cn", "www.modelscope.ai"):
        try:
            url = f"https://{domain}/api/v1/models/{repo_id}/repo/files?Recursive=true"
            req = urllib.request.Request(url, headers={"User-Agent": "fileagent/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
            for f in (data.get("Data", {}).get("Files", []) or []):
                if f.get("Name") == filename or f.get("Path") == filename:
                    sz = int(f.get("Size", 0))
                    if sz > 0:
                        _gguf_size_cache[cache_key] = sz
                        return sz
        except Exception:
            continue
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, files_metadata=True)
        for sib in (info.siblings or []):
            if getattr(sib, "rfilename", "") == filename:
                sz = getattr(sib, "size", None) or getattr(sib, "lfs", {})
                if isinstance(sz, int) and sz > 0:
                    _gguf_size_cache[cache_key] = sz
                    return sz
                if isinstance(sz, dict):
                    val = int(sz.get("size", 0))
                    if val > 0:
                        _gguf_size_cache[cache_key] = val
                    return val
    except Exception:
        pass
    return 0


def _query_gguf_file_metadata_modelscope(repo_id: str, filename: str) -> Dict[str, Any]:
    cache_key = f"modelscope:{repo_id}/{filename}"
    if cache_key in _gguf_metadata_cache:
        return _gguf_metadata_cache[cache_key]

    import urllib.request
    import json as _json

    domains: List[str] = []
    for raw in (os.environ.get("MODELSCOPE_DOMAIN"), "www.modelscope.cn", "www.modelscope.ai"):
        domain = str(raw or "").strip().replace("https://", "").replace("http://", "").strip("/")
        if domain and domain not in domains:
            domains.append(domain)

    for domain in domains:
        try:
            url = f"https://{domain}/api/v1/models/{repo_id}/repo/files?Recursive=true"
            req = urllib.request.Request(url, headers={"User-Agent": "fileagent/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
            for f in (data.get("Data", {}).get("Files", []) or []):
                if f.get("Name") == filename or f.get("Path") == filename:
                    meta = {
                        "size": int(f.get("Size") or 0),
                        "sha256": str(f.get("Sha256") or "").lower(),
                        "domain": domain,
                    }
                    _gguf_metadata_cache[cache_key] = meta
                    return meta
        except Exception as e:
            logger.debug(f"Failed to query ModelScope metadata for {repo_id}/{filename} on {domain}: {e}")
            continue

    _gguf_metadata_cache[cache_key] = {}
    return {}


def _query_gguf_file_metadata_hf(repo_id: str, filename: str) -> Dict[str, Any]:
    cache_key = f"hf:{repo_id}/{filename}"
    if cache_key in _gguf_metadata_cache:
        return _gguf_metadata_cache[cache_key]

    try:
        from huggingface_hub import HfApi

        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        info = HfApi(endpoint=endpoint).model_info(repo_id, files_metadata=True)
        for sib in (info.siblings or []):
            if getattr(sib, "rfilename", "") != filename:
                continue
            size = getattr(sib, "size", None)
            sha256 = ""
            lfs = getattr(sib, "lfs", None)
            if lfs is not None:
                size = getattr(lfs, "size", size)
                sha256 = str(getattr(lfs, "sha256", "") or "").lower()
            meta = {"size": int(size or 0), "sha256": sha256}
            _gguf_metadata_cache[cache_key] = meta
            return meta
    except Exception as e:
        logger.debug(f"Failed to query HuggingFace metadata for {repo_id}/{filename}: {e}")

    _gguf_metadata_cache[cache_key] = {}
    return {}


def _gguf_metadata_matches_modelscope(repo_id: str, filename: str) -> bool:
    ms = _query_gguf_file_metadata_modelscope(repo_id, filename)
    hf = _query_gguf_file_metadata_hf(repo_id, filename)
    if not ms or not hf:
        logger.info(f"HF source not verified for {repo_id}/{filename}: missing metadata")
        return False
    same_size = int(ms.get("size") or 0) == int(hf.get("size") or 0)
    ms_sha = str(ms.get("sha256") or "").lower()
    hf_sha = str(hf.get("sha256") or "").lower()
    same_sha = bool(ms_sha and hf_sha and ms_sha == hf_sha)
    if not (same_size and same_sha):
        logger.info(
            f"HF source rejected for {repo_id}/{filename}: ModelScope and HF differ "
            f"(ms_size={int(ms.get('size') or 0)}, hf_size={int(hf.get('size') or 0)})"
        )
        return False
    return True


def _gguf_probe_url(source: str, repo_id: str, filename: str) -> str:
    import urllib.parse

    if source == "hf":
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        return f"{endpoint}/{repo_id}/resolve/main/{urllib.parse.quote(filename, safe='/')}"
    if source == "modelscope":
        # Probe the same domain used by _do_gguf_download_modelscope.
        return (
            f"https://www.modelscope.cn/api/v1/models/{repo_id}/repo?"
            f"Revision=master&FilePath={urllib.parse.quote(filename, safe='')}"
        )
    return ""


def _test_gguf_source_speed(source: str, repo_id: str, filename: str) -> Dict[str, float]:
    import urllib.request
    import time as _time

    try:
        probe_bytes = int(os.getenv("FILEAGENT_SOURCE_PROBE_BYTES", "262144") or 262144)
    except Exception:
        probe_bytes = 262144
    probe_bytes = max(16 * 1024, min(probe_bytes, 1024 * 1024))

    url = _gguf_probe_url(source, repo_id, filename)
    if not url:
        return {"ok": 0.0, "latency_ms": -1.0, "bytes_per_sec": 0.0}

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "fileagent/1.0",
            "Range": f"bytes=0-{probe_bytes - 1}",
        },
        method="GET",
    )
    start_t = _time.time()
    try:
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            read_bytes = 0
            while read_bytes < probe_bytes:
                chunk = resp.read(min(64 * 1024, probe_bytes - read_bytes))
                if not chunk:
                    break
                read_bytes += len(chunk)
        elapsed = max(_time.time() - start_t, 0.001)
        return {
            "ok": 1.0,
            "latency_ms": elapsed * 1000,
            "bytes_per_sec": (read_bytes / elapsed) if read_bytes > 0 else 0.0,
        }
    except Exception:
        return {"ok": 0.0, "latency_ms": -1.0, "bytes_per_sec": 0.0}


def _get_gguf_source_order(repo_id: str, filename: str) -> List[str]:
    sources = ["modelscope"]
    if _gguf_metadata_matches_modelscope(repo_id, filename):
        sources.append("hf")

    if len(sources) <= 1:
        logger.info(f"GGUF auto source order for {repo_id}/{filename}: {sources}")
        return sources

    results = []
    for source in sources:
        probe = _test_gguf_source_speed(source, repo_id, filename)
        results.append({"source": source, **probe})
        logger.info(
            f"GGUF source speed test: source={source} file={filename} "
            f"latency={probe.get('latency_ms', -1):.2f}ms speed={probe.get('bytes_per_sec', 0):.0f}B/s"
        )

    valid = [r for r in results if r.get("ok", 0.0) > 0 and r.get("latency_ms", -1.0) >= 0]
    valid.sort(key=lambda x: (0 if x.get("bytes_per_sec", 0) > 0 else 1, -x.get("bytes_per_sec", 0), x.get("latency_ms", 999999)))
    order = [r["source"] for r in valid]
    order.extend([s for s in sources if s not in order])
    logger.info(f"GGUF auto source order for {repo_id}/{filename}: {order}")
    return order


def _probe_file_size(local_dir: str, filename: str) -> int:
    """Probe downloaded size by checking common locations (incl. ModelScope temp dirs)."""
    candidates = [
        os.path.join(local_dir, filename),
        os.path.join(local_dir, f"{filename}.part"),
        os.path.join(local_dir, "._____temp", filename),
        os.path.join(local_dir, ".msc", filename),
    ]
    best = 0
    for p in candidates:
        try:
            if os.path.exists(p):
                best = max(best, os.path.getsize(p))
        except Exception:
            pass
    if best > 0:
        return best
    skip_suffixes = (".downloading", ".tmp", ".part")
    try:
        for root, _dirs, files in os.walk(local_dir):
            for fn in files:
                if any(fn.endswith(s) for s in skip_suffixes):
                    continue
                if fn == filename or fn.startswith(filename) or filename in fn:
                    try:
                        best = max(best, os.path.getsize(os.path.join(root, fn)))
                    except Exception:
                        pass
    except Exception:
        pass
    return best


def _is_transient_path(path: str) -> bool:
    p = str(path or "").replace("\\", "/").lower()
    if any(seg in p for seg in ("/._____temp/", "/.msc/", "/.mv/")):
        return True
    if p.endswith((".downloading", ".tmp", ".part", ".incomplete")):
        return True
    return False

def _is_gguf_complete(path: str, expected_size: int = 0) -> bool:
    """Check if a GGUF file at *path* is complete by comparing with remote size."""
    try:
        if not os.path.isfile(path):
            return False
        if _is_transient_path(path):
            return False
        sz = os.path.getsize(path)
        if sz < 4:
            return False
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return False
        if expected_size > 0:
            return 0.99 <= sz / expected_size <= 1.01
        # If expected size is unknown, but it has a GGUF header and is large enough (e.g. > 1MB), we assume it is valid
        return sz >= 1_000_000
    except Exception:
        return False


def _is_gguf_payload_complete(path: str, expected_size: int = 0) -> bool:
    """Validate GGUF bytes even when the path itself is a transient .part file."""
    try:
        if not os.path.isfile(path):
            return False
        sz = os.path.getsize(path)
        if sz < 4:
            return False
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return False
        if expected_size > 0:
            return 0.99 <= sz / expected_size <= 1.01
        return sz >= 1_000_000
    except Exception:
        return False


def _is_gguf_loadable(path: str) -> bool:
    """
    Best-effort runtime validation when remote size is unavailable.
    This catches many truncated GGUF cases that still have a valid header.
    """
    try:
        from llama_cpp import Llama  # type: ignore
        llm = Llama(
            model_path=path,
            n_ctx=32,
            n_batch=32,
            n_threads=1,
            embeddings=True,
            verbose=False,
        )
        try:
            del llm
        except Exception:
            pass
        return True
    except Exception as e:
        logger.warning(f"GGUF runtime validation failed for {path}: {e}")
        return False


def _is_cancelled(should_cancel: Optional[Callable[[], bool]]) -> bool:
    if should_cancel is None:
        return False
    try:
        return bool(should_cancel())
    except Exception:
        return False


class ResumableDownloadUnavailable(RuntimeError):
    """Direct resumable download URL is unavailable; callers may fall back to SDK download."""


class ResumableDownloadInterrupted(RuntimeError):
    """Download was interrupted after a partial file existed; preserve it for the next retry."""


class ResumableDownloadCancelled(RuntimeError):
    """Download was cancelled by the caller; preserve any partial file."""


def _safe_size(path: str) -> int:
    try:
        return int(os.path.getsize(path)) if os.path.isfile(path) else 0
    except Exception:
        return 0


def _content_range_total(value: str) -> int:
    try:
        if "/" not in value:
            return 0
        total = value.rsplit("/", 1)[1].strip()
        return int(total) if total and total != "*" else 0
    except Exception:
        return 0


def _http_error_content_range_total(error: Exception) -> int:
    try:
        headers = getattr(error, "headers", None) or getattr(error, "hdrs", None)
        content_range = headers.get("Content-Range") if headers is not None else ""
        return _content_range_total(content_range or "")
    except Exception:
        return 0


def _has_gguf_prefix(path: str) -> bool:
    try:
        sz = _safe_size(path)
        if sz <= 0:
            return True
        if sz < 4:
            return True
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except Exception:
        return False


def _gguf_download_urls(
    source: str,
    repo_id: str,
    filename: str,
    *,
    modelscope_domain: Optional[str] = None,
) -> List[str]:
    from urllib.parse import quote

    quoted_filename = quote(str(filename).lstrip("/"), safe="/")
    source = (source or "").strip().lower()
    if source == "hf":
        endpoint = (os.getenv("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
        return [f"{endpoint}/{repo_id}/resolve/main/{quoted_filename}"]

    if source == "modelscope":
        domains: List[str] = []
        for domain in (
            modelscope_domain,
            os.getenv("MODELSCOPE_DOMAIN"),
            "www.modelscope.cn",
            "www.modelscope.ai",
        ):
            domain = (domain or "").strip().removeprefix("https://").removeprefix("http://").rstrip("/")
            if domain and domain not in domains:
                domains.append(domain)
        urls: List[str] = []
        for domain in domains:
            for revision in ("master", "main"):
                urls.append(f"https://{domain}/models/{repo_id}/resolve/{revision}/{quoted_filename}")
        return urls

    return []


def download_gguf_with_resume(
    source: str,
    repo_id: str,
    filename: str,
    target_path: str,
    *,
    expected_size: int = 0,
    on_progress: Optional[ProgressCallback] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    modelscope_domain: Optional[str] = None,
) -> bool:
    """
    Download a GGUF file with an app-owned .part file and HTTP Range resume.
    The .part file is intentionally preserved on cancellation and network errors.
    """
    urls = _gguf_download_urls(source, repo_id, filename, modelscope_domain=modelscope_domain)
    if not urls:
        raise ResumableDownloadUnavailable(f"no_direct_url:{source}")

    target_path = os.path.abspath(target_path)
    part_path = f"{target_path}.part"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    expected_size = int(expected_size or 0)

    def _cancelled() -> bool:
        return _is_cancelled(should_cancel)

    def _emit(downloaded: int, total: int, speed: float = 0.0, force: bool = False) -> None:
        if on_progress is None:
            return
        now = time.time()
        if not force and now - _emit.last_ts < 0.5:
            return
        _emit.last_ts = now
        total = int(max(0, total))
        downloaded = int(max(0, downloaded))
        pct = (downloaded / total) * 100.0 if total > 0 else 0.0
        eta = ((total - downloaded) / speed) if speed > 0 and total > downloaded else 0.0
        on_progress({
            "downloaded_bytes": min(downloaded, total) if total > 0 else downloaded,
            "total_bytes": total,
            "percent": max(0.0, min(99.9, pct)),
            "speed_bytes_per_sec": float(max(0.0, speed)),
            "eta_seconds": float(max(0.0, eta)),
        })
    _emit.last_ts = 0.0  # type: ignore[attr-defined]

    def _promote_if_complete(total: int, *, exact_size: bool) -> bool:
        complete_size = int(total or 0)
        if complete_size <= 0 and exact_size:
            complete_size = _safe_size(part_path)
        if complete_size <= 0:
            complete_size = expected_size
        if _is_gguf_payload_complete(part_path, complete_size):
            os.replace(part_path, target_path)
            final_size = _safe_size(target_path)
            _write_gguf_size_hint(target_path, final_size)
            _emit(final_size, max(final_size, complete_size), 0.0, force=True)
            logger.info(f"GGUF direct download completed: {target_path} ({final_size / 1e6:.1f} MB)")
            return True
        return False

    if _is_gguf_complete(target_path, expected_size):
        size = _safe_size(target_path)
        _write_gguf_size_hint(target_path, size)
        _emit(size, max(expected_size, size), 0.0, force=True)
        return True

    target_size = _safe_size(target_path)
    part_size = _safe_size(part_path)
    if target_size > 0 and not _is_gguf_complete(target_path, expected_size):
        if part_size <= 0 or target_size > part_size:
            try:
                os.replace(target_path, part_path)
                part_size = _safe_size(part_path)
            except Exception:
                pass

    if part_size > 0 and not _has_gguf_prefix(part_path):
        logger.warning(f"Discarding invalid GGUF partial without header: {part_path}")
        try:
            os.unlink(part_path)
        except OSError:
            pass
        part_size = 0

    if expected_size > 0 and part_size > expected_size + max(1_048_576, int(expected_size * 0.02)):
        logger.warning(
            f"Discarding oversized GGUF partial: path={part_path}, partial={part_size}, expected={expected_size}"
        )
        try:
            os.unlink(part_path)
        except OSError:
            pass
        part_size = 0

    if part_size > 0 and _promote_if_complete(expected_size, exact_size=expected_size > 0):
        return True

    total_size = max(expected_size, part_size)
    _emit(part_size, total_size, 0.0, force=True)

    import urllib.error
    import urllib.request

    last_unavailable: Optional[Exception] = None
    for url in urls:
        downloaded = _safe_size(part_path)
        if _cancelled():
            raise ResumableDownloadCancelled("gguf_download_cancelled")

        for attempt in range(1, 3):
            if _cancelled():
                raise ResumableDownloadCancelled("gguf_download_cancelled")

            downloaded = _safe_size(part_path)
            headers = {"User-Agent": "Unfoldly/1.0", "Accept-Encoding": "identity"}
            resume_allowed = downloaded > 0
            if resume_allowed:
                headers["Range"] = f"bytes={downloaded}-"

            req = urllib.request.Request(url, headers=headers)
            start_offset = downloaded
            start_ts = time.time()
            got_response = False
            got_bytes = False
            response_total_exact = False
            try:
                logger.info(
                    f"Downloading GGUF with Range resume: source={source} file={filename} "
                    f"offset={downloaded} attempt={attempt}/2"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    got_response = True
                    code = getattr(resp, "status", None) or resp.getcode()
                    if resume_allowed and code != 206:
                        logger.warning(f"Server ignored Range for {filename}; restarting this partial from 0.")
                        downloaded = 0
                        mode = "wb"
                    else:
                        mode = "ab" if resume_allowed else "wb"

                    start_offset = downloaded
                    content_range_total = _content_range_total(resp.headers.get("Content-Range") or "")
                    content_length = int(resp.headers.get("Content-Length") or 0)
                    if content_range_total > 0:
                        total_size = max(expected_size, content_range_total)
                        response_total_exact = True
                    elif content_length > 0:
                        total_size = max(expected_size, downloaded + content_length)
                        response_total_exact = True

                    with open(part_path, mode) as f:
                        while True:
                            if _cancelled():
                                raise ResumableDownloadCancelled("gguf_download_cancelled")
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            got_bytes = True
                            f.write(chunk)
                            downloaded += len(chunk)
                            elapsed = max(0.001, time.time() - start_ts)
                            speed = max(0, downloaded - start_offset) / elapsed
                            _emit(downloaded, total_size, speed)

                if _promote_if_complete(total_size, exact_size=response_total_exact or expected_size > 0):
                    return True

                actual = _safe_size(part_path)
                if expected_size > 0 and actual < expected_size:
                    raise ResumableDownloadInterrupted(
                        f"incomplete_gguf_download:{filename}:actual={actual}:expected={expected_size}"
                    )
                if actual > 0:
                    raise ResumableDownloadInterrupted(f"incomplete_gguf_download:{filename}:actual={actual}")
                raise ResumableDownloadUnavailable(f"empty_gguf_download:{filename}")

            except ResumableDownloadCancelled:
                raise
            except urllib.error.HTTPError as e:
                if e.code == 416:
                    range_total = _http_error_content_range_total(e)
                    if _promote_if_complete(range_total or expected_size, exact_size=True):
                        return True
                    try:
                        os.unlink(part_path)
                    except OSError:
                        pass
                    downloaded = 0
                    _emit(0, max(expected_size, range_total), 0.0, force=True)
                    continue
                if e.code in (401, 403, 404) and (not got_bytes) and _safe_size(part_path) <= 0:
                    last_unavailable = e
                    break
                raise ResumableDownloadInterrupted(f"gguf_download_http_error:{e.code}:{filename}") from e
            except Exception as e:
                if _safe_size(part_path) > 0 or got_bytes or resume_allowed:
                    raise ResumableDownloadInterrupted(f"gguf_download_interrupted:{filename}:{e}") from e
                if got_response:
                    raise ResumableDownloadInterrupted(f"gguf_download_interrupted:{filename}:{e}") from e
                last_unavailable = e
                break

    raise ResumableDownloadUnavailable(str(last_unavailable or f"direct_download_unavailable:{filename}"))


def ensure_gguf_downloaded(
    repo_id: str,
    filename: str,
    local_path: str,
    on_progress: Optional[ProgressCallback] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> bool:
    """
    Download a single GGUF file via ModelScope SDK (with HuggingFace fallback).
    SDK handles resume / temp caching internally — we only check the final result.
    """
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    remote_expected = _query_gguf_file_size(repo_id, filename)
    hint_expected = max(
        _read_gguf_size_hint(local_path),
        _read_gguf_size_hint(os.path.join(local_dir, filename)),
    )
    expected_size = max(int(remote_expected or 0), int(hint_expected or 0))

    if _is_cancelled(should_cancel):
        return False

    if _is_gguf_complete(local_path, expected_size):
        logger.info(f"GGUF already complete: {local_path} ({os.path.getsize(local_path) / 1e6:.1f} MB)")
        _write_gguf_size_hint(local_path, os.path.getsize(local_path))
        return True

    target = os.path.join(local_dir, filename)
    if target != local_path and _is_gguf_complete(target, expected_size):
        import shutil
        shutil.copy2(target, local_path)
        logger.info(f"GGUF found at {target}, copied to {local_path}")
        _write_gguf_size_hint(local_path, os.path.getsize(local_path))
        return True

    source_order = _get_gguf_source_order(repo_id, filename)

    direct_enabled = (os.getenv("FILEAGENT_DIRECT_GGUF_DOWNLOAD", "1") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    if direct_enabled:
        for direct_source in source_order:
            if _is_cancelled(should_cancel):
                return False
            try:
                ok = download_gguf_with_resume(
                    direct_source,
                    repo_id,
                    filename,
                    target,
                    expected_size=expected_size,
                    on_progress=on_progress,
                    should_cancel=should_cancel,
                    modelscope_domain="www.modelscope.cn" if direct_source == "modelscope" else None,
                )
                if ok:
                    if os.path.abspath(target) != os.path.abspath(local_path):
                        import shutil
                        shutil.copy2(target, local_path)
                    final_sz = os.path.getsize(local_path)
                    _write_gguf_size_hint(local_path, final_sz)
                    _write_gguf_size_hint(target, final_sz)
                    return True
            except ResumableDownloadCancelled:
                logger.info(f"Direct GGUF download cancelled: {filename}")
                return False
            except ResumableDownloadInterrupted as e:
                logger.warning(f"Direct GGUF download interrupted; keeping partial for resume: {e}")
                return False
            except ResumableDownloadUnavailable as e:
                logger.info(f"Direct GGUF download unavailable via {direct_source}; trying next path: {e}")
            except Exception as e:
                logger.warning(f"Direct GGUF download failed via {direct_source}; keeping SDK fallback available: {e}")

    state = {"total_bytes": expected_size, "stop": False}

    def _watch():
        prev_sz = 0
        prev_ts = time.time()
        last_speed = 0.0
        stale_ticks = 0
        retry_size_count = 0
        while not state["stop"]:
            if _is_cancelled(should_cancel):
                return
            sz = _probe_file_size(local_dir, filename)
            now = time.time()
            dt = now - prev_ts
            if dt > 0.1 and sz > prev_sz:
                last_speed = (sz - prev_sz) / dt
                stale_ticks = 0
            else:
                stale_ticks += 1
                if stale_ticks > 10:
                    last_speed = 0
            prev_sz, prev_ts = sz, now

            tb = state["total_bytes"]
            if tb <= 0 and retry_size_count < 3:
                retry_size_count += 1
                tb = _query_gguf_file_size(repo_id, filename)
                if tb > 0:
                    state["total_bytes"] = tb

            if on_progress:
                display_sz = min(sz, tb) if tb > 0 else sz
                pct = (display_sz / tb * 100) if tb > 0 else 0
                remaining = max(tb - sz, 0) if tb > 0 else 0
                eta = remaining / last_speed if last_speed > 0 and remaining > 0 else 0
                on_progress({
                    "downloaded_bytes": display_sz,
                    "total_bytes": tb,
                    "percent": min(pct, 99.9),
                    "speed_bytes_per_sec": last_speed,
                    "eta_seconds": eta,
                })
            time.sleep(1.0)

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()

    ok = False
    try:
        if _is_cancelled(should_cancel):
            return False
        for source in source_order:
            if source == "hf":
                ok = _do_gguf_download_hf(repo_id, filename, local_dir, should_cancel=should_cancel)
            else:
                ok = _do_gguf_download_modelscope(repo_id, filename, local_dir, should_cancel=should_cancel)
            if ok:
                break
            if _is_cancelled(should_cancel):
                return False
    finally:
        state["stop"] = True
        try:
            watcher.join(timeout=2.0)
        except Exception:
            pass

    if _is_cancelled(should_cancel):
        return False

    if ok:
        final_target = os.path.join(local_dir, filename)
        final_expected = int(expected_size or 0)
        if final_expected <= 0:
            final_expected = _query_gguf_file_size(repo_id, filename)
            if final_expected > 0:
                state["total_bytes"] = final_expected

        final_size = 0
        try:
            if os.path.isfile(final_target):
                final_size = int(os.path.getsize(final_target))
        except Exception:
            final_size = 0

        if final_expected > 0:
            if not _is_gguf_complete(final_target, final_expected):
                ratio = (final_size / final_expected) if final_size > 0 else 0.0
                logger.warning(
                    f"Downloaded file size mismatch, trying runtime validation fallback: "
                    f"path={final_target}, actual={final_size}, expected={final_expected}, ratio={ratio:.3f}"
                )
                runtime_ok = False
                try:
                    if (
                        final_size >= 1_000_000
                        and os.path.isfile(final_target)
                        and (not _is_transient_path(final_target))
                    ):
                        with open(final_target, "rb") as _vf:
                            has_header = (_vf.read(4) == b"GGUF")
                        if has_header:
                            runtime_ok = _is_gguf_loadable(final_target)
                except Exception:
                    runtime_ok = False

                if runtime_ok:
                    logger.warning(
                        f"Accepting GGUF despite size mismatch because runtime validation passed: {final_target}"
                    )
                    state["total_bytes"] = max(int(state.get("total_bytes") or 0), int(final_size or 0))
                else:
                    logger.error(f"Downloaded file failed size validation: {final_target}")
                    ok = False
        elif os.path.isfile(final_target):
            try:
                with open(final_target, "rb") as _vf:
                    if _vf.read(4) != b"GGUF":
                        logger.error(f"Downloaded file has invalid GGUF header: {final_target}")
                        ok = False
                if ok and (not _is_gguf_loadable(final_target)):
                    logger.error(f"Downloaded file failed runtime validation: {final_target}")
                    ok = False
            except Exception:
                ok = False

    if ok:
        final_target = os.path.join(local_dir, filename)
        if os.path.abspath(final_target) != os.path.abspath(local_path):
            import shutil
            shutil.copy2(final_target, local_path)
        try:
            final_sz2 = os.path.getsize(local_path)
            _write_gguf_size_hint(local_path, final_sz2)
            _write_gguf_size_hint(final_target, final_sz2)
        except Exception:
            pass
        if on_progress:
            final_sz = os.path.getsize(local_path)
            on_progress({
                "downloaded_bytes": final_sz,
                "total_bytes": max(state["total_bytes"], final_sz),
                "percent": 100.0,
                "speed_bytes_per_sec": 0,
                "eta_seconds": 0,
            })
    else:
        logger.error(f"Failed to download GGUF file {filename} from all sources.")
    return ok


def _find_file_in_tree(local_dir: str, filename: str, min_size: int = 0) -> Optional[str]:
    """Search local_dir recursively for `filename` with size >= min_size."""
    try:
        for root, _dirs, files in os.walk(local_dir):
            if filename in files:
                p = os.path.join(root, filename)
                try:
                    if os.path.getsize(p) >= min_size:
                        return p
                except OSError:
                    pass
    except Exception:
        pass
    return None


def _ensure_at_target(result_path: str, local_dir: str, filename: str) -> bool:
    """Only accept success when the target file itself exists or can be securely copied from result_path."""
    target = os.path.join(local_dir, filename)
    try:
        if os.path.isfile(target) and os.path.getsize(target) > 0:
            return True
    except Exception:
        pass

    sources = []
    try:
        if result_path and os.path.isfile(result_path) and os.path.getsize(result_path) > 0:
            sources.append(result_path)
    except Exception:
        pass

    found = _find_file_in_tree(local_dir, filename, min_size=1)
    if found and found not in sources:
        sources.append(found)

    last_err = None
    for src in sources:
        try:
            if _is_transient_path(src):
                continue
            if os.path.getsize(src) <= 0:
                continue
            if os.path.abspath(src) != os.path.abspath(target):
                import shutil
                shutil.copy2(src, target)
            if os.path.isfile(target) and os.path.getsize(target) > 0:
                return True
        except Exception as e:
            last_err = e
            continue

    if last_err is not None:
        logger.warning(f"Failed to place downloaded file at target {target}: {last_err}")
    return False


def _do_gguf_download_modelscope(
    repo_id: str,
    filename: str,
    local_dir: str,
    *,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> bool:
    try:
        from modelscope.hub.file_download import model_file_download
    except Exception as e:
        logger.warning(f"Failed to import ModelScope downloader: {e}")
        return False

    try:
        attempts = int(os.getenv("FILEAGENT_MODELSCOPE_DOWNLOAD_ATTEMPTS", "2") or 2)
    except Exception:
        attempts = 2
    attempts = max(1, min(attempts, 5))

    for attempt in range(1, attempts + 1):
        if _is_cancelled(should_cancel):
            logger.info(f"ModelScope download cancelled before attempt {attempt}: {filename}")
            return False
        try:
            old_domain = os.environ.get("MODELSCOPE_DOMAIN")
            os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.cn"
            try:
                logger.info(
                    f"Downloading {filename} from ModelScope (modelscope.cn) {repo_id} "
                    f"[attempt {attempt}/{attempts}] ..."
                )
                result_path = model_file_download(model_id=repo_id, file_path=filename, local_dir=local_dir)
            finally:
                if old_domain is not None:
                    os.environ["MODELSCOPE_DOMAIN"] = old_domain
                else:
                    os.environ.pop("MODELSCOPE_DOMAIN", None)

            ok = _ensure_at_target(result_path or "", local_dir, filename)
            if ok:
                logger.info(f"Successfully downloaded {filename} from ModelScope -> {os.path.join(local_dir, filename)}")
                return True

            logger.warning(
                f"ModelScope SDK returned {result_path} but file not at target "
                f"[attempt {attempt}/{attempts}]"
            )
        except Exception as e:
            logger.warning(f"Failed to download {filename} from ModelScope [attempt {attempt}/{attempts}]: {e}")

        if attempt < attempts:
            if _is_cancelled(should_cancel):
                logger.info(f"ModelScope download cancelled after attempt {attempt}: {filename}")
                return False
            time.sleep(1.5)

    return False


def _do_gguf_download_hf(
    repo_id: str,
    filename: str,
    local_dir: str,
    *,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> bool:
    if _is_cancelled(should_cancel):
        logger.info(f"HuggingFace download cancelled before start: {filename}")
        return False
    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"Downloading {filename} from HuggingFace {repo_id} ...")
        result_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        ok = _ensure_at_target(result_path or "", local_dir, filename)
        if ok:
            logger.info(f"Successfully downloaded {filename} from HuggingFace -> {os.path.join(local_dir, filename)}")
        else:
            logger.warning(f"HF SDK returned {result_path} but file not at target")
        return ok
    except Exception as e:
        logger.warning(f"Failed to download {filename} from HuggingFace: {e}")
        return False
