
import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

_logger: Optional[logging.Logger] = None


def _parse_level(name: str, default: int = logging.INFO) -> int:
    raw = (os.getenv(name) or "").strip().upper()
    if not raw:
        return default
    return getattr(logging, raw, default)


def _quiet_third_party() -> None:
    raw = (os.getenv("FILEAGENT_LOG_QUIET", "1") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return
    for name in (
        "urllib3",
        "urllib3.connectionpool",
        "httpx",
        "httpcore",
        "chromadb",
        "chromadb.telemetry",
        "watchfiles",
        "uvicorn.access",
        "uvicorn.error",
        "asyncio",
        "multipart",
        "fsspec",
        "filelock",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def configure_logging() -> logging.Logger:
    return get_logger()


def get_logger() -> logging.Logger:
    global _logger, _configured

    if _logger is not None:
        _logger.propagate = False
        return _logger

    # Prevent BrokenPipeError from crashing the loop if stderr is closed by Tauri
    logging.raiseExceptions = False

    logger = logging.getLogger("backend")
    level = _parse_level("FILEAGENT_LOG_LEVEL", logging.INFO)
    logger.setLevel(level)

    if logger.handlers:
        logger.propagate = False
        _quiet_third_party()
        _logger = logger
        return logger

    data_dir = (os.getenv("FILEAGENT_DATA_DIR") or "").strip()
    if data_dir:
        data_dir = os.path.abspath(os.path.expanduser(data_dir))
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")

    log_dir = os.path.join(data_dir, "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass

    log_file = os.path.join(log_dir, "backend.log")

    fmt = os.getenv(
        "FILEAGENT_LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ).strip()
    datefmt = os.getenv("FILEAGENT_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S").strip()
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt or None)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"[Logger] Failed to setup file handler: {e}", file=sys.stderr)

    logger.propagate = False

    _quiet_third_party()
    _logger = logger
    return logger


def get_child_logger(module_name: str) -> logging.Logger:
    get_logger()
    name = (module_name or "").strip()
    if name.startswith("backend."):
        child = logging.getLogger(name)
    else:
        child = logging.getLogger(f"backend.{name}")
    child.propagate = True
    return child


def log_info(msg: str):
    get_logger().info(msg)


def log_error(msg: str, exc_info=False):
    get_logger().error(msg, exc_info=exc_info)


def log_warning(msg: str):
    get_logger().warning(msg)


def log_debug(msg: str):
    get_logger().debug(msg)
