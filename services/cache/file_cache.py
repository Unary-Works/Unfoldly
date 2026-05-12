
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _default_data_dir() -> str:
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/Unfoldly")
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "Unfoldly")
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "Unfoldly")


def _cache_db_path() -> str:
    data_dir = os.environ.get("FILEAGENT_DATA_DIR") or _default_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "parse_cache.db")


class FileParseCache:

    VERSION = 2

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or _cache_db_path()
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parse_cache (
                    cache_key   TEXT PRIMARY KEY,
                    file_path   TEXT NOT NULL,
                    mtime       REAL NOT NULL,
                    file_size   INTEGER NOT NULL,
                    text        TEXT,
                    summary     TEXT,
                    category    TEXT,
                    ocr_text    TEXT,
                    image_desc  TEXT,
                    llm_model   TEXT,
                    cached_at   REAL NOT NULL,
                    version     INTEGER NOT NULL DEFAULT 1
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_file_path ON parse_cache(file_path)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Key computation
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(file_path: str, mtime: float, file_size: int) -> str:
        raw = f"{file_path}:{mtime:.3f}:{file_size}:v{FileParseCache.VERSION}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def _file_stat(self, file_path: str) -> Tuple[float, int]:
        try:
            st = os.stat(file_path)
            return st.st_mtime, st.st_size
        except OSError:
            return -1.0, -1

    # ------------------------------------------------------------------
    # Hit / Miss
    # ------------------------------------------------------------------

    def get(self, file_path: str) -> Optional[Dict[str, Any]]:
        mtime, file_size = self._file_stat(file_path)
        if mtime < 0:
            return None
        key = self._cache_key(file_path, mtime, file_size)
        with self._lock:
            try:
                with self._connect() as conn:
                    row = conn.execute(
                        "SELECT * FROM parse_cache WHERE cache_key = ?", (key,)
                    ).fetchone()
                    if row:
                        return dict(row)
            except Exception as e:
                logger.debug(f"[ParseCache] get error: {e}")
        return None

    def put(
        self,
        file_path: str,
        *,
        text: Optional[str] = None,
        summary: Optional[str] = None,
        category: Optional[str] = None,
        ocr_text: Optional[str] = None,
        image_desc: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        mtime, file_size = self._file_stat(file_path)
        if mtime < 0:
            return
        key = self._cache_key(file_path, mtime, file_size)
        with self._lock:
            try:
                with self._connect() as conn:
                    existing = conn.execute(
                        "SELECT * FROM parse_cache WHERE cache_key = ?", (key,)
                    ).fetchone()
                    if existing:
                        existing = dict(existing)
                        updates = {
                            "text": text if text is not None else existing.get("text"),
                            "summary": summary if summary is not None else existing.get("summary"),
                            "category": category if category is not None else existing.get("category"),
                            "ocr_text": ocr_text if ocr_text is not None else existing.get("ocr_text"),
                            "image_desc": image_desc if image_desc is not None else existing.get("image_desc"),
                            "llm_model": llm_model if llm_model is not None else existing.get("llm_model"),
                            "cached_at": time.time(),
                        }
                        conn.execute("""
                            UPDATE parse_cache
                            SET text=:text, summary=:summary, category=:category,
                                ocr_text=:ocr_text, image_desc=:image_desc,
                                llm_model=:llm_model, cached_at=:cached_at
                            WHERE cache_key=:cache_key
                        """, {**updates, "cache_key": key})
                    else:
                        conn.execute("""
                            INSERT INTO parse_cache
                              (cache_key, file_path, mtime, file_size,
                               text, summary, category, ocr_text, image_desc,
                               llm_model, cached_at, version)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """, (
                            key, file_path, mtime, file_size,
                            text, summary, category, ocr_text, image_desc,
                            llm_model, time.time(), self.VERSION,
                        ))
            except Exception as e:
                logger.debug(f"[ParseCache] put error: {e}")

    def invalidate(self, file_path: str) -> None:
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        "DELETE FROM parse_cache WHERE file_path = ?", (file_path,)
                    )
            except Exception as e:
                logger.debug(f"[ParseCache] invalidate error: {e}")

    def evict_missing(self) -> int:
        removed = 0
        with self._lock:
            try:
                with self._connect() as conn:
                    rows = conn.execute(
                        "SELECT DISTINCT file_path FROM parse_cache"
                    ).fetchall()
                    for row in rows:
                        fp = row[0]
                        if not os.path.exists(fp):
                            conn.execute(
                                "DELETE FROM parse_cache WHERE file_path = ?", (fp,)
                            )
                            removed += 1
            except Exception as e:
                logger.debug(f"[ParseCache] evict error: {e}")
        if removed:
            logger.info(f"[ParseCache] 清理了 {removed} 条过期缓存")
        return removed

    def stats(self) -> Dict[str, Any]:
        try:
            with self._connect() as conn:
                total = conn.execute("SELECT COUNT(*) FROM parse_cache").fetchone()[0]
                has_text = conn.execute(
                    "SELECT COUNT(*) FROM parse_cache WHERE text IS NOT NULL"
                ).fetchone()[0]
                has_summary = conn.execute(
                    "SELECT COUNT(*) FROM parse_cache WHERE summary IS NOT NULL"
                ).fetchone()[0]
                has_category = conn.execute(
                    "SELECT COUNT(*) FROM parse_cache WHERE category IS NOT NULL"
                ).fetchone()[0]
            return {
                "total": total,
                "has_text": has_text,
                "has_summary": has_summary,
                "has_category": has_category,
                "db_path": self._db_path,
            }
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_global_cache: Optional[FileParseCache] = None
_global_cache_lock = threading.Lock()


def get_parse_cache() -> FileParseCache:
    global _global_cache
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = FileParseCache()
    return _global_cache
