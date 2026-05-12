import os
import sqlite3
import threading
from typing import Dict, Any, List, Optional
from utils.logger import get_logger

logger = get_logger()

class MetadataIndex:
    """
    Sidecar SQLite Metadata Index for Unfoldly.
    Maintains a structured, fast-to-query cache of file metadata alongside ChromaDB.
    Greatly speeds up `count_by_category`, keyword filtering, and filepath matching.
    """
    
    def __init__(self, db_dir: str):
        self.db_path = os.path.join(db_dir, "metadata_sidecar.sqlite")
        self._lock = threading.RLock()
        self._init_db()
        
    def _get_connection(self):
        # sqlite3 needs check_same_thread=False if shared across threads, or thread-local connections.
        # Since this is a lightweight wrapper, we create connection per query for thread safety.
        # It's fast enough for local SQLite.
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = self._get_connection()
            try:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    file_path TEXT PRIMARY KEY,
                    file_name TEXT,
                    file_name_no_ext TEXT,
                    file_extension TEXT,
                    file_size_kb REAL,
                    modified_time TEXT,
                    parent_folder TEXT,
                    doc_category TEXT,
                    doc_summary TEXT,
                    doc_summary_model_id TEXT,
                    doc_summary_saved_at TEXT,
                    keywords TEXT,
                    entities TEXT
                )
                """)
                # Create indices for hot fields
                conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON file_metadata(doc_category)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_folder ON file_metadata(parent_folder)")
                conn.commit()
            except Exception as e:
                logger.error(f"[MetadataIndex] Failed to initialize db: {e}")
            finally:
                conn.close()

    def upsert_file_metadata(self, file_path: str, metadata: Dict[str, Any]):
        """Upsert metadata for a specific file."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                INSERT INTO file_metadata (
                    file_path, file_name, file_name_no_ext, file_extension,
                    file_size_kb, modified_time, parent_folder, doc_category,
                    doc_summary, doc_summary_model_id, doc_summary_saved_at,
                    keywords, entities
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_name=excluded.file_name,
                    file_name_no_ext=excluded.file_name_no_ext,
                    file_extension=excluded.file_extension,
                    file_size_kb=excluded.file_size_kb,
                    modified_time=excluded.modified_time,
                    parent_folder=excluded.parent_folder,
                    doc_category=excluded.doc_category,
                    doc_summary=excluded.doc_summary,
                    doc_summary_model_id=excluded.doc_summary_model_id,
                    doc_summary_saved_at=excluded.doc_summary_saved_at,
                    keywords=excluded.keywords,
                    entities=excluded.entities
                """, (
                    file_path,
                    metadata.get("file_name", ""),
                    metadata.get("file_name_no_ext", ""),
                    metadata.get("file_extension", ""),
                    metadata.get("file_size_kb", 0.0),
                    metadata.get("modified_time", ""),
                    metadata.get("parent_folder", ""),
                    metadata.get("doc_category", "other"),
                    metadata.get("doc_summary", ""),
                    metadata.get("doc_summary_model_id", ""),
                    metadata.get("doc_summary_saved_at", ""),
                    metadata.get("keywords", ""),
                    metadata.get("entities", "")
                ))
                conn.commit()
            except Exception as e:
                logger.error(f"[MetadataIndex] Failed to upsert metadata for {os.path.basename(file_path)}: {e}")
            finally:
                conn.close()

    def count_by_category(self, active_paths: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Count documents grouped by category.
        Replaces the slow ChromaDB sequential scan.
        """
        conn = self._get_connection()
        try:
            if not active_paths:
                cursor = conn.execute("SELECT doc_category, COUNT(*) as cnt FROM file_metadata GROUP BY doc_category")
                return {row["doc_category"]: row["cnt"] for row in cursor}
            else:
                # Optimized chunk batching for active_paths (SQLite limits vars to 999)
                results = {}
                # Active paths matching usually requires exact matches or LIKE for directories
                # If active_paths are exact file_paths or prefixes:
                for path in active_paths:
                    cursor = conn.execute("SELECT doc_category, COUNT(*) as cnt FROM file_metadata WHERE file_path LIKE ? GROUP BY doc_category", (f"{path}%",))
                    for row in cursor:
                        cat = row["doc_category"]
                        results[cat] = results.get(cat, 0) + row["cnt"]
                return results
        except Exception as e:
            logger.error(f"[MetadataIndex] Failed to count categories: {e}")
            return {}
        finally:
            conn.close()

    def get_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific file."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM file_metadata WHERE file_path = ?", (file_path,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"[MetadataIndex] Failed to get metadata for {os.path.basename(file_path)}: {e}")
            return None
        finally:
            conn.close()

    def delete_metadata(self, file_path: str):
        """Remove file from metadata index."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM file_metadata WHERE file_path = ?", (file_path,))
                conn.commit()
            except Exception as e:
                logger.error(f"[MetadataIndex] Failed to delete metadata for {os.path.basename(file_path)}: {e}")
            finally:
                conn.close()
