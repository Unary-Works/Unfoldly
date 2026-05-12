import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from utils.logger import get_logger

logger = get_logger()

class PersonalInfoDB:
    """
    Sidecar SQLite Database for Personal and High-Value Information extracted during indexing.
    """
    
    def __init__(self, db_dir: str):
        self.db_path = os.path.join(db_dir, "personal_info.sqlite")
        self._lock = threading.RLock()
        self._init_db()
        
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = self._get_connection()
            try:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS personal_info_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_name TEXT,
                    info_type TEXT NOT NULL,
                    description TEXT,
                    content TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    source_file_name TEXT NOT NULL,
                    extracted_at TEXT NOT NULL
                )
                """)
                # Create indices for hot fields and dedup
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_dedup ON personal_info_records(info_type, content, source_file)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON personal_info_records(info_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON personal_info_records(source_file)")
                conn.commit()
            except Exception as e:
                logger.error(f"[PersonalInfoDB] Failed to initialize db: {e}")
            finally:
                conn.close()

    def upsert_batch(self, records: List[Dict[str, Any]]):
        """
        Batch insert or ignore personal info records.
        Records should have: owner_name, info_type, description, content, source_file, source_file_name
        """
        if not records:
            return
            
        now = datetime.now(timezone.utc).isoformat()
        
        with self._lock:
            conn = self._get_connection()
            try:
                # Use INSERT OR IGNORE to avoid duplicating the exact same content from the same file
                conn.executemany("""
                INSERT INTO personal_info_records (
                    owner_name, info_type, description, content, source_file, source_file_name, extracted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(info_type, content, source_file) DO UPDATE SET
                    owner_name=excluded.owner_name,
                    description=excluded.description,
                    extracted_at=excluded.extracted_at,
                    source_file_name=excluded.source_file_name
                """, [
                    (
                        r.get("owner", "Unknown"),
                        r.get("type", "other"),
                        r.get("description", ""),
                        r.get("content", ""),
                        r.get("source_file", ""),
                        r.get("source_file_name", ""),
                        now
                    )
                    for r in records
                ])
                conn.commit()
            except Exception as e:
                logger.error(f"[PersonalInfoDB] Failed to upsert batch: {e}")
            finally:
                conn.close()

    def search(self, query: str = "", types: Optional[List[str]] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search for personal info items based on a query or specific types.
        """
        conn = self._get_connection()
        results = []
        try:
            sql = "SELECT * FROM personal_info_records WHERE 1=1"
            params = []
            
            if query:
                # Simple exact/LIKE match on content, description, owner
                sql += " AND (content LIKE ? OR description LIKE ? OR owner_name LIKE ? OR source_file_name LIKE ?)"
                like_q = f"%{query}%"
                params.extend([like_q, like_q, like_q, like_q])
                
            if types and len(types) > 0:
                expanded_types = []
                contact_subtypes = {"phone", "email", "address", "social_media", "contact"}
                for t in types:
                    if t == "联系方式":
                        expanded_types.extend(contact_subtypes)
                    else:
                        expanded_types.append(t)
                        
                placeholders = ",".join("?" * len(expanded_types))
                sql += f" AND info_type IN ({placeholders})"
                params.extend(expanded_types)
                
            sql += " ORDER BY extracted_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(sql, tuple(params))
            for row in cursor:
                results.append(dict(row))
            return results
        except Exception as e:
            logger.error(f"[PersonalInfoDB] Failed to search: {e}")
            return []
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregated stats by info_type.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT info_type, COUNT(*) as cnt FROM personal_info_records GROUP BY info_type ORDER BY cnt DESC")
            stats = []
            total = 0
            contact_count = 0
            contact_subtypes = {"phone", "email", "address", "social_media", "contact"}
            
            for row in cursor:
                t = row["info_type"]
                c = row["cnt"]
                total += c
                if t in contact_subtypes:
                    contact_count += c
                else:
                    stats.append({"type": t, "count": c})
            
            if contact_count > 0:
                stats.append({"type": "联系方式", "count": contact_count})
                
            # Re-sort since we aggregated contacts
            stats.sort(key=lambda x: x["count"], reverse=True)
            
            return {"stats": stats, "total": total}
        except Exception as e:
            logger.error(f"[PersonalInfoDB] Failed to get stats: {e}")
            return {"stats": [], "total": 0}
        finally:
            conn.close()

    def delete_by_file(self, file_path: str):
        """Remove info extracted from a specific file."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM personal_info_records WHERE source_file = ?", (file_path,))
                conn.commit()
            except Exception as e:
                logger.error(f"[PersonalInfoDB] Failed to delete records for file {os.path.basename(file_path)}: {e}")
            finally:
                conn.close()
                
    def delete_by_folder(self, folder_path: str):
        """Remove info extracted from files within a folder."""
        with self._lock:
            conn = self._get_connection()
            try:
                # Append % to match any file paths starting with the folder path
                # ensure folder path ends with slash to not mismatch siblings
                folder_prefix = folder_path if folder_path.endswith('/') or folder_path.endswith('\\') else folder_path + '/'
                # Also delete exact match in case folder_path is actually a specific file
                conn.execute("DELETE FROM personal_info_records WHERE source_file = ? OR source_file LIKE ?", (folder_path, f"{folder_prefix}%"))
                conn.commit()
            except Exception as e:
                logger.error(f"[PersonalInfoDB] Failed to delete records for folder {os.path.basename(folder_path.rstrip(os.sep))}: {e}")
            finally:
                conn.close()

    def clear_all(self):
        """Clear all records."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM personal_info_records")
                conn.commit()
            except Exception as e:
                logger.error(f"[PersonalInfoDB] Failed to clear all: {e}")
            finally:
                conn.close()
