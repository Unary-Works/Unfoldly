import os
import threading
from typing import Dict, Any

class HistoryManager:
    def __init__(self, base_dir: str):
        self.path = os.path.join(base_dir, "chat_history.json")
        self.lock = threading.Lock()
        self.ensure_file()

    def ensure_file(self):
        if not os.path.exists(self.path):
            with self.lock:
                import json
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump({}, f)

    def load_all(self) -> Dict[str, Any]:
        with self.lock:
            try:
                import json
                if not os.path.exists(self.path):
                    return {}
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    result = {}
                    for item in data:
                        if isinstance(item, dict) and "id" in item:
                            result[item["id"]] = item
                    return result
                elif isinstance(data, dict):
                    return data
                else:
                    return {}
            except Exception:
                return {}

    def save_session(self, session_id: str, session_data: Dict[str, Any]):
        with self.lock:
            import json
            try:
                if os.path.exists(self.path):
                    with open(self.path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    data = {}
            except Exception:
                data = {}
            if isinstance(data, list):
                converted = {}
                for item in data:
                    if isinstance(item, dict) and "id" in item:
                        converted[item["id"]] = item
                data = converted
            elif not isinstance(data, dict):
                data = {}
            data[session_id] = session_data
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def delete_session(self, session_id: str):
        with self.lock:
            import json
            try:
                if os.path.exists(self.path):
                    with open(self.path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    return
            except Exception:
                return
            if not isinstance(data, dict):
                return
            if session_id in data:
                del data[session_id]
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
