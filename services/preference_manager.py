import os
import json
from typing import Optional, Dict, Any

from utils.logger import get_child_logger

logger = get_child_logger(__name__)

class PreferenceManager:
    def __init__(self, base_dir: str = ""):
        if not base_dir:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pref_path = (os.getenv("FILEAGENT_PREFERENCES_PATH") or os.getenv("FILEAGENT_PREFS_PATH") or "").strip()
        if not pref_path:
            data_dir = (os.getenv("FILEAGENT_DATA_DIR") or "").strip()
            data_dir = os.path.abspath(os.path.expanduser(data_dir)) if data_dir else ""
            if data_dir:
                pref_path = os.path.join(data_dir, "user_preferences.json")
        if pref_path:
            self.config_path = os.path.abspath(os.path.expanduser(pref_path))
        else:
            candidate = os.path.join(base_dir, "data", "user_preferences.json")
            legacy = os.path.join(base_dir, "user_preferences.json")
            self.config_path = legacy if (os.path.exists(legacy) and not os.path.exists(candidate)) else candidate
        self.preferences: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.config_path):
            self.preferences = {}
            return
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.preferences = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load user preferences: {e}")
            self.preferences = {}

    def save(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.preferences, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self.preferences.get(key, default)

    def set(self, key: str, value: Any):
        self.preferences[key] = value
        self.save()

    def get_selected_model_id(self) -> Optional[str]:
        return self.preferences.get("selected_model_id")

    def set_selected_model_id(self, model_id: str):
        self.preferences["selected_model_id"] = model_id
        self.save()

    def get_selected_index_model_id(self) -> Optional[str]:
        return self.preferences.get("selected_index_model_id")

    def set_selected_index_model_id(self, model_id: str):
        self.preferences["selected_index_model_id"] = model_id
        self.save()

    # ---- Per-model quantization preference ----
    # shape: {"model_quantization": {"model_id": "some.gguf"}}

    def get_model_quantization_map(self) -> Dict[str, str]:
        m = self.preferences.get("model_quantization", {})
        if isinstance(m, dict):
            # ensure string values
            return {str(k): str(v) for k, v in m.items() if v is not None}
        return {}

    def get_selected_quantization_file(self, model_id: str) -> Optional[str]:
        mid = (model_id or "").strip()
        if not mid:
            return None
        m = self.get_model_quantization_map()
        v = m.get(mid)
        return (v or "").strip() or None

    def set_selected_quantization_file(self, model_id: str, quantization_file: str) -> None:
        mid = (model_id or "").strip()
        qf = (quantization_file or "").strip()
        if not mid or not qf:
            return
        m = self.preferences.get("model_quantization", {})
        if not isinstance(m, dict):
            m = {}
        m[mid] = qf
        self.preferences["model_quantization"] = m
        self.save()
