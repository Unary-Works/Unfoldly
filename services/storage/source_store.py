import os
import threading
from typing import List

class SourceStore:
    def __init__(self, base_dir: str):
        self.new_path = os.path.join(base_dir, "indexed_sources.json")
        self.legacy_path = os.path.join(base_dir, "indexed_folders.json")
        self.folders: List[str] = []
        self.files: List[str] = []
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        try:
            import json
            if os.path.exists(self.new_path):
                with open(self.new_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self.folders = [str(x) for x in raw.get("folders", []) if x]
                    self.files = [str(x) for x in raw.get("files", []) if x]
                    return
            if os.path.exists(self.legacy_path):
                with open(self.legacy_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    self.folders = [str(x) for x in raw if x]
                    self.files = []
                    self.save()
                    return
            self.folders = []
            self.files = []
        except Exception:
            self.folders = []
            self.files = []

    def save(self) -> None:
        try:
            import json
            data = {"folders": self.folders, "files": self.files}
            with open(self.new_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_folder(self, folder: str) -> None:
        folder = os.path.abspath(os.path.expanduser(folder))
        with self._lock:
            if folder not in self.folders:
                self.folders.append(folder)
                self.save()

    def add_file(self, file_path: str) -> None:
        file_path = os.path.abspath(os.path.expanduser(file_path))
        with self._lock:
            if file_path not in self.files:
                self.files.append(file_path)
                self.save()

    def add_files(self, file_paths: List[str]) -> None:
        with self._lock:
            for fp in file_paths:
                fp_abs = os.path.abspath(os.path.expanduser(fp))
                if fp_abs not in self.files:
                    self.files.append(fp_abs)
            self.save()

    def remove(self, path: str) -> None:
        path = os.path.abspath(os.path.expanduser(path))
        with self._lock:
            self.folders = [f for f in self.folders if f != path]
            self.files = [f for f in self.files if f != path]
            self.save()

    def get_all_paths(self) -> List[str]:
        return self.folders + self.files

    def add(self, folder: str) -> None:
        self.add_folder(folder)
