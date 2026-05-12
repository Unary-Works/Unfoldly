import os
from watchdog.events import FileSystemEventHandler
from config import settings

class FileEventHandler(FileSystemEventHandler):
    def __init__(self, rag_engine):
        self.rag = rag_engine
        self.home_dir = os.path.abspath(settings.HOME_DIR)

    def _should_ignore(self, file_path):
        abs_path = os.path.abspath(file_path)
        
        if not abs_path.startswith(self.home_dir): 
            return True

        rel_path = os.path.relpath(abs_path, self.home_dir)
        parts = rel_path.split(os.sep)
        
        if parts[0] in settings.IGNORE_TOP_LEVEL_DIRS: 
            return True
            
        for part in parts:
            if part.startswith(".") and len(part) > 1:
                return True
            if part in settings.IGNORE_PATTERNS: 
                return True
        
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in settings.ALLOWED_EXTENSIONS: 
            return True

        exclude_paths = getattr(settings, 'EXCLUDE_PATHS', set())
        for exclude in exclude_paths:
            if exclude in abs_path:
                return True

        return False

    def on_created(self, event):
        if not event.is_directory and not self._should_ignore(event.src_path):
            print(f"[FileMonitor] created: {os.path.basename(event.src_path)}")
            self.rag.ingest_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and not self._should_ignore(event.src_path):
            print(f"[FileMonitor] modified: {os.path.basename(event.src_path)}")
            self.rag.ingest_file(event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory and not self._should_ignore(event.src_path):
            print(f"[FileMonitor] deleted: {os.path.basename(event.src_path)}")
            self.rag.delete_file(event.src_path)
    
    def on_moved(self, event):
        if not event.is_directory:
            if not self._should_ignore(event.src_path):
                print(f"[FileMonitor] moved: {os.path.basename(event.src_path)} -> {os.path.basename(event.dest_path)}")
                self.rag.delete_file(event.src_path)
            
            if not self._should_ignore(event.dest_path):
                self.rag.ingest_file(event.dest_path)
