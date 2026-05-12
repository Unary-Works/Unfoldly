import os
import uuid
from typing import Any, Dict, List, Optional, Set

from config import settings


def _stable_id(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _icon_type_for_path(path: str) -> str:
    ext = os.path.splitext(path.lower())[1].lstrip(".")
    if ext in {"pdf"}:
        return "pdf"
    if ext in {"doc", "docx", "md", "txt", "rtf"}:
        return "doc"
    if ext in {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"}:
        return "image"
    if ext in {"xls", "xlsx", "csv"}:
        return "sheet"
    if ext in {"mp4", "mov", "avi", "mkv", "webm", "flv", "wmv"}:
        return "video"
    if ext in {"mp3", "wav", "m4a", "flac", "aac", "ogg"}:
        return "audio"
    return "doc"


def _is_indexable_file_for_sources(file_path: str, *, treat_parent_as_explicit_source: bool = True) -> bool:
    try:
        file_name = os.path.basename(file_path)
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()
        abs_path = os.path.abspath(os.path.expanduser(file_path))

        lower_name = file_name.lower()
        if lower_name.startswith(".~") or lower_name.startswith("~$") or lower_name.startswith("._"):
            return False
        if lower_name.endswith((".swp", ".swo", ".tmp", ".bak")):
            return False

        if ext not in getattr(settings, "ALLOWED_EXTENSIONS", set()):
            return False
        if file_name in getattr(settings, "EXCLUDE_FILENAMES", set()):
            return False
        for prefix in getattr(settings, "EXCLUDE_FILENAME_PREFIXES", set()):
            try:
                if file_name.lower().startswith(str(prefix).lower()):
                    return False
            except Exception:
                continue

        if getattr(settings, "USE_WHITELIST_MODE", False):
            if not treat_parent_as_explicit_source:
                in_whitelist = False
                for include_path in getattr(settings, "INCLUDE_PATHS", set()):
                    try:
                        ip = os.path.abspath(os.path.expanduser(str(include_path)))
                    except Exception:
                        ip = str(include_path)
                    if ip and abs_path.startswith(ip):
                        in_whitelist = True
                        break
                if not in_whitelist:
                    return False
        else:
            for pattern in getattr(settings, "IGNORE_PATTERNS", set()):
                try:
                    if str(pattern) in abs_path:
                        return False
                except Exception:
                    continue
            for exclude_path in getattr(settings, "EXCLUDE_PATHS", set()):
                try:
                    if str(exclude_path) in abs_path:
                        return False
                except Exception:
                    continue
        return True
    except Exception:
        return False


def _build_file_node(file_path: str, status: str = "indexed") -> Dict[str, Any]:
    name = os.path.basename(file_path)
    node_id = _stable_id(f"file:{file_path}")
    return {
        "id": node_id,
        "name": name,
        "type": "file",
        "iconType": _icon_type_for_path(file_path),
        "path": file_path,
        "status": status,
    }


def _source_path_key(path: str) -> str:
    try:
        expanded = os.path.expanduser(path)
        real = os.path.realpath(expanded)
        return os.path.normcase(os.path.normpath(real))
    except Exception:
        return os.path.normcase(os.path.normpath(os.path.abspath(os.path.expanduser(path))))


def _is_folder_fully_indexed(folder_path: str, indexed_path_keys: Optional[Set[str]]) -> bool:
    if not indexed_path_keys:
        return False
    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.startswith("."):
                continue
            p = os.path.join(root, f)
            if _is_indexable_file_for_sources(p, treat_parent_as_explicit_source=True):
                if _source_path_key(p) not in indexed_path_keys:
                    return False
    return True


def _collect_indexable_file_paths(root: str) -> Set[str]:
    out: Set[str] = set()
    try:
        root_abs = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root_abs):
            return out
        for walk_root, dirs, files in os.walk(root_abs):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fn in files:
                if fn.startswith("."):
                    continue
                p = os.path.join(walk_root, fn)
                try:
                    if _is_indexable_file_for_sources(p, treat_parent_as_explicit_source=True):
                        out.add(_source_path_key(p))
                except Exception:
                    continue
    except Exception:
        pass
    return out


def _folder_has_relevant_indexable_file(folder_path: str, relevant_paths: Set[str]) -> bool:
    if not relevant_paths:
        return False
    folder_key = _source_path_key(folder_path).rstrip(os.sep)
    prefix = folder_key + os.sep
    for pk in relevant_paths:
        if pk.startswith(prefix) or pk == folder_key:
            return True
    return False


def _build_folder_node(
    folder_path: str,
    status: str = "indexed",
    max_children: int = 400,
    indexed_path_keys: Optional[Set[str]] = None,
    depth: int = 0,
    *,
    relevant_indexable_paths: Optional[Set[str]] = None,
    prune_empty_subfolders: bool = False,
) -> Dict[str, Any]:
    name = os.path.basename(folder_path.rstrip(os.sep)) or folder_path
    node_id = _stable_id(f"folder:{folder_path}")
    children: List[Dict[str, Any]] = []

    current_folder_status = status
    if status == "indexing" and indexed_path_keys is not None:
        if _is_folder_fully_indexed(folder_path, indexed_path_keys):
            current_folder_status = "indexed"

    if depth >= 15:
        return {
            "id": node_id,
            "name": name,
            "type": "folder",
            "iconType": "folder",
            "path": folder_path,
            "status": current_folder_status,
            "children": [],
        }

    try:
        entries = os.listdir(folder_path)
        dirs = []
        files = []
        for e in entries:
            if e.startswith("."):
                continue
            p = os.path.join(folder_path, e)
            if os.path.isdir(p):
                dirs.append(e)
            else:
                files.append(e)

        dirs.sort()
        files.sort()

        for d in dirs[:100]:
            d_path = os.path.join(folder_path, d)
            if prune_empty_subfolders and relevant_indexable_paths is not None:
                if not _folder_has_relevant_indexable_file(d_path, relevant_indexable_paths):
                    continue
            child_node = _build_folder_node(
                d_path,
                current_folder_status,
                max_children,
                indexed_path_keys,
                depth + 1,
                relevant_indexable_paths=relevant_indexable_paths,
                prune_empty_subfolders=prune_empty_subfolders,
            )
            children.append(child_node)

        remaining = max(0, int(max_children) - len(children))
        for f in files:
            if remaining <= 0:
                break
            f_path = os.path.join(folder_path, f)
            if not _is_indexable_file_for_sources(f_path, treat_parent_as_explicit_source=True):
                continue
            
            file_status = current_folder_status
            fk = _source_path_key(f_path)
            if indexed_path_keys is None:
                file_status = "indexed"
            elif current_folder_status == "indexing":
                if fk in indexed_path_keys:
                    file_status = "indexed"
                else:
                    file_status = "pending"
            elif current_folder_status == "indexed":
                if fk not in indexed_path_keys:
                    continue

            children.append({
                "id": _stable_id(f"file:{f_path}"),
                "name": f,
                "type": "file",
                "iconType": _icon_type_for_path(f_path),
                "path": f_path,
                "status": file_status,
            })
            remaining -= 1

    except Exception:
        pass
    
    return {
        "id": node_id,
        "name": name,
        "type": "folder",
        "iconType": "folder",
        "path": folder_path,
        "status": current_folder_status,
        "children": children,
    }
