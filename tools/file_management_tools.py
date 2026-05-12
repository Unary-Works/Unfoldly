
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Optional

from langchain_core.tools import tool

from config import settings
from utils.pdf_utils import HAS_PDF_TEXT, extract_pdf_text
from .registry import ToolRegistry
from .document_tools import get_active_paths


def _default_root_dir() -> str:
    try:
        wd = getattr(settings, "WATCH_DIR", "") or ""
    except Exception:
        wd = ""
    return os.path.expanduser(wd) if wd else os.path.expanduser("~")


FILE_TOOLS_ROOT = os.path.abspath(
    os.path.expanduser(os.getenv("FILE_TOOLS_ROOT", _default_root_dir()))
)


def _is_within_any_root(ap: str, roots: list[str]) -> bool:
    for r in roots:
        rr = os.path.abspath(os.path.expanduser(r))
        if ap == rr or ap.startswith(rr + os.sep):
            return True
    return False


def _safe_abs_in_scope(path: str) -> str:
    ap = os.path.abspath(os.path.expanduser(path))
    active_paths = get_active_paths()

    if active_paths is None:
        return _safe_abs(path)

    if isinstance(active_paths, list) and len(active_paths) == 0:
        raise ValueError("当前未选中任何 Sources（active_paths 为空）。请先在右侧 Sources 面板勾选目录后再进行文件操作。")

    if _is_within_any_root(ap, active_paths):
        return ap

    is_explicit = (path or "").startswith("/") or (path or "").startswith("~")
    if is_explicit:
        return _safe_abs(path)

    raise ValueError(f"路径超出当前选中 Sources 范围：{ap}\n当前允许范围：{active_paths}")


def _safe_abs(path: str) -> str:
    ap = os.path.abspath(os.path.expanduser(path))
    root = FILE_TOOLS_ROOT
    if ap == root or ap.startswith(root + os.sep):
        return ap
    raise ValueError(f"路径超出允许范围：{ap}\n允许范围：{root}\n如需放开，请设置环境变量 FILE_TOOLS_ROOT")


def _try_build_langchain_tools(root_dir: str) -> Optional[list[Any]]:
    try:
        from langchain.tools.file_management import (  # type: ignore
            ReadFileTool,
            WriteFileTool,
            CopyFileTool,
            DeleteFileTool,
            MoveFileTool,
            ListDirectoryTool,
        )
    except Exception:
        try:
            from langchain_community.tools.file_management import (  # type: ignore
                ReadFileTool,
                WriteFileTool,
                CopyFileTool,
                DeleteFileTool,
                MoveFileTool,
                ListDirectoryTool,
            )
        except Exception:
            return None

    tools: list[Any] = []

    def _inst(cls, name: str):
        try:
            t = cls(root_dir=root_dir)
        except Exception:
            t = cls()
            try:
                setattr(t, "root_dir", root_dir)
            except Exception:
                pass
        try:
            setattr(t, "name", name)
        except Exception:
            pass
        return t

    tools.append(_inst(ReadFileTool, "read_file"))
    tools.append(_inst(WriteFileTool, "write_file"))
    tools.append(_inst(CopyFileTool, "copy_file"))
    tools.append(_inst(DeleteFileTool, "delete_file"))
    tools.append(_inst(MoveFileTool, "move_file"))
    tools.append(_inst(ListDirectoryTool, "list_directory"))
    return tools




@tool
def read_file(file_path: str) -> str:
    """Read a text file, limited to selected Sources and FILE_TOOLS_ROOT scope."""
    ap = _safe_abs_in_scope(file_path)
    with open(ap, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


@tool
def open_file(file_path: str) -> str:
    """
    Open and read a file for "open file" or "view original content" workflows.

    Prefer this semantic tool when the user asks to open, inspect, or preview a
    specific file.

    - Uses the same access rules as `read_file`: selected Sources by default,
      and explicit absolute paths remain constrained by FILE_TOOLS_ROOT.
    - Returns content for the frontend Opened view.
    - For supported images, returns a bounded inline preview instead of raw bytes.
    - For unsupported binary formats, avoids loading raw bytes into model context.
    """
    ap = _safe_abs_in_scope(file_path)
    ext = os.path.splitext(ap)[1].lower()

    max_len = int(os.getenv("OPEN_FILE_MAX_CHARS", "60000"))

    if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}:
        try:
            import base64
            import json

            max_image_bytes = int(os.getenv("OPEN_IMAGE_MAX_BYTES", "3000000"))  # default 3MB
            try:
                size = int(os.path.getsize(ap))
            except Exception:
                size = -1

            if size > max_image_bytes and size > 0:
                return json.dumps(
                    {
                        "ok": False,
                        "kind": "image_too_large",
                        "file_path": ap,
                        "ext": ext,
                        "size_bytes": size,
                        "max_bytes": max_image_bytes,
                        "message": "Image is too large to inline preview. Please open it in system viewer.",
                    },
                    ensure_ascii=False,
                )

            with open(ap, "rb") as f:
                data = f.read(max_image_bytes + 1)
            truncated = len(data) > max_image_bytes
            if truncated:
                data = data[:max_image_bytes]

            mime = "image/png"
            if ext in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif ext == ".webp":
                mime = "image/webp"
            elif ext == ".gif":
                mime = "image/gif"
            elif ext == ".bmp":
                mime = "image/bmp"
            elif ext in {".tif", ".tiff"}:
                mime = "image/tiff"

            b64 = base64.b64encode(data).decode("utf-8")
            data_url = f"data:{mime};base64,{b64}"
            return json.dumps(
                {
                    "ok": True,
                    "kind": "image",
                    "file_path": ap,
                    "mime": mime,
                    "size_bytes": (size if size >= 0 else None),
                    "preview_bytes": len(data),
                    "truncated": bool(truncated),
                    "data_url": data_url,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return f"[图片预览失败] {e}\n路径：{ap}"

    if ext in {".txt", ".md", ".json", ".csv", ".log", ".py", ".ts", ".tsx", ".js", ".jsx", ".yml", ".yaml"}:
        with open(ap, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:max_len]

    if ext == ".docx":
        try:
            import docx2txt  # type: ignore

            text = docx2txt.process(ap) or ""
            return text[:max_len]
        except Exception as e:
            return f"[DOCX 解析失败] {e}\n路径：{ap}"

    if ext in {".doc", ".rtf"}:
        try:
            textutil = "/usr/bin/textutil"
            if not os.path.exists(textutil):
                raise RuntimeError("missing /usr/bin/textutil (macOS only)")
            p = subprocess.run(
                [textutil, "-convert", "txt", "-stdout", ap],
                capture_output=True,
                text=True,
                check=False,
            )
            if p.returncode != 0:
                err = (p.stderr or "").strip()
                raise RuntimeError(err or f"textutil exit {p.returncode}")
            return (p.stdout or "")[:max_len]
        except Exception as e:
            return f"[DOC/RTF 解析失败] {e}\n路径：{ap}\n建议：可尝试用系统打开后另存为 .docx/.pdf 再打开。"

    if ext == ".pdf":
        try:
            if not HAS_PDF_TEXT:
                raise RuntimeError("PDF parser is not installed")
            text, _ = extract_pdf_text(ap, max_chars=max_len)
            return text[:max_len]
        except Exception as e:
            return f"[PDF 暂不支持读取或解析失败] {e}\n路径：{ap}"

    size = None
    try:
        size = os.path.getsize(ap)
    except Exception:
        pass
    return f"[不支持直接在文本中打开该格式] 后缀：{ext or '(none)'}\n路径：{ap}\n大小：{size if size is not None else 'unknown'} bytes\n建议：如需查看，请在系统中打开或转换为 txt/pdf 后再打开。"


@tool
def write_file(file_path: str, text: str, mode: str = "w") -> str:
    """Write or append text to a file within selected Sources and FILE_TOOLS_ROOT scope."""
    ap = _safe_abs_in_scope(file_path)
    os.makedirs(os.path.dirname(ap), exist_ok=True)
    if mode not in ("w", "a"):
        raise ValueError("mode 只支持 'w' 或 'a'")
    with open(ap, mode, encoding="utf-8") as f:
        f.write(text)
    return f"已写入：{ap}"


@tool
def copy_file(source_path: str, destination_path: str) -> str:
    """Copy a file within selected Sources and FILE_TOOLS_ROOT scope."""
    src = _safe_abs_in_scope(source_path)
    dst = _safe_abs_in_scope(destination_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return f"已复制：{src} -> {dst}"


@tool
def move_file(source_path: str, destination_path: str) -> str:
    """Move or rename a file within selected Sources and FILE_TOOLS_ROOT scope."""
    src = _safe_abs_in_scope(source_path)
    dst = _safe_abs_in_scope(destination_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    return f"已移动：{src} -> {dst}"


@tool
def delete_file(file_path: str) -> str:
    """Delete a file within selected Sources and FILE_TOOLS_ROOT scope."""
    ap = _safe_abs_in_scope(file_path)
    if not os.path.exists(ap):
        return f"文件不存在：{ap}"
    os.remove(ap)
    return f"已删除：{ap}"


@tool
def list_directory(directory_path: str = ".", max_items: int = 200) -> str:
    """
    List directory contents.

    - If directory_path is "." and the current request has active_paths, return
      the selected source list with one absolute path per line.
    - Otherwise list entries within selected Sources and FILE_TOOLS_ROOT scope.
    """
    active_paths = get_active_paths()
    if directory_path in (".", "", None) and active_paths is not None:
        if len(active_paths) == 0:
            return "[NO_ACTIVE_SOURCES]"
        return "\n".join(active_paths[: max(1, int(max_items))])

    ap = _safe_abs_in_scope(directory_path)
    items = sorted(os.listdir(ap))[: max(1, int(max_items))]
    return "\n".join(items)


@tool
def file_search(query: str, directory_path: str = ".", max_matches: int = 50) -> str:
    """
    Disabled physical disk search.

    Retrieval must use the indexed database tools in tools.document_tools
    (`search_files`, `search_documents`, `count_documents_files`) so query-time
    search never scans local folders or opens arbitrary files.
    """
    return (
        "[DISABLED_DISK_SEARCH] Query-time physical disk search is disabled. "
        "Use indexed database tools: search_files/search_documents/count_documents_files."
    )


def _register_tools():
    ToolRegistry.register_langchain_tool("open_file", open_file)
    # ToolRegistry.register_langchain_tool("read_file", read_file)
    # ToolRegistry.register_langchain_tool("write_file", write_file)
    # ToolRegistry.register_langchain_tool("copy_file", copy_file)
    # ToolRegistry.register_langchain_tool("move_file", move_file)
    # ToolRegistry.register_langchain_tool("delete_file", delete_file)
    # ToolRegistry.register_langchain_tool("list_directory", list_directory)


_register_tools()
