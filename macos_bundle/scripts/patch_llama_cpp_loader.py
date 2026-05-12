#!/usr/bin/env python3
"""
Patch llama_cpp's ctypes loader to prefer bundled dylibs before find_library().

This prevents a bundled macOS app from accidentally resolving ggml-family
libraries from a sibling package (for example pywhispercpp) and then failing
with `dlsym(..., ggml_log_get): symbol not found`.
"""

from __future__ import annotations

import os
import sys


NEW_BLOCK = """    errors = []

    # Prefer bundled libraries next to llama_cpp itself.
    # This avoids resolving conflicting ggml/whisper dylibs from other packages
    # (for example pywhispercpp) during macOS app startup/runtime.
    for base_path in base_paths:
        for lib_name in lib_names:
            lib_path = pathlib.Path(base_path) / lib_name

            if lib_path.exists():
                try:
                    return ctypes.CDLL(str(lib_path), **cdll_args)
                except Exception as e:
                    errors.append(f"{lib_path}: {e}")

    # Fallback to libraries discoverable from the wider runtime.
    lib_path = find_library(lib_base_name)
    if lib_path:
        try:
            return ctypes.CDLL(lib_path, **cdll_args)
        except Exception as e:
            errors.append(f"{lib_path}: {e}")
"""


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_llama_cpp_loader.py <site_dir>", file=sys.stderr)
        return 2

    site_dir = os.path.abspath(sys.argv[1])
    target = os.path.join(site_dir, "llama_cpp", "_ctypes_extensions.py")
    if not os.path.exists(target):
        print(f"[patch_llama_cpp_loader] skip: missing {target}")
        return 0

    with open(target, "r", encoding="utf-8") as f:
        text = f.read()

    if "Prefer bundled libraries next to llama_cpp itself." in text:
        print(f"[patch_llama_cpp_loader] already patched: {target}")
        return 0

    start = text.find("    errors = []\n")
    if start < 0:
        print(f"[patch_llama_cpp_loader] skip: cannot locate loader block start in {target}", file=sys.stderr)
        return 1

    end = text.find("\n    raise RuntimeError(", start)
    if end < 0:
        print(f"[patch_llama_cpp_loader] skip: cannot locate loader block end in {target}", file=sys.stderr)
        return 1

    patched = text[:start] + NEW_BLOCK + text[end:]
    with open(target, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"[patch_llama_cpp_loader] patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
