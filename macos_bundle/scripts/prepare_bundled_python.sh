#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MACOS_BUNDLE="$ROOT_DIR/macos_bundle"
PYTHON_RUNTIME="$MACOS_BUNDLE/python_runtime"
CACHE_DIR="${UNFOLDLY_PYTHON_CACHE:-$MACOS_BUNDLE/cache}"
WHEELHOUSE="$CACHE_DIR/wheelhouse"
PBS_TAG="20260310"
PYTHON_VERSION="3.12.13"
LLAMA_CPP_PYTHON_REPO="${UNFOLDLY_LLAMA_CPP_PYTHON_REPO:-https://github.com/JamePeng/llama-cpp-python.git}"
LLAMA_CPP_PYTHON_REF="${UNFOLDLY_LLAMA_CPP_PYTHON_REF:-ef27f333f367fdc53dc1a729ad8bb6c3c9362514}"
PYWHISPERCPP_REPO="${UNFOLDLY_PYWHISPERCPP_REPO:-https://github.com/absadiki/pywhispercpp.git}"
PYWHISPERCPP_REF="${UNFOLDLY_PYWHISPERCPP_REF:-aaf756bd3c2e8ad38f62bbdc9a32a7549fde9c78}"

if [[ -n "${1:-}" ]]; then
  if [[ "$1" == "arm64" ]]; then
    TARGET_TRIPLE="aarch64-apple-darwin"
  elif [[ "$1" == "x64" ]]; then
    TARGET_TRIPLE="x86_64-apple-darwin"
  else
    echo "[prepare_bundled_python] ERROR: unknown architecture $1 (valid: arm64, x64)"
    exit 2
  fi
else
  case "$(uname -m)" in
    arm64)  TARGET_TRIPLE="aarch64-apple-darwin" ;;
    x86_64) TARGET_TRIPLE="x86_64-apple-darwin" ;;
    *)      echo "[prepare_bundled_python] ERROR: only macOS arm64/x86_64 is supported"; exit 1 ;;
  esac
fi

mkdir -p "$CACHE_DIR" "$PYTHON_RUNTIME" "$WHEELHOUSE"

TARBALL="cpython-${PYTHON_VERSION}+${PBS_TAG}-${TARGET_TRIPLE}-install_only.tar.gz"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${TARBALL}"
if [[ ! -f "$CACHE_DIR/$TARBALL" ]]; then
  echo "[prepare_bundled_python] Downloading $TARBALL ..."
  curl -sSL -o "$CACHE_DIR/$TARBALL" "$URL" || wget -q -O "$CACHE_DIR/$TARBALL" "$URL"
fi
if [[ ! -f "$CACHE_DIR/$TARBALL" ]]; then
  echo "[prepare_bundled_python] ERROR: download failed: $URL"
  exit 1
fi

INSTALL_DIR="$PYTHON_RUNTIME/install"

if [[ ! -x "$INSTALL_DIR/bin/python3" ]]; then
  echo "[prepare_bundled_python] Extracting Python to $INSTALL_DIR ..."
  rm -rf "$INSTALL_DIR" "$PYTHON_RUNTIME/extract"
  mkdir -p "$PYTHON_RUNTIME/extract"
  tar xzf "$CACHE_DIR/$TARBALL" -C "$PYTHON_RUNTIME/extract"
  if [[ -d "$PYTHON_RUNTIME/extract/python/install" ]]; then
    mv "$PYTHON_RUNTIME/extract/python/install" "$INSTALL_DIR"
  else
    TOP=$(ls -1 "$PYTHON_RUNTIME/extract" | head -1)
    mv "$PYTHON_RUNTIME/extract/$TOP" "$INSTALL_DIR"
  fi
  rm -rf "$PYTHON_RUNTIME/extract"
fi
if [[ ! -x "$INSTALL_DIR/bin/python3" ]]; then
  echo "[prepare_bundled_python] ERROR: $INSTALL_DIR/bin/python3 not found after extraction"
  exit 1
fi
echo "[prepare_bundled_python] Interpreter: $INSTALL_DIR/bin/python3 ($("$INSTALL_DIR/bin/python3" --version 2>&1))"

LIBDIR="$INSTALL_DIR/lib"
if ! find "$LIBDIR" -maxdepth 1 -name 'libpython*.dylib' -print -quit 2>/dev/null | grep -q .; then
  echo "[prepare_bundled_python] ERROR: libpython*.dylib not found in $LIBDIR"
  exit 1
fi

export PYTHON_RUNTIME_INSTALL="$INSTALL_DIR"

SITE_DIR="$PYTHON_RUNTIME/site"
FORCE_REBUILD_SITE="${UNFOLDLY_FORCE_REBUILD_SITE:-1}"
FORCE_REBUILD_SITE_LC="$(printf '%s' "$FORCE_REBUILD_SITE" | tr '[:upper:]' '[:lower:]')"
SITE_REBUILD_ENABLED=false
PATCH_LLAMA_CPP_LOADER="$SCRIPT_DIR/patch_llama_cpp_loader.py"
case "$FORCE_REBUILD_SITE_LC" in
  1|true|yes|on) SITE_REBUILD_ENABLED=true ;;
esac

write_grpc_local_stub() {
  local site_dir="$1"
  rm -rf "$site_dir/grpc" "$site_dir"/grpcio-*.dist-info
  mkdir -p "$site_dir/grpc"
  cat > "$site_dir/grpc/__init__.py" <<'PY'
__version__ = "1.76.0"

class _Status:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"StatusCode.{self.name}"


class _StatusCode:
    def __getattr__(self, name):
        value = _Status(name)
        setattr(self, name, value)
        return value


StatusCode = _StatusCode()
for _name in (
    "OK CANCELLED UNKNOWN INVALID_ARGUMENT DEADLINE_EXCEEDED NOT_FOUND "
    "ALREADY_EXISTS PERMISSION_DENIED RESOURCE_EXHAUSTED FAILED_PRECONDITION "
    "ABORTED OUT_OF_RANGE UNIMPLEMENTED INTERNAL UNAVAILABLE DATA_LOSS "
    "UNAUTHENTICATED"
).split():
    setattr(StatusCode, _name, _Status(_name))


class ChannelCredentials:
    pass


class Compression:
    NoCompression = None
    Deflate = "deflate"
    Gzip = "gzip"


class RpcError(Exception):
    def code(self):
        return StatusCode.UNKNOWN

    def trailing_metadata(self):
        return []


class ClientCallDetails:
    pass


class UnaryUnaryClientInterceptor:
    pass


class UnaryStreamClientInterceptor:
    pass


class StreamUnaryClientInterceptor:
    pass


class StreamStreamClientInterceptor:
    pass


class Future:
    pass


class Channel:
    def unary_unary(self, *args, **kwargs):
        def call(*call_args, **call_kwargs):
            raise RpcError("grpc is not bundled in local-only runtime")

        return call


def ssl_channel_credentials(*args, **kwargs):
    return ChannelCredentials()


def insecure_channel(*args, **kwargs):
    return Channel()


def secure_channel(*args, **kwargs):
    return Channel()


def intercept_channel(channel, *interceptors):
    return channel


def unary_unary_rpc_method_handler(*args, **kwargs):
    return object()


def method_handlers_generic_handler(*args, **kwargs):
    return object()
PY
  cat > "$site_dir/grpc/_utilities.py" <<'PY'
def first_version_is_lower(current, required):
    return False
PY
}

thin_runtime_macho_binaries() {
  local site_dir="$1"
  local lipo_arch=""
  case "$TARGET_TRIPLE" in
    aarch64-apple-darwin) lipo_arch="arm64" ;;
    x86_64-apple-darwin)  lipo_arch="x86_64" ;;
  esac
  [[ -n "$lipo_arch" ]] || return 0
  command -v lipo >/dev/null 2>&1 || return 0

  local thin_count=0
  while IFS= read -r -d '' f; do
    if file "$f" | grep -q "universal binary"; then
      tmp="${f}.thin.$$"
      if lipo "$f" -thin "$lipo_arch" -output "$tmp" 2>/dev/null; then
        mv "$tmp" "$f"
        thin_count=$((thin_count + 1))
      else
        rm -f "$tmp"
      fi
    fi
  done < <(find "$site_dir" \( -name "*.so" -o -name "*.dylib" \) -type f -print0)
  if [[ "$thin_count" -gt 0 ]]; then
    echo "[prepare_bundled_python] Thinned universal Mach-O files for ${lipo_arch}: ${thin_count}"
  fi
}

prune_runtime_site() {
  local site_dir="$1"
  echo "[prepare_bundled_python] Pruning optional runtime dependencies..."

  rm -rf "$site_dir"/llama_index*

  rm -rf "$site_dir"/langchain_community*

  rm -rf "$site_dir"/av "$site_dir"/av-*.dist-info

  rm -rf "$site_dir"/jieba/analyse "$site_dir"/jieba/posseg "$site_dir"/jieba/lac_small

  write_grpc_local_stub "$site_dir"

  rm -f "$site_dir"/libggml*.dylib "$site_dir"/libwhisper*.dylib
  rm -f "$site_dir"/pywhispercpp/.dylibs/libwhisper.coreml.dylib

  thin_runtime_macho_binaries "$site_dir"
}

validate_runtime_site() {
  local python_bin="$1"
  local site_dir="$2"
  echo "[prepare_bundled_python] Validating runtime parsing dependencies..."
  PYTHONPATH="$site_dir" "$python_bin" - <<'PY'
import importlib
import shutil
import sys
import tempfile

required = [
    ("pypdf", "PDF text parsing"),
    ("pypdfium2", "PDF page rendering for OCR"),
    ("docx2txt", "DOCX text parsing"),
    ("pptx", "PPTX text parsing"),
    ("lxml", "python-pptx XML dependency"),
    ("openpyxl", "XLSX table parsing"),
    ("xlrd", "XLS table parsing"),
    ("numbers_parser", "Apple Numbers parsing"),
    ("rank_bm25", "hybrid lexical retrieval"),
    ("jieba", "Chinese BM25 tokenization"),
    ("pypinyin", "CJK filename alias generation"),
]
missing = []
for module_name, purpose in required:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(f"{module_name} ({purpose}): {exc}")

if missing:
    print("[runtime-check] ERROR: bundled runtime is missing required parser modules:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(1)

from pypdf import PdfReader  # noqa: F401,E402
import pypdfium2 as pdfium  # noqa: F401,E402
import docx2txt  # noqa: F401,E402
from pptx import Presentation  # noqa: F401,E402
from openpyxl import load_workbook  # noqa: F401,E402
from xlrd import open_workbook  # noqa: F401,E402
from numbers_parser import Document  # noqa: F401,E402
from rank_bm25 import BM25Okapi  # noqa: F401,E402
import jieba  # noqa: F401,E402
from pypinyin import lazy_pinyin  # noqa: F401,E402

from pywhispercpp.model import Model as WhisperCppModel  # noqa: E402
whisper_info = WhisperCppModel.system_info()
if "Metal" not in whisper_info or "COREML = 0" not in whisper_info:
    raise SystemExit(
        "[runtime-check] ERROR: pywhispercpp must be Metal GPU enabled and CoreML disabled; "
        f"system_info={whisper_info!r}"
    )

import chromadb  # noqa: E402
tmp_dir = tempfile.mkdtemp(prefix="unfoldly_chroma_check_")
try:
    client = chromadb.PersistentClient(path=tmp_dir)
    collection = client.get_or_create_collection("runtime_check_collection")
    collection.add(ids=["1"], documents=["runtime check"], embeddings=[[0.1, 0.2, 0.3]])
    if collection.count() != 1:
        raise SystemExit("[runtime-check] ERROR: Chroma local PersistentClient smoke test failed")
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)

print("[runtime-check] ok: pypdf, pypdfium2, docx2txt, pptx/lxml, openpyxl, xlrd, numbers_parser, BM25/jieba/pypinyin, pywhispercpp Metal GPU, chromadb")
PY
}

if [[ -f "$SITE_DIR/.installed_success" ]] && [[ "$SITE_REBUILD_ENABLED" != "true" ]]; then
  echo "[prepare_bundled_python] Existing Python site directory found ($SITE_DIR); skipping environment rebuild. Delete it first to rebuild."
  export PYTHON_RUNTIME_INSTALL="$INSTALL_DIR"
  export PYTHON_RUNTIME_SITE="$SITE_DIR"
  
  echo "[prepare_bundled_python] Incrementally updating backend source into site directory..."
  rm -rf \
    "$SITE_DIR/unfoldly_backend"* \
    "$SITE_DIR/backend_bundle"* \
    "$SITE_DIR/core"* \
    "$SITE_DIR/services"* \
    "$SITE_DIR/tools"* \
    "$SITE_DIR/utils"* \
    "$SITE_DIR/config"* \
    "$SITE_DIR/database"* \
    "$SITE_DIR/api_server.py"* \
    "$SITE_DIR/backend_core.py"*
  "$INSTALL_DIR/bin/python3" -m pip install "$ROOT_DIR" --no-deps --target "$SITE_DIR" --upgrade --force-reinstall
  "$INSTALL_DIR/bin/python3" "$PATCH_LLAMA_CPP_LOADER" "$SITE_DIR"
  prune_runtime_site "$SITE_DIR"
  validate_runtime_site "$INSTALL_DIR/bin/python3" "$SITE_DIR"
  
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
  fi
  exit 0
fi

if [[ -f "$SITE_DIR/.installed_success" ]] && [[ "$SITE_REBUILD_ENABLED" == "true" ]]; then
  echo "[prepare_bundled_python] UNFOLDLY_FORCE_REBUILD_SITE=${FORCE_REBUILD_SITE}; rebuild site directory from scratch."
fi

rm -rf "$SITE_DIR"
mkdir -p "$SITE_DIR"

echo "[prepare_bundled_python] Upgrading pip and installing build tools..."
"$INSTALL_DIR/bin/python3" -m pip install --upgrade pip wheel build

echo "[prepare_bundled_python] Collecting and building dependencies into wheelhouse..."
"$INSTALL_DIR/bin/python3" -m pip wheel -r "$ROOT_DIR/runtime-core.txt" -w "$WHEELHOUSE"

echo "[prepare_bundled_python] Installing dependencies into site directory..."
"$INSTALL_DIR/bin/python3" -m pip install --no-index --find-links="$WHEELHOUSE" -r "$ROOT_DIR/runtime-core.txt" --target "$SITE_DIR" --upgrade

  echo "[prepare_bundled_python] Fetching and building llama-cpp-python from GitHub (JamePeng)..."
  LLAMA_CPP_PYTHON_CACHE="$CACHE_DIR/llama-cpp-python-build"
  mkdir -p "$LLAMA_CPP_PYTHON_CACHE"
  cd "$LLAMA_CPP_PYTHON_CACHE"

  retry_git() {
    local max_attempts="${UNFOLDLY_GIT_RETRY_MAX:-3}"
    local attempt=1
    while true; do
      if "$@"; then
        return 0
      fi
      if [[ "$attempt" -ge "$max_attempts" ]]; then
        return 1
      fi
      echo "  [WARN] git command failed, attempt ${attempt}/${max_attempts}; retrying shortly..."
      sleep $((attempt * 2))
      attempt=$((attempt + 1))
    done
  }


  # export https_proxy=http://127.0.0.1:7897
  # export http_proxy=http://127.0.0.1:7897
  # export all_proxy=http://127.0.0.1:7897
  # use 'git clone ' to test the network speed
  # git clone https://github.com/ggerganov/llama.cpp.git

  if [[ -d "llama-cpp-python/.git" ]]; then
    echo "  Reusing local llama-cpp-python cache..."
    cd llama-cpp-python
    git remote set-url origin "$LLAMA_CPP_PYTHON_REPO"
  else
    echo "  First clone of llama-cpp-python with HTTP/1.1 and retry..."
    retry_git git -c http.version=HTTP/1.1 clone --progress --filter=blob:none --no-checkout "$LLAMA_CPP_PYTHON_REPO" llama-cpp-python
    cd llama-cpp-python
  fi

  echo "  Fetching pinned llama-cpp-python ref: ${LLAMA_CPP_PYTHON_REF}"
  retry_git git -c http.version=HTTP/1.1 fetch --force --tags origin "$LLAMA_CPP_PYTHON_REF" --depth 1
  git checkout --force FETCH_HEAD
  git reset --hard
  git clean -xfd
  echo "  Using llama-cpp-python commit: $(git rev-parse HEAD)"

  git config submodule."vendor/llama.cpp".url "https://github.com/ggerganov/llama.cpp.git"
  echo "  Initializing/updating llama.cpp submodule with HTTP/1.1 and retry..."
  retry_git git -c http.version=HTTP/1.1 submodule sync --recursive
  retry_git git -c http.version=HTTP/1.1 submodule update --init --recursive --progress --depth 1 --jobs 1
  echo "  Pinned submodules:"
  git submodule status --recursive

  echo "  Building and installing Qwen3-VL capable llama-cpp-python into site directory..."
  export CMAKE_ARGS="-DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF"
  "$INSTALL_DIR/bin/python3" -m pip install . --target "$SITE_DIR" --upgrade
  
  cat << 'EOF' > fix_bindings.py
import subprocess
import re
import sys
import os

site_dir = sys.argv[1]
filepath = os.path.join(site_dir, "llama_cpp/llama_cpp.py")
python_bin = sys.executable

while True:
    result = subprocess.run(
        [python_bin, "-c", "import sys; sys.path.insert(0, sys.argv[1]); from llama_cpp import Llama; print('SUCCESS')", site_dir],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": site_dir}
    )
    if "SUCCESS" in result.stdout:
        print("Successfully imported Llama!")
        break
    if result.returncode != 0:
        err = result.stderr
        match = re.search(r"dlsym\([^,]+,\s*([^)]+)\): symbol not found", err)
        if match:
            missing_symbol = match.group(1)
            print(f"Fixing missing symbol: {missing_symbol}")
            with open(filepath, 'r') as f:
                content = f.read()
            pattern = re.compile(r'@ctypes_function\s*\(\s*[\"\']' + re.escape(missing_symbol) + r'[\"\'].*?\)\s*def ' + re.escape(missing_symbol), re.DOTALL)
            new_content = pattern.sub(f'def {missing_symbol}', content)
            if new_content == content:
                new_content = content.replace(f'@ctypes_function("{missing_symbol}")', f'# @ctypes_function("{missing_symbol}")')
            with open(filepath, 'w') as f:
                f.write(new_content)
        else:
            print(f"Unhandled error:\n{err}")
            break
    else:
        print(f"Unknown output: {result.stdout}")
        break
EOF
  "$INSTALL_DIR/bin/python3" fix_bindings.py "$SITE_DIR"

  echo "[prepare_bundled_python] Applying preventive llama_cpp.py lazy-symbol patch for missing fork symbols..."
  "$INSTALL_DIR/bin/python3" - "$SITE_DIR" << 'SYMEOF'
import sys, os, re, ctypes, glob

site = sys.argv[1]
llama_py = os.path.join(site, "llama_cpp", "llama_cpp.py")
if not os.path.exists(llama_py):
    print(f"[sym_patch] {llama_py} not found; skipping")
    sys.exit(0)

dylib_paths = (
    glob.glob(os.path.join(site, "llama_cpp", "lib", "libllama*.dylib")) +
    glob.glob(os.path.join(site, "llama_cpp", "lib", "libggml*.dylib")) +
    glob.glob(os.path.join(site, "llama_cpp", "_llama_cpp*.so"))
)
if not dylib_paths:
    print("[sym_patch] llama_cpp dylib not found; skipping symbol validation")
    sys.exit(0)

exported = set()
for p in dylib_paths:
    try:
        lib = ctypes.CDLL(p)
        import subprocess
        nm_out = subprocess.run(["nm", "-gU", p], capture_output=True, text=True)
        for line in nm_out.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] in ("T", "U", "D"):
                sym = parts[2].lstrip("_")
                exported.add(sym)
    except Exception as e:
        print(f"[sym_patch] failed to read symbol table {p}: {e}")

txt = open(llama_py).read()

bound_syms = re.findall(r'@ctypes_function\(\s*["\']([^"\']+)["\']', txt)
missing = [s for s in bound_syms if s not in exported]

if not missing:
    print("[sym_patch] all bound symbols are exported; no patch needed")
    sys.exit(0)

patched = 0
for sym in missing:
    new_txt = re.sub(
        r'(@ctypes_function\(\s*["\']' + re.escape(sym) + r'["\'][^)]*\))',
        r'# \1  # patched: not exported by compiled lib',
        txt
    )
    if new_txt != txt:
        txt = new_txt
        patched += 1
        print(f"[sym_patch] commented missing symbol: {sym}")

if patched:
    open(llama_py, "w").write(txt)
    print(f"[sym_patch] patched {patched} missing symbols")
SYMEOF
  # ─────────────────────────────────────────────────────────────────────────────
cd "$ROOT_DIR"


echo "[prepare_bundled_python] CrispASR build skipped; speaker diarization is handled by gguf_diarizer.py"

echo "[prepare_bundled_python] Fetching and building pywhispercpp from GitHub (Metal GPU, CoreML disabled)..."
PYWHISPERCPP_CACHE="$CACHE_DIR/pywhispercpp-build"
rm -rf "$PYWHISPERCPP_CACHE"
mkdir -p "$PYWHISPERCPP_CACHE"
cd "$PYWHISPERCPP_CACHE"
retry_git git -c http.version=HTTP/1.1 clone --progress --filter=blob:none --no-checkout "$PYWHISPERCPP_REPO" pywhispercpp
cd pywhispercpp
echo "  Fetching pinned pywhispercpp ref: ${PYWHISPERCPP_REF}"
retry_git git -c http.version=HTTP/1.1 fetch --force --tags origin "$PYWHISPERCPP_REF" --depth 1
git checkout --force FETCH_HEAD
git reset --hard
git clean -xfd
echo "  Using pywhispercpp commit: $(git rev-parse HEAD)"
retry_git git -c http.version=HTTP/1.1 submodule sync --recursive
retry_git git -c http.version=HTTP/1.1 submodule update --init --recursive --progress --depth 1 --jobs 1
echo "  Pinned pywhispercpp submodules:"
git submodule status --recursive
# GGML_METAL: Apple Silicon GPU acceleration. CoreML requires a per-model
# *-encoder.mlmodelc next to every downloaded GGML/GGUF model; without it,
# whisper.cpp can abort in native code during WAV indexing. Keep Metal on and
# CoreML off so packaged ASR remains GPU-backed and stable.
rm -rf "$SITE_DIR"/pywhispercpp "$SITE_DIR"/pywhispercpp-*.dist-info \
       "$SITE_DIR"/_pywhispercpp*.so "$SITE_DIR"/libggml*.dylib "$SITE_DIR"/libwhisper*.dylib
PYTHON_INCLUDE_DIR="$INSTALL_DIR/include/python3.12"
PYTHON_LIBRARY="$INSTALL_DIR/lib/libpython3.12.dylib"
PYBIND11_FINDPYTHON=ON \
Python_EXECUTABLE="$INSTALL_DIR/bin/python3" \
Python_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" \
Python_LIBRARY="$PYTHON_LIBRARY" \
Python3_EXECUTABLE="$INSTALL_DIR/bin/python3" \
Python3_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" \
Python3_LIBRARY="$PYTHON_LIBRARY" \
WHISPER_COREML=0 \
WHISPER_COREML_ALLOW_FALLBACK=0 \
GGML_METAL=1 \
"$INSTALL_DIR/bin/python3" -m pip install . --target "$SITE_DIR" --upgrade
cd "$ROOT_DIR"

echo "[prepare_bundled_python] Packaging and installing backend source..."
"$INSTALL_DIR/bin/python3" -m pip install "$ROOT_DIR" --no-deps --target "$SITE_DIR"
"$INSTALL_DIR/bin/python3" "$PATCH_LLAMA_CPP_LOADER" "$SITE_DIR"

echo "[prepare_bundled_python] Pruning redundant dependencies to reduce bundle size..."
rm -rf "$SITE_DIR"/onnxruntime*
rm -rf "$SITE_DIR"/kubernetes*

rm -rf "$SITE_DIR"/sympy*
rm -rf "$SITE_DIR"/mpmath*


rm -rf "$SITE_DIR"/pandas*

rm -rf "$SITE_DIR"/setuptools*
rm -rf "$SITE_DIR"/build*
rm -rf "$SITE_DIR"/_distutils_hack*
rm -rf "$SITE_DIR"/pkg_resources*
rm -f "$SITE_DIR"/distutils-precedence.pth
rm -rf "$SITE_DIR"/pip*
rm -rf "$SITE_DIR"/wheel*

rm -rf "$SITE_DIR"/networkx*


rm -rf "$SITE_DIR"/nltk*

if [[ -d "$SITE_DIR/modelscope" ]]; then
  for subdir in models pipelines trainers preprocessors exporters metrics swift server; do
    rm -rf "$SITE_DIR/modelscope/$subdir"
  done
  echo "  modelscope: pruned to hub download modules only"
fi

rm -rf "$SITE_DIR"/numpy/f2py*
rm -rf "$SITE_DIR"/numpy/typing/tests*

rm -rf "$SITE_DIR"/tzdata*
rm -rf "$SITE_DIR"/pytz*

rm -rf "$SITE_DIR"/uvloop*

rm -rf "$SITE_DIR"/rich*
rm -rf "$SITE_DIR"/colorama*

rm -rf "$SITE_DIR"/joblib*

rm -rf "$SITE_DIR"/flatbuffers*

rm -rf "$SITE_DIR"/bin

rm -rf "$SITE_DIR"/hf_xet*

# lxml — python-pptx runtime dependency. Keep it, otherwise PPTX indexing fails.

rm -rf "$SITE_DIR"/langchain_classic*

rm -rf "$SITE_DIR"/sqlalchemy*
rm -rf "$SITE_DIR"/greenlet*

rm -rf "$SITE_DIR"/av "$SITE_DIR"/av-*.dist-info

rm -rf "$SITE_DIR"/faster_whisper*
rm -rf "$SITE_DIR"/ctranslate2*
rm -rf "$SITE_DIR"/faster_whisper-*.dist-info
rm -rf "$SITE_DIR"/ctranslate2-*.dist-info

find "$SITE_DIR" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find "$SITE_DIR" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
find "$SITE_DIR" -type d -name "docs" -exec rm -rf {} + 2>/dev/null || true
find "$SITE_DIR" -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true
find "$SITE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SITE_DIR" -name "*.pyc" -delete 2>/dev/null || true

prune_runtime_site "$SITE_DIR"


rm -rf "$SITE_DIR"/llama_cpp/lib/cmake
rm -rf "$SITE_DIR"/llama_cpp/lib/pkgconfig

echo "[prepare_bundled_python] Converting duplicate llama_cpp dylibs to symlinks..."
for versioned in "$SITE_DIR"/llama_cpp/lib/lib*.*.*.dylib "$SITE_DIR"/llama_cpp/lib/lib*.*.*.*.dylib; do
  [[ -f "$versioned" ]] || continue
  base=$(basename "$versioned")
  lib_stem=$(echo "$base" | sed -E 's/^(lib[^.]+)\..*/\1/')
  dir=$(dirname "$versioned")
  for dup in "$dir/${lib_stem}.dylib" "$dir/${lib_stem}.0.dylib"; do
    [[ -f "$dup" && ! -L "$dup" ]] || continue
    rm -f "$dup"
    ln -s "$base" "$dup"
  done
done

rm -rf "$SITE_DIR"/numpy/doc*
rm -rf "$SITE_DIR"/numpy/_pyinstaller*

echo "[prepare_bundled_python] Stripping binary debug symbols and re-signing..."
find "$SITE_DIR" \( -name "*.so" -o -name "*.dylib" \) -print0 | while IFS= read -r -d '' f; do
  strip -x "$f" 2>/dev/null || true
  codesign --force --sign - "$f" 2>/dev/null || true
done

SITE_SIZE_MB=$(du -sm "$SITE_DIR" | awk '{print $1}')
echo "[prepare_bundled_python] Site directory size after pruning: ${SITE_SIZE_MB}MB"

validate_runtime_site "$INSTALL_DIR/bin/python3" "$SITE_DIR"

export PYTHON_RUNTIME_SITE="$SITE_DIR"
touch "$SITE_DIR/.installed_success"
echo "[prepare_bundled_python] Done. INSTALL=${INSTALL_DIR}, SITE=${SITE_DIR}"
