#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
FRONTEND_DIR="$SCRIPT_DIR/apps/desktop"
VENV_DIR="${FILEAGENT_VENV_DIR:-$SCRIPT_DIR/.venv}"

RELEASE_MODE=0
TAURI_ARGS=()
for arg in "$@"; do
  case "$arg" in
    release|--release|-r)
      RELEASE_MODE=1
      ;;
    *)
      TAURI_ARGS+=("$arg")
      ;;
  esac
done

# 1. Rust toolchain
if ! command -v cargo >/dev/null 2>&1; then
  [[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "[tauri-dev] ERROR: cargo not found. Install Rust: https://rustup.rs"
  exit 1
fi

# 2. Python venv
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[tauri-dev] ERROR: Python venv not found at $VENV_DIR"
  echo "           Run: python3 -m venv $VENV_DIR && $VENV_DIR/bin/pip install -r requirements.txt"
  exit 1
fi

PYTHON="$VENV_DIR/bin/python"
PYTHON_LIBDIR="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
PYTHON_FWPREFIX="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX") or "")')"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"

echo "[tauri-dev] Python: $PYTHON ($(${PYTHON} --version 2>&1))"

export FILEAGENT_ENABLE_LLAMA_INDEX_FALLBACK="${FILEAGENT_ENABLE_LLAMA_INDEX_FALLBACK:-0}"

# Keep dev behavior aligned with the bundled app: LlamaIndex is not packaged.
# It can be explicitly retained for experiments by setting either:
#   FILEAGENT_ENABLE_LLAMA_INDEX_FALLBACK=1 or FILEAGENT_KEEP_LLAMA_INDEX=1
_LLAMA_INDEX_FALLBACK_LC="$(printf '%s' "${FILEAGENT_ENABLE_LLAMA_INDEX_FALLBACK:-0}" | tr '[:upper:]' '[:lower:]')"
_KEEP_LLAMA_INDEX_LC="$(printf '%s' "${FILEAGENT_KEEP_LLAMA_INDEX:-0}" | tr '[:upper:]' '[:lower:]')"
case "${_LLAMA_INDEX_FALLBACK_LC}:${_KEEP_LLAMA_INDEX_LC}" in
  1:*|true:*|yes:*|on:*|*:1|*:true|*:yes|*:on)
    echo "[tauri-dev] Keeping LlamaIndex packages for explicit fallback testing."
    ;;
  *)
    if "$PYTHON" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("llama_index") else 1)
PY
    then
      echo "[tauri-dev] Removing LlamaIndex from dev venv to match bundled runtime..."
      "$PYTHON" -m pip uninstall -y \
        llama-index \
        llama-index-core \
        llama-index-instrumentation \
        llama-index-workflows \
        llama-index-llms-openai \
        llama-index-vector-stores-chroma \
        llama-index-readers-file \
        llama-index-embeddings-huggingface >/dev/null || true
      rm -rf "$SITE_PACKAGES"/llama_index "$SITE_PACKAGES"/llama_index-*.dist-info "$SITE_PACKAGES"/llama_index_*.dist-info
    fi
    ;;
esac

if [[ "${FILEAGENT_SKIP_RUNTIME_PARITY_CHECK:-0}" != "1" ]]; then
  if ! "$PYTHON" - <<'PY'
import importlib.util
import sys

required = {
    "docx2txt": "docx text extraction",
    "pptx": "PowerPoint text extraction",
    "pypdf": "PDF text extraction",
    "pypdfium2": "PDF page rendering for OCR",
    "openpyxl": "xlsx parsing",
    "xlrd": "xls parsing",
    "numbers_parser": ".numbers parsing",
}

missing = [f"{mod} ({desc})" for mod, desc in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    sys.stderr.write(
        "[tauri-dev] ERROR: runtime parity check failed. Missing Python modules:\n"
        + "\n".join(f"  - {item}" for item in missing)
        + "\n"
    )
    raise SystemExit(1)
PY
  then
    echo "[tauri-dev] Install dev dependencies again so indexing matches the bundled app:"
    echo "           $PYTHON -m pip install -r requirements.txt"
    echo "           (override with FILEAGENT_SKIP_RUNTIME_PARITY_CHECK=1 if you intentionally want a degraded dev env)"
    exit 1
  fi
fi

export PYO3_PYTHON="$PYTHON"
export LIBRARY_PATH="${PYTHON_LIBDIR}:${LIBRARY_PATH:-}"
if [[ -n "$PYTHON_FWPREFIX" ]]; then
  export DYLD_FALLBACK_FRAMEWORK_PATH="${PYTHON_FWPREFIX}:${DYLD_FALLBACK_FRAMEWORK_PATH:-}"
  export DYLD_FRAMEWORK_PATH="${PYTHON_FWPREFIX}:${DYLD_FRAMEWORK_PATH:-}"
fi
export PYTHONPATH="${SCRIPT_DIR}:${SITE_PACKAGES}:${PYTHONPATH:-}"
unset PYTHONHOME 2>/dev/null || true

export MODELSCOPE_DOMAIN="${MODELSCOPE_DOMAIN:-www.modelscope.ai}"

if [[ "$RELEASE_MODE" == "1" ]]; then
  export FILEAGENT_LOG_LEVEL="${FILEAGENT_LOG_LEVEL:-WARNING}"
  export FILEAGENT_UVICORN_LOG_LEVEL="${FILEAGENT_UVICORN_LOG_LEVEL:-warning}"
  export FILEAGENT_LOG_QUIET="${FILEAGENT_LOG_QUIET:-1}"
  export FILEAGENT_INTENT_STRICT_RETRY="${FILEAGENT_INTENT_STRICT_RETRY:-0}"
  export FILEAGENT_QUERY_STREAM_DEBUG="${FILEAGENT_QUERY_STREAM_DEBUG:-0}"
  export RUST_LOG="${RUST_LOG:-warn}"
  echo "[tauri-dev] release logging mode enabled: FILEAGENT_LOG_LEVEL=${FILEAGENT_LOG_LEVEL}"
fi

# 3. npm deps
cd "$FRONTEND_DIR"

if [[ ! -d "node_modules" ]]; then
  echo "[tauri-dev] Installing npm dependencies..."
  npm install || exit 1
fi

# 4. Launch
echo ""
echo "============================================"
echo "  Unfoldly - Tauri + PyO3 Dev Mode"
echo "  Zero ports / Zero HTTP / In-process"
if [[ "$RELEASE_MODE" == "1" ]]; then
  echo "  Log Mode: release (reduced output)"
fi
echo "============================================"
echo ""

if [[ "$RELEASE_MODE" == "1" ]]; then
  echo "[tauri-dev] release stderr filter enabled for llama embedding spam"
  if (( ${#TAURI_ARGS[@]} > 0 )); then
    exec npx @tauri-apps/cli dev "${TAURI_ARGS[@]}" 2> >(
      awk 'index($0, "init: embeddings required but some input tokens were not marked as outputs -> overriding") == 0 { print > "/dev/stderr"; fflush("/dev/stderr") }'
    )
  else
    exec npx @tauri-apps/cli dev 2> >(
      awk 'index($0, "init: embeddings required but some input tokens were not marked as outputs -> overriding") == 0 { print > "/dev/stderr"; fflush("/dev/stderr") }'
    )
  fi
elif (( ${#TAURI_ARGS[@]} > 0 )); then
  exec npx @tauri-apps/cli dev "${TAURI_ARGS[@]}"
else
  exec npx @tauri-apps/cli dev
fi
