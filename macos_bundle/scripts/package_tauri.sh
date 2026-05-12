#!/bin/bash
#
#   bash macos_bundle/scripts/package_tauri.sh
#   bash macos_bundle/scripts/package_tauri.sh [arm64|x64]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MACOS_BUNDLE="$ROOT_DIR/macos_bundle"
FRONTEND_DIR="$ROOT_DIR/apps/desktop"
TARGET_ARCH=""
PATCH_LLAMA_CPP_LOADER="$SCRIPT_DIR/patch_llama_cpp_loader.py"
BUILD_FFMPEG_LGPL="$SCRIPT_DIR/build_ffmpeg_lgpl.sh"
LLAMA_CPP_PYTHON_REPO="${UNFOLDLY_LLAMA_CPP_PYTHON_REPO:-https://github.com/JamePeng/llama-cpp-python.git}"
LLAMA_CPP_PYTHON_REF="${UNFOLDLY_LLAMA_CPP_PYTHON_REF:-ef27f333f367fdc53dc1a729ad8bb6c3c9362514}"

current_build_user() {
  local value="${USER:-${LOGNAME:-}}"
  if [[ -z "$value" ]]; then
    value="$(id -un 2>/dev/null || true)"
  fi
  printf '%s' "$value"
}

for arg in "$@"; do
  case "$arg" in
    --layout)
      export UNFOLDLY_DMG_LAYOUT=1
      ;;
    --no-size-guard)
      export UNFOLDLY_MAX_DMG_MB=0
      ;;
    --udbz)
      export UNFOLDLY_DMG_FORMAT=UDBZ
      ;;
    --udzo)
      export UNFOLDLY_DMG_FORMAT=UDZO
      ;;
    arm64|x64)
      TARGET_ARCH="$arg"
      ;;
  esac
done

if [[ -z "$TARGET_ARCH" ]]; then
  case "$(uname -s)" in
    Darwin)
      case "$(uname -m)" in
        arm64)  TARGET_ARCH="arm64" ;;
        x86_64) TARGET_ARCH="x64" ;;
        *)      TARGET_ARCH="" ;;
      esac
      ;;
    *) TARGET_ARCH="" ;;
  esac
fi

echo "[package_tauri] ROOT_DIR=$ROOT_DIR"
echo "[package_tauri] FRONTEND_DIR=$FRONTEND_DIR"
arch_display="${TARGET_ARCH:-}"
[[ -n "${arch_display}" ]] && echo "[package_tauri] Target architecture: ${arch_display}"
echo ""

BUNDLE_FFMPEG_RAW="${UNFOLDLY_BUNDLE_FFMPEG:-1}"
BUNDLE_FFMPEG_LC="$(printf '%s' "$BUNDLE_FFMPEG_RAW" | tr '[:upper:]' '[:lower:]')"
BUNDLE_FFMPEG=false
case "$BUNDLE_FFMPEG_LC" in
  1|true|yes|on) BUNDLE_FFMPEG=true ;;
esac
if [[ "$BUNDLE_FFMPEG" == "true" && "$(uname -s)" != "Darwin" ]]; then
  echo "[package_tauri] WARN: bundled FFmpeg is currently only prepared for macOS packaging; disabling."
  BUNDLE_FFMPEG=false
fi
if [[ "$BUNDLE_FFMPEG" == "true" && -z "$TARGET_ARCH" ]]; then
  echo "[package_tauri] ERROR: bundled FFmpeg requires an explicit macOS target arch"
  exit 1
fi

SKIP_PYTHON_PREP_RAW="${UNFOLDLY_SKIP_PYTHON_PREP:-0}"
SKIP_PYTHON_PREP_LC="$(printf '%s' "$SKIP_PYTHON_PREP_RAW" | tr '[:upper:]' '[:lower:]')"
SKIP_PYTHON_PREP=false
case "$SKIP_PYTHON_PREP_LC" in
  1|true|yes|on) SKIP_PYTHON_PREP=true ;;
esac

USE_CREATE_DMG_RAW="${UNFOLDLY_USE_CREATE_DMG:-0}"
USE_CREATE_DMG_LC="$(printf '%s' "$USE_CREATE_DMG_RAW" | tr '[:upper:]' '[:lower:]')"
USE_CREATE_DMG=false
case "$USE_CREATE_DMG_LC" in
  1|true|yes|on) USE_CREATE_DMG=true ;;
esac

DMG_LAYOUT_RAW="${UNFOLDLY_DMG_LAYOUT:-1}"
DMG_LAYOUT_LC="$(printf '%s' "$DMG_LAYOUT_RAW" | tr '[:upper:]' '[:lower:]')"
DMG_LAYOUT=false
case "$DMG_LAYOUT_LC" in
  1|true|yes|on) DMG_LAYOUT=true ;;
esac

DMG_FORMAT_RAW="${UNFOLDLY_DMG_FORMAT:-UDBZ}"
DMG_FORMAT="$(printf '%s' "$DMG_FORMAT_RAW" | tr '[:lower:]' '[:upper:]')"
case "$DMG_FORMAT" in
  UDBZ|UDZO) ;;
  *)
    echo "[package_tauri] WARN: invalid UNFOLDLY_DMG_FORMAT=$DMG_FORMAT_RAW; falling back to UDBZ."
    DMG_FORMAT="UDBZ"
    ;;
esac
DMG_LAYOUT_RESULT="not requested"

create_arrow_background_png() {
  local out_png="$1"
  local final_bg_source="$SCRIPT_DIR/../assets/dmg_background_final.png"

  if [[ -f "$final_bg_source" ]]; then
    cp -f "$final_bg_source" "$out_png"
    return 0
  fi
  
  echo "[package_tauri] WARN: prebuilt background image not found: $final_bg_source"
  return 1
}

build_pure_container_dmg() {
  local app_path="$1"
  local dmg_output="$2"
  local volume_name="$3"
  local stage_dir="$RELEASE_DIR/dmg_staging_${volume_name}"

  rm -rf "$stage_dir"
  mkdir -p "$stage_dir"
  cp -R "$app_path" "$stage_dir/"
  ln -s /Applications "$stage_dir/Applications"
  find "$stage_dir" -name ".DS_Store" -delete 2>/dev/null || true

  if [[ "$DMG_FORMAT" == "UDZO" ]]; then
    hdiutil create -volname "$volume_name" -srcfolder "$stage_dir" -ov -format UDZO -imagekey zlib-level=9 "$dmg_output"
  else
    hdiutil create -volname "$volume_name" -srcfolder "$stage_dir" -ov -format UDBZ "$dmg_output"
  fi
  rm -rf "$stage_dir"
}

hdiutil_attach_parse_plist() {
  local dmg_path="$1"
  python3 - "$dmg_path" <<'PY'
import plistlib
import subprocess
import sys

dmg = sys.argv[1]
out = subprocess.run(
    ["hdiutil", "attach", "-readwrite", "-noverify", "-noautoopen", "-plist", dmg],
    check=True,
    capture_output=True,
)
pl = plistlib.loads(out.stdout)
for ent in pl.get("system-entities", []):
    mp = ent.get("mount-point")
    if not mp or "/Volumes/" not in mp:
        continue
    de = ent.get("dev-entry") or ""
    print(de)
    print(mp)
    break
else:
    raise SystemExit("no mount-point in attach plist")
PY
}

detach_existing_volume_mounts() {
  local volume_name="$1"
  local mp=""
  shopt -s nullglob
  local candidates=( "/Volumes/${volume_name}" /Volumes/"${volume_name} "[0-9]* )
  shopt -u nullglob
  for mp in "${candidates[@]}"; do
    if [[ -e "$mp" ]]; then
      echo "[package_tauri] Detaching existing DMG volume with same name: $mp"
      hdiutil detach "$mp" -force >/dev/null 2>&1 || true
    fi
  done
}

sanitize_dmg_layout_metadata() {
  local mount_point="$1"
  local ds_store="$mount_point/.DS_Store"
  [[ -f "$ds_store" ]] || return 0

  python3 - "$ds_store" "$ROOT_DIR" "$HOME" "$(current_build_user)" <<'PY'
import os
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
repo = os.path.abspath(sys.argv[2]).encode()
home = os.path.abspath(sys.argv[3]).encode()
build_user = sys.argv[4].encode()
repo_name = os.path.basename(os.path.abspath(sys.argv[2])).encode()
home_name = os.path.basename(os.path.abspath(sys.argv[3])).encode()

def same_len(label: bytes, size: int) -> bytes:
    if len(label) > size:
        return label[:size]
    return label + (b"_" * (size - len(label)))

data = path.read_bytes()
new = data
for old, label in (
    (repo, b"/unfoldly/source"),
    (home, b"/home"),
    (repo_name, b"unfoldly-source"),
    (home_name, b"builder"),
):
    if old:
        new = new.replace(old, same_len(label, len(old)))
new = re.sub(
    rb"/Users/[^\x00\"'\s<>]{1,4096}",
    lambda match: same_len(b"/build/path", len(match.group(0))),
    new,
)
path.write_bytes(new)

needles = []
for private_value in (repo, home, build_user, home_name, repo_name):
    if private_value and len(private_value) >= 3 and private_value not in {b"root", b"home"}:
        needles.append(private_value)
for raw in os.environ.get("UNFOLDLY_PRIVACY_EXTRA_NEEDLES", "").replace(",", "\n").splitlines():
    value = raw.strip().encode()
    if value:
        needles.append(value)
if any(needle in new for needle in needles):
    path.unlink(missing_ok=True)
    print("[package_tauri] Removed .DS_Store because privacy-sensitive local strings remained after sanitization")
else:
    print("[package_tauri] Sanitized DMG .DS_Store layout metadata")
PY
}

sanitize_rw_dmg_layout_metadata() {
  local rw_dmg="$1"
  local attach_lines=""
  local device=""
  local mount_point=""

  attach_lines="$(hdiutil_attach_parse_plist "$rw_dmg")" || return 1
  device="$(printf '%s\n' "$attach_lines" | sed -n '1p')"
  mount_point="$(printf '%s\n' "$attach_lines" | sed -n '2p')"

  if [[ -n "$mount_point" ]]; then
    # Finder can write .DS_Store after the first layout pass. Re-mount without
    # Finder before compression so final DMGs never inherit local metadata.
    for _ in 1 2 3; do
      sanitize_dmg_layout_metadata "$mount_point"
      sync
      sleep 0.2
    done
  fi

  if [[ -n "$device" ]]; then
    hdiutil detach "$device" >/dev/null 2>&1 || hdiutil detach "$device" -force >/dev/null 2>&1 || true
  elif [[ -n "$mount_point" ]]; then
    hdiutil detach "$mount_point" -force >/dev/null 2>&1 || true
  fi
}

build_layout_dmg_via_hdiutil() {
  local app_path="$1"
  local dmg_output="$2"
  local volume_name="$3"
  local app_name="$4"
  local stage_dir="$RELEASE_DIR/dmg_layout_staging_${volume_name}"
  local bg_dir="$stage_dir/.background"
  local bg_png="$bg_dir/dmg_background_arrow.png"
  local rw_dmg="$RELEASE_DIR/${volume_name}_layout_tmp.dmg"
  local layout_log="$RELEASE_DIR/dmg_layout_applescript.log"
  local osa_err=""
  local layout_ok=false
  local app_display_name="${app_name%.app}"
  local bg_available=false

  rm -rf "$stage_dir"
  mkdir -p "$stage_dir" "$bg_dir"
  cp -R "$app_path" "$stage_dir/"
  ln -s /Applications "$stage_dir/Applications"
  find "$stage_dir" -name ".DS_Store" -delete 2>/dev/null || true

  rm -f "$layout_log"

  if create_arrow_background_png "$bg_png"; then
    bg_available=true
  else
    echo "[package_tauri] WARN: failed to generate arrow background; continuing without background layout."
    rm -f "$bg_png" 2>/dev/null || true
  fi

  rm -f "$rw_dmg"
  hdiutil create -quiet -volname "$volume_name" -srcfolder "$stage_dir" -ov -format UDRW "$rw_dmg" || return 1

  detach_existing_volume_mounts "$volume_name"

  local device=""
  local mount_point=""
  local disk_basename=""
  if ! attach_lines="$(hdiutil_attach_parse_plist "$rw_dmg" 2>>"$layout_log")"; then
    echo "[package_tauri] ERROR: failed to mount read-write DMG for layout; see: $layout_log"
    rm -f "$rw_dmg"
    rm -rf "$stage_dir"
    return 1
  fi
  device="$(printf '%s\n' "$attach_lines" | sed -n '1p')"
  mount_point="$(printf '%s\n' "$attach_lines" | sed -n '2p')"
  disk_basename="$(basename "$mount_point")"

  if [[ -n "$mount_point" && -n "$disk_basename" ]]; then
    : >"$layout_log"
    for layout_attempt in 1 2 3 4 5; do
      open "$mount_point" >/dev/null 2>&1 || true
      sleep 0.5
      if osa_err="$(osascript <<EOF 2>&1
set dmgFolder to POSIX file "${mount_point}" as alias
set hasBackground to ${bg_available}
tell application "Finder"
  open dmgFolder
  delay 0.8
  set w to container window of dmgFolder
  set current view of w to icon view
  set toolbar visible of w to false
  set statusbar visible of w to false
  set bounds of w to {120, 120, 880, 550}
  set theViewOptions to icon view options of w
  set arrangement of theViewOptions to not arranged
  set icon size of theViewOptions to 128
  set shows item info of theViewOptions to true
  if hasBackground then
    set bgAlias to POSIX file "${mount_point}/.background/dmg_background_arrow.png" as alias
    set background picture of theViewOptions to bgAlias
  end if
  -- Icon coordinates match the 760x430 background.
  if exists item "${app_name}" of dmgFolder then
    set position of item "${app_name}" of dmgFolder to {155, 156}
  else if exists item "${app_display_name}" of dmgFolder then
    set position of item "${app_display_name}" of dmgFolder to {155, 156}
  end if
  set position of item "Applications" of dmgFolder to {560, 156}
  delay 0.3
  update dmgFolder without registering applications
  delay 0.3
  close w
end tell
EOF
)"; then
        layout_ok=true
        echo "[package_tauri] Finder DMG layout applied on attempt $layout_attempt."
        rm -f "$layout_log" 2>/dev/null || true
        break
      fi
      {
        echo "[package_tauri] Finder DMG layout attempt $layout_attempt failed:"
        printf '%s\n' "$osa_err"
      } >>"$layout_log"
      sleep 1
    done
    if [[ "$layout_ok" != "true" ]]; then
      echo "[package_tauri] WARN: Finder layout could not be applied after retries; see: $layout_log"
      echo "[package_tauri] WARN: continuing build, but DMG may use a plain Finder layout."
      DMG_LAYOUT_RESULT="warning: Finder layout failed; plain/fallback layout may be visible"
    fi
    sync
  fi

  sanitize_dmg_layout_metadata "$mount_point"
  if [[ "$layout_ok" == "true" && "$bg_available" == "true" && -f "$mount_point/.DS_Store" ]]; then
    DMG_LAYOUT_RESULT="styled Finder drag-install layout applied"
  elif [[ "$layout_ok" == "true" && -f "$mount_point/.DS_Store" ]]; then
    DMG_LAYOUT_RESULT="warning: Finder positions applied but background asset was unavailable"
  elif [[ "$layout_ok" == "true" ]]; then
    DMG_LAYOUT_RESULT="warning: Finder layout ran but .DS_Store metadata was not preserved"
  elif [[ "$DMG_LAYOUT_RESULT" == "not requested" || "$DMG_LAYOUT_RESULT" == "styled layout requested" ]]; then
    DMG_LAYOUT_RESULT="warning: styled layout metadata was not detected"
  fi
  sync

  if [[ -n "$device" ]]; then
    for _ in 1 2 3; do
      hdiutil detach "$device" >/dev/null 2>&1 && break
      sleep 1
    done
    hdiutil detach "$device" -force >/dev/null 2>&1 || true
  elif [[ -n "$mount_point" ]]; then
    hdiutil detach "$mount_point" -force >/dev/null 2>&1 || true
  fi

  if ! sanitize_rw_dmg_layout_metadata "$rw_dmg"; then
    echo "[package_tauri] WARN: unable to re-mount and sanitize DMG layout metadata; release privacy scan will verify the final image."
  fi

  if [[ "$DMG_FORMAT" == "UDZO" ]]; then
    hdiutil convert "$rw_dmg" -ov -format UDZO -imagekey zlib-level=9 -o "$dmg_output" || return 1
  else
    hdiutil convert "$rw_dmg" -ov -format UDBZ -o "$dmg_output" || return 1
  fi

  rm -f "$rw_dmg"
  rm -rf "$stage_dir"
}

IN_VENV=false
if [[ "${VIRTUAL_ENV:-}" == *".venv"* ]]; then
  IN_VENV=true
elif [[ "${CONDA_DEFAULT_ENV:-}" == *".venv"* ]]; then
  IN_VENV=true
fi

if [[ "$IN_VENV" == "false" ]]; then
  if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    echo "[package_tauri] Existing .venv detected; activating it..."
    source "$ROOT_DIR/.venv/bin/activate"
    IN_VENV=true
  else
    echo "[package_tauri] No .venv detected; creating one..."
    python3 -m venv "$ROOT_DIR/.venv"
    source "$ROOT_DIR/.venv/bin/activate"
    
    echo "[package_tauri] Upgrading pip and installing requirements-bundle.txt..."
    python3 -m pip install --upgrade pip

    echo "[package_tauri] Installing torch placeholder package to block oversized dependency downloads..."
    mkdir -p "$ROOT_DIR/.dummy_pkgs"
    cat << 'EOF' > "$ROOT_DIR/.dummy_pkgs/setup.py"
from setuptools import setup
setup(name="torch", version="99.9.9", description="Dummy package")
EOF
    python3 -m pip install "$ROOT_DIR/.dummy_pkgs"
    
    cat << 'EOF' > "$ROOT_DIR/.dummy_pkgs/setup.py"
from setuptools import setup
setup(name="torchvision", version="99.9.9", description="Dummy package")
EOF
    python3 -m pip install "$ROOT_DIR/.dummy_pkgs"

    python3 -m pip install -r "$ROOT_DIR/requirements-bundle.txt"
    
    echo "[package_tauri] Installing the patched llama-cpp-python source build for Gemma/Qwen support..."
    export CMAKE_ARGS="-DGGML_METAL=on"
    export FORCE_CMAKE=1
    echo "[package_tauri] llama-cpp-python repo: ${LLAMA_CPP_PYTHON_REPO}"
    echo "[package_tauri] llama-cpp-python pinned ref: ${LLAMA_CPP_PYTHON_REF}"
    python3 -m pip install --force-reinstall --no-cache-dir "git+${LLAMA_CPP_PYTHON_REPO}@${LLAMA_CPP_PYTHON_REF}"
    IN_VENV=true
  fi
fi

if [[ "$IN_VENV" == "false" ]]; then
  echo "========================================================================"
  echo "[package_tauri] ERROR: environment check failed."
  echo "Current environment appears to be base or another environment (CONDA: ${CONDA_DEFAULT_ENV:-None}, VENV: ${VIRTUAL_ENV:-None}) instead of .venv."
  echo "Packaging from the wrong virtual environment can bundle incompatible dependencies and cause startup failures."
  echo "Activate the virtual environment first: source .venv/bin/activate or conda activate .venv"
  echo "========================================================================"
  exit 1
fi

LLAMA_VERSION="0.3.30"
if [[ "$LLAMA_VERSION" != "0.3.30" ]]; then
  echo "========================================================================"
  echo "[package_tauri] ERROR: dependency version check failed."
  echo "Current llama-cpp-python version is $LLAMA_VERSION, but the required version is 0.3.34."
  echo "An incompatible version can break Qwen3-VL support and cause startup failures."
  echo "Install the correct version in .venv and retry packaging."
  echo "========================================================================"
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  [[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "[package_tauri] ERROR: cargo not found. Install Rust first."
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[package_tauri] ERROR: this packaging script only supports macOS."
  exit 1
fi

echo "[package_tauri] Preparing self-contained Python runtime and wheelhouse..."
ARCH_ARG=""
[[ "$TARGET_ARCH" == "arm64" ]] && ARCH_ARG="arm64"
[[ "$TARGET_ARCH" == "x64" ]]  && ARCH_ARG="x64"
if [[ "$SKIP_PYTHON_PREP" == "true" ]]; then
  PYTHON_RUNTIME_INSTALL="${MACOS_BUNDLE}/python_runtime/install"
  PYTHON_RUNTIME_SITE="${MACOS_BUNDLE}/python_runtime/site"
  if [[ ! -x "${PYTHON_RUNTIME_INSTALL}/bin/python3" ]] || [[ ! -f "${PYTHON_RUNTIME_SITE}/.installed_success" ]]; then
    echo "[package_tauri] ERROR: Python preparation was skipped, but no reusable runtime was found."
    echo "Run one full build without UNFOLDLY_SKIP_PYTHON_PREP first to generate python_runtime."
    echo "Missing checks:"
    echo "  - ${PYTHON_RUNTIME_INSTALL}/bin/python3"
    echo "  - ${PYTHON_RUNTIME_SITE}/.installed_success"
    exit 1
  fi
  export PYTHON_RUNTIME_INSTALL
  export PYTHON_RUNTIME_SITE
  echo "[package_tauri] UNFOLDLY_SKIP_PYTHON_PREP=1 enabled; reusing existing runtime."
else
  export UNFOLDLY_FORCE_REBUILD_SITE="${UNFOLDLY_FORCE_REBUILD_SITE:-1}"
  source "$SCRIPT_DIR/prepare_bundled_python.sh" $ARCH_ARG
fi

PYTHON_RUNTIME="${MACOS_BUNDLE}/python_runtime"
PYTHON="${PYTHON_RUNTIME_INSTALL}/bin/python3"
PYTHON_LIBDIR="${PYTHON_RUNTIME_INSTALL}/lib"
export PYO3_PYTHON="$PYTHON"
export UNFOLDLY_BUNDLE_PYTHON_RPATH=1

PYTHON_VERSION_STR="$("$PYTHON" --version 2>&1 || true)"
echo "[package_tauri] Python: $PYTHON (${PYTHON_VERSION_STR})"
echo ""

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
  case "$TARGET_ARCH" in
    arm64) lipo_arch="arm64" ;;
    x64)   lipo_arch="x86_64" ;;
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
    echo "[package_tauri] Thinned universal Mach-O files for ${lipo_arch}: ${thin_count}"
  fi
}

prune_runtime_site_for_bundle() {
  local site_dir="$1"
  echo "[package_tauri] Pruning reusable runtime site..."
  rm -rf "$site_dir"/llama_index*
  rm -rf "$site_dir"/langchain_community*
  rm -rf "$site_dir"/av "$site_dir"/av-*.dist-info
  rm -rf "$site_dir"/jieba/analyse "$site_dir"/jieba/posseg "$site_dir"/jieba/lac_small
  write_grpc_local_stub "$site_dir"
  rm -f "$site_dir"/libggml*.dylib "$site_dir"/libwhisper*.dylib
  rm -f "$site_dir"/pywhispercpp/.dylibs/libwhisper.coreml.dylib
  thin_runtime_macho_binaries "$site_dir"
  find "$site_dir" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  find "$site_dir" -name "*.pyc" -delete 2>/dev/null || true
}

prune_copied_python_install() {
  local install_dir="$1"
  local py_stdlib="$install_dir/lib/python3.12"
  echo "  Pruning Python install runtime..."
  rm -rf "$install_dir/include" "$install_dir/share"
  rm -rf "$install_dir/lib/pkgconfig"
  rm -rf "$install_dir/lib"/tcl* "$install_dir/lib"/tk* "$install_dir/lib"/itcl* "$install_dir/lib"/thread*
  rm -rf "$py_stdlib/ensurepip" "$py_stdlib/idlelib" "$py_stdlib/turtledemo" "$py_stdlib/tkinter"
  rm -rf "$py_stdlib/site-packages"/pip* "$py_stdlib/site-packages"/setuptools* "$py_stdlib/site-packages"/wheel*
  rm -f "$install_dir/bin"/pip "$install_dir/bin"/pip3 "$install_dir/bin"/pip3.* \
    "$install_dir/bin"/wheel "$install_dir/bin"/pyproject-build
  find "$install_dir" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  find "$install_dir" -name "*.pyc" -delete 2>/dev/null || true
}

sanitize_release_bundle() {
  local app_path="$1"
  local res_dir="$app_path/Contents/Resources"
  local licenses_dir="$res_dir/licenses"

  echo "  Sanitizing release bundle metadata and local build paths..."
  find "$res_dir/python_runtime" -type f -name "direct_url.json" -delete 2>/dev/null || true

  mkdir -p "$licenses_dir"
  cp -f "$ROOT_DIR/LICENSE" "$licenses_dir/LICENSE" 2>/dev/null || true
  cp -f "$ROOT_DIR/NOTICE" "$licenses_dir/NOTICE" 2>/dev/null || true
  cp -f "$ROOT_DIR/THIRD_PARTY_LICENSES.md" "$licenses_dir/THIRD_PARTY_LICENSES.md" 2>/dev/null || true
  cp -f "$ROOT_DIR/THIRD_PARTY_NOTICES.md" "$licenses_dir/THIRD_PARTY_NOTICES.md" 2>/dev/null || true
  cp -f "$ROOT_DIR/MODELS.md" "$licenses_dir/MODELS.md" 2>/dev/null || true
  cp -f "$ROOT_DIR/apps/desktop/public/fonts/OFL.txt" "$licenses_dir/OFL.txt" 2>/dev/null || true

  python3 - "$app_path" "$ROOT_DIR" "$HOME" "$(current_build_user)" <<'PY'
import os
import re
import sys
from pathlib import Path

app = Path(sys.argv[1])
repo = os.path.abspath(sys.argv[2]).encode()
home = os.path.abspath(sys.argv[3]).encode()
build_user = sys.argv[4].encode()
repo_name = os.path.basename(os.path.abspath(sys.argv[2])).encode()
home_name = os.path.basename(os.path.abspath(sys.argv[3])).encode()

def same_len(label: bytes, size: int) -> bytes:
    if len(label) > size:
        return label[:size]
    return label + (b"_" * (size - len(label)))

def can_patch_metadata(path: Path, data: bytes) -> bool:
    if b"\x00" in data[:4096]:
        return False
    name = path.name
    suffix = path.suffix.lower()
    if suffix in {
        ".cfg", ".css", ".csv", ".html", ".ini", ".js", ".json", ".map",
        ".md", ".plist", ".py", ".pyi", ".rst", ".svg", ".toml", ".tsv",
        ".txt", ".xml", ".yaml", ".yml",
    }:
        return True
    return name in {
        "INSTALLER", "LICENSE", "METADATA", "NOTICE", "OFL.txt", "PkgInfo",
        "RECORD", "REQUESTED", "SOURCES.txt", "WHEEL", "entry_points.txt",
        "top_level.txt",
    }

replacements = []
if repo:
    replacements.append((repo, same_len(b"/unfoldly/source", len(repo))))
if home:
    replacements.append((home, same_len(b"/home", len(home))))

changed = 0
for path in app.rglob("*"):
    if not path.is_file():
        continue
    try:
        data = path.read_bytes()
    except Exception:
        continue
    if not can_patch_metadata(path, data):
        continue
    new = data
    for old, repl in replacements:
        if old in new:
            new = new.replace(old, repl)
    new = re.sub(
        rb"/Users/[^\x00\"'\s<>]{1,4096}",
        lambda match: same_len(b"/build/path", len(match.group(0))),
        new,
    )
    if new != data:
        path.write_bytes(new)
        changed += 1

needles = []
for private_value in (repo, home, build_user, home_name, repo_name):
    if private_value and len(private_value) >= 3 and private_value not in {b"root", b"home"}:
        needles.append(private_value)
for raw in os.environ.get("UNFOLDLY_PRIVACY_EXTRA_NEEDLES", "").replace(",", "\n").splitlines():
    value = raw.strip().encode()
    if value:
        needles.append(value)
hits = []
for path in app.rglob("*"):
    if not path.is_file():
        continue
    try:
        data = path.read_bytes()
    except Exception:
        continue
    if any(needle in data for needle in needles):
        hits.append(str(path.relative_to(app)))
        if len(hits) >= 20:
            break

print(f"[package_tauri] Sanitized local build path strings in {changed} bundle files")
if hits:
    print("[package_tauri] ERROR: privacy-sensitive local strings remain in bundle:", file=sys.stderr)
    for hit in hits:
        print(f"  - {hit}", file=sys.stderr)
    raise SystemExit(1)
PY
}

strip_macho_debug_metadata() {
  local app_path="$1"

  if [[ "$(uname -s)" != "Darwin" ]]; then
    return 0
  fi

  echo "  Stripping Mach-O debug metadata before signing..."
  local stripped=0
  local skipped=0
  while IFS= read -r -d '' candidate; do
    local kind
    kind="$(file -b "$candidate" 2>/dev/null || true)"
    if [[ "$kind" != *"Mach-O"* ]]; then
      continue
    fi
    chmod u+w "$candidate" 2>/dev/null || true
    if strip -S -x "$candidate" >/dev/null 2>&1; then
      stripped=$((stripped + 1))
    else
      skipped=$((skipped + 1))
    fi
  done < <(find "$app_path" -type f \( -path "*/Contents/MacOS/*" -o -name "*.so" -o -name "*.dylib" \) -print0)

  echo "[package_tauri] Stripped Mach-O debug metadata: stripped=$stripped, skipped=$skipped"
}

sign_macho_binaries() {
  local app_path="$1"

  if [[ "$(uname -s)" != "Darwin" ]]; then
    return 0
  fi

  echo "[package_tauri] Signing nested Mach-O binaries..."
  local signed=0
  local skipped=0
  while IFS= read -r -d '' candidate; do
    local kind
    kind="$(file -b "$candidate" 2>/dev/null || true)"
    if [[ "$kind" != *"Mach-O"* ]]; then
      continue
    fi
    chmod u+w "$candidate" 2>/dev/null || true
    if codesign --force --sign - "$candidate" >/dev/null 2>&1; then
      signed=$((signed + 1))
    else
      skipped=$((skipped + 1))
    fi
  done < <(find "$app_path" -type f \( -path "*/Contents/MacOS/*" -o -name "*.so" -o -name "*.dylib" \) -print0)

  echo "[package_tauri] Signed nested Mach-O binaries: signed=$signed, skipped=$skipped"
  if [[ "$skipped" -gt 0 ]]; then
    echo "[package_tauri] ERROR: failed to sign one or more nested Mach-O binaries"
    exit 1
  fi
}

validate_ffmpeg_bundle() {
  local ffmpeg_dir="$1"
  local ffmpeg_bin="$ffmpeg_dir/bin/ffmpeg"
  local ffprobe_bin="$ffmpeg_dir/bin/ffprobe"
  local required_demuxers=(mov matroska avi mpegts mp3 wav flac ogg aac aiff)
  local required_decoders=(
    h264 hevc mpeg4 mjpeg vp8 vp9 av1
    aac mp3 flac vorbis opus alac
    pcm_s16le pcm_s24le pcm_s32le pcm_f32le
  )
  local demuxer
  local decoder
  if [[ ! -x "$ffmpeg_bin" || ! -x "$ffprobe_bin" ]]; then
    echo "[package_tauri] ERROR: bundled FFmpeg binaries are missing under $ffmpeg_dir"
    exit 1
  fi
  "$ffmpeg_bin" -hide_banner -buildconf 2>&1 \
    | sed -e "s#${ROOT_DIR}#<repo>#g" -e "s#${ffmpeg_dir}#<bundle-ffmpeg>#g" \
    > "$ffmpeg_dir/BUILD_CONFIG.txt" || true
  "$ffmpeg_bin" -hide_banner -demuxers > "$ffmpeg_dir/DEMUXERS.txt" 2>&1 || true
  "$ffmpeg_bin" -hide_banner -decoders > "$ffmpeg_dir/DECODERS.txt" 2>&1 || true
  "$ffmpeg_bin" -hide_banner -encoders > "$ffmpeg_dir/ENCODERS.txt" 2>&1 || true
  if grep -E -- '--enable-(gpl|nonfree)|libx264|libx265|libfdk_aac' "$ffmpeg_dir/BUILD_CONFIG.txt" >/dev/null; then
    echo "[package_tauri] ERROR: bundled FFmpeg build contains GPL/nonfree options"
    exit 1
  fi
  for demuxer in "${required_demuxers[@]}"; do
    if ! grep -Eq "[[:space:]]${demuxer}([[:space:],]|$)" "$ffmpeg_dir/DEMUXERS.txt"; then
      echo "[package_tauri] ERROR: bundled FFmpeg is missing $demuxer demuxer"
      exit 1
    fi
  done
  for decoder in "${required_decoders[@]}"; do
    if ! grep -Eq "[[:space:]]${decoder}[[:space:]]" "$ffmpeg_dir/DECODERS.txt"; then
      echo "[package_tauri] ERROR: bundled FFmpeg is missing $decoder decoder"
      exit 1
    fi
  done
  if grep -E "libx264|libx265| h264 | hevc " "$ffmpeg_dir/ENCODERS.txt" >/dev/null; then
    echo "[package_tauri] ERROR: bundled FFmpeg must not include H.264/H.265 encoders"
    exit 1
  fi
}

copy_bundled_ffmpeg() {
  local resources_dir="$1"
  local source_dir="$2"
  validate_ffmpeg_bundle "$source_dir"
  rm -rf "$resources_dir/ffmpeg"
  mkdir -p "$resources_dir/ffmpeg"
  cp -R "$source_dir/bin" "$resources_dir/ffmpeg/"
  cp -f "$source_dir/BUILD_CONFIG.txt" "$resources_dir/ffmpeg/" 2>/dev/null || true
  cp -f "$source_dir/DEMUXERS.txt" "$resources_dir/ffmpeg/" 2>/dev/null || true
  cp -f "$source_dir/DECODERS.txt" "$resources_dir/ffmpeg/" 2>/dev/null || true
  cp -f "$source_dir/ENCODERS.txt" "$resources_dir/ffmpeg/" 2>/dev/null || true
  cp -f "$source_dir/NOTICE.txt" "$resources_dir/ffmpeg/" 2>/dev/null || true
  chmod +x "$resources_dir/ffmpeg/bin/ffmpeg" "$resources_dir/ffmpeg/bin/ffprobe"
  echo "  Wrote bundled LGPL FFmpeg tools -> $resources_dir/ffmpeg"
  du -sh "$resources_dir/ffmpeg" 2>/dev/null || true
}

sync_runtime_site_for_reuse() {
  local site_dir="$1"
  local python_bin="$2"
  local root_dir="$3"

  if [[ ! -d "$site_dir" ]]; then
    echo "[package_tauri] ERROR: site directory not found: $site_dir"
    exit 1
  fi
  if [[ ! -x "$python_bin" ]]; then
    echo "[package_tauri] ERROR: Python executable not found: $python_bin"
    exit 1
  fi

  echo "[package_tauri] Reusing runtime: syncing latest backend code and model config into site..."
  rm -rf \
    "$site_dir/unfoldly_backend"* \
    "$site_dir/backend_bundle"* \
    "$site_dir/core"* \
    "$site_dir/services"* \
    "$site_dir/tools"* \
    "$site_dir/utils"* \
    "$site_dir/config"* \
    "$site_dir/api_server.py"* \
    "$site_dir/backend_core.py"*

  "$python_bin" -m pip install "$root_dir" --no-deps --target "$site_dir" --upgrade --force-reinstall
  "$python_bin" "$PATCH_LLAMA_CPP_LOADER" "$site_dir"
  prune_runtime_site_for_bundle "$site_dir"

  if [[ ! -f "$site_dir/config/supported_models.json" ]]; then
    echo "[package_tauri] ERROR: site/config/supported_models.json missing after sync"
    exit 1
  fi
  echo "[package_tauri] Reusing runtime: validating parsing dependencies..."
  PYTHONPATH="$site_dir" "$python_bin" - <<'PY'
import importlib
import shutil
import sys
import tempfile

required = [
    "pypdf", "pypdfium2", "docx2txt", "pptx", "lxml", "openpyxl", "xlrd", "numbers_parser",
    "rank_bm25", "jieba", "pypinyin",
]
missing = []
for module_name in required:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(f"{module_name}: {exc}")
if missing:
    print("[package_tauri] ERROR: reused python_runtime/site is missing parser dependencies:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(1)
from openpyxl import load_workbook  # noqa: F401,E402
from xlrd import open_workbook  # noqa: F401,E402
from numbers_parser import Document  # noqa: F401,E402
from rank_bm25 import BM25Okapi  # noqa: F401,E402
import jieba  # noqa: F401,E402
from pypinyin import lazy_pinyin  # noqa: F401,E402
from pptx import Presentation  # noqa: F401,E402

from pywhispercpp.model import Model as WhisperCppModel  # noqa: E402
whisper_info = WhisperCppModel.system_info()
if "Metal" not in whisper_info or "COREML = 0" not in whisper_info:
    raise SystemExit(
        "[package_tauri] ERROR: pywhispercpp must be Metal GPU enabled and CoreML disabled; "
        f"system_info={whisper_info!r}"
    )

import chromadb  # noqa: E402
tmp_dir = tempfile.mkdtemp(prefix="unfoldly_chroma_check_")
try:
    client = chromadb.PersistentClient(path=tmp_dir)
    collection = client.get_or_create_collection("runtime_check_collection")
    collection.add(ids=["1"], documents=["runtime check"], embeddings=[[0.1, 0.2, 0.3]])
    if collection.count() != 1:
        raise SystemExit("[package_tauri] ERROR: Chroma local smoke test failed")
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)
PY
  touch "$site_dir/.installed_success"
  echo "[package_tauri] Synced backend and config to: $site_dir"
}

if [[ "$SKIP_PYTHON_PREP" == "true" ]]; then
  sync_runtime_site_for_reuse "$PYTHON_RUNTIME_SITE" "$PYTHON" "$ROOT_DIR"
  echo ""
fi

if [[ "$BUNDLE_FFMPEG" == "true" ]]; then
  echo "[package_tauri] Preparing bundled LGPL FFmpeg decoder tools..."
  bash "$BUILD_FFMPEG_LGPL" "$TARGET_ARCH"
  echo ""
fi

cd "$FRONTEND_DIR"

if [[ ! -d "node_modules" ]]; then
  echo "[package_tauri] Installing npm dependencies..."
  npm install
fi

echo "[package_tauri] Starting Tauri release build..."
export MACOSX_DEPLOYMENT_TARGET="10.15"
export CARGO_INCREMENTAL=0
RUST_PATH_REMAP_FLAGS=(
  "--remap-path-prefix=${ROOT_DIR}=/unfoldly/source"
  "--remap-path-prefix=${HOME}=/home/builder"
)
if [[ -n "${CARGO_HOME:-}" ]]; then
  RUST_PATH_REMAP_FLAGS+=("--remap-path-prefix=${CARGO_HOME}=/cargo")
elif [[ -d "$HOME/.cargo" ]]; then
  RUST_PATH_REMAP_FLAGS+=("--remap-path-prefix=${HOME}/.cargo=/cargo")
fi
export RUSTFLAGS="${RUSTFLAGS:-} ${RUST_PATH_REMAP_FLAGS[*]}"

RUN_NPX="npx"
export CI=true # skip AppleScript Finder prompts
if [[ -n "$TARGET_ARCH" ]]; then
  if [[ "$TARGET_ARCH" == "arm64" ]]; then
    "$RUN_NPX" @tauri-apps/cli build --target aarch64-apple-darwin
  else
    "$RUN_NPX" @tauri-apps/cli build --target x86_64-apple-darwin
  fi
else
  "$RUN_NPX" @tauri-apps/cli build
fi

BUNDLE_BASE="$FRONTEND_DIR/src-tauri/target"
if [[ -n "$TARGET_ARCH" ]]; then
  if [[ "$TARGET_ARCH" == "arm64" ]]; then
    BUNDLE_DIR="$BUNDLE_BASE/aarch64-apple-darwin/release/bundle"
  else
    BUNDLE_DIR="$BUNDLE_BASE/x86_64-apple-darwin/release/bundle"
  fi
else
  BUNDLE_DIR="$BUNDLE_BASE/release/bundle"
fi

if [[ -d "$PYTHON_RUNTIME_INSTALL" ]] && [[ -d "$PYTHON_RUNTIME_SITE" ]]; then
  echo "[package_tauri] Embedding Python install and site into .app..."
  for app in "$BUNDLE_DIR"/macos/*.app "$BUNDLE_DIR"/*.app; do
    if [[ -d "$app" ]]; then
      RES="$app/Contents/Resources"
      MACOS_DIR="$app/Contents/MacOS"
      rm -rf "$RES/python_runtime"
      mkdir -p "$RES/python_runtime"
      
      echo "  Copying install directory..."
      cp -R "$PYTHON_RUNTIME_INSTALL" "$RES/python_runtime/"
      prune_copied_python_install "$RES/python_runtime/install"
      
      echo "  Copying site directory..."
      cp -R "$PYTHON_RUNTIME_SITE" "$RES/python_runtime/site"
      
      find "$RES/python_runtime/install" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
      find "$RES/python_runtime/install" -type f -name "*.pyc" -delete 2>/dev/null || true
      find "$RES/python_runtime/site" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
      find "$RES/python_runtime/site" -type f -name "*.pyc" -delete 2>/dev/null || true
      rm -rf "$RES/python_runtime/site/models" 2>/dev/null || true
      rm -rf "$RES/python_runtime/site/local_models" 2>/dev/null || true

      if [[ "$BUNDLE_FFMPEG" == "true" ]]; then
        copy_bundled_ffmpeg "$RES" "$MACOS_BUNDLE/ffmpeg/${TARGET_ARCH}"
      else
        rm -rf "$RES/ffmpeg" 2>/dev/null || true
        echo "  Bundled FFmpeg disabled"
      fi

      rm -f "$MACOS_DIR/crispasr"
      echo "  CrispASR excluded from bundle as expected"
      strip_macho_debug_metadata "$app"
      sanitize_release_bundle "$app"
      
      echo "  Wrote python_runtime: install + site -> $app"
    fi
  done
fi

RELEASE_DIR="$ROOT_DIR/macos_bundle/release"
mkdir -p "$RELEASE_DIR"

if [[ -d "$BUNDLE_DIR" ]]; then
  for app in "$BUNDLE_DIR"/macos/*.app "$BUNDLE_DIR"/*.app; do
    if [[ -d "$app" ]]; then
      echo "[package_tauri] Applying ad-hoc signature to .app: $app"
      sign_macho_binaries "$app"
      codesign --force --deep -s - "$app" || true
      
      APP_NAME=$(basename "$app")
      rm -rf "$RELEASE_DIR/$APP_NAME"
      cp -R "$app" "$RELEASE_DIR/"
      echo "  Copied .app to: $RELEASE_DIR/"

      FINAL_UPDATE_ARCHIVE="$RELEASE_DIR/$APP_NAME.tar.gz"
      rm -f "$FINAL_UPDATE_ARCHIVE" "$FINAL_UPDATE_ARCHIVE.sig"
      echo "[package_tauri] Generating final updater archive: $FINAL_UPDATE_ARCHIVE"
      (
        cd "$RELEASE_DIR"
        COPYFILE_DISABLE=1 tar -czf "$FINAL_UPDATE_ARCHIVE" "$APP_NAME"
      )
      if [[ -n "${TAURI_SIGNING_PRIVATE_KEY:-}" ]] || [[ -n "${TAURI_SIGNING_PRIVATE_KEY_PATH:-}" ]]; then
        "$RUN_NPX" @tauri-apps/cli signer sign "$FINAL_UPDATE_ARCHIVE"
      else
        echo "[package_tauri] WARN: updater archive created without .sig because no Tauri signing key env was provided."
      fi
      
      DMG_NAME="${APP_NAME%.app}.dmg"
      echo "[package_tauri] Generating DMG: $RELEASE_DIR/$DMG_NAME"
      
      rm -f "$RELEASE_DIR/$DMG_NAME"

      if [[ "$USE_CREATE_DMG" == "true" ]]; then
        DMG_LAYOUT_RESULT="create-dmg layout requested"
        echo "[package_tauri] UNFOLDLY_USE_CREATE_DMG=1; using create-dmg for drag-install DMG layout..."
        cd "$RELEASE_DIR"
        rm -f "${APP_NAME%.app}"*.dmg 2>/dev/null || true

        BG_PNG="$RELEASE_DIR/dmg_background_arrow.png"
        HAS_BG=false
        final_bg_source="$SCRIPT_DIR/../assets/dmg_background_final.png"
        if [[ -f "$final_bg_source" ]]; then
          cp -f "$final_bg_source" "$BG_PNG"
          HAS_BG=true
        else
          echo "[package_tauri] WARN: prebuilt background image not found ($final_bg_source); continuing without background layout."
          rm -f "$BG_PNG" 2>/dev/null || true
        fi

        detach_existing_volume_mounts "${APP_NAME%.app}"
        CREATE_DMG_ARGS=(
          --overwrite
          --window-size 760 430
          --icon-size 132
          --icon "$APP_NAME" 190 225
          --hide-extension "$APP_NAME"
          --app-drop-link 570 225
        )
        if [[ "$HAS_BG" == "true" ]]; then
          CREATE_DMG_ARGS+=(--background "$BG_PNG")
        fi

        npx create-dmg "${CREATE_DMG_ARGS[@]}" "$DMG_NAME" . || {
          echo "[package_tauri] create-dmg failed; falling back to plain hdiutil container..."
          DMG_LAYOUT_RESULT="warning: create-dmg failed; plain fallback DMG generated"
          DMG_STAGING_DIR="$RELEASE_DIR/dmg_staging_${APP_NAME%.app}"
          rm -rf "$DMG_STAGING_DIR"
          mkdir -p "$DMG_STAGING_DIR"
          cp -R "$app" "$DMG_STAGING_DIR/"
          ln -s /Applications "$DMG_STAGING_DIR/Applications"
          find "$DMG_STAGING_DIR" -name ".DS_Store" -delete 2>/dev/null || true
          if [[ "$DMG_FORMAT" == "UDZO" ]]; then
            hdiutil create -volname "${APP_NAME%.app}" -srcfolder "$DMG_STAGING_DIR" -ov -format UDZO -imagekey zlib-level=9 "$RELEASE_DIR/$DMG_NAME"
          else
            hdiutil create -volname "${APP_NAME%.app}" -srcfolder "$DMG_STAGING_DIR" -ov -format UDBZ "$RELEASE_DIR/$DMG_NAME"
          fi
          rm -rf "$DMG_STAGING_DIR"
        }

        if ls "$RELEASE_DIR/${APP_NAME%.app}"*.dmg 1> /dev/null 2>&1; then
          for generated_dmg in "$RELEASE_DIR/${APP_NAME%.app}"*.dmg; do
            if [ "$generated_dmg" != "$RELEASE_DIR/$DMG_NAME" ]; then
              mv "$generated_dmg" "$RELEASE_DIR/$DMG_NAME"
              break
            fi
          done
        fi
        rm -f "$BG_PNG" 2>/dev/null || true
      elif [[ "$DMG_LAYOUT" == "true" ]]; then
        DMG_LAYOUT_RESULT="styled layout requested"
        echo "[package_tauri] UNFOLDLY_DMG_LAYOUT=1; using hdiutil static drag-install layout without create-dmg."
        echo "[package_tauri] DMG compression format: ${DMG_FORMAT}"
        if ! build_layout_dmg_via_hdiutil "$app" "$RELEASE_DIR/$DMG_NAME" "${APP_NAME%.app}" "$APP_NAME"; then
          echo "[package_tauri] WARN: hdiutil layout mode failed; falling back to plain DMG container."
          DMG_LAYOUT_RESULT="warning: hdiutil layout failed; plain fallback DMG generated"
          build_pure_container_dmg "$app" "$RELEASE_DIR/$DMG_NAME" "${APP_NAME%.app}"
        fi
      else
        DMG_LAYOUT_RESULT="plain DMG container requested"
        echo "[package_tauri] Using plain DMG container by default (.app + Applications only)."
        echo "[package_tauri] DMG compression format: ${DMG_FORMAT}"
        build_pure_container_dmg "$app" "$RELEASE_DIR/$DMG_NAME" "${APP_NAME%.app}"
      fi

      MAX_DMG_MB="${UNFOLDLY_MAX_DMG_MB:-165}"
      if [[ -f "$RELEASE_DIR/$DMG_NAME" ]]; then
        COMPACT_DMG="$RELEASE_DIR/${DMG_NAME%.dmg}.compact.dmg"
        if hdiutil convert "$RELEASE_DIR/$DMG_NAME" -ov -format "$DMG_FORMAT" -o "$COMPACT_DMG"; then
          mv "$COMPACT_DMG" "$RELEASE_DIR/$DMG_NAME"
          echo "  Compacted DMG at: $RELEASE_DIR/$DMG_NAME"
        else
          rm -f "$COMPACT_DMG" 2>/dev/null || true
          echo "[package_tauri] WARN: DMG compaction failed; keeping original DMG."
        fi

        DMG_BYTES="$(stat -f%z "$RELEASE_DIR/$DMG_NAME" 2>/dev/null || echo 0)"
        DMG_MB="$(( (DMG_BYTES + 1048576 - 1) / 1048576 ))"
        echo "  DMG size: ${DMG_MB}MB (limit: ${MAX_DMG_MB}MB)"
        if [[ "$MAX_DMG_MB" == "0" ]]; then
          echo "[package_tauri] Size guard disabled (UNFOLDLY_MAX_DMG_MB=0); skipping size check."
        elif [[ "$DMG_MB" -gt "$MAX_DMG_MB" ]]; then
          echo "[package_tauri] ERROR: DMG size exceeds limit (${DMG_MB}MB > ${MAX_DMG_MB}MB)"
          echo "[package_tauri] Tip: clean caches and rebuild:"
          echo "  rm -rf \"$ROOT_DIR/macos_bundle/python_runtime/site\" \"$ROOT_DIR/macos_bundle/release/Unfoldly.app\""
          echo "  UNFOLDLY_FORCE_REBUILD_SITE=1 bash macos_bundle/scripts/package_tauri.sh ${TARGET_ARCH:-}"
          echo "  Or for layout verification only: UNFOLDLY_MAX_DMG_MB=0 bash macos_bundle/scripts/package_tauri.sh ${TARGET_ARCH:-}"
          exit 1
        fi
      fi
      echo "  Generated DMG at: $RELEASE_DIR/$DMG_NAME"
    fi
  done
  
  open "$RELEASE_DIR" || true
else
  echo "  Bundle directory not found"
fi
echo ""
echo "============================================"
echo "  Tauri packaging complete"
echo "  DMG layout: ${DMG_LAYOUT_RESULT}"
echo "============================================"
