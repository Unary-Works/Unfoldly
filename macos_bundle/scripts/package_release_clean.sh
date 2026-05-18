#!/usr/bin/env bash
set -euo pipefail

# Build a distributable macOS package and fail if the resulting .app/.dmg
# contains local build paths, private cache paths, test reports, or user data.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PACKAGE_SCRIPT="$SCRIPT_DIR/package_tauri.sh"

current_build_user() {
  local value="${USER:-${LOGNAME:-}}"
  if [[ -z "$value" ]]; then
    value="$(id -un 2>/dev/null || true)"
  fi
  printf '%s' "$value"
}

ARCH="${1:-}"
if [[ -z "$ARCH" ]]; then
  case "$(uname -m)" in
    arm64) ARCH="arm64" ;;
    x86_64) ARCH="x64" ;;
    *) echo "[package_release_clean] ERROR: unsupported architecture: $(uname -m)" >&2; exit 1 ;;
  esac
fi

case "$ARCH" in
  arm64|x64) ;;
  *) echo "[package_release_clean] ERROR: expected arch arm64 or x64, got: $ARCH" >&2; exit 1 ;;
esac

APP="$ROOT_DIR/macos_bundle/release/Unfoldly.app"
DMG="$ROOT_DIR/macos_bundle/release/Unfoldly.dmg"
PY_RUNTIME="$ROOT_DIR/macos_bundle/python_runtime/install/bin/python3"
PY_SITE="$ROOT_DIR/macos_bundle/python_runtime/site"
DMG_LAYOUT_STATUS="not checked"
RELEASE_CACHE="${UNFOLDLY_RELEASE_CACHE:-/tmp/unfoldly-release-cache-${ARCH}}"
export UNFOLDLY_PYTHON_CACHE="${UNFOLDLY_PYTHON_CACHE:-$RELEASE_CACHE/python-cache}"
export UNFOLDLY_FFMPEG_CACHE="${UNFOLDLY_FFMPEG_CACHE:-$RELEASE_CACHE/ffmpeg-cache}"

runtime_has_private_markers() {
  local runtime_root="$1"
  [[ -d "$runtime_root" ]] || return 1
  python3 - "$runtime_root" "$ROOT_DIR" "$HOME" "$(current_build_user)" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
repo = os.path.abspath(sys.argv[2]).encode()
home = os.path.abspath(sys.argv[3]).encode()
build_user = sys.argv[4].encode()
repo_name = os.path.basename(os.path.abspath(sys.argv[2])).encode()
home_name = os.path.basename(os.path.abspath(sys.argv[3])).encode()
needles = [
    value
    for value in (repo, home, build_user, home_name)
    if value and len(value) >= 3 and value not in {b"root", b"home"}
]
# repo_name omitted by default for product-name == repo-name projects; opt in via env.
if os.environ.get("UNFOLDLY_PRIVACY_STRICT_REPO_NAME") == "1":
    if repo_name and len(repo_name) >= 3 and repo_name not in {b"root", b"home"}:
        needles.append(repo_name)
for path in root.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix == ".pyc" or path.name == "direct_url.json":
        continue
    if path.parent.name == "bin" and path.name in {
        "pip", "pip3", "pip3.12", "wheel", "pyproject-build"
    }:
        continue
    try:
        data = path.read_bytes()
    except Exception:
        continue
    if any(needle in data for needle in needles):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

if [[ -x "$PY_RUNTIME" && -d "$PY_SITE" && "${UNFOLDLY_FORCE_REBUILD_SITE:-0}" != "1" ]]; then
  if runtime_has_private_markers "$ROOT_DIR/macos_bundle/python_runtime"; then
    echo "[package_release_clean] Existing python_runtime contains private build markers; rebuilding runtime site."
    export UNFOLDLY_FORCE_REBUILD_SITE=1
    unset UNFOLDLY_SKIP_PYTHON_PREP
  else
    export UNFOLDLY_SKIP_PYTHON_PREP="${UNFOLDLY_SKIP_PYTHON_PREP:-1}"
  fi
fi

if [[ -x "$PY_RUNTIME" && -d "$PY_SITE" && "${UNFOLDLY_FORCE_REBUILD_SITE:-0}" != "1" ]]; then
  export UNFOLDLY_SKIP_PYTHON_PREP="${UNFOLDLY_SKIP_PYTHON_PREP:-1}"
fi

export UNFOLDLY_MAX_DMG_MB="${UNFOLDLY_MAX_DMG_MB:-165}"

echo "[package_release_clean] Building clean release package for $ARCH"
echo "[package_release_clean] ROOT_DIR=$ROOT_DIR"
echo "[package_release_clean] UNFOLDLY_SKIP_PYTHON_PREP=${UNFOLDLY_SKIP_PYTHON_PREP:-0}"
echo "[package_release_clean] UNFOLDLY_MAX_DMG_MB=$UNFOLDLY_MAX_DMG_MB"
echo "[package_release_clean] UNFOLDLY_PYTHON_CACHE=$UNFOLDLY_PYTHON_CACHE"
echo "[package_release_clean] UNFOLDLY_FFMPEG_CACHE=$UNFOLDLY_FFMPEG_CACHE"

bash "$PACKAGE_SCRIPT" --layout --udbz "$ARCH"

if [[ ! -d "$APP" ]]; then
  echo "[package_release_clean] ERROR: missing app bundle: $APP" >&2
  exit 1
fi
if [[ ! -f "$DMG" ]]; then
  echo "[package_release_clean] ERROR: missing DMG: $DMG" >&2
  exit 1
fi

echo "[package_release_clean] Verifying app signature..."
codesign --verify --deep --strict --verbose=2 "$APP"

echo "[package_release_clean] Verifying DMG checksum..."
hdiutil verify "$DMG"

privacy_scan() {
  local root="$1"
  local label="$2"
  python3 - "$root" "$label" "$ROOT_DIR" "$HOME" "$(current_build_user)" "${UNFOLDLY_PRIVACY_EXTRA_NEEDLES:-}" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
label = sys.argv[2]
repo = os.path.abspath(sys.argv[3]).encode()
home = os.path.abspath(sys.argv[4]).encode()
build_user = sys.argv[5].encode()
extra_needles = sys.argv[6]
repo_name = os.path.basename(os.path.abspath(sys.argv[3])).encode()
home_name = os.path.basename(os.path.abspath(sys.argv[4])).encode()

needles = []
for private_value in (repo, home, build_user, home_name):
    if private_value and len(private_value) >= 3 and private_value not in {b"root", b"home"}:
        needles.append(private_value)
# repo_name omitted by default; opt in via UNFOLDLY_PRIVACY_STRICT_REPO_NAME=1.
if os.environ.get("UNFOLDLY_PRIVACY_STRICT_REPO_NAME") == "1":
    if repo_name and len(repo_name) >= 3 and repo_name not in {b"root", b"home"}:
        needles.append(repo_name)
for raw in extra_needles.replace(",", "\n").splitlines():
    value = raw.strip().encode()
    if value:
        needles.append(value)

hits = []
for path in root.rglob("*"):
    if not path.is_file():
        continue
    try:
        data = path.read_bytes()
    except Exception:
        continue
    for needle in needles:
        if needle in data:
            hits.append((str(path.relative_to(root)), needle.decode("utf-8", "ignore")))
            break
    if len(hits) >= 80:
        break

if hits:
    print(f"[package_release_clean] ERROR: privacy scan failed for {label}", file=sys.stderr)
    for rel, needle in hits:
        print(f"  - {rel}: {needle}", file=sys.stderr)
    raise SystemExit(1)

print(f"[package_release_clean] {label} privacy scan OK")
PY
}

privacy_scan "$APP" "app"

MOUNT_POINT="$(mktemp -d /tmp/unfoldly_release_dmg.XXXXXX)"
cleanup() {
  hdiutil detach "$MOUNT_POINT" -force >/dev/null 2>&1 || true
  rmdir "$MOUNT_POINT" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[package_release_clean] Mounting DMG for privacy scan..."
hdiutil attach "$DMG" -readonly -nobrowse -mountpoint "$MOUNT_POINT" >/dev/null
privacy_scan "$MOUNT_POINT" "dmg"
if [[ -f "$MOUNT_POINT/.DS_Store" && -f "$MOUNT_POINT/.background/dmg_background_arrow.png" ]]; then
  DMG_LAYOUT_STATUS="styled Finder drag-install layout detected"
else
  DMG_LAYOUT_STATUS="WARNING: styled DMG layout metadata not detected; open the DMG before publishing"
fi

DIRECT_URL_COUNT="$(find "$APP/Contents/Resources/python_runtime" -type f -name direct_url.json 2>/dev/null | wc -l | tr -d ' ')"
PIP_SCRIPT_COUNT="$(find "$APP/Contents/Resources/python_runtime/install/bin" -maxdepth 1 \( -name pip -o -name 'pip3*' -o -name wheel -o -name pyproject-build \) 2>/dev/null | wc -l | tr -d ' ')"
PROJECT_TEST_COUNT="$(find "$APP/Contents/Resources/python_runtime/site" -maxdepth 2 \( -path '*/test' -o -path '*/tests' -o -path '*/reports' \) 2>/dev/null | wc -l | tr -d ' ')"

if [[ "$DIRECT_URL_COUNT" != "0" ]]; then
  echo "[package_release_clean] ERROR: direct_url.json files remain in bundle" >&2
  exit 1
fi
if [[ "$PIP_SCRIPT_COUNT" != "0" ]]; then
  echo "[package_release_clean] ERROR: pip/build console scripts remain in bundle" >&2
  exit 1
fi
if [[ "$PROJECT_TEST_COUNT" != "0" ]]; then
  echo "[package_release_clean] ERROR: project test/report directories remain in runtime site" >&2
  exit 1
fi

SHA256="$(shasum -a 256 "$DMG" | awk '{print $1}')"
APP_SIZE="$(du -sh "$APP" | awk '{print $1}')"
DMG_SIZE="$(du -sh "$DMG" | awk '{print $1}')"

echo "[package_release_clean] Clean package ready"
echo "[package_release_clean] APP=$APP ($APP_SIZE)"
echo "[package_release_clean] DMG=$DMG ($DMG_SIZE)"
echo "[package_release_clean] DMG_LAYOUT=$DMG_LAYOUT_STATUS"
echo "[package_release_clean] SHA256=$SHA256"
