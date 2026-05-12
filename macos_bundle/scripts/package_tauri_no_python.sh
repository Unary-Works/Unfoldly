#!/bin/bash
#
#   bash macos_bundle/scripts/package_tauri_no_python.sh [arm64|x64]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export UNFOLDLY_SKIP_PYTHON_PREP=1
export UNFOLDLY_FORCE_REBUILD_SITE=0
export UNFOLDLY_USE_CREATE_DMG=0
export UNFOLDLY_DMG_LAYOUT="${UNFOLDLY_DMG_LAYOUT:-1}"
export UNFOLDLY_DMG_FORMAT="${UNFOLDLY_DMG_FORMAT:-UDBZ}"

PASS_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --layout)
      export UNFOLDLY_DMG_LAYOUT=1
      export UNFOLDLY_USE_CREATE_DMG=0
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
    *)
      PASS_ARGS+=("$arg")
      ;;
  esac
done

exec bash "$SCRIPT_DIR/package_tauri.sh" "${PASS_ARGS[@]}"
