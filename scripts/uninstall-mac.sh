#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Uninstall Unfoldly for macOS.

Usage:
  bash scripts/uninstall-mac.sh
  bash scripts/uninstall-mac.sh --show

Options:
  --show, --dry-run   Show all paths that would be removed, but do not delete.
  -h, --help          Show this help.

This script permanently removes Unfoldly application data, including local
indexes, downloaded models, preferences, chat history, logs, and caches.
Your original source files selected for indexing are not deleted.
EOF
}

SHOW_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --show|--dry-run)
      SHOW_ONLY=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[uninstall] ERROR: unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

paths=(
  "/Applications/Unfoldly.app"
  "/Applications/unfoldly.app"
  "$HOME/Applications/Unfoldly.app"
  "$HOME/Library/Application Support/Unfoldly"
  "$HOME/Library/Caches/Unfoldly"
  "$HOME/Library/Logs/Unfoldly"
  "$HOME/Library/Preferences/Unfoldly.plist"
  "$HOME/Library/Saved Application State/Unfoldly.savedState"
)

crash_report_dir="$HOME/Library/Logs/DiagnosticReports"
crash_report_patterns=(
  "$crash_report_dir/Unfoldly_*"
  "$crash_report_dir/unfoldly_*"
)

print_paths() {
  echo "The following Unfoldly paths will be removed if they exist:"
  echo ""
  for path in "${paths[@]}"; do
    echo "  - $path"
  done
  for pattern in "${crash_report_patterns[@]}"; do
    echo "  - $pattern"
  done
}

echo "=========================================="
echo "Uninstall Unfoldly"
echo "=========================================="
echo ""
echo "This is a destructive uninstall."
echo ""
echo "It removes:"
echo "  - Unfoldly.app"
echo "  - Local ChromaDB indexes and file metadata"
echo "  - Downloaded local models"
echo "  - Preferences and selected source records"
echo "  - Chat history"
echo "  - Logs, caches, and crash reports"
echo ""
echo "It does NOT delete the original files you selected for indexing."
echo ""
print_paths
echo ""

if [[ "$SHOW_ONLY" == "1" ]]; then
  echo "[uninstall] Show-only mode. No files were deleted."
  exit 0
fi

echo "Type 'yes' and press Enter to permanently delete these paths."
read -r reply

if [[ "$reply" != "yes" ]]; then
  echo "[uninstall] Cancelled. No files were deleted."
  exit 0
fi

echo ""
echo "[uninstall] Stopping Unfoldly if it is running..."
pkill -f "/Unfoldly.app/Contents/MacOS/unfoldly" 2>/dev/null || true
pkill -x "Unfoldly" 2>/dev/null || true
pkill -x "unfoldly" 2>/dev/null || true
sleep 1

remove_path() {
  local path="$1"
  if [[ -e "$path" || -L "$path" ]]; then
    echo "[uninstall] Removing: $path"
    rm -rf -- "$path"
  else
    echo "[uninstall] Not found: $path"
  fi
}

for path in "${paths[@]}"; do
  remove_path "$path"
done

if [[ -d "$crash_report_dir" ]]; then
  while IFS= read -r -d '' path; do
    remove_path "$path"
  done < <(
    find "$crash_report_dir" -maxdepth 1 -type f \
      \( -name 'Unfoldly_*' -o -name 'unfoldly_*' \) -print0
  )
fi

echo ""
echo "=========================================="
echo "Unfoldly uninstall complete."
echo "=========================================="
