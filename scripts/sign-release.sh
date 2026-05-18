#!/usr/bin/env bash
# Build, sign, notarize, and staple a distributable Unfoldly release for macOS.
# Wraps scripts/build-release.sh with all environment required for Developer ID
# signing + Apple notarization, plus preflight checks that surface missing
# prerequisites with actionable error messages.
#
# Usage:
#   bash scripts/sign-release.sh              # auto-detect host arch
#   bash scripts/sign-release.sh arm64        # explicit arch
#   bash scripts/sign-release.sh x64
#
# Optional overrides (rarely needed):
#   UNFOLDLY_SIGNING_IDENTITY    Override Developer ID auto-detection
#   UNFOLDLY_NOTARY_PROFILE      Default: unfoldly-notary
#   TAURI_UPDATER_KEY_FILE       Default: signing/updater.key
#   TAURI_UPDATER_KEY_PASSWORD   Default: empty (matches a passwordless key)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ARCH="${1:-}"
if [[ -z "$ARCH" ]]; then
  case "$(uname -m)" in
    arm64)  ARCH=arm64 ;;
    x86_64) ARCH=x64 ;;
    *) echo "[sign-release] ERROR: unsupported arch $(uname -m)" >&2; exit 1 ;;
  esac
fi
case "$ARCH" in arm64|x64) ;; *)
  echo "[sign-release] ERROR: arch must be arm64 or x64, got: $ARCH" >&2; exit 1 ;;
esac

NOTARY_PROFILE="${UNFOLDLY_NOTARY_PROFILE:-unfoldly-notary}"
UPDATER_KEY="${TAURI_UPDATER_KEY_FILE:-$ROOT_DIR/signing/updater.key}"
UPDATER_KEY_PASSWORD="${TAURI_UPDATER_KEY_PASSWORD-}"

echo "[sign-release] ============================================"
echo "[sign-release] Building Unfoldly signed + notarized release"
echo "[sign-release] arch=$ARCH"
echo "[sign-release] root=$ROOT_DIR"
echo "[sign-release] ============================================"

# 1. Rust toolchain (cargo)
if ! command -v cargo >/dev/null 2>&1 && [[ -f "$HOME/.cargo/env" ]]; then
  . "$HOME/.cargo/env"
fi
if ! command -v cargo >/dev/null 2>&1; then
  cat >&2 <<'EOF'
[sign-release] ERROR: cargo not found. Install Rust:
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
EOF
  exit 1
fi
echo "[sign-release] cargo: $(cargo --version)"

# 2. Developer ID Application certificate
if [[ -n "${UNFOLDLY_SIGNING_IDENTITY:-}" ]]; then
  IDENTITY="$UNFOLDLY_SIGNING_IDENTITY"
else
  IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
              | awk -F'"' '/Developer ID Application/ {print $2; exit}')"
fi
if [[ -z "$IDENTITY" ]]; then
  cat >&2 <<'EOF'
[sign-release] ERROR: no "Developer ID Application" certificate in login keychain.
  Create one at https://developer.apple.com/account/resources/certificates/add
  (choose "Developer ID Application" with G2 Sub-CA), then double-click the
  downloaded .cer to install. Make sure the matching private key is also in
  the login keychain (use a CSR generated on this Mac, or import a .p12).
EOF
  exit 1
fi
echo "[sign-release] signing identity: $IDENTITY"

# 3. Notarization keychain profile
if ! xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" \
        --output-format json >/dev/null 2>&1; then
  cat >&2 <<EOF
[sign-release] ERROR: notarytool keychain profile '$NOTARY_PROFILE' is missing
  or invalid. Create one with:
    xcrun notarytool store-credentials "$NOTARY_PROFILE" \\
      --apple-id <your-apple-id-email> \\
      --team-id <your-team-id> \\
      --password <app-specific-password from appleid.apple.com>
EOF
  exit 1
fi
echo "[sign-release] notary profile: $NOTARY_PROFILE (validated)"

# 4. Tauri updater minisign key (must match plugins.updater.pubkey in tauri.conf.json)
if [[ ! -f "$UPDATER_KEY" ]]; then
  cat >&2 <<EOF
[sign-release] ERROR: Tauri updater key not found at: $UPDATER_KEY
  Generate a fresh keypair with:
    npx --yes @tauri-apps/cli signer generate --ci -p '' \\
      -w "$ROOT_DIR/signing/updater.key"
  Then paste the contents of signing/updater.key.pub into
  apps/desktop/src-tauri/tauri.conf.json -> plugins.updater.pubkey.
EOF
  exit 1
fi
echo "[sign-release] updater key: $UPDATER_KEY"

# 5. Export env and dispatch to the existing release flow
export PATH="$HOME/.cargo/bin:$PATH"
export UNFOLDLY_SIGNING_IDENTITY="$IDENTITY"
export UNFOLDLY_NOTARY_PROFILE="$NOTARY_PROFILE"
export TAURI_SIGNING_PRIVATE_KEY="$(cat "$UPDATER_KEY")"
export TAURI_SIGNING_PRIVATE_KEY_PASSWORD="$UPDATER_KEY_PASSWORD"

echo ""
echo "[sign-release] Handing off to scripts/build-release.sh $ARCH"
echo "[sign-release] First clean build is slow (~30-60 min):"
echo "  bundled Python prep -> LGPL FFmpeg build -> llama-cpp-python compile"
echo "  -> Rust release build -> codesign nested binaries -> notarize .app"
echo "  -> notarize DMG -> staple -> spctl assess."
echo ""

bash "$ROOT_DIR/scripts/build-release.sh" "$ARCH"

# 6. Post-build independent verification
APP="$ROOT_DIR/macos_bundle/release/Unfoldly.app"
DMG="$ROOT_DIR/macos_bundle/release/Unfoldly.dmg"

echo ""
echo "[sign-release] ============================================"
echo "[sign-release] Artifacts:"
ls -lh "$APP" "$DMG" \
       "$ROOT_DIR/macos_bundle/release/Unfoldly.app.tar.gz" \
       "$ROOT_DIR/macos_bundle/release/Unfoldly.app.tar.gz.sig" \
       2>/dev/null | awk '{printf "  %s\n", $0}'

echo ""
echo "[sign-release] Independent verification:"
spctl_app="$(spctl --assess --type execute --verbose=4 "$APP" 2>&1 || true)"
spctl_dmg="$(spctl --assess --type open --context context:primary-signature --verbose=4 "$DMG" 2>&1 || true)"
staple_app="$(xcrun stapler validate "$APP" 2>&1 | tail -1)"
staple_dmg="$(xcrun stapler validate "$DMG" 2>&1 | tail -1)"
sha="$(shasum -a 256 "$DMG" | awk '{print $1}')"

echo "  spctl  .app : $spctl_app"
echo "  spctl  .dmg : $spctl_dmg"
echo "  staple .app : $staple_app"
echo "  staple .dmg : $staple_dmg"
echo "  sha256 .dmg : $sha"

if echo "$spctl_dmg" | grep -q "Notarized Developer ID"; then
  echo "[sign-release] OK - DMG accepted by Gatekeeper as Notarized Developer ID"
else
  echo "[sign-release] WARN - DMG not flagged as Notarized; review the spctl line above"
fi
echo "[sign-release] ============================================"
