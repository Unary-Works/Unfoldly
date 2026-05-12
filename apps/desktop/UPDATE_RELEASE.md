# App Updater Release Flow

Unfoldly uses Tauri updater with GitHub Releases from `Unary-Works/Unfoldly`.

## One-time setup

The updater public key is committed in `src-tauri/tauri.conf.json`.

Keep the Tauri updater private signing key and password out of git. Release
operators should provide them through local environment variables when building.

## Release a new version

1. Sync app versions:

   ```bash
   cd apps/desktop
   npm run version:sync -- 1.0.0
   npm install --package-lock-only
   ```

2. Build signed updater artifacts from the repository root:

   ```bash
   export TAURI_SIGNING_PRIVATE_KEY="$(cat path/to/updater-private.key)"
   export TAURI_SIGNING_PRIVATE_KEY_PASSWORD="$(cat path/to/updater-private.password)"
   cd ../..
   bash scripts/build-release.sh arm64
   ```

   This clean release wrapper creates the signed updater archive, the optional
   manual-install DMG, verifies signatures and checksums, and scans the app and
   mounted DMG for local private paths and test data.

3. Generate `latest.json`:

   ```bash
   cd apps/desktop
   npm run updater:manifest -- --repo Unary-Works/Unfoldly --tag v1.0.0 --notes "Release v1.0.0"
   ```

4. In GitHub release `v1.0.0`, upload these updater files from `macos_bundle/release/`:

   - `Unfoldly.app.tar.gz`
   - `Unfoldly.app.tar.gz.sig`
   - `latest.json`

`Unfoldly.dmg` is optional. It is only for manual download and install, not required by in-app updates.

## Version rules

- App internal version: `1.0.0`
- Git tag / GitHub release tag: `v1.0.0`
- `latest.json.version`: `1.0.0`

Do not put the `v` prefix into `package.json`, `Cargo.toml`, or `tauri.conf.json`.
