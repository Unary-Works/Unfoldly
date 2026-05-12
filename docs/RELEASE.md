# Release Publishing

[Chinese version](zh/RELEASE.md)

This guide describes how to build a macOS release package, upload it to GitHub
Releases, and make the Tauri in-app updater discover it.

## 1. Bump the App Version

Before building a release, update the version consistently:

```bash
cd apps/desktop
npm run version:sync -- 1.0.0
npm install --package-lock-only
cd ../..
```

Use a SemVer version such as `1.0.0`, and create the Git tag as `v1.0.0`.
Do not include the `v` prefix in `package.json`, `Cargo.toml`, or
`tauri.conf.json`.

## 2. Build the Release Package

From the repository root:

```bash
export TAURI_SIGNING_PRIVATE_KEY="$(cat path/to/updater-private.key)"
export TAURI_SIGNING_PRIVATE_KEY_PASSWORD="$(cat path/to/updater-private.password)"
bash scripts/build-release.sh arm64
```

This uses the clean release wrapper and creates the highest-compression DMG path used by this project:

```text
macos_bundle/release/Unfoldly.app
macos_bundle/release/Unfoldly.app.tar.gz
macos_bundle/release/Unfoldly.app.tar.gz.sig
macos_bundle/release/Unfoldly.dmg
```

The script verifies the app signature, verifies the DMG checksum, and scans both the app and mounted DMG for local private paths and test data.

For a completely fresh dependency/runtime build:

```bash
rm -rf macos_bundle/python_runtime macos_bundle/release
bash scripts/build-release.sh arm64
```

## 3. Generate the Updater Manifest

From `apps/desktop`:

```bash
npm run updater:manifest -- --repo Unary-Works/Unfoldly --tag v1.0.0 --notes "Release v1.0.0"
```

This writes:

```text
macos_bundle/release/latest.json
```

## 4. Create the GitHub Release

Push the release commit and tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Then create a GitHub release from the tag.

Required in-app updater assets:

```text
macos_bundle/release/Unfoldly.app.tar.gz
macos_bundle/release/Unfoldly.app.tar.gz.sig
macos_bundle/release/latest.json
```

Optional manual install asset:

```text
macos_bundle/release/Unfoldly.dmg
```

If you use the GitHub web UI:

1. Open the repository on GitHub.
2. Go to **Releases**.
3. Click **Draft a new release**.
4. Select or create tag `v1.0.0`.
5. Upload `Unfoldly.app.tar.gz`, `Unfoldly.app.tar.gz.sig`, and `latest.json`.
6. Optionally upload `Unfoldly.dmg` for manual installation.
7. Publish the release.

If you use GitHub CLI:

```bash
gh release create v1.0.0 \
  macos_bundle/release/Unfoldly.app.tar.gz \
  macos_bundle/release/Unfoldly.app.tar.gz.sig \
  macos_bundle/release/latest.json \
  macos_bundle/release/Unfoldly.dmg \
  --repo Unary-Works/Unfoldly \
  --title "Unfoldly v1.0.0" \
  --notes "Release notes here"
```

## 5. How the App Checks for Updates

The Settings > Updates page checks:

```text
https://github.com/Unary-Works/Unfoldly/releases/latest/download/latest.json
```

It compares:

- current app version from the packaged Tauri app;
- `latest.json.version`, such as `1.0.0`.

If `latest.json.version` is newer, the app downloads the signed
`Unfoldly.app.tar.gz` package, verifies it with the committed updater public
key, installs it, and prompts the user to restart.

Important constraints:

- The GitHub repository and release assets must be publicly accessible for end users.
- Draft releases are not visible to the app.
- The `latest.json` file must reference the release asset URL for the same tag.
- Do not upload local test data, source directories, or unsigned modified app bundles as release assets.
