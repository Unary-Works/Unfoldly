# Unfoldly macOS Build Guide

[Chinese version](zh/BUILD_MACOS.md)

---

This directory contains scripts and assets for packaging the Unfoldly macOS application into a standalone `.app` and `.dmg`. The packaging process bundles a clean Python runtime and backend dependencies into the Tauri application. User models and databases remain runtime data and should be placed under `FILEAGENT_DATA_DIR`.

## Prerequisites

Before building the application, ensure your macOS machine has the underlying build toolchains. **You do not need to manually create virtual environments or install Python dependencies** — the build script fully automates `.venv` creation, requirements installation, and from-source compilation of `llama-cpp-python` (with Metal acceleration) out of the box.

1. **Python 3.12** is recommended. The script will automatically create a `.venv` at your project root using your system `python3`.
2. **Rust Toolkit**: Required for Tauri. Install via [rustup.rs](https://rustup.rs/).
3. **Node.js / npm**: Required for frontend builds. Install via [nodejs.org](https://nodejs.org/) or `brew install node`.
4. **Git & CMake**: Required for automatically compiling `llama-cpp-python` from source during the build process.
5. **Disk Space**: Ensure at least **10GB** of free disk space for standalone Python extraction and LLM artifacts.

---

## Build Command Matrix

Run all commands from the repository root. Use `scripts/build-release.sh` for public release builds. The lower-level scripts under `macos_bundle/scripts` are maintained for packaging internals and advanced incremental workflows.

| Workflow | Command |
|----------|---------|
| Clean release build, Apple Silicon | `bash scripts/build-release.sh arm64` |
| Clean release build, Intel | `bash scripts/build-release.sh x64` |
| Clean release build, auto-detect architecture | `bash scripts/build-release.sh` |
| Lower-level full build, Apple Silicon | `bash macos_bundle/scripts/package_tauri.sh arm64` |
| Lower-level full build, Intel | `bash macos_bundle/scripts/package_tauri.sh x64` |
| Force a fresh runtime-site rebuild | `rm -rf macos_bundle/python_runtime/site macos_bundle/release && UNFOLDLY_FORCE_REBUILD_SITE=1 bash macos_bundle/scripts/package_tauri.sh arm64` |
| Fast incremental build after source-only changes | `bash macos_bundle/scripts/package_tauri_no_python.sh arm64` |
| Equivalent incremental build through the main script | `UNFOLDLY_SKIP_PYTHON_PREP=1 bash macos_bundle/scripts/package_tauri.sh arm64` |
| Smallest high-compression layout DMG after runtime exists | `UNFOLDLY_SKIP_PYTHON_PREP=1 UNFOLDLY_MAX_DMG_MB=0 bash macos_bundle/scripts/package_tauri.sh --layout --udbz arm64` |
| Styled drag-install DMG with default size guard | `bash macos_bundle/scripts/package_tauri_no_python.sh --layout --udbz arm64` |
| Layout test with size guard disabled | `bash macos_bundle/scripts/package_tauri_no_python.sh --layout --udbz --no-size-guard arm64` |

## Full Build

Use the clean release build the first time on a machine, after runtime dependency changes, or before publishing a DMG.

```bash
bash scripts/build-release.sh arm64
```

The clean release build does all of the following:

- creates or activates `.venv`;
- prepares `macos_bundle/python_runtime/install`;
- rebuilds `macos_bundle/python_runtime/site` by default;
- compiles the pinned JamePeng `llama-cpp-python` fork with Metal support;
- builds a minimal LGPL FFmpeg / FFprobe decoder toolset for bundled media
  probing, common MP4/MOV/MKV/WebM/AVI/audio decoding, frame extraction, and
  WAV extraction;
- builds the pinned `pywhispercpp` source tree with Metal enabled and CoreML
  disabled for packaged ASR;
- validates parser dependencies, pywhispercpp Metal, and ChromaDB;
- runs the Tauri release build;
- copies `.app` and `.dmg` to `macos_bundle/release/`.
- verifies the final app signature;
- verifies the DMG checksum;
- scans the app and mounted DMG for local paths, private markers, and test data.

The lower-level equivalent is:

```bash
bash macos_bundle/scripts/package_tauri.sh arm64
```

The default pinned fork commit is:

```text
ef27f333f367fdc53dc1a729ad8bb6c3c9362514
```

Override it only after auditing and testing a new commit:

```bash
UNFOLDLY_LLAMA_CPP_PYTHON_REF=<commit-sha> bash macos_bundle/scripts/package_tauri.sh arm64
```

The packaged ASR runtime also pins `pywhispercpp`:

```text
aaf756bd3c2e8ad38f62bbdc9a32a7549fde9c78
```

Override it only after auditing and testing a new commit:

```bash
UNFOLDLY_PYWHISPERCPP_REF=<commit-sha> bash macos_bundle/scripts/package_tauri.sh arm64
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the fork and
submodule commits recorded for open-source release review.

To force only the reusable site directory to rebuild from scratch:

```bash
rm -rf macos_bundle/python_runtime/site macos_bundle/release
UNFOLDLY_FORCE_REBUILD_SITE=1 bash macos_bundle/scripts/package_tauri.sh arm64
```

To force a completely fresh Python runtime extraction and site rebuild:

```bash
rm -rf macos_bundle/python_runtime macos_bundle/release
bash macos_bundle/scripts/package_tauri.sh arm64
```

## Incremental Build

Use the incremental path after a successful full build when only frontend or Python source files changed. It reuses `macos_bundle/python_runtime/install` and `macos_bundle/python_runtime/site`, then syncs the current backend code into the runtime site before building.

```bash
bash macos_bundle/scripts/package_tauri_no_python.sh arm64
```

The equivalent main-script form is:

```bash
UNFOLDLY_SKIP_PYTHON_PREP=1 bash macos_bundle/scripts/package_tauri.sh arm64
```

If the incremental script fails with a missing reusable runtime, run one full build first.

## High-Compression DMG Build

For release candidates, prefer the guarded clean build:

```bash
bash scripts/build-release.sh arm64
```

After one successful full build has prepared `macos_bundle/python_runtime`, use this advanced command for the smallest styled local DMG without the release privacy/signature wrapper:

```bash
UNFOLDLY_SKIP_PYTHON_PREP=1 UNFOLDLY_MAX_DMG_MB=0 bash macos_bundle/scripts/package_tauri.sh --layout --udbz arm64
```

This path:

- reuses the prepared Python runtime;
- syncs the current backend code into the packaged runtime site;
- reuses or rebuilds the minimal bundled FFmpeg decoder tools;
- builds the Tauri release app;
- creates the static Finder drag-install layout;
- uses `UDBZ` compression.

`UNFOLDLY_MAX_DMG_MB=0` disables the size guard. If you want a guarded release check, set a positive limit such as `UNFOLDLY_MAX_DMG_MB=165`.

## DMG Style and Size Modes

By default, `package_tauri.sh` creates a plain DMG container with `Unfoldly.app` and an `Applications` symlink. This is the simplest mode and uses `UDBZ` compression unless overridden.

| Mode | Command | Notes |
|------|---------|-------|
| Plain DMG, default compression | `bash macos_bundle/scripts/package_tauri.sh arm64` | Default. Good for CI and basic release builds. |
| Smallest practical DMG | `UNFOLDLY_SKIP_PYTHON_PREP=1 UNFOLDLY_MAX_DMG_MB=0 bash macos_bundle/scripts/package_tauri.sh --layout --udbz arm64` | Reuses the Python runtime, uses UDBZ, and disables the local size guard. |
| Faster but usually larger DMG | `bash macos_bundle/scripts/package_tauri_no_python.sh --udzo arm64` | Uses UDZO zlib compression. |
| Styled Finder layout | `bash macos_bundle/scripts/package_tauri_no_python.sh --layout --udbz arm64` | Uses `hdiutil` with `macos_bundle/assets/dmg_background_final.png`. |
| Styled layout smoke test | `bash macos_bundle/scripts/package_tauri_no_python.sh --layout --udbz --no-size-guard arm64` | Only for local visual checks when a temporary build exceeds the size limit. |
| create-dmg layout path | `UNFOLDLY_USE_CREATE_DMG=1 bash macos_bundle/scripts/package_tauri.sh arm64` | Optional alternate path that shells out to `create-dmg`. |

The default size guard is `UNFOLDLY_MAX_DMG_MB=165`. Do not disable it for a release candidate. Use `--no-size-guard` only when testing DMG layout.

## Output Locations

The release copies are written here:

```text
macos_bundle/release/Unfoldly.app
macos_bundle/release/Unfoldly.dmg
```

Tauri also keeps build output under:

```text
apps/desktop/src-tauri/target/aarch64-apple-darwin/release/bundle/
apps/desktop/src-tauri/target/x86_64-apple-darwin/release/bundle/
```

---

## Active Development Mode

If you are developing features and want to test without building `.dmg`s, isolate runtime data first. This prevents test ChromaDB data, downloaded models, preferences, and chat history from being written into the repository or your production data directory.

```bash
export FILEAGENT_DATA_DIR="$HOME/UnfoldlyData-Test"
export DB_PATH="$FILEAGENT_DATA_DIR/chroma_db"
export FILEAGENT_LOCAL_MODELS_DIR="$FILEAGENT_DATA_DIR/local_models"
export FILEAGENT_PREFERENCES_PATH="$FILEAGENT_DATA_DIR/user_preferences.json"
mkdir -p "$FILEAGENT_DATA_DIR"
./scripts/start-dev.sh
```

For a packaged-app smoke test:

```bash
export FILEAGENT_DATA_DIR="$HOME/UnfoldlyData-Test"
export DB_PATH="$FILEAGENT_DATA_DIR/chroma_db"
export FILEAGENT_LOCAL_MODELS_DIR="$FILEAGENT_DATA_DIR/local_models"
export FILEAGENT_PREFERENCES_PATH="$FILEAGENT_DATA_DIR/user_preferences.json"
./macos_bundle/release/Unfoldly.app/Contents/MacOS/Unfoldly
```

## Uninstalling Local App Data

To inspect the app, database, model, log, cache, and preference paths that would
be removed:

```bash
bash scripts/uninstall-mac.sh --show
```

To fully remove Unfoldly and its local runtime data:

```bash
bash scripts/uninstall-mac.sh
```

The script is intentionally destructive and requires typing `yes` before it
deletes anything. It removes local indexes, downloaded models, preferences, chat
history, logs, and caches. It does not delete the original files selected for
indexing.

---

## Troubleshooting and FAQ

### 1. The App Crashes Immediately on Launch
Check Unfoldly's local logs located at `~/Library/Logs/Unfoldly/`.
- **`startup.log` is missing**: The macOS `dyld` linker is failing to locate the internal Python framework, or Gatekeeper is blocking the unsigned binary.
- **`dyld: Library not loaded` error when running via Terminal**: Run a clean rebuild using `rm -rf macos_bundle/python_runtime` and running the clean build step.

### 2. "App cannot be opened because of a problem" (Gatekeeper & Deployment Target)
macOS deployment targets and ad-hoc signatures can cause the application to be blocked on unfamiliar Macs.
- Verify that your system is running macOS 10.15 (Catalina) or higher.
- Strip Apple's quarantine attributes if Gatekeeper blocks the unnotarized application:
  ```bash
  xattr -cr /Applications/Unfoldly.app
  ```

### 3. OSError: [Errno 28] No space left on device
The standalone packaging requires copying your entire environment. Overcome disk limits by pointing the build cache to a larger drive:
```bash
export UNFOLDLY_PYTHON_CACHE="/Volumes/ExternalDrive/unfoldly-build-cache"
bash macos_bundle/scripts/package_tauri.sh arm64
```
