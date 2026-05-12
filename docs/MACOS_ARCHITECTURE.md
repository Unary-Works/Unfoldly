# Unfoldly macOS Architecture & Data Paths

[Chinese version](zh/MACOS_ARCHITECTURE.md)

---

This directory contains resources and scripts for packaging Unfoldly into a standalone macOS application using **Tauri** and **PyO3**.
Unfoldly natively embeds the Python backend directly within the Tauri Rust process, eliminating the need for local ports and resulting in a seamless, zero-config launch.

For build instructions, please read [BUILD_MACOS.md](BUILD_MACOS.md).

## 1. Data Preservation & Persistence

Packaged applications on macOS bundle all resources inside `.app/Contents/Resources`. Because this application directory is strictly read-only after signing, Unfoldly must persist data (like vector search indices, user preferences, and downloaded AI models) outside of the `.app` bundle.

### Default Behaviors
When Unfoldly is started without any specific environment variables configurations:
- **macOS Application**: It sets the overarching data directory dynamically depending on where you opened it (e.g., if launched from `/Applications`, data persists in `/Applications/...` - highly NOT recommended).
- **Development Environment**: It targets the repository root.

### Modifying the Data Route (`FILEAGENT_DATA_DIR`)
You can force Unfoldly to compartmentalize and store all user-generated databases and logs into a dedicated folder (like the standard `Application Support` directory) by configuring the environment variable:

```bash
export FILEAGENT_DATA_DIR="$HOME/Library/Application Support/Unfoldly"
open /Applications/Unfoldly.app
```

*Note: For the official packaged releases, this variable is natively hardcoded and injected during the Tauri launch sequence to guarantee data safety.*

## 2. Directory Structure Under the Data Route

Inside of your designated `FILEAGENT_DATA_DIR`, Unfoldly creates the following structure dynamically:

```text
Unfoldly/
├── chroma_db/               # High-speed Vector database matrices
├── local_models/            # Downloaded chat models (GGUF, etc.)
├── models/                  # Downloaded embedding & reranker utility models
├── user_preferences.json    # End-user visual and systemic config states
└── logs/                    # Application rotation logs (backend.log, crash.log)
```
*(Models like `bge-small-zh` or `bge-reranker` will be downloaded automatically by the native backend to the `models/` folder directly without any extra user setup.)*

## 3. Crashing Scenarios

If the newly created Unfoldly app immediately closes unprompted:
1. Try analyzing the system crash logs located under `~/Library/Logs/Unfoldly/`.
2. A missing `startup.log` indicates that the internal Python dyld (Dynamic Link Editor) bindings have cracked. Please review the build script to ensure `rpath` injected the internal `lib/` directory successfully.
3. If it generates `crash.log` containing `UnicodeEncodeError`, Apple's bare Finder terminal passed un-encoded path strings during the cold start. Repackage the bundle ensuring the macOS target flags `PYTHONUTF8=1` and `LC_ALL=en_US.UTF-8` correctly.
