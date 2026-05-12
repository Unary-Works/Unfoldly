# Unfoldly Frontend

This directory contains the Tauri frontend and Rust app shell for Unfoldly.

## Key Files

| Path | Purpose |
|------|---------|
| `App.tsx` | Main React application. |
| `backend.ts` | Frontend bridge for Tauri backend commands. |
| `components/` | Application UI components. |
| `src-tauri/` | Rust/Tauri app shell and PyO3 backend embedding. |

## Development

Run frontend and Tauri development mode from the repository root:

```bash
cd apps/desktop
npm install
cd ../..
./scripts/start-dev.sh
```

Use an isolated runtime data directory for local tests:

```bash
export FILEAGENT_DATA_DIR="$HOME/UnfoldlyData-Test"
export DB_PATH="$FILEAGENT_DATA_DIR/chroma_db"
export FILEAGENT_LOCAL_MODELS_DIR="$FILEAGENT_DATA_DIR/local_models"
export FILEAGENT_PREFERENCES_PATH="$FILEAGENT_DATA_DIR/user_preferences.json"
mkdir -p "$FILEAGENT_DATA_DIR"
./scripts/start-dev.sh
```

## Packaging

Release packaging should be launched from the repository root with `scripts/build-release.sh`. See [../../docs/BUILD_MACOS.md](../../docs/BUILD_MACOS.md).
