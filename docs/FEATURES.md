# Unfoldly Feature Overview

This document summarizes Unfoldly's main backend, indexing, retrieval, model-management, and packaging capabilities. The checklist items are lightweight manual verification notes for maintainers.

---

## Contents

1. [File Indexing and Ingestion](#1-file-indexing-and-ingestion)
2. [Knowledge Base Retrieval](#2-knowledge-base-retrieval)
3. [Intent Routing](#3-intent-routing)
4. [Multi-Turn Chat](#4-multi-turn-chat)
5. [Scope Isolation](#5-scope-isolation)
6. [Model Management](#6-model-management)
7. [Personal Info Extraction](#7-personal-info-extraction)
8. [REST API](#8-rest-api)
9. [Build and Deployment](#9-build-and-deployment)
10. [Configuration and Storage](#10-configuration-and-storage)
11. [QA Checklist](#11-qa-checklist)

---

## 1. File Indexing and Ingestion

### 1.1 Supported File Types

| Format | Processing path | Status |
|--------|-----------------|--------|
| `.pdf` | pypdf text extraction with pypdfium2 page rendering for optional vision-language OCR | Implemented |
| `.docx` | `docx2txt` text extraction plus image OCR through Qwen3-VL | Implemented |
| `.pptx` | `python-pptx` slide text and table extraction | Implemented |
| `.txt` / `.md` | Direct UTF-8 read path | Implemented |
| `.csv` / `.tsv` | pandas parsing with encoding fallback from `utf-8-sig` to `gb18030` to `latin-1`; pipe characters are escaped for Markdown tables | Implemented |
| `.xlsx` / `.xls` / `.numbers` | pandas or LlamaIndex extraction converted to Markdown tables | Implemented |
| `.jpg` / `.jpeg` / `.png` / `.gif` / `.bmp` / `.webp` / `.heic` / `.tiff` | Image description and summary through Qwen3-VL | Implemented |
| `.mp3` / `.wav` / `.flac` / `.aac` / `.ogg` / `.m4a` / `.mp4` / `.mov` / `.m4v` / `.mkv` / `.webm` / `.avi` | Media metadata, audio extraction, H.264 / HEVC decoding, and frame extraction through bundled minimal FFmpeg on macOS, with system FFmpeg as a development fallback | Implemented |
| Other formats | LlamaIndex fallback extraction | Implemented |

QA:

- [ ] Index an image-only PDF and verify OCR text is searchable.
- [ ] Index a GBK-encoded CSV and verify decoding is correct.
- [ ] Index a PPTX and verify each slide can be retrieved by content.

### 1.2 Document Classification

Documents are categorized during indexing.

| Category | Meaning |
|----------|---------|
| `document` | General document files such as PDF, DOCX, and DOC |
| `report` | Reports and summaries |
| `paper` | Academic papers |
| `book` | Long-form books or ebook-like documents |
| `manual` | Manuals and documentation files such as Markdown, text, RST, and README files |
| `resume` | Resumes and CVs |
| `contract` | Contracts and agreements |
| `invoice` | Invoices |
| `presentation` | Slide decks |
| `data` | Spreadsheets and structured data files |
| `image` / `photo` | Images |
| `audio/video` | Audio and video files |
| `other` | Uncategorized files |

The classifier includes an extension guard. If a category conflicts with the file extension, for example a `.docx` being labeled as an image, the system falls back to a safe default.

QA:

- [ ] Index a PDF and verify `doc_category` is reasonable.
- [ ] Index an image and verify it is not categorized as a document.

### 1.3 Indexing Pipeline

- Smart indexing: LLM-generated summary, classification, and keyword extraction.
- Incremental skip: previously indexed files are skipped by stable path hash.
- Resume support: interrupted indexing jobs can resume on the next startup.
- Progress reporting: current file, total files, completed files, ETA, and status are surfaced to the frontend.
- Concurrency guard: model switching is blocked during indexing to reduce GPU and Metal resource contention.
- CSV table safety: pipe characters are escaped so Markdown table structure remains valid.

---

## 2. Knowledge Base Retrieval

### 2.1 Vector Search

- ChromaDB local vector storage.
- `allowed_paths` support for source-scope filtering.
- Local embedding models, including BGE-Small-ZH and BGE-M3.
- Query translation and multilingual augmentation for cross-language retrieval.
- Category and file-extension filtering.
- Path validation through `_is_path_allowed()` for both folder prefixes and exact file matches.

### 2.2 Reranking

- BGE-Reranker based reranking on vector candidates.
- Configurable relevance threshold through `RELEVANCE_THRESHOLD`.
- File-level deduplication so repeated chunks from one file do not dominate results.

### 2.3 Lexical Search

- Filename stem matching, for example `sensevoice` can match `sensevoice.png`.
- ChromaDB metadata lookup by `file_name`, `file_name_no_ext`, and `file_path`.
- Hybrid candidate merging across semantic, lexical, category, and folder-inventory routes.

QA:

- [ ] Search for a file stem and verify files with different extensions are returned.
- [ ] Search by a category keyword and verify recall quality.

---

## 3. Intent Routing

All user input is classified before execution and routed to a matching action pipeline.

### 3.1 Supported Intents

| Intent | Scenario | Handler |
|--------|----------|---------|
| `search` | Semantic or keyword search | Vector search plus rerank |
| `count` | Quantity questions | `_handle_count` |
| `summarize` | Category-specific summary | `_handle_summarize` |
| `summarize_all` | Global or selected-source overview | `_handle_summarize_all` |
| `process_previous` | Follow-up questions about prior results | `_handle_process_previous` plus a lightweight sub-agent router |
| `list_selected` | Questions about selected files | Routed to selected-scope count or summary |
| `view_detail` | Open a specific ranked result for detail | `_handle_view_detail` |
| `open_file` | Open a file through the system or frontend bridge | File-content cache plus frontend event |
| `chat` | General conversation and capability explanation | LLM chat |
| `db_clear` | Clear index/database requests | Database cleanup pipeline |

### 3.2 Analysis Flow

- Two-stage parsing: fast rules first, then LLM-based parsing for ambiguous requests.
- Context injection: recent search results are included in intent analysis for follow-ups.
- Bilingual routing rules for English and Chinese queries.
- LLM intent correction through `correct_llm_intent`.

### 3.3 Follow-Up Router

Follow-up requests are handled by a lightweight sub-agent router.

| Router output | Meaning | Action |
|---------------|---------|--------|
| `global_summary` | Summary or statistics over previous results | Summarize the current context file list |
| `search` | Entity or topic query within previous context | Search inside the previous result set |

QA:

- [ ] Ask a follow-up about a previous result and verify it stays in context.
- [ ] Ask for a global summary after a search and verify it uses the prior result set.

---

## 4. Multi-Turn Chat

- Session history is isolated by `session_id`.
- The system keeps recent search and count results for follow-up routing.
- Follow-up hints are set after selected operations to improve short follow-up handling.
- New requests can interrupt prior unfinished requests in the same session.
- Server-Sent Events are used for streaming responses.
- Client disconnects are detected during streaming so backend inference can stop early.

QA:

- [ ] Send two requests in one session and verify the first can be interrupted.
- [ ] Disconnect a streaming client and verify backend logs show a client disconnect.

---

## 5. Scope Isolation

When the frontend passes `active_paths`, all retrieval actions must stay physically limited to those selected sources.

### 5.1 Enforcement by Intent

| Intent | Scope enforced | Notes |
|--------|----------------|-------|
| `search` | Yes | `vector_search(allowed_paths=active_paths)` |
| `count` / `count_all` | Yes | No global-count bypass when selected sources are active |
| `summarize` | Yes | Additional `_path_allowed()` filtering |
| `summarize_all` | Yes | `count_by_category(allowed_paths=active_paths)` |
| `process_previous` | Yes | Prior result context is intersected with active paths |
| Tool Agent tools | Yes | Tools read active paths from request context |

### 5.2 Selection Change Reset

- Selection changes are detected per request.
- Session history, previous results, follow-up hints, and count-scope context are cleared when source scope changes.
- A global fallback session key is used when no explicit `session_id` exists.

QA:

- [ ] Select all files, ask for a summary, then narrow the source selection and ask again.
- [ ] Verify the second answer only uses the new selected files.
- [ ] Verify previous search results are cleared after source-scope changes.

---

## 6. Model Management

### 6.1 Chat Models

Supported product models are defined in `config/supported_models.json` and filtered for UI display. Current families include:

| Model family | Type | Notes |
|--------------|------|-------|
| Qwen3 | Text and vision-language | Main local chat and OCR-capable models |
| Gemma | Text | Larger high-quality local models |
| Llama | Text | Meta Llama variants |
| DeepSeek-R1 Distill | Text/reasoning | Distilled reasoning model variants |
| Ministral | Text | Mistral small and medium local models |

### 6.2 Model Actions

- One-click GGUF downloads from Hugging Face and ModelScope sources.
- Automatic source selection based on source probing and download performance.
- Quantization selection per model file.
- Hot model switching without app restart.
- Download cancellation.
- Model deletion by quantized file.
- Persistent model and quantization preferences in `user_preferences.json`.

QA:

- [ ] Download a model and verify progress updates.
- [ ] Switch chat models and verify the next response uses the selected model.
- [ ] Cancel an active download and verify the next attempt can resume or restart cleanly.

### 6.3 Embedding and Reranker Models

- Core models can be downloaded automatically when onboarding requires them.
- Onboarding avoids background downloads before the setup flow is active.
- Embedding/reranker models are stored separately from chat models.
- Runtime readiness checks prevent entering the next onboarding step when a model is downloaded but not loadable.

---

## 7. Personal Info Extraction

- Structured personal-information extraction is available during indexing.
- Extracted entities are stored in a local `PersonalInfoDB` SQLite database.
- Public HTTP debug APIs are disabled by default and require `FILEAGENT_ENABLE_PERSONAL_INFO_API=1`.

Privacy note: public builds should not include user-derived fixtures, personal datasets, local paths, or test reports containing private data.

QA:

- [ ] Index a synthetic contact-info fixture and verify local lookup behavior.
- [ ] Verify `/api/personal_info/search` and `/api/personal_info/stats` are disabled by default.

---

## 8. REST API

### 8.1 Core Endpoints

| Path | Method | Purpose |
|------|--------|---------|
| `/health` | GET | Health check |
| `/api/runtime/paths` | GET | Runtime data and model paths |
| `/api/sources` | GET | Added sources and file tree |
| `/api/sources/add` | POST | Add a source |
| `/api/sources/remove` | POST | Remove a source |
| `/api/index/start` | POST | Start folder indexing |
| `/api/index/files` | POST | Index explicit file paths |
| `/api/index/status` | GET | Indexing status |
| `/api/index/cancel` | POST | Cancel indexing |
| `/api/index/active` | GET | Active indexing job |
| `/api/query` | POST | Synchronous chat query |
| `/api/query_stream` | POST | SSE streaming chat query |
| `/api/query/abort` | POST | Abort active stream |

### 8.2 Model Endpoints

| Path | Method | Purpose |
|------|--------|---------|
| `/api/models` | GET | List models and status |
| `/api/models/select` | POST | Select chat model |
| `/api/models/quantization/select` | POST | Select quantized file |
| `/api/models/download` | POST | Start model download |
| `/api/models/cancel` | POST | Cancel model download |
| `/api/models/delete` | POST | Delete model file |
| `/api/core_models/status` | GET | Embedding/reranker status |
| `/api/core_models/download` | POST | Download embedding/reranker models |

### 8.3 Other Endpoints

| Path | Method | Purpose |
|------|--------|---------|
| `/api/history` | GET | Read chat history |
| `/api/history/sync` | POST | Sync chat history |
| `/api/history/delete` | POST | Delete a chat-history item |
| `/api/personal_info/search` | GET | Search personal info; disabled by default |
| `/api/personal_info/stats` | GET | Personal-info stats; disabled by default |

### 8.4 SSE Event Types

| Event | Meaning |
|-------|---------|
| `status` | Processing state |
| `text` | Text delta |
| `files` | Retrieved file list |
| `opened_file` | Opened file event with content |
| `trace_append` | Tool-call trace event |
| `done` | Final event with metadata |

---

## 9. Build and Deployment

### 9.1 Tauri Packaging

- Self-contained macOS app with bundled Python runtime and dependencies.
- Apple Silicon and Intel architecture support.
- Metal acceleration for Apple Silicon llama.cpp builds.
- `llama-cpp-python` is built from the JamePeng fork for newer model support.

```bash
bash scripts/build-release.sh arm64   # Apple Silicon
bash scripts/build-release.sh x64     # Intel
```

### 9.2 Development Mode

```bash
bash scripts/start-dev.sh
```

### 9.3 Environment Variables

| Variable | Meaning | Default |
|----------|---------|---------|
| `FILEAGENT_DATA_DIR` | Root data directory | App/executable directory |
| `DB_PATH` | ChromaDB vector database path | `$FILEAGENT_DATA_DIR/chroma_db` |
| `FILEAGENT_LOCAL_MODELS_DIR` | GGUF chat model directory | `$FILEAGENT_DATA_DIR/local_models` |
| `FILEAGENT_PREFERENCES_PATH` | User preferences JSON | `$FILEAGENT_DATA_DIR/user_preferences.json` |
| `FILEAGENT_DISABLE_WARMUP` | Disable background warmup | `false` |
| `FILEAGENT_PRELOAD_AGENT` | Preload FileAgent at startup | `false` |
| `DEV_NO_MODEL_LOAD` | Skip model loading in development | `false` |
| `MAX_TABLE_INDEX_CHARS` | Max spreadsheet indexing characters | `5000000` |
| `OCR_MAX_TOKENS` | Max image OCR tokens | `2200` |

---

## 10. Configuration and Storage

### 10.1 Data Directory

```text
$FILEAGENT_DATA_DIR/
├── chroma_db/              # ChromaDB vector index
├── local_models/           # GGUF chat models
├── models/                 # Embedding and reranker models
├── user_preferences.json   # Preferences, selected model, quantization, onboarding state
├── indexed_folders.json    # Indexed source folders and files
└── logs/                   # Runtime logs
```

### 10.2 Persistent Preferences

- Selected chat model and quantized file.
- Onboarding completion and current step.
- Added source folders and files.
- Optional chat history sync.

---

## 11. QA Checklist

### Smoke Test

- [ ] Start the app and verify `/health` returns 200.
- [ ] Add a folder as a source.
- [ ] Start indexing and verify progress updates.
- [ ] Ask what files are available after indexing completes.
- [ ] Verify the Sources panel shows the expected file list.
- [ ] Ask a follow-up summary question and verify `process_previous` routing.

### Scope Isolation Test

- [ ] Select all sources and ask for a summary.
- [ ] Narrow the Sources selection to a small subset.
- [ ] Ask for a summary again and verify only the selected subset is used.
- [ ] Ask for a file count and verify it matches the selected subset, not the full library.
- [ ] Ask a follow-up question and verify the answer stays inside the selected subset.

### Multimodal Test

- [ ] Index an image and ask for its content.
- [ ] Index a scanned PDF and verify OCR-based retrieval.
- [ ] Index a PPTX and search for slide-specific text.

---

Generated from the current codebase for public documentation.
