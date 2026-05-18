<p align="center">
  <img src="docs/assets/brand/unfoldly-logo.png" alt="Unfoldly" width="120">
</p>

# Unfoldly

<p>
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
  <a href="https://www.unfoldly.io/"><img alt="Website" src="https://img.shields.io/badge/website-unfoldly.io-111111"></a>
  <img alt="Platform: macOS" src="https://img.shields.io/badge/platform-macOS-black">
  <img alt="Local-first private AI" src="https://img.shields.io/badge/local--first-private%20AI-00A66A">
  <img alt="Status: beta" src="https://img.shields.io/badge/status-beta-orange">
</p>

**Find any file, even when you forgot its name.**

Unfoldly is built for the messy folders we all have: Documents, Downloads, Desktop, old hard drives, and random project folders filled with videos, images, audio files, PDFs, screenshots, and notes you no longer remember how to search for.

Instead of typing exact file names, just describe what you are looking for:

- "Find images of my dog."
- "Show me the video where I presented the demo."
- "Find the audio recording from the investor meeting."
- "Show me screenshots about Stripe setup."
- "Find the PDF that mentioned pricing strategy."

Unfoldly indexes your local files and lets you search them by meaning, not just by filename or folder path. Once you find the right files, you can keep asking follow-up questions: summarize a document, ask about a video or audio file, compare notes, extract key points, or chat with a folder.

Everything runs locally. No account required. Your files stay on your device.

<p align="center">
  <a href="https://www.unfoldly.io/">Website</a> |
  <a href="https://www.unfoldly.io/">Download</a> |
  <a href="#demos">Demos</a> |
  <a href="#privacy">Privacy</a> |
  <a href="#build-from-source">Build</a> |
  <a href="#uninstall">Uninstall</a> |
  <a href="docs/zh/README.md">中文</a>
</p>

> **Early beta:** The current public release is macOS-only and distributed as a signed, notarized macOS build.

## Product Overview

Unfoldly turns the files and folders you choose into a private, searchable memory layer on your computer.

1. **Choose your sources**  
   Add the folders or files you want Unfoldly to index.

2. **Build local memory**  
   Unfoldly processes your files locally, extracting text, visual signals, metadata, transcripts, and searchable embeddings depending on the file type.

3. **Search and ask**  
   Describe what you are looking for, inspect the source files Unfoldly finds, then ask follow-up questions in context.

Indexes, downloaded models, preferences, logs, and chat history are stored in the local app data directory.

## Demos

### Find the photo you remember

Search local images by what is inside them, not by filename.

[![Find the photo you remember](docs/assets/demos/find-my-dogs.gif)](docs/assets/videos/find-my-dogs.mp4)

[Watch full video](docs/assets/videos/find-my-dogs.mp4)

### Find the video from a moment

Describe the moment you remember. Unfoldly finds the local video and the relevant moment.

[![Find the video from a moment](docs/assets/demos/manta.gif)](docs/assets/videos/manta.mp4)

[Watch full video](docs/assets/videos/manta.mp4)

### Find the answer hidden in your files

Find the right local file first, then ask a follow-up question. Unfoldly keeps the source in context so you can inspect where the answer came from.

[![Find the answer hidden in your files](docs/assets/demos/excel-analysis.gif)](docs/assets/videos/excel-analysis.mp4)

[Watch full video](docs/assets/videos/excel-analysis.mp4)

## What You Can Search

Unfoldly is designed for everyday personal files.

| Type | Formats |
| --- | --- |
| Documents | PDF, DOC, DOCX, TXT, Markdown, RTF, EPUB, MOBI |
| Spreadsheets | XLSX, XLS, CSV, TSV, Numbers, ODS |
| Slides | PPTX, PPT, Keynote, ODP |
| Images | JPG, PNG, HEIC, HEIF, WEBP, GIF, TIFF, BMP, SVG |
| Audio | MP3, WAV, M4A, FLAC, AAC, OGG, WMA, AIFF |
| Video | MP4, MOV, MKV, AVI, WEBM, M4V, WMV |
| Other structured files | JSONL, XML, SQL, YAML |

Support quality can vary by file structure, size, parser availability, and selected model.

## Core Features

- **Search by memory**: find files by what you remember, not just filenames or exact keywords.
- **Search visual content**: make photos, screenshots, scanned files, and video moments searchable.
- **Ask source-aware questions**: retrieve the right file, then ask follow-up questions in context.
- **Choose what gets indexed**: Unfoldly only indexes the files and folders you select.
- **Run locally**: the whole search workflow is designed to run on your machine.

## Privacy

Unfoldly is built around a simple principle:

**Your local files should stay local.**

Unfoldly keeps the whole search and chat workflow on your device.

| Data | Where it lives |
| --- | --- |
| Source files | Your selected local folders |
| Search indexes | Local data directory |
| Downloaded models | Local data directory |
| Preferences | Local data directory |
| Chat history | Local data directory |
| Logs | Local data directory |

Unfoldly may connect to the internet to download model files or application releases, but your personal file contents are not uploaded.

## Installation

[Download Now](https://www.unfoldly.io/)

The current public build is focused on macOS and is distributed as a signed, notarized macOS app.

## Quick Start

1. Download and open Unfoldly.
2. Add the files or folders you want to search.
3. Download or select a supported local model.
4. Wait for indexing to complete.
5. Search by memory, meaning, or visual content.
6. Open the source file or ask a follow-up question.

## Uninstall

Unfoldly includes a macOS uninstall helper for users who want to inspect or remove local app data.

Show the app, database, model, preference, log, cache, and crash-report paths that would be removed:

```bash
bash scripts/uninstall-mac.sh --show
```

Fully remove Unfoldly and its local runtime data:

```bash
bash scripts/uninstall-mac.sh
```

The uninstall script is intentionally destructive and requires typing `yes` before deletion. It removes local indexes, downloaded models, preferences, selected-source records, chat history, logs, caches, crash reports, and installed app bundles. It does not delete the original files you selected for indexing.

## Technical Architecture

Unfoldly is built as a local desktop application.

| Layer | Technology |
| --- | --- |
| Desktop shell | Tauri |
| Frontend | React, TypeScript, Vite |
| Native bridge | Rust + PyO3 |
| Backend | Python |
| Vector store | ChromaDB |
| Retrieval | Embeddings, lexical search, reranking, source-scoped filtering |
| Model runtime | Local GGUF models through llama.cpp / llama-cpp-python |
| Storage | Local app data directory |

The packaged desktop app embeds the Python backend into the Tauri application, so normal packaged use does not require a separate local HTTP server.

## Build from Source

For development and release packaging, see the full [macOS build guide](docs/BUILD_MACOS.md).
For publishing GitHub Releases and enabling in-app update checks, see [Release Publishing](docs/RELEASE.md).

Minimum local toolchain:

- macOS
- Python 3.12
- Rust
- Node.js and npm
- Git and CMake

Start the local development app:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./scripts/start-dev.sh
```

Create a clean macOS release build:

```bash
bash scripts/build-release.sh arm64
```

Release artifacts are written to:

```text
macos_bundle/release/Unfoldly.app
macos_bundle/release/Unfoldly.dmg
```

Runtime data such as indexes, downloaded models, preferences, logs, and chat history is stored in the local app data directory.

## Models and Retrieval

Unfoldly is designed to run supported local AI models through a local runtime.

The current model registry includes GGUF-based text and vision-language models from families such as:

- Gemma
- Qwen / Qwen-VL
- GLM
- Llama
- Ministral
- DeepSeek-R1 Distill
- gpt-oss

The retrieval system currently uses **ChromaDB** as the local vector store, with BGE-family embedding and reranking models.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

All pull requests are reviewed by Unary Works maintainers before merge.

## License

Unfoldly is licensed under the [Apache License 2.0](LICENSE).

Copyright (c) 2026 Unary Works LLC.

Bundled third-party dependencies retain their own licenses. See [Third-Party License Notes](docs/THIRD_PARTY_DEPENDENCIES.md), [Third-Party Notices](docs/THIRD_PARTY_NOTICES.md), and [NOTICE](NOTICE) for dependency license and notice details.
