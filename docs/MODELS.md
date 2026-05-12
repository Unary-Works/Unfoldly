# Model Sources

Unfoldly does not commit or redistribute model weights in this repository.
Models are downloaded at runtime into the user's configured data directory.

## Chat Models

Supported chat model metadata lives in `config/supported_models.json`. Each
entry records the Hugging Face and ModelScope repository used by the downloader.
The downloader may choose the fastest available source, but the selected file is
expected to be the same model artifact named in the metadata.

Users and distributors must review and comply with the upstream license for each
downloaded model repository before redistribution or commercial use.

## Retrieval Models

The default retrieval model repositories are configured in `config/settings.py`:

- Embedding model: `gpustack/bge-m3-GGUF`
- Reranker model: `gpustack/bge-reranker-v2-m3-GGUF`

These weights are also runtime downloads and are not part of the source tree.

## Runtime Storage

Downloaded model files should live outside the repository:

- Chat GGUF models: `$FILEAGENT_DATA_DIR/local_models`
- Embedding and reranker models: `$FILEAGENT_DATA_DIR/models`

Do not commit downloaded weights, cache directories, or model conversion outputs.
