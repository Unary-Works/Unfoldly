# Third-Party Notices

This file records source-built third-party components used by the macOS
packaging workflow.

## llama-cpp-python Fork

- Repository: `https://github.com/JamePeng/llama-cpp-python.git`
- License: MIT
- Pinned commit: `ef27f333f367fdc53dc1a729ad8bb6c3c9362514`
- Build override: set `UNFOLDLY_LLAMA_CPP_PYTHON_REF` to use a different
  pinned commit.

The macOS packaging scripts build this fork from source for newer GGUF model
runtime support. The commit is pinned so release builds are reproducible and do
not silently change when the upstream repository HEAD moves.

## llama.cpp Submodule

The pinned `llama-cpp-python` commit references this submodule:

- Repository: `https://github.com/ggerganov/llama.cpp.git`
- License: MIT
- Submodule path: `vendor/llama.cpp`
- Submodule commit: `e48034dfc9e5705248fd39dc437ca887dc55a528`

The build script initializes the submodule at the commit recorded by the pinned
`llama-cpp-python` tree.

## pywhispercpp Source Build

- Repository: `https://github.com/absadiki/pywhispercpp.git`
- License: MIT
- Pinned commit: `aaf756bd3c2e8ad38f62bbdc9a32a7549fde9c78`
- Build override: set `UNFOLDLY_PYWHISPERCPP_REF` to use a different pinned
  commit.

The macOS packaging script builds `pywhispercpp` from source with Metal enabled
and CoreML disabled for packaged ASR. The pinned commit records these submodules:

- `whisper.cpp`: `4979e04f5dcaccb36057e059bbaed8a2f5288315`
- `pybind11`: `b70b8eb332fadf55d7e22b492da0e954c1a4fcb7`

## Runtime License Notes

See `THIRD_PARTY_DEPENDENCIES.md` for the release-facing summary of packaged
runtime dependencies, PDF parsing, media decoding tools, fonts, and bundled
assets.

## FFmpeg Decoder Tools

- Project: `FFmpeg`
- Source: `https://ffmpeg.org/releases/ffmpeg-7.1.1.tar.xz`
- License for the bundled configuration: LGPL-2.1-or-later
- Build script: `macos_bundle/scripts/build_ffmpeg_lgpl.sh`

The macOS package builds a minimal FFmpeg / FFprobe command-line toolset from
source. It is used as a separate local process for media probing, common
container decoding, frame extraction, and WAV extraction. The bundled build
enables MP4/MOV/M4A, MKV/WebM, AVI, MP3, WAV, FLAC, AAC, AIFF, and OGG
containers with H.264, HEVC/H.265, MPEG-4, MJPEG, VP8, VP9, AV1, AAC, MP3,
FLAC, Vorbis, Opus, ALAC, and PCM decoders. The configure line disables GPL and
nonfree components and does not include x264/x265 encoder libraries. The
packaged app includes the sanitized `BUILD_CONFIG.txt`, `DEMUXERS.txt`,
`DECODERS.txt`, `ENCODERS.txt`, and `NOTICE.txt` files under
`Contents/Resources/ffmpeg/`.

## Model Weights

No model weights are stored in this repository. See `MODELS.md` for runtime
model source metadata and download storage notes.
