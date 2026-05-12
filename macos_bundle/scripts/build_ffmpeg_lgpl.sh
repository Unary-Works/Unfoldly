#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MACOS_BUNDLE="$ROOT_DIR/macos_bundle"
CACHE_DIR="${UNFOLDLY_FFMPEG_CACHE:-$MACOS_BUNDLE/cache}"
FFMPEG_VERSION="${UNFOLDLY_FFMPEG_VERSION:-7.1.1}"
FFMPEG_URL="${UNFOLDLY_FFMPEG_URL:-https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz}"
TARGET_ARCH="${1:-}"

current_build_user() {
  local value="${USER:-${LOGNAME:-}}"
  if [[ -z "$value" ]]; then
    value="$(id -un 2>/dev/null || true)"
  fi
  printf '%s' "$value"
}

if [[ -z "$TARGET_ARCH" ]]; then
  case "$(uname -m)" in
    arm64) TARGET_ARCH="arm64" ;;
    x86_64) TARGET_ARCH="x64" ;;
    *) echo "[build_ffmpeg_lgpl] ERROR: unsupported arch $(uname -m)"; exit 1 ;;
  esac
fi

case "$TARGET_ARCH" in
  arm64)
    FFMPEG_ARCH="arm64"
    ;;
  x64|x86_64)
    TARGET_ARCH="x64"
    FFMPEG_ARCH="x86_64"
    ;;
  *)
    echo "[build_ffmpeg_lgpl] ERROR: unsupported target arch: $TARGET_ARCH"
    exit 1
    ;;
esac

ARCHIVE="$CACHE_DIR/ffmpeg-${FFMPEG_VERSION}.tar.xz"
SRC_DIR="$CACHE_DIR/ffmpeg-${FFMPEG_VERSION}"
BUILD_DIR="$CACHE_DIR/ffmpeg-build-${FFMPEG_VERSION}-${TARGET_ARCH}"
INSTALL_DIR="$MACOS_BUNDLE/ffmpeg/${TARGET_ARCH}"
INSTALL_PREFIX="/unfoldly/ffmpeg"

REQUIRED_DEMUXERS=(
  mov matroska avi mpegts mp3 wav flac ogg aac aiff
)
REQUIRED_DECODERS=(
  h264 hevc mpeg4 mjpeg vp8 vp9 av1
  aac mp3 flac vorbis opus alac
  pcm_s16le pcm_s24le pcm_s32le pcm_f32le
)

has_required_demuxers() {
  local demuxer
  for demuxer in "${REQUIRED_DEMUXERS[@]}"; do
    if ! grep -Eq "[[:space:]]${demuxer}([[:space:],]|$)" "$INSTALL_DIR/DEMUXERS.txt"; then
      echo "[build_ffmpeg_lgpl] Missing demuxer: $demuxer"
      return 1
    fi
  done
  return 0
}

has_required_decoders() {
  local decoder
  for decoder in "${REQUIRED_DECODERS[@]}"; do
    if ! grep -Eq "[[:space:]]${decoder}[[:space:]]" "$INSTALL_DIR/DECODERS.txt"; then
      echo "[build_ffmpeg_lgpl] Missing decoder: $decoder"
      return 1
    fi
  done
  return 0
}

has_private_build_markers() {
  local root="$1"
  [[ -d "$root" ]] || return 1
  python3 - "$root" "$ROOT_DIR" "$HOME" "$(current_build_user)" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
repo = os.path.abspath(sys.argv[2]).encode()
home = os.path.abspath(sys.argv[3]).encode()
build_user = sys.argv[4].encode()
repo_name = os.path.basename(os.path.abspath(sys.argv[2])).encode()
home_name = os.path.basename(os.path.abspath(sys.argv[3])).encode()
needles = [
    value
    for value in (repo, home, build_user, repo_name, home_name)
    if value and len(value) >= 3 and value not in {b"root", b"home"}
]
for path in root.rglob("*"):
    if not path.is_file():
        continue
    try:
        data = path.read_bytes()
    except Exception:
        continue
    if any(needle in data for needle in needles):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

mkdir -p "$CACHE_DIR" "$MACOS_BUNDLE/ffmpeg"

if [[ -x "$INSTALL_DIR/bin/ffmpeg" && -x "$INSTALL_DIR/bin/ffprobe" && -f "$INSTALL_DIR/BUILD_CONFIG.txt" ]]; then
  echo "[build_ffmpeg_lgpl] Existing LGPL FFmpeg bundle found: $INSTALL_DIR"
  "$INSTALL_DIR/bin/ffmpeg" -hide_banner -buildconf 2>&1 \
    | sed -e "s#${ROOT_DIR}#<repo>#g" -e "s#${INSTALL_DIR}#<bundle-ffmpeg>#g" \
    > "$INSTALL_DIR/BUILD_CONFIG.txt" || true
  "$INSTALL_DIR/bin/ffmpeg" -hide_banner -demuxers > "$INSTALL_DIR/DEMUXERS.txt" 2>&1 || true
  "$INSTALL_DIR/bin/ffmpeg" -hide_banner -decoders > "$INSTALL_DIR/DECODERS.txt" 2>&1 || true
  "$INSTALL_DIR/bin/ffmpeg" -hide_banner -encoders > "$INSTALL_DIR/ENCODERS.txt" 2>&1 || true
  if grep -E -- '--enable-(gpl|nonfree)|libx264|libx265|libfdk_aac' "$INSTALL_DIR/BUILD_CONFIG.txt" >/dev/null \
    || ! has_required_demuxers \
    || ! has_required_decoders \
    || grep -E "libx264|libx265| h264 | hevc " "$INSTALL_DIR/ENCODERS.txt" >/dev/null \
    || has_private_build_markers "$INSTALL_DIR"; then
    echo "[build_ffmpeg_lgpl] Existing bundle failed LGPL/decoder validation; rebuilding."
    rm -rf "$INSTALL_DIR"
  fi
fi

if [[ -x "$INSTALL_DIR/bin/ffmpeg" && -x "$INSTALL_DIR/bin/ffprobe" ]]; then
  exit 0
fi

if [[ ! -f "$ARCHIVE" ]]; then
  echo "[build_ffmpeg_lgpl] Downloading $FFMPEG_URL"
  curl -L --fail -o "$ARCHIVE" "$FFMPEG_URL"
fi

if [[ ! -d "$SRC_DIR" ]]; then
  echo "[build_ffmpeg_lgpl] Extracting $ARCHIVE"
  tar -xf "$ARCHIVE" -C "$CACHE_DIR"
fi

rm -rf "$BUILD_DIR" "$INSTALL_DIR"
mkdir -p "$BUILD_DIR" "$INSTALL_DIR"
cp -R "$SRC_DIR"/. "$BUILD_DIR/"

cd "$BUILD_DIR"

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-10.15}"

CONFIGURE_FLAGS=(
  "--prefix=$INSTALL_PREFIX"
  "--datadir=share/ffmpeg"
  "--cc=clang"
  "--arch=$FFMPEG_ARCH"
  "--target-os=darwin"
  "--enable-cross-compile"
  "--disable-autodetect"
  "--disable-doc"
  "--disable-debug"
  "--disable-network"
  "--disable-gpl"
  "--disable-nonfree"
  "--disable-programs"
  "--enable-ffmpeg"
  "--enable-ffprobe"
  "--disable-everything"
  "--enable-small"
  "--enable-protocol=file,pipe"
  "--enable-demuxer=mov,matroska,avi,mpegts,mp3,wav,flac,ogg,aac,aiff"
  "--enable-muxer=image2,wav"
  "--enable-decoder=h264,hevc,mpeg4,mjpeg,vp8,vp9,av1,aac,mp3,mp3float,flac,vorbis,opus,alac,pcm_s16le,pcm_s24le,pcm_s32le,pcm_f32le"
  "--enable-parser=h264,hevc,mpeg4video,vp8,vp9,av1,mpegaudio,aac,vorbis,opus"
  "--enable-encoder=mjpeg,pcm_s16le"
  "--enable-filter=aresample,anull,aformat,format,scale"
  "--enable-avcodec"
  "--enable-avformat"
  "--enable-avfilter"
  "--enable-swresample"
  "--enable-swscale"
)

echo "[build_ffmpeg_lgpl] Configuring FFmpeg ${FFMPEG_VERSION} (${TARGET_ARCH})"
./configure "${CONFIGURE_FLAGS[@]}"

echo "[build_ffmpeg_lgpl] Building FFmpeg"
make -j"$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
STAGED_INSTALL="$BUILD_DIR/stage"
rm -rf "$STAGED_INSTALL"
make install "DESTDIR=$STAGED_INSTALL"
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
if [[ -d "$STAGED_INSTALL/$INSTALL_PREFIX" ]]; then
  cp -R "$STAGED_INSTALL/$INSTALL_PREFIX"/. "$INSTALL_DIR/"
else
  echo "[build_ffmpeg_lgpl] ERROR: staged FFmpeg install missing: $STAGED_INSTALL/$INSTALL_PREFIX"
  exit 1
fi

strip "$INSTALL_DIR/bin/ffmpeg" "$INSTALL_DIR/bin/ffprobe" 2>/dev/null || true

"$INSTALL_DIR/bin/ffmpeg" -hide_banner -buildconf 2>&1 \
  | sed -e "s#${ROOT_DIR}#<repo>#g" -e "s#${INSTALL_DIR}#<bundle-ffmpeg>#g" \
  > "$INSTALL_DIR/BUILD_CONFIG.txt" || true
"$INSTALL_DIR/bin/ffmpeg" -hide_banner -demuxers > "$INSTALL_DIR/DEMUXERS.txt" 2>&1 || true
"$INSTALL_DIR/bin/ffmpeg" -hide_banner -decoders > "$INSTALL_DIR/DECODERS.txt" 2>&1 || true
"$INSTALL_DIR/bin/ffmpeg" -hide_banner -encoders > "$INSTALL_DIR/ENCODERS.txt" 2>&1 || true

if grep -E -- '--enable-(gpl|nonfree)|libx264|libx265|libfdk_aac' "$INSTALL_DIR/BUILD_CONFIG.txt" >/dev/null; then
  echo "[build_ffmpeg_lgpl] ERROR: forbidden GPL/nonfree option found in FFmpeg build config"
  exit 1
fi
if ! has_required_demuxers || ! has_required_decoders; then
  echo "[build_ffmpeg_lgpl] ERROR: bundled FFmpeg is missing required media format support"
  exit 1
fi
if grep -E "libx264|libx265| h264 | hevc " "$INSTALL_DIR/ENCODERS.txt" >/dev/null; then
  echo "[build_ffmpeg_lgpl] ERROR: H.264/H.265 encoders must not be included"
  exit 1
fi

cat > "$INSTALL_DIR/NOTICE.txt" <<EOF
Bundled FFmpeg
===============

Project: FFmpeg
Version: ${FFMPEG_VERSION}
Source: ${FFMPEG_URL}
License: LGPL-2.1-or-later for this build configuration.

This Unfoldly build includes FFmpeg and FFprobe as separate command-line tools
for local media decoding, probing, frame extraction, and WAV extraction. It
enables common local decode paths for MP4/MOV/M4A, MKV/WebM, AVI, MP3, WAV,
FLAC, AAC, AIFF, and OGG containers with H.264, HEVC/H.265, MPEG-4, MJPEG,
VP8, VP9, AV1, AAC, MP3, FLAC, Vorbis, Opus, ALAC, and PCM decoders. It is
configured without --enable-gpl, without --enable-nonfree, and without x264/x265
encoding libraries. H.264 and HEVC/H.265 support is decoder-only.

The exact configure output is stored in BUILD_CONFIG.txt.
EOF

du -sh "$INSTALL_DIR" || true
echo "[build_ffmpeg_lgpl] Installed LGPL FFmpeg bundle: $INSTALL_DIR"
