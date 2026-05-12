# Third-Party License Notes

This file summarizes the public-release license posture for the dependencies
that are most relevant to packaged desktop distribution. It is not a substitute
for a generated software bill of materials for a final signed release.

## PDF Parsing

- `pypdf`: BSD-3-Clause. Used for text extraction from text-based PDFs.
- `pypdfium2`: Apache-2.0 / BSD-3-Clause. Used to render PDF pages as images
  for the existing vision-language OCR fallback.

## Audio And Video

- `FFmpeg` / `FFprobe`: LGPL-2.1-or-later for the bundled source-built
  command-line toolset used by the macOS package for media probing, common
  container decoding, frame extraction, and WAV extraction.

The bundled FFmpeg build is configured without GPL or nonfree components and
does not include x264/x265 encoder libraries. The packaged app includes the
sanitized FFmpeg build configuration, demuxer list, decoder list, encoder list,
and notice file under `Contents/Resources/ffmpeg/`.

## Fonts

Bundled UI fonts:

- Inter
- Lora
- Manrope

These font families are distributed under the SIL Open Font License 1.1 in their
upstream releases. The OFL 1.1 text is included with the redistributed font
files at `apps/desktop/public/fonts/OFL.txt`.

## Application Assets

The bundled app icons, logo images, and DMG background assets under
`apps/desktop/assets/` and `macos_bundle/assets/` are treated as first-party
Unfoldly release assets.
