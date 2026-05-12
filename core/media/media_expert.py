"""
MediaExpert — Unified audio/video processing engine.

Provides:
  - ASR transcription with timestamps (via pywhispercpp / whisper.cpp local models)
  - Optional speaker diarization for meeting minutes (via local GGUF diarizer)
  - Video keyframe extraction (via system ffmpeg, with optional PyAV opt-in)
  - LLM-based content summarization and meeting minute generation
  - Time-based content lookup (transcript segment at time T, frame at time T)

Design:
  - All heavy dependencies are lazy-imported
  - Falls back gracefully when dependencies are missing
  - Models downloaded on first use (not bundled); CrispASR binary is not bundled
  - Uses Metal GPU acceleration on Apple Silicon
"""
from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from utils.logger import get_child_logger
    logger = get_child_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

# ── File Extension Sets ───────────────────────────────────────────────────────
AUDIO_EXTENSIONS = frozenset({
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape",
})
VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".m4v", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".ts",
})
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ASRSegment:
    """A single timestamped segment from ASR transcription."""
    start: float           # seconds
    end: float             # seconds
    text: str              # transcribed text
    speaker: str = ""      # speaker label e.g. "SPEAKER_00" (populated by diarization)

    def contains_time(self, t: float, margin: float = 2.0) -> bool:
        """Check if time t falls within this segment (with margin)."""
        return (self.start - margin) <= t <= (self.end + margin)


@dataclass
class KeyframeInfo:
    """A single extracted video keyframe with VL description."""
    time_sec: float
    description: str = ""
    ocr_text: str = ""
    frame_path: str = ""   # path to cached frame image (optional)


@dataclass
class MediaResult:
    """Result of processing an audio or video file."""
    file_path: str
    media_type: str                                # "audio" | "video"
    duration_sec: float = 0.0
    transcript: str = ""                           # full ASR transcript
    transcript_summary: str = ""                   # LLM-generated summary
    asr_segments: List[ASRSegment] = field(default_factory=list)
    keyframes: List[KeyframeInfo] = field(default_factory=list)
    video_metadata: dict = field(default_factory=dict)  # device, GPS, creation_time, etc.

    def get_segments_at(self, time_sec: float, margin: float = 5.0) -> List[ASRSegment]:
        """Return ASR segments that overlap with [time_sec - margin, time_sec + margin]."""
        return [s for s in self.asr_segments if s.contains_time(time_sec, margin)]

    def get_transcript_at(self, time_sec: float, margin: float = 5.0) -> str:
        """Return transcript text around a given timestamp."""
        segs = self.get_segments_at(time_sec, margin)
        if not segs:
            # Widen search
            segs = self.get_segments_at(time_sec, margin=15.0)
        if not segs:
            return ""
        return " ".join(s.text.strip() for s in segs)

    def get_nearest_keyframe(self, time_sec: float) -> Optional[KeyframeInfo]:
        """Return the keyframe closest to the given timestamp."""
        if not self.keyframes:
            return None
        return min(self.keyframes, key=lambda kf: abs(kf.time_sec - time_sec))


# ── MediaExpert Engine ────────────────────────────────────────────────────────

class MediaExpert:
    """
    Unified audio/video processing expert.

    Usage:
        expert = MediaExpert(llm_client=client)
        result = expert.process_audio("/path/to/audio.mp3")
        result = expert.process_video("/path/to/video.mp4")

        # Time-based lookup
        text = result.get_transcript_at(30.0)
        kf = result.get_nearest_keyframe(60.0)
    """

    # ASR model configuration
    # pywhispercpp engine: uses whisper.cpp local models (no PyTorch / onnxruntime)
    # Model name maps to ggerganov/whisper.cpp HuggingFace repo filenames.
    # "large-v3-turbo-q5_0" → ggml-large-v3-turbo-q5_0.bin (547MB, high-accuracy mode)
    # "base"           → ggml-base.bin (142MB, fast fallback)
    # Default to large-v3-turbo-q5_0 for best observed Chinese/English accuracy.
    DEFAULT_ASR_MODEL = os.environ.get("MEDIA_ASR_MODEL", "large-v3-turbo-q5_0")
    DEFAULT_ASR_DEVICE = os.environ.get("MEDIA_ASR_DEVICE", "auto")
    DEFAULT_ASR_COMPUTE = os.environ.get("MEDIA_ASR_COMPUTE", "auto")  # auto: CUDA→float16, CPU→int8
    DEFAULT_ASR_THREADS = int(os.environ.get("MEDIA_ASR_THREADS", "2"))

    # pywhispercpp model storage — alongside other local models
    _WHISPER_MODEL_DIR = os.path.join(
        os.path.expanduser("~"), "Library", "Application Support", "Unfoldly", "whisper_models"
    )
    # Map friendly name → (HF repo file basename, URL)
    _GGML_MODEL_MAP: Dict[str, Tuple[str, str]] = {
        "large-v3-turbo": (
            "ggml-large-v3-turbo-q5_0.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        ),
        "large-v3-turbo-q5_0": (
            "ggml-large-v3-turbo-q5_0.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        ),
        "base-q5": (
            "ggml-base-q5_1.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin",
        ),
        "base-q5_1": (
            "ggml-base-q5_1.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin",
        ),
        "base": (
            "ggml-base.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        ),
        "small-q5": (
            "ggml-small-q5_1.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin",
        ),
        "small-q5_1": (
            "ggml-small-q5_1.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin",
        ),
        "small-q8": (
            "ggml-small-q8_0.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin",
        ),
        "small-q8_0": (
            "ggml-small-q8_0.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin",
        ),
        "small": (
            "ggml-small.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        ),
        "medium-q4": (
            "ggml-medium-q4_k.bin",
            "https://huggingface.co/Pomni/whisper-medium-ggml-allquants/resolve/main/ggml-medium-q4_k.bin",
        ),
        "medium-q4_k": (
            "ggml-medium-q4_k.bin",
            "https://huggingface.co/Pomni/whisper-medium-ggml-allquants/resolve/main/ggml-medium-q4_k.bin",
        ),
        "medium-q4_0": (
            "ggml-medium-q4_0.bin",
            "https://huggingface.co/Pomni/whisper-medium-ggml-allquants/resolve/main/ggml-medium-q4_0.bin",
        ),
        "medium-q5_0": (
            "ggml-medium-q5_0.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin",
        ),
        "medium": (
            "ggml-medium-q5_0.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin",
        ),
        "large-v3": (
            "ggml-large-v3-q5_0.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin",
        ),
    }
    _GGML_MODEL_SIZE_HINTS: Dict[str, int] = {
        # Conservative byte-size hints used for progress and partial-file detection.
        "large-v3-turbo": 547 * 1024 * 1024,
        "large-v3-turbo-q5_0": 547 * 1024 * 1024,
        "base-q5": 57 * 1024 * 1024,
        "base-q5_1": 57 * 1024 * 1024,
        "base": 142 * 1024 * 1024,
        "small-q5": 182 * 1024 * 1024,
        "small-q5_1": 182 * 1024 * 1024,
        "small-q8": 253 * 1024 * 1024,
        "small-q8_0": 253 * 1024 * 1024,
        "small": 465 * 1024 * 1024,
        "medium-q4": 424 * 1024 * 1024,
        "medium-q4_k": 424 * 1024 * 1024,
        "medium-q4_0": 424 * 1024 * 1024,
        "medium-q5_0": 515 * 1024 * 1024,
        "medium": 515 * 1024 * 1024,
        "large-v3": 1500 * 1024 * 1024,
    }

    # Keyframe extraction configuration
    DEFAULT_KEYFRAME_INTERVAL = int(os.environ.get("MEDIA_KEYFRAME_INTERVAL", "30"))
    DEFAULT_MAX_KEYFRAMES = int(os.environ.get("MEDIA_MAX_KEYFRAMES", "20"))
    DEFAULT_MAX_KEYFRAMES_WITH_ASR = int(os.environ.get("MEDIA_MAX_KEYFRAMES_WITH_ASR", "6"))
    DEFAULT_MAX_KEYFRAMES_NO_ASR = int(os.environ.get("MEDIA_MAX_KEYFRAMES_NO_ASR", "18"))
    DEFAULT_MAX_KEYFRAMES_SCREEN_NO_ASR = int(os.environ.get("MEDIA_MAX_KEYFRAMES_SCREEN_NO_ASR", "18"))

    # Transcript chunk size for indexing
    CHUNK_DURATION_SEC = 60  # 1 minute per chunk

    # ── CUDA / device detection (class-level cache) ──────────────────────
    _fw_device: Optional[str] = None       # faster-whisper device ("cuda" | "cpu")
    _fw_compute_type: Optional[str] = None # faster-whisper compute type

    # whisper.cpp owns native GPU state; concurrent init/transcribe can abort
    # the whole app. Share one Metal-backed context and serialize ASR calls.
    _whisper_model_lock = threading.RLock()
    _whisper_transcribe_lock = threading.Lock()
    _whisper_download_lock = threading.Lock()
    _shared_whisper_model: Optional[Any] = None
    _shared_whisper_model_key: Optional[Tuple[str, int]] = None
    _ffmpeg_path_cache: Optional[str] = None
    _ffprobe_path_cache: Optional[str] = None

    _PLACEHOLDER_ASR_PATTERNS = (
        re.compile(r'^\(?speaking in (?:a )?foreign language\)?[.!。 ]*$', re.IGNORECASE),
        re.compile(r'^\(?foreign language\)?[.!。 ]*$', re.IGNORECASE),
    )
    _LOW_SIGNAL_ASR_PHRASES = {
        "silence",
        "silent",
        "no speech",
        "no speech detected",
        "no spoken words",
        "blank audio",
        "empty audio",
        "empty transcript",
        "静音",
        "无语音",
        "没有语音",
        "未检测到语音",
        "空白音频",
    }
    _LOW_SIGNAL_ASR_TOKEN_RE = re.compile(
        r'[\[\(<]?\s*(?:silence|silent|no speech(?: detected)?|no spoken words?|blank audio|empty audio|empty transcript|静音|无语音|没有语音|未检测到语音|空白音频)\s*[\]\)>]?',
        re.IGNORECASE,
    )
    _LOW_SIGNAL_SUMMARY_MARKERS = (
        "speech-to-text transcript, which consists only of silence",
        "transcript consists only of silence",
        "transcript contains only silence",
        "contains only silence",
        "consists only of silence",
        "only silence",
        "blank audio",
        "only blank audio",
        "silent audio",
        "empty transcript",
        "no speech",
        "no spoken",
        "no dialogue",
        "no discernible content",
        "no content to summarize",
        "no content to index",
        "no indexable",
        "cannot generate the requested structured meeting minutes",
        "无法生成摘要",
        "没有可总结",
        "无可索引",
        "未检测到语音",
        "空白音频",
        "静音",
    )

    @classmethod
    def _detect_fw_device(cls) -> Tuple[str, str]:
        """
        Auto-detect the best device + compute_type for faster-whisper / CTranslate2.
        Priority: CUDA → CPU.  Result is cached at class level.
        Returns (device, compute_type).
        """
        if cls._fw_device is not None:
            return cls._fw_device, cls._fw_compute_type  # type: ignore[return-value]

        # Honour explicit env overrides
        env_dev = os.environ.get("MEDIA_ASR_DEVICE", "auto").strip().lower()
        env_ctype = os.environ.get("MEDIA_ASR_COMPUTE", "auto").strip().lower()

        if env_dev != "auto":
            cls._fw_device = env_dev
            cls._fw_compute_type = env_ctype if env_ctype != "auto" else (
                "float16" if env_dev == "cuda" else "int8"
            )
            return cls._fw_device, cls._fw_compute_type  # type: ignore[return-value]

        device = "cpu"
        compute_type = "int8"
        try:
            import ctranslate2
            n_cuda = ctranslate2.get_cuda_device_count()
            if n_cuda > 0:
                device = "cuda"
                # float16 on CUDA is fastest; int8_float16 for large models
                compute_type = "float16"
                logger.info(f"[MediaExpert] CUDA detected ({n_cuda} device(s)) — using CUDA float16 for ASR")
            else:
                logger.info("[MediaExpert] No CUDA — using CPU int8 for ASR")
        except Exception as _e:
            logger.debug(f"[MediaExpert] CUDA detection failed ({_e}), defaulting to CPU")

        if env_ctype != "auto":
            compute_type = env_ctype

        cls._fw_device = device
        cls._fw_compute_type = compute_type
        return device, compute_type

    def __init__(
        self,
        llm_client: Any = None,
        vl_describe_fn: Optional[Callable] = None,
        frame_ocr_fn: Optional[Callable[[str], str]] = None,
        asr_model: Optional[str] = None,
        on_frame_progress: Optional[Callable[[int, int], None]] = None,
        on_media_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Args:
            llm_client: OpenAI-compatible client for LLM summary generation
            vl_describe_fn: Optional function(image_path) -> str for VL frame description
            frame_ocr_fn: Optional function(image_path) -> str for lightweight OCR on selected frames
            asr_model: Whisper model key (for example "small-q8_0", "medium-q4_k",
                       "large-v3-turbo-q5_0"). Defaults to DEFAULT_ASR_MODEL
                       (MEDIA_ASR_MODEL env var to override).
            on_frame_progress: Optional callback(current_frame, total_frames) called before
                               each keyframe VL analysis to report sub-file progress
            on_media_progress: Optional callback({"stage": ..., ...}) for richer
                               sub-file progress such as ASR/audio seconds and
                               keyframe throughput.
        """
        self._llm_client = llm_client
        self._vl_describe_fn = vl_describe_fn
        self._frame_ocr_fn = frame_ocr_fn
        self._asr_model_name = asr_model or self.DEFAULT_ASR_MODEL
        self._on_frame_progress = on_frame_progress
        self._on_media_progress = on_media_progress

    def _emit_progress(self, stage: str, **payload: Any) -> None:
        cb = self._on_media_progress
        if not cb:
            return
        try:
            cb({"stage": stage, **payload})
        except Exception:
            pass

    @staticmethod
    def _env_enabled(name: str, default: str = "0") -> bool:
        return str(os.environ.get(name, default)).strip().lower() in {
            "1", "true", "yes", "on"
        }

    @staticmethod
    def _filename_looks_like_screen_recording(file_path: str) -> bool:
        try:
            name = os.path.basename(str(file_path or "")).strip().lower()
            compact = re.sub(r"[\s_.-]+", "", name)
            hints = (
                "screenrecording",
                "screenrecord",
                "screencapture",
                "录屏",
                "屏幕录制",
                "屏幕录像",
            )
            return any(hint in compact for hint in hints)
        except Exception:
            return False

    def _get_index_model_id(self) -> str:
        """Resolve the exact model selected by the indexing UI."""
        try:
            from services.preference_manager import PreferenceManager
            import config.settings as agent_settings

            base_dir = getattr(
                agent_settings,
                "BASE_DIR",
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            )
            pm = PreferenceManager(base_dir)
            mid = (pm.get_selected_index_model_id() or "").strip()
            if mid:
                return mid
        except Exception:
            pass
        logger.warning("[MediaExpert] Add Sources index model is not configured; skipping index-side LLM/VL call")
        return ""

    # ── Public API ────────────────────────────────────────────────────────

    def process_audio(self, file_path: str) -> MediaResult:
        """
        Process an audio file: ASR transcription + LLM summary.

        Returns MediaResult with transcript, segments, and summary.
        """
        logger.info(f"[MediaExpert] Processing audio: {os.path.basename(file_path)}")
        result = MediaResult(file_path=file_path, media_type="audio")

        # 1. Get duration + metadata (author, source, recording type)
        result.duration_sec = self._get_media_duration(file_path)
        result.video_metadata = self._extract_video_metadata(file_path)
        logger.info(f"[MediaExpert] Audio metadata: {result.video_metadata}")

        # 2. ASR transcription (with optional speaker diarization when configured)
        try:
            self._emit_progress(
                "transcribing_audio",
                current_audio_sec=0.0,
                total_audio_sec=round(float(result.duration_sec or 0.0), 2),
                stage_rate=0.0,
            )
            segments = self.transcribe_with_diarization(file_path)
            result.asr_segments = self._meaningful_asr_segments(segments)
            result.transcript = " ".join(s.text.strip() for s in result.asr_segments)
            if segments and not result.asr_segments:
                logger.info("[MediaExpert] Audio ASR contained only low-signal placeholders; indexing as metadata-only")
            logger.info(
                f"[MediaExpert] ASR complete: {len(segments)} segments, "
                f"{len(result.transcript)} chars, duration={result.duration_sec:.1f}s"
            )
        except Exception as e:
            logger.error(f"[MediaExpert] ASR failed for {os.path.basename(file_path)}: {e}", exc_info=True)
            result.transcript = ""

        # 3. LLM summary of transcript
        if result.transcript and self._llm_client:
            try:
                result.transcript_summary = self._generate_transcript_summary(
                    result.transcript, os.path.basename(file_path)
                )
            except Exception as e:
                logger.error(f"[MediaExpert] Summary failed: {e}", exc_info=True)

        return result

    def process_video(self, file_path: str) -> MediaResult:
        """
        Process a video file:
          1. Extract audio track → ASR transcription + summary
          2. Extract keyframes → VL description for each

        Returns MediaResult with transcript, segments, summary, and keyframes.
        """
        logger.info(f"[MediaExpert] Processing video: {os.path.basename(file_path)}")
        result = MediaResult(file_path=file_path, media_type="video")

        # 1. Get duration + rich file metadata
        result.duration_sec = self._get_media_duration(file_path)
        result.video_metadata = self._extract_video_metadata(file_path)
        logger.info(f"[MediaExpert] Video metadata: {result.video_metadata}")
        has_audio_track = bool(result.video_metadata.get("has_audio"))
        recording_type = str(result.video_metadata.get("recording_type") or "").strip().lower()
        should_run_asr = self._should_run_video_asr(
            has_audio_track=has_audio_track,
            recording_type=recording_type,
        )

        # 2. Extract audio track and run ASR (with speaker diarization when available)
        tmp_wav = None
        if should_run_asr:
            try:
                tmp_wav = self._extract_audio_track(file_path)
                if tmp_wav:
                    self._emit_progress(
                        "transcribing_audio",
                        current_audio_sec=0.0,
                        total_audio_sec=round(float(result.duration_sec or 0.0), 2),
                        stage_rate=0.0,
                    )
                    segments = self.transcribe_with_diarization(tmp_wav)
                    result.asr_segments = self._meaningful_asr_segments(segments)
                    result.transcript = " ".join(s.text.strip() for s in result.asr_segments)
                    if segments and not result.asr_segments:
                        logger.info(
                            "[MediaExpert] Video ASR contained only low-signal placeholders; "
                            "using visual-first indexing"
                        )
                    logger.info(
                        f"[MediaExpert] Video ASR complete: {len(segments)} segments, "
                        f"{len(result.transcript)} chars"
                    )
            except Exception as e:
                logger.error(f"[MediaExpert] Video audio extraction/ASR failed: {e}", exc_info=True)
            finally:
                if tmp_wav and os.path.exists(tmp_wav):
                    try:
                        os.unlink(tmp_wav)
                    except OSError:
                        pass
        else:
            logger.info(
                f"[MediaExpert] Skip video ASR: has_audio={has_audio_track} "
                f"recording_type={recording_type or 'unknown'} file={os.path.basename(file_path)}"
            )

        # 3. LLM summary of transcript
        if result.transcript and self._llm_client:
            try:
                result.transcript_summary = self._generate_transcript_summary(
                    result.transcript, os.path.basename(file_path)
                )
            except Exception as e:
                logger.error(f"[MediaExpert] Video summary failed: {e}", exc_info=True)

        # 4. Extract keyframes and describe with VL
        try:
            has_meaningful_asr = self._has_meaningful_asr_transcript(result.transcript)
            keyframes = self._extract_keyframes(
                file_path,
                has_audio_track=has_audio_track,
                has_asr_transcript=has_meaningful_asr,
                recording_type=recording_type,
            )
            run_frame_ocr = self._should_run_frame_ocr(
                has_audio_track=has_audio_track,
                has_asr_transcript=has_meaningful_asr,
                recording_type=recording_type,
            )
            logger.info(
                f"[MediaExpert] Frame OCR plan: enabled={bool(run_frame_ocr)} "
                f"recording_type={recording_type or 'unknown'} "
                f"has_audio={has_audio_track} has_asr={has_meaningful_asr}"
            )
            ocr_indices = self._select_keyframe_indices_for_ocr(len(keyframes)) if run_frame_ocr else set()
            if keyframes and self._vl_describe_fn:
                import inspect
                _vl_accepts_prev = "prev_description" in inspect.signature(self._vl_describe_fn).parameters
                prev_desc = ""
                total_kf = len(keyframes)
                frame_started_at = time.time()
                self._emit_progress(
                    "analyzing_frames",
                    current_frame=0,
                    total_frames=total_kf,
                    stage_rate=0.0,
                )
                for kf_idx, kf in enumerate(keyframes, 1):
                    if self._on_frame_progress:
                        try:
                            self._on_frame_progress(kf_idx, total_kf)
                        except Exception:
                            pass
                    elapsed_frames = max(0.001, time.time() - frame_started_at)
                    self._emit_progress(
                        "analyzing_frames",
                        current_frame=kf_idx,
                        total_frames=total_kf,
                        stage_rate=round(kf_idx / elapsed_frames, 2),
                    )
                    if kf.frame_path and os.path.exists(kf.frame_path):
                        try:
                            if _vl_accepts_prev:
                                kf.description = self._vl_describe_fn(kf.frame_path, prev_description=prev_desc)
                            else:
                                kf.description = self._vl_describe_fn(kf.frame_path)
                            if kf.description:
                                prev_desc = kf.description
                        except Exception as e:
                            logger.warning(f"[MediaExpert] VL describe failed at {kf.time_sec}s: {e}")
                    if (kf_idx - 1) in ocr_indices and self._frame_ocr_fn and kf.frame_path and os.path.exists(kf.frame_path):
                        try:
                            kf.ocr_text = self._frame_ocr_fn(kf.frame_path) or ""
                        except Exception as e:
                            logger.warning(f"[MediaExpert] Frame OCR failed at {kf.time_sec}s: {e}")
            elif keyframes and self._llm_client:
                # Fallback: use LLM client with image if it supports VL
                total_kf = len(keyframes)
                frame_started_at = time.time()
                self._emit_progress(
                    "analyzing_frames",
                    current_frame=0,
                    total_frames=total_kf,
                    stage_rate=0.0,
                )
                for kf_idx, kf in enumerate(keyframes, 1):
                    if self._on_frame_progress:
                        try:
                            self._on_frame_progress(kf_idx, total_kf)
                        except Exception:
                            pass
                    elapsed_frames = max(0.001, time.time() - frame_started_at)
                    self._emit_progress(
                        "analyzing_frames",
                        current_frame=kf_idx,
                        total_frames=total_kf,
                        stage_rate=round(kf_idx / elapsed_frames, 2),
                    )
                    if kf.frame_path and os.path.exists(kf.frame_path):
                        try:
                            kf.description = self._describe_frame_with_vl(kf.frame_path)
                        except Exception as e:
                            logger.warning(f"[MediaExpert] VL frame describe failed at {kf.time_sec}s: {e}")
                    if (kf_idx - 1) in ocr_indices and self._frame_ocr_fn and kf.frame_path and os.path.exists(kf.frame_path):
                        try:
                            kf.ocr_text = self._frame_ocr_fn(kf.frame_path) or ""
                        except Exception as e:
                            logger.warning(f"[MediaExpert] Frame OCR failed at {kf.time_sec}s: {e}")
            result.keyframes = keyframes
            logger.info(f"[MediaExpert] Extracted {len(keyframes)} keyframes")
        except Exception as e:
            logger.error(f"[MediaExpert] Keyframe extraction failed: {e}", exc_info=True)

        return result

    def extract_frame_at(self, video_path: str, time_sec: float) -> Optional[str]:
        """
        Extract a single frame from a video at the given timestamp.
        Returns the path to the saved frame image, or None on failure.
        Uses ffmpeg by default. Optional PyAV support is only used when
        UNFOLDLY_ENABLE_PYAV=1 is explicitly set.
        """
        if self._check_av():
            frame_path = self._extract_frame_pyav(video_path, time_sec)
            if frame_path:
                return frame_path
        ffmpeg = self._ffmpeg_cmd()
        if not ffmpeg:
            return None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="media_frame_")
            frame_path = os.path.join(tmp_dir, f"frame_{time_sec:.1f}.jpg")
            cmd = [
                ffmpeg, "-ss", str(time_sec),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                frame_path, "-y",
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                return frame_path
        except Exception as e:
            logger.error(f"[MediaExpert] Frame extraction at {time_sec}s failed: {e}")
        return None

    def extract_frames_around(self, video_path: str, time_sec: float, window_sec: float = 2.0, count: int = 3) -> List[str]:
        """
        Extract a sequence of frames around a target timestamp.
        Returns a list of paths to extracted images.
        """
        if count <= 1:
            p = self.extract_frame_at(video_path, time_sec)
            return [p] if p else []

        start = max(0, time_sec - window_sec / 2)
        step = window_sec / (count - 1) if count > 1 else 0
        paths = []
        for i in range(count):
            t = start + i * step
            p = self.extract_frame_at(video_path, t)
            if p:
                paths.append(p)
        return paths

    @staticmethod
    def _extract_frame_pyav(video_path: str, time_sec: float) -> Optional[str]:
        """Extract a single video frame with optional PyAV."""
        try:
            import av as _av
            tmp_dir = tempfile.mkdtemp(prefix="media_frame_")
            frame_path = os.path.join(tmp_dir, f"frame_{time_sec:.1f}.jpg")
            with _av.open(video_path) as container:
                stream = container.streams.video[0]
                # Seek to target time
                ts = int(time_sec / stream.time_base)
                container.seek(ts, stream=stream)
                for frame in container.decode(stream):
                    img = frame.to_image()
                    img.save(frame_path, "JPEG", quality=90)
                    break
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                return frame_path
        except Exception as e:
            logger.debug(f"[MediaExpert] PyAV frame extract at {time_sec}s failed: {e}")
        return None

    def describe_frame_at(self, video_path: str, time_sec: float) -> str:
        """
        Extract a frame at the given time and describe it using VL model.
        Returns the VL description string.
        """
        frame_path = self.extract_frame_at(video_path, time_sec)
        if not frame_path:
            return f"Unable to extract frame at {time_sec}s."
        try:
            if self._vl_describe_fn:
                return self._vl_describe_fn(frame_path)
            return self._describe_frame_with_vl(frame_path)
        except Exception as e:
            logger.error(f"[MediaExpert] VL describe at {time_sec}s failed: {e}")
            return f"Frame extracted at {time_sec}s but VL description failed."
        finally:
            # Cleanup
            try:
                os.unlink(frame_path)
                os.rmdir(os.path.dirname(frame_path))
            except OSError:
                pass

    # ── ASR Engine ────────────────────────────────────────────────────────

    @classmethod
    def _resolve_ggml_model_spec(cls, model_name: Optional[str] = None) -> Dict[str, Any]:
        name = model_name or cls.DEFAULT_ASR_MODEL
        if name not in cls._GGML_MODEL_MAP:
            if os.path.isfile(name):
                path = os.path.abspath(name)
                return {
                    "name": os.path.basename(path),
                    "filename": os.path.basename(path),
                    "url": "",
                    "path": path,
                    "tmp_path": f"{path}.tmp",
                    "expected_size": max(0, os.path.getsize(path)),
                    "direct_path": True,
                }
            logger.warning(f"[MediaExpert] Unknown GGML model name '{name}', falling back to 'base'")
            name = "base"

        filename, url = cls._GGML_MODEL_MAP[name]
        path = os.path.join(cls._WHISPER_MODEL_DIR, filename)
        persisted_size = cls._read_ggml_size_hint(path)
        expected_size = persisted_size if persisted_size > 0 else int(cls._GGML_MODEL_SIZE_HINTS.get(name) or 0)
        return {
            "name": name,
            "filename": filename,
            "url": url,
            "path": path,
            "tmp_path": f"{path}.tmp",
            "expected_size": expected_size,
            "direct_path": False,
        }

    @staticmethod
    def _safe_size(path: str) -> int:
        try:
            return int(os.path.getsize(path)) if os.path.exists(path) else 0
        except Exception:
            return 0

    @staticmethod
    def _ggml_size_hint_path(path: str) -> str:
        return f"{path}.size.json"

    @classmethod
    def _read_ggml_size_hint(cls, path: str) -> int:
        try:
            import json
            hp = cls._ggml_size_hint_path(path)
            if not os.path.isfile(hp):
                return 0
            with open(hp, "r", encoding="utf-8") as f:
                data = json.load(f)
            size = int(data.get("size_bytes") or 0)
            return size if size > 0 else 0
        except Exception:
            return 0

    @classmethod
    def _write_ggml_size_hint(cls, path: str, size_bytes: int) -> None:
        try:
            import json
            size = int(size_bytes or 0)
            if size <= 0:
                return
            hp = cls._ggml_size_hint_path(path)
            os.makedirs(os.path.dirname(hp), exist_ok=True)
            tmp = f"{hp}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"size_bytes": size}, f, ensure_ascii=False)
            os.replace(tmp, hp)
        except Exception:
            pass

    @classmethod
    def _is_ggml_model_complete(cls, path: str, expected_size: int = 0) -> bool:
        try:
            if not os.path.isfile(path):
                return False
            size = os.path.getsize(path)
            if size < 10 * 1024 * 1024:
                return False
            if expected_size > 0 and size < int(expected_size * 0.98):
                return False
            with open(path, "rb") as f:
                magic = f.read(4)
            # whisper.cpp legacy .bin files commonly expose the ggml magic as lmgg
            # on disk; newer GGUF files start with GGUF.
            if magic and magic not in {b"lmgg", b"ggml", b"GGUF"}:
                logger.warning(f"[MediaExpert] Whisper model has unexpected header {magic!r}: {os.path.basename(path)}")
                return False
            return True
        except Exception:
            return False

    @classmethod
    def get_ggml_model_status(cls, model_name: Optional[str] = None) -> Dict[str, Any]:
        spec = cls._resolve_ggml_model_spec(model_name)
        path = str(spec["path"])
        tmp_path = str(spec["tmp_path"])
        expected = int(spec.get("expected_size") or 0)
        installed = cls._is_ggml_model_complete(path, expected)
        final_size = cls._safe_size(path)
        tmp_size = 0 if installed else cls._safe_size(tmp_path)
        if not installed and tmp_size > 0 and cls._is_ggml_model_complete(tmp_path, expected):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                os.replace(tmp_path, path)
                installed = cls._is_ggml_model_complete(path, expected)
                final_size = cls._safe_size(path)
                tmp_size = 0 if installed else cls._safe_size(tmp_path)
                if installed and final_size > 0:
                    cls._write_ggml_size_hint(path, final_size)
            except Exception as e:
                logger.warning(f"[MediaExpert] Could not promote completed Whisper temp file: {e}")
        if installed:
            if final_size > 0:
                cls._write_ggml_size_hint(path, final_size)
            downloaded = final_size if final_size > 0 else expected
            total = downloaded
        else:
            downloaded = max(final_size, tmp_size)
            total = max(expected, downloaded)
        percent = 100.0 if installed else ((downloaded / total) * 100.0 if total > 0 else 0.0)
        return {
            "installed": bool(installed),
            "status": "installed" if installed else "idle",
            "model_name": spec.get("name"),
            "filename": spec.get("filename"),
            "path": path,
            "tmp_path": tmp_path,
            "downloaded_bytes": int(downloaded),
            "total_bytes": int(total),
            "percent": round(max(0.0, min(100.0, percent)), 1),
        }

    @staticmethod
    def _remote_content_length(url: str) -> int:
        if not url:
            return 0
        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                method="HEAD",
                headers={"User-Agent": "Unfoldly/1.0", "Accept-Encoding": "identity"},
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                return int(resp.headers.get("Content-Length") or 0)
        except Exception:
            return 0

    @staticmethod
    def _content_range_total(value: str) -> int:
        try:
            # Example: bytes 123-456/789
            if "/" not in value:
                return 0
            total = value.rsplit("/", 1)[1].strip()
            return int(total) if total and total != "*" else 0
        except Exception:
            return 0

    @classmethod
    def _http_error_content_range_total(cls, error: Exception) -> int:
        try:
            headers = getattr(error, "headers", None) or getattr(error, "hdrs", None)
            content_range = headers.get("Content-Range") if headers is not None else ""
            return cls._content_range_total(content_range or "")
        except Exception:
            return 0

    @classmethod
    def ensure_ggml_model_downloaded(
        cls,
        model_name: Optional[str] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Optional[str]:
        """
        Ensure the whisper.cpp model file is present locally.
        Downloads are resumable via the .tmp file used by onboarding.
        """
        spec = cls._resolve_ggml_model_spec(model_name)
        path = str(spec["path"])
        tmp_path = str(spec["tmp_path"])
        url = str(spec.get("url") or "")
        expected = int(spec.get("expected_size") or 0)

        if spec.get("direct_path"):
            return path if cls._is_ggml_model_complete(path, expected) else None

        def _cancelled() -> bool:
            if should_cancel is None:
                return False
            try:
                return bool(should_cancel())
            except Exception:
                return False

        def _emit(downloaded: int, total: int, speed: float = 0.0, force: bool = False) -> None:
            if on_progress is None:
                return
            now = time.time()
            if not force and now - _emit.last_ts < 0.5:
                return
            _emit.last_ts = now
            pct = (downloaded / total) * 100.0 if total > 0 else 0.0
            eta = ((total - downloaded) / speed) if speed > 0 and total > downloaded else 0.0
            on_progress({
                "percent": max(0.0, min(100.0, pct)),
                "downloaded_bytes": int(max(0, downloaded)),
                "total_bytes": int(max(0, total)),
                "speed_bytes_per_sec": float(max(0.0, speed)),
                "eta_seconds": float(max(0.0, eta)),
            })
        _emit.last_ts = 0.0  # type: ignore[attr-defined]

        with cls._whisper_download_lock:
            if cls._is_ggml_model_complete(path, expected):
                size = cls._safe_size(path)
                cls._write_ggml_size_hint(path, size)
                _emit(size, max(expected, size), 0.0, force=True)
                logger.info(f"[MediaExpert] Using cached GGML model: {path} ({size//1024//1024}MB)")
                return path

            os.makedirs(cls._WHISPER_MODEL_DIR, exist_ok=True)

            final_size = cls._safe_size(path)
            if final_size > 0 and not cls._is_ggml_model_complete(path, expected):
                tmp_size = cls._safe_size(tmp_path)
                if tmp_size <= 0 or final_size > tmp_size:
                    try:
                        os.replace(path, tmp_path)
                    except Exception:
                        pass

            remote_size = cls._remote_content_length(url)
            total_size = remote_size if remote_size > 0 else max(expected, cls._safe_size(tmp_path), cls._safe_size(path))
            downloaded = cls._safe_size(tmp_path)

            def _promote_tmp_if_complete(complete_size: int, reason: str, *, exact_size: bool) -> Optional[str]:
                complete_size = int(complete_size or 0)
                local_size = cls._safe_size(tmp_path)
                if local_size <= 0 or complete_size <= 0:
                    return None
                if exact_size and local_size > max(complete_size + 1024 * 1024, int(complete_size * 1.02)):
                    logger.warning(
                        f"[MediaExpert] Whisper temp file is larger than server size; restarting: "
                        f"local={local_size}, remote={complete_size}"
                    )
                    return None
                if not cls._is_ggml_model_complete(tmp_path, complete_size):
                    return None
                os.replace(tmp_path, path)
                final = cls._safe_size(path)
                cls._write_ggml_size_hint(path, final)
                _emit(final, max(complete_size, final), 0.0, force=True)
                logger.info(f"[MediaExpert] GGML model {reason}: {path} ({final//1024//1024}MB)")
                return path

            promoted = _promote_tmp_if_complete(
                total_size,
                "completed from temp file",
                exact_size=remote_size > 0,
            )
            if promoted:
                return promoted

            if remote_size > 0 and downloaded >= remote_size:
                # Local partial claims to be at/after EOF but failed validation. A
                # ranged request would return 416, so discard it and restart cleanly.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                downloaded = 0
            _emit(downloaded, total_size, 0.0, force=True)
            if _cancelled():
                raise RuntimeError("asr_model_download_cancelled")

            logger.info(f"[MediaExpert] Downloading GGML Whisper model '{spec.get('name')}' from {url}")
            import urllib.request

            resume_allowed = downloaded > 0
            for attempt in range(1, 3):
                if _cancelled():
                    raise RuntimeError("asr_model_download_cancelled")

                headers = {"User-Agent": "Unfoldly/1.0", "Accept-Encoding": "identity"}
                if resume_allowed and downloaded > 0:
                    headers["Range"] = f"bytes={downloaded}-"
                req = urllib.request.Request(url, headers=headers)

                start_offset = downloaded
                start_ts = time.time()
                last_log_pct = -1
                response_total_exact = False
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        code = getattr(resp, "status", None) or resp.getcode()
                        if resume_allowed and downloaded > 0 and code != 206:
                            downloaded = 0
                            mode = "wb"
                        else:
                            mode = "ab" if resume_allowed and downloaded > 0 else "wb"
                        start_offset = downloaded
                        content_range_total = cls._content_range_total(resp.headers.get("Content-Range") or "")
                        content_length = int(resp.headers.get("Content-Length") or 0)
                        if content_range_total > 0:
                            total_size = max(downloaded, content_range_total)
                            response_total_exact = True
                        elif content_length > 0:
                            total_size = max(downloaded, downloaded + content_length)
                            response_total_exact = True

                        with open(tmp_path, mode) as f:
                            while True:
                                if _cancelled():
                                    raise RuntimeError("asr_model_download_cancelled")
                                chunk = resp.read(1024 * 1024)
                                if not chunk:
                                    break
                                f.write(chunk)
                                downloaded += len(chunk)
                                elapsed = max(0.001, time.time() - start_ts)
                                speed = max(0, downloaded - start_offset) / elapsed
                                _emit(downloaded, total_size, speed)
                                if total_size > 0:
                                    pct = int((downloaded / total_size) * 100)
                                    if pct >= last_log_pct + 10:
                                        last_log_pct = pct
                                        logger.info(f"[MediaExpert] Download progress: {min(100, pct)}%")

                    promoted = _promote_tmp_if_complete(
                        total_size,
                        "downloaded",
                        exact_size=response_total_exact,
                    )
                    if promoted:
                        return promoted
                    if not response_total_exact:
                        promoted = _promote_tmp_if_complete(
                            cls._safe_size(tmp_path),
                            "downloaded with unknown total",
                            exact_size=True,
                        )
                        if promoted:
                            return promoted
                    raise RuntimeError(
                        f"ASR model incomplete after download: {cls._safe_size(tmp_path)} / {total_size} bytes"
                    )
                except Exception as e:
                    if getattr(e, "code", None) == 416:
                        range_total = cls._http_error_content_range_total(e)
                        for candidate_total, exact in (
                            (range_total, True),
                            (remote_size, True),
                            (expected, False),
                        ):
                            promoted = _promote_tmp_if_complete(
                                candidate_total,
                                "completed after 416 response",
                                exact_size=exact,
                            )
                            if promoted:
                                return promoted

                        if attempt < 2:
                            logger.warning(
                                f"[MediaExpert] Server rejected resume range for Whisper model; "
                                f"restarting full download (local={cls._safe_size(tmp_path)}, "
                                f"server_total={range_total or remote_size or 0})"
                            )
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                            downloaded = 0
                            total_size = range_total or remote_size or expected
                            resume_allowed = False
                            _emit(0, total_size, 0.0, force=True)
                            continue

                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                    logger.error(f"[MediaExpert] Failed to download GGML model: {e}")
                    raise

    def _ensure_ggml_model(self, model_name: Optional[str] = None) -> Optional[str]:
        """
        Ensure the ggml model file is present locally. Downloads if missing.
        Returns the absolute path to the .bin file, or None on failure.
        """
        try:
            return type(self).ensure_ggml_model_downloaded(model_name or self._asr_model_name)
        except Exception:
            return None

    def _get_pywhispercpp_model(self):
        """Lazy-load the shared pywhispercpp (whisper.cpp GGUF / Metal) model."""
        try:
            from pywhispercpp.model import Model as WhisperCppModel
        except ImportError:
            raise ImportError(
                "pywhispercpp is required for ASR. "
                "Install with: GGML_METAL=1 pip install git+https://github.com/absadiki/pywhispercpp"
            )

        model_path = self._ensure_ggml_model()
        if not model_path:
            raise RuntimeError(
                f"[MediaExpert] GGML model '{self._asr_model_name}' not available and download failed."
            )

        n_threads = self.DEFAULT_ASR_THREADS
        model_key = (os.path.abspath(model_path), n_threads)
        cls = type(self)
        with cls._whisper_model_lock:
            if cls._shared_whisper_model is not None and cls._shared_whisper_model_key == model_key:
                return cls._shared_whisper_model

            logger.info(
                f"[MediaExpert] Loading whisper.cpp Metal model: {model_path} (n_threads={n_threads})"
            )
            t0 = time.time()
            model = WhisperCppModel(model_path, n_threads=n_threads)
            cls._shared_whisper_model = model
            cls._shared_whisper_model_key = model_key
            logger.info(f"[MediaExpert] whisper.cpp model loaded in {time.time() - t0:.1f}s")
            return model

    def _run_asr(self, audio_path: str) -> List[ASRSegment]:
        """
        Run ASR on an audio file, returning timestamped segments.
        Tries pywhispercpp (whisper.cpp GGUF, Metal GPU) first, falls back to faster-whisper.
        """
        # ── Primary: pywhispercpp (whisper.cpp + Metal GPU, no PyTorch) ──
        try:
            segments = self._run_asr_pywhispercpp(audio_path)
            if self._should_retry_with_zh_hint(segments):
                logger.info(
                    "[MediaExpert] ASR auto-detect returned placeholder foreign-language captions; "
                    "retrying whisper.cpp with zh language hint"
                )
                retry_segments = self._run_asr_pywhispercpp(audio_path, language_hint="zh")
                if retry_segments and not self._looks_like_placeholder_asr_result(retry_segments):
                    return retry_segments
            return segments
        except ImportError:
            logger.info("[MediaExpert] pywhispercpp not available, falling back to faster-whisper")
        except Exception as e:
            if self._should_retry_pywhisper_with_normalized_audio(e):
                normalized_audio = self._convert_audio_to_whisper_wav(audio_path)
                if normalized_audio:
                    try:
                        logger.info(
                            "[MediaExpert] Retrying pywhispercpp ASR with normalized 16k mono WAV input"
                        )
                        segments = self._run_asr_pywhispercpp(normalized_audio)
                        if self._should_retry_with_zh_hint(segments):
                            logger.info(
                                "[MediaExpert] Normalized ASR still produced placeholder captions; "
                                "retrying with zh language hint"
                            )
                            retry_segments = self._run_asr_pywhispercpp(normalized_audio, language_hint="zh")
                            if retry_segments and not self._looks_like_placeholder_asr_result(retry_segments):
                                return retry_segments
                        return segments
                    finally:
                        try:
                            os.unlink(normalized_audio)
                        except OSError:
                            pass
                logger.warning(
                    "[MediaExpert] Audio could not be normalized for whisper.cpp retry; "
                    "skipping ASR and keeping metadata-only media indexing"
                )
                return []
            logger.warning(f"[MediaExpert] pywhispercpp ASR failed ({e}), falling back to faster-whisper")

        # ── Fallback: faster-whisper ──
        return self._run_asr_faster_whisper(audio_path)

    @staticmethod
    def _should_retry_pywhisper_with_normalized_audio(error: Exception) -> bool:
        message = str(error).lower()
        retry_markers = (
            "16000 hz",
            "16000hz",
            "must be 16000",
            "ffmpeg is not installed",
            "not in path",
            "provide a wav file",
            "unsupported format",
            "unknown format",
            "invalid data found",
            "file does not start with riff",
        )
        return any(marker in message for marker in retry_markers)

    def _convert_audio_to_whisper_wav(self, audio_path: str) -> Optional[str]:
        """
        Convert standalone audio into a 16kHz mono WAV that whisper.cpp accepts.
        Prefer ffmpeg CLI because it is already used elsewhere in the media pipeline.
        """
        ffmpeg = self._ffmpeg_cmd()
        if not ffmpeg:
            logger.warning("[MediaExpert] ffmpeg unavailable — cannot normalize audio for whisper.cpp retry")
            return None

        tmp_wav = tempfile.mktemp(suffix=".wav", prefix="media_asr_")
        cmd = [
            ffmpeg, "-i", audio_path,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            tmp_wav, "-y",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            if os.path.exists(tmp_wav) and os.path.getsize(tmp_wav) > 0:
                logger.info(
                    f"[MediaExpert] Audio normalized for ASR via ffmpeg: {os.path.getsize(tmp_wav) / 1024:.1f}KB"
                )
                return tmp_wav
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"[MediaExpert] ffmpeg audio normalization failed: {e.stderr[:200] if e.stderr else ''}"
            )
        except subprocess.TimeoutExpired:
            logger.warning("[MediaExpert] ffmpeg audio normalization timed out (300s)")

        try:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)
        except OSError:
            pass
        return None

    @classmethod
    def _looks_like_placeholder_asr_text(cls, text: str) -> bool:
        normalized = re.sub(r'\s+', ' ', str(text or '')).strip().lower()
        if not normalized:
            return False
        return any(pat.fullmatch(normalized) for pat in cls._PLACEHOLDER_ASR_PATTERNS)

    @classmethod
    def _looks_like_placeholder_asr_result(cls, segments: List[ASRSegment]) -> bool:
        texts = [str(seg.text or "").strip() for seg in segments if str(seg.text or "").strip()]
        if not texts:
            return False
        placeholder_count = sum(1 for text in texts if cls._looks_like_placeholder_asr_text(text))
        return placeholder_count == len(texts)

    @classmethod
    def _looks_like_low_signal_asr_text(cls, text: str) -> bool:
        """Return True for ASR noise that should not count as real content."""
        normalized = re.sub(r'\s+', ' ', str(text or '')).strip().lower()
        if not normalized:
            return False
        if cls._looks_like_placeholder_asr_text(normalized):
            return True
        stripped = normalized.strip("[]()<> \t\r\n.!。！？?,，")
        if stripped in cls._LOW_SIGNAL_ASR_PHRASES:
            return True
        return bool(cls._LOW_SIGNAL_ASR_TOKEN_RE.fullmatch(normalized))

    @classmethod
    def _looks_like_low_signal_transcript(cls, text: str) -> bool:
        normalized = re.sub(r'\s+', ' ', str(text or '')).strip().lower()
        if not normalized:
            return False
        if cls._looks_like_low_signal_asr_text(normalized):
            return True
        residual = cls._LOW_SIGNAL_ASR_TOKEN_RE.sub(" ", normalized)
        residual = re.sub(r'[\s,.;:!?。！？\-_/|]+', '', residual)
        return not residual

    @classmethod
    def _looks_like_low_signal_summary_text(cls, text: str) -> bool:
        normalized = re.sub(r'\s+', ' ', str(text or '')).strip().lower()
        if not normalized:
            return False
        if cls._looks_like_low_signal_transcript(normalized):
            return True
        return any(marker in normalized for marker in cls._LOW_SIGNAL_SUMMARY_MARKERS)

    @classmethod
    def _has_meaningful_asr_transcript(cls, text: str) -> bool:
        return bool(str(text or "").strip()) and not cls._looks_like_low_signal_transcript(text)

    @classmethod
    def _meaningful_asr_segments(cls, segments: List[ASRSegment]) -> List[ASRSegment]:
        return [
            seg for seg in list(segments or [])
            if str(seg.text or "").strip()
            and not cls._looks_like_low_signal_asr_text(str(seg.text or ""))
        ]

    def _should_retry_with_zh_hint(self, segments: List[ASRSegment]) -> bool:
        explicit_lang = os.environ.get("MEDIA_ASR_LANG", "").strip()
        if explicit_lang:
            return False
        return self._looks_like_placeholder_asr_result(segments)

    @staticmethod
    def _env_truthy(name: str, default: str = "0") -> bool:
        return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _whispercpp_segment_end_sec(seg: Any) -> float:
        try:
            return float(getattr(seg, "t1", 0) or 0) / 100.0
        except Exception:
            return 0.0

    @classmethod
    def _whispercpp_raw_to_asr_segments(
        cls,
        raw_segments: List[Any],
        offset_sec: float = 0.0,
    ) -> List[ASRSegment]:
        segments: List[ASRSegment] = []
        for seg in raw_segments or []:
            text = seg.text.strip() if hasattr(seg, "text") else str(seg).strip()
            if not text:
                continue
            start = float(getattr(seg, "t0", 0) or 0) / 100.0
            end = cls._whispercpp_segment_end_sec(seg)
            segments.append(ASRSegment(
                start=round(offset_sec + start, 2),
                end=round(offset_sec + end, 2),
                text=text,
            ))
        return segments

    @classmethod
    def _should_expand_auto_language_tail(
        cls,
        raw_segments: List[Any],
        total_audio_sec: float,
    ) -> bool:
        if total_audio_sec < 8.0:
            return False
        end_sec = max((cls._whispercpp_segment_end_sec(seg) for seg in raw_segments or []), default=0.0)
        if end_sec <= 0.0:
            return False
        min_coverage = float(os.environ.get("MEDIA_ASR_AUTO_TAIL_MIN_COVERAGE", "0.80"))
        return end_sec < total_audio_sec * min_coverage

    def _expand_auto_language_tails(
        self,
        model: Any,
        audio_path: str,
        first_raw_segments: List[Any],
        total_audio_sec: float,
        transcribe_kwargs: Dict[str, Any],
    ) -> List[ASRSegment]:
        """Recover later language switches by re-running auto detection on unprocessed tails."""
        try:
            audio = model._load_audio(audio_path)
        except Exception as e:
            logger.warning(f"[MediaExpert] Could not load audio for multilingual ASR tail pass: {e}")
            return self._whispercpp_raw_to_asr_segments(first_raw_segments)

        sample_rate = 16000.0
        min_tail_sec = float(os.environ.get("MEDIA_ASR_AUTO_TAIL_MIN_SEC", "1.0"))
        segments = self._whispercpp_raw_to_asr_segments(first_raw_segments)
        cursor = max((seg.end for seg in segments), default=0.0)
        iterations = 0
        while total_audio_sec - cursor > min_tail_sec and iterations < 24:
            iterations += 1
            start_index = max(0, int(cursor * sample_rate))
            if start_index >= len(audio):
                break
            tail = audio[start_index:]
            try:
                raw_tail = model.transcribe(tail, **transcribe_kwargs)
            except Exception as e:
                logger.warning(f"[MediaExpert] Multilingual ASR tail pass failed: {e}")
                break
            tail_segments = self._whispercpp_raw_to_asr_segments(raw_tail, offset_sec=cursor)
            if not tail_segments:
                break
            new_end = max(seg.end for seg in tail_segments)
            segments.extend(tail_segments)
            cursor = new_end
            remaining_total = max(0.001, total_audio_sec - (tail_segments[0].start if tail_segments else cursor))
            covered = max(0.0, new_end - (tail_segments[0].start if tail_segments else cursor))
            if covered >= remaining_total * 0.80:
                break

        segments.sort(key=lambda s: (s.start, s.end))
        return segments

    def _run_asr_pywhispercpp(self, audio_path: str, language_hint: Optional[str] = None) -> List[ASRSegment]:
        """ASR via pywhispercpp (whisper.cpp GGML + Metal GPU on Mac)."""
        with type(self)._whisper_transcribe_lock:
            model = self._get_pywhispercpp_model()
            t0 = time.time()
            total_audio_sec = round(float(self._get_media_duration(audio_path) or 0.0), 2)
            self._emit_progress(
                "transcribing_audio",
                current_audio_sec=0.0,
                total_audio_sec=total_audio_sec,
                stage_rate=0.0,
            )

            # language="auto" is intentional for pywhispercpp: leaving language
            # unset can translate non-English speech to English instead of
            # transcribing it in-place.
            lang = (language_hint or os.environ.get("MEDIA_ASR_LANG", "")).strip()  # "" = auto-detect
            transcribe_kwargs: dict = {"language": lang or "auto"}
            if self._env_truthy("MEDIA_ASR_TRANSLATE"):
                transcribe_kwargs["translate"] = True

            raw_segments = model.transcribe(audio_path, **transcribe_kwargs)
            if (lang or "auto") == "auto" and self._should_expand_auto_language_tail(raw_segments, total_audio_sec):
                logger.info(
                    "[MediaExpert] ASR auto language pass covered only part of the audio; "
                    "expanding unprocessed multilingual tail"
                )
                segments = self._expand_auto_language_tails(
                    model,
                    audio_path,
                    raw_segments,
                    total_audio_sec,
                    transcribe_kwargs,
                )
            else:
                segments = self._whispercpp_raw_to_asr_segments(raw_segments)

        for seg in segments:
            processed_sec = min(total_audio_sec, round(seg.end, 2)) if total_audio_sec > 0 else round(seg.end, 2)
            elapsed = max(0.001, time.time() - t0)
            rate = (processed_sec / elapsed) if processed_sec > 0 else 0.0
            self._emit_progress(
                "transcribing_audio",
                current_audio_sec=processed_sec,
                total_audio_sec=total_audio_sec,
                stage_rate=round(rate, 2),
            )

        elapsed = time.time() - t0
        logger.info(
            f"[MediaExpert] whisper.cpp ASR done in {elapsed:.1f}s: {len(segments)} segments"
        )
        return segments

    def _run_asr_faster_whisper(self, audio_path: str) -> List[ASRSegment]:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("Neither pywhispercpp nor faster-whisper is available for ASR.")

        device, compute_type = self._detect_fw_device()

        _VALID_FW_MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3")
        model_name = (
            self._asr_model_name
            if self._asr_model_name in _VALID_FW_MODELS
            else "small"
        )

        if not hasattr(self, "_fw_model") or self._fw_model is None:
            logger.info(
                f"[MediaExpert] Loading faster-whisper '{model_name}' "
                f"(device={device}, compute={compute_type})"
            )
            t0 = time.time()
            self._fw_model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                num_workers=1,
            )
            logger.info(f"[MediaExpert] faster-whisper model loaded in {time.time() - t0:.1f}s")

        t0 = time.time()
        total_audio_sec = round(float(self._get_media_duration(audio_path) or 0.0), 2)
        self._emit_progress(
            "transcribing_audio",
            current_audio_sec=0.0,
            total_audio_sec=total_audio_sec,
            stage_rate=0.0,
        )
        lang_hint = os.environ.get("MEDIA_ASR_LANG", "").strip() or None
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="faster_whisper")
            segments_iter, info = self._fw_model.transcribe(
                audio_path,
                beam_size=1,
                word_timestamps=False,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
                language=lang_hint,
            )
        segments = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                segments.append(ASRSegment(
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    text=text,
                ))
                processed_sec = min(total_audio_sec, round(seg.end, 2)) if total_audio_sec > 0 else round(seg.end, 2)
                elapsed = max(0.001, time.time() - t0)
                rate = (processed_sec / elapsed) if processed_sec > 0 else 0.0
                self._emit_progress(
                    "transcribing_audio",
                    current_audio_sec=processed_sec,
                    total_audio_sec=total_audio_sec,
                    stage_rate=round(rate, 2),
                )
        elapsed = time.time() - t0
        lang_det = getattr(info, "language", "unknown")
        logger.info(
            f"[MediaExpert] faster-whisper ASR done in {elapsed:.1f}s: "
            f"lang={lang_det}, device={device}, {len(segments)} segments"
        )
        return segments

    # ── Audio Track Extraction ────────────────────────────────────────────

    @staticmethod
    @classmethod
    def _truthy_env(cls, name: str, default: str = "0") -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _media_binary_cmd(cls, name: str) -> Optional[str]:
        """Resolve packaged media binaries before falling back to PATH."""
        if name == "ffmpeg" and cls._ffmpeg_path_cache:
            return cls._ffmpeg_path_cache
        if name == "ffprobe" and cls._ffprobe_path_cache:
            return cls._ffprobe_path_cache

        env_bin = os.getenv("UNFOLDLY_FFMPEG_BIN" if name == "ffmpeg" else "UNFOLDLY_FFPROBE_BIN", "").strip()
        candidates: List[Path] = []
        if env_bin:
            candidates.append(Path(env_bin))

        env_dir = os.getenv("UNFOLDLY_FFMPEG_DIR", "").strip()
        if env_dir:
            candidates.append(Path(env_dir) / name)
            candidates.append(Path(env_dir) / "bin" / name)

        machine = platform.machine().lower()
        arch_dir = "arm64" if machine == "arm64" else "x64"
        roots: List[Path] = []
        for raw in (getattr(sys, "executable", ""), __file__):
            try:
                p = Path(raw).resolve()
            except Exception:
                continue
            roots.append(p if p.is_dir() else p.parent)
            roots.extend((p if p.is_dir() else p.parent).parents)

        seen = set()
        for root in roots:
            for candidate in (
                root / "ffmpeg" / "bin" / name,
                root / "Resources" / "ffmpeg" / "bin" / name,
                root / "macos_bundle" / "ffmpeg" / arch_dir / "bin" / name,
                root / "macos_bundle" / "ffmpeg" / "bin" / name,
            ):
                key = str(candidate)
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)

        system_binary = shutil.which(name)
        if system_binary:
            candidates.append(Path(system_binary))

        for candidate in candidates:
            try:
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    resolved = str(candidate)
                    if name == "ffmpeg":
                        cls._ffmpeg_path_cache = resolved
                    else:
                        cls._ffprobe_path_cache = resolved
                    logger.debug("[MediaExpert] using %s binary: %s", name, resolved)
                    return resolved
            except OSError:
                continue
        return None

    @classmethod
    def _ffmpeg_cmd(cls) -> Optional[str]:
        return cls._media_binary_cmd("ffmpeg")

    @classmethod
    def _ffprobe_cmd(cls) -> Optional[str]:
        return cls._media_binary_cmd("ffprobe")

    @classmethod
    def _check_ffmpeg(cls) -> bool:
        """Check if bundled or system ffmpeg CLI is available."""
        ffmpeg = cls._ffmpeg_cmd()
        if not ffmpeg:
            return False
        try:
            subprocess.run(
                [ffmpeg, "-version"],
                capture_output=True, timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _check_av() -> bool:
        """Check whether optional PyAV support was explicitly enabled."""
        if os.getenv("UNFOLDLY_ENABLE_PYAV", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            return False
        try:
            import av as _av  # noqa: F401
            return True
        except ImportError:
            return False

    def _extract_audio_track(self, video_path: str) -> Optional[str]:
        """
        Extract audio track from video as WAV (16kHz mono for Whisper).
        Uses ffmpeg by default. Optional PyAV support is only used when
        UNFOLDLY_ENABLE_PYAV=1 is explicitly set.
        """
        if self._check_av():
            result = self._extract_audio_pyav(video_path)
            if result:
                return result
            logger.debug("[MediaExpert] optional PyAV audio extraction failed, trying system ffmpeg")

        # Fallback: system ffmpeg CLI
        ffmpeg = self._ffmpeg_cmd()
        if not ffmpeg:
            logger.warning("[MediaExpert] system ffmpeg unavailable — cannot extract audio")
            return None

        tmp_wav = tempfile.mktemp(suffix=".wav", prefix="media_audio_")
        cmd = [
            ffmpeg, "-i", video_path,
            "-vn",                   # no video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "16000",          # 16kHz (Whisper optimal)
            "-ac", "1",              # mono
            tmp_wav, "-y",           # overwrite
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            if os.path.exists(tmp_wav) and os.path.getsize(tmp_wav) > 0:
                logger.info(f"[MediaExpert] Audio extracted via ffmpeg CLI: {os.path.getsize(tmp_wav) / 1024:.1f}KB")
                return tmp_wav
        except subprocess.CalledProcessError as e:
            logger.error(f"[MediaExpert] ffmpeg audio extraction failed: {e.stderr[:200] if e.stderr else ''}")
        except subprocess.TimeoutExpired:
            logger.error("[MediaExpert] ffmpeg audio extraction timed out (300s)")
        return None

    @staticmethod
    def _extract_audio_pyav(video_path: str) -> Optional[str]:
        """
        Extract audio track from video using PyAV and write as 16kHz mono WAV.
        This optional path is disabled by default for public release builds.
        """
        try:
            import av as _av
            import wave

            tmp_wav = tempfile.mktemp(suffix=".wav", prefix="media_audio_")
            pcm_frames: list = []

            with _av.open(video_path) as container:
                audio_stream = next(
                    (s for s in container.streams if s.type == "audio"), None
                )
                if audio_stream is None:
                    logger.warning("[MediaExpert] PyAV: no audio stream found in video")
                    return None

                resampler = _av.audio.resampler.AudioResampler(
                    format="s16",
                    layout="mono",
                    rate=16000,
                )
                for frame in container.decode(audio_stream):
                    for resampled in resampler.resample(frame):
                        pcm_frames.append(resampled.to_ndarray().tobytes())
                # Flush resampler
                for resampled in resampler.resample(None):
                    pcm_frames.append(resampled.to_ndarray().tobytes())

            if not pcm_frames:
                return None

            pcm_data = b"".join(pcm_frames)
            with wave.open(tmp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)   # 16-bit = 2 bytes
                wf.setframerate(16000)
                wf.writeframes(pcm_data)

            if os.path.exists(tmp_wav) and os.path.getsize(tmp_wav) > 0:
                logger.info(f"[MediaExpert] Audio extracted via PyAV: {os.path.getsize(tmp_wav) / 1024:.1f}KB")
                return tmp_wav
        except Exception as e:
            logger.debug(f"[MediaExpert] PyAV audio extraction failed: {e}")
        return None

    # ── Media Metadata ────────────────────────────────────────────────────

    @staticmethod
    def _extract_video_metadata(file_path: str) -> dict:
        """
        Extract rich metadata from video/audio file.
        Returns a dict with keys like creation_time, device_make, device_model,
        location, author, recording_type, resolution, fps, has_audio, etc.
        """
        info = {}
        if MediaExpert._check_av():
            try:
                import av as _av
                container = _av.open(file_path)
                meta = container.metadata or {}

                # ── Creation time ──
                ct = meta.get("com.apple.quicktime.creationdate") or meta.get("creation_time")
                if ct:
                    info["creation_time"] = ct

                # ── Device info ──
                make = meta.get("com.apple.quicktime.make")
                model = meta.get("com.apple.quicktime.model")
                if make:
                    info["device_make"] = make
                if model:
                    info["device_model"] = model

                # ── Author / software ──
                author = meta.get("com.apple.quicktime.author") or meta.get("artist") or meta.get("author")
                if author:
                    info["author"] = author
                sw = meta.get("com.apple.quicktime.software") or meta.get("encoder")
                if sw:
                    info["software"] = sw

                # ── GPS / Location ──
                loc = meta.get("com.apple.quicktime.location.ISO6709")
                if loc:
                    info["location_iso6709"] = loc
                loc_acc = meta.get("com.apple.quicktime.location.accuracy.horizontal")
                if loc_acc:
                    info["location_accuracy_m"] = loc_acc

                # ── Video stream info ──
                has_audio = False
                for stream in container.streams:
                    if stream.type == "video":
                        info["resolution"] = f"{stream.width}x{stream.height}"
                        try:
                            info["fps"] = round(float(stream.average_rate), 2)
                        except Exception:
                            pass
                        codec = getattr(stream.codec_context, "name", None)
                        if codec:
                            info["video_codec"] = codec
                    elif stream.type == "audio":
                        has_audio = True
                        channels = getattr(stream, "channels", None) or getattr(
                            stream.codec_context, "channels", None
                        )
                        sample_rate = getattr(stream, "sample_rate", None) or getattr(
                            stream.codec_context, "sample_rate", None
                        )
                        if channels:
                            info["audio_channels"] = channels
                        if sample_rate:
                            info["audio_sample_rate"] = sample_rate
                        codec = getattr(stream.codec_context, "name", None)
                        if codec:
                            info["audio_codec"] = codec
                info["has_audio"] = has_audio

                container.close()
            except Exception as e:
                logger.debug(f"[MediaExpert] PyAV metadata extraction failed: {e}")

        if "has_audio" not in info:
            for key, value in MediaExpert._extract_video_metadata_ffprobe(file_path).items():
                info.setdefault(key, value)

        # Enrich with macOS extended attributes (download source, etc.)
        info.update(MediaExpert._extract_macos_source(file_path))

        # Refine recording_type using all available signals + filename hints
        info["recording_type"] = MediaExpert._classify_media_origin(info, file_path=file_path)

        return info

    @staticmethod
    def _extract_video_metadata_ffprobe(file_path: str) -> dict:
        """Metadata probe through system ffprobe."""
        info: dict = {}
        ffprobe = MediaExpert._ffprobe_cmd()
        if not ffprobe:
            return info
        try:
            import json

            cmd = [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                (
                    "stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,"
                    "sample_rate,channels:format_tags=creation_time,artist,author,encoder"
                ),
                "-of",
                "json",
                file_path,
            ]
            raw = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if raw.returncode != 0 or not raw.stdout.strip():
                return info
            data = json.loads(raw.stdout)

            tags = ((data.get("format") or {}).get("tags") or {})
            creation_time = tags.get("creation_time")
            if creation_time:
                info["creation_time"] = creation_time
            author = tags.get("artist") or tags.get("author")
            if author:
                info["author"] = author
            encoder = tags.get("encoder")
            if encoder:
                info["software"] = encoder

            has_audio = False
            for stream in data.get("streams") or []:
                stream_type = str(stream.get("codec_type") or "")
                codec = str(stream.get("codec_name") or "")
                if stream_type == "video":
                    width = stream.get("width")
                    height = stream.get("height")
                    if width and height:
                        info["resolution"] = f"{width}x{height}"
                    rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "")
                    try:
                        if "/" in rate:
                            num, den = rate.split("/", 1)
                            fps = float(num) / float(den or 1)
                        else:
                            fps = float(rate)
                        if fps > 0:
                            info["fps"] = round(fps, 2)
                    except Exception:
                        pass
                    if codec:
                        info["video_codec"] = codec
                elif stream_type == "audio":
                    has_audio = True
                    if codec:
                        info["audio_codec"] = codec
                    try:
                        channels = int(stream.get("channels") or 0)
                        if channels > 0:
                            info["audio_channels"] = channels
                    except Exception:
                        pass
                    try:
                        sample_rate = int(stream.get("sample_rate") or 0)
                        if sample_rate > 0:
                            info["audio_sample_rate"] = sample_rate
                    except Exception:
                        pass
            info["has_audio"] = has_audio
        except Exception as e:
            logger.debug(f"[MediaExpert] ffprobe metadata extraction failed: {e}")
        return info

    @staticmethod
    def _extract_macos_source(file_path: str) -> dict:
        """
        Read macOS extended attributes to determine where the file came from.
        Extracts kMDItemWhereFroms (download URL) and kMDItemDownloadedDate.
        """
        info: dict = {}
        try:
            import plistlib
            raw = subprocess.run(
                ["xattr", "-px", "com.apple.metadata:kMDItemWhereFroms", file_path],
                capture_output=True, text=True, timeout=5,
            )
            if raw.returncode == 0 and raw.stdout.strip():
                hex_str = raw.stdout.replace(" ", "").replace("\n", "")
                plist_bytes = bytes.fromhex(hex_str)
                urls = plistlib.loads(plist_bytes)
                if isinstance(urls, list) and urls:
                    info["source_url"] = urls[0]
                    if len(urls) > 1:
                        info["source_page"] = urls[1]
        except Exception as e:
            logger.debug(f"[MediaExpert] xattr source extraction failed: {e}")

        try:
            raw_q = subprocess.run(
                ["xattr", "-p", "com.apple.quarantine", file_path],
                capture_output=True, text=True, timeout=5,
            )
            if raw_q.returncode == 0 and raw_q.stdout.strip():
                parts = raw_q.stdout.strip().split(";")
                if len(parts) >= 3:
                    info["quarantine_agent"] = parts[2]
        except Exception:
            pass

        return info

    @staticmethod
    def _classify_media_origin(info: dict, file_path: str = "") -> str:
        """
        Determine media origin type based on all available metadata signals.
        Returns one of: screen_recording, camera, voice_recording,
                        downloaded, app_generated, unknown
        """
        author = str(info.get("author") or "").lower()
        software = str(info.get("software") or "").lower()
        source_url = str(info.get("source_url") or "").lower()
        quarantine_agent = str(info.get("quarantine_agent") or "").lower()
        make = info.get("device_make")
        model = info.get("device_model")

        if "replaykitrecording" in author or "replaykit" in software:
            return "screen_recording"

        if MediaExpert._filename_looks_like_screen_recording(file_path):
            return "screen_recording"

        if make or model:
            return "camera"

        if source_url:
            return "downloaded"

        if quarantine_agent:
            if "safari" in quarantine_agent or "chrome" in quarantine_agent or "firefox" in quarantine_agent:
                return "downloaded"
            return "app_generated"

        _voice_kws = ("voice", "recorder", "memo", "录音", "语音")
        if any(kw in author for kw in _voice_kws) or any(kw in software for kw in _voice_kws):
            return "voice_recording"

        return "unknown"

    @staticmethod
    def _format_video_metadata_text(meta: dict, duration_sec: float) -> str:
        """Format media metadata into human-readable text for indexing."""
        parts = []

        # Recording type / media origin
        _ORIGIN_LABELS = {
            "screen_recording": "screen recording",
            "camera": "camera recording",
            "downloaded": "downloaded media",
            "app_generated": "app-generated media",
            "voice_recording": "voice recording",
        }
        rtype = meta.get("recording_type", "unknown")
        label = _ORIGIN_LABELS.get(rtype)
        if label:
            parts.append(f"Source type: {label}")

        # Device
        make = meta.get("device_make", "")
        model = meta.get("device_model", "")
        if make or model:
            parts.append(f"Recording device: {(make + ' ' + model).strip()}")

        # Creation time
        ct = meta.get("creation_time")
        if ct:
            parts.append(f"Created at: {ct}")

        # Author
        author = meta.get("author")
        if author and author != "ReplayKitRecording":
            parts.append(f"Author: {author}")

        # Download source
        source_url = meta.get("source_url")
        if source_url:
            from urllib.parse import urlparse
            try:
                domain = urlparse(source_url).netloc or source_url[:80]
            except Exception:
                domain = source_url[:80]
            parts.append(f"Download source: {domain}")

        # Quarantine agent (which app downloaded it)
        q_agent = meta.get("quarantine_agent")
        if q_agent:
            parts.append(f"Downloaded by: {q_agent}")

        # Resolution & fps
        res = meta.get("resolution")
        fps = meta.get("fps")
        if res:
            fps_str = f" @ {fps}fps" if fps else ""
            parts.append(f"Resolution: {res}{fps_str}")

        # Duration
        m, s = divmod(int(duration_sec), 60)
        h, m = divmod(m, 60)
        if h:
            parts.append(f"Duration: {h}h {m}m {s}s")
        elif m:
            parts.append(f"Duration: {m}m {s}s")
        else:
            parts.append(f"Duration: {s}s")

        # Audio
        if meta.get("has_audio"):
            ch = meta.get("audio_channels", 0)
            ch_str = "mono" if ch == 1 else ("stereo" if ch == 2 else f"{ch} channels")
            parts.append(f"Audio: present ({ch_str})")
        elif meta.get("has_audio") is not None:
            parts.append("Audio: none")

        # GPS
        if meta.get("location_iso6709"):
            parts.append(f"Location: {meta['location_iso6709']}")

        return " | ".join(parts)

    # ── Media Duration ────────────────────────────────────────────────────

    @staticmethod
    def _get_media_duration(file_path: str) -> float:
        """
        Get media file duration in seconds.
        Uses ffprobe by default; optional PyAV is disabled for public release builds.
        """
        if MediaExpert._check_av():
            try:
                import av as _av
                with _av.open(file_path) as container:
                    if container.duration is not None:
                        return float(container.duration) / 1_000_000  # microseconds → seconds
            except Exception as e:
                logger.debug(f"[MediaExpert] PyAV duration failed: {e}")

        # Fallback: ffprobe CLI
        ffprobe = MediaExpert._ffprobe_cmd()
        if not ffprobe:
            return 0.0
        try:
            cmd = [
                ffprobe, "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception as e:
            logger.debug(f"[MediaExpert] ffprobe duration failed: {e}")
        return 0.0

    # ── Keyframe Extraction ───────────────────────────────────────────────

    def _should_run_video_asr(
        self,
        *,
        has_audio_track: bool,
        recording_type: str,
    ) -> bool:
        if not has_audio_track:
            return False
        if not self._env_enabled("MEDIA_ENABLE_VIDEO_ASR", "1"):
            return False
        if recording_type == "screen_recording":
            return self._env_enabled("MEDIA_ENABLE_SCREEN_RECORDING_ASR", "0")
        return True

    def _should_run_frame_ocr(
        self,
        *,
        has_audio_track: bool,
        has_asr_transcript: bool,
        recording_type: str,
    ) -> bool:
        """
        Lightweight OCR is only worth paying for when ASR is missing and visual
        text becomes primary signal (screen recordings, PPT-like UI, silent video).
        """
        if not self._frame_ocr_fn or has_asr_transcript:
            return False
        if recording_type == "screen_recording":
            return self._env_enabled("MEDIA_ENABLE_SCREEN_RECORDING_FRAME_OCR", "0")
        if not has_audio_track:
            return True
        return self._env_enabled("MEDIA_ENABLE_FRAME_OCR_FOR_NO_ASR", "1")

    def _select_keyframe_indices_for_ocr(self, total_frames: int) -> set[int]:
        if total_frames <= 0:
            return set()
        try:
            max_ocr_frames = int(os.environ.get("MEDIA_OCR_MAX_FRAMES", "6"))
        except Exception:
            max_ocr_frames = 6
        max_ocr_frames = max(1, min(max_ocr_frames, total_frames))
        if max_ocr_frames >= total_frames:
            return set(range(total_frames))
        step = (total_frames - 1) / max(max_ocr_frames - 1, 1)
        return {min(total_frames - 1, round(i * step)) for i in range(max_ocr_frames)}

    def _get_keyframe_timestamps(
        self,
        duration: float,
        *,
        has_audio_track: bool,
        has_asr_transcript: bool,
        recording_type: str = "",
    ) -> List[float]:
        """
        Build a compute-aware keyframe plan.

        Strategy:
          - If ASR already gives us searchable text, keep frames sparse.
          - If ASR is missing or the video has no audio, spend more budget on visuals.
          - Screen recordings without ASR get the densest default budget because
            UI text and layout changes are often the main signal.
        """
        if duration <= 0:
            return []

        priority = "visual_fallback"
        max_frames = self.DEFAULT_MAX_KEYFRAMES_NO_ASR
        target_interval = 15.0

        if has_asr_transcript:
            priority = "asr_first"
            max_frames = self.DEFAULT_MAX_KEYFRAMES_WITH_ASR
            if duration <= 60:
                target_interval = 10.0
            elif duration <= 300:
                target_interval = 45.0
            elif duration <= 900:
                target_interval = 120.0
            else:
                target_interval = 600.0
        elif not has_audio_track:
            priority = "visual_only"
            max_frames = self.DEFAULT_MAX_KEYFRAMES_NO_ASR
            if duration <= 60:
                target_interval = 4.0
            elif duration <= 300:
                target_interval = 15.0
            elif duration <= 900:
                target_interval = 45.0
            else:
                target_interval = 180.0
        else:
            priority = "visual_fallback"
            max_frames = self.DEFAULT_MAX_KEYFRAMES_NO_ASR
            if duration <= 60:
                target_interval = 5.0
            elif duration <= 300:
                target_interval = 20.0
            elif duration <= 900:
                target_interval = 60.0
            else:
                target_interval = 180.0

        if recording_type == "screen_recording" and not has_asr_transcript:
            priority = "screen_visual_priority"
            max_frames = max(max_frames, self.DEFAULT_MAX_KEYFRAMES_SCREEN_NO_ASR)
            if duration <= 60:
                target_interval = 4.0
            elif duration <= 300:
                target_interval = 15.0
            elif duration <= 900:
                target_interval = 45.0
            else:
                target_interval = 180.0

        max_frames = max(3, min(max_frames, self.DEFAULT_MAX_KEYFRAMES))
        target_frames = min(max_frames, max(1, int(duration / max(target_interval, 1.0))))
        interval = duration / target_frames if target_frames > 1 else duration

        timestamps: List[float] = []
        t = 0.0
        while t < duration and len(timestamps) < max_frames:
            timestamps.append(round(t, 1))
            t += interval
        if not timestamps or timestamps[-1] < duration - interval * 0.5:
            timestamps.append(round(max(0.0, duration - 1), 1))

        deduped = list(dict.fromkeys(timestamps))
        if len(deduped) > max_frames:
            tail_ts = round(max(0.0, duration - 1), 1)
            deduped = deduped[: max_frames - 1] + [tail_ts]
            deduped = list(dict.fromkeys(deduped))
        logger.info(
            f"[MediaExpert] keyframe budget: duration={duration:.1f}s "
            f"priority={priority} has_audio={has_audio_track} has_asr={has_asr_transcript} "
            f"recording_type={recording_type or 'unknown'} max_frames={max_frames} "
            f"target_interval={target_interval}s actual_frames={len(deduped)}"
        )
        return deduped[:max_frames]

    def _extract_keyframes(
        self,
        video_path: str,
        *,
        has_audio_track: bool = True,
        has_asr_transcript: bool = False,
        recording_type: str = "",
    ) -> List[KeyframeInfo]:
        """
        Extract keyframes from a video using a compute-aware budget.
        Uses ffmpeg by default. Optional PyAV support is only used when
        UNFOLDLY_ENABLE_PYAV=1 is explicitly set.
        """
        duration = self._get_media_duration(video_path)
        if duration <= 0:
            return []

        timestamps = self._get_keyframe_timestamps(
            duration,
            has_audio_track=has_audio_track,
            has_asr_transcript=has_asr_transcript,
            recording_type=recording_type,
        )
        if not timestamps:
            return []

        if self._check_av():
            keyframes = self._extract_keyframes_pyav(video_path, timestamps)
            if keyframes:
                logger.info(f"[MediaExpert] Extracted {len(keyframes)}/{len(timestamps)} keyframes via PyAV")
                return keyframes
            logger.debug("[MediaExpert] optional PyAV keyframe extraction failed, trying system ffmpeg")

        ffmpeg = self._ffmpeg_cmd()
        if not ffmpeg:
            logger.warning("[MediaExpert] system ffmpeg unavailable — cannot extract keyframes")
            return []

        tmp_dir = tempfile.mkdtemp(prefix="media_keyframes_")
        keyframes = []
        for ts in timestamps:
            frame_path = os.path.join(tmp_dir, f"kf_{ts:.1f}.jpg")
            cmd = [
                ffmpeg, "-ss", str(ts),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                frame_path, "-y",
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=15)
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    keyframes.append(KeyframeInfo(time_sec=ts, frame_path=frame_path))
            except Exception as e:
                logger.debug(f"[MediaExpert] ffmpeg keyframe at {ts}s failed: {e}")

        logger.info(f"[MediaExpert] Extracted {len(keyframes)}/{len(timestamps)} keyframes via ffmpeg CLI")
        return keyframes

    @staticmethod
    def _extract_keyframes_pyav(video_path: str, timestamps: List[float]) -> List[KeyframeInfo]:
        """Extract keyframes with optional PyAV."""
        try:
            import av as _av
            tmp_dir = tempfile.mkdtemp(prefix="media_keyframes_")
            keyframes = []

            with _av.open(video_path) as container:
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = "NONREF"  # only decode keyframes for speed

                for ts in timestamps:
                    seek_ts = int(ts / stream.time_base)
                    try:
                        container.seek(seek_ts, stream=stream)
                        for frame in container.decode(stream):
                            frame_path = os.path.join(tmp_dir, f"kf_{ts:.1f}.jpg")
                            img = frame.to_image()
                            img.save(frame_path, "JPEG", quality=85)
                            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                                keyframes.append(KeyframeInfo(time_sec=ts, frame_path=frame_path))
                            break
                    except Exception as e:
                        logger.debug(f"[MediaExpert] PyAV keyframe at {ts}s failed: {e}")

            return keyframes
        except Exception as e:
            logger.debug(f"[MediaExpert] PyAV keyframe extraction failed: {e}")
            return []

    # ── LLM Summary ───────────────────────────────────────────────────────

    # ── ASR Text Cleaning (deterministic, zero LLM cost) ─────────────────

    # Chinese filler words that ASR commonly inserts
    _ZH_FILLER_RE = re.compile(
        r'(?<![\u4e00-\u9fff])'
        r'(?:就是说|就是|那个|这个|然后就|就是那个)'
        r'|(?:^|(?<=[\s\uff0c。？！]))'  # at word boundary
        r'(?:嗯[嗯]*|啊[啊]*|哦[哦]*|哈[哈]*|呢|呀|喔|問|喇)'
        r'(?=[\s\uff0c。\u4e00-\u9fff]|$)',
        re.IGNORECASE,
    )
    _EN_FILLER_RE = re.compile(
        r'\b(?:um+|uh+|er+|ah+|you know,?|i mean,?|like,|basically,?|literally,?'  # noqa
        r'|so+ so+|right\?|okay so|and and|but but)\b',
        re.IGNORECASE,
    )
    _REPEAT_RE = re.compile(r'(\S)\1{2,}')
    _NON_ENGLISH_INDEX_RE = re.compile(
        r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af\u0400-\u04ff]"
    )

    @classmethod
    def _simple_clean_asr_text(cls, text: str) -> str:
        """
        Deterministic ASR artifact removal — no LLM, zero latency.
        Removes common filler words and character repetitions.
        """
        text = cls._ZH_FILLER_RE.sub('', text)
        text = cls._EN_FILLER_RE.sub('', text)
        text = cls._REPEAT_RE.sub(r'\1', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    @classmethod
    def _detect_transcript_lang(cls, transcript: str) -> str:
        """Return 'zh' if transcript is predominantly Chinese, else 'en'."""
        if not transcript:
            return 'en'
        cjk_count = sum(1 for ch in transcript if '\u4e00' <= ch <= '\u9fff')
        return 'zh' if cjk_count / max(len(transcript), 1) > 0.15 else 'en'

    @staticmethod
    def _strip_model_thinking(text: str) -> str:
        """Remove local reasoning wrappers before text is stored in the index."""
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def _non_english_index_ratio(cls, text: str) -> float:
        visible = [ch for ch in str(text or "") if not ch.isspace()]
        if not visible:
            return 0.0
        non_en = sum(1 for ch in visible if cls._NON_ENGLISH_INDEX_RE.search(ch))
        return non_en / max(len(visible), 1)

    @classmethod
    def _needs_english_index_rewrite(cls, text: str) -> bool:
        return cls._non_english_index_ratio(text) > 0.08

    def _ensure_english_index_text(
        self,
        text: str,
        *,
        max_len: int = 900,
        context_label: str = "media index text",
    ) -> str:
        """Keep media index summaries English-first while preserving source details."""
        raw = self._strip_model_thinking(text)
        if not raw:
            return ""
        if not self._needs_english_index_rewrite(raw):
            return raw[:max_len]
        if not self._llm_client:
            return raw[:max_len]

        index_model_id = self._get_index_model_id()
        if not index_model_id:
            return raw[:max_len]

        source = raw[: max(max_len * 2, 1200)]
        prompt = (
            f"Rewrite the following {context_label} into concise natural English for retrieval indexing.\n"
            "- Translate Chinese or other non-English content into English.\n"
            "- Preserve names, filenames, timestamps, numbers, technical terms, and visible screen text when useful.\n"
            "- Keep retrieval anchors and factual meaning; do not add new facts.\n"
            "- Output English only, no markdown, no extra explanation.\n\n"
            f"Source text:\n---\n{source}\n---"
        )
        try:
            response = self._llm_client.chat.completions.create(
                model=index_model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=min(800, max(96, max_len)),
                temperature=0.0,
                stream=False,
            )
            rewritten = self._strip_model_thinking(response.choices[0].message.content or "")
            return (rewritten or raw)[:max_len]
        except Exception as e:
            logger.debug(f"[MediaExpert] English index rewrite failed for {context_label}: {e}")
            return raw[:max_len]

    def _generate_transcript_summary(self, transcript: str, file_name: str) -> str:
        """
        Generate a structured meeting-minutes or content summary from ASR transcript.

        • Always writes index summaries in English, even when the transcript is not.
        • Produces meeting-minutes style output (topics / key points / conclusions /
          action items) rather than a generic 3-sentence summary.
        • Strips raw thinking blocks (<think>...</think>) from model output.
        """
        if not self._llm_client:
            return ""

        max_chars = 6000
        truncated = transcript[:max_chars]
        if len(transcript) > max_chars:
            truncated += f"\n\n... [Transcript truncated, full length: {len(transcript)} characters]"

        prompt = (
            f'Below is the speech-to-text transcript from the media file "{file_name}". '
            "The transcript may be in Chinese or another language and may contain ASR errors or filler words.\n"
            "Please do two things:\n"
            "1. Generate a structured meeting-minutes / content summary including: "
            "[Main Topics], [Key Points], [Conclusions] (if any), [Action Items] (if any).\n"
            "2. Write a 100-200 word first paragraph as a plain-text search index summary.\n\n"
            "Language policy for indexing:\n"
            "- Output in English, even if the transcript is Chinese or another language.\n"
            "- Translate or paraphrase spoken content into natural English.\n"
            "- Preserve original proper nouns, filenames, timestamps, numbers, and technical terms when useful.\n"
            "- Do not add facts that are not supported by the transcript.\n\n"
            f"Transcript:\n---\n{truncated}\n---\n\n"
            "Start with the English summary paragraph, then the structured notes."
        )

        try:
            index_model_id = self._get_index_model_id()
            if not index_model_id:
                return ""
            response = self._llm_client.chat.completions.create(
                model=index_model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.0,
                stream=False,
            )
            raw = (response.choices[0].message.content or "").strip()
            raw = self._strip_model_thinking(raw)
            return self._ensure_english_index_text(
                raw,
                max_len=900,
                context_label="media transcript summary",
            )
        except Exception as e:
            logger.error(f"[MediaExpert] LLM transcript summary failed: {e}")
            return ""

    # ── VL Frame Description ─────────────────────────────────────────────

    def _describe_frame_with_vl(self, frame_path: str, lang: str = "en") -> str:
        """Describe a video frame using the VL model via LLM client.

        Args:
            frame_path: Path to the JPEG frame image.
            lang: Language for the description ('zh' or 'en').
        """
        if not self._llm_client:
            return ""
        import base64
        try:
            with open(frame_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            if lang == "zh":
                desc_prompt = (
                    "请用中文描述这个视频帧（2-3句话）。"
                    "重点说明：画面中显示了什么内容、有哪些人物或物体、"
                    "屏幕上是否有文字、整体场景是什么。"
                )
            else:
                desc_prompt = (
                    "Describe this video frame in 2-3 sentences. "
                    "Focus on: what is visible, any people/objects, "
                    "text on screen, and the overall scene. "
                    "Use searchable concrete nouns for visible animals, vehicles, products, places, "
                    "logos, activities, and environments. If an animal or object is ambiguous, "
                    "describe the cautious broad class plus any visually plausible type, without inventing details. "
                    "Answer entirely in English. If visible text is Chinese or another language, "
                    "translate it into English and preserve the original text only when useful."
                )

            index_model_id = self._get_index_model_id()
            if not index_model_id:
                return ""
            response = self._llm_client.chat.completions.create(
                model=index_model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                        {"type": "text", "text": desc_prompt},
                    ],
                }],
                max_tokens=200,
                temperature=0.0,
                stream=False,
            )
            raw = (response.choices[0].message.content or "").strip()
            raw = self._strip_model_thinking(raw)
            if lang != "zh":
                raw = self._ensure_english_index_text(
                    raw,
                    max_len=300,
                    context_label="video frame description",
                )
            return raw[:300]
        except Exception as e:
            logger.error(f"[MediaExpert] VL frame describe failed: {e}")
            return ""

    # ── Chunking for Indexing ─────────────────────────────────────────────

    @staticmethod
    def _build_visual_summary_text(
        result: MediaResult,
        file_name: str,
        *,
        fallback: bool = True,
    ) -> str:
        kf_snippets = []
        for kf in result.keyframes:
            parts = []
            if kf.description:
                parts.append(kf.description[:160])
            if kf.ocr_text:
                parts.append(f"OCR: {kf.ocr_text[:120]}")
            if parts:
                kf_snippets.append(f"[{kf.time_sec:.0f}s] " + " | ".join(parts))
        if kf_snippets:
            return "Visual content: " + " | ".join(kf_snippets)
        return f"Media file: {file_name}" if fallback else ""

    def build_index_chunks(
        self, result: MediaResult, base_metadata: dict
    ) -> List[Tuple[str, dict]]:
        """
        Build (text, metadata) chunks for ChromaDB indexing.

        Produces:
          1. Summary chunk — full transcript summary (main searchable chunk)
          2. Transcript window chunks — grouped by ~60s intervals for coarse semantic recall
          3. Precise transcript segment chunks — one per ASR segment for exact timestamp lookup
          4. Keyframe chunks — one per keyframe with VL description (video only)
        """
        chunks: List[Tuple[str, dict]] = []
        file_name = os.path.basename(result.file_path)

        if result.transcript_summary:
            result.transcript_summary = self._ensure_english_index_text(
                result.transcript_summary,
                max_len=900,
                context_label="media transcript summary",
            )
        for kf in result.keyframes:
            if kf.description:
                kf.description = self._ensure_english_index_text(
                    kf.description,
                    max_len=300,
                    context_label="video keyframe description",
                )

        # ── Chunk 0: Summary / overview ───────────────────────────────────
        # When there is no ASR transcript (e.g. silent screen recording), build a
        # visual summary from the keyframe descriptions so chunk 0 still carries
        # meaningful searchable content instead of a bare filename placeholder.
        has_meaningful_asr = self._has_meaningful_asr_transcript(result.transcript)
        transcript_summary = (
            result.transcript_summary
            if not self._looks_like_low_signal_summary_text(result.transcript_summary)
            else ""
        )
        visual_summary = self._build_visual_summary_text(result, file_name, fallback=False)
        summary_parts: List[str] = []
        if transcript_summary:
            summary_parts.append(f"Speech/content summary: {transcript_summary}")
        elif has_meaningful_asr:
            summary_parts.append(f"Speech/content transcript: {result.transcript[:500]}")
        if visual_summary:
            summary_parts.append(visual_summary)
        if summary_parts:
            summary_text = re.sub(r'\s+', ' ', " ".join(summary_parts)).strip()
        elif result.keyframes:
            summary_text = self._build_visual_summary_text(result, file_name)
        else:
            summary_text = f"Media file: {file_name}"
        summary_text = self._ensure_english_index_text(
            summary_text,
            max_len=1200,
            context_label="media overview summary",
        )

        # Build metadata description text
        meta_text = ""
        if result.video_metadata:
            meta_text = self._format_video_metadata_text(result.video_metadata, result.duration_sec)

        media_label = "video" if result.media_type == "video" else "audio"
        overview_parts = [f"File name: {file_name}", f"Type: {media_label} file"]
        if meta_text:
            overview_parts.append(meta_text)
        overview_parts.append(f"Content summary: {summary_text}")
        overview = "\n".join(overview_parts)

        _vm = result.video_metadata or {}
        _structured_fields = {}
        for _fk in ("device_make", "device_model", "recording_type",
                     "creation_time", "author", "software",
                     "source_url", "quarantine_agent"):
            if _vm.get(_fk):
                _structured_fields[_fk] = str(_vm[_fk])

        meta_0 = {
            **base_metadata,
            "chunk_type": "media_summary",
            "media_type": result.media_type,
            "media_duration_sec": round(result.duration_sec, 1),
            "has_asr_transcript": has_meaningful_asr,
            "has_keyframes": bool(result.keyframes),
            "has_keyframe_ocr": bool(any(kf.ocr_text for kf in result.keyframes)),
            **_structured_fields,
        }
        chunks.append((overview, meta_0))

        if result.media_type == "video":
            audio_summary_text = transcript_summary or (result.transcript[:500] if has_meaningful_asr else "")
            if audio_summary_text:
                audio_summary_clean = re.sub(r'\s+', ' ', audio_summary_text).strip()
                audio_summary_clean = self._ensure_english_index_text(
                    audio_summary_clean,
                    max_len=900,
                    context_label="video audio summary",
                )
                audio_parts = [f"File name: {file_name}", "Type: video audio summary"]
                if meta_text:
                    audio_parts.append(meta_text)
                audio_parts.append(f"Audio summary: {audio_summary_clean}")
                meta_audio = {
                    **base_metadata,
                    "chunk_type": "media_audio_summary",
                    "media_type": "video",
                    "media_duration_sec": round(result.duration_sec, 1),
                    "has_asr_transcript": has_meaningful_asr,
                    **_structured_fields,
                }
                chunks.append(("\n".join(audio_parts), meta_audio))

            if visual_summary:
                visual_parts = [f"File name: {file_name}", "Type: video visual summary"]
                if meta_text:
                    visual_parts.append(meta_text)
                visual_parts.append(f"Visual summary: {visual_summary}")
                meta_visual = {
                    **base_metadata,
                    "chunk_type": "media_visual_summary",
                    "media_type": "video",
                    "media_duration_sec": round(result.duration_sec, 1),
                    "has_keyframes": bool(result.keyframes),
                    "has_keyframe_ocr": bool(any(kf.ocr_text for kf in result.keyframes)),
                    **_structured_fields,
                }
                chunks.append(("\n".join(visual_parts), meta_visual))

        # ── Chunks 1..N: Transcript windows (grouped by CHUNK_DURATION_SEC) ──
        meaningful_segments = self._meaningful_asr_segments(result.asr_segments)
        if meaningful_segments:
            current_start = 0.0
            current_texts: List[str] = []
            chunk_end = float(self.CHUNK_DURATION_SEC)

            for seg in meaningful_segments:
                if seg.start >= chunk_end and current_texts:
                    # Flush current chunk — apply filler-word cleaning before storage
                    raw_window = " ".join(current_texts)
                    clean_window = self._simple_clean_asr_text(raw_window)
                    if self._looks_like_low_signal_transcript(clean_window):
                        clean_window = ""
                    chunk_text = (
                        f"[{file_name} @ {current_start:.0f}s-{chunk_end:.0f}s] "
                        + clean_window
                    )
                    if clean_window:
                        meta_seg = {
                            **base_metadata,
                            "chunk_type": "asr_transcript",
                            "media_type": result.media_type,
                            "asr_start_sec": round(current_start, 1),
                            "asr_end_sec": round(chunk_end, 1),
                        }
                        chunks.append((chunk_text, meta_seg))

                    # Advance window
                    current_start = chunk_end
                    chunk_end = current_start + self.CHUNK_DURATION_SEC
                    current_texts = []

                current_texts.append(seg.text.strip())

            # Flush remaining
            if current_texts:
                actual_end = meaningful_segments[-1].end if meaningful_segments else chunk_end
                raw_window = " ".join(current_texts)
                clean_window = self._simple_clean_asr_text(raw_window)
                if self._looks_like_low_signal_transcript(clean_window):
                    clean_window = ""
                chunk_text = (
                    f"[{file_name} @ {current_start:.0f}s-{actual_end:.0f}s] "
                    + clean_window
                )
                if clean_window:
                    meta_seg = {
                        **base_metadata,
                        "chunk_type": "asr_transcript",
                        "media_type": result.media_type,
                        "asr_start_sec": round(current_start, 1),
                        "asr_end_sec": round(actual_end, 1),
                    }
                    chunks.append((chunk_text, meta_seg))

            # can retrieve exact spans instead of re-scanning minute-sized windows.
            total_segments = len(meaningful_segments)
            for seg_idx, seg in enumerate(meaningful_segments, 1):
                clean_seg_text = self._simple_clean_asr_text(seg.text)
                if not clean_seg_text or self._looks_like_low_signal_asr_text(clean_seg_text):
                    continue
                seg_prefix = f"[{file_name} speech @ {seg.start:.1f}s-{seg.end:.1f}s]"
                if seg.speaker:
                    seg_prefix += f" [{seg.speaker}]"
                meta_precise = {
                    **base_metadata,
                    "chunk_type": "asr_segment",
                    "media_type": result.media_type,
                    "asr_start_sec": round(seg.start, 1),
                    "asr_end_sec": round(seg.end, 1),
                    "asr_mid_sec": round((float(seg.start) + float(seg.end)) / 2.0, 1),
                    "asr_segment_index": seg_idx,
                    "asr_segment_count": total_segments,
                    "speaker": seg.speaker or "",
                }
                chunks.append((f"{seg_prefix} {clean_seg_text}", meta_precise))

        # ── Chunks K: Keyframe descriptions (video only) ──────────────────
        for kf in result.keyframes:
            if kf.description or kf.ocr_text:
                _parts = []
                if kf.description:
                    _parts.append(kf.description)
                if kf.ocr_text:
                    _parts.append(f"OCR: {kf.ocr_text}")
                kf_text = f"[{file_name} visual @ {kf.time_sec:.0f}s] " + " | ".join(_parts)
                meta_kf = {
                    **base_metadata,
                    "chunk_type": "keyframe",
                    "media_type": "video",
                    "keyframe_time_sec": round(kf.time_sec, 1),
                    "keyframe_description": kf.description,
                    "keyframe_ocr_text": kf.ocr_text,
                }
                chunks.append((kf_text, meta_kf))

        n_asr = sum(1 for _, m in chunks if m.get('chunk_type') == 'asr_transcript')
        n_asr_segments = sum(1 for _, m in chunks if m.get('chunk_type') == 'asr_segment')
        n_kf = sum(1 for _, m in chunks if m.get('chunk_type') == 'keyframe')
        logger.info(
            f"[MediaExpert] Built {len(chunks)} index chunks "
            f"(1 summary + {n_asr} transcript + {n_asr_segments} precise_asr + {n_kf} keyframe)"
        )
        return chunks

    # ── Serialization (for caching ASR results) ───────────────────────────

    def serialize_segments(self, segments: List[ASRSegment]) -> List[dict]:
        """Serialize ASR segments to JSON-safe dicts."""
        return [{"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker} for s in segments]

    @staticmethod
    def deserialize_segments(data: List[dict]) -> List[ASRSegment]:
        """Deserialize ASR segments from JSON dicts."""
        return [ASRSegment(
            start=d["start"], end=d["end"], text=d["text"],
            speaker=d.get("speaker", ""),
        ) for d in data]

    # ── Speaker Diarization Pipeline ─────────────────────────────────────
    #
    # ─────────────────────────────────────────────────────────────────────

    # Cached gguf diarizer instance
    _gguf_diarizer = None
    _gguf_diarizer_model_path: Optional[str] = None

    DIARIZATION_MAX_DURATION_SEC = float(
        os.environ.get("MEDIA_DIARIZATION_MAX_DURATION", "900")
    )

    def transcribe_with_diarization(
        self,
        audio_path: str,
        whisper_model_path: Optional[str] = None,
        num_speakers: Optional[int] = None,
    ) -> List[ASRSegment]:
        asr_segments = self._run_asr(audio_path)
        if not asr_segments:
            return []

        diar_mode = os.environ.get("MEDIA_ENABLE_DIARIZATION", "").strip().lower()
        if not diar_mode:
            return asr_segments

        duration = self._get_media_duration(audio_path)
        if duration > self.DIARIZATION_MAX_DURATION_SEC:
            logger.warning(
                f"[MediaExpert] Diarization skipped: audio {duration/60:.1f}min > "
                f"{self.DIARIZATION_MAX_DURATION_SEC/60:.0f}min threshold. "
                "Returning ASR only. Adjust MEDIA_DIARIZATION_MAX_DURATION to override."
            )
            return asr_segments

        if diar_mode == "gguf":
            return self._run_gguf_diarization(audio_path, asr_segments, num_speakers)

        logger.warning(
            f"[MediaExpert] Unknown MEDIA_ENABLE_DIARIZATION='{diar_mode}' — returning ASR only. "
            "Valid values: 'gguf'"
        )
        return asr_segments

    def _run_gguf_diarization(
        self,
        audio_path: str,
        asr_segments: List["ASRSegment"],
        num_speakers: Optional[int] = None,
    ) -> List["ASRSegment"]:
        pyannote_gguf = os.path.join(
            self._WHISPER_MODEL_DIR, "pyannote-seg-3.0.gguf"
        )
        if not os.path.isfile(pyannote_gguf):
            logger.warning(
                f"[MediaExpert] GGUF diarizer model not found at {pyannote_gguf} "
                "— returning ASR only. Run download to get pyannote-seg-3.0.gguf."
            )
            return asr_segments

        try:
            from core.media.gguf_diarizer import GGUFDiarizer
            if (
                MediaExpert._gguf_diarizer is None
                or MediaExpert._gguf_diarizer_model_path != pyannote_gguf
            ):
                t0 = time.time()
                MediaExpert._gguf_diarizer = GGUFDiarizer(pyannote_gguf)
                MediaExpert._gguf_diarizer_model_path = pyannote_gguf
                logger.info(
                    f"[MediaExpert] GGUFDiarizer initialized in {time.time()-t0:.2f}s"
                )

            t0 = time.time()
            diar_segments = MediaExpert._gguf_diarizer.diarize(audio_path)
            logger.info(
                f"[MediaExpert] GGUF Diarization done in {time.time()-t0:.1f}s: "
                f"{len(diar_segments)} segments, "
                f"{len({s['speaker'] for s in diar_segments})} speakers"
            )

            asr_as_dict = [
                {"start": s.start, "end": s.end, "text": s.text} for s in asr_segments
            ]
            merged = MediaExpert._gguf_diarizer.merge_with_asr(diar_segments, asr_as_dict)

            result: List[ASRSegment] = []
            for m, orig in zip(merged, asr_segments):
                result.append(ASRSegment(
                    start=orig.start,
                    end=orig.end,
                    text=orig.text,
                    speaker=m.get("speaker", ""),
                ))
            return result

        except Exception as e:
            logger.error(
                f"[MediaExpert] GGUF diarization failed: {e} — returning ASR only",
                exc_info=True,
            )
            return asr_segments

    # ── Meeting Minutes Generation ────────────────────────────────────────

    def generate_meeting_minutes(
        self,
        segments: List[ASRSegment],
        file_name: str = "",
        language: str = "zh",
        speaker_names: Optional[Dict[str, str]] = None,
    ) -> str:
        if not self._llm_client:
            return self._format_transcript_only(segments, speaker_names)

        # Build diarized transcript text
        transcript_lines = []
        for seg in segments:
            speaker_label = seg.speaker or "Unknown"
            if speaker_names and speaker_label in speaker_names:
                speaker_label = speaker_names[speaker_label]
            start_str = f"{int(seg.start // 60):02d}:{int(seg.start % 60):02d}"
            transcript_lines.append(f"[{start_str}] {speaker_label}: {seg.text}")

        transcript_text = "\n".join(transcript_lines)
        max_chars = 6000
        if len(transcript_text) > max_chars:
            transcript_text = transcript_text[:max_chars] + "\n...[截断]"

        has_speakers = any(seg.speaker for seg in segments)

        if language == "zh":
            prompt = (
                f"以下是一段{'带说话人标注的' if has_speakers else ''}会议录音转录{'（来自文件：' + file_name + '）' if file_name else ''}：\n\n"
                f"```\n{transcript_text}\n```\n\n"
                f"请生成一份结构化的会议纪要，包含：\n"
                f"1. **会议概要**（2-3句话总结主题）\n"
                f"2. **主要讨论内容**（按话题分段）\n"
                f"3. **关键决议/行动项**（如有）\n"
                f"4. **待跟进事项**（如有）\n\n"
                f"请用中文输出，格式清晰，不要重复转录原文。"
            )
        else:
            prompt = (
                f"Below is a {'speaker-diarized ' if has_speakers else ''}meeting transcript"
                f"{' from file: ' + file_name if file_name else ''}:\n\n"
                f"```\n{transcript_text}\n```\n\n"
                f"Generate structured meeting minutes including:\n"
                f"1. **Summary** (2-3 sentences on main topics)\n"
                f"2. **Key Discussion Points** (organized by topic)\n"
                f"3. **Decisions / Action Items** (if any)\n"
                f"4. **Follow-ups** (if any)\n\n"
                f"Output in English. Be concise, do not repeat verbatim transcript."
            )

        try:
            index_model_id = self._get_index_model_id()
            if not index_model_id:
                return self._format_transcript_only(segments, speaker_names)
            response = self._llm_client.chat.completions.create(
                model=index_model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.3,
                stream=False,
            )
            raw = (response.choices[0].message.content or "").strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return raw
        except Exception as e:
            logger.error(f"[MediaExpert] Meeting minutes LLM failed: {e}")
            return self._format_transcript_only(segments, speaker_names)

    @staticmethod
    def _format_transcript_only(
        segments: List[ASRSegment],
        speaker_names: Optional[Dict[str, str]] = None,
    ) -> str:
        """Fallback: format diarized transcript as simple text when LLM unavailable."""
        lines = ["# 会议转录\n"]
        current_speaker = None
        for seg in segments:
            speaker = seg.speaker or "Unknown"
            if speaker_names and speaker in speaker_names:
                speaker = speaker_names[speaker]
            start_str = f"{int(seg.start // 60):02d}:{int(seg.start % 60):02d}"
            if speaker != current_speaker:
                lines.append(f"\n**{speaker}** [{start_str}]")
                current_speaker = speaker
            lines.append(f"  {seg.text}")
        return "\n".join(lines)
