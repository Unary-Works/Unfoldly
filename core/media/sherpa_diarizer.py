

class SherpaDiarizer:
    """Stub: sherpa-onnx diarizer has been disabled. See module docstring."""

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "SherpaDiarizer is disabled.\n"
            "sherpa-onnx CUDA provider fallback makes Mac performance unacceptable.\n"
            "See core/media/sherpa_diarizer.py for alternative diarization approaches."
        )

    def diarize(self, audio_path: str):
        raise ImportError("SherpaDiarizer is disabled.")

    def merge_with_asr(self, diar_segments, asr_segments):
        raise ImportError("SherpaDiarizer is disabled.")
