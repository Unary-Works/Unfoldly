from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Set, List

@dataclass
class IndexJobState:
    job_id: str
    folder: str
    is_indexing: bool = True
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    eta_seconds: int = 0
    current_file: str = ""
    current_path: str = ""
    started_at: float = 0.0
    finished_at: Optional[float] = None
    error: Optional[str] = None
    message: Optional[str] = None
    skipped_files: Set[str] = field(default_factory=set)
    indexed_paths: List[str] = field(default_factory=list)
    # Sub-progress for media files (keyframe VL analysis)
    current_frame: int = 0
    total_frames: int = 0
    current_audio_sec: float = 0.0
    total_audio_sec: float = 0.0
    stage_rate: float = 0.0
    stage: str = ""  # e.g. "analyzing_frames"

    def to_payload(self) -> Dict[str, Any]:
        d = asdict(self)
        d["completed_files"] = min(d["completed_files"], d["total_files"])
        d["current_audio_sec"] = round(float(d.get("current_audio_sec") or 0.0), 2)
        d["total_audio_sec"] = round(float(d.get("total_audio_sec") or 0.0), 2)
        d["stage_rate"] = round(float(d.get("stage_rate") or 0.0), 2)
        eta = self.eta_seconds
        if not eta:
            if self.is_indexing and self.total_files > self.completed_files:
                eta_str = "…"
            else:
                eta_str = "—"
        elif eta < 60:
            eta_str = f"{eta}s"
        elif eta < 3600:
            eta_str = f"{eta//60}min"
        else:
            eta_str = f"{eta/3600:.1f}h"
        d["eta"] = eta_str
        d.pop("skipped_files", None)
        return d
