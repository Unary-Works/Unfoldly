from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FollowupHint:
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    expires_turn: int = 0
    uses_left: int = 1


@dataclass
class SessionState:
    session_id: Optional[str]
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_search_results: List[Dict[str, Any]] = field(default_factory=list)
    recent_search_result_sets: List[List[Dict[str, Any]]] = field(default_factory=list)
    count_scope_context: Optional[Dict[str, Any]] = None
    prompt_language: Optional[str] = None
    followup_hint: Optional[FollowupHint] = None
    active_paths: Tuple[str, ...] = field(default_factory=tuple)

    def clear_runtime(self, *, clear_history: bool = True, clear_language: bool = False) -> None:
        if clear_history:
            self.history.clear()
        self.last_search_results.clear()
        self.recent_search_result_sets.clear()
        self.count_scope_context = None
        self.followup_hint = None
        if clear_language:
            self.prompt_language = None
