"""
SessionMemory — session state repository for the FileAgent.

This module keeps runtime session data in-memory, but the underlying data model
is now the domain-level `SessionState` object rather than scattered dicts.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from core.domain import FollowupHint, SessionState

logger = logging.getLogger(__name__)


class SessionMemory:
    """
    In-memory repository managing `SessionState` objects.

    - `SessionState` is the domain model for one conversation/session.
    - `SessionMemory` is only responsible for indexing, lookup, cleanup, and
      lightweight mutation helpers used by the current FileAgent code paths.
    """

    _MAX_SESSIONS = 50
    _CLEANUP_THRESHOLD = 20

    def __init__(self):
        self._states: Dict[str, SessionState] = {}
        self._global_state = SessionState(session_id=None)

    def _gc_states(self) -> None:
        """Garbage-collect oldest sessions if the store grows too large."""
        if len(self._states) > self._MAX_SESSIONS:
            sorted_keys = sorted(self._states.keys())
            for old_key in sorted_keys[:self._CLEANUP_THRESHOLD]:
                self._states.pop(old_key, None)

    def get_state(self, session_id: Optional[str]) -> SessionState:
        """Return the mutable session domain object for a session."""
        sid = (session_id or "").strip()
        if not sid:
            return self._global_state
        self._gc_states()
        state = self._states.get(sid)
        if state is None:
            state = SessionState(session_id=sid)
            self._states[sid] = state
        return state

    # ── History ───────────────────────────────────────────────────────────

    def get_history_ref(self, session_id: Optional[str]) -> List[Dict[str, Any]]:
        return self.get_state(session_id).history

    def clear_history(self) -> None:
        self._global_state.history.clear()
        for state in self._states.values():
            state.history.clear()

    # ── Last Search Results ───────────────────────────────────────────────

    def get_last_search_results_ref(self, session_id: Optional[str]) -> List[Dict[str, Any]]:
        return self.get_state(session_id).last_search_results

    def set_last_search_results(self, session_id: Optional[str], results: List[Dict[str, Any]]) -> None:
        cleaned: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in list(results or []):
            if not isinstance(row, dict):
                continue
            key = str(row.get("file_path") or row.get("file_name") or "").strip()
            if key:
                if key in seen:
                    continue
                seen.add(key)
            cleaned.append(row)
        state = self.get_state(session_id)
        state.last_search_results.clear()
        state.last_search_results.extend(cleaned)
        if cleaned:
            snapshot = [dict(row) for row in cleaned[:50] if isinstance(row, dict)]
            snapshot_key = tuple(
                str(row.get("file_path") or row.get("file_name") or "").strip()
                for row in snapshot
            )
            prev_key = tuple(
                str(row.get("file_path") or row.get("file_name") or "").strip()
                for row in (state.recent_search_result_sets[-1] if state.recent_search_result_sets else [])
            )
            if snapshot_key and snapshot_key != prev_key:
                state.recent_search_result_sets.append(snapshot)
                del state.recent_search_result_sets[:-6]

    def get_recent_search_result_sets(self, session_id: Optional[str], *, limit: int = 4) -> List[List[Dict[str, Any]]]:
        state = self.get_state(session_id)
        n = max(1, int(limit or 1))
        return [
            [dict(row) for row in batch if isinstance(row, dict)]
            for batch in state.recent_search_result_sets[-n:]
        ]

    # ── Count Scope Context ───────────────────────────────────────────────

    def set_count_scope_context(
        self, session_id: Optional[str], context: Optional[Dict[str, Any]]
    ) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        state = self.get_state(sid)
        state.count_scope_context = dict(context) if isinstance(context, dict) and context else None

    def get_count_scope_context(self, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        ctx = self.get_state(sid).count_scope_context
        if not isinstance(ctx, dict):
            return None
        return dict(ctx)

    def clear_count_scope_context(self, session_id: Optional[str], reason: str = "manual") -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        state = self.get_state(sid)
        existed = isinstance(state.count_scope_context, dict) and bool(state.count_scope_context)
        state.count_scope_context = None
        if existed:
            logger.info(f"[count-scope] cleared sid={sid} reason={reason}")

    # ── Active Paths / Session Scope ──────────────────────────────────────

    def sync_active_paths(self, session_id: Optional[str], active_paths: Optional[List[str]]) -> bool:
        """
        Persist the latest selected source paths for a session.

        Returns True when the selection changed compared with the previous turn.
        """
        sid = (session_id or "").strip()
        if not sid:
            return False
        state = self.get_state(sid)
        curr_paths_key = tuple(sorted(active_paths or []))
        prev_paths_key = state.active_paths
        state.active_paths = curr_paths_key
        return bool(prev_paths_key) and prev_paths_key != curr_paths_key

    def clear_session_runtime_state(
        self,
        session_id: Optional[str],
        *,
        clear_history: bool = True,
        clear_language: bool = False,
        reason: str = "manual",
    ) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        state = self.get_state(sid)
        state.clear_runtime(clear_history=clear_history, clear_language=clear_language)
        logger.info(
            f"[session-state] cleared sid={sid} reason={reason} "
            f"clear_history={clear_history} clear_language={clear_language}"
        )

    @staticmethod
    def build_count_scope_from_sources(sources: list) -> Dict[str, Any]:
        cat_counter: dict = Counter()
        cat_samples: dict = defaultdict(list)
        sample_limit = max(1, int(os.getenv("COUNT_SCOPE_SAMPLES_PER_CATEGORY", "10")))
        for s in sources:
            cat = str(s.get("doc_category") or "other").strip() or "other"
            cat_counter[cat] += 1
            if len(cat_samples[cat]) < sample_limit:
                cat_samples[cat].append(
                    {
                        "file_name": str(s.get("file_name") or ""),
                        "file_path": str(s.get("file_path") or ""),
                        "doc_summary": str(s.get("doc_summary") or "")[:200],
                    }
                )
        category_counts = [
            {"category": cat, "count": cnt}
            for cat, cnt in cat_counter.most_common(24)
        ]
        return {
            "kind": "count_all",
            "total_files": len(sources),
            "category_counts": category_counts,
            "samples_by_category": dict(cat_samples),
            "stored_at": float(time.time()),
        }

    # ── Followup Hints ────────────────────────────────────────────────────

    def set_followup_hint(
        self,
        session_id: Optional[str],
        *,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        ttl_turns: int = 2,
        uses: int = 1,
    ) -> None:
        sid = (session_id or "").strip() or "__default__"
        state = self.get_state(sid if sid != "__default__" else None)
        current_turn = len(state.history)
        state.followup_hint = FollowupHint(
            action=str(action or "").strip(),
            params=dict(params or {}),
            expires_turn=current_turn + max(1, int(ttl_turns or 1)),
            uses_left=max(1, int(uses or 1)),
        )
        logger.info(
            f"[followup-guard] stage=hint_lifecycle decision=set sid={sid} "
            f"action={action} ttl_turns={ttl_turns} uses={uses} turn={current_turn}"
        )

    def get_followup_hint(self, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        sid = (session_id or "").strip() or "__default__"
        state = self.get_state(sid if sid != "__default__" else None)
        hint = state.followup_hint
        if not hint:
            return None
        current_turn = len(state.history)
        if hint.uses_left <= 0 or (hint.expires_turn and current_turn > hint.expires_turn):
            state.followup_hint = None
            return None
        return {
            "action": hint.action,
            "params": dict(hint.params),
            "expires_turn": hint.expires_turn,
            "uses_left": hint.uses_left,
        }

    def consume_followup_hint(self, session_id: Optional[str]) -> None:
        sid = (session_id or "").strip() or "__default__"
        state = self.get_state(sid if sid != "__default__" else None)
        hint = state.followup_hint
        if not hint:
            return
        hint.uses_left -= 1
        if hint.uses_left <= 0:
            state.followup_hint = None
            logger.info(f"[followup-guard] stage=hint_lifecycle decision=consume_end sid={sid}")
            return
        logger.info(f"[followup-guard] stage=hint_lifecycle decision=consume sid={sid} uses_left={hint.uses_left}")

    def clear_followup_hint(self, session_id: Optional[str], reason: str = "manual") -> None:
        sid = (session_id or "").strip() or "__default__"
        state = self.get_state(sid if sid != "__default__" else None)
        existed = state.followup_hint is not None
        state.followup_hint = None
        if existed:
            logger.info(f"[followup-guard] stage=hint_lifecycle decision=clear sid={sid} reason={reason}")

    # ── Logging ───────────────────────────────────────────────────────────

    @staticmethod
    def log_followup_guard(
        *,
        stage: str,
        decision: str,
        reason: str,
        session_id: Optional[str],
        question: str,
        action_before: str = "",
        action_after: str = "",
        hint_action: str = "",
        brief_followup: Optional[bool] = None,
        last_results_count: int = 0,
    ) -> None:
        sid = (session_id or "").strip() or "__default__"
        q = " ".join(str(question or "").split())
        if len(q) > 120:
            q = q[:120] + "..."
        extra = (
            f" action_before={action_before} action_after={action_after}"
            f" hint_action={hint_action} brief_followup={brief_followup}"
            f" last_results={last_results_count}"
        )
        logger.info(
            f"[followup-guard] stage={stage} decision={decision} reason={reason} sid={sid}"
            f"{extra} q={json.dumps(q, ensure_ascii=False)}"
        )

    # ── Language ──────────────────────────────────────────────────────────

    def remember_prompt_language(self, session_id: Optional[str], prompt_language: str) -> None:
        from config.prompts import normalize_prompt_language

        sid = (session_id or "").strip()
        if not sid:
            return
        state = self.get_state(sid)
        state.prompt_language = normalize_prompt_language(prompt_language, fallback="en")

    def get_remembered_language(self, session_id: Optional[str]) -> Optional[str]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        return self.get_state(sid).prompt_language

    # ── Global Reset ──────────────────────────────────────────────────────

    def clear_all(self) -> None:
        self._global_state = SessionState(session_id=None)
        self._states.clear()
