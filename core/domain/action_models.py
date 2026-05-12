from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QueryExecutionMode(str, Enum):
    HANDLER = "handler"
    TOOL_AGENT = "tool_agent"
    SEARCH_PIPELINE = "search_pipeline"
    CHAT = "chat"
    CLARIFY = "clarify"
    SYSTEM = "system"


@dataclass(frozen=True)
class QueryContext:
    question: str
    normalized_question: str
    session_id: Optional[str]
    prompt_language: str
    user_language: str
    active_paths: List[str]
    opened_file_path: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_results: List[Dict[str, Any]] = field(default_factory=list)
    total_searchable: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntentDecision:
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    source: str = "intent_analyzer"
    normalized_internal_en: bool = False

    @classmethod
    def from_legacy(cls, payload: Dict[str, Any], source: str = "legacy") -> "IntentDecision":
        data = dict(payload or {})
        return cls(
            action=str(data.get("action") or "").strip() or "search",
            params=dict(data.get("params") or {}),
            confidence=data.get("confidence"),
            source=source,
            normalized_internal_en=bool(data.get("_normalized_internal_en")),
        )


@dataclass(frozen=True)
class ActionRequest:
    action: str
    params: Dict[str, Any]
    query: str
    session_id: Optional[str]
    prompt_language: str
    user_language: str
    execution_mode: QueryExecutionMode
    query_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_intent(
        cls,
        decision: IntentDecision,
        *,
        query_context: QueryContext,
        execution_mode: QueryExecutionMode,
        query_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ActionRequest":
        merged_metadata = dict(query_context.metadata)
        if metadata:
            merged_metadata.update(metadata)
        merged_metadata.setdefault("active_paths", list(query_context.active_paths))
        merged_metadata.setdefault("opened_file_path", query_context.opened_file_path)
        merged_metadata.setdefault("intent_source", decision.source)
        merged_metadata.setdefault("intent_confidence", decision.confidence)
        merged_metadata.setdefault("normalized_internal_en", decision.normalized_internal_en)
        return cls(
            action=decision.action,
            params=dict(decision.params or {}),
            query=query_context.question,
            session_id=query_context.session_id,
            prompt_language=query_context.prompt_language,
            user_language=query_context.user_language,
            execution_mode=execution_mode,
            query_type=query_type,
            metadata=merged_metadata,
        )


@dataclass
class ActionResult:
    ok: bool = True
    query_type: str = "search"
    sources: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
