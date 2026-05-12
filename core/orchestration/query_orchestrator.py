from __future__ import annotations

import os
import json
import re
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
from core.intent.expert_arbiter import ExpertIntentArbiter
from core.domain.action_models import (
    ActionRequest,
    IntentDecision,
    QueryContext,
    QueryExecutionMode,
)
from core.handlers.context import HandlerContext
from core.intent.entity_experts import CategoryListExpert
from core.retrieval.filename_canonicalizer import looks_like_specific_filename_candidate
from core.intent.search_scope_disambiguation import (
    clarify_message,
    has_explicit_previous_scope,
    has_explicit_selected_scope,
    has_prior_entity_reference,
    looks_like_contextual_followup_request,
    looks_like_personal_attribute_request,
    looks_like_search_request,
    resolve_pending_scope_choice,
    result_paths,
    selected_scope_matches_previous,
)

logger = get_logger()


_HISTORY_RESULT_CONTEXT_RE = re.compile(
    r"\bfound\s+\d+\s+(?:relevant\s+)?(?:files?|documents?|results?)\b"
    r"|\brelevant\s+(?:files?|documents?|results?)\b"
    r"|\bno\s+(?:highly\s+)?relevant\s+(?:indexed\s+)?(?:content|files?|documents?|results?)\b"
    r"|\bmatched\s+(?:files?|documents?|results?)\b"
    r"|匹配文件|相关文件|相关文档|找到\s*\d+\s*(?:个|份)?(?:文件|文档|结果)|未找到.*(?:文件|文档|资料|内容)",
    re.IGNORECASE,
)


class QueryOrchestrator:
    """
    Thin orchestration layer that standardizes:
    - query context construction
    - legacy intent payload -> typed action request normalization
    - execution mode selection
    - query_type mapping

    It intentionally does not replace the existing dispatch/search pipeline yet.
    Instead, it gives the existing system a stable protocol surface so future
    refactors can move logic out of dispatch without changing external behavior.
    """

    _HANDLER_ACTIONS = {
        "translate_response",
        "summarize_all",
        "db_clear",
        "clarify",
        "chat",
        "process_previous",
        "count",
        "summarize",
        "view_detail",
        "open_file",
        "media_export",
        "media_content_search",
    }

    _TOOL_ACTIONS = {"tools", "tool_agent"}

    _SEARCH_LIKE_ACTIONS = {"search", "media_content_search"}

    _QUERY_TYPE_MAP = {
        "translate_response": "translate",
        "summarize_all": "summarize_all",
        "db_clear": "db_clear",
        "clarify": "clarify",
        "chat": "chat",
        "process_previous": "process",
        "count": "count",
        "summarize": "summarize",
        "view_detail": "detail",
        "open_file": "open_file",
        "media_export": "media_export",
        "media_content_search": "media_content_search",
        "search": "search",
        "tools": "tools",
        "tool_agent": "agent",
    }

    def __init__(self, agent: Any):
        self.agent = agent
        self.expert_arbiter = ExpertIntentArbiter(agent)

    def build_query_context(
        self,
        *,
        question: str,
        normalized_question: Optional[str],
        session_id: Optional[str],
        prompt_language: str,
        user_language: str,
        active_paths: Optional[List[str]],
        opened_file_path: Optional[str] = None,
        total_searchable: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QueryContext:
        history = list(self.agent._get_history_ref(session_id))
        last_results = list(self.agent._get_last_search_results_ref(session_id))
        return QueryContext(
            question=str(question or ""),
            normalized_question=str(normalized_question or question or ""),
            session_id=session_id,
            prompt_language=prompt_language,
            user_language=user_language,
            active_paths=list(active_paths or []),
            opened_file_path=opened_file_path,
            history=history,
            last_results=last_results,
            total_searchable=total_searchable,
            metadata=dict(metadata or {}),
        )

    def analyze(self, query_context: QueryContext) -> Tuple[ActionRequest, Dict[str, Any]]:
        analyze_t0 = time.time()
        arbiter_t0 = time.time()
        intent_payload = None
        intent_source = "arbiter"
        arbiter_source = "unresolved"
        pending_choice = self._resolve_pending_search_scope_choice(query_context)
        if pending_choice is not None:
            intent_payload = {
                "action": pending_choice.action,
                "params": pending_choice.params,
                "confidence": 1.0,
            }
            intent_source = "scope_disambiguation"
            arbiter_source = pending_choice.reason
        else:
            clarify_gate = self._run_pre_intent_clarify_gate(query_context)
            if isinstance(clarify_gate, dict):
                gate_decision = str(clarify_gate.get("decision") or "").strip().lower()
                query_context.metadata["pre_intent_clarify_gate"] = gate_decision
                query_context.metadata["pre_intent_clarify_raw"] = str(clarify_gate.get("raw") or "")
                if gate_decision == "true":
                    clarify_params = {
                        "question": clarify_message(query_context.user_language or query_context.prompt_language),
                        "query": query_context.question,
                        "_clarify_kind": "pre_intent",
                        "_clarify_context": clarify_gate.get("context") or {},
                        "_clarify_with_context": True,
                    }
                    self._maybe_set_pre_intent_scope_hint(query_context, clarify_params)
                    decision = IntentDecision(
                        action="clarify",
                        params=clarify_params,
                        confidence=1.0,
                        source="pre_intent_clarify_gate",
                    )
                    request = ActionRequest.from_intent(
                        decision,
                        query_context=query_context,
                        execution_mode=self.execution_mode_for(decision.action),
                        query_type=self.query_type_for(decision.action),
                        metadata={
                            "arbiter_source": "pre_intent_clarify_gate",
                            "intent_source": "pre_intent_clarify_gate",
                            "clarify_gate_decision": "true",
                            "used_legacy_fallback": False,
                            "timing": {
                                "arbiter_ms": 0,
                                "legacy_fallback_ms": 0,
                                "normalize_ms": 0,
                                "analyze_total_ms": int((time.time() - analyze_t0) * 1000),
                            },
                        },
                    )
                    return request, {"action": "clarify", "params": clarify_params, "confidence": 1.0}

            try:
                intent_payload, arbiter_source = self.expert_arbiter.analyze(query_context)
            except Exception as e:
                logger.warning(f"[QueryOrchestrator] expert arbiter failed: {e}")
                intent_payload = None
                arbiter_source = "arbiter_error"
        arbiter_ms = int((time.time() - arbiter_t0) * 1000)

        allow_legacy_fallback = os.environ.get("FILEAGENT_ENABLE_LEGACY_INTENT_FALLBACK", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        legacy_fallback_ms = 0
        if intent_payload is None and allow_legacy_fallback:
            legacy_t0 = time.time()
            logger.info("[QueryOrchestrator] arbiter returned no intent, using legacy fallback")
            intent_payload = self.agent._analyze_intent_with_context(
                query_context.normalized_question,
                session_id=query_context.session_id,
                prompt_language=query_context.prompt_language,
            )
            intent_source = "legacy_fallback"
            arbiter_source = "legacy_fallback"
            legacy_fallback_ms = int((time.time() - legacy_t0) * 1000)

        if not isinstance(intent_payload, dict):
            if not query_context.active_paths:
                intent_payload = {"action": "count", "params": {"category": "all"}}
                arbiter_source = "default_count_all"
            else:
                intent_payload = {"action": "search", "params": {"query": query_context.normalized_question}}
                arbiter_source = "default_search_active_scope"
            intent_source = "orchestrator_default"

        normalize_t0 = time.time()
        normalized = self.agent._normalize_intent_to_internal_en(
            query_context.normalized_question,
            intent_payload,
            session_id=query_context.session_id,
        )
        normalize_ms = int((time.time() - normalize_t0) * 1000)
        decision = IntentDecision.from_legacy(normalized, source=intent_source)
        decision, scope_guard_metadata = self._guard_ambiguous_search_scope(query_context, decision)
        decision_params = dict(decision.params or {})
        expert_route = str(decision_params.get("_expert_route") or "").strip()
        dispatch_skill = str(decision_params.get("_skill_name") or "").strip()
        dispatch_scope = str(decision_params.get("_scope_kind") or decision_params.get("scope") or "").strip()
        dispatch_operation = str(decision_params.get("_context_operation") or decision_params.get("operation") or "").strip()
        dispatch_reason = str(decision_params.get("_dispatch_reason") or "").strip()
        candidate_scopes = list(decision_params.get("_candidate_scopes") or [])
        request = ActionRequest.from_intent(
            decision,
            query_context=query_context,
            execution_mode=self.execution_mode_for(decision.action),
            query_type=self.query_type_for(decision.action),
            metadata={
                "normalized_intent": normalized,
                "arbiter_source": arbiter_source,
                "intent_source": intent_source,
                "expert_route": expert_route,
                "dispatch_skill": dispatch_skill,
                "dispatch_scope": dispatch_scope,
                "dispatch_operation": dispatch_operation,
                "dispatch_reason": dispatch_reason,
                "dispatch_candidate_scopes": candidate_scopes[:6],
                "used_legacy_fallback": intent_source == "legacy_fallback",
                **scope_guard_metadata,
                "timing": {
                    "arbiter_ms": arbiter_ms,
                    "legacy_fallback_ms": legacy_fallback_ms,
                    "normalize_ms": normalize_ms,
                    "analyze_total_ms": int((time.time() - analyze_t0) * 1000),
                },
            },
        )
        return request, normalized

    def _count_scope_context(self, session_id: Optional[str]) -> Dict[str, Any]:
        try:
            getter = getattr(self.agent, "_get_count_scope_context", None)
            if callable(getter):
                ctx = getter(session_id)
                if isinstance(ctx, dict):
                    return dict(ctx)
        except Exception as exc:
            logger.debug("[QueryOrchestrator] count-scope lookup failed: %s", exc)
        return {}

    def _resolve_pending_search_scope_choice(self, query_context: QueryContext):
        try:
            getter = getattr(self.agent, "_get_followup_hint", None)
            if not callable(getter):
                return None
            hint = getter(query_context.session_id)
            if not isinstance(hint, dict):
                return None
            if str(hint.get("action") or "") != "search_scope_clarify":
                return None
            decision = resolve_pending_scope_choice(
                query_context.question,
                dict(hint.get("params") or {}),
            )
            if decision is None:
                if looks_like_personal_attribute_request(query_context.question) or looks_like_search_request(query_context.question):
                    clearer = getattr(self.agent, "_clear_followup_hint", None)
                    if callable(clearer):
                        clearer(query_context.session_id, reason="search_scope_choice_replaced")
                return None
            clearer = getattr(self.agent, "_clear_followup_hint", None)
            if callable(clearer):
                clearer(query_context.session_id, reason="search_scope_choice_resolved")
            return decision
        except Exception as exc:
            logger.debug("[QueryOrchestrator] pending search-scope choice failed: %s", exc)
            return None

    def _set_search_scope_clarify_hint(
        self,
        query_context: QueryContext,
        *,
        query: str,
        search_params: Dict[str, Any],
    ) -> None:
        try:
            setter = getattr(self.agent, "_set_followup_hint", None)
            if callable(setter):
                setter(
                    query_context.session_id,
                    action="search_scope_clarify",
                    params={
                        "query": query,
                        "search_params": dict(search_params or {}),
                    },
                    ttl_turns=1,
                    uses=1,
                )
        except Exception as exc:
            logger.debug("[QueryOrchestrator] pending search-scope hint failed: %s", exc)

    def _build_clarify_context(self, query_context: QueryContext, *, query: Optional[str] = None) -> Dict[str, Any]:
        last_results = list(query_context.last_results or [])
        active_paths = list(query_context.active_paths or [])
        previous_files = []
        for row in last_results[:8]:
            if not isinstance(row, dict):
                continue
            previous_files.append(
                {
                    "file_name": str(row.get("file_name") or row.get("name") or "").strip(),
                    "file_path": str(row.get("file_path") or row.get("path") or "").strip(),
                    "doc_summary": str(row.get("doc_summary") or row.get("summary") or "").strip()[:220],
                }
            )

        selected_examples = []
        for path in active_paths[:8]:
            raw_path = str(path or "").strip()
            if raw_path:
                selected_examples.append({"file_name": os.path.basename(raw_path), "file_path": raw_path})

        previous_paths = result_paths(last_results)
        same_scope = selected_scope_matches_previous(
            active_paths=active_paths,
            last_results=last_results,
            count_scope_context=self._count_scope_context(query_context.session_id),
            total_searchable=query_context.total_searchable,
        )
        recent_history = []
        for item in list(query_context.history or [])[-4:]:
            if not isinstance(item, dict):
                continue
            recent_history.append(
                {
                    "q": str(item.get("q") or item.get("content") or "").strip()[:180],
                    "a": str(item.get("a") or "").strip()[:220],
                }
            )

        return {
            "user_question": str(query_context.question or ""),
            "candidate_query": str(query or query_context.normalized_question or query_context.question or ""),
            "user_language": str(query_context.user_language or query_context.prompt_language or ""),
            "previous_result_count": len(previous_paths),
            "previous_results": previous_files,
            "current_selected_count": len([p for p in active_paths if str(p or "").strip()]),
            "current_selected_examples": selected_examples,
            "total_searchable": query_context.total_searchable,
            "selected_matches_previous": bool(same_scope),
            "recent_history": recent_history,
        }

    def _format_clarify_context(self, context: Dict[str, Any]) -> str:
        prev = context.get("previous_results") or []
        selected = context.get("current_selected_examples") or []
        history = context.get("recent_history") or []
        lines = [
            f"User question: {context.get('user_question', '')}",
            f"Candidate query: {context.get('candidate_query', '')}",
            f"Previous result count: {context.get('previous_result_count', 0)}",
            f"Current selected count: {context.get('current_selected_count', 0)}",
            f"Total searchable: {context.get('total_searchable', '')}",
            f"Selected matches previous: {context.get('selected_matches_previous', False)}",
        ]
        if prev:
            lines.append("Previous results:")
            for item in prev:
                name = str(item.get("file_name") or os.path.basename(str(item.get("file_path") or "")) or "")
                summary = str(item.get("doc_summary") or "")
                path = str(item.get("file_path") or "")
                lines.append(f"- {name} | {path} | {summary}")
        if selected:
            lines.append("Current selected examples:")
            for item in selected:
                lines.append(f"- {item.get('file_name', '')} | {item.get('file_path', '')}")
        if history:
            lines.append("Recent conversation:")
            for item in history:
                lines.append(f"- user: {item.get('q', '')}")
                if item.get("a"):
                    lines.append(f"  assistant: {item.get('a', '')}")
        return "\n".join(lines)

    def _build_clarify_gate_prompt(self, context_text: str) -> str:
        return (
            "ClarifyGate v2. You are a fast boolean gate that runs BEFORE intent routing.\n"
            "Your entire output MUST be exactly one lowercase token: true or false.\n"
            "No JSON. No markdown. No punctuation. No explanation.\n\n"
            "Question: should the assistant ask a clarification question now?\n\n"
            "Return false when the normal pipeline can safely continue, including:\n"
            "- clear search/list/count/summarize/translate/open/tool requests;\n"
            "- follow-up questions answerable from previous results or recent conversation;\n"
            "- pronoun/entity follow-ups (he/she/his/her/they/them/it/this/that/他/她/它/这些/上面) when context identifies the subject or scope;\n"
            "- ordinal/deictic result references (the first paper/report/file, this report, that file, 第一篇, 这篇报告) when previous results exist;\n"
            "- person/profile attributes: email, phone, address, home/residence, school, graduation school, education, degree, major, company, employer, position;\n"
            "- explicit scope: previous results, above files, these/them/it, current selected files;\n"
            "- current selected files are the same set as previous results;\n"
            "- there is enough context for retrieval to try even if the final answer may say not found.\n\n"
            "Return true ONLY when clarification is required before any safe routing, for example:\n"
            "- the latest message is too vague to form any action or query (e.g. 'that one', 'do it', 'yes' with no pending choice);\n"
            "- a new search is requested and previous results and current selected files are different plausible scopes, with no words indicating previous/current scope;\n"
            "- multiple incompatible actions or targets are requested and context cannot resolve them.\n\n"
            "Bias toward false when context makes a reasonable route possible. Never use true just because the query uses a pronoun; use the context.\n\n"
            "<Context>\n"
            f"{context_text}\n"
            "</Context>\n\n"
            "Output one token:"
        )

    @staticmethod
    def _looks_like_scope_ambiguous_summary_request(question: str) -> bool:
        q = str(question or "").strip()
        if not q or looks_like_search_request(q):
            return False
        ql = q.lower()
        return bool(
            re.search(
                r"\b(summar(?:y|ize|ise|ization)|overview|recap|analy[sz]e|analysis|"
                r"explain|describe|tell\s+me\s+about)\b",
                ql,
                re.IGNORECASE,
            )
            or re.search(
                r"(总结|概括|归纳|汇总|分析|解释|说明|介绍|讲讲|说说|梳理|要点|重点|结论)",
                q,
            )
        )

    @staticmethod
    def _looks_like_explicit_result_followup_scope(question: str) -> bool:
        q = str(question or "").strip()
        if not q:
            return False
        return bool(
            re.search(
                r"\b(?:this|that|the)\s+"
                r"(?:report|paper|document|doc|file|resume|invoice|presentation|image|photo|video|audio|result|item)\b"
                r"|\b(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|previous)\s+"
                r"(?:one|result|file|document|doc|report|paper|item|resume|invoice|presentation|image|photo|video|audio)\b"
                r"|\b(?:result|file|document|doc|report|paper)\s*(?:#\s*)?\d+\b",
                q,
                re.IGNORECASE,
            )
            or re.search(
                r"(?:这|那)(?:篇|份|个|张|条|项)?"
                r"(?:报告|论文|文档|文件|资料|简历|发票|表格|图片|照片|视频|音频|结果)"
                r"|(?:第[一二三四五六七八九十百千万\d]+|前[一二三四五六七八九十百千万\d]+|最后一?)"
                r"(?:篇|份|个|张|条|项|部)?"
                r"(?:报告|论文|文档|文件|资料|简历|发票|表格|图片|照片|视频|音频|结果)?",
                q,
            )
        )

    @staticmethod
    def _looks_like_clear_inventory_request(question: str) -> bool:
        try:
            intent = CategoryListExpert.analyze(question, has_content_qualifier=False)
        except Exception:
            return False
        if not intent:
            return False
        action = str(intent.get("action") or "").strip()
        params = intent.get("params") if isinstance(intent.get("params"), dict) else {}
        if action == "count":
            return True
        return action == "search" and str(params.get("_inventory_mode") or "").strip().lower() == "category"

    def _parse_clarify_gate_bool(self, raw: str) -> Optional[str]:
        text = str(raw or "").strip()
        if not text:
            return None
        lowered = text.lower().strip()
        cleaned = re.sub(r"^```(?:json|text)?|```$", "", lowered.strip(), flags=re.IGNORECASE).strip()
        tokens = re.findall(r"\b(true|false)\b", cleaned, flags=re.IGNORECASE)
        unique_tokens = {tok.lower() for tok in tokens}
        if len(unique_tokens) > 1:
            return None
        if len(unique_tokens) == 1:
            return next(iter(unique_tokens))

        def _json_bool(value: Any) -> Optional[str]:
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, dict):
                for key in (
                    "decision",
                    "answer",
                    "clarify",
                    "should_clarify",
                    "needs_clarification",
                    "need_clarification",
                    "need_clarify",
                ):
                    if key in value:
                        parsed = _json_bool(value.get(key))
                        if parsed:
                            return parsed
            return None

        try:
            parsed_json = json.loads(text)
            parsed_bool = _json_bool(parsed_json)
            if parsed_bool:
                return parsed_bool
        except Exception:
            pass

        obj_start = text.find("{")
        obj_end = text.rfind("}")
        if 0 <= obj_start < obj_end:
            try:
                parsed_json = json.loads(text[obj_start : obj_end + 1])
                parsed_bool = _json_bool(parsed_json)
                if parsed_bool:
                    return parsed_bool
            except Exception:
                pass

        anchored = re.match(
            r"^\s*(?:answer|decision|result|output|clarify|should_clarify|needs_clarification)?"
            r"\s*[:=\-]?\s*(true|false)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if anchored:
            return anchored.group(1).lower()

        tagged = re.search(r"<(?:answer|decision|result)>\s*(true|false)\s*</(?:answer|decision|result)>", cleaned)
        if tagged:
            return tagged.group(1).lower()
        return None

    def _extract_explicit_filename_ref(self, question: str) -> Optional[Dict[str, Any]]:
        extractor = getattr(self.agent, "_extract_explicit_file_reference", None)
        if not callable(extractor):
            return None
        try:
            ref = extractor(str(question or ""))
        except Exception:
            return None
        if not isinstance(ref, dict):
            return None
        raw_name = str(ref.get("raw_name") or ref.get("search_term") or "").strip()
        search_term = str(ref.get("search_term") or raw_name).strip()
        if not raw_name:
            return None
        if re.search(r"\.[A-Za-z0-9]{1,12}$", os.path.basename(raw_name)):
            return ref
        q = str(question or "").strip()
        if re.search(
            r"\b(?:file\s+named|file\s+called|filename|file\s*name|named\s+file|called\s+file)\b"
            r"|文件名|名为|叫.{0,40}(?:文件|文档|pdf|docx|xlsx|csv)",
            q,
            re.IGNORECASE,
        ):
            return ref
        if search_term and looks_like_specific_filename_candidate(search_term):
            explicit_file_scope = bool(
                re.search(
                    r"\b(?:file|document|doc|image|photo|picture|spreadsheet|table|audio|video|clip|recording|song|music)\b"
                    r"|文件|文档|图片|照片|表格|音频|视频|录音|歌曲|音乐",
                    q,
                    re.IGNORECASE,
                )
            )
            if explicit_file_scope:
                return ref
        return None

    def _run_pre_intent_clarify_gate(self, query_context: QueryContext) -> Optional[Dict[str, Any]]:
        context = self._build_clarify_context(query_context)
        context_text = self._format_clarify_context(context)
        has_result_context = bool(
            query_context.last_results
            or self._count_scope_context(query_context.session_id)
            or self._history_has_prior_result_context(query_context.history)
        )
        if self._extract_explicit_filename_ref(query_context.question):
            return {
                "decision": "false",
                "raw": "deterministic_explicit_filename_request",
                "prompt": "",
                "context": context,
            }
        if self._looks_like_clear_inventory_request(query_context.question):
            return {
                "decision": "false",
                "raw": "deterministic_inventory_request",
                "prompt": "",
                "context": context,
            }
        if bool(context.get("selected_matches_previous")):
            return {
                "decision": "false",
                "raw": "deterministic_selected_matches_previous",
                "prompt": "",
                "context": context,
            }
        question = str(query_context.question or "").strip()
        if (
            has_result_context
            and question
            and not looks_like_search_request(question)
            and (
                has_explicit_previous_scope(question)
                or looks_like_contextual_followup_request(question)
            )
        ):
            return {
                "decision": "false",
                "raw": "deterministic_context_followup",
                "prompt": "",
                "context": context,
            }
        if (
            has_result_context
            and question
            and not looks_like_search_request(question)
            and self._looks_like_explicit_result_followup_scope(question)
        ):
            return {
                "decision": "false",
                "raw": "deterministic_context_followup",
                "prompt": "",
                "context": context,
            }
        if self._is_obvious_context_followup(query_context):
            return {
                "decision": "false",
                "raw": "deterministic_context_followup",
                "prompt": "",
                "context": context,
            }
        if (
            (query_context.last_results or self._count_scope_context(query_context.session_id))
            and query_context.active_paths
            and not has_explicit_previous_scope(query_context.question)
            and not has_explicit_selected_scope(query_context.question)
            and self._looks_like_scope_ambiguous_summary_request(query_context.question)
        ):
            return {
                "decision": "true",
                "raw": "deterministic_ambiguous_summary_scope",
                "prompt": "",
                "context": context,
            }
        try:
            from core.skills import MediaFollowupSkill

            media_exts = {
                ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
                ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv",
            }
            active_media_paths = [
                str(path or "").strip()
                for path in list(query_context.active_paths or [])
                if str(path or "").strip() and os.path.splitext(str(path or "").strip())[1].lower() in media_exts
            ]
            if active_media_paths and MediaFollowupSkill.looks_like_generic_overview_query(
                str(query_context.question or ""),
                file_hint=os.path.basename(active_media_paths[0]) if len(active_media_paths) == 1 else "",
            ):
                return {
                    "decision": "false",
                    "raw": "deterministic_media_overview",
                    "prompt": "",
                    "context": context,
                }
        except Exception:
            pass
        llm_getter = getattr(self.agent, "_get_llm_service", None)
        if not callable(llm_getter):
            return None
        try:
            llm = llm_getter(
                detailed=False,
                session_id=query_context.session_id,
                prompt_language=query_context.user_language or query_context.prompt_language,
            )
        except Exception as exc:
            logger.debug("[QueryOrchestrator] pre-intent clarify gate failed: %s", exc)
            return None

        prompt = self._build_clarify_gate_prompt(context_text)
        try:
            try:
                raw = str(llm.generate(prompt, history=[], system_prompt=None) or "").strip()
            except TypeError:
                raw = str(llm.generate(prompt) or "").strip()
        except Exception as exc:
            logger.debug("[QueryOrchestrator] pre-intent clarify gate generate failed: %s", exc)
            return None
        parsed = self._parse_clarify_gate_bool(raw)
        if parsed in {"true", "false"}:
            return {
                "decision": parsed,
                "raw": raw,
                "prompt": prompt,
                "context": context,
            }

        logger.warning("[QueryOrchestrator] clarify gate returned non-boolean: %r", raw[:200])
        return None

    def _is_obvious_context_followup(self, query_context: QueryContext) -> bool:
        has_history_result_context = self._history_has_prior_result_context(query_context.history)
        if not (
            query_context.last_results
            or self._count_scope_context(query_context.session_id)
            or has_history_result_context
        ):
            return False
        question = str(query_context.question or "").strip()
        if not question or looks_like_search_request(question):
            return False
        if has_explicit_previous_scope(question) or looks_like_contextual_followup_request(question):
            return True
        try:
            from core.intent.context_followup_expert import ContextFollowupExpert

            prior_ctx = {
                "prior_was_search": bool(query_context.last_results) or has_history_result_context,
                "prior_was_content": bool(query_context.last_results) or has_history_result_context,
                "prior_was_media": False,
                "prior_was_count": bool(self._count_scope_context(query_context.session_id)),
                "prior_search_failed": False,
                "n_prior_files": len(query_context.last_results or []),
                "prior_user_query": "history" if has_history_result_context else "",
                "focused_file": (
                    (query_context.last_results or [{}])[0].get("file_name")
                    if len(query_context.last_results or []) == 1
                    else None
                ),
            }
            return bool(
                ContextFollowupExpert.analyze_context_followup(
                    question,
                    prior_ctx,
                    last_results=query_context.last_results,
                    active_paths=query_context.active_paths,
                )
            )
        except Exception:
            return False

    @staticmethod
    def _history_has_prior_result_context(history: List[Dict[str, Any]]) -> bool:
        for item in reversed(list(history or [])[-4:]):
            if not isinstance(item, dict):
                continue
            answer = str(item.get("a") or item.get("content") or "").strip()
            if answer and _HISTORY_RESULT_CONTEXT_RE.search(answer):
                return True
        return False

    def _maybe_set_pre_intent_scope_hint(self, query_context: QueryContext, clarify_params: Dict[str, Any]) -> None:
        question = str(query_context.question or "").strip()
        operation = "search" if looks_like_search_request(question) else ""
        if not operation and self._looks_like_scope_ambiguous_summary_request(question):
            operation = "summarize"
        if not operation:
            return
        try:
            setter = getattr(self.agent, "_set_followup_hint", None)
            if callable(setter):
                setter(
                    query_context.session_id,
                    action="search_scope_clarify",
                    params={
                        "query": question,
                        "search_params": {"query": question},
                        "operation": operation,
                        "clarify_context": dict(clarify_params.get("_clarify_context") or {}),
                    },
                    ttl_turns=1,
                    uses=1,
                )
        except Exception as exc:
            logger.debug("[QueryOrchestrator] pre-intent scope hint failed: %s", exc)

    def _guard_ambiguous_search_scope(
        self,
        query_context: QueryContext,
        decision: IntentDecision,
    ) -> Tuple[IntentDecision, Dict[str, Any]]:
        params = dict(decision.params or {})
        action = str(decision.action or "").strip()
        if params.get("_scope_disambiguation"):
            return decision, {"scope_disambiguation": str(params.get("_scope_disambiguation") or "")}

        question = str(query_context.question or "").strip()
        expert_route = str(params.get("_expert_route") or "").strip().lower()
        has_prior_file_scope = bool(query_context.last_results) or bool(self._count_scope_context(query_context.session_id))
        searchish_action = action in self._SEARCH_LIKE_ACTIONS
        searchish_process_previous = (
            action == "process_previous"
            and expert_route != "context_followup"
            and looks_like_search_request(question)
        )
        query = str(params.get("query") or question).strip() or question
        selected_scope_preserved = bool(
            expert_route == "selection"
            or params.get("_preserve_selected_scope")
            or params.get("_selection_media_scope")
            or str(params.get("_scope") or "").strip().lower() == "selected"
            or str(params.get("scope") or "").strip().lower() == "selected"
        )

        explicit_file_ref = params.get("_explicit_file_ref") if isinstance(params.get("_explicit_file_ref"), dict) else None
        if explicit_file_ref is None:
            explicit_file_ref = self._extract_explicit_filename_ref(question)
        if explicit_file_ref:
            # Do not collapse already-scoped selected-file/image intents into a
            # global explicit-filename lookup just because the extractor found
            # "selected file/image" surface text in the question.
            if selected_scope_preserved:
                return decision, {}
            lookup_name = str(explicit_file_ref.get("raw_name") or explicit_file_ref.get("search_term") or query).strip()
            if action in {"media_export", "media_content_search"}:
                routed_params = dict(params)
                if lookup_name:
                    routed_params.setdefault("file_hint", lookup_name)
                routed_params["_explicit_file_ref"] = explicit_file_ref
                routed_params.setdefault("_scope_disambiguation", "explicit_media_filename_request")
                return (
                    IntentDecision(
                        action=action,
                        params=routed_params,
                        confidence=max(float(decision.confidence or 0.0), 0.98),
                        source=decision.source,
                        normalized_internal_en=decision.normalized_internal_en,
                    ),
                    {"scope_disambiguation": "explicit_media_filename_request"},
                )
            routed_params = dict(params)
            routed_params["query"] = lookup_name or query
            routed_params["_explicit_file_ref"] = explicit_file_ref
            routed_params["_expert_route"] = "explicit_filename"
            for key in ("_scope", "_context_scope", "scope", "_scope_kind", "_preserve_selected_scope"):
                routed_params.pop(key, None)
            routed_params["_scope_disambiguation"] = "explicit_filename_request"
            return (
                IntentDecision(
                    action="search",
                    params=routed_params,
                    confidence=max(float(decision.confidence or 0.0), 0.98),
                    source=decision.source,
                    normalized_internal_en=decision.normalized_internal_en,
                ),
                {"scope_disambiguation": "explicit_filename_request"},
            )

        if action == "process_previous" and expert_route == "context_followup":
            return decision, {}

        personal_attribute_like = expert_route == "personal_attribute" or looks_like_personal_attribute_request(question)
        if personal_attribute_like:
            routed_params = dict(params)
            routed_params["query"] = query
            routed_params.setdefault("_expert_route", "personal_attribute")
            routed_action = "search" if action != "media_content_search" else action
            if has_prior_file_scope and has_prior_entity_reference(question):
                routed_params["_scope"] = "previous"
                routed_params["_scope_disambiguation"] = "personal_attribute_previous_scope"
                routed_params["_context_scope"] = "last_results"
                return (
                    IntentDecision(
                        action=routed_action,
                        params=routed_params,
                        confidence=decision.confidence,
                        source=decision.source,
                        normalized_internal_en=decision.normalized_internal_en,
                    ),
                    {"scope_disambiguation": "personal_attribute_previous_scope"},
                )
            if routed_params != params or routed_action != action:
                return (
                    IntentDecision(
                        action=routed_action,
                        params=routed_params,
                        confidence=decision.confidence,
                        source=decision.source,
                        normalized_internal_en=decision.normalized_internal_en,
                    ),
                    {},
                )
            return decision, {}

        if action == "search" and has_prior_file_scope and looks_like_contextual_followup_request(question):
            routed_params = dict(params)
            routed_params["_scope_disambiguation"] = "contextual_followup"
            return (
                IntentDecision(
                    action="process_previous",
                    params=routed_params,
                    confidence=decision.confidence,
                    source=decision.source,
                    normalized_internal_en=decision.normalized_internal_en,
                ),
                {"scope_disambiguation": "contextual_followup"},
            )

        if not searchish_action and not searchish_process_previous:
            return decision, {}

        if has_explicit_selected_scope(question):
            routed_params = dict(params)
            routed_params["query"] = query
            routed_params.pop("_scope", None)
            routed_params["_scope_disambiguation"] = "explicit_selected_scope"
            routed_action = "search" if searchish_process_previous else action
            return (
                IntentDecision(
                    action=routed_action,
                    params=routed_params,
                    confidence=decision.confidence,
                    source=decision.source,
                    normalized_internal_en=decision.normalized_internal_en,
                ),
                {"scope_disambiguation": "explicit_selected_scope"},
            )

        if has_explicit_previous_scope(question):
            routed_params = dict(params)
            routed_params["query"] = query
            routed_params["_scope"] = "previous"
            routed_params["_scope_disambiguation"] = "explicit_previous_scope"
            routed_action = "search" if searchish_process_previous else action
            return (
                IntentDecision(
                    action=routed_action,
                    params=routed_params,
                    confidence=decision.confidence,
                    source=decision.source,
                    normalized_internal_en=decision.normalized_internal_en,
                ),
                {"scope_disambiguation": "explicit_previous_scope"},
            )

        if not has_prior_file_scope:
            return decision, {}
        count_scope = self._count_scope_context(query_context.session_id)

        if selected_scope_matches_previous(
            active_paths=query_context.active_paths,
            last_results=query_context.last_results,
            count_scope_context=count_scope,
            total_searchable=query_context.total_searchable,
        ):
            routed_params = dict(params)
            routed_params["query"] = query
            routed_params.pop("_scope", None)
            routed_params["_scope_disambiguation"] = "selected_matches_previous"
            routed_action = "search" if searchish_process_previous else action
            return (
                IntentDecision(
                    action=routed_action,
                    params=routed_params,
                    confidence=decision.confidence,
                    source=decision.source,
                    normalized_internal_en=decision.normalized_internal_en,
                ),
                {"scope_disambiguation": "selected_matches_previous"},
            )

        if str(query_context.metadata.get("pre_intent_clarify_gate") or "").strip().lower() == "false":
            return decision, {"clarify_gate_decision": "false"}

        self._set_search_scope_clarify_hint(
            query_context,
            query=query,
            search_params=params,
        )
        clarify_params = {
            "question": clarify_message(query_context.user_language or query_context.prompt_language),
            "query": query,
            "_scope_disambiguation": "ambiguous_search_scope",
        }
        return (
            IntentDecision(
                action="clarify",
                params=clarify_params,
                confidence=1.0,
                source="scope_disambiguation",
                normalized_internal_en=decision.normalized_internal_en,
            ),
            {"scope_disambiguation": "ambiguous_search_scope"},
        )

    def execution_mode_for(self, action: str) -> QueryExecutionMode:
        act = str(action or "").strip()
        if act in self._TOOL_ACTIONS:
            return QueryExecutionMode.TOOL_AGENT
        if act in self._HANDLER_ACTIONS:
            if act == "chat":
                return QueryExecutionMode.CHAT
            if act == "clarify":
                return QueryExecutionMode.CLARIFY
            if act == "db_clear":
                return QueryExecutionMode.SYSTEM
            return QueryExecutionMode.HANDLER
        return QueryExecutionMode.SEARCH_PIPELINE

    def query_type_for(self, action: str) -> str:
        return self._QUERY_TYPE_MAP.get(str(action or "").strip(), "search")

    def is_handler_action(self, action: str) -> bool:
        return self.execution_mode_for(action) == QueryExecutionMode.HANDLER

    def supports_structured_execution(self, action: str) -> bool:
        return self.execution_mode_for(action) != QueryExecutionMode.SEARCH_PIPELINE

    def build_handler_context(self, request: ActionRequest) -> HandlerContext:
        return HandlerContext(
            question=request.query,
            params=dict(request.params or {}),
            active_paths=list(request.metadata.get("active_paths") or []),
            session_id=request.session_id,
            lang=request.user_language,
            kb=self.agent.kb,
            llm_service=self.agent._get_llm_service(
                detailed=False,
                session_id=request.session_id,
                prompt_language=request.user_language,
            ),
            prompt_formatter=self.agent._prompt,
            normalize_category=self.agent._normalize_category_name,
            is_generic_category=self.agent._is_generic_file_scope_category,
            get_category_keywords=self.agent._get_rule_category_keywords,
            history=list(self.agent._get_history_ref(request.session_id)),
            last_results=list(self.agent._get_last_search_results_ref(request.session_id)),
            abort_checker=lambda: self.agent.is_aborted(request.session_id),
            log_fn=logger.info,
        )

    def as_debug_payload(self, request: ActionRequest) -> Dict[str, Any]:
        return asdict(request)
