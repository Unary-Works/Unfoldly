"""
IntentValidator — centralized correction and validation layer.

Consolidates correction logic that was previously scattered across 4 locations:
  1. IntentAnalyzer.correct_llm_intent()
  2. FileOpAgent._postprocess()
  3. _normalize_intent_to_internal_en() overrides
  4. dispatch.py followup_hint logic

Design: Pure deterministic rules, no LLM calls.
All corrections are logged with reasons for observability.
"""
from __future__ import annotations

import re
import logging
from typing import Optional, List, Dict, Any, Callable

logger = logging.getLogger(__name__)


class IntentValidator:
    """
    Centralized intent validation/correction pipeline.
    
    All agent outputs pass through this layer before dispatch.
    Corrections are applied as a chain of guards, each returning
    either a corrected intent or None (pass through).
    """

    @classmethod
    def validate(
        cls,
        question: str,
        result: dict,
        *,
        last_results: Optional[List[dict]] = None,
        history: Optional[List[dict]] = None,
        active_paths: Optional[List[str]] = None,
        prompt_language: str = "en",
    ) -> dict:
        """
        Apply all correction guards in priority order.
        
        Args:
            question: Original user query
            result: Raw intent from agent pipeline {"action": ..., "params": ...}
            last_results: Previous search/count results for context
            history: Conversation history
            active_paths: Currently selected file paths
            prompt_language: "zh" or "en"
            
        Returns:
            Validated/corrected intent dict
        """
        action = str(result.get("action", "")).strip()
        params = result.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        qn = (question or "").strip()
        ql = qn.lower()
        confidence = float(result.get("confidence", 0.7))

        # Apply guard chain
        guard_chain = (
            cls._SKILL_DISPATCH_GUARD_CHAIN
            if cls._is_skill_dispatch_route(params)
            else cls._GUARD_CHAIN
        )
        for guard_fn in guard_chain:
            corrected = guard_fn(
                ql=ql, qn=qn, action=action, params=params,
                confidence=confidence,
                last_results=last_results,
                history=history,
                active_paths=active_paths,
                prompt_language=prompt_language,
            )
            if corrected is not None:
                corrected.setdefault("confidence", confidence)
                return corrected

        # No correction needed
        result.setdefault("confidence", confidence)
        return result

    @staticmethod
    def _is_deterministic_expert_route(params: Optional[Dict[str, Any]]) -> bool:
        """High-confidence routes that should not be overridden by heuristic guards.
        
        - Deterministic fast gates: personal_attribute, filename, explicit_filename, selection, media
        - SkillDispatcher (skill_dispatch): single LLM call with full skill context.
          Its search/process_previous decision should be trusted over regex heuristics.
        """
        route = str((params or {}).get("_expert_route") or "").strip().lower()
        return route in {"personal_attribute", "filename", "explicit_filename", "selection", "media", "skill_dispatch"}

    @staticmethod
    def _is_skill_dispatch_route(params: Optional[Dict[str, Any]]) -> bool:
        route = str((params or {}).get("_expert_route") or "").strip().lower()
        return route == "skill_dispatch"

    @staticmethod
    def _infer_requested_media_type(ql: str) -> str:
        video = bool(
            re.search(
                r'\b(video|videos|movie|movies|clip|clips|film|films|footage'
                r'|mp4|mov|mkv|avi|webm|m4v|wmv|flv|ts)\b'
                r'|视频|影片|录像|录屏|短片',
                ql,
                re.IGNORECASE,
            )
        )
        audio = bool(
            re.search(
                r'\b(audio|audios|podcast|podcasts|song|songs|music'
                r'|mp3|wav|m4a|aac|flac|ogg|wma|aiff|ape)\b'
                r'|音频|录音|播客|歌曲|音乐',
                ql,
                re.IGNORECASE,
            )
        )
        generic = bool(re.search(r'\b(media)\b|音视频|影音', ql, re.IGNORECASE))
        if video and not audio:
            return "video"
        if audio and not video:
            return "audio"
        if generic or video or audio:
            return "all"
        return ""

    @staticmethod
    def _build_selected_media_params(media_type: str, *, selection_mode: str = "") -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "category": media_type if media_type in {"audio", "video"} else "audio/video",
            "_scope": "selected",
        }
        if selection_mode:
            params["_selection_mode"] = selection_mode
        if media_type in {"audio", "video"}:
            params["media_type"] = media_type
        return params

    @staticmethod
    def _parse_size_threshold_bytes(ql: str) -> Optional[Dict[str, Any]]:
        m = re.search(
            r"\b(?:larger|bigger|greater|more|over|above)\s+than\s+(\d+(?:\.\d+)?)\s*(kb|mb|gb|bytes?|b)\b"
            r"|\b(?:smaller|less|under|below)\s+than\s+(\d+(?:\.\d+)?)\s*(kb|mb|gb|bytes?|b)\b",
            ql,
            re.IGNORECASE,
        )
        if not m:
            return None
        raw_value = m.group(1) or m.group(3)
        raw_unit = (m.group(2) or m.group(4) or "b").lower()
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None
        factor = {
            "b": 1,
            "byte": 1,
            "bytes": 1,
            "kb": 1024,
            "mb": 1024 * 1024,
            "gb": 1024 * 1024 * 1024,
        }.get(raw_unit, 1)
        key = "min_file_size_bytes" if m.group(1) else "max_file_size_bytes"
        return {key: int(value * factor), "_metadata_filter": "file_size"}

    @staticmethod
    def _guard_scoped_metadata_count(*, ql, qn, action, last_results, **kwargs) -> Optional[dict]:
        if not last_results or action not in {"search", "chat", "clarify", "process_previous", "count"}:
            return None
        if not re.search(r"\b(?:how\s+many|count|number\s+of)\b", ql, re.IGNORECASE):
            return None
        size_params = IntentValidator._parse_size_threshold_bytes(ql)
        if not size_params:
            return None
        params = {
            "category": "all",
            "scope": "last_results",
            "_scope_disambiguation": "contextual_metadata_count",
            **size_params,
        }
        logger.info("[validator] scoped metadata count: %s -> count(last_results, %s)", action, size_params)
        return {"action": "count", "params": params}

    @staticmethod
    def _build_selected_summary_params(
        raw_params: Optional[Dict[str, Any]] = None,
        *,
        media_type: str = "",
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(raw_params or {})
        params.setdefault("_scope", "selected")
        params["_preserve_selected_scope"] = True
        if media_type:
            params.update(IntentValidator._build_selected_media_params(media_type))
            params["_preserve_selected_scope"] = True
            params["_selection_media_scope"] = True
        return params

    # ════════════════════════════════════════════════════════════════════════
    # Individual guard functions (each returns dict or None)
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _guard_list_selected(*, ql, qn, action, params, **kwargs) -> Optional[dict]:
        """list_selected → direct selected-item listing or summarize_all (content)."""
        if action != "list_selected":
            return None
        _LIST_KWS = [
            "有哪些", "哪些文件", "哪些文档", "有多少", "几个", "几份", "列出", "清单",
            "what files", "which files", "list", "how many", "what are",
        ]
        if any(kw in ql for kw in _LIST_KWS):
            logger.info("[validator] list_selected → count(scope=selected_items)")
            return {
                "action": "count",
                "params": {"category": "all", "_scope": "selected", "_selection_mode": "selected_items"},
            }
        logger.info("[validator] list_selected → summarize_all")
        return {"action": "summarize_all", "params": params}

    @staticmethod
    def _guard_meta_followup(*, ql, qn, action, last_results, prompt_language, confidence=0.7, **kwargs) -> Optional[dict]:
        """Meta followup short phrases should go to process_previous, not search.
        
        Confidence-aware: if LLM is confident (>=0.88) the query is a search, respect it.
        """
        if not last_results:
            return None
        params = kwargs.get("params") or {}
        if IntentValidator._is_deterministic_expert_route(params):
            return None
        # Respect high-confidence LLM search decisions — don't override with meta-followup heuristic
        if action == "search" and float(confidence) >= 0.88:
            return None
        # Import here to avoid circular imports
        from core.intent_analyzer import IntentAnalyzer
        if not IntentAnalyzer.looks_like_meta_followup_on_last_results(qn, prompt_language):
            return None
        # Guard: explicit new-search verbs should NOT be redirected
        _STRONG_ACT = re.compile(
            r'^(find|search|show|list|get|display|retrieve|找|搜|显示|列出)\b',
            re.IGNORECASE,
        )
        if _STRONG_ACT.match(ql):
            return None
        if action in {"search", "summarize", "clarify"}:
            logger.info(f"[validator] meta-followup: {action} → process_previous (conf={confidence:.2f})")
            return {"action": "process_previous", "params": {}}
        return None

    @staticmethod
    def _guard_prev_ref_keywords(*, ql, qn, action, last_results, confidence=0.7, **kwargs) -> Optional[dict]:
        """Queries with explicit previous-result references + pronoun 'it' → process_previous.
        
        Confidence-aware: if LLM is very confident (>=0.90) the query is a new search, respect it.
        """
        if not last_results:
            return None
        params = kwargs.get("params") or {}
        if IntentValidator._is_deterministic_expert_route(params):
            return None
        if action not in {"search", "clarify", "summarize_all"}:
            return None
        # Respect very-high-confidence LLM search decisions
        if action == "search" and float(confidence) >= 0.90:
            return None
        from core.intent_analyzer import IntentAnalyzer, IntentKeywords
        has_pronoun = re.search(r'\b(it|he|she|they|his|her|its|their|him|them)\b|(他|她|它|这家|这几|那个|那几|这位|这些人|那些人)', ql)
        has_prev_ref = IntentAnalyzer._is_kw_match(ql, IntentKeywords.PREV_REF_KWS) or has_pronoun
        if not has_prev_ref:
            return None
        _STRONG_NEW = re.compile(
            r'^(find\s+new|search\s+new|another\s+search|ignore)\b',
            re.IGNORECASE,
        )
        if _STRONG_NEW.match(ql):
            return None
        logger.info(f"[validator] prev-ref keywords: {action} → process_previous (conf={confidence:.2f})")
        return {"action": "process_previous", "params": {}}

    @staticmethod
    def _guard_pronoun_followup(*, ql, qn, action, last_results, confidence=0.7, **kwargs) -> Optional[dict]:
        """Queries with topic continuation and pronoun → process_previous.

        Guards (prevent false positives):
          - Personal attribute lookups (e.g. "what is his email") → let search handle it
          - Long queries (> 12 words) are likely independent queries with incidental pronouns
          - Explicit new-scope / recount patterns
          - High-confidence LLM search decision (>=0.88) → respect it
        """
        if not last_results:
            return None

        params = kwargs.get("params") or {}
        if IntentValidator._is_deterministic_expert_route(params):
            return None

        if action not in {"search", "clarify", "summarize_all", "chat"}:
            return None

        # Respect high-confidence LLM search decisions — pronouns in long queries are often incidental
        if action == "search" and float(confidence) >= 0.88:
            return None

        from core.intent.context_followup_expert import ContextFollowupExpert

        word_count = len([w for w in ql.split() if w])
        if word_count > 12:
            return None

        # Anti-guard: personal attribute queries must go to search, not continuation
        if ContextFollowupExpert._ATTR_LOOKUP_RE.search(ql):
            return None

        if ContextFollowupExpert._NEW_SCOPE_RE.search(ql) or ContextFollowupExpert._RECOUNT_RE.search(ql):
            return None

        has_pronoun = bool(ContextFollowupExpert._PRONOUN_FOLLOWUP_RE.search(ql))
        has_topic = bool(ContextFollowupExpert._TOPIC_FOLLOWUP_SHORT_RE.search(ql))

        if has_pronoun or has_topic:
            logger.info(f"[validator] pronoun/topic followup: {action} → process_previous (pronoun={has_pronoun}, topic={has_topic})")
            return {"action": "process_previous", "params": {}}

        return None

    @staticmethod
    def _guard_how_many_files(*, ql, qn, action, **kwargs) -> Optional[dict]:
        """'how many files' should always be count(all)."""
        params = dict(kwargs.get("params") or {})
        last_results = list(kwargs.get("last_results") or [])
        if action == "summarize_all" and (
            params.get("_preserve_selected_scope")
            or params.get("_selection_media_scope")
            or str(params.get("_scope") or "").strip().lower() == "selected"
            or str(params.get("scope") or "").strip().lower() in {"selected", "selected_items", "selected_folder"}
        ):
            return None

        def _with_contextual_scope(out_params: Dict[str, Any]) -> Dict[str, Any]:
            contextual_count_scope = bool(
                last_results
                and re.search(
                    r"\b(?:there|in\s+there|of\s+them|of\s+these|these\s+results|those\s+results|"
                    r"this\s+(?:topic|subject|area|kind|type)|that\s+(?:topic|subject|area|kind|type)|"
                    r"same\s+(?:topic|subject|area|kind|type))\b"
                    r"|这方面|这类|这批|这些|那些|其中|里面|上面|上述|相关|同类|同主题",
                    qn,
                    re.IGNORECASE,
                )
            )
            if contextual_count_scope:
                out_params.setdefault("scope", "last_results")
                out_params.setdefault("_scope_disambiguation", "contextual_count_followup")
            return out_params

        try:
            from core.retrieval.category_engine import (
                is_generic_file_scope_category,
                match_dynamic_category_from_query,
            )
            dynamic_category = match_dynamic_category_from_query(qn, refresh_if_missing=True)
        except Exception:
            dynamic_category = ""
            is_generic_file_scope_category = lambda _value: False  # type: ignore

        specific_category = dynamic_category
        if not specific_category:
            try:
                from core.intent.entity_experts import CategoryListExpert

                category_intent = CategoryListExpert.analyze(qn, has_content_qualifier=False)
                specific_category = str(((category_intent or {}).get("params") or {}).get("category") or "").strip().lower()
            except Exception:
                specific_category = ""
        if not specific_category:
            try:
                from core.kb.knowledge_base import _normalize_category_en

                normalized_category = str(_normalize_category_en(qn, default="") or "").strip().lower()
                explicit_count_categories = {
                    "resume",
                    "report",
                    "contract",
                    "note",
                    "manual",
                    "paper",
                    "presentation",
                    "data",
                    "email",
                    "image",
                    "audio",
                    "video",
                    "audio/video",
                    "book",
                    "code",
                    "invoice",
                    "quotation",
                }
                if normalized_category in explicit_count_categories and not is_generic_file_scope_category(normalized_category):
                    specific_category = normalized_category
            except Exception:
                pass

        media_type = IntentValidator._infer_requested_media_type(ql)

        def _build_media_count_params() -> Dict[str, Any]:
            params: Dict[str, Any] = {
                "category": media_type if media_type in {"audio", "video"} else "audio/video"
            }
            if media_type in {"audio", "video"}:
                params["media_type"] = media_type
            return params

        def _build_dynamic_count_params() -> Dict[str, Any]:
            if media_type and specific_category in {"audio", "video", "audio/video"}:
                return _build_media_count_params()
            return {"category": specific_category}

        scoped_content_count_en = bool(
            action == "process_previous"
            and last_results
            and re.search(
                r"\bhow\s+many\s+"
                r"(?!(?:files?|documents?|docs?|folders?|sources?|images?|photos?|pictures?|videos?|audios?|"
                r"recordings?|resumes?|reports?|papers?|contracts?|invoices?)\b)"
                r"[a-z][a-z0-9_-]{1,40}\b.{0,60}"
                r"\b(?:remain(?:ing)?|left|listed|mentioned|included|covered|available)\b",
                ql,
                re.IGNORECASE,
            )
            and not re.search(
                r"\b(?:do\s+i\s+have|do\s+we\s+have|are\s+there|all\s+my|all\s+of\s+my|"
                r"my\s+(?:files?|documents?|library)|indexed|in\s+my\s+(?:files?|documents?|library))\b",
                ql,
                re.IGNORECASE,
            )
        )
        if scoped_content_count_en:
            logger.info("[validator] scoped English content count follow-up: keep process_previous")
            return None

        if re.search(r"\bhow\s+many\b", ql) and (
            re.search(r"\b(files?|documents?|docs?|sources?)\b", ql)
            or bool(specific_category)
        ):
            if media_type:
                logger.info(
                    f"[validator] how-many-files: {action} → count({media_type}, media_type={media_type})"
                )
                return {"action": "count", "params": _with_contextual_scope(_build_media_count_params())}
            if specific_category and not is_generic_file_scope_category(specific_category):
                logger.info(
                    f"[validator] how-many-files: {action} → count(category={specific_category})"
                )
                return {"action": "count", "params": _with_contextual_scope(_build_dynamic_count_params())}
            if action != "count":
                logger.info(f"[validator] how-many-files: {action} → count(all)")
            return {"action": "count", "params": _with_contextual_scope({"category": "all"})}

        zh_quantity = bool(
            re.search(
                r"(?:有|共有|总共|一共|统计|数一下|算一下)?[^。！？?]{0,12}"
                r"(?:多少|几)(?:份|个|篇|张|条|项|部)?",
                qn,
            )
        )
        explicit_count_signal = bool(
            re.search(
                r"\b(?:how\s+many|number\s+of|count|total)\b"
                r"|有多少|多少(?:份|个|篇|张|条|项|部)?|数量|统计|数一下|算一下|总共|一共|共有",
                qn,
                re.IGNORECASE,
            )
        )
        search_or_listing_command = bool(
            re.match(
                r"^\s*(?:please\s+|pls\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+)?"
                r"(?:find|search(?:\s+for)?|look\s+for|locate|get(?:\s+me)?|show(?:\s+me)?|"
                r"retrieve|fetch|list|display|browse)\b"
                r"|^\s*(?:请|麻烦你|帮我|给我)?\s*"
                r"(?:找|搜|查|检索|查找|搜索|找找|找下|搜下|查下|找一下|搜一下|查一下|列出|显示|看看|看下)",
                qn,
                re.IGNORECASE,
            )
        )
        if zh_quantity and search_or_listing_command and not explicit_count_signal:
            return None
        zh_file_scope_noun = bool(
            re.search(
                r"文件|文档|资料|报告|论文|图片|照片|视频|音频|录音|简历|发票|表格|数据表|幻灯片|PPT|ppt|pdf|PDF",
                qn,
            )
        )
        zh_prior_scope_ref = bool(
            last_results
            and re.search(r"这方面|这类|这批|这些|那些|这几|那几|其中|里面|上面|上述|相关|同类|同主题", qn)
        )
        zh_scoped_comparison = bool(
            re.search(
                r"(?:这几份|这些|那几份|那些|其中|上面|上述|前面|这批|那批)?.{0,8}"
                r"(?:哪一份|哪份|哪个|谁).{0,24}(?:最|更).{0,8}"
                r"(?:匹配|适合|符合|契合|对应|合适)",
                qn,
            )
        )
        zh_semantic_quantity = bool(
            re.search(
                r"金额|总金额|合计|加起来|费用|价格|规模|增速|增长|预期|预测|架构|模型|区别|差异|质量|品质|最好|最高|市场|背景|要求",
                qn,
                re.IGNORECASE,
            )
        )
        zh_file_count_ask = bool(
            re.search(
                r"(?:多少|几)(?:份|个|篇|张|条|项|部)?\s*(?:文件|文档|资料|报告|论文|图片|照片|视频|音频|录音|简历|发票|表格|数据表|幻灯片|PPT|ppt|pdf|PDF)"
                r"|(?:文件|文档|资料|报告|论文|图片|照片|视频|音频|录音|简历|发票|表格|数据表|幻灯片|PPT|ppt|pdf|PDF).{0,10}(?:多少|几)",
                qn,
            )
        )
        if action == "process_previous" and zh_quantity and zh_prior_scope_ref and (zh_semantic_quantity or not zh_file_count_ask):
            logger.info("[validator] scoped content quantity follow-up: keep process_previous")
            return None
        if action == "process_previous" and zh_scoped_comparison and (zh_prior_scope_ref or zh_file_scope_noun):
            logger.info("[validator] scoped comparison follow-up: keep process_previous")
            return None
        if action == "process_previous" and zh_quantity and zh_semantic_quantity:
            logger.info("[validator] semantic quantity follow-up: keep process_previous")
            return None
        zh_item_scope = bool(zh_file_scope_noun or zh_prior_scope_ref)
        if zh_quantity and (zh_item_scope or bool(specific_category)):
            if media_type:
                logger.info(
                    f"[validator] zh-how-many-files: {action} → count({media_type}, media_type={media_type})"
                )
                return {"action": "count", "params": _with_contextual_scope(_build_media_count_params())}
            if specific_category and not is_generic_file_scope_category(specific_category):
                logger.info(
                    f"[validator] zh-how-many-files: {action} → count(category={specific_category})"
                )
                return {"action": "count", "params": _with_contextual_scope(_build_dynamic_count_params())}
            if action != "count":
                logger.info(f"[validator] zh-how-many-files: {action} → count(all)")
            return {"action": "count", "params": _with_contextual_scope({"category": "all"})}

        # Also check pattern-based all-files-list
        from core.intent_analyzer import IntentAnalyzer
        if IntentAnalyzer._looks_like_all_files_list_query(qn):
            if media_type:
                logger.info(
                    f"[validator] all-files-list: {action} → count({media_type}, media_type={media_type})"
                )
                return {"action": "count", "params": _with_contextual_scope(_build_media_count_params())}
            if action != "count":
                logger.info(f"[validator] all-files-list: {action} → count(all)")
            return {"action": "count", "params": _with_contextual_scope({"category": "all"})}
        return None

    @staticmethod
    def _guard_category_retrieval_not_count(*, ql, qn, action, params, **kwargs) -> Optional[dict]:
        """
        Queries like "find my resume", "show photos", "find all worksheets"
        should retrieve matching files, not count them.

        SkillDispatcher may still collapse category-listing requests into
        selected-scope count when a large folder is active.  Allow this guard
        to correct only that soft LLM route while keeping deterministic
        filename/media/selection experts protected from regex overrides.
        """
        if action != "count":
            return None
        if (
            IntentValidator._is_deterministic_expert_route(params)
            and not IntentValidator._is_skill_dispatch_route(params)
        ):
            return None
        if re.search(r"\bhow\s+many\b", ql):
            return None

        from core.intent.entity_experts import CategoryListExpert
        has_content_qualifier = bool(
            re.search(
                r'\b(detail|details|content|contents|about|describe|explanation|explain|'
                r'summary|summarize|analysis|analyze|inside)\b',
                ql,
                re.IGNORECASE,
            )
        )
        cat_result = CategoryListExpert.analyze(qn, has_content_qualifier=has_content_qualifier)
        if cat_result and str(cat_result.get("action")) == "search":
            logger.info(f"[validator] category-retrieval: count → search(category={cat_result.get('params', {}).get('category')!r})")
            return cat_result
        return None

    @staticmethod
    def _guard_document_topic_not_media_category(*, ql, qn, action, params, **kwargs) -> Optional[dict]:
        """Document searches whose topic contains media words must not use media filters."""
        if action != "search":
            return None
        current_category = str((params or {}).get("category") or "").strip().lower()
        current_media_type = str((params or {}).get("media_type") or "").strip().lower()
        if current_category not in {"audio/video", "audio", "video"} and current_media_type not in {"audio", "video"}:
            return None

        has_document_target = bool(
            re.search(
                r"\b(?:papers?|articles?|documents?|docs?|pdfs?|reports?|publications?|theses|thesis)\b"
                r"|论文|文章|文档|报告|资料|PDF|pdf",
                qn,
                re.IGNORECASE,
            )
        )
        has_media_topic = bool(
            re.search(r"\b(?:audio|video|speech|music|sound)\b|音频|视频|语音|声音|音乐", qn, re.IGNORECASE)
        )
        if not (has_document_target and has_media_topic):
            return None

        repaired_params = dict(params or {})
        repaired_params.pop("media_type", None)
        if current_category in {"audio/video", "audio", "video"}:
            repaired_params.pop("category", None)
        query = str(repaired_params.get("query") or qn or "").strip()
        if query and not re.search(
            r"\b(?:papers?|articles?|documents?|docs?|pdfs?|reports?|publications?|theses|thesis)\b"
            r"|论文|文章|文档|报告|资料|PDF|pdf",
            query,
            re.IGNORECASE,
        ):
            query = f"{query} papers"
        repaired_params["query"] = query or qn
        repaired_params.setdefault(
            "_dispatch_reason",
            "Document retrieval with media-domain topic terms should not be filtered as audio/video media.",
        )
        logger.info("[validator] document media-topic search: removed audio/video filter")
        return {"action": "search", "params": repaired_params}

    @staticmethod
    def _guard_scoped_file_search(*, ql, qn, action, params, **kwargs) -> Optional[dict]:
        """Scoped file queries (with topic) should search, not count.
        Skipped for skill_dispatch — it already chose count deliberately.
        """
        if action != "count":
            return None
        if IntentValidator._is_deterministic_expert_route(params):
            return None
        from core.intent_analyzer import IntentAnalyzer
        if IntentAnalyzer._looks_like_scoped_file_search_query(qn):
            logger.info(f"[validator] scoped-file → search (not count)")
            return {"action": "search", "params": {"query": qn}}
        return None

    @staticmethod
    def _guard_file_content_analysis(*, ql, qn, action, **kwargs) -> Optional[dict]:
        """File content analysis requests should search, not count.
        Skipped for skill_dispatch — it already chose the right action.
        """
        from core.intent_analyzer import IntentAnalyzer
        last_results = kwargs.get("last_results") or []
        prompt_language = kwargs.get("prompt_language")
        params = kwargs.get("params") or {}
        if action == "process_previous":
            return None
        if IntentValidator._is_deterministic_expert_route(params):
            logger.info("[validator] file-content-analysis guard skipped for deterministic/skill_dispatch route")
            return None
        if last_results and (
            IntentAnalyzer.looks_like_meta_followup_on_last_results(qn, prompt_language)
            or IntentAnalyzer.looks_like_content_followup_on_prior_results(qn)
        ):
            return None
        if IntentAnalyzer._looks_like_file_content_analysis_query(qn):
            focus = IntentAnalyzer._extract_file_analysis_focus_query(qn) or qn
            logger.info(f"[validator] file-content-analysis → search")
            return {"action": "search", "params": {"query": focus}}
        return None

    @staticmethod
    def _guard_short_affirm_view_detail(*, ql, qn, action, last_results, history, **kwargs) -> Optional[dict]:
        if not last_results:
            return None
        from core.intent_analyzer import IntentKeywords
        q_compact = re.sub(r'[\s。！？！?,.!]+', '', ql)
        short_affirm_kws = IntentKeywords.SHORT_AFFIRM_KWS.get("zh", []) + IntentKeywords.SHORT_AFFIRM_KWS.get("en", [])
        if q_compact not in short_affirm_kws:
            return None

        prev_assistant = ""
        try:
            hist = history or []
            if len(hist) >= 2 and hist[-1].get("role") == "user" and hist[-1].get("content") == qn:
                prev_assistant = str(hist[-2].get("content") or "")
            elif hist:
                prev_assistant = str(hist[-1].get("content") or "")
        except Exception:
            pass

        prev_lower = prev_assistant.lower()
        followup_markers = [
            "需要我详细介绍某一份文件吗", "需要查看哪一份", "查看哪一份",
            "详细介绍某一份", "which one", "which file", "view which",
        ]
        if any(m in prev_assistant or m in prev_lower for m in followup_markers):
            if len(last_results) == 1:
                logger.info("[validator] short-affirm + single file → view_detail(index=1)")
                return {"action": "view_detail", "params": {"index": 1}}
        return None

    @staticmethod
    def _guard_global_summary(*, ql, qn, action, **kwargs) -> Optional[dict]:
        """'summary of all documents/files' → summarize_all."""
        if action not in {"search", "chat", "clarify", "count", "summarize"}:
            return None
        # English patterns
        _SUMMARIZE_ALL_RE = re.compile(
            r'\b(summar|overview|recap).{0,20}\b(all|every|my|the)\b.{0,15}\b('
            r'document|file|doc|知识|资料|文件|文档)'
            r'|'
            r'\b(all|every|my)\b.{0,15}\b(document|file|doc|文档|资料|文件).{0,10}\b(summar|overview|recap)',
            re.IGNORECASE,
        )
        # Chinese patterns
        _ZH_SUMMARIZE_ALL_RE = re.compile(
            r'(总结|概括|归纳|概述|梳理).{0,6}(所有|全部|所有的|我的).{0,6}(文件|文档|资料|知识)'
            r'|'
            r'(所有|全部|我的).{0,6}(文件|文档|资料).{0,6}(总结|概括|归纳|概述)',
        )
        if _SUMMARIZE_ALL_RE.search(ql) or _ZH_SUMMARIZE_ALL_RE.search(ql):
            logger.info(f"[validator] global-summary: {action} → summarize_all")
            return {"action": "summarize_all", "params": {}}
        return None

    @staticmethod
    def _guard_selected_extension_count(*, ql, qn, action, **kwargs) -> Optional[dict]:
        """'my selected PDF/docx/wav files' → count(category, extension)."""
        if action not in {"search", "chat", "summarize_all", "clarify", "list_selected"}:
            return None
        raw_params = kwargs.get("params") or {}
        if action == "summarize_all" and (
            raw_params.get("_preserve_selected_scope")
            or raw_params.get("_selection_media_scope")
            or str(raw_params.get("_scope") or "").strip().lower() == "selected"
        ):
            return None
        _SELECTED_RE = re.compile(
            r'\b((my\s+)?selected|seleted|selectd|slected|chosen|chosn|picked)\b'
            r'|'
            r'(选中|已选|已勾选)',
            re.IGNORECASE,
        )
        _CONTENT_RE = re.compile(
            r'\b(tell\s+me\s+about|summarize|summary|overview|explain|describe|details?|content|contents?|'
            r'most\s+important(?:\s+information)?|important(?:\s+information)?|key\s+points?|main\s+points?)\b'
            r'|\b(heard|hear|said|spoken|can\s+be\s+heard|is\s+heard|can\s+i\s+hear|is\s+said)\b'
            r'|'
            r'(介绍|总结|概括|讲讲|内容|详情|详细)',
            re.IGNORECASE,
        )
        _LISTING_RE = re.compile(
            r'\b(show|list|what|which|display|browse|see)\b'
            r'|'
            r'(看看|看下|查看|列出|有哪些|都有哪些|有什么)',
            re.IGNORECASE,
        )
        if _SELECTED_RE.search(ql) and not _CONTENT_RE.search(ql):
            media_type = IntentValidator._infer_requested_media_type(ql)
            if media_type:
                selection_mode = "selected_items" if _LISTING_RE.search(ql) or action == "list_selected" else ""
                logger.info(
                    f"[validator] selected-media inventory: {action} → count(media_type={media_type}, selection_mode={selection_mode or 'scope'})"
                )
                return {
                    "action": "count",
                    "params": IntentValidator._build_selected_media_params(
                        media_type, selection_mode=selection_mode
                    ),
                }
        if _CONTENT_RE.search(ql):
            return None
        _SELECTED_RE = re.compile(
            r'\b((my\s+)?selected|seleted|selectd|slected|chosen|chosn|picked)\b.{0,15}\b('
            r'pdf|docx?|xlsx?|xls|csv|pptx?|ppt|wav|mp[34]|m4a|mov|jpg|jpeg|png|gif|image|video|audio'
            r')\b',
            re.IGNORECASE,
        )
        m = _SELECTED_RE.search(ql)
        if m:
            ext = m.group(3).lower()
            logger.info(f"[validator] selected-extension: {action} → count(ext={ext})")
            return {"action": "count", "params": {"category": "all", "_scope": "selected", "extension": ext}}
        return None

    @staticmethod
    def _guard_selected_scope_listing(*, ql, qn, action, active_paths, **kwargs) -> Optional[dict]:
        """Selected-scope listing requests should surface the selected items themselves."""
        if not active_paths:
            return None
        if action not in {"search", "chat", "clarify", "count", "list_selected"}:
            return None

        _SELECTED_SCOPE_RE = re.compile(
            r'\b((my\s+)?selected|seleted|selectd|slected|chosen|chosn|picked)\b'
            r'|'
            r'\b(these|those)\s+(files?|documents?|docs?)\b'
            r'|'
            r'(选中|已选|已勾选|这些文件|这些文档|那些文件|那些文档)',
            re.IGNORECASE,
        )
        _LISTING_RE = re.compile(
            r'\b(show|list|what|which|display|browse|see)\b'
            r'|'
            r'(看看|看下|查看|列出|有哪些|都有哪些|有什么)',
            re.IGNORECASE,
        )
        _CONTENT_RE = re.compile(
            r'\b(tell\s+me\s+about|summarize|summary|overview|explain|describe|details?|content|contents?|'
            r'most\s+important(?:\s+information)?|important(?:\s+information)?|key\s+points?|main\s+points?)\b'
            r'|\b(heard|hear|said|spoken|can\s+be\s+heard|is\s+heard|can\s+i\s+hear|is\s+said)\b'
            r'|'
            r'(介绍|总结|概括|讲讲|内容|详情|详细)',
            re.IGNORECASE,
        )
        if not _SELECTED_SCOPE_RE.search(ql):
            return None
        if _CONTENT_RE.search(ql):
            return None
        if not _LISTING_RE.search(ql):
            return None

        media_type = IntentValidator._infer_requested_media_type(ql)
        if media_type:
            logger.info(f"[validator] selected-scope media listing → count(scope=selected_items, media_type={media_type})")
            return {
                "action": "count",
                "params": IntentValidator._build_selected_media_params(
                    media_type, selection_mode="selected_items"
                ),
            }

        logger.info("[validator] selected-scope listing → count(scope=selected_items)")
        return {
            "action": "count",
            "params": {"category": "all", "_scope": "selected", "_selection_mode": "selected_items"},
        }

    @staticmethod
    def _guard_selected_scope_summary(*, ql, qn, action, active_paths, **kwargs) -> Optional[dict]:
        """Selected-scope content requests stay inside selection; media requests keep media subtype."""
        if not active_paths:
            return None
        if action not in {"search", "chat", "count", "clarify"}:
            return None
        raw_params = kwargs.get("params") or {}
        preserved_params: Dict[str, Any] = {}
        expert_route = str(raw_params.get("_expert_route") or "").strip()
        if expert_route:
            preserved_params["_expert_route"] = expert_route

        _SELECTED_SCOPE_RE = re.compile(
            r'\b(selected|seleted|selectd|slected|chosen|chosn|picked)\b'
            r'|'
            r'\b(these|those)\s+(files?|documents?|docs?)\b'
            r'|'
            r'(选中|已选|已勾选|这些文件|这些文档|那些文件|那些文档)',
            re.IGNORECASE,
        )
        _CONTENT_RE = re.compile(
            r'\b(tell(?:\s+me)?\s+about|what\s+are\s+these\s+files\s+about|summarize|summary|overview|'
            r'explain|describe|details?|content|contents?|most\s+important(?:\s+information)?|important(?:\s+information)?|'
            r'key\s+points?|main\s+points?)\b'
            r'|\b(heard|hear|said|spoken|can\s+be\s+heard|is\s+heard|can\s+i\s+hear|is\s+said)\b'
            r'|'
            r'(介绍|总结|概括|讲讲|讲一下|内容|详情|详细)',
            re.IGNORECASE,
        )
        if _SELECTED_SCOPE_RE.search(ql) and _CONTENT_RE.search(ql):
            media_type = IntentValidator._infer_requested_media_type(ql)
            preserved_params = dict(raw_params)
            preserved_params.setdefault("_expert_route", expert_route)
            if media_type:
                logger.info(f"[validator] selected-scope media content → summarize(media_type={media_type})")
                preserved_params.update(IntentValidator._build_selected_media_params(media_type))
                return {
                    "action": "summarize",
                    "params": preserved_params,
                }
            logger.info("[validator] selected-scope content → summarize_all")
            return {"action": "summarize_all", "params": preserved_params}
        return None

    @staticmethod
    def _guard_find_request_not_count(*, ql, qn, action, params, **kwargs) -> Optional[dict]:
        """Explicit file-finding requests with a concrete target should search, not count.
        Skipped for skill_dispatch — SkillDispatcher chose count deliberately.
        """
        if action != "count":
            return None

        # Selection-expert + deictic conflict is now resolved upstream in
        # ExpertIntentArbiter._selection_intent (deictic wins over inventory).
        # No special pre-bail needed here.

        if IntentValidator._is_deterministic_expert_route(params):
            return None


        scope = str((params or {}).get("_scope") or "").strip().lower()
        selection_mode = str((params or {}).get("_selection_mode") or "").strip().lower()
        if scope == "selected" or selection_mode == "selected_items":
            # "show me all selected files" / "which selected files" → count; bail.
            # "show me this file" / "show it" (singular deictic pronoun) → describe, not count.
            # NOTE: only "this" (demonstrative) and "it" (pronoun) are unambiguously singular.
            # "the" is a definite article that can refer to any set, so we exclude it here.
            #
            # NOTE: confidence threshold is intentionally NOT checked here.
            # This guard fires for SelectionExpert (non-skill_dispatch) routes only.
            # SelectionExpert always returns hardcoded high confidence (e.g. 0.97),
            # so a confidence check would never trigger — making it meaningless.
            # The LLM-priority confidence check belongs in _SKILL_DISPATCH_GUARD_CHAIN (guard 5).
            active_paths = kwargs.get("active_paths") or []

            _SINGULAR_DEICTIC_RE = re.compile(
                # "show/tell me this [file]" or "show/tell me it"
                r'\b(show\s+me|tell\s+me|display|read)\s+(this|it)\b'
                # also: "show/tell me this file/doc" explicitly
                r'|\b(show\s+me|tell\s+me)\s+this\s+(file|doc|document)\b'
                # bare "show it" / "display it"
                r'|\b(show|display|read)\s+it\b'
                r'|给我看这[个份]|展示这[个份]|读这[个份]',
                re.IGNORECASE,
            )
            if active_paths and _SINGULAR_DEICTIC_RE.search(ql):
                logger.debug("[validator] show/tell-me-this/it (selected count) → summarize_all query_chars=%s", len(ql or ""))
                media_type = IntentValidator._infer_requested_media_type(ql)
                return {
                    "action": "summarize_all",
                    "params": IntentValidator._build_selected_summary_params(
                        params,
                        media_type=media_type,
                    ),
                }
            logger.info("[validator] explicit find request guard skipped for selected-scope count/listing")
            return None


        _EXPLICIT_FIND_RE = re.compile(
            r'^(find|show|search|look\s+for|retrieve|get\s+me|give\s+me|display|locate)\b'
            r'|^(找|搜|查|给我看|给我找|帮我找|搜索|查找|查看|看看)',
            re.IGNORECASE,
        )
        if not _EXPLICIT_FIND_RE.search(ql):
            return None

        _COUNT_ONLY_RE = re.compile(
            r'\b(how\s+many|which\s+files\s+do\s+i\s+have|what\s+files\s+do\s+i\s+have|list\b)\b'
            r'|(有多少|哪些文件|列出)',
            re.IGNORECASE,
        )
        if _COUNT_ONLY_RE.search(ql):
            return None

        from core.intent_analyzer import IntentAnalyzer
        from core.intent.entity_experts import CategoryListExpert

        has_content_qualifier = bool(
            re.search(
                r'\b(about|regarding|containing|inside|content|detail|details|describe|summary|summarize|analysis|analyze)\b',
                ql,
                re.IGNORECASE,
            )
        )
        cat_result = CategoryListExpert.analyze(qn, has_content_qualifier=has_content_qualifier)
        if cat_result and str(cat_result.get("action")) == "search":
            logger.info("[validator] explicit find request: count → search(category)")
            return cat_result

        focus = IntentAnalyzer._extract_file_analysis_focus_query(qn)
        if focus:
            logger.info("[validator] explicit find request: count → search(query)")
            return {"action": "search", "params": {"query": qn}}
        return None

    @staticmethod
    def _strip_completed_action_preface(text: str) -> str:
        """Drop context-setting prefaces like "after count," before routing."""
        return re.sub(
            r"^\s*(?:after|following|once|after\s+doing|when\s+done\s+with)\s+"
            r"(?:the\s+)?(?:count|counting|count\s+(?:query|request|result|step))\s*,?\s*"
            r"(?:(?:then|next|now)\s+)?",
            "",
            str(text or ""),
            flags=re.IGNORECASE,
        ).strip()

    @staticmethod
    def _guard_skill_dispatch_find_request_not_count(*, ql, qn, action, params, **kwargs) -> Optional[dict]:
        """SkillDispatcher may over-weight prior count context in "after count, find ...".

        Only repair explicit retrieval commands over file/document categories. True
        count requests still pass through _guard_how_many_files before this guard.
        """
        if action != "count" or not IntentValidator._is_skill_dispatch_route(params):
            return None

        command_text = IntentValidator._strip_completed_action_preface(qn)
        command_lower = command_text.lower()
        if not command_text:
            return None

        if re.search(r"\b(?:how\s+many|number\s+of|total|count)\b|多少|几个|几份|统计|数量", command_lower):
            return None

        explicit_retrieval = bool(
            re.search(
                r"^\s*(?:find|search(?:\s+for)?|look\s+for|locate|retrieve|get(?:\s+me)?|fetch)\b"
                r"|^\s*(?:找|搜|查找|搜索|检索|给我找)",
                command_lower,
                re.IGNORECASE,
            )
        )
        file_scope = bool(
            re.search(
                r"\b(?:files?|documents?|docs?|spreadsheets?|worksheets?|tables?|data|csv|xlsx|xls|"
                r"images?|photos?|pictures?|videos?|audios?|media|reports?|resumes?)\b"
                r"|文件|文档|资料|表格|数据|图片|照片|视频|音频|报告|简历",
                command_text,
                re.IGNORECASE,
            )
        )
        if not explicit_retrieval or not file_scope:
            return None

        from core.intent.entity_experts import CategoryListExpert

        cat_result = CategoryListExpert.analyze(command_text)
        if cat_result and str(cat_result.get("action")) == "search":
            logger.info("[validator] skill_dispatch retrieval command: count → search(category)")
            return cat_result

        logger.info("[validator] skill_dispatch retrieval command: count → search(query)")
        return {"action": "search", "params": {"query": command_text}}

    @staticmethod
    def _guard_tell_me_this_or_it_to_process_previous(
        *, ql, qn, action, active_paths, last_results, **kwargs
    ) -> Optional[dict]:
        """'tell me this file', 'tell me it', 'show it' with selected/prev files → describe.

        skill_dispatch LLM sometimes classifies these as 'count' because it detects
        a selected-file reference. But the user wants a description/summary, not a count.

        Routing:
          - last_results exist  → process_previous (describes previous search hits)
          - active_paths only   → summarize_all    (describes directly selected files)
        """
        if action != "count":
            return None
        if not active_paths and not last_results:
            return None
        # LLM-priority: if LLM returned count with high confidence (e.g. a smarter model that
        # correctly distinguishes count vs summarize), trust it. Regex guard is a fallback only.
        if float(kwargs.get("confidence") or 0.7) >= 0.90:
            logger.info("[validator] tell-me-this/it guard deferred to high-confidence LLM count")
            return None
        # Detect explicit "tell me" describe intent ONLY — "show/describe" are too ambiguous
        # (e.g. "show me how many" / "show me all" are legitimate count queries)
        _DESCRIBE_RE = re.compile(
            r'\b(tell\s+me)\b'
            r'|(\u8bb2\u8bb2|\u8bf4\u8bf4|\u8bb2\u4e00\u4e0b)',
            re.IGNORECASE,
        )
        if not _DESCRIBE_RE.search(ql):
            return None
        # Detect pronoun/demonstrative reference to selected/previous files
        _REF_RE = re.compile(
            r'\b(it|this|these|them|this\s+file|these\s+files|the\s+file)\b'
            r'|(\u8fd9\u4e2a|\u8fd9\u4efd|\u8fd9\u4e9b|\u8fd9|\u5b83|\u5b83\u4eec)',
            re.IGNORECASE,
        )
        if not _REF_RE.search(ql):
            return None
        # Choose target action based on context
        if last_results:
            # Previous search results exist — describe those results
            logger.info(
                "[validator] tell-me-this/it (skill_dispatch count) → process_previous (has last_results, ql=%r)", ql
            )
            return {"action": "process_previous", "params": {}}
        else:
            # No previous results — user is referring to their directly-selected files
            logger.info(
                "[validator] tell-me-this/it (skill_dispatch count) → summarize_all (active_paths only, ql=%r)", ql
            )
            media_type = IntentValidator._infer_requested_media_type(ql)
            return {
                "action": "summarize_all",
                "params": IntentValidator._build_selected_summary_params(
                    kwargs.get("params") or {},
                    media_type=media_type,
                ),
            }


    @staticmethod
    def _guard_how_many_of_them(*, ql, qn, action, last_results, **kwargs) -> Optional[dict]:
        """'how many of them are PDFs' → count (not process_previous)."""
        if not last_results:
            return None
        m = re.search(r'\bhow\s+many\b.{0,15}\b(of\s+them|of\s+these)\b', ql)
        if m:
            logger.info(f"[validator] how-many-of-them: {action} → count")
            return {"action": "count", "params": {"category": "all"}}
        return None

    @staticmethod
    def _guard_open_file_requires_explicit(*, ql, qn, action, params, **kwargs) -> Optional[dict]:
        if action != "open_file":
            return None
        _EXPLICIT_OPEN_KWS = {"打开", "开启", "启动", "open", "launch"}
        if any(kw in ql for kw in _EXPLICIT_OPEN_KWS):
            return None  # pass through — user really wants to open
        file_name = str((params or {}).get("file_name") or "").strip()
        search_query = file_name or qn
        logger.info(
            f"[validator] open_file rejected (no explicit keyword), → search query_chars={len(search_query or '')}"
        )
        return {"action": "search", "params": {"query": search_query}}

    # ── Guard chain: ordered by priority ──────────────────────────────────
    # Removed from chain (handled by SkillDispatcher skill descriptions):
    #   _guard_prev_ref_keywords     — fully covered by _guard_pronoun_followup
    #   _guard_global_summary        — summarize_all is now a registered Skill
    _GUARD_CHAIN = [
        _guard_open_file_requires_explicit.__func__,
        _guard_scoped_metadata_count.__func__,
        _guard_selected_scope_summary.__func__,
        _guard_selected_extension_count.__func__,
        _guard_selected_scope_listing.__func__,
        _guard_list_selected.__func__,
        _guard_how_many_of_them.__func__,
        _guard_meta_followup.__func__,
        _guard_pronoun_followup.__func__,
        _guard_find_request_not_count.__func__,
        _guard_category_retrieval_not_count.__func__,
        _guard_file_content_analysis.__func__,
        _guard_scoped_file_search.__func__,
        _guard_how_many_files.__func__,
        _guard_short_affirm_view_detail.__func__,
    ]

    _SKILL_DISPATCH_GUARD_CHAIN = [
        _guard_open_file_requires_explicit.__func__,
        _guard_document_topic_not_media_category.__func__,
        _guard_scoped_metadata_count.__func__,
        _guard_how_many_files.__func__,
        _guard_skill_dispatch_find_request_not_count.__func__,
        # Correct skill_dispatch mis-classifying "tell me this/it" as count
        _guard_tell_me_this_or_it_to_process_previous.__func__,
    ]
