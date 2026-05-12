from __future__ import annotations

import os
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
from core.intent_analyzer import IntentAnalyzer, IntentContext
from core.intent.entity_experts import CategoryListExpert, EntitySearchExpert, FilenameExpert
from core.intent.search_scope_disambiguation import looks_like_search_request
from core.retrieval.filename_canonicalizer import (
    classify_reference_target,
    looks_like_thematic_lookup_candidate,
)
logger = get_logger()


def _category_for_media_type(media_type: str) -> str:
    value = str(media_type or "").strip().lower()
    return value if value in {"audio", "video"} else "audio/video"


_EXPLICIT_SELECTION_SCOPE_RE = re.compile(
    r"\b(selected|seleted|selectd|slected|chosen|picked|current\s+selection|"
    r"selected\s+(?:files?|documents?|docs?|folders?|items?|sources?))\b"
    r"|选中|已选|已勾选|当前选中|勾选",
    re.IGNORECASE,
)

_SELECTION_INVENTORY_RE = re.compile(
    r"\b(list|show|display|browse|count|how many|which files|what files|what are|show me)\b"
    r"|有哪些|列出|多少|几个|清单|查看",
    re.IGNORECASE,
)

_SELECTION_SEMANTIC_RE = re.compile(
    r"\b(tell(?:\s+me)?\s+about|summarize|summary|describe|overview|analyze|analysis|detail|details|"
    r"content|contains text|most important|most detailed|most relevant|best|worst|"
    r"focus only on|strongest sources|supporting files|evidence|heard|hear|speech|spoken|"
    r"what is said|what can be heard|if .* selected|if the selected)\b"
    r"|总结|概括|详情|内容|最重要|最详细|聚焦|证据|语音|说了什么|听到什么",
    re.IGNORECASE,
)

_NON_ACTION_UTTERANCE_RE = re.compile(
    r"^\s*(?:ok(?:ay)?|sure|yeah|yep|nope|thanks?|thank\s+you|"
    r"no\s+need|no\s+thanks|never\s*mind|all\s+set|cancel\s+that|"
    r"lol|lmao|haha(?:ha)*|hmm+|嗯+|哦+|好+|行+|谢谢|哈哈+|不用了|先不用|算了|不需要了)\s*[.!。！?？]*\s*$"
    r"|^\s*[\U0001F300-\U0001FAFF\u2600-\u27BF]+(?:\s*[\U0001F300-\U0001FAFF\u2600-\u27BF]+)*\s*$",
    re.IGNORECASE,
)

_EXPLICIT_FIND_REQUEST_RE = re.compile(
    r"^\s*(?:now\s+|please\s+|can\s+you\s+|could\s+you\s+)?"
    r"(?!(?:find\s+out)\b)(?:find|search(?:\s+for)?|look\s+for|locate|retrieve|get\s+me|show\s+me|pull\s+up)\b"
    r"|^\s*(?:找|搜|搜索|查找|检索|给我找|帮我找)",
    re.IGNORECASE,
)

_EXPLICIT_FIND_DEICTIC_RE = re.compile(
    r"\b(?:it|its|them|they|this|that|these|those|selected|current|same|previous|last|"
    r"first\s+(?:one|result|file)|second\s+(?:one|result|file)|"
    r"the\s+(?:file|document|doc|video|audio|result|same|above|previous|last))\b"
    r"|这个|那个|这些|那些|上一个|上一份|同一个|同一份",
    re.IGNORECASE,
)

_GLOBAL_SUMMARY_REQUEST_RE = re.compile(
    r"\b(?:summari[sz]e|summary|overview|digest|analy[sz]e|explain|tell\s+me\s+about)\b"
    r".{0,40}\b(?:all|everything|entire|whole|my\s+(?:files|documents|docs)|the\s+corpus|indexed\s+files)\b"
    r"|(?:总结|概括|总览|整体分析).{0,20}(?:全部|所有|整个|全库)",
    re.IGNORECASE,
)

_FILETYPE_FRAGMENT_FILLER_RE = re.compile(
    r"\b(?:looking|look|searching|search|find|show|list|get|for|only|just|"
    r"please|pls|kind|kindly|the|a|an|my|mine|me|i|want|need|would|like|to|"
    r"files?|documents?|docs?|items?)\b",
    re.IGNORECASE,
)

_PRIOR_SEARCH_TOPIC_FILLER_RE = re.compile(
    r"\b(?:please|pls|can|could|would|you|help|me|to|find|search|look|for|"
    r"locate|retrieve|get|show|list|display|browse|all|every|my|mine|the|a|an|"
    r"files?|documents?|docs?|items?|images?|photos?|pictures?|screenshots?|"
    r"videos?|audios?|recordings?|pdfs?|pdf|csv|xlsx|xls|pptx|ppt)\b",
    re.IGNORECASE,
)


class ExpertIntentArbiter:
    """
    Two-layer expert-first arbiter placed in front of the legacy IntentAnalyzer.

    Design goals:
    - keep only high-confidence deterministic routes as hard fast paths
    - move weak semantic routing to a compact LLM-mediated arbitration layer
    - avoid growing more brittle inline rules in dispatch / analyzer
    - keep legacy IntentAnalyzer only as the final fallback
    """

    def __init__(self, agent: Any):
        self.agent = agent

    @staticmethod
    def _repair_skill_dispatch_category_listing(ctx: IntentContext, result: Dict[str, Any]) -> Dict[str, Any]:
        """Recover category-listing requests that the LLM collapsed to count.

        This is intentionally narrow: only the soft SkillDispatcher route is
        repaired, and only when the existing CategoryListExpert can identify a
        file category.  Explicit quantity questions still stay count.
        """
        action = str((result or {}).get("action") or "").strip()
        if action != "count":
            return result
        question = str(getattr(ctx, "question", "") or "").strip()
        if re.search(r"\b(how\s+many|count|number\s+of|total)\b|有多少|多少个|多少份|统计", question, re.IGNORECASE):
            return result
        has_content_qualifier = bool(
            re.search(
                r"\b(detail|details|content|contents|about|describe|explain|analysis|analyze|summary|summarize)\b"
                r"|内容|详情|解释|分析|总结",
                question,
                re.IGNORECASE,
            )
        )
        cat_result = CategoryListExpert.analyze(question, has_content_qualifier=has_content_qualifier)
        if not cat_result or str(cat_result.get("action") or "") != "search":
            return result

        original_params = dict((result or {}).get("params") or {})
        repaired_params = dict(cat_result.get("params") or {})
        for key in ("_dispatch_reason", "_skill_name", "_candidate_scopes"):
            if key in original_params:
                repaired_params.setdefault(key, original_params[key])
        repaired_params.setdefault("_dispatch_repair", "category_listing_not_count")
        logger.info(
            "[ExpertIntentArbiter] repaired skill_dispatch count -> search(category=%r) for category listing",
            repaired_params.get("category"),
        )
        return {
            "action": "search",
            "params": repaired_params,
            "confidence": max(float(result.get("confidence", 0.0) or 0.0), float(cat_result.get("confidence", 0.0) or 0.0), 0.86),
        }

    @staticmethod
    def _repair_skill_dispatch_topic_inventory(ctx: IntentContext, result: Dict[str, Any]) -> Dict[str, Any]:
        """Do not let broad category inventory erase a topical qualifier."""
        action = str((result or {}).get("action") or "").strip()
        if action != "search":
            return result
        params = dict((result or {}).get("params") or {})
        if str(params.get("_inventory_mode") or "").strip().lower() != "category":
            return result

        question = str(getattr(ctx, "question", "") or "").strip()
        if not question:
            return result
        has_content_qualifier = bool(
            re.search(
                r"\b(of|with|containing|contains?|about|regarding|related\s+to|mentioning|"
                r"that\s+mention|that\s+contain|featuring|showing|depicting|involving)\b"
                r"|关于|包含|含有|带有|里面有|出现|展示|显示",
                question,
                re.IGNORECASE,
            )
        )
        if not has_content_qualifier:
            return result

        cat_result = CategoryListExpert.analyze(question, has_content_qualifier=True)
        if not cat_result or str(cat_result.get("action") or "") != "search":
            return result
        repaired_params = dict(cat_result.get("params") or {})
        if str(repaired_params.get("_inventory_mode") or "").strip().lower() == "category":
            return result
        for key in ("_dispatch_reason", "_skill_name", "_candidate_scopes"):
            if key in params:
                repaired_params.setdefault(key, params[key])
        repaired_params["_dispatch_repair"] = "topic_qualifier_not_inventory"
        logger.info(
            "[ExpertIntentArbiter] repaired skill_dispatch category inventory -> topical search "
            "(category=%r)",
            repaired_params.get("category"),
        )
        return {
            "action": "search",
            "params": repaired_params,
            "confidence": max(float((result or {}).get("confidence", 0.0) or 0.0), 0.86),
        }

    @staticmethod
    def _repair_skill_dispatch_missing_category_filter(ctx: IntentContext, result: Dict[str, Any]) -> Dict[str, Any]:
        """Preserve explicit file-type filters that the LLM omitted.

        SkillDispatcher owns the semantic decision, but small local models can
        keep the topic in ``params.query`` while dropping the filter implied by
        words such as "papers", "reports", "documents", "photos", or "CSV".
        Reusing CategoryListExpert here keeps the repair taxonomy-based and
        generic; it never changes a non-search action and never overwrites the
        LLM's retrieval query.
        """
        action = str((result or {}).get("action") or "").strip()
        if action != "search":
            return result

        params = dict((result or {}).get("params") or {})
        existing_category = str(params.get("category") or "").strip()
        if existing_category and existing_category.lower() not in {"all", "other", "unknown"}:
            return result

        question = str(getattr(ctx, "question", "") or "").strip()
        if not question:
            return result

        has_content_qualifier = bool(
            re.search(
                r"\b(of|with|containing|contains?|about|regarding|related\s+to|mentioning|"
                r"that\s+mention|that\s+contain|featuring|showing|depicting|involving)\b"
                r"|关于|包含|含有|带有|里面有|出现|展示|显示",
                question,
                re.IGNORECASE,
            )
        )
        cat_result = CategoryListExpert.analyze(question, has_content_qualifier=has_content_qualifier)
        if not cat_result or str(cat_result.get("action") or "") != "search":
            return result

        cat_params = dict(cat_result.get("params") or {})
        category = str(cat_params.get("category") or "").strip()
        if not category or category.lower() in {"all", "other", "unknown"}:
            return result
        if (
            category.lower() == "document"
            and str(cat_params.get("_inventory_mode") or "").strip().lower() != "category"
        ):
            logger.info(
                "[ExpertIntentArbiter] skipped generic document category repair for topical search"
            )
            return result

        repaired_params = dict(params)
        repaired_params["category"] = category
        for key in ("media_type", "file_extensions", "_inventory_mode"):
            if key in cat_params and cat_params.get(key) not in (None, ""):
                repaired_params.setdefault(key, cat_params.get(key))
        repaired_params["_dispatch_repair"] = "preserve_file_type_filter"
        logger.info(
            "[ExpertIntentArbiter] repaired skill_dispatch search by preserving category=%r",
            category,
        )
        return {
            "action": "search",
            "params": repaired_params,
            "confidence": max(float((result or {}).get("confidence", 0.0) or 0.0), 0.86),
        }

    @staticmethod
    def _infer_filetype_fragment_category(text: str) -> str:
        """Infer a file category only when the text is basically a type clarification."""
        raw = str(text or "").strip()
        if not raw:
            return ""
        ql = raw.lower()
        categories: List[str] = []
        aliases: List[str] = []
        for token, category in CategoryListExpert._TOKEN_TO_CAT.items():
            token_l = str(token or "").strip().lower()
            if not token_l:
                continue
            is_cjk = bool(re.search(r"[\u4e00-\u9fff]", token_l))
            if (is_cjk and token_l in ql) or (not is_cjk and re.search(rf"\b{re.escape(token_l)}\b", ql)):
                categories.append(str(category or "").strip())
                aliases.append(token_l)
        categories = [cat for cat in dict.fromkeys(categories) if cat]
        if len(categories) != 1:
            return ""

        remainder = ql
        for alias in sorted(set(aliases), key=len, reverse=True):
            if re.search(r"[\u4e00-\u9fff]", alias):
                remainder = remainder.replace(alias, " ")
            else:
                remainder = re.sub(rf"\b{re.escape(alias)}\b", " ", remainder)
        remainder = _FILETYPE_FRAGMENT_FILLER_RE.sub(" ", remainder)
        remainder = re.sub(r"[\W_]+", " ", remainder)
        leftover = [part for part in remainder.split() if part]
        return categories[0] if not leftover else ""

    @staticmethod
    def _extract_prior_search_topic(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        topic = raw.lower()
        topic = _PRIOR_SEARCH_TOPIC_FILLER_RE.sub(" ", topic)
        for token in sorted(CategoryListExpert._TOKEN_TO_CAT.keys(), key=lambda item: len(str(item)), reverse=True):
            token_l = str(token or "").strip().lower()
            if not token_l:
                continue
            if re.search(r"[\u4e00-\u9fff]", token_l):
                topic = topic.replace(token_l, " ")
            else:
                topic = re.sub(rf"\b{re.escape(token_l)}\b", " ", topic)
        topic = re.sub(r"[\W_]+", " ", topic).strip()
        topic = re.sub(r"\s+", " ", topic)
        return topic[:120].strip()

    def _prior_failed_search_query(self, ctx: IntentContext) -> str:
        getter = getattr(self.agent, "_get_followup_hint", None)
        if callable(getter):
            try:
                hint = getter(getattr(ctx, "session_id", None))
            except TypeError:
                hint = getter()
            except Exception:
                hint = None
            if isinstance(hint, dict) and str(hint.get("action") or "") == "process_previous":
                params = hint.get("params") if isinstance(hint.get("params"), dict) else {}
                if params.get("allow_without_results") and str(params.get("anchor") or "") == "search_topic":
                    prior = str(params.get("prior_search_query") or "").strip()
                    if prior:
                        return prior

        for item in reversed(list(getattr(ctx, "history", None) or [])[-4:]):
            if not isinstance(item, dict):
                continue
            answer = str(item.get("a") or item.get("content") or "").strip().lower()
            prior_q = str(item.get("q") or "").strip()
            if prior_q and re.search(r"\b(no|not)\s+(?:relevant\s+)?(?:indexed\s+)?(?:files?|content|results?)\b|clarify|unclear|scope|未找到|没有找到|不清楚", answer):
                return prior_q
        return ""

    def _repair_skill_dispatch_filetype_clarification(self, ctx: IntentContext, result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge a short file-type clarification with the previous failed search topic."""
        action = str((result or {}).get("action") or "").strip()
        if action != "search":
            return result
        params = dict((result or {}).get("params") or {})
        question = str(getattr(ctx, "question", "") or "").strip()
        fragment_category = self._infer_filetype_fragment_category(question)
        if not fragment_category:
            return result

        prior_query = self._prior_failed_search_query(ctx)
        prior_topic = self._extract_prior_search_topic(prior_query)
        if not prior_topic:
            return result

        current_query = str(params.get("query") or "").strip()
        current_topic = self._extract_prior_search_topic(current_query)
        if current_topic and current_topic != self._extract_prior_search_topic(question):
            return result

        repaired_params = dict(params)
        repaired_params["query"] = prior_topic
        repaired_params["category"] = self.agent._normalize_category_name(fragment_category) if hasattr(self.agent, "_normalize_category_name") else fragment_category
        repaired_params["_dispatch_repair"] = "filetype_clarification_merged_prior_topic"
        repaired_params["_prior_search_query"] = prior_query
        logger.info(
            "[ExpertIntentArbiter] repaired file-type clarification by merging prior topic=%r category=%r",
            prior_topic,
            repaired_params.get("category"),
        )
        return {
            "action": "search",
            "params": repaired_params,
            "confidence": max(float((result or {}).get("confidence", 0.0) or 0.0), 0.86),
        }

    @staticmethod
    def _repair_skill_dispatch_explicit_find(ctx: IntentContext, result: Dict[str, Any]) -> Dict[str, Any]:
        """Recover explicit retrieval requests that the LLM collapsed to summarize_all.

        The SkillDispatcher prompt already says specific reports/plans/entities
        should search.  This narrow exit repair handles the recurring failure
        mode without adding file-specific rules or disturbing true global
        overview requests.
        """
        action = str((result or {}).get("action") or "").strip()
        if action not in {"summarize_all", "process_previous"}:
            return result

        question = str(getattr(ctx, "question", "") or "").strip()
        if not question or not _EXPLICIT_FIND_REQUEST_RE.search(question):
            return result
        has_deictic_scope = bool(_EXPLICIT_FIND_DEICTIC_RE.search(question))
        has_selected_scope = bool(_EXPLICIT_SELECTION_SCOPE_RE.search(question))
        if has_selected_scope and getattr(ctx, "active_paths", None):
            original_params = dict((result or {}).get("params") or {})
            query = str(original_params.get("query") or question).strip()
            repaired_params = {
                **original_params,
                "query": query,
                "_scope": "selected",
                "_preserve_selected_scope": True,
                "_dispatch_repair": "selected_find_not_global_summary",
            }
            for key in ("scope", "_scope_kind", "operation", "_context_operation"):
                repaired_params.pop(key, None)
            logger.info(
                "[ExpertIntentArbiter] repaired skill_dispatch %s -> selected scoped search for explicit find request",
                action,
            )
            return {
                "action": "search",
                "params": repaired_params,
                "confidence": max(float((result or {}).get("confidence", 0.0) or 0.0), 0.86),
            }
        if has_deictic_scope:
            return result
        if _GLOBAL_SUMMARY_REQUEST_RE.search(question):
            return result

        original_params = dict((result or {}).get("params") or {})
        query = str(original_params.get("query") or question).strip()
        repaired_params = {
            **original_params,
            "query": query,
            "_dispatch_repair": "explicit_find_not_global_summary",
        }
        for key in (
            "scope",
            "_scope",
            "_scope_kind",
            "operation",
            "_context_operation",
            "_candidate_scopes",
            "_preserve_selected_scope",
        ):
            repaired_params.pop(key, None)
        logger.info(
            "[ExpertIntentArbiter] repaired skill_dispatch %s -> search for explicit find request",
            action,
        )
        return {
            "action": "search",
            "params": repaired_params,
            "confidence": max(float((result or {}).get("confidence", 0.0) or 0.0), 0.86),
        }

    @staticmethod
    def _repair_skill_dispatch_contextual_refine(ctx: IntentContext, result: Dict[str, Any]) -> Dict[str, Any]:
        """Keep contextual refinement on the prior turn instead of summarizing the whole library.

        SkillDispatcher sometimes labels compare/rewrite follow-ups as
        contextual_refine but the selected-scope planner converts them to
        summarize_all when last_results are not populated.  If the visible
        active scope is effectively the whole library, summarize_all becomes a
        corpus summary.  Route those refinement operations back through
        process_previous, where follow-up context and active media/doc context
        are handled.
        """
        action = str((result or {}).get("action") or "").strip()
        if action != "summarize_all":
            return result

        params = dict((result or {}).get("params") or {})
        skill_name = str(params.get("_skill_name") or "").strip()
        if skill_name != "contextual_refine":
            return result

        operation = str(params.get("operation") or params.get("_context_operation") or "").strip().lower()
        if operation not in {"qa", "rewrite", "support"}:
            return result

        scope = str(params.get("scope") or params.get("_scope_kind") or params.get("_scope") or "").strip()
        active_count = len(getattr(ctx, "active_paths", None) or [])
        history_items = list(getattr(ctx, "history", None) or [])
        has_prior_context = bool(getattr(ctx, "last_results", None)) or len(history_items) > 1
        if scope in {"selected_items", "selected_folder", "selected"} and active_count <= 50 and not has_prior_context:
            return result

        repaired_params = dict(params)
        repaired_params["_dispatch_repair"] = "contextual_refine_not_global_summary"
        repaired_params.pop("_preserve_selected_scope", None)
        logger.info(
            "[ExpertIntentArbiter] repaired skill_dispatch contextual_refine summarize_all -> process_previous "
            "(operation=%s, scope=%s, active_paths=%s)",
            operation,
            scope,
            active_count,
        )
        return {
            "action": "process_previous",
            "params": repaired_params,
            "confidence": max(float((result or {}).get("confidence", 0.0) or 0.0), 0.86),
        }

    @staticmethod
    def _repair_skill_dispatch_scoped_comparison(ctx: IntentContext, result: Dict[str, Any]) -> Dict[str, Any]:
        """Keep deictic comparison/ranking questions on the previous result set.

        When prior-context hydration is weak, SkillDispatcher may turn queries
        like "which one fits best" into a fresh search because the query
        contains role or job words. These are still scoped judgments over the
        already-found items and should go through process_previous.
        """
        action = str((result or {}).get("action") or "").strip()
        if action != "search":
            return result

        question = str(getattr(ctx, "question", "") or "").strip()
        if not question or looks_like_search_request(question):
            return result

        history_items = list(getattr(ctx, "history", None) or [])
        has_prior_context = bool(getattr(ctx, "last_results", None)) or len(history_items) > 1
        if not has_prior_context:
            return result

        from core.intent.context_followup_expert import ContextFollowupExpert

        if not ContextFollowupExpert._SCOPED_COMPARISON_RE.search(question):
            return result

        repaired_params = dict((result or {}).get("params") or {})
        repaired_params["_dispatch_repair"] = "scoped_comparison_not_search"
        logger.info(
            "[ExpertIntentArbiter] repaired skill_dispatch search -> process_previous "
            "for scoped comparison/ranking follow-up"
        )
        return {
            "action": "process_previous",
            "params": repaired_params,
            "confidence": max(float((result or {}).get("confidence", 0.0) or 0.0), 0.86),
        }

    @staticmethod
    def _repair_skill_dispatch_scoped_content_followup(ctx: IntentContext, result: Dict[str, Any]) -> Dict[str, Any]:
        """Keep content questions after a prior search scoped to the prior result set.

        Small local dispatch models sometimes see a short topical question like
        "how many stages" or "what parameters" as a fresh search/count because
        it contains no pronoun. If there is prior search/content context and no
        explicit new-search wording, this is usually a QA follow-up over the
        previous files, not a global corpus operation.
        """
        action = str((result or {}).get("action") or "").strip()
        if action not in {"search", "count"}:
            return result

        question = str(getattr(ctx, "question", "") or "").strip()
        if not question or looks_like_search_request(question):
            return result

        from core.intent.context_followup_expert import ContextFollowupExpert

        ql = question.lower()
        prior_ctx = IntentAnalyzer._extract_prior_action_context(ctx)
        history_items = list(getattr(ctx, "history", None) or [])
        has_prior_context = bool(getattr(ctx, "last_results", None)) or len(history_items) > 1 or bool(
            prior_ctx.get("prior_was_search")
            or prior_ctx.get("prior_was_content")
            or prior_ctx.get("focused_file")
        )
        if not has_prior_context:
            return result

        if (
            ContextFollowupExpert._NEW_SCOPE_RE.search(ql)
            or ContextFollowupExpert._GLOBAL_SCOPE_SUMMARIZE_RE.search(ql)
            or ContextFollowupExpert._SEARCH_VERB_ENTITY_RE.search(ql)
        ):
            return result
        query_file_type_hints = ContextFollowupExpert._query_file_type_hints(question)
        if query_file_type_hints:
            prior_file_type_hints = ContextFollowupExpert._result_file_type_hints(
                getattr(ctx, "last_results", None)
            )
            if prior_file_type_hints and not query_file_type_hints.intersection(prior_file_type_hints):
                logger.info(
                    "[ExpertIntentArbiter] scoped content repair skipped for explicit file-type switch: "
                    "query_hints=%s prior_hints=%s",
                    sorted(query_file_type_hints),
                    sorted(prior_file_type_hints),
                )
                return result

        explicit_file_recount = bool(ContextFollowupExpert._RECOUNT_RE.search(ql))
        scoped_count_question = bool(
            re.search(
                r"\b(?:how\s+many|number\s+of|count|total)\b"
                r"|多少|几个|几份|几条|几项|几种|几类|分几",
                ql,
                re.IGNORECASE,
            )
        ) and not explicit_file_recount
        scoped_content_question = bool(
            ContextFollowupExpert._CONTENT_QUESTION_RE.search(question)
            or ContextFollowupExpert._SCOPED_MENTION_QA_RE.search(question)
            or ContextFollowupExpert._SCOPED_METADATA_QA_RE.search(question)
            or (len([w for w in ql.split() if w]) <= 10 and ContextFollowupExpert._SHORT_FRAGMENT_FOLLOWUP_RE.search(question))
            or scoped_count_question
        )
        if not scoped_content_question:
            return result

        if prior_ctx.get("prior_was_media") and ContextFollowupExpert._MEDIA_FOLLOWUP_SIGNAL_RE.search(ql):
            return result

        repaired_params = dict((result or {}).get("params") or {})
        repaired_params["_dispatch_repair"] = (
            "scoped_content_count_not_global_count"
            if action == "count"
            else "scoped_content_question_not_search"
        )
        logger.info(
            "[ExpertIntentArbiter] repaired skill_dispatch %s -> process_previous "
            "for scoped content follow-up",
            action,
        )
        return {
            "action": "process_previous",
            "params": repaired_params,
            "confidence": max(float((result or {}).get("confidence", 0.0) or 0.0), 0.86),
        }

    @staticmethod
    def _skill_dispatch_image_content_misroute(ctx: IntentContext, params: Dict[str, Any]) -> bool:
        """Reject media_content_search for still-image understanding.

        media_content_search is backed by audio/video transcript, keyframe, and
        media-summary indexes. A still image question should stay in selected or
        previous-file reasoning so the normal image/document handlers can answer.
        """
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".heic"}
        media_type = str((params or {}).get("media_type") or "").strip().lower()
        if media_type in {"image", "images", "photo", "photos", "picture", "pictures"}:
            return True

        file_hint = str(
            (params or {}).get("file_hint")
            or (params or {}).get("focused_file")
            or ""
        ).strip()
        if os.path.splitext(file_hint)[1].lower() in image_exts:
            return True

        active_paths = [str(path or "").strip() for path in list(getattr(ctx, "active_paths", None) or []) if str(path or "").strip()]
        if len(active_paths) == 1 and os.path.splitext(active_paths[0])[1].lower() in image_exts:
            question = str(getattr(ctx, "question", "") or "")
            return bool(re.search(r"\b(?:image|photo|picture)\b|图片|照片|图里|这张图", question, re.IGNORECASE))
        return False

    def _build_intent_context(self, query_context: Any) -> IntentContext:
        return IntentContext(
            question=query_context.normalized_question,
            prompt_language=query_context.prompt_language,
            user_lang=query_context.user_language,
            history=list(query_context.history or []),
            last_results=list(query_context.last_results or []),
            get_category_keywords_fn=self.agent._get_rule_category_keywords,
            is_generic_category_fn=self.agent._is_generic_file_scope_category,
            normalize_category_fn=self.agent._normalize_category_name,
            llm_service=self.agent._get_llm_service(
                detailed=False,
                session_id=query_context.session_id,
                prompt_language=query_context.prompt_language,
            ),
            category_info=self.agent._get_category_stats(prompt_language=query_context.prompt_language),
            prompt_formatter=self.agent._prompt,
            log_followup_guard_fn=getattr(self.agent, "_log_followup_guard", None),
            session_id=query_context.session_id,
            active_paths=list(query_context.active_paths or []),
            opened_file_path=getattr(query_context, "opened_file_path", None),
        )

    @staticmethod
    def _infer_filename_category(question: str, explicit_ref: Dict[str, str]) -> str:
        ql = str(question or "").lower()
        raw_name = str(explicit_ref.get("raw_name") or "").lower()
        search_term = str(explicit_ref.get("search_term") or "").lower()
        combined = " ".join([ql, raw_name, search_term])

        has_doc_cue = any(token in combined for token in [
            "document", "documents", "doc", "docs", "paper", "papers", "pdf",
            "text file", "text files", "plain text", "txt", "markdown", "md",
            "文件", "文档", "论文", "资料", "文本", "文字",
        ])
        has_media_extension = any(token in " ".join([raw_name, search_term]) for token in [
            ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
            ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg",
            ".mp4", ".mov", ".mkv", ".avi", ".webm",
        ])
        if has_doc_cue and not has_media_extension:
            return "document"

        if any(token in combined for token in ["image", "images", "photo", "picture", "图片", "照片", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]):
            return "image"
        if any(token in combined for token in ["spreadsheet", "table", "tables", "csv", "excel", "xlsx", "xls", "numbers", "表格", "数据表"]):
            return "data"
        if any(token in combined for token in ["audio", "recording", "sound", "wav", "mp3", "m4a", "音频", "录音"]):
            return "audio"
        if any(token in combined for token in ["video", "clip", "movie", "mp4", "mov", "mkv", "视频"]):
            return "video"
        return ""

    def _explicit_filename_intent(self, query_context: Any) -> Optional[Dict[str, Any]]:
        extractor = getattr(self.agent, "_extract_explicit_file_reference", None)
        if not callable(extractor):
            return None

        question = str(getattr(query_context, "question", "") or "").strip()
        if not question:
            return None
        explicit_ref = extractor(question)
        if not isinstance(explicit_ref, dict):
            return None

        raw_name = str(explicit_ref.get("raw_name") or "").strip()
        search_term = str(explicit_ref.get("search_term") or "").strip()
        lookup_name = raw_name or search_term
        if not lookup_name:
            return None
        ql = question.lower()
        thematic_filelike_lookup = bool(
            re.search(
                r"\b(?:documentation|docs?|guide|guides?|manual|manuals?|papers?|articles?|"
                r"frameworks?|runtime|runtimes?)\s+(?:about|for|on|related\s+to)\b"
                r"|\b(?:about|regarding|related\s+to|concerning)\b.{0,60}"
                r"\b(?:frameworks?|runtime|runtimes?|documentation|docs?|guides?|manuals?)\b",
                ql,
                re.IGNORECASE,
            )
        )
        has_extension = bool(re.search(r"\.[A-Za-z0-9]{1,12}$", os.path.basename(lookup_name)))
        has_filename_marker = bool(
            re.search(
                r"\b(?:file\s+named|file\s+called|filename|file\s*name|named\s+file|called\s+file)\b"
                r"|文件名|名为|叫.{0,40}(?:文件|文档|pdf|docx|xlsx|csv)",
                question,
                re.IGNORECASE,
            )
        )
        if not (has_extension or has_filename_marker):
            return None
        thematic_filelike_lookup = thematic_filelike_lookup or looks_like_thematic_lookup_candidate(
            question,
            lookup_name,
        )
        if thematic_filelike_lookup and not has_filename_marker:
            logger.info(
                "[ExpertIntentArbiter] explicit filename expert deferred topical file-like lookup: %r",
                question,
            )
            return None

        open_request = bool(
            re.search(r"\b(?:open|launch)\b|打开|开启|启动", question, re.IGNORECASE)
        )

        params: Dict[str, Any] = {
            "query": lookup_name,
            "_explicit_file_ref": explicit_ref,
            "_expert_route": "explicit_filename",
        }
        category = self._infer_filename_category(question, explicit_ref)
        if category:
            params["category"] = category

        if open_request:
            params["file_name"] = lookup_name
            intent = {"action": "open_file", "params": params, "confidence": 0.98}
        else:
            intent = {"action": "search", "params": params, "confidence": 0.98}
        logger.info(f"[ExpertIntentArbiter] explicit filename expert matched: {intent}")
        return intent

    def _selection_intent(self, ctx: IntentContext) -> Optional[Dict[str, Any]]:
        from core.intent.selection_expert import SelectionExpert

        qn = str(ctx.question or "")
        ql = qn.lower().strip()
        has_prior_context = bool(ctx.last_results or ctx.history)
        ref_kind = str(classify_reference_target(qn).get("kind") or "").strip().lower()
        followup_rewrite_signal = bool(
            re.search(
                r"\b(make it shorter|shorten that|make it more detailed|more detail|"
                r"strongest sources|what evidence supports|supporting files|focus only on)\b"
                r"|缩短|简短一点|更详细|证据|依据|支撑文件|只看",
                ql,
                re.IGNORECASE,
            )
        )
        if has_prior_context and followup_rewrite_signal:
            return None
        if not SelectionExpert.should_activate(qn, ctx.active_paths):
            return None

        _EXPLICIT_COUNT_RE = re.compile(
            r"\b(how\s+many|count|number\s+of|how\s+much|几个|多少|有多少|统计)\b",
            re.IGNORECASE,
        )
        _DEICTIC_FILE_SCOPE_RE = re.compile(
            r"\b(this|that|these|those|selected|current)\s+"
            r"(files?|documents?|docs?|folders?|items?|sources?|videos?|audios?|images?|photos?|pictures?|recordings?)\b"
            r"|这(个|些)(文件|文档|资料|视频|音频|图片)|那(个|些)(文件|文档|资料|视频|音频|图片)",
            re.IGNORECASE,
        )
        explicit_selection_scope = bool(_EXPLICIT_SELECTION_SCOPE_RE.search(qn))
        _prior_action_arb = str(getattr(ctx, "prior_intent_action", "") or "")

        if (
            ref_kind == "deictic"
            and ctx.history
            and not explicit_selection_scope
            and not _DEICTIC_FILE_SCOPE_RE.search(qn)
        ):
            logger.info(
                "[ExpertIntentArbiter] bare deictic follow-up deferred away from selected scope: %r",
                qn,
            )
            return None

        if (
            ref_kind == "deictic"
            and has_prior_context
            and not explicit_selection_scope
            and _prior_action_arb in {"search", "count", "summarize", "summarize_all", "process_previous"}
        ):
            logger.info(
                "[ExpertIntentArbiter] selected deictic deferred to prior file context instead of selected scope: %r",
                qn,
            )
            return None

        if (
            ref_kind == "deictic"
            and len(ctx.active_paths or []) == 1
            and getattr(ctx, "llm_service", None) is not None
            and not _EXPLICIT_COUNT_RE.search(ql)
        ):
            logger.info(
                "[ExpertIntentArbiter] selected single-file deictic deferred to skill_dispatch "
                "for semantic operation choice: %r",
                qn,
            )
            return None

        result = SelectionExpert.classify(
            qn,
            ctx.active_paths or [],
            ctx.last_results,
            llm_service=ctx.llm_service,
            lang=ctx.prompt_language,
            prior_action=_prior_action_arb,
        )
        intent = result.to_intent()
        if intent is None:
            logger.info(
                "[ExpertIntentArbiter] selection expert deferred to skill dispatch after non-selection verdict: %r",
                qn,
            )
        if intent is not None:
            intent.setdefault("params", {})["_expert_route"] = "selection"
            logger.info(f"[ExpertIntentArbiter] selection expert matched: {intent}")
        return intent

    def _category_list_intent(self, ctx: IntentContext) -> Optional[Dict[str, Any]]:
        qn = str(ctx.question or "")
        if (ctx.last_results or ctx.history) and re.search(
            r"\b(?:this|that|these|those|them|it|previous|prior|above|last)\b"
            r"|这几份|这几个|这些|那些|它们|这个|那个|上面|上述|前面|之前|里面",
            qn,
            re.IGNORECASE,
        ):
            logger.info(
                "[ExpertIntentArbiter] category list deferred to context follow-up for scoped reference: %r",
                qn,
            )
            return None
        has_content_qualifier = bool(
            IntentAnalyzer._looks_like_file_content_analysis_query(ctx.question)
        )
        folder_hint = self._folder_listing_intent(
            SimpleNamespace(
                question=ctx.question,
                active_paths=list(ctx.active_paths or []),
                last_results=list(ctx.last_results or []),
                session_id=ctx.session_id,
            )
        )
        intent = CategoryListExpert.analyze(
            ctx.question,
            has_content_qualifier=has_content_qualifier,
        )
        if intent is not None:
            params = intent.setdefault("params", {})
            folder_params = dict((folder_hint or {}).get("params") or {})
            explicit_folder = str(folder_params.get("folder") or "").strip()
            folder_has_specific_category = bool(str(folder_params.get("category") or "").strip())
            if explicit_folder and not folder_has_specific_category:
                logger.info(
                    "[ExpertIntentArbiter] category list expert deferred to explicit folder listing: "
                    "generic folder listing should keep folder/file/content merge intact"
                )
                return None
            if (
                getattr(ctx, "llm_service", None) is not None
                and str(params.get("_inventory_mode") or "").strip().lower() != "category"
            ):
                logger.info(
                    "[ExpertIntentArbiter] category list expert deferred to skill dispatch: "
                    "non-inventory category search should preserve topic/query semantics"
                )
                return None
            intent.setdefault("params", {})["_expert_route"] = "category_list"
            logger.info(f"[ExpertIntentArbiter] category list expert matched: {intent}")
        return intent

    def _entity_search_intent(self, query_context: Any) -> Optional[Dict[str, Any]]:
        question = str(query_context.question or "").strip()
        if not question:
            return None

        if FilenameExpert.is_bare_filename(question):
            intent = FilenameExpert.to_intent(question)
            intent.setdefault("params", {})["_expert_route"] = "bare_filename"
            return intent

        if EntitySearchExpert.is_bare_entity(question, active_paths=list(query_context.active_paths or [])):
            intent = EntitySearchExpert.to_intent(question)
            intent.setdefault("params", {})["_expert_route"] = "bare_entity"
            return intent

        return None

    def _personal_attribute_intent(self, ctx: IntentContext) -> Optional[Dict[str, Any]]:
        from core.intent.context_followup_expert import ContextFollowupExpert

        ql = str(ctx.question or "").lower().strip()
        if not ql or not ContextFollowupExpert._ATTR_LOOKUP_RE.search(ql):
            return None

        intent = {
            "action": "search",
            "params": {
                "query": ctx.question,
                "_expert_route": "personal_attribute",
            },
            "confidence": 0.99,
        }
        logger.info(f"[ExpertIntentArbiter] personal attribute expert matched: {intent}")
        return intent

    def _context_followup_intent(self, ctx: IntentContext) -> Optional[Dict[str, Any]]:
        """Reuse the canonical follow-up expert before broad media/search routing."""
        from core.intent.context_followup_expert import ContextFollowupExpert
        from core.intent.selection_expert import SelectionExpert

        if not (ctx.last_results or ctx.history):
            return None
        if (
            ctx.active_paths
            and _EXPLICIT_SELECTION_SCOPE_RE.search(str(ctx.question or ""))
            and SelectionExpert.should_activate(ctx.question, ctx.active_paths)
        ):
            logger.info(
                "[ExpertIntentArbiter] context follow-up deferred explicit selected scope: %r",
                ctx.question,
            )
            return None
        prior_ctx = IntentAnalyzer._extract_prior_action_context(ctx)
        intent = ContextFollowupExpert.analyze_context_followup(
            ctx.question,
            prior_ctx,
            last_results=ctx.last_results,
            active_paths=ctx.active_paths,
        )
        if intent is None:
            question = str(ctx.question or "").strip()
            ql = question.lower()
            ref_kind = str(classify_reference_target(question).get("kind") or "").strip().lower()
            weak_followup_signal = (
                ref_kind == "deictic"
                or IntentAnalyzer.looks_like_meta_followup_on_last_results(
                    question,
                    ctx.prompt_language,
                )
                or IntentAnalyzer.looks_like_content_followup_on_prior_results(question)
                or bool(
                    re.search(
                        r"\b(what|which|who|where|when|why|how)\b"
                        r"|什么|哪些|哪个|谁|哪里|为何|为什么|怎么|如何",
                        ql,
                        re.IGNORECASE,
                    )
                )
            )
            if (
                prior_ctx.get("prior_was_search")
                and prior_ctx.get("prior_search_failed")
                and weak_followup_signal
                and not looks_like_search_request(question)
            ):
                intent = {
                    "action": "process_previous",
                    "params": {},
                    "confidence": 0.78,
                }
                logger.info(
                    "[ExpertIntentArbiter] weak-result follow-up rescue → process_previous: %r",
                    question,
                )
            elif (
                prior_ctx.get("prior_was_search")
                and ref_kind == "deictic"
                and weak_followup_signal
                and not looks_like_search_request(question)
            ):
                intent = {
                    "action": "process_previous",
                    "params": {},
                    "confidence": 0.8,
                }
                logger.info(
                    "[ExpertIntentArbiter] deictic history follow-up rescue → process_previous: %r",
                    question,
                )
        if intent is None:
            return None
        params = intent.setdefault("params", {})
        if str(intent.get("action") or "") in {"media_export", "media_content_search"} and not params.get("file_hint"):
            prior_media = [
                str(item.get("file_path") or item.get("file_name") or "").strip()
                for item in list(ctx.last_results or [])
                if self._results_are_predominantly_media([item])
            ]
            prior_basenames = [
                os.path.basename(path).strip()
                for path in prior_media
                if os.path.basename(path).strip()
            ]
            unique_basenames = {name.lower() for name in prior_basenames}
            if prior_basenames and (
                len(prior_basenames) == 1
                or len(unique_basenames) == 1
            ):
                params["file_hint"] = prior_basenames[0]
        params["_expert_route"] = "context_followup"
        logger.info(f"[ExpertIntentArbiter] context follow-up expert matched: {intent}")
        return intent

    def _non_action_intent(self, ctx: IntentContext) -> Optional[Dict[str, Any]]:
        qn = str(ctx.question or "").strip()
        if not qn or not _NON_ACTION_UTTERANCE_RE.search(qn):
            return None
        question = (
            "I can't tell what you'd like me to do with the files yet. "
            "Try a specific request like \"summarize this file\" or \"what happens around 15 seconds?\""
        )
        if str(ctx.prompt_language or ctx.user_lang or "").lower().startswith("zh"):
            question = "我还不确定你想让我对文件做什么。可以直接说“总结这个文件”或“看一下 15 秒附近发生了什么”。"
        intent = {
            "action": "clarify",
            "params": {
                "question": question,
                "_expert_route": "non_action",
            },
            "confidence": 0.99,
        }
        logger.info(f"[ExpertIntentArbiter] non-action utterance matched: {intent}")
        return intent

    def _folder_listing_intent(self, query_context: Any) -> Optional[Dict[str, Any]]:
        question = str(query_context.question or "").strip()
        if not question:
            return None

        q = question.strip()
        ql = q.lower()
        folder = ""
        kind = ""

        zh = re.match(
            r"^\s*(?:找|查找|搜索|搜一下|查一下|看看|看下)\s*(?P<folder>.+?)\s*(?:目录|文件夹)(?:里|中的|下)?(?:的)?\s*"
            r"(?P<kind>视频|音频|录音|图片|照片|文档|文件|表格|数据表|数据|工作表|简历|手册|说明书|报告|论文|发票|幻灯片|演示文稿|代码)?\s*$",
            q,
            re.IGNORECASE,
        )
        if zh:
            folder = str(zh.group("folder") or "").strip()
            kind = str(zh.group("kind") or "").strip().lower()
        else:
            en = re.match(
                r"^\s*(?:find|show|search\s+for|look\s+for|get)\s+"
                r"(?P<kind>videos?|audio(?:\s+files?)?|recordings?|images?|photos?|pictures?|documents?|docs?|files?|"
                r"spreadsheets?|worksheets?|tables?|resumes?|manuals?|guides?|reports?|papers?|"
                r"invoices?|slides?|presentations?|code)\s+"
                r"in\s+(?:the\s+)?(?:folder|directory)\s+(?P<folder>.+?)\s*$",
                q,
                re.IGNORECASE,
            )
            if en:
                folder = str(en.group("folder") or "").strip()
                kind = str(en.group("kind") or "").strip().lower()
                kind = re.sub(r"\s+files?$", "", kind).strip()
            else:
                folder_lookup = re.match(
                    r"^\s*(?:please\s+)?(?:help\s+me\s+(?:to\s+)?|can\s+you\s+)?"
                    r"(?:find|show|search\s+for|look\s+for|get|open|locate)\s+"
                    r"(?:the\s+)?(?:folder|directory|dir)\s+(?:named\s+|called\s+)?(?P<folder>.+?)\s*$",
                    q,
                    re.IGNORECASE,
                )
                if folder_lookup:
                    folder = str(folder_lookup.group("folder") or "").strip()
                    kind = ""
                else:
                    tail_folder_lookup = re.match(
                        r"^\s*(?:please\s+)?(?:help\s+me\s+(?:to\s+)?|can\s+you\s+)?"
                        r"(?:find|show|search\s+for|look\s+for|get|open|locate|找|查找|搜索|搜一下|查一下|看看|看下)\s+"
                        r"(?P<folder>.+?(?:folder|directory|dir|文件夹|目录))\s*$",
                        q,
                        re.IGNORECASE,
                    )
                    if tail_folder_lookup:
                        folder = str(tail_folder_lookup.group("folder") or "").strip()
                        kind = ""

        if not folder:
            return None

        folder = folder.strip(" \"'“”‘’.")

        category = ""
        retrieval_query = folder
        if kind in {"视频", "video", "videos"}:
            category = "video"
            retrieval_query = "video"
        elif kind in {"音频", "录音", "audio", "recording", "recordings"}:
            category = "audio"
            retrieval_query = "audio"
        elif kind in {"图片", "照片", "image", "images", "photo", "photos", "picture", "pictures"}:
            category = "image"
            retrieval_query = "image"
        elif kind in {"文档", "document", "documents", "doc", "docs"}:
            category = "document"
            retrieval_query = "document"
        elif kind in {"表格", "数据表", "数据", "工作表", "spreadsheet", "spreadsheets", "worksheet", "worksheets", "table", "tables"}:
            category = "data"
            retrieval_query = "data"
        elif kind in {"简历", "resume", "resumes"}:
            category = "resume"
            retrieval_query = "resume"
        elif kind in {"手册", "说明书", "manual", "manuals", "guide", "guides"}:
            category = "manual"
            retrieval_query = "manual"
        elif kind in {"报告", "report", "reports"}:
            category = "report"
            retrieval_query = "report"
        elif kind in {"论文", "paper", "papers"}:
            category = "paper"
            retrieval_query = "paper"
        elif kind in {"发票", "invoice", "invoices"}:
            category = "invoice"
            retrieval_query = "invoice"
        elif kind in {"幻灯片", "演示文稿", "slide", "slides", "presentation", "presentations"}:
            category = "presentation"
            retrieval_query = "presentation"
        elif kind in {"代码", "code"}:
            category = "code"
            retrieval_query = "code"

        params: Dict[str, Any] = {
            "query": retrieval_query,
            "_expert_route": "folder_listing",
            "folder": folder,
        }
        if category:
            params["category"] = category
        if kind:
            params["_folder_listing_kind"] = kind

        intent = {
            "action": "search",
            "params": params,
            "confidence": 0.99,
        }
        logger.info(f"[ExpertIntentArbiter] folder listing expert matched: {intent}")
        return intent

    @staticmethod
    def _results_are_predominantly_media(last_results: List[Dict[str, Any]]) -> bool:
        if not last_results:
            return False
        media_exts = {
            ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
            ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv",
        }
        sample = list(last_results[:12])
        media_hits = 0
        from core.retrieval.category_engine import is_media_category_value
        for doc in sample:
            doc_category = str(doc.get("doc_category") or "").strip().lower()
            file_name = str(doc.get("file_name") or doc.get("file_path") or "").strip()
            ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
            if is_media_category_value(doc_category) or (ext and f".{ext}" in media_exts):
                media_hits += 1
        return bool(sample) and media_hits >= max(1, (len(sample) + 1) // 2)

    @staticmethod
    def _active_media_scope(active_paths: Optional[List[str]]) -> List[str]:
        media_exts = {
            ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
            ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv",
        }
        scoped: List[str] = []
        for raw_path in list(active_paths or []):
            fp = str(raw_path or "").strip()
            if not fp:
                continue
            if os.path.splitext(fp)[1].lower() in media_exts:
                scoped.append(fp)
        return scoped

    @staticmethod
    def _query_is_media_specific(question: str) -> bool:
        return bool(
            re.search(
                r"\b(?:audio|video|recording|podcast|clip|media|mp3|wav|m4a|mp4|mov|"
                r"heard|hear|said|sung|transcript|scene|frame|timestamp|footage)\b"
                r"|音频|视频|录音|录像|媒体|听到|说了|唱了|转写|画面|场景|时间戳",
                str(question or ""),
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _selected_video_frame_position(question: str) -> str:
        ql = str(question or "").strip().lower()
        if not ql:
            return ""
        if re.search(r"\b(?:first|initial|opening|beginning|start)\s+(?:frame|scene|shot)\b", ql):
            return "first"
        if re.search(r"\b(?:last|final|ending|end)\s+(?:frame|scene|shot)\b", ql):
            return "last"
        if re.search(r"\b(?:middle|midpoint|center|centre)\b.{0,24}\b(?:frame|scene|shot|video)\b", ql):
            return "middle"
        return ""

    def _media_intent(self, ctx: IntentContext) -> Optional[Dict[str, Any]]:
        from core.intent.media_query_expert import MediaQueryExpert
        from core.skills import MediaFollowupSkill

        qn = str(ctx.question or "").strip()
        active_media_paths = self._active_media_scope(ctx.active_paths)
        active_scope_paths = [str(path or "").strip() for path in list(ctx.active_paths or []) if str(path or "").strip()]
        active_scope_is_media_only = bool(active_media_paths) and len(active_media_paths) == len(active_scope_paths)
        mixed_selected_scope = bool(active_media_paths) and not active_scope_is_media_only
        media_specific_query = self._query_is_media_specific(qn)
        focused_media_path = ""
        opened_file_path = str(getattr(ctx, "opened_file_path", "") or "").strip()
        if opened_file_path and self._active_media_scope([opened_file_path]):
            focused_media_path = opened_file_path
        if not focused_media_path and active_scope_is_media_only and len(active_media_paths) == 1:
            focused_media_path = active_media_paths[0]
        if not focused_media_path and len(getattr(ctx, "last_results", None) or []) == 1:
            prior_media = [
                str(item.get("file_path") or item.get("file_name") or "").strip()
                for item in list(getattr(ctx, "last_results", None) or [])
                if self._results_are_predominantly_media([item])
            ]
            if len(prior_media) == 1 and prior_media[0]:
                focused_media_path = prior_media[0]
        focused_media_name = os.path.basename(focused_media_path) if focused_media_path else ""
        if MediaQueryExpert.looks_like_explicit_media_file_search(qn):
            logger.info(
                "[ExpertIntentArbiter] media expert routed explicit media file search -> search: %r",
                qn,
            )
            ql = qn.lower()
            audio_only = bool(re.search(r"\b(?:audio|recording|recordings|podcast|song|songs|music|mp3|wav|m4a|flac|aac|ogg)\b|音频|录音", ql))
            video_only = bool(re.search(r"\b(?:video|videos|movie|movies|clip|clips|mp4|mov|mkv|avi|webm)\b|视频|录像|影片|短片", ql))
            media_type = "audio" if (audio_only and not video_only) else ("video" if (video_only and not audio_only) else "")
            file_hint = MediaQueryExpert._extract_file_hint(qn)
            if not file_hint:
                logger.info(
                    "[ExpertIntentArbiter] media expert deferred topical media inventory to skill dispatch: %r",
                    qn,
                )
                return None
            params: Dict[str, Any] = {
                "query": qn,
                "_expert_route": "media_explicit_file_search",
                "_dispatch_reason": "Explicit media file inventory or filename lookup should stay on the indexed file search pipeline.",
            }
            if media_type:
                params["media_type"] = media_type
            params["category"] = _category_for_media_type(media_type)
            return {
                "action": "search",
                "params": params,
                "confidence": 0.98,
            }
        if ctx.last_results and IntentAnalyzer.looks_like_meta_followup_on_last_results(qn, ctx.prompt_language):
            logger.info(
                "[ExpertIntentArbiter] media expert deferred to meta followup on prior results: "
                f"query_chars={len(qn or '')}"
            )
            return None
        media_context_active = bool(active_media_paths) or (
            ctx.last_results and self._results_are_predominantly_media(ctx.last_results)
        )
        if media_context_active and MediaFollowupSkill.looks_like_generic_overview_query(
            qn,
            file_hint=focused_media_name,
        ):
            if mixed_selected_scope and not media_specific_query:
                logger.info(
                    "[ExpertIntentArbiter] media expert deferred mixed selected overview: query_chars=%s",
                    len(qn or ""),
                )
                return None
            if active_media_paths:
                selected_exts = {os.path.splitext(path)[1].lower() for path in active_media_paths}
                video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv"}
                audio_exts = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape"}
                media_type = (
                    "video" if selected_exts and selected_exts <= video_exts
                    else "audio" if selected_exts and selected_exts <= audio_exts
                    else "all"
                )
                params: Dict[str, Any] = {
                    "category": _category_for_media_type(media_type),
                    "media_type": media_type,
                    "_scope": "selected",
                    "_preserve_selected_scope": True,
                    "_selection_media_scope": True,
                }
                if focused_media_name:
                    params["file_hint"] = focused_media_name
                elif len(active_media_paths) == 1:
                    params["file_hint"] = os.path.basename(active_media_paths[0])
                params["_expert_route"] = "media"
                logger.info(
                    "[ExpertIntentArbiter] selected media overview → summarize_all: query_chars=%s",
                    len(qn or ""),
                )
                return {"action": "summarize_all", "params": params, "confidence": 0.96}
            logger.info(
                "[ExpertIntentArbiter] media expert deferred generic media overview to skill dispatch: query_chars=%s",
                len(qn or ""),
            )
            return None
        if ctx.last_results and self._results_are_predominantly_media(ctx.last_results):
            if IntentAnalyzer.looks_like_content_followup_on_prior_results(qn):
                logger.info(
                    "[ExpertIntentArbiter] media expert deferred to media followup on prior results: "
                    f"query_chars={len(qn or '')}"
                )
                return None

        result = MediaQueryExpert.analyze(
            ctx.question,
            last_results=ctx.last_results,
            llm_service=ctx.llm_service,
        )
        if result is None:
            has_time_signal = bool(MediaQueryExpert._HAS_TIME_SIGNAL.search(MediaQueryExpert._normalize_time_query(qn).lower()))
            if has_time_signal and MediaQueryExpert._looks_like_calendar_year_reference(
                MediaQueryExpert._normalize_time_query(qn).lower()
            ):
                has_time_signal = False
            if active_media_paths and has_time_signal and (focused_media_path or len(active_media_paths) == 1):
                selected_path = focused_media_path or active_media_paths[0]
                selected_name = os.path.basename(selected_path)
                selected_ext = os.path.splitext(selected_name)[1].lower()
                selected_kind = "video" if selected_ext in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv"} else "audio"
                augmented_query = qn
                ql = qn.lower()
                if selected_name.lower() not in ql:
                    augmented_query = f"{qn} in {selected_name}"
                elif selected_kind not in ql:
                    augmented_query = f"{qn} in the selected {selected_kind}"
                logger.info(
                    "[ExpertIntentArbiter] retrying media expert with selected media scope: %r",
                    augmented_query,
                )
                result = MediaQueryExpert.analyze(
                    augmented_query,
                    last_results=ctx.last_results,
                    llm_service=ctx.llm_service,
                )
                if result is not None:
                    params = result.setdefault("params", {})
                    params.setdefault("file_hint", selected_name)
                    params["_selection_media_scope"] = True
        if result is not None:
            params = result.setdefault("params", {})
            if active_media_paths and str(result.get("action") or "") in {"media_export", "media_content_search"}:
                selected_exts = {os.path.splitext(path)[1].lower() for path in active_media_paths}
                video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv"}
                audio_exts = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape"}
                if len(active_media_paths) == 1:
                    params.setdefault("file_hint", os.path.basename(active_media_paths[0]))
                if selected_exts and selected_exts <= video_exts:
                    params.setdefault("media_type", "video")
                elif selected_exts and selected_exts <= audio_exts:
                    params.setdefault("media_type", "audio")
                params["_scope"] = "selected"
                params["_preserve_selected_scope"] = True
                params["_selection_media_scope"] = True
            params["_expert_route"] = "media"
            logger.info(f"[ExpertIntentArbiter] media expert matched: {result}")
        return result

    def _selected_media_priority_intent(self, ctx: IntentContext) -> Optional[Dict[str, Any]]:
        from core.intent.media_query_expert import MediaQueryExpert
        from core.skills import MediaFollowupSkill

        active_media_paths = self._active_media_scope(ctx.active_paths)
        if not active_media_paths:
            return None

        qn = str(ctx.question or "").strip()
        if not qn or _NON_ACTION_UTTERANCE_RE.search(qn):
            return None
        if MediaQueryExpert.looks_like_explicit_media_file_search(qn):
            return None
        focused_media_path = ""
        opened_file_path = str(getattr(ctx, "opened_file_path", "") or "").strip()
        if opened_file_path and self._active_media_scope([opened_file_path]):
            focused_media_path = opened_file_path

        normalized = MediaQueryExpert._normalize_time_query(qn).lower()
        has_time_signal = bool(MediaQueryExpert._HAS_TIME_SIGNAL.search(normalized))
        if has_time_signal and MediaQueryExpert._looks_like_calendar_year_reference(normalized):
            has_time_signal = False
        if not focused_media_path and len(active_media_paths) == 1:
            focused_media_path = active_media_paths[0]
        if not focused_media_path:
            prior_media = [
                str(item.get("file_path") or item.get("file_name") or "").strip()
                for item in list(getattr(ctx, "last_results", None) or [])
                if self._results_are_predominantly_media([item])
            ]
            prior_basenames = {
                os.path.basename(path).strip().lower()
                for path in prior_media
                if os.path.basename(path).strip()
            }
            if prior_media and (
                len(prior_media) == 1
                or len(prior_basenames) == 1
            ):
                focused_media_path = prior_media[0]
        focused_media_name = os.path.basename(focused_media_path) if focused_media_path else ""
        active_scope_paths = [str(path or "").strip() for path in list(ctx.active_paths or []) if str(path or "").strip()]
        single_selected_media_scope = len(active_scope_paths) == 1 and len(active_media_paths) == 1
        mixed_selected_scope = bool(active_media_paths) and len(active_media_paths) < len(active_scope_paths)
        media_specific_query = self._query_is_media_specific(qn)
        frame_position = self._selected_video_frame_position(qn)
        if frame_position:
            selected_ext = os.path.splitext(active_media_paths[0])[1].lower()
            if selected_ext in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv"}:
                params: Dict[str, Any] = {
                    "query": qn,
                    "target_type": "video_visual",
                    "sub_intent": "point_lookup",
                    "frame_position": frame_position,
                    "media_type": "video",
                    "_scope": "selected",
                    "_preserve_selected_scope": True,
                    "_selection_media_scope": True,
                    "_expert_route": "media",
                }
                if frame_position == "first":
                    params["time_sec"] = 0.0
                if focused_media_name:
                    params["file_hint"] = focused_media_name
                intent = {"action": "media_export", "params": params, "confidence": 0.97}
                logger.info(f"[ExpertIntentArbiter] selected video frame query matched: {intent}")
                return intent
        if mixed_selected_scope and not media_specific_query:
            logger.info(
                "[ExpertIntentArbiter] selected media priority deferred mixed non-media-specific scope: %r",
                qn,
            )
            return None
        if active_media_paths and MediaFollowupSkill.looks_like_generic_overview_query(
            qn,
            file_hint=focused_media_name,
        ):
            selected_ext = os.path.splitext(active_media_paths[0])[1].lower()
            video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv"}
            audio_exts = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape"}
            if selected_ext in video_exts or selected_ext in audio_exts:
                is_video = selected_ext in video_exts
                params = {
                    "query": qn,
                    "target_type": "video_visual" if is_video else "audio_content",
                    "sub_intent": "range_summary",
                    "time_sec": 0.0,
                    "media_type": "video" if is_video else "audio",
                    "_scope": "selected",
                    "_preserve_selected_scope": True,
                    "_selection_media_scope": True,
                    "_expert_route": "media",
                }
                if focused_media_name:
                    params["file_hint"] = focused_media_name
                intent = {"action": "summarize_all", "params": params, "confidence": 0.97}
                logger.info(f"[ExpertIntentArbiter] selected media overview matched: {intent}")
                return intent
        should_prioritize = (
            has_time_signal
            or bool(MediaQueryExpert._HAS_MEDIA_SEARCH_SIGNAL.search(normalized))
            or bool(MediaQueryExpert._HAS_MEDIA_CONTENT_RE.search(qn))
            or MediaFollowupSkill.looks_like_generic_overview_query(qn, file_hint=focused_media_name)
        )
        if not should_prioritize:
            return None

        intent = self._media_intent(ctx)
        if intent is not None:
            intent.setdefault("params", {})["_expert_route"] = "media"
            logger.info(f"[ExpertIntentArbiter] selected media priority matched: {intent}")
        return intent

    def _semantic_intent(self, ctx: IntentContext) -> Optional[Dict[str, Any]]:
        """Route to SkillDispatcher (default) or legacy Router+Agent pipeline."""
        use_skill_dispatch = os.environ.get(
            "FILEAGENT_USE_SKILL_DISPATCH", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}

        if use_skill_dispatch:
            return self._skill_dispatch_intent(ctx)
        else:
            return self._legacy_semantic_intent(ctx)

    def _skill_dispatch_intent(self, ctx: IntentContext) -> Optional[Dict[str, Any]]:
        """Single LLM call via SkillDispatcher — replaces Router+Agent double call."""
        from core.intent.skill_dispatcher import SkillDispatcher
        from core.intent.media_query_expert import MediaQueryExpert
        from core.skills import ContextualRefineSkill, MediaFollowupSkill

        result = SkillDispatcher.dispatch(ctx)
        action = str(result.get("action") or "").strip()

        if not action:
            return None

        params = dict(result.get("params") or {})
        params.setdefault("_skill_name", action)
        result["params"] = params

        if action == "search" and MediaQueryExpert.looks_like_explicit_media_file_search(ctx.question):
            file_hint = MediaQueryExpert._extract_file_hint(ctx.question)
            if file_hint:
                file_hint = re.sub(
                    r"^(?:audio|video|recording|clip|media)\s+file\s+|^file\s+",
                    "",
                    file_hint,
                    flags=re.IGNORECASE,
                ).strip()
                params["query"] = file_hint
                params.setdefault("category", _category_for_media_type(str(params.get("media_type") or "")))
        elif action == "media_content_search" and self._skill_dispatch_image_content_misroute(ctx, params):
            repaired_params = dict(params)
            repaired_params.setdefault("operation", "qa")
            repaired_params.setdefault("_context_operation", "qa")
            repaired_params.setdefault(
                "_dispatch_reason",
                "Still-image understanding is selected-file reasoning, not audio/video content search.",
            )
            if ctx.active_paths:
                repaired_params["_scope"] = "selected"
                repaired_params["_preserve_selected_scope"] = True
                action = "summarize_all"
            else:
                repaired_params.setdefault("scope", "last_results")
                action = "process_previous"
            result = {"action": action, "params": repaired_params}
            params = repaired_params
        elif action == "media_content_search" and MediaQueryExpert.looks_like_explicit_media_file_search(ctx.question):
            action = "search"
            repaired_params = dict(params)
            repaired_params.setdefault("query", str(params.get("query") or ctx.question or "").strip())
            if "media_type" in params:
                repaired_params.setdefault("media_type", params.get("media_type"))
            repaired_params.setdefault("category", _category_for_media_type(str(repaired_params.get("media_type") or "")))
            repaired_params.setdefault(
                "_dispatch_reason",
                "Explicit media file inventory wording should stay on indexed file search, not media content search.",
            )
            result = {"action": action, "params": repaired_params}
            params = repaired_params

        if action == "media_content_search" and MediaFollowupSkill.supports_ctx(ctx):
            media_state = MediaFollowupSkill.build_state(ctx)
            if MediaFollowupSkill.looks_like_generic_overview_query(
                str(params.get("query") or ctx.question or ""),
                file_hint=str(params.get("file_hint") or media_state.focused_file or ""),
            ):
                action = "media_followup"
                params["operation"] = "summary"
                result = {"action": action, "params": params}
        elif (
            action == "media_export"
            and MediaQueryExpert.looks_like_explicit_media_file_search(ctx.question)
            and not MediaQueryExpert.looks_like_media_operation_request(ctx.question)
        ):
            action = "search"
            repaired_params = {"query": ctx.question, **params}
            repaired_params.setdefault("category", _category_for_media_type(str(repaired_params.get("media_type") or "")))
            result = {"action": action, "params": repaired_params}

        if action == "list_selected":
            params.setdefault(
                "scope",
                "selected_folder" if any(os.path.isdir(str(p)) for p in (ctx.active_paths or [])) else "selected_items",
            )
            params.setdefault("operation", "list")
            action = "contextual_refine"
        elif action == "summarize_selected":
            params.setdefault(
                "scope",
                "selected_folder" if any(os.path.isdir(str(p)) for p in (ctx.active_paths or [])) else "selected_items",
            )
            params.setdefault("operation", "summary")
            action = "contextual_refine"
        elif action == "process_previous":
            params.setdefault("scope", "last_results")
            params.setdefault("operation", "summary")
            action = "contextual_refine"
        elif action == "media_timequery":
            params.setdefault("operation", "range_summary" if params.get("time_end_sec") is not None else "time_lookup")
            action = "media_followup"

        if action == "contextual_refine":
            plan = ContextualRefineSkill.plan_from_params(
                ctx.question,
                params,
                active_paths=ctx.active_paths,
                last_results=ctx.last_results,
            )
            params["scope"] = plan.scope
            params["_scope_kind"] = plan.scope
            params["operation"] = plan.operation
            params["_context_operation"] = plan.operation
            if plan.focus_extension:
                params["focus_extension"] = plan.focus_extension
                params.setdefault("file_extensions", plan.focus_extension)
            if plan.rewrite_mode:
                params["rewrite_mode"] = plan.rewrite_mode
            if plan.file_hint:
                params.setdefault("file_hint", plan.file_hint)
            if plan.reason:
                params.setdefault("_dispatch_reason", plan.reason)

            if plan.operation == "list" and plan.scope in {"selected_items", "selected_folder"}:
                result = {
                    "action": "count",
                    "params": {
                        **params,
                        "category": "all",
                        "_scope": "selected",
                        "_selection_mode": "selected_items",
                    },
                }
            elif plan.scope in {"selected_items", "selected_folder"} and not ctx.last_results:
                summary_params = dict(params)
                # First-turn selected-scope reasoning should preserve the
                # selected corpus and let the summarizer answer within it,
                # rather than collapsing the visible scope to a filtered subset.
                summary_params.pop("focus_extension", None)
                summary_params.pop("focus_extensions", None)
                summary_params.pop("file_extensions", None)
                if len(ctx.active_paths or []) > 1 and not self._query_is_media_specific(ctx.question):
                    summary_params.pop("file_hint", None)
                    summary_params.pop("focused_file", None)
                result = {
                    "action": "summarize_all",
                    "params": {
                        **summary_params,
                        "_scope": "selected",
                        "_preserve_selected_scope": True,
                    },
                }
            else:
                result = {"action": "process_previous", "params": params}
            action = str(result.get("action") or "").strip()

        if action == "media_followup":
            plan = MediaFollowupSkill.plan_from_params(
                ctx.question,
                params,
                last_results=ctx.last_results,
                active_paths=ctx.active_paths,
            )
            active_media_scope = self._active_media_scope(ctx.active_paths)
            params["operation"] = plan.operation
            params["media_type"] = plan.media_type
            if plan.file_hint:
                params["file_hint"] = plan.file_hint
            if plan.target_type:
                params["target_type"] = plan.target_type
            if plan.time_sec is not None:
                params["time_sec"] = plan.time_sec
            if plan.time_end_sec is not None:
                params["time_end_sec"] = plan.time_end_sec
                params.setdefault("sub_intent", "range_summary")
            if plan.reason:
                params.setdefault("_dispatch_reason", plan.reason)

            if plan.operation in {"time_lookup", "range_summary"}:
                result = {"action": "media_export", "params": params}
            elif plan.operation == "topic_search":
                result = {"action": "media_content_search", "params": params}
            elif plan.operation == "summary" and active_media_scope:
                summary_params = dict(params)
                result = {
                    "action": "summarize_all",
                    "params": {
                        **summary_params,
                        "_scope": "selected",
                        "_preserve_selected_scope": True,
                    },
                }
            elif ctx.last_results:
                result = {"action": "process_previous", "params": params}
            else:
                result = {
                    "action": "summarize",
                    "params": {
                        **params,
                        "category": _category_for_media_type(str(params.get("media_type") or "")),
                    },
                }
            action = str(result.get("action") or "").strip()

        # Normalize media sub-actions the same way the legacy path does
        if action == "media_count":
            params = dict(result.get("params") or {})
            result = {"action": "count", "params": {"category": _category_for_media_type(str(params.get("media_type") or "")), **params}}
        elif action == "media_summarize":
            params = dict(result.get("params") or {})
            result = {"action": "summarize", "params": {"category": _category_for_media_type(str(params.get("media_type") or "")), **params}}
        elif action == "media_timequery":
            # Map to media_content_search which QueryOrchestrator understands
            params = dict(result.get("params") or {})
            result = {"action": "media_content_search", "params": params}

        result = self._repair_skill_dispatch_category_listing(ctx, result)
        result = self._repair_skill_dispatch_topic_inventory(ctx, result)
        result = self._repair_skill_dispatch_missing_category_filter(ctx, result)
        result = self._repair_skill_dispatch_filetype_clarification(ctx, result)
        result = self._repair_skill_dispatch_explicit_find(ctx, result)
        result = self._repair_skill_dispatch_contextual_refine(ctx, result)
        result = self._repair_skill_dispatch_scoped_comparison(ctx, result)
        result = self._repair_skill_dispatch_scoped_content_followup(ctx, result)
        result.setdefault("params", {})["_expert_route"] = "skill_dispatch"
        result["confidence"] = max(float(result.get("confidence", 0.0)), 0.82)

        logger.info(f"[ExpertIntentArbiter] skill_dispatch → {result.get('action')}, params={result.get('params')}")
        return result

    def _legacy_semantic_intent(self, ctx: IntentContext) -> Optional[Dict[str, Any]]:
        """Legacy Router → ContinuationAgent/FileOpAgent/MediaSubAgent double LLM call.
        Kept as rollback via FILEAGENT_USE_SKILL_DISPATCH=0."""
        from core.intent.continuation_agent import ContinuationAgent
        from core.intent.file_op_agent import FileOpAgent
        from core.intent.media_sub_agent import MediaSubAgent
        from core.intent.router import ConversationRouter

        route = ConversationRouter.route(ctx)
        logger.info(f"[ExpertIntentArbiter] semantic router selected: {route}")

        if route == "continuation":
            result = ContinuationAgent.analyze(ctx)
            action = str(result.get("action") or "").strip()
            if action == "fallback_to_file_op":
                route = "file_op"
            else:
                result.setdefault("params", {})["_expert_route"] = "semantic_continuation"
                result["confidence"] = max(float(result.get("confidence") or 0.0), 0.82)
                return result

        if route == "file_op":
            ql = str(ctx.question or "").strip().lower()
            has_prior_context = bool(ctx.last_results or ctx.history)
            explicit_new_request = bool(
                re.match(
                    r'^(find|search|show\s+me|look\s+for|retrieve|locate|get\s+me|list|count|'
                    r'what\s+files\b|what\s+documents?\b|what\s+docs\b|what\s+items\b|what\s+are\s+my\b|'
                    r'how\s+many|which\s+files?\b|which\s+documents?\b|do\s+i\s+have)\b'
                    r'|^(找|搜|查找|搜索|显示|列出|查一下|搜一下|帮我找|(我)?有哪些|(我)?有(什么|哪些)|(我)?(一共|总计)?有多少)',
                    ql,
                    re.IGNORECASE,
                )
            )
            if has_prior_context and not explicit_new_request:
                continuation_retry = ContinuationAgent.analyze(ctx)
                continuation_action = str(continuation_retry.get("action") or "").strip()
                if continuation_action not in {"fallback_to_file_op", "chat"}:
                    continuation_retry.setdefault("params", {})["_expert_route"] = "semantic_continuation_rescue"
                    continuation_retry["confidence"] = max(float(continuation_retry.get("confidence") or 0.0), 0.81)
                    logger.info(f"[ExpertIntentArbiter] continuation rescue accepted: {continuation_retry}")
                    return continuation_retry
            result = FileOpAgent.analyze(ctx)
            result.setdefault("params", {})["_expert_route"] = "semantic_file_op"
            result["confidence"] = max(float(result.get("confidence") or 0.0), 0.8)
            return result

        if route == "media":
            result = self._media_intent(ctx)
            if result is None:
                result = MediaSubAgent.analyze(ctx)
                action = str(result.get("action") or "").strip()
                params = dict(result.get("params") or {})
                if action == "media_count":
                    result = {"action": "count", "params": {"category": _category_for_media_type(str(params.get("media_type") or "")), **params}}
                elif action == "media_summarize":
                    result = {"action": "summarize", "params": {"category": _category_for_media_type(str(params.get("media_type") or "")), **params}}
            result.setdefault("params", {})["_expert_route"] = "semantic_media"
            result["confidence"] = max(float(result.get("confidence") or 0.0), 0.84)
            return result

        if route == "chat":
            return {
                "action": "chat",
                "params": {
                    "_expert_route": "semantic_chat",
                },
                "confidence": 0.8,
            }

        return None

    def analyze(self, query_context: Any) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Returns:
          (intent_dict, source_name) if any high-confidence expert matched
          (None, "legacy") otherwise
        """
        ctx = self._build_intent_context(query_context)

        folder_intent = self._folder_listing_intent(query_context)
        extracted_folder = ""
        folder_category = ""
        folder_has_specific_category = False
        if folder_intent and isinstance(folder_intent.get("params"), dict):
            extracted_folder = folder_intent["params"].get("folder", "")
            folder_category = str(folder_intent["params"].get("category") or "").strip()
            folder_has_specific_category = bool(folder_category)

        for source, fn in (
            ("non_action", self._non_action_intent),
            ("selected_media_priority", self._selected_media_priority_intent),
            ("selection", self._selection_intent),
            ("category_list", self._category_list_intent),
            ("explicit_filename", lambda _ctx: self._explicit_filename_intent(query_context)),
            # Let the canonical follow-up expert see prior-result references
            # before broad attribute words like "company" force a global search.
            ("context_followup", self._context_followup_intent),
            ("personal_attribute", self._personal_attribute_intent),
            ("media", self._media_intent),
            ("entity", lambda _ctx: self._entity_search_intent(query_context)),
            ("folder_listing", lambda _ctx: folder_intent),
        ):
            intent = fn(ctx)
            if intent is not None:
                if source in {"category_list", "media"} and extracted_folder and not folder_has_specific_category:
                    logger.info(
                        "[ExpertIntentArbiter] %s superseded by explicit folder listing (folder=%r)",
                        source,
                        extracted_folder,
                    )
                    intent = folder_intent
                    source = "folder_listing"
                if (
                    extracted_folder
                    and str((intent or {}).get("action") or "").strip() == "search"
                    and source != "folder_listing"
                ):
                    if "params" not in intent:
                        intent["params"] = {}
                    params = intent["params"]
                    if not params.get("folder"):
                        params["folder"] = extracted_folder
                        params["_expert_merged_folder"] = True
                    if folder_category:
                        current_category = str(params.get("category") or "").strip()
                        if current_category != folder_category:
                            if current_category:
                                params["_category_before_folder_kind_repair"] = current_category
                            params["category"] = folder_category
                            params["_dispatch_repair"] = "folder_listing_kind_category_precedence"

                if source == "entity":
                    route = str((intent.get("params") or {}).get("_expert_route") or "entity")
                    return intent, route
                return intent, source

        semantic_intent = self._semantic_intent(ctx)
        if semantic_intent is not None:
            if extracted_folder:
                if "params" not in semantic_intent:
                    semantic_intent["params"] = {}
                if not semantic_intent["params"].get("folder"):
                    semantic_intent["params"]["folder"] = extracted_folder
                    semantic_intent["params"]["_expert_merged_folder"] = True
            return semantic_intent, str((semantic_intent.get("params") or {}).get("_expert_route") or "semantic")

        return None, "legacy"
