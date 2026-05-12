"""
SelectionExpert MoE Agent — lightweight binary classifier for "selected files" queries.

Replaces the brittle 80-line regex stack in intent_analyzer.py fp0 with a semantically
aware, deterministic classifier that can optionally fall back to a tiny LLM call.

Design philosophy:
  - Primary: rule-based heuristic (covers 95% of cases, zero latency)
  - Optional: lightweight LLM call for ambiguous cases (~80 token prompt)
  
Why not pure LLM? Because "selected" queries are high-frequency and we want sub-50ms.
Why not pure regex? Because the old fp0 had 80 lines of regex and still missed edge cases
like "files about selected topics" vs "the selected files".
"""
from __future__ import annotations

import os
import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

from core.retrieval.filename_canonicalizer import classify_reference_target


class SelectionResult:
    """Result from the SelectionExpert."""
    __slots__ = ("is_selection", "intent", "confidence", "reason", "params")

    def __init__(
        self,
        is_selection: bool,
        intent: str = "",
        confidence: float = 0.0,
        reason: str = "",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.is_selection = is_selection
        self.intent = intent  # "summarize_all" | "summarize_selected" | "count_selected" | "list_selected" | "" (not selection)
        self.confidence = confidence
        self.reason = reason
        self.params = dict(params or {})

    def __repr__(self):
        return f"SelectionResult(is_selection={self.is_selection}, intent={self.intent!r}, conf={self.confidence:.2f})"

    def to_intent(self) -> Optional[dict]:
        """Convert to intent dict, or None if not a selection query."""
        if not self.is_selection:
            return None
        base_params = dict(self.params or {})
        if self.intent == "clarify":
            return {
                "action": "clarify",
                "params": {"question": str(base_params.get("question") or "").strip()},
                "confidence": self.confidence,
            }
        if self.intent == "list_selected":
            params = {
                "category": base_params.get("category", "all"),
                "_scope": "selected",
                "_selection_mode": "selected_items",
            }
            if base_params.get("media_type") in {"audio", "video"}:
                params["media_type"] = base_params["media_type"]
            return {
                "action": "count",
                "params": params,
                "confidence": self.confidence,
            }
        if self.intent == "count_selected":
            params = {
                "category": base_params.get("category", "all"),
                "_scope": "selected",
            }
            if base_params.get("media_type") in {"audio", "video"}:
                params["media_type"] = base_params["media_type"]
            return {"action": "count", "params": params, "confidence": self.confidence}
        if self.intent == "summarize_selected":
            params = {
                "category": base_params.get("category", "all"),
                "_scope": "selected",
                "_preserve_selected_scope": True,
            }
            if base_params.get("media_type") in {"audio", "video"}:
                params["media_type"] = base_params["media_type"]
            if params.get("category") == "audio/video":
                params["_selection_media_scope"] = True
                return {"action": "summarize_all", "params": params, "confidence": self.confidence}
            return {"action": "summarize", "params": params, "confidence": self.confidence}
        if self.intent == "summarize_all":
            params = {
                **base_params,
                "_scope": "selected",
                "_preserve_selected_scope": True,
            }
            return {"action": "summarize_all", "params": params, "confidence": self.confidence}
        return None


class SelectionExpert:
    """
    Lightweight MoE expert for "selected/chosen files" queries.
    
    Activation: only runs when the query contains selection-related keywords
    AND active_paths is non-empty.
    """

    # ── Trigger keywords (cheap string match) ─────────────────────────────
    _EN_TRIGGERS = frozenset({
        "selected", "seleted", "selectd", "slected",  # common typos
        "chosen", "chosn", "picked",
    })
    _ZH_TRIGGERS = frozenset({
        "选中", "已选", "已勾选", "当前选中", "勾选",
    })
    _DEMONSTRATIVE_TRIGGERS = frozenset({
        "these files", "these documents", "these docs",
        "this file", "this document", "this doc",
        "current files", "current documents", "current selection",
        "这些文件", "这些文档", "那些文件", "那些文档",
    })
    # ── Anti-patterns: query is ABOUT the topic "selected", not about user's file selection ──
    _SEARCH_ESCAPE_RE = re.compile(
        r'\b(about|regarding|containing|that\s+mention|that\s+contain|that\s+have'
        r'|with\s+the\s+word|mentioning|related\s+to|involving)\b',
        re.IGNORECASE,
    )
    _EXPLICIT_FILE_REQUEST_RE = re.compile(
        r'^(?:find|search(?:\s+for)?|look\s+for|locate|retrieve|get|show|list|open)\b'
        r'|^(?:找|搜|搜索|查找|检索|列出|查看|打开)',
        re.IGNORECASE,
    )
    _SELECTION_SCOPE_NOUN_RE = re.compile(
        r'\b(files?|documents?|docs?|folders?|items?|sources?|pdfs?|pdf|docx?|xlsx?|xls|csv|pptx?|ppt|'
        r'images?|photos?|pictures?|videos?|audios?|media|movies?|clips?)\b'
        r'|文件|文档|资料|数据源|图片|照片|视频|音频',
        re.IGNORECASE,
    )
    _TOPIC_SELECTION_ESCAPE_RE = re.compile(
        r'\b(about|regarding|related\s+to|containing|mentioning|featuring|on)\s+'
        r'(?:the\s+)?(selected|seleted|selectd|slected|chosen|chosn|picked)\b',
        re.IGNORECASE,
    )

    # ── Patterns that indicate listing vs content reading ─────────────────
    _COUNT_KWS = frozenset({
        "有多少", "几个", "几份", "count", "how many",
    })

    _LIST_KWS = frozenset({
        "有哪些", "哪些文件", "哪些文档", "列出", "清单", "看看", "看下", "查看",
        "what files", "which files", "list", "show", "display", "browse", "what are",
    })
    _INVENTORY_PREFIX_RE = re.compile(
        r'^\s*(?:what\s+are|which\s+files|what\s+files|show(?:\s+me)?|list|display|browse|tell\s+me)\b'
        r'|有哪些|列出|看看|看下|查看|告诉我',
        re.IGNORECASE,
    )

    _CONTENT_INTENT_RE = re.compile(
        r'\b(detail|details|content|contents|about|describe|explanation|explain'
        r'|contains?|has\s+text|have\s+text|includes?|has\s+(any|a)'  # conditional phrasing
        r'|what\s+(is|does|in|inside|can\s+be\s+heard|is\s+heard|can\s+i\s+hear|is\s+said)'
        r'|tell\s+(?:me\s+)?(?:more\s+)?about|inside|in\s+the\s+(file|doc|document)'
        r'|summarize|summary|recap|overview|analyze|analysis|heard|hear|said|spoken'
        r'|most\s+(detailed|comprehensive|important|relevant|complete)'  # comparative/ranking
        r'|ranking|ranked|rank|compare|comparison|best|worst'
        r'|which\s+.{0,20}\s+(is|are)\s+(the\s+)?(most|best|worst|least))\b',
        re.IGNORECASE,
    )

    _VIDEO_SCOPE_RE = re.compile(
        r'\b(video|videos|movie|movies|clip|clips|film|films|footage'
        r'|mp4|mov|mkv|avi|webm|m4v|wmv|flv|ts)\b'
        r'|视频|影片|录像|录屏|短片',
        re.IGNORECASE,
    )
    _AUDIO_SCOPE_RE = re.compile(
        r'\b(audio|audios|podcast|podcasts|song|songs|music'
        r'|mp3|wav|m4a|aac|flac|ogg|wma|aiff|ape)\b'
        r'|音频|录音|播客|歌曲|音乐',
        re.IGNORECASE,
    )
    _MEDIA_SCOPE_RE = re.compile(
        r'\b(media)\b|音视频|影音',
        re.IGNORECASE,
    )

    _SELECTED_EXTENSION_RE = re.compile(
        r'\b(selected|seleted|selectd|slected|chosen|chosn|picked)\b.{0,15}\b('
        r'pdf|docx?|xlsx?|xls|csv|pptx?|ppt|wav|mp3|mp4|m4a|mov|jpg|jpeg|png|gif|'
        r'image|images|photo|photos|video|videos|movie|movies|clip|clips|audio|audios|media)\b',
        re.IGNORECASE,
    )

    # ── Previous-result escape ────────────────────────────────────────────
    _PREV_ESCAPE_KWS = frozenset({
        "above", "previous", "previously", "last", "earlier",
        "刚才", "上次", "前面", "上述", "以上", "上面", "上文",
    })

    @classmethod
    def should_activate(cls, query: str, active_paths: Optional[List[str]]) -> bool:
        """Quick check: should we even run the selection expert?"""
        if not active_paths:
            return False
        ql = (query or "").lower()
        if classify_reference_target(query).get("kind") == "deictic":
            return True
        # Check English trigger words
        for trigger in cls._EN_TRIGGERS:
            if trigger in ql:
                return True
        # Check Chinese triggers
        for trigger in cls._ZH_TRIGGERS:
            if trigger in ql:
                return True
        # Check demonstrative patterns
        for pattern in cls._DEMONSTRATIVE_TRIGGERS:
            if pattern in ql:
                return True
        return False

    @classmethod
    def infer_scope_filters(cls, query: str) -> Dict[str, Any]:
        ql = (query or "").lower()
        has_video = bool(cls._VIDEO_SCOPE_RE.search(ql))
        has_audio = bool(cls._AUDIO_SCOPE_RE.search(ql))
        has_media = bool(cls._MEDIA_SCOPE_RE.search(ql))

        if has_video and not has_audio:
            return {"category": "video", "media_type": "video"}
        if has_audio and not has_video:
            return {"category": "audio", "media_type": "audio"}
        if has_media or has_video or has_audio:
            return {"category": "audio/video"}
        return {}

    @classmethod
    def _looks_like_topic_search_not_selection(cls, query: str) -> bool:
        ql = (query or "").lower()
        if not ql or not cls._EXPLICIT_FILE_REQUEST_RE.search(ql):
            return False
        match = cls._TOPIC_SELECTION_ESCAPE_RE.search(ql)
        if not match:
            return False
        tail = ql[match.end(): match.end() + 32]
        if cls._SELECTION_SCOPE_NOUN_RE.search(tail):
            return False
        return True

    @classmethod
    def _looks_like_selection_inventory_query(cls, query: str) -> bool:
        ql = (query or "").lower()
        if not ql:
            return False
        if cls._CONTENT_INTENT_RE.search(ql):
            return False
        if cls._SELECTED_EXTENSION_RE.search(ql):
            return True
        if any(kw in ql for kw in cls._COUNT_KWS):
            return True
        if any(kw in ql for kw in cls._LIST_KWS):
            return True
        return bool(cls._INVENTORY_PREFIX_RE.search(ql))

    @classmethod
    def _classify_deictic_selection(
        cls,
        query: str,
        *,
        has_last_results: bool,
        active_count: int,
        lang: str = "en",
        prior_action: str = "",
    ) -> Optional[SelectionResult]:
        q = str(query or "").strip()
        if not q:
            return None
        ref_target = classify_reference_target(query)
        if ref_target.get("kind") != "deictic":
            return None
        scope_params = cls.infer_scope_filters(query)

        # Fix C: When prior action was a summarize or process_previous, deictic
        # pronouns like "it" / "them" refer to the whole prior result set —
        # NOT to an ambiguous individual selected file. Skip singular clarify
        # entirely and let the continuation pipeline handle it as process_previous.
        _SUMMARIZE_PRIORS = {
            "summarize_all", "summarize", "summarize_selected", "process_previous",
        }
        if prior_action in _SUMMARIZE_PRIORS:
            logger.info(
                f"[selection_expert] skip deictic clarify: prior_action={prior_action!r} "
                "(deictic 'it' refers to whole prior result, not individual file)"
            )
            return None

        if has_last_results:
            logger.info("[selection_expert] deictic reference deferred to previous results context")
            return SelectionResult(False, reason="deictic_prefers_previous_results")

        if ref_target.get("number") == "singular" and active_count != 1:
            msg = (
                "\u6211\u8fd8\u6ca1\u6cd5\u786e\u5b9a\u4f60\u6307\u7684\u662f\u54ea\u4e00\u4efd\u6587\u4ef6\u3002\u8bf7\u76f4\u63a5\u8bf4\u51fa\u6587\u4ef6\u540d\uff0c\u6211\u5c31\u7ee7\u7eed\u5e2e\u4f60\u770b\u3002"
                if str(lang).lower().startswith("zh")
                else "I can't tell which file you mean yet. Please name the file and I'll continue."
            )
            logger.info("[selection_expert] singular deictic without unique selected file \u2192 clarify")
            return SelectionResult(True, "clarify", 0.98, "deictic_selected_ambiguous", params={"question": msg})

        if ref_target.get("number") == "singular" and active_count == 1:
            if any(kw in q.lower() for kw in cls._COUNT_KWS):
                logger.info("[selection_expert] explicit count on unique selected file \u2192 count_selected")
                return SelectionResult(True, "count_selected", 0.95, "deictic_unique_selected_count", params=scope_params)
            intent = "summarize_selected" if scope_params.get("category") in {"audio", "video", "audio/video"} else "summarize_all"
            logger.info(f"[selection_expert] unique selected deictic \u2192 {intent}")
            return SelectionResult(True, intent, 0.94, "deictic_unique_selected_file", params=scope_params)

        if cls._CONTENT_INTENT_RE.search(q.lower()):
            intent = "summarize_selected" if scope_params.get("category") in {"audio", "video", "audio/video"} else "summarize_all"
            logger.info(f"[selection_expert] deictic content without prior results \u2192 {intent}")
            return SelectionResult(True, intent, 0.96, "deictic_selected_content", params=scope_params)

        logger.info("[selection_expert] deictic listing without prior results \u2192 list_selected")
        return SelectionResult(True, "list_selected", 0.97, "deictic_selected_listing", params=scope_params)

    @classmethod
    def classify(
        cls,
        query: str,
        active_paths: List[str],
        last_results: Optional[List] = None,
        llm_service: Any = None,
        lang: str = "en",
        prior_action: str = "",
    ) -> SelectionResult:
        """
        Classify whether the query is about the user's file selection using an LLM.
        Falls back to regex if LLM is unavailable or fails.

        Args:
            prior_action: The intent action from the previous turn (used to skip
                          deictic singular clarify when already in summarize context).
        """
        ql = (query or "").lower()
        if cls._looks_like_topic_search_not_selection(query):
            logger.info("[selection_expert] explicit file search uses selection term as topic, not scope.")
            return SelectionResult(False, reason="selection_term_used_as_topic")
        active_identity_keys = {
            os.path.basename(str(path or "")).strip().lower()
            for path in list(active_paths or [])
            if os.path.basename(str(path or "")).strip()
        }
        effective_active_count = len(active_identity_keys) if active_identity_keys else len(active_paths or [])
        ref_kind = str(classify_reference_target(query).get("kind") or "").strip().lower()
        if ref_kind != "deictic":
            regex_prefilter = cls._legacy_regex_classify(
                query,
                active_paths,
                last_results,
                lang=lang,
                prior_action=prior_action,
            )
            if regex_prefilter.is_selection and regex_prefilter.reason in {
                "selected_extension_inventory",
                "selection_inventory_phrase",
                "count_intent_on_selection",
                "list_intent_on_selection",
                "default_selection_intent",
            }:
                logger.info(
                    "[selection_expert] high-precision regex fast-path → %s",
                    regex_prefilter.intent,
                )
                return regex_prefilter
        deictic_result = cls._classify_deictic_selection(
            query,
            has_last_results=bool(last_results),
            active_count=effective_active_count,
            lang=lang,
            prior_action=prior_action,
        )
        if deictic_result is not None:
            return deictic_result
        if not llm_service:
            logger.info("[selection_expert] No llm_service provided, falling back to regex.")
            return cls._legacy_regex_classify(query, active_paths, last_results, lang=lang, prior_action=prior_action)

        import json

        # We inject up to 15 basenames so the MoE has concrete context of what's selected
        selected_names = [os.path.basename(p) for p in active_paths[:15]]
        
        system = """You are a precise intent routing classifier.
The user currently has {n} files actively selected in their workspace:
{file_list}

[Task]
Determine if the user's query refers to THEIR ACTIVELY SELECTED FILES (as the target scope) or if they are doing a generic CONTENT SEARCH.

[Examples]
- "show me my selected files" -> is_selection: true
- "find files about the selected candidates" -> is_selection: false (keyword search)
- "which of them are PDFs" -> is_selection: true
- "怎么过滤这些选中的内容" -> is_selection: true
- "其中有多少张照片" -> is_selection: true
- "tell me about the selected movies" -> is_selection: true
- "show me the selected movies" -> is_selection: true
- "summarize the selected audio files" -> is_selection: true
- "Which selected file is most detailed?" -> is_selection: true
- "what connections can you infer between these selected files?" -> is_selection: true
- "compare these selected files" -> is_selection: true
- "what do these selected files have in common?" -> is_selection: true

If is_selection is true, determine the operation:
- "count": user asks to "count", "subset", "filter", "show which ones", "数量", "几个", "过滤", "筛选", "哪些" among the selection.
- "summarize": user asks for summary, content details, explanation, comparison, ranking, relationship, synthesis, "讲什么", "总结", "内容" of the selection. Questions like "which selected file is most detailed", "compare them", or "what connects these selected files" are summarize, not count.

[Query] {query}
[Output]
Return ONLY a valid JSON object. No markdown formatting.
{{"is_selection": true or false, "operation": "count" or "summarize"}}"""

        prompt = system.format(
            n=len(active_paths),
            file_list="\n".join([f"- {name}" for name in selected_names]),
            query=query
        )

        try:
            # We want deterministic, logical output
            resp = llm_service.generate(prompt, temperature=0.1)
            resp = resp.strip()
            
            # Extract JSON
            start = resp.find("{")
            end = resp.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(resp[start:end+1])
                is_sel = data.get("is_selection", False)
                scope_params = cls.infer_scope_filters(query)
                
                if not is_sel:
                    logger.info("[selection_expert] LLM determined query is NOT about active selection.")
                    return SelectionResult(False, reason="llm_not_selection")
                    
                # Trust LLM's operation field; fall back to regex ONLY when LLM didn't answer.
                op = data.get("operation") or ""
                if not op:
                    if cls._CONTENT_INTENT_RE.search(ql):
                        op = "summarize"
                    elif any(kw in ql for kw in cls._COUNT_KWS) or any(kw in ql for kw in cls._LIST_KWS):
                        op = "count"
                    else:
                        op = "summarize"  # safe default
                elif op == "count" and cls._CONTENT_INTENT_RE.search(ql) and not any(kw in ql for kw in cls._COUNT_KWS):
                    # The lightweight model occasionally treats selected CSV/schema
                    # analysis as inventory. Explicit content verbs should win.
                    op = "summarize"
                if op == "count":
                    intent = "count_selected"
                elif scope_params.get("category") in {"audio", "video", "audio/video"}:
                    intent = "summarize_selected"
                else:
                    intent = "summarize_all"
                logger.info(f"[selection_expert] LLM routed to {intent} (operation={op})")
                return SelectionResult(True, intent, 0.95, "llm_routed", params=scope_params)
                
            logger.warning("[selection_expert] JSON parsing failed from LLM output, falling back to regex.")
        except Exception as e:
            logger.error(f"[selection_expert] LLM Error: {e}, falling back to regex.")

        return cls._legacy_regex_classify(query, active_paths, last_results, prior_action=prior_action)

    @classmethod
    def _legacy_regex_classify(
        cls,
        query: str,
        active_paths: List[str],
        last_results: Optional[List] = None,
        *,
        lang: str = "en",
        prior_action: str = "",
    ) -> SelectionResult:
        ql = (query or "").lower()
        deictic_result = cls._classify_deictic_selection(
            query,
            has_last_results=bool(last_results),
            active_count=len(active_paths or []),
            lang=lang,
            prior_action=prior_action,
        )
        if deictic_result is not None:
            return deictic_result
        scope_params = cls.infer_scope_filters(query)

        # ── Step 1: Check for previous-result references ──────────────────
        if last_results:
            for kw in cls._PREV_ESCAPE_KWS:
                if kw.isascii():
                    if re.search(r'\b' + re.escape(kw) + r'\b', ql):
                        logger.info(f"[selection_expert] bypassing: explicit reference to previous results")
                        return SelectionResult(False, reason="prev_result_reference")
                else:
                    if kw in ql:
                        logger.info(f"[selection_expert] bypassing: explicit reference to previous results (zh)")
                        return SelectionResult(False, reason="prev_result_reference")

        # ── Step 2: Verify selection keyword + file keyword co-occurrence ─
        # English: selection word must appear BEFORE file word
        _SEL_WORD_PATS = [
            re.compile(r"\bsele?c?te?d\b"),
            re.compile(r"\bsl[e]ct[e]?d\b"),
            re.compile(r"\bchose?n\b"),
            re.compile(r"\bchosn\b"),
            re.compile(r"\bpicked\b"),
        ]
        _FILE_CTX_RE = re.compile(
            r"\b(files?|documents?|docs?|sources?|folders?|data"
            r"|csv|pdf|txt|docx?|xlsx?|pptx?|md|json|xml|html"
            r"|jpg|jpeg|png|gif|mp3|mp4|wav|zip|rar|numbers|pages|keynote"
            r"|image|images|photo|photos|audio|audios|video|videos|movie|movies|clip|clips|media)\b",
            re.IGNORECASE,
        )

        # Chinese patterns
        _ZH_PATS = [
            re.compile(r"选中.{0,4}(文件|文档|资料|数据源)"),
            re.compile(r"已选.{0,4}(文件|文档|资料|数据源)"),
            re.compile(r"(这些|那些)(文件|文档)"),
            re.compile(r"当前选中"),
            re.compile(r"已勾选"),
        ]

        has_zh_match = any(p.search(ql) for p in _ZH_PATS)

        sel_match = None
        for p in _SEL_WORD_PATS:
            m = p.search(ql)
            if m:
                sel_match = m
                break
        file_match = _FILE_CTX_RE.search(ql)

        en_ok = False
        if sel_match and file_match and sel_match.start() <= file_match.start():
            between = ql[sel_match.end():file_match.start()]
            if not cls._SEARCH_ESCAPE_RE.search(between):
                en_ok = True

        # Check demonstrative patterns
        has_demo = any(p in ql for p in cls._DEMONSTRATIVE_TRIGGERS)

        if not (has_zh_match or en_ok or has_demo):
            return SelectionResult(False, reason="no_selection_signal")

        if cls._SELECTED_EXTENSION_RE.search(ql) and cls._looks_like_selection_inventory_query(ql):
            logger.info(
                f"[selection_expert] selected+extension inventory → count_selected ({len(active_paths)} files)"
            )
            return SelectionResult(True, "count_selected", 0.96, "selected_extension_inventory", params=scope_params)

        if cls._INVENTORY_PREFIX_RE.search(ql) and cls._looks_like_selection_inventory_query(ql):
            logger.info(f"[selection_expert] explicit inventory phrasing on selected → list_selected ({len(active_paths)} items)")
            return SelectionResult(True, "list_selected", 0.96, "selection_inventory_phrase", params=scope_params)

        # ── Step 3: Determine intent sub-type ─────────────────────────────
        # Content intent → summarize_all
        if cls._CONTENT_INTENT_RE.search(ql):
            if scope_params.get("category") in {"audio", "video", "audio/video"}:
                logger.info(
                    f"[selection_expert] content query on selected media → summarize_selected "
                    f"({len(active_paths)} files, media_type={scope_params.get('media_type') or 'all'})"
                )
                return SelectionResult(True, "summarize_selected", 0.95, "content_intent_on_selected_media", params=scope_params)
            logger.info(f"[selection_expert] content query on selected → summarize_all ({len(active_paths)} files)")
            return SelectionResult(True, "summarize_all", 0.95, "content_intent_on_selection")

        # Numeric count intent → count_selected
        if any(kw in ql for kw in cls._COUNT_KWS):
            logger.info(f"[selection_expert] count query on selected → count_selected ({len(active_paths)} files)")
            return SelectionResult(True, "count_selected", 0.95, "count_intent_on_selection", params=scope_params)

        # Plain listing intent → list currently selected items, not every descendant under selected folders.
        if any(kw in ql for kw in cls._LIST_KWS):
            logger.info(f"[selection_expert] list query on selected → list_selected ({len(active_paths)} items)")
            return SelectionResult(True, "list_selected", 0.96, "list_intent_on_selection", params=scope_params)

        # Default: bare "selected files" is more naturally a request to surface the selected items.
        logger.info(f"[selection_expert] default selected → list_selected ({len(active_paths)} items)")
        return SelectionResult(True, "list_selected", 0.9, "default_selection_intent", params=scope_params)
