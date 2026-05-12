"""
IntentAnalyzer — orchestrates the micro-agent pipeline for intent recognition.

v3 Architecture: Each fast-path has been extracted to a dedicated micro-agent.
This file is now thin (~200 lines) — it only orchestrates the agent pipeline.

Pipeline order:
  1. SelectionExpert      (if active_paths)
  2. ContextFollowupExpert (if has history/context)
  3. CountExpert          (how many files)
  4. FilenameExpert       (bare filename)
  5. CategoryListExpert   (find + file-type)
  6. EntitySearchExpert   (bare ≤3 word entity)
  7. LLM Router fallback  (ConversationRouter → L2 agents)
  8. IntentValidator      (correction chain)
"""
import logging
import json
import os
import re
from typing import Optional, List, Dict, Callable, Any

logger = logging.getLogger(__name__)


class IntentKeywords:
    """Centralized Registry for Intent Recognition Keywords"""
    
    # Generic content queries
    CONTENT_KWS = {
        "zh": ["讲什么", "怎么说", "内容", "总结", "归纳", "关于什么", "细节", "要点", "描述", "说明", "介绍", "分析", "在做什么", "干什么", "分别", "结论"],
        "en": ["content", "summary", "summarize", "sunmery", "sumary", "summery", "summaries", "details", "key points", "about what", "describe", "description", "explain", "analysis", "doing", "in each", "for each", "conclude", "conclusion"]
    }
    
    # File object queries
    FILE_OBJ_KWS = {
        "zh": ["文件", "文档", "资料", "数据源", "图片", "照片", "pdf", "word", "表格", "截图", "图像", "csv", "excel", "txt", "docx", "doc", "jpg", "png", "mp3", "mp4", "ppt", "pptx", "xls", "xlsx", "表", "工作表", "数据表", "数据集", "视频", "音频", "录音", "音视频", "视频文件", "音频文件"],
        "en": ["file", "files", "document", "documents", "source", "sources", "doc", "docs", "image", "images", "photo", "photos", "picture", "pictures", "img", "pdf", "word", "csv", "excel", "txt", "docx", "jpg", "png", "mp3", "mp4", "ppt", "pptx", "xls", "xlsx", "worksheet", "worksheets", "spreadsheet", "spreadsheets", "dataset", "datasets", "table", "tables", "video", "videos", "audio", "audios", "recording", "recordings"]
    }
    
    # Generic listing requests
    ASK_KWS = {
        "zh": ["有哪些", "有多少", "多少个", "多少份", "列出", "清单", "全部", "所有", "一共", "总共", "什么文件", "有什么文件", "哪些文件", "哪些文档", "什么文档", "有什么文档", "有没有", "找哪些", "查哪些", "相关文件", "相关文档"],
        "en": ["what files", "which files", "how many", "list", "all", "total", "show me", "do i have", "find files", "show files", "related files", "matching files"]
    }
    
    # Explicit counting identifiers
    EXPLICIT_COUNT_KWS = {
        "zh": ["多少份", "多少个", "一共", "总共", "共有", "几份", "几个文件", "几个文档"],
        "en": ["how many", "count", "total", "number of"]
    }

    # Reference to previous results
    PREV_REF_KWS = {
        "zh": ["刚才", "上次", "上一轮", "前面", "上述", "以上", "上面那些", "上面的内容", "上面列出", "上文", "查看之后", "看完之后", "看过之后", "前面的内容", "以上列出的", "你提供的", "这批", "这些文件", "这些结果", "这个结果", "其中", "这些里面", "这批里面", "这个文档", "该文档", "这份文档", "这个文件", "该文件", "这份文件", "这个资料", "这些", "那些", "他们", "它们", "这几个", "那几个", "上面", "上面的", "上的", "前文", "刚才的", "之前的", "这其中"],
        "en": ["previous", "last", "earlier", "above", "those files", "these files", "those results", "these results", "among them", "in these", "number ", "this document", "that document", "this file", "that file", "this one", "that one", "they", "them", "these", "those"]
    }
    
    # Short summary follow-ups
    SHORT_FOLLOWUP_SUMMARY_KWS = {
        "zh": ["总结一下", "总结一下吧", "总结下", "总结下吧", "概括一下", "概括一下吧", "归纳一下", "归纳一下吧", "总结一下他们", "总结他们", "总结它们", "总结一下它们", "结论", "总结", "归纳", "概括", "汇总", "讲一下", "说说看", "介绍下", "得出结论", "要点", "关键要点是什么", "核心内容"],
        "en": ["summarize", "summarize it", "summarize them", "summary please", "conclusion for me", "a conclusion", "give me a conclusion", "give me a summary", "summary for me", "summarize for me", "recap", "recap for me", "tldr", "in short", "conclusion", "conclude", "summary", "overview", "wrap up", "wrap-up", "briefly", "key takeaways", "takeaways"]
    }
    
    # Short affirmation follow-ups
    SHORT_AFFIRM_KWS = {
        "zh": ["需要", "要", "好的", "好", "行", "可以", "嗯", "嗯嗯", "是的", "继续", "然后呢", "再来", "确认下", "确认一下", "帮我确认下", "核对下", "核对一下", "检查下", "检查一下", "详细说说", "详细一点"],
        "en": ["yes", "yeah", "yep", "sure", "ok", "okay", "go on", "continue", "confirm", "verify", "check this", "help me confirm", "please confirm"]
    }

    # Explicit Summarize All keywords
    SUMMARIZE_ALL_KWS = {
        "zh": ["总结下我的所有", "总结所有文件", "整理所有文件", "全局总结", "所有内容", "总结下所有", "结论", "所有结论"],
        "en": [
            "summarize all files", "overall summary", "summarize everything",
            "summarize all my files", "summery all files", "sunmery all files",
            "summary of all my", "summary all my", "give me a summary of all my",
            "conclude all my files", "conclude all files", "conclude my files", "conclusion of all my",
            "recap all my", "recap all files", "wrap up all",
        ]
    }
    
    # Stopwords/Noise allowed to bypass scoped search rules
    NOISE_TOKENS = {
        "zh": ["相关的", "相关", "有关的", "有关", "关于", "跟", "与", "里面", "其中", "当前", "选中的", "我的", "我", "帮我", "给我", "看看", "看下", "查看", "查一下", "查下", "找一下", "找下", "列出", "显示", "筛选", "搜索", "一下", "下", "吧", "告诉", "告诉我", "说一下"],
        "en": ["what", "which", "find", "show", "list", "related", "matching", "about", "with", "my", "me", "the", "that", "those", "these", "are", "is", "do", "did", "does", "have", "has", "had", "i", "you", "we", "they", "he", "she", "it", "a", "an", "any", "some", "all", "of", "in", "to", "for", "on", "at", "by", "from", "can", "could", "would", "will", "please", "selected", "current"]
    }


class IntentContext:
    def __init__(self,
                 question: str,
                 prompt_language: str,
                 history: List[Dict],
                 last_results: List[Dict],
                 get_category_keywords_fn: Callable[[], List[str]],
                 is_generic_category_fn: Callable[[str], bool],
                 normalize_category_fn: Callable[[str], str],
                 llm_service: Any,
                 category_info: str,
                 prompt_formatter: Callable[[str, str], str],
                 log_followup_guard_fn: Callable = None,
                 session_id: Optional[str] = None,
                 active_paths: Optional[List[str]] = None,
                 opened_file_path: Optional[str] = None,
                 user_lang: Optional[str] = None):
        self.question = question
        self.prompt_language = prompt_language
        self.user_lang = user_lang or prompt_language
        self.history = history
        self.last_results = last_results
        self.get_category_keywords = get_category_keywords_fn
        self.is_generic_category = is_generic_category_fn
        self.normalize_category = normalize_category_fn
        self.llm_service = llm_service
        self.category_info = category_info
        self.prompt_formatter = prompt_formatter
        self.log_followup_guard = log_followup_guard_fn
        self.session_id = session_id
        self.active_paths = active_paths
        self.opened_file_path = opened_file_path


class IntentAnalyzer:
    """
    Orchestrates the micro-agent pipeline for intent recognition.
    
    v3: Each fast-path has been extracted to a dedicated micro-agent.
    This class only wires the agents together in correct priority order.
    """
    
    @classmethod
    def _is_kw_match(cls, text: str, kws_dict: Dict[str, List[str]], lang: str = "all") -> bool:
        text_lower = text.lower()
        if lang in kws_dict:
            return any((k.lower() in text_lower) for k in kws_dict[lang])
        elif lang == "all":
            return any((k.lower() in text_lower) for kw_list in kws_dict.values() for k in kw_list)
        return False

    @classmethod
    def _extract_prior_action_context(cls, ctx: 'IntentContext') -> dict:
        """
        Analyze conversation history to understand what the previous exchange was about.
        Returns a context dict used by context-aware fast-paths.
        """
        out: dict = {
            "prior_action": None,
            "prior_was_count": False,
            "prior_was_content": False,
            "prior_was_media": False,
            "prior_was_search": False,
            "prior_search_failed": False,
            "n_prior_files": len(ctx.last_results or []),
            "focused_file": None,
            "prior_user_query": None,
        }

        history = ctx.history or []
        if not history:
            return out

        last_q = ""
        for msg in reversed(history):
            q = str(msg.get("q") or "")
            if q:
                last_q = q.strip()
                break
            if msg.get("role") == "user":
                last_q = str(msg.get("content") or "").strip()
                if last_q:
                    break
        if last_q:
            out["prior_user_query"] = last_q[:160]

        last_a = ""
        for msg in reversed(history):
            a = str(msg.get("a") or "")
            if a:
                last_a = a.strip()
                break
            if msg.get("role") == "assistant":
                last_a = str(msg.get("content") or "").strip()
                if last_a:
                    break

        if not last_a:
            return out

        _SEARCH_FAIL_SIGS = [
            r"no relevant indexed content found",
            r"no highly relevant indexed content found",
            r"no directly relevant files remained",
            r"no indexed files matched",
            r"no files?\s+with\s+the\s+exact\s+requested\s+name\s+were\s+found",
            r"no files?\s+matching\s+the\s+requested\s+name\s+were\s+found",
            r"couldn't find.*results",
            r"未找到.*相关.*(文件|文档|资料|内容)",
            r"没有找到.*相关.*(文件|文档|资料|内容)",
            r"没有找到上一轮的查询结果",
        ]
        _SEARCH_SUCCESS_SIGS = [
            r"found\s+\d+\s+relevant files",
            r"found\s+\d+\s+files",
            r"relevant files",
            r"匹配文件",
            r"相关文件",
        ]
        if any(re.search(sig, last_a, re.IGNORECASE) for sig in _SEARCH_FAIL_SIGS):
            out["prior_action"] = "search"
            out["prior_was_search"] = True
            out["prior_search_failed"] = True
        elif any(re.search(sig, last_a, re.IGNORECASE) for sig in _SEARCH_SUCCESS_SIGS):
            out["prior_action"] = out["prior_action"] or "search"
            out["prior_was_search"] = True

        _COUNT_SIGS = [
            r"\U0001F4CA",
            r"there are \d+\s+(document|file)",
            r"\d+\s+(document|file)s?\s+in the selected",
            r"category\s*[\|┃]\s*count",
            r"can further drill down",
            r"largest group",
            r"this distribution",
            r"共\s*\d+\s*(个|份)?\s*(文件|文档)",
            r"一共\s*\d+",
            r"总计\s*\d+",
            r"no.*files?\s+(found|match)",
        ]
        for sig in _COUNT_SIGS:
            if re.search(sig, last_a, re.IGNORECASE):
                out["prior_action"] = "count"
                out["prior_was_count"] = True
                break

        _CONTENT_SIGS = [
            r'document titled ["「『]',
            r'file (titled|named|called)\s+["「『]',
            r'this document (is|serves|provides|covers|explains|describes|shows)',
            r'the document (discusses|covers|describes|contains|provides|explains)',
            r'the file (discusses|covers|describes|contains|provides|explains)',
            r'(firmware|烧录|固件|用于|主要内容|本文档|本文件)',
        ]
        for sig in _CONTENT_SIGS:
            if re.search(sig, last_a, re.IGNORECASE):
                out["prior_action"] = "summarize"
                out["prior_was_content"] = True
                break

        _MEDIA_SIGS = [
            r'🎙️', r'🖼️', r'📝', r'\[(?:约\s*)?\d+(?:(?:分|分钟)?\d+)?(?:秒|s)\]', r'\[~\s*\d+[hms\s]+\]',
            r'asr transcript', r'keyframe', r'视频帧', r'音频片段', r'画面',
        ]
        for sig in _MEDIA_SIGS:
            if re.search(sig, last_a, re.IGNORECASE):
                out["prior_was_media"] = True
                out["prior_was_content"] = True
                break

        last_results = ctx.last_results or []
        if not out["prior_was_media"]:
            prior_user_query = str(out.get("prior_user_query") or "").lower()
            if re.search(r'\b(video|videos|audio|audios|recording|recordings|clip|clips|movie|movies)\b|视频|音频|录音|录像', prior_user_query, re.IGNORECASE):
                out["prior_was_media"] = True
            elif last_results:
                media_exts = {
                    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
                    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv",
                }
                sample = list(last_results[:12])
                media_hits = 0
                from core.retrieval.category_engine import is_media_category_value
                for doc in sample:
                    doc_category = str(doc.get("doc_category") or "").strip().lower()
                    file_name = str(doc.get("file_name") or doc.get("file_path") or "")
                    ext = os.path.splitext(file_name)[1].lower() if file_name else ""
                    if is_media_category_value(doc_category) or ext in media_exts:
                        media_hits += 1
                if sample and media_hits >= max(1, (len(sample) + 1) // 2):
                    out["prior_was_media"] = True
        unique_result_names: List[str] = []
        unique_result_keys: set[str] = set()
        for row in list(last_results or []):
            key = str(row.get("file_path") or row.get("file_name") or "").strip()
            if not key or key in unique_result_keys:
                continue
            unique_result_keys.add(key)
            unique_result_names.append(str(row.get("file_name") or os.path.basename(key) or "").strip())
        if len(unique_result_names) == 1:
            out["focused_file"] = unique_result_names[0] or None

        return out
        
    @classmethod
    def analyze(cls, ctx: IntentContext) -> dict:
        """
        Context-aware intent routing via micro-agent pipeline.

        Architecture selection:
          - Default (v1): sequential rule fast-path (Agents 0-6) → LLM Router
          - v2 (FILEAGENT_INTENT_ARCH=v2): IntentClassifier (LLM) → Expert Groups

        Agent priority (v1):
          0. ExplicitContinuationExpert(fast-path for "continue", "in english")
          1. SelectionExpert       (active_paths + selection keyword)
          2. ContextFollowupExpert (post-count/post-content/implicit ref)
          3. CountExpert           ("how many files")
          4. FilenameExpert        (bare filename with extension)
          5. CategoryListExpert    ("find all pdf")
          6. EntitySearchExpert    (≤3 word bare entity)
          7. LLM Router fallback   (ConversationRouter → L2 agents)
          … all outputs pass through IntentValidator
        """
        import os
        if os.environ.get("FILEAGENT_INTENT_ARCH", "").strip().lower() == "v2":
            return cls._analyze_v2(ctx)

        qn = (ctx.question or "").strip()
        ql = qn.lower()
        lang = ctx.prompt_language

        # ── Agent 0: ExplicitContinuationExpert ───────────────────────────
        from core.intent.continuation_agent import ExplicitContinuationExpert
        explicit_cont = ExplicitContinuationExpert.analyze(qn, ctx.history)
        if explicit_cont is not None:
            return explicit_cont

        # ── Agent 0.5: MediaQueryExpert ────────────────────────────────────
        try:
            from core.intent.media_query_expert import MediaQueryExpert
            if ctx.last_results and (
                cls.looks_like_meta_followup_on_last_results(qn, lang)
                or cls.looks_like_content_followup_on_prior_results(qn)
            ):
                logger.debug(f"[IntentAnalyzer] Skip MediaQueryExpert for prior-results followup query_chars={len(qn)}")
            else:
                logger.debug(f"[IntentAnalyzer] Testing MediaQueryExpert query_chars={len(qn)}")
                media_result = MediaQueryExpert.analyze(
                    qn, 
                    last_results=ctx.last_results, 
                    llm_service=ctx.llm_service
                )
                if media_result is not None:
                    logger.info(f"[IntentAnalyzer] MediaQueryExpert MATCH: {media_result}")
                    return media_result
                else:
                    logger.info(f"[IntentAnalyzer] MediaQueryExpert NO MATCH")
        except Exception as e:
            logger.error(f"[IntentAnalyzer] MediaQueryExpert failed: {e}", exc_info=True)

        # ── Agent 1: SelectionExpert ──────────────────────────────────────
        from core.intent.selection_expert import SelectionExpert
        if SelectionExpert.should_activate(qn, ctx.active_paths):
            _prior_ctx_sel = cls._extract_prior_action_context(ctx)
            _prior_action_sel = str(_prior_ctx_sel.get("prior_action") or "")
            sel_result = SelectionExpert.classify(
                qn, ctx.active_paths, ctx.last_results,
                llm_service=ctx.llm_service, lang=ctx.prompt_language,
                prior_action=_prior_action_sel,
            )
            intent = sel_result.to_intent()
            if intent is not None:
                return intent

        # ── Agent 1.5: PersonalAttributeGuard ──────────────────────────
        # Intercept "Name + attribute" queries BEFORE ContextFollowupExpert.
        # Uses the canonical regex from ContextFollowupExpert (single source of truth).
        from core.intent.context_followup_expert import ContextFollowupExpert
        if ContextFollowupExpert._ATTR_LOOKUP_RE.search(ql):
            logger.debug(f"[IntentAnalyzer] personal attr guard → search query_chars={len(qn)}")
            return {"action": "search", "params": {"query": qn}, "confidence": 0.95}

        # ── Agent 2: ContextFollowupExpert ────────────────────────────────
        # (ContextFollowupExpert already imported above)
        prior_ctx = cls._extract_prior_action_context(ctx)
        followup = ContextFollowupExpert.analyze_context_followup(
            qn, prior_ctx,
            last_results=ctx.last_results,
            active_paths=ctx.active_paths,
        )
        if followup is not None:
            return followup


        # ── Agent 3: CountExpert ──────────────────────────────────────────
        from core.intent.count_expert import CountExpert
        if CountExpert.is_how_many_files(qn):
            return CountExpert.to_intent()

        # ── Agent 4: FilenameExpert ───────────────────────────────────────
        from core.intent.entity_experts import FilenameExpert
        if FilenameExpert.is_bare_filename(qn):
            return FilenameExpert.to_intent(qn)

        # ── Agent 5: CategoryListExpert ───────────────────────────────────
        from core.intent.entity_experts import CategoryListExpert
        has_qualifier = cls._is_kw_match(ql, IntentKeywords.CONTENT_KWS)
        cat_result = CategoryListExpert.analyze(qn, has_content_qualifier=has_qualifier)
        if cat_result is not None:
            return cat_result

        # ── Agent 6: EntitySearchExpert ───────────────────────────────────
        from core.intent.entity_experts import EntitySearchExpert
        if EntitySearchExpert.is_bare_entity(qn, active_paths=ctx.active_paths):
            return EntitySearchExpert.to_intent(qn)

        # ── Agent 7: LLM Router fallback ─────────────────────────────────
        return cls._run_llm_intent_parser(ctx)

    @classmethod
    def _analyze_v2(cls, ctx: IntentContext) -> dict:
        """
        v2 Architecture: LLM-first top-level routing + Expert Groups.

        Stage 0: Deterministic rules (no LLM, microseconds)
          - ExplicitContinuationExpert, FilenameExpert, PersonalAttributeGuard
        Stage 1: IntentClassifier (1 LLM call, ~130 tok) → group
          - group ∈ {continuation, selection, media, file_op, chat}
        Stage 2: Expert Group dispatch (rules + optional group LLM)
          - ContinuationGroup: ContextFollowupExpert rules + FollowupClassifier LLM
          - SelectionGroup: SelectionExpert.classify (already MoE)
          - MediaGroup: MediaQueryExpert rules + MediaOpAgent LLM
          - FileOpGroup: Count/Category/Entity rules + FileOpAgent LLM
        Stage 3: IntentValidator (confidence-aware, existing)
        """
        from core.intent.intent_classifier import IntentClassifier
        from core.intent.validator import IntentValidator

        qn = (ctx.question or "").strip()
        ql = qn.lower()
        lang = ctx.prompt_language
        active_paths = ctx.active_paths or []
        last_results = ctx.last_results or []

        # ── Stage 0: Deterministic fast-paths (no LLM) ───────────────────
        from core.intent.continuation_agent import ExplicitContinuationExpert
        r = ExplicitContinuationExpert.analyze(qn, ctx.history)
        if r is not None:
            return r

        from core.intent.entity_experts import FilenameExpert
        if FilenameExpert.is_bare_filename(qn):
            return FilenameExpert.to_intent(qn)

        from core.intent.context_followup_expert import ContextFollowupExpert
        if ContextFollowupExpert._ATTR_LOOKUP_RE.search(ql):
            logger.debug(f"[v2] personal attr guard → search query_chars={len(qn)}")
            return {"action": "search", "params": {"query": qn}, "confidence": 0.95}

        # ── Stage 1: IntentClassifier (LLM top-level router) ─────────────
        group, group_conf = IntentClassifier.classify(ctx)
        logger.info(f"[v2] IntentClassifier: group={group!r} conf={group_conf:.2f}")

        # ── Stage 2: Expert Group dispatch ───────────────────────────────
        result: dict = {"action": "search", "params": {"query": qn}, "confidence": 0.6}

        if group == "continuation":
            from core.intent.continuation_group import ContinuationGroup
            result = ContinuationGroup.route(qn, ctx)
            # ContinuationGroup may decide it's a new request → re-route to file_op
            if result.get("action") == "fallback_to_file_op":
                group = "file_op"
            else:
                result.setdefault("confidence", group_conf)

        if group == "selection" and active_paths:
            from core.intent.selection_expert import SelectionExpert
            if SelectionExpert.should_activate(qn, active_paths):
                # Pass prior_action so deictic 'it' in summarize context skips clarify
                _prior_action_v2 = str(getattr(ctx, "prior_intent_action", "") or "")
                sel_result = SelectionExpert.classify(
                    qn, active_paths, last_results,
                    llm_service=ctx.llm_service, lang=lang,
                    prior_action=_prior_action_v2,
                )
                intent = sel_result.to_intent()
                if intent is not None:
                    result = intent
                else:
                    group = "file_op"
            else:
                group = "file_op"

        if group == "media":
            from core.intent.media_query_expert import MediaQueryExpert
            media_r = MediaQueryExpert.analyze(
                qn, last_results=last_results, llm_service=ctx.llm_service
            )
            if media_r is not None:
                result = media_r
            else:
                from core.intent.media_sub_agent import MediaSubAgent
                result = MediaSubAgent.analyze(ctx)

        if group == "chat":
            result = {"action": "chat", "params": {}, "confidence": 0.90}

        if group == "file_op":
            from core.intent.count_expert import CountExpert
            if CountExpert.is_how_many_files(qn):
                result = CountExpert.to_intent()
            else:
                from core.intent.entity_experts import EntitySearchExpert
                if EntitySearchExpert.is_bare_entity(qn, active_paths=active_paths):
                    # Bare ≤3-word entity (e.g. "person name", "anker", "tencent.pdf") — sub-ms rule
                    result = EntitySearchExpert.to_intent(qn)
                else:
                    # Everything else (category search, find-verb+type, summarize, open, etc.)
                    # → FileOpAgent LLM: its prompt already handles all these cases well
                    from core.intent.file_op_agent import FileOpAgent
                    result = FileOpAgent.analyze(ctx)

        # ── Stage 3: IntentValidator (confidence-aware) ───────────────────
        result.setdefault("confidence", group_conf)
        return IntentValidator.validate(
            qn, result,
            last_results=last_results,
            history=ctx.history,
            active_paths=active_paths,
            prompt_language=lang,
        )

    @classmethod
    def _has_file_op_signal(cls, question: str) -> bool:
        """Check if query contains strong file operation or thematic search signals."""
        q = (question or "").strip().lower()
        if not q:
            return False
            
        file_op_verbs = re.compile(
            r'\b(find|search|look\s+for|look\s+up|show\s+me|retrieve|locate|get\s+me|list|do\s+i\s+have)\b|'
            r'(查找|搜索|找|检索|查询|调出|列出|帮我找|搜一下|查一下|找一下)', re.IGNORECASE
        )
        
        thematic_nouns = re.compile(
            r'\b(files?|documents?|docs?|resume|paper|papers|invoice|invoices?|report|recording|recordings?|'
            r'photo|photos|image|images|slides?|presentation|manual|datasheet|config|'
            r'csv|pdf|wav|mp3|pptx?|docx?|xlsx?|'
            r'chip|diagram|brief|survey|article|book)s?\b',
            re.IGNORECASE
        )
        
        # Also detect Chinese thematic nouns
        zh_nouns = re.compile(
            r'(文件|文档|简历|论文|发票|报告|录音|幻灯片|配置|数据|图片|手册|芯片|简介|音效|表格)'
        )
        
        has_verb = bool(file_op_verbs.search(q))
        has_noun = bool(thematic_nouns.search(q) or zh_nouns.search(q))
        
        # Strong signal: verb + noun together
        if has_verb and has_noun:
            return True
        # Also strong: query starts with a file-op verb
        if re.match(r'^(find|search|show me|look for|list|retrieve|locate|get me)\b', q):
            return True
        # Implicit search: "any papers", "are there any documents"
        if has_noun and re.search(r'\b(any|are there any|do we have|is there any)\b', q):
            return True
        # Chinese: query contains search verb
        if re.search(r'(搜索|查找|帮我找|搜一下|查一下|找一下)', q):
            return True
        return False

    @classmethod
    def _run_llm_intent_parser(cls, ctx: IntentContext) -> dict:
        """
        3-layer multi-agent routing (v2 architecture).

        Layer 1: ConversationRouter  — lightweight LLM call; outputs continuation/file_op/chat
        Layer 2A: ContinuationAgent  — translate_response | process_previous | chat
        Layer 2B: FileOpAgent        — search | count | summarize | summarize_all

        Falls back to v1 (monolithic LLM) when FILEAGENT_ROUTER_MODE=v1.
        """
        import os
        if os.environ.get("FILEAGENT_ROUTER_MODE", "").strip().lower() == "v1":
            return cls._run_llm_intent_parser_v1(ctx)

        from core.intent.router import ConversationRouter
        from core.intent.continuation_agent import ContinuationAgent
        from core.intent.file_op_agent import FileOpAgent
        from core.intent.media_sub_agent import MediaSubAgent
        from core.intent.validator import IntentValidator

        try:
            route = ConversationRouter.route(ctx)
        except Exception as e:
            logger.error(f"[Router] failed: {e} — falling back to file_op", exc_info=True)
            route = "file_op"

        if route == "continuation":
            result = ContinuationAgent.analyze(ctx)
            if result.get("action") == "fallback_to_file_op":
                logger.info("[Router] ContinuationAgent determined it's a new request, falling back to FileOpAgent.")
                result = FileOpAgent.analyze(ctx)
            elif result.get("action") == "chat" and cls._has_file_op_signal(ctx.question):
                logger.info("[Router] ContinuationAgent returned chat but file-op signal detected → FileOpAgent")
                result = FileOpAgent.analyze(ctx)
        elif route == "media":
            result = MediaSubAgent.analyze(ctx)
        elif route == "chat":
            if cls._has_file_op_signal(ctx.question):
                logger.info("[Router] chat overridden → file_op (file-op signal detected)")
                result = FileOpAgent.analyze(ctx)
            else:
                result = {"action": "chat", "params": {}}
        else:  # file_op (default)
            result = FileOpAgent.analyze(ctx)

        # Apply centralized validation (replaces old correct_llm_intent)
        return IntentValidator.validate(
            ctx.question,
            result,
            last_results=ctx.last_results,
            history=ctx.history,
            active_paths=ctx.active_paths,
            prompt_language=ctx.prompt_language,
        )

    @classmethod
    def _run_llm_intent_parser_v1(cls, ctx: IntentContext) -> dict:
        """Legacy monolithic LLM intent parser (v1). Used when FILEAGENT_ROUTER_MODE=v1."""
        ql = (ctx.question or "").strip().lower()
        lang = ctx.prompt_language
        context_info = ""
        last_results = ctx.last_results or []

        if ctx.active_paths:
            import os
            selected_names = [os.path.basename(p) for p in ctx.active_paths[:10]]
            n_sel = len(ctx.active_paths)
            bullets = ["  - " + n for n in selected_names]
            sel_str = chr(10).join(bullets)
            if lang == "en":
                context_info += (
                    f"\n[ACTIVE SELECTION - {n_sel} file(s) in Sources panel]\n"
                    f"User ticked these {n_sel} file(s):\n{sel_str}\n\n"
                    "CRITICAL: 'selected', 'the selected documents', 'these files', 'seleted' (typos ok)"
                    f" ALL mean EXACTLY these {n_sel} files above - NOT a keyword search.\n"
                    "For queries like 'tell me about the selected', 'what are these files':"
                    " use action='list_selected' (empty params). NEVER search or process_previous.\n\n"
                )
            else:
                context_info += (
                    f"\n[当前选区 - 已勾选 {n_sel} 个文件]\n"
                    f"用户选中：\n{sel_str}\n\n"
                    "'选中的文件'、'这些文件'、'这些文档' 只指上面"
                    f" {n_sel} 个，与'我的所有文件'完全不同，与全库无关。\n"
                    "告诉我选中文件/介绍这些文件 → action='list_selected'，"
                    "禁止 search 或 process_previous。\n\n"
                )

        if last_results and (cls._is_kw_match(ctx.question, IntentKeywords.PREV_REF_KWS) or cls.looks_like_meta_followup_on_last_results(ctx.question, lang)):
            _STRONG_ACTION_PAT = re.compile(
                r'^(find|search|show|list|get|display|retrieve|找|搜|显示|列出)\b',
                re.IGNORECASE
            )
            if _STRONG_ACTION_PAT.match(ql):
                pass
            else:
                total = len(last_results)
                if lang == "en":
                    context_info = f"[Previous search/stat result] Assistant just showed {total} files to the user.\n"
                else:
                    context_info = f"[上次搜索/统计结果] 助手刚才向用户展示了 {total} 份文件\n"
                for i, doc in enumerate(last_results[:10], 1):
                    context_info += f"{i}. {doc.get('file_name', '')} - {doc.get('doc_summary', '')[:50]}\n"
                if lang == "en":
                    context_info += (
                        "\nMulti-turn: if the current message is a short follow-up about THOSE listed files "
                        "(e.g. 'conclusion for me', 'summarize', 'recap', 'tldr', 'key points', 'what do these say'), "
                        "you MUST use action=process_previous. Do NOT treat such text as a new search query.\n"
                        "If the user uses pronouns ('them', 'these') or asks to summarize/filter within that list, also process_previous.\n\n"
                    )
                else:
                    context_info += (
                        "\n多轮对话：若当前输入是针对上述已列出文件的短跟进（如「结论」「总结一下」「说说要点」），"
                        "必须使用 process_previous，不要把整句当成新的 search 检索词。\n"
                        "若使用代词或要求在该列表内总结/筛选，同样用 process_previous。\n\n"
                    )
        
        from tools import IntentRegistry
        from services.local_llm import get_local_llm_manager
        actions_block = IntentRegistry.render_actions_block(language=lang)
        rules_block = IntentRegistry.render_rules_block(language=lang)
        rules_text = f"\n{'[Important Rules]' if lang == 'en' else '[重要规则]'}\n{rules_block}\n" if rules_block else ""

        prompt = ctx.prompt_formatter("INTENT_DETECTION_PROMPT", lang).format(
            category_info=ctx.category_info,
            context_info=context_info,
            query=ctx.question,
            actions_text=actions_block + rules_text
        )
        system = ctx.prompt_formatter("INTENT_DETECTION_SYSTEM_PROMPT", lang)
        
        try:
            llm_mgr = get_local_llm_manager()
            model_id = llm_mgr.current_model_id
            if model_id:
                cfg = llm_mgr.get_target_model_config(model_id) or {}
                suffix = cfg.get("intent_prompt_suffix", "")
                if suffix:
                    prompt += f"\n{suffix}"
        except Exception:
            pass

        if (os.getenv("FILEAGENT_LOG_PROMPTS", "") or "").strip().lower() in {"1", "true", "yes", "on"}:
            logger.debug(f"===== INTENT PROMPT =====\n{system}\n{prompt}\n=====================")
        else:
            logger.debug(
                "Intent prompt prepared: system_chars=%s prompt_chars=%s query_chars=%s",
                len(system or ""),
                len(prompt or ""),
                len(ctx.question or ""),
            )
        response = ctx.llm_service.generate(prompt, history=ctx.history, system_prompt=system)
        response_stripped = (response or "").strip()

        summarize_kws = IntentKeywords.SUMMARIZE_ALL_KWS.get("en", []) + IntentKeywords.SUMMARIZE_ALL_KWS.get("zh", [])
        if any(kw in ctx.question.lower() for kw in summarize_kws) and (
            ("文件" in ctx.question or "内容" in ctx.question or "资料" in ctx.question) or ("file" in ctx.question.lower() or "content" in ctx.question.lower())
        ):
            logger.info(f"🔧 校正: 包含全局总结关键词 → 强制 summarize_all")
            return {"action": "summarize_all", "params": {}}

        try:
            start = response_stripped.find("{")
            if start < 0:
                logger.info(f"意图分析无法提取 JSON，raw={response_stripped[:100]}")
                return {"action": "search", "params": {"query": ctx.question}}
            decoder = json.JSONDecoder()
            result, _ = decoder.raw_decode(response_stripped, start)
            return cls.correct_llm_intent(ctx, result)
        except Exception as e:
            logger.error(f"意图分析 JSON 解析失败: {e}", exc_info=True)
            return {"action": "search", "params": {"query": ctx.question}}

    @classmethod
    def correct_llm_intent(cls, ctx: IntentContext, result: dict) -> dict:
        """
        Apply validation constraints atop LLM outputs.
        NOTE: In v3, most logic is in IntentValidator. This remains for v1 backward compat.
        Delegates to IntentValidator for the actual correction logic.
        """
        from core.intent.validator import IntentValidator
        return IntentValidator.validate(
            ctx.question,
            result,
            last_results=ctx.last_results,
            history=ctx.history,
            active_paths=ctx.active_paths,
            prompt_language=ctx.prompt_language,
        )

    # Patterns that unambiguously mean "list / count ALL my files" regardless of
    # what _looks_like_scoped_file_search_query says.  These fire BEFORE the
    # all my files" are never mis-classified as a topic-scoped search.
    # English "list all files" shortcut patterns only — Chinese is handled inline below.
    _ALL_FILES_EN_RE = re.compile(
        r'\b(what|which)\s+(files?|documents?|docs?)\s+(do\s+)?(i|we)\s+have\b'
        r'|'
        r'\b(show|list|display)\s+(me\s+)?(all\s+)?(my\s+)?(files?|documents?|docs?)\b',
        re.IGNORECASE,
    )
    # Tokens that, when they appear between the listing phrase and the file-type word,
    # indicate a topic-scoped query rather than a generic "list all" query.
    _ALL_FILES_TOPIC_RE = re.compile(
        r'关于|有关|相关|涉及|about|regarding|related|concerning|of\s+type|类型|格式|格式的',
        re.IGNORECASE,
    )
    # Specific file-format words (category filters, not generic "file")
    _FILE_FORMAT_WORDS = {
        "pdf", "word", "excel", "ppt", "pptx", "docx", "doc", "xls", "xlsx",
        "csv", "txt", "mp3", "mp4", "jpg", "jpeg", "png", "gif", "zip",
        "spreadsheet", "spreadsheets", "worksheet", "worksheets", "workbook", "workbooks",
        "table", "tables", "dataset", "datasets", "data",
        # English media/type words — must be here so "what video files do i have" is
        # treated as a scoped search (not a generic "list all files" = count(all))
        "video", "videos", "audio", "audios", "image", "images",
        "photo", "photos", "picture", "pictures", "recording", "recordings",
        "music", "movie", "movies", "clip", "clips", "screenshot", "screenshots",
        # Chinese equivalents
        "图片", "照片", "音频", "视频", "音乐", "代码", "录音", "截图",
        "表格", "数据表", "数据集", "数据", "工作表",
    }

    @classmethod
    def _looks_like_all_files_list_query(cls, question: str) -> bool:
        q = (question or "").strip()
        if not q:
            return False
        q_norm = q.lower()
        if cls._is_kw_match(q_norm, IntentKeywords.CONTENT_KWS):
            return False
        try:
            from core.retrieval.filename_canonicalizer import (
                extract_filename_query_surfaces,
                looks_like_specific_filename_candidate,
                normalize_filename_candidate,
            )

            explicit_surfaces = extract_filename_query_surfaces(q, max_candidates=1)
            if explicit_surfaces:
                candidate = normalize_filename_candidate(str(explicit_surfaces[0] or "").strip())
                if candidate and looks_like_specific_filename_candidate(candidate):
                    return False
        except Exception:
            pass

        # Pre-compute scoped signals before the broad English fast-path below.
        # Queries like "find the EPD module product list document" or
        # "show files comparing AI chips" are still file retrieval requests,
        # but they are scoped by a topic/qualifier and should not collapse into
        # a global inventory/count intent.
        has_topic = bool(cls._ALL_FILES_TOPIC_RE.search(q_norm))
        has_format = any(fmt in q_norm for fmt in cls._FILE_FORMAT_WORDS)
        has_scoped_search = cls._looks_like_scoped_file_search_query(question)

        # ── Fast-path A: English "list all files" patterns ───────────────────
        if cls._ALL_FILES_EN_RE.search(q_norm):
            if has_topic or has_format or has_scoped_search:
                return False
            return True

        # ── Fast-path B: Chinese first-person listing patterns ────────────────
        _ZH_LISTING_PHRASES = ("有哪些", "都有哪些", "有什么", "有多少", "有几个", "有几份")
        _ZH_FIRST_PERSON = ("我", "我的", "我这里", "我这边", "我手上", "我目前")
        _ZH_LIST_VERBS = ("查看", "列出", "看看", "看下", "显示")
        _ZH_FILE_WORDS = ("文件", "文档", "资料", "内容")
        _ZH_SCOPE_WORDS = ("所有", "全部", "的", "我的", "我有的", "全部的")

        # Pre-compute topic/format flags once to avoid redundant checks
        if not has_topic and not has_format:
            # but do not collapse category-scoped inventory queries such as
            for fw in _ZH_FILE_WORDS:
                for phrase in _ZH_LISTING_PHRASES:
                    if (fw + phrase) in q_norm or (phrase + fw) in q_norm:
                        return True

        for verb in _ZH_LIST_VERBS:
            if q_norm.startswith(verb):
                rest = q_norm[len(verb):]
                for sw in _ZH_SCOPE_WORDS:
                    rest = rest.replace(sw, "")
                rest = rest.strip()
                if rest in _ZH_FILE_WORDS or rest == "":
                    return True

        has_file = cls._is_kw_match(q_norm, IntentKeywords.FILE_OBJ_KWS)
        has_ask = cls._is_kw_match(q_norm, IntentKeywords.ASK_KWS)

        if not (has_file and has_ask):
            return False
        # even if _looks_like_scoped_file_search_query doesn't catch them.
        if not has_topic and not has_format:
            pass  # already computed above
        else:
            return False
        if has_scoped_search:
            return False
        return True

    @classmethod
    def _looks_like_scoped_file_search_query(cls, question: str) -> bool:
        q = (question or "").strip()
        if not q:
            return False
        ql = q.lower()
        if not cls._is_kw_match(ql, IntentKeywords.FILE_OBJ_KWS):
            return False
        if not cls._is_kw_match(ql, IntentKeywords.ASK_KWS):
            return False
        if cls._is_kw_match(ql, IntentKeywords.EXPLICIT_COUNT_KWS):
            return False
        if cls._is_kw_match(ql, IntentKeywords.CONTENT_KWS):
            return False

        # ── Early-exit: detect "find/search files ABOUT <topic>" pattern ─────
        # e.g. "find files about image", "find files about dogs"
        # The word after "about" is the topic; this is always a scoped search.
        _about_pat = re.search(r'\babout\s+(\w+)', ql)
        if _about_pat:
            topic = _about_pat.group(1)
            # If the topic word is itself only a file-object word (e.g. "about files"),
            # this is still a listing query — let it fall through.
            _pure_file_words = {"file", "files", "document", "documents", "doc", "docs"}
            if topic not in _pure_file_words:
                return True  # "about image", "about dogs", "about project" → scoped

        # ── Early-exit: "find/search + explicit media category" pattern ───────
        # e.g. "find image files", "search photo files", "look for audio files"
        # These are category-filtered searches, not generic "list all" queries.
        _SEARCH_VERBS = re.compile(
            r'\b(find|search|get|look|show|display|retrieve|查找|搜索|找|搜)\b',
            re.IGNORECASE
        )
        _CATEGORY_WORDS = {
            # Media types that are also in FILE_OBJ_KWS — distinguish from generic "files"
            "image", "images", "photo", "photos", "picture", "pictures",
            "video", "videos", "audio", "music", "mp3", "mp4", "wav",
            "pdf", "word", "excel", "powerpoint", "csv", "code",
            "图片", "照片", "音频", "视频", "音乐", "代码",
        }
        if _SEARCH_VERBS.search(ql):
            for cw in _CATEGORY_WORDS:
                if cw in ql:
                    return True  # "find image files" is a category search

        stripped = ql
        noise_tokens = IntentKeywords.FILE_OBJ_KWS['zh'] + IntentKeywords.FILE_OBJ_KWS['en'] + \
                       IntentKeywords.ASK_KWS['zh'] + IntentKeywords.ASK_KWS['en'] + \
                       IntentKeywords.EXPLICIT_COUNT_KWS['zh'] + IntentKeywords.EXPLICIT_COUNT_KWS['en'] + \
                       IntentKeywords.NOISE_TOKENS['zh'] + IntentKeywords.NOISE_TOKENS['en']
                       
        for tok in sorted(set(t for t in noise_tokens if t), key=len, reverse=True):
            stripped = stripped.replace(tok, " ")

        stripped = re.sub(r"""[\s\-_.,!?;:，。！？；：、/\\()\[\]{}<>"'`]+""", " ", stripped)
        stripped = stripped.replace("的", " ").replace("里", " ").replace("中", " ")
        stripped = " ".join(part for part in stripped.split() if part)
        if not stripped:
            return False

        if any("\u4e00" <= ch <= "\u9fff" for ch in stripped):
            return True

        tokens = [t for t in stripped.split() if len(t) >= 3]
        return len(tokens) >= 1

    @classmethod
    def _looks_like_file_content_analysis_query(cls, question: str) -> bool:
        q = str(question or "").strip()
        if not q:
            return False
        ql = q.lower()
        file_terms = IntentKeywords.FILE_OBJ_KWS['zh'] + IntentKeywords.FILE_OBJ_KWS['en']
        analysis_terms = IntentKeywords.CONTENT_KWS['zh'] + IntentKeywords.CONTENT_KWS['en'] + ["看下", "看看"]
        selection_terms = ["找", "找到", "筛", "筛出", "哪些", "所有", "全部", "相关", "find", "show", "list", "all", "relevant", "matching"]

        has_file_term = any(k in ql for k in file_terms)
        has_analysis_term = any(k in ql for k in analysis_terms)
        has_selection_term = any(k in ql for k in selection_terms)

        if not (has_file_term and has_analysis_term and has_selection_term):
            return False
            
        return bool(cls._extract_file_analysis_focus_query(question))

    @classmethod
    def _extract_file_analysis_focus_query(cls, question: str) -> Optional[str]:
        q = str(question or "").strip()
        if not q:
            return None
        ql = q.lower()

        focus = ""
        en_patterns = [
            r"(?:photos?|images?|pictures?)\s+of\s+(.+?)(?:\s+(?:and|that|to|where|who|which)\b|$)",
            r"(?:files?|documents?)\s+about\s+(.+?)(?:\s+(?:and|that|to|where|who|which)\b|$)",
            r"(?:about|related to)\s+(.+?)(?:\s+(?:and|that|to|where|who|which)\b|$)",
        ]
        for pat in en_patterns:
            m = re.search(pat, ql)
            if m:
                focus = str(m.group(1) or "").strip()
                if focus:
                    break
        if not focus:
            focus = ql

        noise_tokens = [
            "find", "all", "show", "me", "the", "a", "an", "my", "our", "their",
            "photo", "photos", "image", "images", "picture", "pictures",
            "file", "files", "document", "documents",
            "describe", "description", "explain", "analysis", "analyze",
            "summarize", "summary", "summaries", "recap", "conclude", "conclusion",
            "compare", "comparing", "comparison", "versus", "vs",
            "总结", "归纳", "概括", "讲了什么", "讲解", "查找", "寻找", "寻找关于",
            "关于", "查一下", "找下", "看看", "看下", "找", "的", "给", "跟我", "所有", "全部",
            "tell me about", "what he is doing in each", "what she is doing in each",
            "what they are doing in each", "what is happening in each",
            "in each", "for each", "each one", "each photo", "each image",
        ] + IntentKeywords.FILE_OBJ_KWS.get("zh", []) + IntentKeywords.FILE_OBJ_KWS.get("en", []) \
          + IntentKeywords.NOISE_TOKENS.get("zh", []) + IntentKeywords.NOISE_TOKENS.get("en", [])

        # Build boundaries: for english words, we don't want partial matches like 'in' inside 'find'
        for tok in sorted(set(noise_tokens), key=len, reverse=True):
            if not tok: continue
            if tok.isascii():
                pat = r'(?<![a-zA-Z])' + re.escape(tok) + r'(?![a-zA-Z])'
                focus = re.sub(pat, " ", focus, flags=re.IGNORECASE)
            elif tok in focus:
                focus = focus.replace(tok, " ")

        focus = " ".join(focus.split())
        
        has_cjk = bool(re.search(r'[\u4e00-\u9fff]', focus))
        min_len = 1 if has_cjk else 3
        
        return focus if len(focus) >= min_len else None

    @classmethod
    def looks_like_meta_followup_on_last_results(cls, question: str, prompt_language: Optional[str] = None) -> bool:
        q = (question or "").strip()
        if not q:
            return False
        ql = q.lower()
        if prompt_language == "zh":
            markers = ["结论", "总结", "归纳", "概括", "汇总", "讲一下", "说说看", "介绍下"]
            if not any(m in q for m in markers):
                return False
            return len(q) <= 36
        markers = [
            "conclusion", "recap", "tldr", "summarize", "summary", "overview",
            "wrap up", "wrap-up", "in short", "briefly", "key takeaways", "takeaways",
            "detail", "details", "more detail", "show me the detail", "show me more",
            "tell me more", "explain", "elaborate", "what's in", "what is in",
        ]
        if not any(m in ql for m in markers):
            return False
        words = [w for w in ql.split() if w]
        return len(words) <= 8

    @classmethod
    def looks_like_content_followup_on_prior_results(cls, question: str) -> bool:
        q = str(question or "").strip()
        if not q:
            return False
        ql = q.lower()
        q_compact = re.sub(r"\s+", "", q)

        has_prev_ref = cls._is_kw_match(ql, IntentKeywords.PREV_REF_KWS, "all")
        has_explicit_result_ref = bool(
            re.search(
                r"\b(these|those|them|the above|the previous|the listed)\b"
                r"|这些|那些|它们|他们|这几个|那几个|上面|以上|前面|其中|上述",
                ql,
                re.IGNORECASE,
            )
        )
        has_content_signal = bool(
            re.search(
                r"\b(summary|summarize|overview|recap|details?|more detail|inside|contents?|"
                r"what(?:'s|\s+is)\s+in|what\s+do\s+these\s+say|explain)\b"
                r"|内容|总结|概括|归纳|汇总|要点|详细|展开|讲什么|说什么|介绍|解释|里面",
                ql,
                re.IGNORECASE,
            )
        )
        has_registry_content_signal = any(
            kw and kw.lower() in ql
            for kw in (
                IntentKeywords.CONTENT_KWS.get("zh", [])
                + IntentKeywords.CONTENT_KWS.get("en", [])
            )
        )
        has_relation_signal = bool(
            re.search(
                r"\b(why|how\s+come|related|relevance|why\s+is|why\s+are)\b"
                r"|为什么|怎么会|为何|有关|相关|关系|关联|为什么会跟",
                ql,
                re.IGNORECASE,
            )
        )
        has_attribute_or_per_item_signal = bool(
            re.search(
                r"\b(for\s+each|in\s+each|each\s+one|each\s+video|each\s+file|respectively)\b"
                r"|\b(what|which|who|where)\b.{0,24}\b("
                r"animal|animals|person|people|brand|brands|company|companies|product|products|"
                r"place|places|location|locations|scene|scenes|object|objects|topic|topics|theme|themes"
                r")\b"
                r"|(?:什么|哪些|哪个|谁|哪位|哪里).{0,12}(动物|人|人物|品牌|公司|产品|地点|位置|场景|画面|内容|主题|东西)"
                r"|分别|各自|逐个|逐一|一个个|每个|每条|每段|都说明|说明一下|描述一下|介绍一下|解释一下",
                ql,
                re.IGNORECASE,
            )
        )
        is_short_bare_followup = (
            len(q_compact) <= 10
            and bool(
                re.search(
                    r"\b(summarize|summary|details?|more detail|go on|continue|and then|what next)\b"
                    r"|总结一下|总结下|概括一下|归纳一下|详细一点|详细说说|展开讲|继续说|继续讲|然后呢|接着呢|还有呢|后面呢",
                    ql,
                    re.IGNORECASE,
                )
            )
        )
        has_implicit_content_followup = bool(
            re.search(
                r"\b(what\s+does\s+it\s+say|what\s+is\s+in\s+it|tell\s+me\s+more\s+about\s+them)\b"
                r"|里面有什么|里面是(什么|啥)|说了什么|讲了什么|内容是(什么|啥)",
                ql,
                re.IGNORECASE,
            )
        )
        explicit_fresh_search = bool(
            re.search(
                r"^\s*\b(find|search|look\s+for|show\s+me|list|display|retrieve|what\s+files|which\s+files|do\s+i\s+have)\b",
                ql,
                re.IGNORECASE,
            )
            or any(
                q.startswith(prefix)
                for prefix in (
                    "我有哪些", "我有什么", "有哪些", "有什么", "有多少", "多少个", "多少份",
                    "列出", "显示", "查找", "搜索", "搜一下", "搜下", "查一下", "查下",
                    "找一下", "找下", "帮我找", "帮我搜", "帮我查",
                )
            )
        )

        if explicit_fresh_search and not has_prev_ref and not has_explicit_result_ref:
            return False

        if (has_prev_ref or has_explicit_result_ref) and (
            has_content_signal
            or has_registry_content_signal
            or has_relation_signal
            or has_attribute_or_per_item_signal
        ):
            return True
        if has_implicit_content_followup:
            return True
        if is_short_bare_followup:
            return True
        return False
