"""
EntitySearchExpert — agent for bare entity queries (≤3 words) → search.

Replaces fp3 regex in intent_analyzer.py with a structured classifier.
Handles queries like "person name", "tencent", "anker", "invoice" — short queries
that are almost certainly entity searches.
"""
from __future__ import annotations

import re
import logging
from typing import Optional, List, Dict, Set

logger = logging.getLogger(__name__)


class EntitySearchExpert:
    """
    Detects bare entity queries (≤3 words, no instruction verbs) and routes to search.
    """

    _NO_ENTITY_KWS = frozenset({
        "summary", "summarize", "count", "list", "show", "open", "help", "how many",
        "selected", "seleted", "chosen",  # Let SelectionExpert handle these
        "有哪些", "多少", "列出", "hello", "hi", "hey", "thanks", "你好", "谢谢",
        # English function words that should not be treated as entity searches
        "find", "tell", "get", "retrieve", "where", "when", "what", "why", "who",
        "which", "how", "please", "can", "could", "would", "should",
    })

    _PREV_REF_MARKERS = frozenset({
        "conclusion", "recap", "summarize", "summary", "tldr", "them", "these",
        "those", "结论", "总结", "归纳", "概括", "它们", "这些", "那些",
    })

    @classmethod
    def is_bare_entity(
        cls,
        query: str,
        *,
        active_paths: Optional[List[str]] = None,
    ) -> bool:
        """Check if query is a bare entity search (≤3 words, no commands)."""
        qn = (query or "").strip()
        ql = qn.lower()
        if not qn:
            return False

        # Calculate effective word length (CJK chars count as 0.6 words each)
        words = [w for w in qn.split() if w]
        cjk_chars = len(re.findall(r'[\u4e00-\u9fff]', qn))
        effective_length = len(words) if cjk_chars == 0 else (len(words) + cjk_chars * 0.6)

        if effective_length > 3.5:
            return False

        # Exclude instruction keywords
        if any(kw in ql for kw in cls._NO_ENTITY_KWS):
            return False

        # Exclude previous-result references
        if any(m in ql for m in cls._PREV_REF_MARKERS):
            return False

        # Don't trigger with active_paths (let SelectionExpert handle)
        if active_paths:
            return False

        return True

    @classmethod
    def to_intent(cls, query: str) -> dict:
        """Convert a bare entity query to a search intent."""
        logger.debug(f"[entity_search] bare entity → search query_chars={len(query or '')}")
        return {"action": "search", "params": {"query": query}, "confidence": 0.85}


class FilenameExpert:
    """
    Detects bare filename queries (e.g. "sample-audio.wav", "test.pdf") and routes to search.
    """

    _KNOWN_EXTS = frozenset({
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tiff",
        ".pdf", ".txt", ".md", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
        ".mp3", ".mp4", ".wav", ".m4a", ".mov", ".avi", ".zip", ".rar",
        ".py", ".js", ".ts", ".json", ".csv", ".xml", ".html",
    })

    @classmethod
    def is_bare_filename(cls, query: str) -> bool:
        """Check if query is a bare filename (e.g. 'sample-audio.wav')."""
        qn = (query or "").strip()
        if not qn:
            return False
        words = qn.split()
        has_ext = any(qn.lower().endswith(ext) for ext in cls._KNOWN_EXTS)
        return has_ext and len(words) <= 2

    @classmethod
    def to_intent(cls, query: str) -> dict:
        """Convert a bare filename to a search intent."""
        logger.debug(f"[filename_expert] bare filename → search query_chars={len(query or '')}")
        return {"action": "search", "params": {"query": query.strip()}, "confidence": 0.95}


class CategoryListExpert:
    """
    Detects "find/show/list/get + file-type" queries and routes them to
    category-scoped SEARCH, not count(category).

    Examples:
      - "find my resume"        → search(category=resume)
      - "show photos"           → search(category=image)
      - "find all csv files"    → search(category=data)

    Explicit counting remains handled elsewhere:
      - "how many resumes do I have" → CountExpert / validator
      - generic "what files do I have" style inventory → count(all)
    """

    _FIND_VERBS_EN = re.compile(
        r'^\s*(?:(?:please|kindly)\s+)?'
        r'(?:(?:can|could|would)\s+you\s+)?'
        r'(?:(?:i|we)\s+(?:want|need|would\s+like)\s+(?:you\s+to\s+)?|(?:help\s+me\s+(?:to\s+)?)?)?'
        r'(find|show|list|get|search(?:\s+for)?|look\s+for|give\s+me|display|retrieve|browse|fetch)\b',
        re.IGNORECASE,
    )
    _ZH_FIND_PREFIXES = ("找", "给我", "显示", "列出", "获取", "查找", "展示", "看看", "查看", "查一下", "搜", "搜索")

    _CONTENT_OPERATION_RE = re.compile(
        r'\b(describe|inside|content|detail|details|analyze|analysis|explain|explanation|'
        r'summary|summarize|what\s+(is|are|in)|inside\s+the|in\s+the\s+(file|doc))\b',
        re.IGNORECASE,
    )
    _TOPIC_QUALIFIER_RE = re.compile(
        r'\b(of|with|containing|contains?|about|regarding|related\s+to|mentioning|'
        r'that\s+mention|that\s+contain|featuring|showing|depicting|involving)\b'
        r'|关于|有关|相关|包含|含有|带有|里面有|出现|展示|显示',
        re.IGNORECASE,
    )
    _DOCUMENT_TARGET_RE = re.compile(
        r'\b(?:papers?|articles?|documents?|docs?|documentation|pdfs?|reports?|manuals?|guides?|'
        r'publications?|theses|thesis|whitepapers?|text\s+files?)\b'
        r'|论文|文章|文档|报告|资料|手册|指南|PDF|pdf|白皮书|文本|文字',
        re.IGNORECASE,
    )
    _MEDIA_TOPIC_RE = re.compile(
        r'\b(?:audio|video|speech|music|sound|voice|recording|recordings|'
        r'image|photo|picture|diagram|screenshot|visual)\b'
        r'|音频|视频|语音|声音|音乐|录音|图片|照片|图像|截图|图表|架构图',
        re.IGNORECASE,
    )

    _TOKEN_TO_CAT: Dict = {
        "pdf": "document", "pdfs": "document", "docx": "document", "doc": "document", "docs": "document",
        "document": "document", "documents": "document",
        "txt": "document", "txts": "document", "md": "document", "rtf": "document", "odt": "document",
        "文档": "document", "文本": "document", "资料": "document",
        "image": "image", "images": "image", "photo": "image", "photos": "image",
        "picture": "image", "pictures": "image",
        "jpg": "image", "jpeg": "image", "png": "image", "gif": "image",
        "screenshot": "image", "screenshots": "image",
        "图片": "image", "照片": "image", "截图": "image", "图像": "image",
        "csv": "data", "excel": "data", "xlsx": "data", "xls": "data",
        "spreadsheet": "data", "spreadsheets": "data", "dataset": "data",
        "data": "data", "worksheet": "data", "worksheets": "data",
        "workbook": "data", "workbooks": "data", "table": "data", "tables": "data",
        "表格": "data", "数据表": "data", "数据集": "data", "数据": "data", "工作表": "data",
        "ppt": "presentation", "pptx": "presentation",
        "slide": "presentation", "slides": "presentation",
        "presentation": "presentation", "presentations": "presentation",
        "幻灯片": "presentation",
        "mp4": "video", "mov": "video", "mkv": "video", "avi": "video", "webm": "video", "m4v": "video",
        "video": "video", "videos": "video", "movie": "video", "movies": "video", "clip": "video", "clips": "video",
        "视频": "video", "录像": "video", "影片": "video",
        "mp3": "audio", "wav": "audio", "m4a": "audio", "flac": "audio", "aac": "audio", "ogg": "audio",
        "audio": "audio", "audios": "audio", "recording": "audio", "recordings": "audio",
        "podcast": "audio", "podcasts": "audio", "song": "audio", "songs": "audio", "music": "audio",
        "音频": "audio", "录音": "audio", "音乐": "audio", "歌曲": "audio",
        "resume": "resume", "resumes": "resume", "简历": "resume",
        "manual": "manual", "manuals": "manual", "手册": "manual", "说明书": "manual",
        "report": "report", "reports": "report", "报告": "report",
        "paper": "paper", "papers": "paper", "article": "paper", "articles": "paper",
        "publication": "paper", "publications": "paper", "thesis": "paper", "theses": "paper",
        "论文": "paper", "文献": "paper",
        "book": "book", "books": "book", "书": "book", "书籍": "book",
        "invoice": "invoice", "invoices": "invoice", "发票": "invoice",
        "receipt": "invoice", "receipts": "invoice", "收据": "invoice",
        "code": "code", "代码": "code", "源码": "code",
    }

    _GENERIC_FILE_LIST = re.compile(
        r'\b(my\s+files?|all\s+(my\s+)?files?|all\s+(my\s+)?documents?|my\s+documents?'
        r'|all\s+files?|all\s+documents?'
        r'|我的文件|所有文件|所有我的文件|全部文件)\b',
        re.IGNORECASE,
    )
    _EN_INVENTORY_ASK = re.compile(
        r'\b((what|which)\s+(?:my\s+)?|do\s+(i|we)\s+have|are\s+there)\b',
        re.IGNORECASE,
    )
    _ZH_INVENTORY_ASK = re.compile(
        r'(我(都)?有(哪些|什么)|有哪些|都有哪些|有什么|有没有)',
        re.IGNORECASE,
    )
    _EN_INVENTORY_SCOPE_RE = re.compile(
        r'\b(all|my|mine|every|do\s+(i|we)\s+have|what|which)\b',
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
    _EN_FILE_SCOPE_RE = re.compile(
        r'\b(files?|documents?|docs?|images?|photos?|pictures?|screenshots?|videos?|audios?|'
        r'recordings?|podcasts?|songs?|music|media|worksheets?|spreadsheets?|tables?)\b',
        re.IGNORECASE,
    )
    _EN_INVENTORY_NOISE_RE = re.compile(
        r'\b(find|show|list|get|search|give|display|retrieve|browse|fetch|what|which|all|'
        r'every|my|mine|me|the|a|an|do|does|did|i|we|have|are|there|please|can|you)\b',
        re.IGNORECASE,
    )
    _ZH_INVENTORY_NOISE = (
        "给我看看", "帮我看看", "给我看", "查看一下", "查一下", "看一下", "看下", "看看",
        "列出", "显示", "展示", "获取", "查找", "搜索", "找一下", "找", "搜一下", "搜",
        "所有", "全部", "我的", "我有", "我都", "都有哪些", "有哪些", "有什么", "文件", "文档",
        "资料", "图片", "照片", "截图", "视频", "音频", "录音", "音乐", "表格", "数据", "工作表",
    )

    @classmethod
    def _infer_requested_media_type(cls, query: str) -> str:
        ql = (query or "").lower()
        has_video = bool(cls._VIDEO_SCOPE_RE.search(ql))
        has_audio = bool(cls._AUDIO_SCOPE_RE.search(ql))
        if has_video and not has_audio:
            return "video"
        if has_audio and not has_video:
            return "audio"
        return ""

    @classmethod
    def _category_aliases(cls, category: str) -> Set[str]:
        aliases: Set[str] = set()
        normalized_category = str(category or "").strip().lower()
        if normalized_category:
            aliases.add(normalized_category)
        for token, mapped in cls._TOKEN_TO_CAT.items():
            if mapped == category:
                aliases.add(str(token or "").strip().lower())
        try:
            from core.retrieval.category_engine import build_dynamic_category_aliases

            aliases.update(build_dynamic_category_aliases(normalized_category))
        except Exception:
            pass
        return {alias for alias in aliases if alias}

    @classmethod
    def _category_token_is_negated(cls, query: str, token: str) -> bool:
        q = str(query or "").lower()
        raw_token = str(token or "").strip().lower()
        if not q or not raw_token:
            return False
        if re.search(r'[\u4e00-\u9fff]', raw_token):
            return bool(
                re.search(
                    r"(?:不要|别要|别|不(?:要|看|找|含|包含|包括)?|不是|非|排除|剔除|去掉|除外|除了).{0,8}"
                    + re.escape(raw_token),
                    q,
                )
            )
        return bool(
            re.search(
                r"\b(?:not|no|without|exclude|excluding|except|omit|skip|avoid|non)\b"
                r"(?:\s+\w+){0,4}\s+"
                + re.escape(raw_token)
                + r"s?\b",
                q,
                re.IGNORECASE,
            )
        )

    @classmethod
    def _inventory_remainder(cls, query: str, category_aliases: Set[str]) -> str:
        stripped = str(query or "").lower()
        stripped = cls._FIND_VERBS_EN.sub(" ", stripped, count=1)
        stripped = cls._EN_INVENTORY_ASK.sub(" ", stripped)
        stripped = cls._EN_INVENTORY_SCOPE_RE.sub(" ", stripped)
        stripped = cls._EN_FILE_SCOPE_RE.sub(" ", stripped)
        stripped = cls._EN_INVENTORY_NOISE_RE.sub(" ", stripped)

        for token in sorted(category_aliases, key=len, reverse=True):
            if not token:
                continue
            if re.search(r'[\u4e00-\u9fff]', token):
                stripped = stripped.replace(token, " ")
            else:
                stripped = re.sub(rf'\b{re.escape(token)}\b', " ", stripped)

        for token in cls._ZH_INVENTORY_NOISE:
            stripped = stripped.replace(token, " ")

        stripped = re.sub(r'[\W_]+', ' ', stripped)
        leftovers = [part for part in stripped.split() if part]
        return " ".join(leftovers)

    @classmethod
    def _looks_like_broad_inventory_query(
        cls,
        query: str,
        *,
        category: str,
        extra_aliases: Optional[Set[str]] = None,
    ) -> bool:
        q = str(query or "").strip()
        ql = q.lower()
        if not q:
            return False

        has_inventory_scope = bool(
            cls._EN_INVENTORY_ASK.search(ql)
            or cls._ZH_INVENTORY_ASK.search(ql)
            or cls._EN_INVENTORY_SCOPE_RE.search(ql)
            or any(tok in q for tok in ("所有", "全部", "我的", "查看", "看看", "给我看", "列出"))
        )
        if not has_inventory_scope:
            return False

        category_aliases = cls._category_aliases(category)
        if extra_aliases:
            category_aliases.update(str(alias or "").strip().lower() for alias in extra_aliases if alias)
        return not cls._inventory_remainder(query, category_aliases)

    @classmethod
    def _looks_like_bare_category_file_phrase(cls, query: str, *, category: str) -> bool:
        q = str(query or "").strip()
        ql = q.lower()
        if not q:
            return False
        has_file_scope = bool(
            cls._EN_FILE_SCOPE_RE.search(ql)
            or any(tok in q for tok in ("文件", "文档", "资料", "表格", "工作表"))
        )
        if not has_file_scope:
            return False
        return not cls._inventory_remainder(q, cls._category_aliases(category))

    @classmethod
    def _looks_like_category_semantic_file_phrase(cls, query: str, *, category: str) -> bool:
        ql = str(query or "").strip().lower()
        if not ql:
            return False
        if category == "image":
            return bool(
                re.search(
                    r"\b(?:all|every|my|mine)\s+(?:images?|photos?|pictures?|screenshots?)\b"
                    r"|\b(?:images?|photos?|pictures?|screenshots?)\s+(?:of|with|containing)\b",
                    ql,
                    re.IGNORECASE,
                )
            )
        return False

    @classmethod
    def _looks_like_folder_category_listing(cls, query: str) -> bool:
        q = str(query or "").strip()
        ql = q.lower()
        if not q:
            return False
        return bool(
            re.search(
                r"\b(?:in|inside|under|within|from)\s+(?:the\s+)?(?:folder|directory|dir)\b",
                ql,
                re.IGNORECASE,
            )
            or any(tok in q for tok in ("目录里", "目录中的", "目录下", "文件夹里", "文件夹中的", "文件夹下"))
        )

    @classmethod
    def analyze(cls, query: str, has_content_qualifier: bool = False) -> Optional[dict]:
        """
        Check if query is a "find + file-type" listing request.
        
        Returns intent dict if matched, None otherwise.
        """
        ql = (query or "").lower()

        # Let the specialized media expert own timestamp / scene / transcript
        # generic category search for videos.
        try:
            from core.intent.media_query_expert import MediaQueryExpert
            if MediaQueryExpert.analyze(query, last_results=None, llm_service=None) is not None:
                logger.info("[category_list] skip: specialized media query detected")
                return None
        except Exception:
            pass

        # Check for explicit listing / retrieval ask
        has_find = (
            bool(cls._FIND_VERBS_EN.match(ql))
            or ql.startswith(cls._ZH_FIND_PREFIXES)
        )
        has_inventory_ask = bool(
            cls._EN_INVENTORY_ASK.search(ql)
            or cls._ZH_INVENTORY_ASK.search(ql)
        )
        has_listing_request = has_find or has_inventory_ask

        has_topic_qualifier = bool(cls._TOPIC_QUALIFIER_RE.search(ql))

        # Exclude true content-operation queries, but keep topical constraints
        # such as "images of my dog" / "PDFs about budget" as category-scoped
        # searches instead of broad inventory.
        if cls._CONTENT_OPERATION_RE.search(ql) and not has_topic_qualifier:
            return None

        if has_content_qualifier and not has_topic_qualifier:
            return None

        # Token-level category matching
        en_tokens = set(re.sub(r'[\u4e00-\u9fff\W]', ' ', ql).split())
        has_cjk = bool(re.search(r'[\u4e00-\u9fff]', ql))

        matched_category = ""
        for token, cat in cls._TOKEN_TO_CAT.items():
            is_cjk_token = bool(re.search(r'[\u4e00-\u9fff]', token))
            if is_cjk_token:
                if token in ql and not cls._category_token_is_negated(query, token):
                    matched_category = cat
                    logger.info(f"[category_list] zh-token='{token}' → search(category={cat!r})")
                    break
            else:
                if token in en_tokens and not cls._category_token_is_negated(query, token):
                    matched_category = cat
                    logger.info(f"[category_list] en-token='{token}' → search(category={cat!r})")
                    break

        if matched_category:
            if (
                matched_category in {"audio", "video", "image"}
                and (has_topic_qualifier or matched_category == "image")
                and cls._DOCUMENT_TARGET_RE.search(query)
                and cls._MEDIA_TOPIC_RE.search(query)
            ):
                logger.info(
                    "[category_list] skip media category: media word is topic of document retrieval"
                )
                return None
            if (
                not has_listing_request
                and not cls._looks_like_bare_category_file_phrase(query, category=matched_category)
                and not cls._looks_like_category_semantic_file_phrase(query, category=matched_category)
            ):
                return None
            params: Dict[str, str] = {"query": query, "category": matched_category}
            if matched_category == "audio/video":
                media_type = cls._infer_requested_media_type(query)
                if media_type in {"audio", "video"}:
                    params["media_type"] = media_type
            if (
                cls._looks_like_broad_inventory_query(query, category=matched_category)
                or (
                    cls._looks_like_folder_category_listing(query)
                    and not has_topic_qualifier
                )
            ):
                params["_inventory_mode"] = "category"
            return {
                "action": "search",
                "params": params,
                "confidence": 0.9,
            }

        if not has_listing_request:
            return None

        # Registry/leaf taxonomy names are often topical phrases rather than
        # filterable file buckets (for example "business plan").  Leave those
        # to the LLM/search prompt so the phrase stays in params.query instead
        # of becoming a hard category filter that can erase recall.

        # Generic file listing (my files / all files) still routes to count(all),
        # because that is a broad inventory request rather than a category retrieval.
        if cls._GENERIC_FILE_LIST.search(ql) and not has_content_qualifier:
            logger.info(f"[category_list] generic file list → count(all)")
            return {"action": "count", "params": {"category": "all"}, "confidence": 0.9}

        return None
