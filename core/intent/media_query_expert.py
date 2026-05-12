from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _generate_media_llm_text(llm_service: any, prompt: str) -> str:
    if hasattr(llm_service, "run_local_llm"):
        return llm_service.run_local_llm(prompt)
    if hasattr(llm_service, "generate"):
        return llm_service.generate(prompt, temperature=0.1, max_tokens=256)
    raise AttributeError("LLM service does not support run_local_llm or generate")


class MediaQueryExpert:
    """
    Detects time-based media queries and extracts:
      - target timestamp (seconds)
      - target type (audio_content / video_visual / video_audio)
      - optional file hint
    """

    # ── Time extraction patterns ──────────────────────────────────────────

    # Matches "at 30 seconds", "around 1:20", "at 45s", "at minute 2"
    _EN_TIME_PAT = re.compile(
        r'(?:(?:at|around|near|about|approximately|in|within|on)\s+)?'
        r'(?:'
        r'(\d{1,3}):(\d{2})'           # MM:SS format
        r'|(?<![A-Za-z])(\d+)\s*(?:seconds?|secs?|s)\b'  # Ns / N seconds
        r'|(?<![A-Za-z])(\d+)\s*(?:minutes?|mins?|m)\b'  # Nm / N minutes
        r'|(?<![A-Za-z])(\d+)\s*(?:hours?|hrs?|h)\b'     # Nh / N hours
        r')',
        re.IGNORECASE,
    )

    # Matches "from 10 to 20 seconds", "between 1:00 and 2:00"
    _EN_RANGE_PAT = re.compile(
        r'(?:from|between)\s+'
        r'(?:(\d{1,3}):(\d{2})|(\d+)\s*(?:seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)?)\s*'
        r'(?:to|and|-)\s*'
        r'(?:(\d{1,3}):(\d{2})|(\d+)\s*(?:seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)?)',
        re.IGNORECASE,
    )

    _EN_UNIT_RANGE_PAT = re.compile(
        r'(?:from|between)\s+'
        r'(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\s*'
        r'(?:to|and|-)\s*'
        r'(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b',
        re.IGNORECASE,
    )

    _EN_SHARED_UNIT_RANGE_PAT = re.compile(
        r'(?:from|between)\s+'
        r'(\d+)\s*(?:to|and|-)\s*(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b',
        re.IGNORECASE,
    )

    _CLOCK_TIME_PAT = re.compile(r'(\d{1,2}:\d{2}(?::\d{2})?)')
    _CLOCK_RANGE_PAT = re.compile(
        r'(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:to|and|-|到|至)\s*(\d{1,2}:\d{2}(?::\d{2})?)',
        re.IGNORECASE,
    )

    # Matches "first 20 minutes", "opening 30 seconds"
    _EN_PREFIX_RANGE_PAT = re.compile(
        r'\b(?:first|initial|opening|beginning(?:\s+part)?|start(?:ing)?\s+(?:part\s+of\s+)?(?:the\s+)?)\s+'
        r'(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b',
        re.IGNORECASE,
    )

    _EN_BEGINNING_POINT_PAT = re.compile(
        r'\b(?:(?:at|around|near|about)\s+(?:the\s+)?(?:beginning|start|opening|intro)|'
        r'(?:the\s+)?(?:beginning|opening|intro)|(?:the\s+)?start\s+of)\b',
        re.IGNORECASE,
    )

    _ZH_TIME_PAT = re.compile(
        r'第\s*(?:(\d+)\s*分钟?\s*)?(\d+)\s*秒'
        r'|第\s*(\d+)\s*(?:分钟|分)'
        r'|(?<![A-Za-z])(\d+)\s*(?:秒|s(?=[^A-Za-z0-9_]|$))\s*(?:左右|附近|处|的)?',
        re.IGNORECASE,
    )

    _ZH_RANGE_PAT = re.compile(
        r'(?:第?\s*(\d+)\s*(?:秒|s)?\s*(?:到|至|-)\s*第?\s*(\d+)\s*(?:秒|s))',
    )

    _ZH_MINUTE_RANGE_PAT = re.compile(
        r'(?:第?\s*(\d+)\s*(?:分钟|分|份)?\s*(?:到|至|-)\s*第?\s*(\d+)\s*(?:分钟|分|份))(?:之间)?',
        re.IGNORECASE,
    )

    _ZH_PREFIX_RANGE_PAT = re.compile(
        r'(?:前|开头|开始(?:的)?|最开始的)\s*(\d+)\s*(小时|分钟|分|秒|s)',
        re.IGNORECASE,
    )

    _EN_MINUTE_MARK_PAT = re.compile(
        r'(\d+)\s*(?:-| )?(?:minute|minutes|min)\s+mark\b',
        re.IGNORECASE,
    )

    _ZH_MINUTE_AROUND_PAT = re.compile(
        r'(\d+)\s*(?:分钟|分)\s*(?:左右|附近|处)',
        re.IGNORECASE,
    )

    _ZH_HOUR_PAT = re.compile(
        r'(?:第\s*)?(\d+)\s*小时'
        r'(?:\s*(\d+)\s*(?:分钟|分))?'
        r'(?:\s*(\d+)\s*秒)?'
        r'(?:左右|附近|处|的)?',
        re.IGNORECASE,
    )

    # ── Media type detection ──────────────────────────────────────────────

    # Audio indicators in the query
    _AUDIO_INDICATOR = re.compile(
        r'\b(audio|recording|podcast|song|music|说什么|说了什么|在说|讲了什么|说的内容)\b'
        r'|音频|录音|播客|歌曲',
        re.IGNORECASE,
    )

    _VIDEO_VISUAL_INDICATOR = re.compile(
        r'\b(scene|frame|visual|image|screen|picture|view|look|watch|see|saw|seen|show|shown|showing|doing|happens|happening)\b'
        r'|视频.*画面|画面|场景|镜头|屏幕|做什么|在做什么|看什么|干什么',
        re.IGNORECASE,
    )

    # Video general
    _VIDEO_INDICATOR = re.compile(
        r'\b(video|movie|clip)\b|视频|影片|短片',
        re.IGNORECASE,
    )

    # ── File hint extraction ──────────────────────────────────────────────

    # Extract filename with common media extensions.
    # ① Allows a space followed by a digit-led segment so that filenames like
    # ② Uses (?=[^a-zA-Z0-9]|$) instead of \b: Python 3's \b is Unicode-aware
    #   detection after the extension.  The ASCII-only lookahead is safe here.
    _FILE_HINT_PAT = re.compile(
        r'[\w\-.\u4e00-\u9fff]+(?:\s+\d[\w\-.]*)*\.(?:mp3|wav|flac|aac|ogg|m4a|wma|aiff|ape'
        r'|mp4|m4v|avi|mov|mkv|webm|flv|wmv|ts)(?=[^a-zA-Z0-9]|$)',
        re.IGNORECASE,
    )

    # ── Core detection: does this query contain a time reference? ──────

    _HAS_TIME_SIGNAL = re.compile(
        r'\b(?:at|around|near|about|from|between|within|on)\s+\d'
        r'|(?<![A-Za-z])\d+\s*(?:seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)(?=[^A-Za-z0-9_]|$)'
        r'|(?<![A-Za-z])\d+\s*(?:-| )?(?:minute|minutes|min)\s+mark\b'
        r'|\b(?:(?:at|around|near|about)\s+(?:the\s+)?(?:beginning|start|opening|intro)|'
        r'(?:the\s+)?(?:beginning|opening|intro)|(?:the\s+)?start\s+of)\b'
        r'|(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)'
        r'|第\s*\d+\s*(?:秒|分|小时)'
        r'|第?\s*\d+\s*(?:秒|s)?\s*(?:到|至|-)\s*第?\s*\d+\s*(?:秒|s)'
        r'|\d+\s*秒'
        r'|(?<![A-Za-z])\d+\s*s(?=[^A-Za-z0-9_]|$)\s*(?:左右|附近|处|的)?'
        r'|\d+\s*(?:分钟|分)'
        r'|\d+\s*小时',
        re.IGNORECASE,
    )

    _CALENDAR_YEAR_REFERENCE_RE = re.compile(
        r'\b(?:from|between|since|in|by|before|after|until|through)\s+(?:19|20)\d{2}\b'
        r'|\b(?:19|20)\d{2}\s*(?:to|and|-|through|until)\s*(?:now|present|(?:19|20)\d{2})\b',
        re.IGNORECASE,
    )

    # Queries asking to *find* a time or scene
    _HAS_MEDIA_SEARCH_SIGNAL = re.compile(
        r'\b(where|when|which second|what second)\b'
        r'|第几(?:秒|分)|哪些(?:视频|音频)|有没有出现|有出现|画面中[有]?没有|视频里[有]?没有',
        re.IGNORECASE,
    )

    # Media context: the query must also mention audio/video or asking about content
    _HAS_MEDIA_SIGNAL = re.compile(
        r'\b(audio|video|recording|podcast|song|music|clip|movie'
        r'|happens?|happening|scene|frame|said|saying|talking|discussing|speaking|playing)\b'
        r'|做什么|在做什么|说什么|说了|在说|画面|场景|内容|讲了|音频|视频|录音|播客',
        re.IGNORECASE,
    )

    # Content-query signal: user asks about WHAT is in the media (no timestamp needed)
    _HAS_MEDIA_CONTENT_RE = re.compile(
        r'(?:音频|视频|录音|播客).*?(?:提到|讲到|说到|关于|内容|讲的|说的|提及)'
        r'|(?:提到|讲到|说到|关于|内容).*?(?:音频|视频|录音)'
        r'|\b(?:audio|video|recording|podcast)\b.*?\b(?:mention|discuss|say|talk|about|content|contain)\b'
        r'|\b(?:mention|discuss|say|talk|about|content|contain)\b.*?\b(?:audio|video|recording|podcast)\b'
        r'|(?:音频|视频|录音).{0,24}(?:里|里面|中).{0,24}(?:提到|讲到|说到|关于|内容|讲的|说的|提及)'
        r'|(?:提到|讲到|说到|关于|内容).{0,24}(?:音频|视频|录音).{0,24}(?:里|里面|中)',
        re.IGNORECASE,
    )

    _SUMMARY_SIGNAL_RE = re.compile(
        r'\b(summary|summarize|overview|recap|key\s+takeaways?|main\s+points?)\b'
        r'|\bwhat\s+(?:is|was)\s+discussed\b'
        r'|\bwhat\s+(?:is|was)\s+(?:being\s+)?described\b'
        r'|\btell\s+me\s+about\b'
        r'|总结|概括|归纳|概要|要点|主要内容|讲了什么|在讲什么|讨论了什么|描述了什么|在描述什么',
        re.IGNORECASE,
    )

    _DURATION_QUERY_RE = re.compile(
        r'\b(?:how\s+long|duration|runtime|run\s*time|length|total\s+(?:time|duration))\b'
        r'|(?:时长|总时长|时间多长|多长时间|有多长|多久|几分钟|多少分钟|多少秒)',
        re.IGNORECASE,
    )

    # Explicit "find/search/list/show ... video/audio" should stay on the file-search
    # pipeline. MediaQueryExpert is reserved for content/time questions about media.
    _EXPLICIT_MEDIA_FILE_SEARCH_RE = re.compile(
        r'^\s*(?:'
        r'(?:find|search|look\s+for|show(?:\s+me)?|list|display|browse|retrieve|locate|get\s+me|give\s+me)\b'
        r'|(?:look\s+up)\b'
        r'|(?:查找|搜索|搜一下|查一下|帮我找|找一下|找到|找|调出|列出|显示)'
        r').*?(?:'
        r'\b(?:video|videos|audio|audios|recording|recordings|clip|clips|movie|movies|podcast|podcasts|song|songs|wav|mp3|m4a|flac|aac|ogg|mp4|mov|mkv|avi|webm)\b'
        r'|视频|音频|录音|录屏|录像|影片|短片'
        r')',
        re.IGNORECASE,
    )

    _DOCUMENT_RETRIEVAL_WITH_MEDIA_TOPIC_RE = re.compile(
        r'^\s*(?:from\s+scratch,?\s*)?'
        r'(?:find|search|look\s+for|look\s+up|show\s+me|list|retrieve|locate|get\s+me)\b'
        r'.{0,80}\b(?:papers?|articles?|documents?|docs?|documentation|pdfs?|reports?|manuals?|guides?|publications?|theses|thesis|text\s+files?)\b'
        r'.{0,80}\b(?:audio|video|speech|music|sound|image|photo|picture|diagram|screenshot|visual)\b'
        r'|^\s*(?:from\s+scratch,?\s*)?'
        r'(?:find|search|look\s+for|look\s+up|show\s+me|list|retrieve|locate|get\s+me)\b'
        r'.{0,80}\b(?:audio|video|speech|music|sound|image|photo|picture|diagram|screenshot|visual)\b'
        r'.{0,80}\b(?:papers?|articles?|documents?|docs?|documentation|pdfs?|reports?|manuals?|guides?|publications?|theses|thesis|text\s+files?)\b'
        r'|^\s*(?:重新|从头|另外|换个)?\s*(?:找|搜索|查找|帮我找|列出|显示)'
        r'.{0,80}(?:论文|文章|文档|报告|资料|pdf|PDF)'
        r'.{0,80}(?:音频|视频|语音|声音|音乐|图片|照片|图像|截图|图表|架构图)'
        r'|^\s*(?:重新|从头|另外|换个)?\s*(?:找|搜索|查找|帮我找|列出|显示)'
        r'.{0,80}(?:音频|视频|语音|声音|音乐|图片|照片|图像|截图|图表|架构图)'
        r'.{0,80}(?:论文|文章|文档|报告|资料|pdf|PDF)',
        re.IGNORECASE,
    )

    _MEDIA_OPERATION_RE = re.compile(
        r'\b(?:'
        r'export|extract|clip|cut|trim|crop|save|download|convert|transcode|'
        r'screenshot|thumbnail|frame|transcribe|caption|subtitle|'
        r'duration|runtime|length|how\s+long|'
        r'what\s+(?:happens?|is\s+(?:shown|said|heard)|can\s+(?:i\s+)?(?:see|hear))|'
        r'what\s+does\s+(?:it|this|that|the\s+(?:audio|video|recording|clip))\s+(?:say|show|contain)|'
        r'what\s+is\s+(?:said|shown|heard|visible)|'
        r'between|from|first|last|around|at'
        r')\b'
        r'|导出|截取|剪辑|裁剪|保存|下载|转换|转码|截图|封面|缩略图|画面|帧|转写|字幕|'
        r'时长|总时长|时间多长|多长时间|多久|'
        r'第\s*\d+\s*(?:秒|分|分钟)|前\s*\d+\s*(?:秒|分|分钟)|'
        r'(?:从|第)?\s*\d+\s*(?:秒|分|分钟|s|m)?\s*(?:到|至|-)\s*(?:第)?\s*\d+\s*(?:秒|分|分钟|s|m)|'
        r'在\s*\d+\s*(?:秒|分|分钟).{0,12}(?:说|讲|展示|出现|画面|听到|看到)',
        re.IGNORECASE,
    )

    # Very broad English questions like "what is in the video about cats" are
    # too ambiguous to be treated as a deterministic media-content intent.
    _AMBIGUOUS_GENERIC_MEDIA_QUESTION = re.compile(
        r'^\s*what\s+is\s+in\s+(?:the\s+)?(?:audio|video|recording|podcast)\b',
        re.IGNORECASE,
    )
    _GENERIC_MEDIA_OVERVIEW_QUESTION = re.compile(
        r'^\s*(?:tell(?:\s+me)?\s+about|describe|summarize|summary\s+of|overview\s+of)?\s*'
        r'(?:the\s+|this\s+|that\s+|selected\s+|current\s+|a\s+|an\s+)*'
        r'(?:audio|video|recording|podcast|clip|media)(?:\s+file)?\s*'
        r'(?:(?:content|contents|about|contain|contains|inside|overview|summary)\s*)?[?.!]*\s*$'
        r'|^\s*what\s+(?:is|are)\s+(?:in|inside)\s+'
        r'(?:the\s+|this\s+|that\s+|selected\s+|current\s+|a\s+|an\s+)*'
        r'(?:audio|video|recording|podcast|clip|media)(?:\s+file)?\s*[?.!]*\s*$'
        r'|^\s*what\s+(?:can\s+be\s+heard|can\s+i\s+hear)\s+(?:in\s+)?'
        r'(?:the\s+|this\s+|that\s+|selected\s+|current\s+|a\s+|an\s+)*'
        r'(?:audio|video|recording|podcast|clip|media)(?:\s+file)?\s*[?.!]*\s*$'
        r'|^\s*what\s+does\s+(?:the\s+|this\s+|that\s+|selected\s+|current\s+|a\s+|an\s+)*'
        r'(?:audio|video|recording|podcast|clip|media)(?:\s+file)?\s+contain\s*[?.!]*\s*$'
        r'|^\s*(?:what\s+is\s+)?(?:the\s+)?content\s+of\s+'
        r'(?:the\s+|this\s+|that\s+|selected\s+|current\s+|a\s+|an\s+)*'
        r'(?:audio|video|recording|podcast|clip|media)(?:\s+file)?\s*[?.!]*\s*$'
        r'|^\s*(?:介绍一下|讲讲|概括|总结|概述)?\s*'
        r'(?:这个|那个|所选|当前)?(?:音频|视频|录音|播客|媒体)(?:文件)?'
        r'(?:的)?(?:内容是什么|是什么内容|有什么内容|里面是什么|里面有什么|讲的是什么|说的是什么|在讲什么|在说什么|主要内容)?\s*[。？！.!?]*\s*$',
        re.IGNORECASE,
    )

    @classmethod
    def analyze(
        cls,
        query: str,
        last_results: Optional[List[Dict]] = None,
        llm_service: Optional[any] = None,
    ) -> Optional[dict]:
        """
        Consolidated entry point for media intent and parameter extraction.
        
        If a media signal is detected, this expert 'locks' the intent to media_export
        and uses regex + ML fallback to ensure parameters are found.
        """
        if not query or len(query) < 5:
            return None

        q = cls._normalize_time_query(query.strip())
        if cls.looks_like_explicit_media_file_search(q) and not cls.looks_like_media_operation_request(q):
            logger.info(
                "[MediaQueryExpert] Skip explicit media file search so it can stay on file search: %r",
                q,
            )
            return None
        if cls._looks_like_document_retrieval_with_media_topic(q):
            logger.info(
                "[MediaQueryExpert] Skip document retrieval whose topic contains media terms: %r",
                q,
            )
            return None

        ql = q.lower()
        
        # ── Step 1: Broad detection ───────────────────────────────────────
        has_time = bool(cls._HAS_TIME_SIGNAL.search(ql))
        if has_time and cls._looks_like_calendar_year_reference(ql):
            has_time = False
        has_media_search = bool(cls._HAS_MEDIA_SEARCH_SIGNAL.search(ql))
        file_hint = cls._extract_file_hint(q)
        has_media = bool(cls._HAS_MEDIA_SIGNAL.search(ql) or file_hint)
        has_media_content = bool(cls._HAS_MEDIA_CONTENT_RE.search(q))  # content query
        has_duration = bool(cls._DURATION_QUERY_RE.search(q))

        if has_duration and has_media:
            logger.debug(f"[MediaQueryExpert] Duration lookup path query_chars={len(q or '')}")
            target_type = cls._detect_target_type(ql)
            params = {
                "query": q,
                "target_type": target_type,
                "sub_intent": "duration_lookup",
            }
            if file_hint:
                params["file_hint"] = file_hint
            return {
                "action": "media_export",
                "params": params,
                "confidence": 0.96,
            }

        # ── Branch A: Content search without timestamp ─────────────────
        # No time needed. Use deterministic routing only for a concrete topic;
        # generic overviews should be left to selection/LLM summary routing.
        if has_media_content and not has_time:
            if cls._AMBIGUOUS_GENERIC_MEDIA_QUESTION.search(q):
                logger.debug(f"[MediaQueryExpert] Skip ambiguous generic media question query_chars={len(q or '')}")
                return None
            if cls._GENERIC_MEDIA_OVERVIEW_QUESTION.search(q):
                logger.debug(f"[MediaQueryExpert] Defer generic media overview to selection/LLM query_chars={len(q or '')}")
                return None
            logger.debug(f"[MediaQueryExpert] Content-search path (no time) query_chars={len(q or '')}")
            # Use LLM to extract the search concept if available
            search_concept = q  # default to full query
            if llm_service:
                try:
                    from core.intent.media_op_agent import MediaOpAgent
                    extracted = MediaOpAgent.analyze(q, llm_service)
                    if extracted and extracted.get("action") in {"media_content_search", "media_export"}:
                        return extracted
                    # If LLM returned something else (point_lookup), check its concept
                    if extracted and extracted.get("params", {}).get("search_concept"):
                        search_concept = extracted["params"]["search_concept"]
                except Exception as _e:
                    logger.debug(f"[MediaQueryExpert] LLM content-search extraction failed: {_e}")

            # Determine media_type from query
            is_audio_only = bool(re.search(r'\b(?:audio|recording|podcast|song|music|mp3|wav)\b|音频|录音', ql))
            is_video_only = bool(re.search(r'\b(?:video|movie|clip)\b|视频', ql))
            media_type = "audio" if (is_audio_only and not is_video_only) else (
                "video" if (is_video_only and not is_audio_only) else "all"
            )

            params = {
                "query": search_concept,
                "media_type": media_type,
                "sub_intent": "content_search",
            }
            if file_hint:
                params["file_hint"] = file_hint
            return {
                "action": "media_content_search",
                "params": params,
                "confidence": 0.90,
            }

        # ── Branch B: Time-based query (existing path) ───────────────
        # If we don't have a time/search signal OR it's not a media query, skip.
        if not ((has_time or has_media_search) and has_media):
            return None

        logger.debug(f"[MediaQueryExpert] High confidence media query detected: query_chars={len(q or '')}")

        # ── Step 2: Parameter Extraction (Deterministic) ─────────────────
        time_sec = cls._extract_time(ql)
        time_end = cls._extract_time_range_end(ql)
        target_type = cls._detect_target_type(ql)
        # ── Step 3: LLM Refinement via MediaOpAgent ───────────────────────
        # If regex failed to find a precise timestamp OR it's a broad search query,
        # we delegate the rest of the parsing to the core MediaOpAgent.
        if (time_sec is None or has_media_search) and llm_service:
            logger.info("[MediaQueryExpert] Handing off to MediaOpAgent for deep intent classification...")
            from core.intent.media_op_agent import MediaOpAgent
            # Creating a dummy Context-like object if needed, or we just pass the query
            extracted = MediaOpAgent.analyze(q, llm_service)
            if extracted:
                return extracted

        # Final check if we somehow skipped the subagent
        if time_sec is None:
            if target_type == "video_visual":
                time_sec = 0.0
            else:
                return None

        params = {
            "query": q,
            "time_sec": time_sec,
            "target_type": target_type,
            "sub_intent": "range_summary" if (time_end is not None and cls._SUMMARY_SIGNAL_RE.search(ql)) else "point_lookup",
        }
        if time_end is not None:
            params["time_end_sec"] = time_end
        if file_hint:
            params["file_hint"] = file_hint

        return {
            "action": "media_export",
            "params": params,
            "confidence": 0.98,
        }

    @classmethod
    def _looks_like_calendar_year_reference(cls, query: str) -> bool:
        """Avoid treating business/history year ranges as media timestamps."""
        q = str(query or "")
        if not cls._CALENDAR_YEAR_REFERENCE_RE.search(q):
            return False
        explicit_media_time = re.search(
            r'\b(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|minute\s+mark|timestamp|timecode)\b'
            r'|\b\d{1,2}:\d{2}(?::\d{2})?\b|第\s*\d+\s*(?:秒|分|小时)|\d+\s*(?:秒|分钟|分|小时)',
            q,
            re.IGNORECASE,
        )
        return explicit_media_time is None

    @classmethod
    def looks_like_duration_query(cls, query: str) -> bool:
        return bool(cls._DURATION_QUERY_RE.search(str(query or "")))

    @classmethod
    def _extract_params_via_llm(cls, query: str, llm_service: any) -> Optional[dict]:
        """Targeted mini-prompt for structured media parameter extraction."""
        prompt = (
            "You are a media parameter extractor. Extract the target timestamp and file info from the user query.\n"
            "Query: \"{query}\"\n\n"
            "Return valid JSON ONLY:\n"
            "{{\n"
            "  \"time_sec\": float or null, (e.g. 5.0 for 5s, 90.0 for 1:30)\n"
            "  \"time_end_sec\": float or null,\n"
            "  \"target_type\": \"audio_content\" | \"video_visual\" | \"video_audio\",\n"
            "  \"file_hint\": \"filename.ext\" or \"\"\n"
            "}}\n"
            "Rules:\n"
            "- If user asks 'what happens' or 'see scene' or '做什么', target_type = 'video_visual'.\n"
            "- If user asks 'what is said', target_type = 'audio_content'.\n"
        ).format(query=query)
        
        try:
            # Use a fast, low-temperature call
            resp = _generate_media_llm_text(llm_service, prompt)
            if not resp: return None
            
            import json
            # Extract JSON from potential markdown markers
            clean_resp = resp.strip()
            if "```json" in clean_resp:
                clean_resp = clean_resp.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_resp:
                clean_resp = clean_resp.split("```")[1].split("```")[0].strip()
            
            data = json.loads(clean_resp)
            return {
                "time_sec": data.get("time_sec"),
                "time_end_sec": data.get("time_end_sec"),
                "target_type": data.get("target_type", "video_visual"),
                "file_hint": data.get("file_hint", "")
            }
        except Exception as e:
            logger.error(f"[MediaQueryExpert] LLM fallback extraction failed: {e}")
            return None

    @classmethod
    def _extract_time(cls, ql: str) -> Optional[float]:
        """Extract the primary timestamp in seconds from the query."""
        # Prefer explicit range patterns first so "5-10s" returns the start
        # boundary instead of accidentally matching the trailing "10s".
        m = cls._EN_PREFIX_RANGE_PAT.search(ql)
        if m:
            return 0.0

        if cls._EN_BEGINNING_POINT_PAT.search(ql):
            return 0.0

        m = cls._ZH_PREFIX_RANGE_PAT.search(ql)
        if m:
            return 0.0

        m = cls._CLOCK_RANGE_PAT.search(ql)
        if m:
            return cls._parse_clock_token(m.group(1))

        m = cls._CLOCK_TIME_PAT.search(ql)
        if m:
            return cls._parse_clock_token(m.group(1))

        m = cls._EN_SHARED_UNIT_RANGE_PAT.search(ql)
        if m:
            return cls._convert_number_with_unit(m.group(1), m.group(3))

        m = cls._EN_UNIT_RANGE_PAT.search(ql)
        if m:
            return cls._convert_number_with_unit(m.group(1), m.group(2))

        m = cls._ZH_MINUTE_RANGE_PAT.search(ql)
        if m:
            return float(m.group(1)) * 60

        m = cls._EN_RANGE_PAT.search(ql)
        if m:
            if m.group(1) and m.group(2):
                return float(m.group(1)) * 60 + float(m.group(2))
            if m.group(3):
                return float(m.group(3))

        m = cls._ZH_RANGE_PAT.search(ql)
        if m:
            if m.group(1):
                return float(m.group(1))

        m = cls._EN_MINUTE_MARK_PAT.search(ql)
        if m:
            return float(m.group(1)) * 60

        m = cls._ZH_MINUTE_AROUND_PAT.search(ql)
        if m:
            return float(m.group(1)) * 60

        m = cls._ZH_HOUR_PAT.search(ql)
        if m:
            hours = float(m.group(1) or 0)
            minutes = float(m.group(2) or 0)
            seconds = float(m.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds

        # Try English MM:SS / Ns / Nm / Nh patterns
        m = cls._EN_TIME_PAT.search(ql)
        if m:
            if m.group(1) and m.group(2):  # MM:SS
                return float(m.group(1)) * 60 + float(m.group(2))
            if m.group(3):  # N seconds
                return float(m.group(3))
            if m.group(4):  # N minutes
                return float(m.group(4)) * 60
            if m.group(5):  # N hours
                return float(m.group(5)) * 3600

        # Try Chinese patterns
        m = cls._ZH_TIME_PAT.search(ql)
        if m:
            if m.group(1) and m.group(2):
                return float(m.group(1)) * 60 + float(m.group(2))
            if m.group(2):
                return float(m.group(2))
            if m.group(3):
                return float(m.group(3)) * 60
            if m.group(4):
                return float(m.group(4))

        return None

    @classmethod
    def _extract_time_range_end(cls, ql: str) -> Optional[float]:
        """Extract the end timestamp from a range query (e.g. 'from 10 to 20 seconds')."""
        m = cls._EN_PREFIX_RANGE_PAT.search(ql)
        if m:
            amount = float(m.group(1))
            unit = str(m.group(2) or "").lower()
            if unit.startswith("h"):
                return amount * 3600
            return amount * 60 if unit.startswith("m") else amount

        m = cls._ZH_PREFIX_RANGE_PAT.search(ql)
        if m:
            amount = float(m.group(1))
            unit = str(m.group(2) or "").lower()
            if unit.startswith("小时"):
                return amount * 3600
            return amount * 60 if unit.startswith("分") or unit.startswith("分钟") else amount

        m = cls._CLOCK_RANGE_PAT.search(ql)
        if m:
            return cls._parse_clock_token(m.group(2))

        m = cls._EN_SHARED_UNIT_RANGE_PAT.search(ql)
        if m:
            return cls._convert_number_with_unit(m.group(2), m.group(3))

        m = cls._EN_UNIT_RANGE_PAT.search(ql)
        if m:
            return cls._convert_number_with_unit(m.group(3), m.group(4))

        m = cls._ZH_MINUTE_RANGE_PAT.search(ql)
        if m:
            return float(m.group(2)) * 60

        m = cls._EN_RANGE_PAT.search(ql)
        if m:
            if m.group(4) and m.group(5):
                return float(m.group(4)) * 60 + float(m.group(5))
            if m.group(6):
                return float(m.group(6))

        m = cls._ZH_RANGE_PAT.search(ql)
        if m:
            if m.group(2):
                return float(m.group(2))

        return None

    @staticmethod
    def _parse_clock_token(token: str) -> Optional[float]:
        parts = [p.strip() for p in str(token or "").split(":") if p.strip()]
        if len(parts) == 2:
            minutes, seconds = parts
            if minutes.isdigit() and seconds.isdigit():
                return float(int(minutes) * 60 + int(seconds))
            return None
        if len(parts) == 3:
            hours, minutes, seconds = parts
            if hours.isdigit() and minutes.isdigit() and seconds.isdigit():
                return float(int(hours) * 3600 + int(minutes) * 60 + int(seconds))
            return None
        return None

    @staticmethod
    def _convert_number_with_unit(amount: str, unit: str) -> float:
        value = float(amount)
        unit_lower = str(unit or "").lower()
        if unit_lower.startswith("h"):
            return value * 3600
        return value * 60 if unit_lower.startswith("m") else value

    @staticmethod
    def _normalize_time_query(query: str) -> str:
        return re.sub(
            r'(\d)\s*份(?=(?:\s*(?:钟|分钟|分|到|至|-|左右|附近|处|之间))|\b)',
            r'\1分',
            str(query or ""),
            flags=re.IGNORECASE,
        )

    @classmethod
    def looks_like_explicit_media_file_search(cls, query: str) -> bool:
        q = cls._normalize_time_query(str(query or "").strip())
        if not q or not cls._EXPLICIT_MEDIA_FILE_SEARCH_RE.search(q):
            return False
        if cls._SUMMARY_SIGNAL_RE.search(q) and re.search(
            r'\b(?:selected|current|this|that)\b|选中|当前|这个|那个',
            q,
            re.IGNORECASE,
        ):
            return False

        scrubbed = q
        file_hint = cls._extract_file_hint(q)
        if file_hint:
            scrubbed = scrubbed.replace(file_hint, " ")

        ql = scrubbed.lower()
        has_time = cls._extract_time(ql) is not None or cls._extract_time_range_end(ql) is not None
        has_media_search = bool(cls._HAS_MEDIA_SEARCH_SIGNAL.search(ql))
        return not has_time and not has_media_search

    @classmethod
    def looks_like_media_operation_request(cls, query: str) -> bool:
        """Detect generic media operations that should route to media_export.

        The signal is operation-level, not dataset-specific: timestamp/range
        lookup, frame/screenshot extraction, transcript/caption requests, and
        clip/export/convert wording. Plain media inventory like "find videos
        about X" remains ordinary search.
        """
        q = cls._normalize_time_query(str(query or "").strip())
        if not q:
            return False
        ql = q.lower()
        has_time = bool(cls._HAS_TIME_SIGNAL.search(ql))
        if has_time and cls._looks_like_calendar_year_reference(ql):
            has_time = False
        has_media = bool(cls._HAS_MEDIA_SIGNAL.search(ql) or cls._extract_file_hint(q))
        if has_time and has_media:
            return True
        if not has_media:
            return False
        return bool(cls._MEDIA_OPERATION_RE.search(q))

    @classmethod
    def _looks_like_document_retrieval_with_media_topic(cls, query: str) -> bool:
        """Return true when media words are the topic of a document search.

        Example: "search for papers about audio deep learning" should retrieve
        papers, not inspect audio files. The guard is intentionally about broad
        document target types, not any private file title or local dataset term.
        """
        q = cls._normalize_time_query(str(query or "").strip())
        return bool(q and cls._DOCUMENT_RETRIEVAL_WITH_MEDIA_TOPIC_RE.search(q))

    @classmethod
    def _detect_target_type(cls, ql: str) -> str:
        """Determine whether the user wants audio content or video visual."""
        has_audio = bool(cls._AUDIO_INDICATOR.search(ql))
        has_video = bool(cls._VIDEO_INDICATOR.search(ql))
        if has_audio and not has_video:
            return "audio_content"
        if cls._VIDEO_VISUAL_INDICATOR.search(ql):
            return "video_visual"
        if has_audio:
            return "audio_content"
        if has_video:
            # Video mentioned but no explicit visual request → default to audio track content
            return "video_audio"
        # Default: audio content
        return "audio_content"

    @classmethod
    def _extract_file_hint(cls, query: str) -> str:
        m = cls._FILE_HINT_PAT.search(query)
        if not m:
            return ""
        matched = re.sub(r"^(?:in|at|from|within|inside)\s+", "", m.group(0), flags=re.IGNORECASE).strip()

        # Walk character-by-character to find the "true" filename start.
        for i in range(len(matched) - 1):
            ch = matched[i]
            nxt = matched[i + 1]
            # CJK char immediately followed by a digit → this is the boundary where
            # the filename stem begins.  Walk back to include the whole CJK run.
            if ('\u4e00' <= ch <= '\u9fff') and nxt.isdigit():
                # Scan backward to include all contiguous CJK chars before this
                # digit that are separated from the leading verb block by a gap.
                # If the CJK run from position 0 is unbroken up to here, use
                start = i
                run_start = start
                while run_start > 0 and '\u4e00' <= matched[run_start - 1] <= '\u9fff':
                    run_start -= 1
                # If there's a continuous CJK run from the very beginning, take
                # only the final 2 chars of it as the filename prefix (heuristic:
                if run_start == 0 and i > 1:
                    start = max(i - 1, 0)  # include one char before the digit-adjacent char
                return matched[start:]
            # First ASCII filename-stem char → filename starts here.  Ignore the
            # extension dot so pure-CJK stems keep the full name.
            if ch.isascii() and (ch.isalnum() or ch in "_-") and "." in matched[i + 1:]:
                return matched[i:]

        return matched
