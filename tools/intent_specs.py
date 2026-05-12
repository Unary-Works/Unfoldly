
from __future__ import annotations

import re

from .intent_registry import IntentRegistry, IntentSpec


def _has_prev_results(ctx) -> bool:
    return bool(getattr(ctx, "last_results", None))


def _has_active_paths(ctx) -> bool:
    return bool(getattr(ctx, "active_paths", None))


def _has_contextual_scope(ctx) -> bool:
    return bool(getattr(ctx, "active_paths", None) or getattr(ctx, "last_results", None))


def _is_explicit_open_request(ctx) -> bool:
    query = str(getattr(ctx, "question", "") or "")
    return bool(re.search(r"\b(open|launch)\b|打开|开启|启动", query, re.IGNORECASE))


def _has_active_media(ctx) -> bool:
    """Only expose media_timequery when a media file is selected."""
    import os
    _MEDIA_EXTS = {
        ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
        ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv",
    }
    paths = getattr(ctx, "active_paths", None) or []
    return any(
        os.path.splitext(str(p))[1].lower() in _MEDIA_EXTS
        for p in paths
    )


def _has_media_scope(ctx) -> bool:
    import os

    _MEDIA_EXTS = {
        ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
        ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv",
    }
    paths = getattr(ctx, "active_paths", None) or []
    if any(os.path.splitext(str(p))[1].lower() in _MEDIA_EXTS for p in paths):
        return True

    last_results = list(getattr(ctx, "last_results", None) or [])[:12]
    if not last_results:
        return False
    media_hits = 0
    for item in last_results:
        category = str(item.get("doc_category") or item.get("doc_category_family") or "").strip().lower()
        file_name = str(item.get("file_name") or item.get("file_path") or "").strip()
        if os.path.splitext(file_name)[1].lower() in _MEDIA_EXTS or category in {"audio", "video", "audio/video"}:
            media_hits += 1
    return media_hits >= max(1, (len(last_results) + 1) // 2)


def _has_media_export_signal(ctx) -> bool:
    """Expose media_export only when the request is about media operations."""
    try:
        from core.intent.media_query_expert import MediaQueryExpert

        question = str(getattr(ctx, "question", "") or "")
        return _has_media_scope(ctx) or MediaQueryExpert.looks_like_media_operation_request(question)
    except Exception:
        return _has_media_scope(ctx)


def _register_default_intents():

    # ── 1. search ──────────────────────────────────────────────────────────
    IntentRegistry.register(
        IntentSpec(
            name="search",
            description="核心语义搜索。搜索内容、回答问题，或从数据源中提取信息（RAG语义搜索）。",
            description_en=(
                "Core semantic retrieval. Search indexed content, answer questions, "
                "and extract facts from sources."
            ),
            params={
                "query": "完整搜索词或问题",
                "category": "类别过滤（可选；只在清晰文件类型时填写，文件/文档等泛词不要单独变成 document 过滤）",
                "keywords": "核心名词关键词（仅名词，不带'哪些'/'我的'等无意义词）",
                "folder": "限制在特定文件夹（可选）",
                "file_extensions": "文件后缀（逗号分隔，如 .pdf,.docx），用户明确提到格式时填写",
            },
            params_en={
                "query": "Full search question or query text",
                "category": "Optional category filter; use only for clear file-type buckets, not for generic words like file/document/docs",
                "keywords": "Core noun keywords for precise matching",
                "folder": "Optional folder name/path keyword restriction",
                "file_extensions": "Optional comma-separated extensions (e.g. .pdf,.docx)",
            },
            when_to_use=[
                "🔥 用户询问文件内容、找答案、总结要点，优先使用 search",
                "🔥 用户提问具体的人名、公司名、电话、地址/住址、流程等事实类问题（如 “某人的电话号是多少”、“这公司的地址是”、“他的家在哪”），哪怕没提到“文件”二字，也必须走 search",
                "🔥 询问\"有没有xxx的内容\"、\"关于xxx的信息\"、\"找xxx文件\" → search",
                "🔥 短名词短语（如\"初中数学试卷\"）且没有明确说统计数量 → 默认 search",
                "🔥 查找媒体文件本身（文件名、相机编号、扩展名、或'find/show/list ... video/image/audio file'）→ search；不要当作媒体内容检索",
                "🔥 '找所有pdf'、'列出word文件'、'我的pdf有哪些'、'显示mp4' → search + file_extensions='.pdf'/'.docx'/'.mp4'",
                "🔥 按文件类型查找/列出（find/list/show/get + 文件类型）→ search，不是 count",
                "🔥 count 只用于统计数量（'有多少pdf'、'共几个文件'），不用于列出文件",
            ],
            when_to_use_en=[
                "🔥 MUST use search for content questions, finding answers, key-point requests.",
                "🔥 If user asks factual entity questions like 'What is X's phone number?', 'where is his home/residence?', or 'Where is company Y', MUST use search, even without file keywords.",
                "🔥 If user asks about specific topics or wants to find particular files → search.",
                "🔥 A short noun phrase without counting intent MUST default to search.",
                "🔥 'find files about X', 'show me resumes about Y', 'documents related to Z' → search.",
                "🔥 'file', 'document', and 'docs' are generic container words unless the user clearly asks for a document inventory/format; keep catalog/list/table/title words in query and avoid category='document' when unsure.",
                "🔥 'find/show/list videos or audio files about X' is file retrieval → search with category/audio-video filters. Only use media_content_search for inside-content wording like mention/say/show/contain/transcript/scene.",
                "🔥 Finding a media file by filename, camera-style id, extension, or 'find/show/list ... video/image/audio file' wording → search. Do NOT use media_content_search for file inventory or filename lookup.",
                "🔥 'find all the pdf files', 'list my word docs', 'show me mp4 files', 'get all xlsx' → search with file_extensions param (.pdf / .docx / .mp4 / .xlsx).",
                "🔥 Extension-based LISTING (find/list/show/get + file type) → search + file_extensions. NOT count.",
                "🔥 count is ONLY for statistics: 'how many pdfs do I have', 'count my files'. NOT for listing.",
                "🔥 '我的pdf文件有哪些', '找所有pdf', 'find pdf', 'list all pdfs' → search, file_extensions='.pdf'.",
            ],
            priority=1,
        )
    )

    # ── 1b. media_content_search ──────────────────────────────────────────
    IntentRegistry.register(
        IntentSpec(
            name="media_content_search",
            description="按主题在音频/视频内容、转写、关键帧或画面描述中检索。",
            description_en=(
                "Search inside audio/video content by topic, including transcripts, "
                "keyframes, scene descriptions, or spoken/visible content."
            ),
            params={
                "query": "要在音视频内容中查找的主题，使用英文检索关键词",
                "media_type": "video / audio / all",
                "file_hint": "可选：目标媒体文件名提示",
            },
            params_en={
                "query": "English topic keywords to search inside media content",
                "media_type": "video | audio | all",
                "file_hint": "Optional target media filename hint",
            },
            when_to_use=[
                "🔥 用户询问视频/音频里关于某主题说了什么、展示了什么、出现了什么、提到什么 → media_content_search",
                "🔥 '哪些视频/音频提到 X'、'视频里有没有 X'、'音频关于 X 说了什么' → media_content_search",
                "🔥 媒体文件清单（如'列出视频文件'）不是这个动作，应使用 search + category/extension",
                "🔥 静态图片的内容追问（如 this image / selected photo）不是这个动作，应继续处理上一轮结果或选中文件",
            ],
            when_to_use_en=[
                "🔥 Use this when the user asks what audio/video says, shows, mentions, contains, or is about regarding a topic.",
                "🔥 'which videos mention X', 'what is in the video about X', 'does any recording discuss X' → media_content_search.",
                "🔥 Do NOT use this for file retrieval wording like 'find videos about X' or 'show audio files about X'; that is search.",
                "🔥 Media file inventory such as 'list my videos' is NOT this action; use search with category/extension instead.",
                "🔥 Still-image follow-ups such as 'what is in this image/photo' are NOT this action; continue on the previous result or selected file instead.",
            ],
            priority=2,
        )
    )

    # ── 1c. media_export ─────────────────────────────────────────────────
    IntentRegistry.register(
        IntentSpec(
            name="media_export",
            description="对音频/视频执行时间点、时间段、画面、转写或导出类操作。",
            description_en=(
                "Operate on audio/video media: timestamp lookup, interval summary, "
                "frame/screenshot extraction, transcript/caption lookup, or clip/export style requests."
            ),
            params={
                "query": "完整媒体操作请求",
                "sub_intent": "point_lookup / range_summary / media_summary",
                "media_type": "video / audio / all",
                "target_type": "audio_content / video_visual / video_audio",
                "time_sec": "可选：开始时间（秒）",
                "time_end_sec": "可选：结束时间（秒）",
                "file_hint": "可选：目标媒体文件名提示",
            },
            params_en={
                "query": "Full media operation request",
                "sub_intent": "point_lookup | range_summary | media_summary",
                "media_type": "video | audio | all",
                "target_type": "audio_content | video_visual | video_audio",
                "time_sec": "Optional start time in seconds",
                "time_end_sec": "Optional end time in seconds",
                "file_hint": "Optional target media filename hint",
            },
            when_to_use=[
                "🔥 用户要求查看/总结音视频某个时间点或时间段、截帧、截图、转写、字幕、剪辑、导出 → media_export。",
                "🔥 如果用户要对媒体文件执行操作，即使目标文件还需要查找，也使用 media_export，不要退化成普通 search。",
            ],
            when_to_use_en=[
                "🔥 Use this for timestamp/range media operations: what happens at 1:20, first 30 seconds, between 10s and 20s.",
                "🔥 Use this for frame/screenshot/transcript/caption/clip/export/convert requests on audio/video.",
                "🔥 If the user asks to operate on a media file, use media_export even when the target file must be located first; do not downgrade to search.",
            ],
            priority=2,
            expose_condition=_has_media_export_signal,
        )
    )

    # ── 1a. contextual_refine ──────────────────────────────────────────────
    IntentRegistry.register(
        IntentSpec(
            name="contextual_refine",
            description="在当前上下文范围内继续处理：已选文件、已选文件夹、或上一轮结果。",
            description_en=(
                "Continue working INSIDE the current conversational scope "
                "(selected files, selected folder, or previous results) without starting a fresh global search."
            ),
            params={
                "scope": "范围：selected_items / selected_folder / last_results",
                "operation": "操作：list / summary / qa / rewrite / support",
                "query": "可选：范围内继续追问的主题或条件",
                "focus_extension": "可选：文件类型过滤，如 .pdf / .txt",
                "rewrite_mode": "可选：shorter / more_detail / supporting_files",
            },
            params_en={
                "scope": "selected_items | selected_folder | last_results",
                "operation": "list | summary | qa | rewrite | support",
                "query": "Optional narrowed follow-up query inside the current scope",
                "focus_extension": "Optional file-type filter such as .pdf or .txt",
                "rewrite_mode": "Optional rewrite mode: shorter | more_detail | supporting_files",
            },
            when_to_use=[
                "🔥 只在当前上下文里继续处理：已选文件、已选文件夹、或上一轮结果，优先用 contextual_refine。",
            ],
            when_to_use_en=[
                "🔥 Prefer this for ANY follow-up that stays inside selected files, a selected folder, or previous results: "
                "'these files', 'only the text files', 'make it shorter', 'which files support that'.",
            ],
            priority=2,
            expose_condition=_has_contextual_scope,
        )
    )

    # ── 2. list_selected ───────────────────────────────────────────────────
    # Progressive: only shown when user has files selected in Sources panel
    IntentRegistry.register(
        IntentSpec(
            name="list_selected",
            description=(
                "列出或总结当前在左侧 Sources 面板中已勾选的文件。"
                "注意：'选中的文件有哪些' 和 '我的文件有哪些' 含义完全不同！"
                "'选中的文件'只代表用户勾选的那几个文件，不等于全部文件库。"
            ),
            description_en=(
                "List or describe ONLY the files currently selected in the Sources panel. "
                "IMPORTANT: 'what are the selected files' is DIFFERENT from 'what files do I have'. "
                "'Selected files' = only the N files the user ticked in the sidebar. "
                "Use for ANY query that mentions 'selected', 'the selected documents' (when a selection is active), "
                "'these files', 'chosen files', even with typos like 'seleted'."
            ),
            params={},
            params_en={},
            when_to_use=[
                "🔥 '选中的文件有哪些' / '已选文件是哪些' → list_selected（不是 count(all)）",
                "🔥 '这些文件讲什么' / '告诉我选中的文档' / '介绍这些文件' → list_selected",
                "🔥 '当前选中' / '已选择' / '这些文件' / '选中的文档' → list_selected",
                "🔥 严格区分：'我的文件有哪些' = 全库 count；'选中的文件有哪些' = list_selected",
            ],
            when_to_use_en=[
                "🔥 'what are the selected files', 'list the selected documents' → list_selected (NOT count all).",
                "🔥 'tell me about the selected documents', 'summarize the selected files' → list_selected.",
                "🔥 'what are these documents', 'tell me about the documents' (when files are selected) → list_selected.",
                "🔥 'selected', 'chosen', 'these files', 'the selected documents', 'seleted' (typo) with active selection → list_selected.",
                "🔥 KEY: 'what files do I have' = ALL files (use count); 'what are the selected files' = list_selected.",
                "🔥 NEVER use search or process_previous for selected-file queries.",
            ],
            examples=[
                "tell me about the selected documents",
                "tell me about the documents",
                "what are the selected files?",
                "which files are selected?",
                "summarize what I selected",
                "选中的文件是哪些",
                "选中的文件有哪些",
                "给我介绍这些文档",
            ],
            examples_en=[
                "tell me about the selected documents",
                "tell me about the documents",
                "what are the selected files",
                "which files did I select",
                "summarize the selected files",
                "what are these files about",
                "describe what I've selected",
            ],
            priority=2,
            expose_condition=_has_active_paths,
        )
    )


    # ── 3. summarize_all ───────────────────────────────────────────────────
    IntentRegistry.register(
        IntentSpec(
            name="summarize_all",
            description="对所有文件（或指定格式的文件）进行全局视角的总结，不区分类别。",
            description_en=(
                "Summarize all indexed files globally, optionally filtered by file extension. "
                "Use for 'summarize everything', 'overview of all my files', "
                "'summarize all my pdfs'."
            ),
            params={
                "focus": "用户关注的特定重点（可选）",
                "file_extensions": "文件后缀（逗号分隔，如 .pdf,.docx）",
            },
            params_en={
                "focus": "Optional user focus keyword",
                "file_extensions": "Optional comma-separated file extensions (.pdf, .docx, etc.)",
            },
            when_to_use=[
                "🔥 用户要求\"总结我所有文件\"、\"整理所有资料\"、\"帮我汇总全部内容\" → summarize_all",
                "🔥 用户要求 summarize/overview/digest/analyze/explain all/everything/my files/my documents/my docs → summarize_all",
                "🔥 用户要求总结特定格式的全部文件（如\"总结所有html\"、\"归纳我的所有mp4\"）→ summarize_all + file_extensions",
                "🔥 真正的\"所有内容\"范围，没有限定特定业务分类",
            ],
            when_to_use_en=[
                "🔥 User requests global summary of all files/documents/docs → summarize_all.",
                "🔥 'summarize all my documents', 'overview of all files', 'digest everything', 'analyze my docs' → summarize_all.",
                "🔥 User asks to summarize a specific file type (all pdfs, all mp4s) → summarize_all + file_extensions.",
                "🔥 When query is 'overview', 'big picture', 'summarize everything' → summarize_all.",
            ],
            priority=5,
        )
    )

    # ── 3a. summarize_selected ─────────────────────────────────────────────
    # Progressive: only shown when user has files/folders selected in Sources
    IntentRegistry.register(
        IntentSpec(
            name="summarize_selected",
            description="对左侧已勾选的文件或文件夹进行整体总结。",
            description_en=(
                "Summarize the currently selected files or folder (from Sources panel). "
                "Use when user says 'summarize these', 'summarize the selected files', "
                "'summarize this folder', or any summarize request when files/folders are active."
            ),
            params={"focus": "可选：用户关注的重点或文件类型过滤"},
            params_en={"focus": "Optional: focus keyword or file-type filter (e.g. 'text files', 'PDFs')"},
            when_to_use=[
                "🔥 '总结这些文件' / '总结已选文件' / '总结这个文件夹' → summarize_selected",
                "🔥 已选文件/文件夹总结后的 T2 精炼：'只看文本文件' / '更简洁地总结' → summarize_selected",
                "🔥 有 active_paths 时，优先用此而非 summarize_all",
            ],
            when_to_use_en=[
                "🔥 'summarize these files', 'summarize the selected docs', 'summarize this folder' → summarize_selected.",
                "🔥 T2 refinements after a selected-file/folder summary: 'focus only on text files', "
                "'now only images', 'summarize it more briefly', 'which files support that' → summarize_selected.",
                "🔥 ALWAYS prefer this over summarize_all when active_paths is non-empty.",
            ],
            examples_en=[
                "summarize these files",
                "summarize the selected documents",
                "summarize this folder",
                "focus only on text files",
                "now only images",
                "summarize it more briefly",
            ],
            priority=4,
            expose_condition=_has_active_paths,  # 🔑 only when files/folders selected
        )
    )

    # ── 3b. media_followup ─────────────────────────────────────────────────
    IntentRegistry.register(
        IntentSpec(
            name="media_followup",
            description="在当前音频/视频上下文里继续处理：主题追问、时间点查询、区间总结。",
            description_en=(
                "Continue working inside the current audio/video scope: "
                "topic follow-ups, timestamp lookups, interval summaries, or media-only rewrites."
            ),
            params={
                "operation": "操作：topic_search / time_lookup / range_summary / summary / rewrite",
                "query": "范围内要找的主题或原始问题",
                "media_type": "audio / video / all",
                "file_hint": "可选：目标媒体文件名提示",
                "target_type": "audio_content / video_visual / video_audio",
                "time_sec": "可选：开始时间（秒）",
                "time_end_sec": "可选：结束时间（秒）",
            },
            params_en={
                "operation": "topic_search | time_lookup | range_summary | summary | rewrite",
                "query": "Topic or original follow-up question within the media scope",
                "media_type": "audio | video | all",
                "file_hint": "Optional target media filename hint",
                "target_type": "audio_content | video_visual | video_audio",
                "time_sec": "Optional start time in seconds",
                "time_end_sec": "Optional end time in seconds",
            },
            when_to_use=[
                "🔥 当前上下文主要是音频/视频时，围绕这批媒体继续追问、点查或总结，优先用 media_followup。",
            ],
            when_to_use_en=[
                "🔥 Prefer this over generic search/process_previous when the user is still talking about the current media files.",
            ],
            priority=3,
            expose_condition=_has_media_scope,
        )
    )

    # ── 4. summarize (category) ────────────────────────────────────────────
    IntentRegistry.register(
        IntentSpec(
            name="summarize",
            description="总结某一特定类别（如简历、报告、发票）的文件内容，提取关键信息。",
            description_en="Summarize files in a specific category and extract key points.",
            params={
                "category": "类别名称（必须从[已选数据源类别]中提取）",
                "query": "用户关注的特定重点（可选）",
                "file_extensions": "文件后缀（逗号分隔，可选）",
            },
            params_en={
                "category": "Category name from selected source categories (required)",
                "query": "Optional focus keyword/topic",
                "file_extensions": "Optional comma-separated file extensions",
            },
            when_to_use=[
                "🔥 用户要求\"总结所有简历\"、\"整理我的报告内容\"、\"查看音频并总结\" → summarize + category",
                "🔥 必须填写 category 参数，偏向分析特定分类的具体内容",
            ],
            when_to_use_en=[
                "🔥 User requests summary for one specific category (e.g. all resumes/all reports) → summarize.",
                "🔥 Category param is required. This is analytic summarization within a category.",
            ],
            priority=5,
        )
    )

    # ── 5. process_previous ────────────────────────────────────────────────
    # Progressive: only shown when previous results exist
    IntentRegistry.register(
        IntentSpec(
            name="process_previous",
            description="对上一轮找出的文件进行二次处理（总结、筛选或回答特定问题）。",
            description_en=(
                "Post-process previously returned files (summarize/filter/QA within last results). "
                "Use for explicit prior-result references and for qualitative follow-up questions "
                "that continue comparing, ranking, judging, or interpreting the already returned items."
            ),
            params={
                "operation": "操作类型（summarize / filter / qa）",
                "query": "用户的具体问题或筛选条件",
            },
            params_en={
                "operation": "Operation type (summarize/filter/qa)",
                "query": "Specific condition or follow-up question",
            },
            when_to_use=[
                "🔥 用户要\"总结这批文件\"、\"对上面的结果归纳\"→ process_previous",
                "🔥 使用代词（\"他们\"、\"这些\"、\"them\"）且上一轮刚列出文件时 → process_previous",
                "🔥 极短跟进词（\"结论\"、\"recap\"、\"tldr\"）且上一轮有文件列表 → process_previous",
                "🔥 绝对不要把这些短句当成新的 search 查询词",
            ],
            when_to_use_en=[
                "🔥 User asks to summarize/filter/QA over the PREVIOUS result set → process_previous.",
                "🔥 Short pronouns ('them', 'these', 'those') following a file list → process_previous.",
                "🔥 Ordinal and short-fragment follow-ups after a file list ('the first one', 'item 2', 'holiday policy?', 'any mobile ones?') → process_previous.",
                "🔥 Comparative or compound QA over prior results ('compared to...', 'how many items are listed and what categories do they cover') → process_previous.",
                "🔥 After a single image/photo result, 'what is in this image/photo/picture?' or 'describe this image' → process_previous.",
                "🔥 Short meta-requests ('conclusion', 'recap', 'tldr', 'in short') after list → process_previous.",
                "🔥 After prior results or a prior comparison, qualitative questions such as 'who is more X', "
                "'which one is stronger at Y', 'who fits better', or 'what is the difference' → process_previous.",
                "🔥 Refine or narrow a prior summary: 'focus only on text files', 'summarize it more briefly', "
                "'which files support that', 'make it shorter', 'now only images' → process_previous.",
                "🔥 MUST stay within last shown files; do NOT start a fresh search.",
            ],
            priority=6,
            expose_condition=_has_prev_results,
        )
    )

    # ── 6. count ───────────────────────────────────────────────────────────
    IntentRegistry.register(
        IntentSpec(
            name="count",
            description=(
                "统计或列出文件清单（按类别或文件格式）。"
                "当用户问'有几份'、'有哪些pdf'、'我的所有docx'时使用。"
            ),
            description_en=(
                "Count files by category or file type — for STATISTICS only. "
                "Use ONLY when user asks how many files exist. "
                "For listing files by type ('find all pdfs'), use search instead."
            ),
            params={
                "category": "类别名称（从[已选数据源类别]提取）或 all",
                "file_extensions": "文件后缀（逗号分隔，如 .pdf,.docx），用户提到特定格式时必须填写",
            },
            params_en={
                "category": "Known category from selected sources, or 'all'",
                "file_extensions": "Comma-separated extensions (.pdf,.docx,.mp3 etc). MUST fill if user mentions file type.",
            },
            when_to_use=[
                "🔥 用户问 '有几份'、'有多少个文件'、'有多少pdf' → count（纯统计数量）",
                "🔥 '有哪些简历的数量'、'帮我统计所有报告' → count + category",
                "🔥 用户说 '找pdf'、'列出word文件'、'查看所有mp4' → 不是 count，用 search + file_extensions",
                "🔥 如果用户问内容（如 'XX有什么内容'）→ 用 search，不是 count",
            ],
            when_to_use_en=[
                "🔥 User explicitly asks how many / which files → count.",
                "🔥 'how many pdfs', 'count my docs', 'number of files' → count.",
                "🔥 'which resume files', 'list all contracts' → count + category.",
                "🔥 'find all pdfs', 'list my word docs', 'show mp4 files' → NOT count → search + file_extensions.",
                "🔥 count = quantity/statistics. search = find/list/show/get actual files.",
            ],
            priority=10,
        )
    )

    # ── 7. view_detail ─────────────────────────────────────────────────────
    IntentRegistry.register(
        IntentSpec(
            name="view_detail",
            description="查看搜索结果中第 N 份文件的详细信息（搜索后输入纯数字时触发）。",
            description_en="Open detail/summary for the Nth file in current search results.",
            params={"index": "序号（从1开始的整数）"},
            params_en={"index": "1-based index in current result list"},
            when_to_use=[
                "🔥 搜索后用户输入纯数字（\"1\"、\"2\"） → view_detail",
            ],
            when_to_use_en=[
                "🔥 User replies with a plain index number after search (e.g. '1', '2') → view_detail.",
            ],
            priority=12,
            expose_condition=_has_prev_results,
        )
    )

    # ── 8. open_file ───────────────────────────────────────────────────────
    IntentRegistry.register(
        IntentSpec(
            name="open_file",
            description="用系统/文件查看器打开指定文件；只在用户明确说打开/启动时使用。",
            description_en="Open a specific file in the OS/file viewer; only for explicit open/launch requests.",
            params={"file_name": "要打开的文件名"},
            params_en={"file_name": "Target file name to open"},
            when_to_use=["🔥 明确说 \"打开文件XXX\" / \"帮我打开XXX\" / \"启动XXX\" → open_file；\"给我看这个文件\" 通常是总结/描述，不是 open_file。"],
            when_to_use_en=["🔥 Explicit 'open file X' or 'launch X' request → open_file; 'show me this file' usually means summarize/describe, not open_file."],
            priority=13,
            expose_condition=_is_explicit_open_request,
        )
    )

    # ── 9. chat ────────────────────────────────────────────────────────────
    IntentRegistry.register(
        IntentSpec(
            name="chat",
            description="日常问候、能力介绍等，不涉及文件查询或内容检索。",
            description_en="General chat, greetings, and capability questions without retrieval intent.",
            params={},
            params_en={},
            when_to_use=[],
            when_to_use_en=[
                "Use for 'hello', 'what can you do', 'how does this work' etc.",
                "NOT for any file-related queries.",
            ],
            priority=99,
        )
    )


    # ── 10. clarify ────────────────────────────────────────────────────────
    IntentRegistry.register(
        IntentSpec(
            name="clarify",
            description="当用户意图完全不清晰、极其模糊，不知该执行哪种操作时，向用户提问以澄清需求。",
            description_en="When the user intent is extremely vague or unclear and you don't know what to do, ask the user for clarification.",
            params={"question": "向用户提问的澄清问题"},
            params_en={"question": "The clarification question to ask the user"},
            when_to_use=[
                "🔥 输入意图极度模糊且缺乏实体（如“怎么弄”、“帮我弄一下”），无法判断是搜索、统计还是日常聊天时使用。",
                "🔥 当你觉得不知道该做什么时（When you don't know what to do），使用 clarify 与用户确认。",
            ],
            when_to_use_en=[
                "🔥 Use ONLY when the user's input is extremely vague and you don't know what to do.",
                "🔥 Use this to check back with the user by asking a clarifying question.",
            ],
            priority=98,
        )
    )

    # ── 11. media_timequery ───────────────────────────────────────────────
    IntentRegistry.register(
        IntentSpec(
            name="media_timequery",
            description="查询音频/视频在特定时间戳的内容。支持音频转录内容查询和视频画面查询。",
            description_en=(
                "Query audio/video content at a specific timestamp. "
                "Audio: look up what is being said at time T. "
                "Video: describe the visual scene at time T, or look up spoken content."
            ),
            params={
                "query": "完整查询文本",
                "time_sec": "目标时间戳（秒）",
                "target_type": "查询类型：audio_content / video_visual / video_audio",
                "file_hint": "可选的文件名提示",
            },
            params_en={
                "query": "Full query text",
                "time_sec": "Target timestamp in seconds",
                "target_type": "audio_content | video_visual | video_audio",
                "file_hint": "Optional filename hint extracted from query",
            },
            when_to_use=[
                "🔥 用户询问'音频第30秒在说什么' → media_timequery",
                "🔥 用户询问'视频1分20秒画面是什么' → media_timequery",
                "🔥 包含时间戳 + 音视频关键词 → media_timequery",
            ],
            when_to_use_en=[
                "🔥 'what is said at 30 seconds in audio.mp3' → media_timequery.",
                "🔥 'what happens at 1:20 in the video' → media_timequery.",
                "🔥 Any query with a time reference + audio/video context → media_timequery.",
            ],
            priority=3,
            expose_condition=_has_active_media,  # 🔑 only when media file is selected
        )
    )

    # ── 12. translate_response ─────────────────────────────────────────────
    # Progressive: only shown when previous results exist
    IntentRegistry.register(
        IntentSpec(
            name="translate_response",
            description="将上一轮回答翻译成另一种语言。",
            description_en="Translate the previous response into another language.",
            params={"lang": "目标语言代码（en 或 zh）"},
            params_en={"lang": "Target language code: 'en' or 'zh'"},
            when_to_use=[
                "🔥 用户说'用英文'、'in english'、'translate'、'翻译成中文' → translate_response",
            ],
            when_to_use_en=[
                "🔥 User says 'in english', 'in chinese', 'translate', 'reply in english' → translate_response.",
            ],
            priority=4,
            expose_condition=_has_prev_results,
        )
    )

_register_default_intents()
