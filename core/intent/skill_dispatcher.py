"""
SkillDispatcher — single LLM call intent router driven by IntentRegistry.

Replaces the two-call pattern (ConversationRouter → ContinuationAgent/FileOpAgent)
with ONE compact LLM call. The prompt is dynamically assembled from IntentRegistry,
showing only context-relevant skills (progressive disclosure).

Design goals:
  - 1 LLM call instead of 2 → ~50% latency reduction on LLM-routed paths
  - LLM reads actual skill descriptions → better decisions, fewer hallucinations
  - IntentRegistry is the single source of truth for skill definitions
  - All deterministic fast-gates still fire BEFORE this dispatcher

Performance: prompt ~200 tokens, max_tokens=80 (JSON output).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_NON_ACTION_UTTERANCE_RE = re.compile(
    r"^\s*(?:"
    r"no\s+need|no\s+thanks|never\s*mind|all\s+set|cancel\s+that|"
    r"(?:ha){2,}|h{2,}|hehe+|lol+|lmao|rofl|xd+|"
    r"[哈呵嘿]{2,}|233+|笑死(?:我了)?|"
    r"不用了|先不用|算了|不需要了|[😂🤣]+"
    r")\s*[!！?？.。~～…]*\s*$",
    re.IGNORECASE,
)

_EXPLICIT_ACTIVE_SCOPE_RE = re.compile(
    r"\b(selected|seleted|selectd|slected|chosen|picked|checked|current\s+selection|"
    r"selected\s+(?:files?|documents?|docs?|folders?|items?|sources?))\b"
    r"|选中|已选|已勾选|当前选中|勾选",
    re.IGNORECASE,
)
_STRONG_DEICTIC_SCOPE_RE = re.compile(
    r"\b(?:this|that|these|those|current)\s+"
    r"(?:files?|documents?|docs?|folders?|reports?|guides?|videos?|audios?|images?|photos?|items?|results?)\b"
    r"|这个文件|那个文件|这些文件|那些文件|这份|那份|当前文件|当前文档",
    re.IGNORECASE,
)
_BARE_PRONOUN_FOLLOWUP_RE = re.compile(
    r"^\s*(?:please\s+|pls\s+|can\s+you\s+|could\s+you\s+)?"
    r"(?:show(?:\s+me)?|tell(?:\s+me)?(?:\s+about)?|describe|summari[sz]e|read|open|explain)\s+"
    r"(?:it|this|that)\s*[.!?。！？]*\s*$"
    r"|^\s*(?:给我看|看看|看下|讲讲|介绍|总结|概括|打开)\s*(?:这个|那个|这份|它)\s*$",
    re.IGNORECASE,
)
_PLURAL_CONTEXT_FOLLOWUP_RE = re.compile(
    r"^\s*(?:which|what|how\s+many|list|show|summari[sz]e|describe|compare|analy[sz]e)\b"
    r".{0,80}\b(?:them|these|those)\b"
    r"|这些|那些|它们|他们|她们",
    re.IGNORECASE,
)

_KNOWN_ACTIONS = {
    "search", "count", "chat", "summarize", "summarize_all", "summarize_selected",
    "process_previous", "view_detail", "open_file", "clarify", "list_selected",
    "media_timequery", "media_content_search", "media_export", "contextual_refine", "media_followup",
    "translate_response",
}

_ACTION_MAP = {action: action for action in _KNOWN_ACTIONS}

_AUDIO_EXTENSION_HINTS = {"wav", "mp3", "m4a", "flac", "aac", "ogg", "wma", "aiff", "ape"}
_VIDEO_EXTENSION_HINTS = {"mp4", "mov", "mkv", "avi", "webm", "flv", "wmv", "m4v", "ts"}


def _looks_like_non_action_utterance(query: str) -> bool:
    """Return True for pure laughter/noise that should never trigger file work."""
    q = str(query or "").strip()
    if not q:
        return False
    return bool(_NON_ACTION_UTTERANCE_RE.fullmatch(q))


def _non_action_clarify_message(ctx: Any, query: str) -> str:
    user_lang = str(getattr(ctx, "user_lang", "") or getattr(ctx, "prompt_language", "") or "").lower()
    use_zh = user_lang.startswith("zh") or any("\u4e00" <= ch <= "\u9fff" for ch in str(query or ""))
    if use_zh:
        return "我还没法判断你接下来想让我对文件做什么。你可以说得更具体一点，比如“详细解读这个视频”或“总结选中文件”。"
    return "I can't tell what you'd like me to do with the files yet. Try something more specific, like “tell me about this video” or “summarize the selected files.”"


def _should_expose_active_scope_to_llm(ctx: Any, query: str) -> bool:
    """Expose active_paths as contextual selection only when the user asks for it."""
    active_paths = list(getattr(ctx, "active_paths", None) or [])
    if not active_paths:
        return False
    text = str(query or "").strip()
    if not text:
        return False
    if _EXPLICIT_ACTIVE_SCOPE_RE.search(text) or _STRONG_DEICTIC_SCOPE_RE.search(text):
        return True
    if len(active_paths) == 1 and _BARE_PRONOUN_FOLLOWUP_RE.search(text):
        return True
    return bool(_PLURAL_CONTEXT_FOLLOWUP_RE.search(text))


def _build_context_hint(ctx: Any, *, lang: str = "en") -> str:
    """Build a compact context string for the LLM prompt."""
    from core.skills import ContextualRefineSkill, MediaFollowupSkill

    lines: List[str] = []
    use_en = lang.startswith("en")

    # Prior action / history
    history = getattr(ctx, "history", None) or []
    last_results = getattr(ctx, "last_results", None) or []
    active_paths = getattr(ctx, "active_paths", None) or []
    expose_active_scope = _should_expose_active_scope_to_llm(ctx, getattr(ctx, "question", "") or "")

    prior_user_query = ""
    if history:
        # Find last user query
        for msg in reversed(history):
            q = str(msg.get("q") or "").strip()
            if q:
                prior_user_query = q[:80]
                break
            if msg.get("role") == "user":
                prior_user_query = str(msg.get("content") or "")[:80]
                if prior_user_query:
                    break

    n_results = len(last_results)
    n_selected = len(active_paths)

    if use_en:
        lines.append(f"Prior results: {n_results} files")
        if expose_active_scope:
            lines.append(f"Context-selected files: {n_selected}")
        else:
            lines.append(
                f"Active source scope: {n_selected} indexed file(s) available for retrieval filtering; "
                "do not treat this as selected files unless the user says selected/current/this/these."
            )
        if prior_user_query:
            lines.append(f"Prior user query: {prior_user_query}")
        if n_results > 0:
            names = [str(r.get("file_name") or "").strip() for r in last_results[:4] if r.get("file_name")]
            if names:
                suffix = f" +{n_results - 4} more" if n_results > 4 else ""
                lines.append(f"Result files: {', '.join(names)}{suffix}")
        if expose_active_scope and n_selected > 0:
            sel_names = [os.path.basename(p) for p in active_paths[:4]]
            suffix = f" +{n_selected - 4} more" if n_selected > 4 else ""
            lines.append(f"Context-selected: {', '.join(sel_names)}{suffix}")
    else:
        lines.append(f"上轮结果: {n_results} 个文件")
        if expose_active_scope:
            lines.append(f"上下文选中文件: {n_selected} 个")
        else:
            lines.append(
                f"当前可检索来源范围: {n_selected} 个索引文件；这只是检索边界，"
                "除非用户明确说选中/current/this/these/这个/这些，否则不要当成选中文件。"
            )
        if prior_user_query:
            lines.append(f"上轮用户问题: {prior_user_query}")
        if n_results > 0:
            names = [str(r.get("file_name") or "").strip() for r in last_results[:4] if r.get("file_name")]
            if names:
                suffix = f" +{n_results - 4} more" if n_results > 4 else ""
                lines.append(f"结果文件: {', '.join(names)}{suffix}")
        if expose_active_scope and n_selected > 0:
            sel_names = [os.path.basename(p) for p in active_paths[:4]]
            suffix = f" +{n_selected - 4} more" if n_selected > 4 else ""
            lines.append(f"上下文选中: {', '.join(sel_names)}{suffix}")

    contextual_block = ContextualRefineSkill.render_prompt_block(ctx, include_active_scope=expose_active_scope)
    if contextual_block:
        lines.append(contextual_block)

    media_block = MediaFollowupSkill.render_prompt_block(ctx) if (expose_active_scope or last_results) else ""
    if media_block:
        lines.append(media_block)

    return "\n".join(lines)


def _build_decision_contract(lang: str) -> str:
    """High-signal routing guidance for ambiguous file intent decisions."""
    use_en = str(lang or "en").lower().startswith("en")
    if use_en:
        return (
            "[Decision Contract]\n"
            "- First identify four separate things: operation (search/count/summarize/qa), content target, file-type filter, and scope/filter. Do not let any one dimension erase the others.\n"
            "- Use this priority order: explicit quantity words -> count; explicit selected/current/these files understanding/comparison -> selected-scope summary; explicit prior-result rewrite/follow-up with no new topic -> contextual_refine; audio/video timestamp, interval, transcript, frame/screenshot, clip/export/convert operation -> media_export; audio/video internal content questions without a fixed time -> media_content_search; global all-files summary -> summarize_all; explicit find/search/retrieve/locate/show/list/display with a new file/topic target -> search; category-only listing -> search with category/extension; casual non-file text -> chat/clarify.\n"
            "- If active selected files exist and the user asks to tell/describe/explain/summarize/overview/compare/rank/connect the selected/current/this/these file(s), document(s), or item(s), choose summarize_selected or summarize_all scoped to the selection. Use contextual_refine for selected scope only when the user is rewriting/filtering/listing the selected set. Never search for the literal word 'selected'.\n"
            "- open_file is only for explicit OS/file-viewer commands such as 'open this file' or 'launch the document'. In selected/current/this-file context, 'show me this file' means describe/summarize the selected file unless the user explicitly says open/launch.\n"
            "- Treat words like 'all/every/my' as scope words, not as permission to ignore the topic.\n"
            "- Treat 'start over', 'ignore previous', and similar reset phrases as a fresh search/list request when the same query contains a new file type or topic. Do not route them to prior-result refinement or media follow-up.\n"
            "- In a prior-results conversation, qualitative follow-up questions that ask for a judgment, comparison, ranking, synthesis, or interpretation of the already found items should stay on the previous results. Phrases like 'who is more...', 'which is stronger at...', 'who fits better for...', 'compare their...', 'compared to...', or 'what is the difference/commonality...' are process_previous/contextual_refine QA unless the user explicitly says find/search/list new files.\n"
            "- In a prior-results conversation, ordinal references and short fragments are follow-ups, not new global searches: 'the first one', 'tell me about item 2', 'summarize the first video', 'holiday policy?', 'any mobile ones?', 'which columns are important?'. Use process_previous/contextual_refine QA or summary.\n"
            "- If the previous turn was a failed or clarified search with a real topic, and the current message supplies only a file type or scope clarification such as 'photos', 'looking for photos', 'PDFs', or 'only videos', combine the prior topic with the new type and return search. Example: prior query 'show all red bicycle', current 'looking for photos' -> {\"action\":\"search\",\"params\":{\"query\":\"red bicycle\",\"category\":\"image\"}}. Do not ask another clarification.\n"
            "- Compound questions over a prior result, such as 'how many items are listed and what categories do they cover', are previous-result QA. Do not turn the nouns into a fresh global search.\n"
            "- active_paths/candidate scopes are available context, not automatic intent. Do not choose contextual_refine(selected_items) just because many files are active. Use selected/current/this/these/that wording, an explicit 'selected files/folder' phrase, or a real prior-results follow-up before choosing selected scope.\n"
            "- For quantity questions that refer to prior results, a listed folder, or the current conversational location with words such as there, in there, those, them, these results, that folder, previous results, or current results, choose count and set params.scope='last_results' while preserving category/media_type filters. Example: 'how many videos are in there?' -> {\"action\":\"count\",\"params\":{\"category\":\"video\",\"media_type\":\"video\",\"scope\":\"last_results\"}}. Never count the full indexed corpus for these follow-ups.\n"
            "- For explicit find/search/retrieve/locate/show/list/display requests with a new topic or file type and no selected/current/this/these wording, choose search over contextual_refine, even if active_paths are present.\n"
            "- Media operations are not file inventory. If the user asks to inspect, extract, export, clip, convert, transcribe, caption, screenshot, or describe a timestamp/range of audio/video, choose media_export even when the wording starts with show/find/get and even when the target file must first be located.\n"
            "- For media_export, set params.sub_intent='point_lookup' for a single timestamp, 'range_summary' for a time range such as first N seconds or between A and B, and 'media_summary' for a whole-file media overview. Set target_type to audio_content, video_visual, or video_audio when clear.\n"
            "- If the user asks what an audio/video/recording/clip says, shows, mentions, contains, is about, or has inside regarding a topic, choose media_content_search. Example: 'what is in the video about refund policy' -> {\"action\":\"media_content_search\",\"params\":{\"query\":\"refund policy\",\"media_type\":\"video\"}}. Use ordinary search for media file inventory such as 'show my videos' and 'find videos about product launch'.\n"
            "- The verb matters for media topics: 'find/show/list videos about X' means retrieve video files; 'which videos mention X' or 'what does the video say/show about X' means search inside media content.\n"
            "- Media filename lookup is ordinary search, not media_content_search. If the user says find/show/list/locate a video, audio, image, clip, recording, or photo by filename-like text, camera-style id, extension, or title, choose search with the filename/id/title in params.query and category/media_type as a filter when useful. media_content_search is only for searching inside audio/video content by topic.\n"
            "- Media inventory requests stay on search even when they contain topical qualifiers. Queries like 'find recordings with rain sounds', 'bird song recordings', 'videos about product launch', or 'wav audio with ocean noise' are still file retrieval requests. The qualifier describes which files to retrieve, not an instruction to inspect media internals.\n"
            "- Document retrieval requests stay on search even when the topic contains media-domain words. Queries like 'search for papers about audio deep learning', 'find reports on video models', or 'PDFs about speech recognition' ask for documents; audio/video/speech are topic terms, not media files to inspect.\n"
            "- When a search query names a broader class and then gives examples with words like 'like', 'such as', 'including', or 'featuring', keep the class/topic in params.query and treat the examples as hints, not as hard requirements. A compact retrieval query may add close sibling terms from the same class when that improves recall.\n"
            "- For broad topical media inventory searches, params.query should be compact retrieval keywords, not the full natural-language sentence. Keep the topic head plus the strongest example terms, e.g. class + exemplars such as 'ambient sounds rain traffic wav'.\n"
            "- Still images are not audio/video content search. If the user asks 'what is in this image/photo/picture?' after a single image result, or asks to describe/explain the current image, choose process_previous/contextual_refine or a selected-scope summary, not media_content_search.\n"
            "- If the user asks to summarize, overview, digest, analyze, or explain all/everything/my files/my documents/my docs, choose summarize_all. Do not turn 'summarize all my documents' into a search query.\n"
            "- For 'show/list/display everything/files under/in/inside a folder', choose search scoped to that folder. Do not choose count unless the user asks for a quantity.\n"
            "- If a category phrase is followed by a content qualifier such as 'of', 'with', 'containing', 'contains', 'about', 'that mention', 'featuring', or 'showing', choose search. Keep the qualifier target in params.query and set params.category when obvious. Do not choose count or a broad inventory.\n"
            "- If the query has a folder qualifier such as 'in/inside/under/from the folder ...', keep that as params.folder when the schema supports it, and still keep the content target in params.query. Folder scope plus category is not global category inventory.\n"
            "- If the user explicitly names a file extension or format token such as pdf, csv, xlsx, wav, mp3, mp4, mov, png, jpg, or jpeg, keep the topic words in params.query and also set params.file_extensions when helpful. Do not drop the topic just because an extension is present.\n"
            "- If the user names a file-type noun such as papers, articles, reports, photos/images, audio, video, spreadsheets, invoices, resumes, manuals, books, or code together with a topic, preserve BOTH: put the topic in params.query and the file type in params.category when clear.\n"
            "- Treat 'file', 'files', 'document', and 'documents' as generic container words unless the user clearly asks for a document inventory or a specific document format. Do not set category='document' merely because those words appear. For title-like or catalog/list/table requests, leaving category empty is often safer than filtering out data/spreadsheet-like PDFs.\n"
            "- If the query contains a proper noun, filename-like phrase, invoice/order number, amount, product code, or title-like phrase, treat that as the retrieval anchor. Do not drop it just because generic words like file/document/report/image/video/data are also present.\n"
            "- Examples must stay JSON-shaped: 'find all images of a red bicycle' -> {\"action\":\"search\",\"params\":{\"query\":\"red bicycle\",\"category\":\"image\"}}; 'show all photos with bicycles' -> {\"action\":\"search\",\"params\":{\"query\":\"bicycles\",\"category\":\"image\"}}; 'find images of a red bicycle' -> {\"action\":\"search\",\"params\":{\"query\":\"red bicycle\",\"category\":\"image\"}}.\n"
            "- params.category is a filter bucket, not a place for topical phrases. Use it only for clear indexed file/media/type buckets such as image, document, data/spreadsheet, presentation, audio, video, resume, invoice, manual, report, paper, book, or code. If unsure, leave category empty and keep the words in params.query.\n"
            "- If the query includes a strong anchor such as a product/model code, identifier, exact title phrase, or filename-like text, do not narrow it to category=document/manual/report/paper unless the user is explicitly asking for a broad type inventory. Keep the anchor in params.query; leaving category empty is usually better than filtering out the target.\n"
            "- Keep every meaningful subject/type noun in params.query. Never collapse a retrieval query to only a command verb. For example: 'retrieve all my pdf invoices' -> {\"action\":\"search\",\"params\":{\"query\":\"pdf invoices\",\"file_extensions\":\"pdf\"}}; 'find the business plan about smart home sensors' -> {\"action\":\"search\",\"params\":{\"query\":\"business plan smart home sensors\"}} unless the user explicitly asks for a narrow report/document inventory.\n"
            "- For title-like document requests, preserve the complete title phrase in params.query. If the user says 'market expansion analysis report', keep 'market expansion analysis report'; do not reduce it to 'market expansion' or category='document'. If the user explicitly says report, category='report' is usually better than category='document'.\n"
            "- A bare inventory has no content target after the file type: 'find all images', 'show my screenshots', 'all PDFs'. Choose search with the category/extension; use count only for explicit quantity wording such as 'how many', 'count', 'number of', or 'total'.\n"
            "- Phrases like 'find a few papers', 'show several reports', or similar retrieval commands ask for matching results, not a count. Return search unless the user explicitly asks 'how many', 'number of', 'count', or 'total'.\n"
            "- If exactly one file is selected and the user says 'this file', 'it', or 'current file' with show/tell/read/describe/summarize/compare/explain, choose summarize_selected or summarize_all scoped to the selection. Never choose count or search for that.\n"
            "- Use contextual_refine list only for true selected-item inventory, e.g. 'list selected files', 'which selected files', or 'show selected files'.\n"
            "- Preserve important nouns after prepositions. Do not rewrite 'images of a red bicycle' to just 'images'.\n"
        )
    return (
        "[决策契约]\n"
        "- 先分别识别四件事：操作类型（search/count/summarize/qa）、内容目标、文件类型过滤、范围/文件夹过滤。任何一个维度都不能把其他维度抹掉。\n"
        "- 判断优先级：明确数量词 -> count；明确问 selected/current/this/these/选中/当前/这个/这些 文件的理解/比较/关联 -> 选中范围 summary；明确基于上一轮结果做改写/细化/追问且没有新主题 -> contextual_refine；音视频时间点/时间段/转写/截帧/截图/剪辑/导出/转换操作 -> media_export；没有固定时间点的音视频内部内容问题 -> media_content_search；全库/所有文件总结 -> summarize_all；明确 find/search/retrieve/locate/show/list/display/找/搜索/列出/展示 且有新的文件/主题目标 -> search；只有类别/格式清单 -> search + category/extension；普通闲聊 -> chat/clarify。\n"
        "- 如果当前有选中文件，用户要求 tell/describe/explain/summarize/overview/compare/rank/connect/讲讲/介绍/描述/总结/比较/联系 selected/current/this/these/选中/当前/这个/这些 文件或文档，选择 summarize_selected 或 summarize_all 且限定在 selected。只有在用户要改写/筛选/列出选中集时，才使用 contextual_refine。绝不要把 selected/选中 当成关键词去 search。\n"
        "- open_file 只用于用户明确要求用系统/文件查看器打开文件，例如 'open this file'、'打开这个文件'、'launch the document'。在 selected/current/this-file/这个文件 的上下文里，'show me this file'、'给我看这个文件' 默认是描述/总结选中文件，除非用户明确说 open/打开/launch。\n"
        "- 'all/every/my/所有/全部/我的' 只是范围词，不能因此丢掉用户真正要找的主题。\n"
        "- 'start over'、'ignore previous'、'重新/从头/忽略之前' 这类重置表达，如果同一句里带了新的文件类型或主题，应当视为新的 search/list；不要继续上轮结果，也不要走 media follow-up。\n"
        "- 在已有结果的对话里，用户问判断、比较、排序、综合或解释已有对象时，应继续处理上一轮结果。比如 'who is more...'、'which is stronger at...'、'who fits better for...'、'compare their...'、'compared to...'、'区别/共同点/谁更适合/谁更偏向...'，除非用户明确说 find/search/list/找/搜索新的文件，否则选择 process_previous/contextual_refine QA。\n"
        "- 在已有结果的对话里，序号引用和短片段问题是追问，不是新的全局搜索：'the first one'、'tell me about item 2'、'summarize the first video'、'holiday policy?'、'any mobile ones?'、'第一个文件/第一篇/讲讲第一个文件'。选择 process_previous/contextual_refine QA 或 summary。\n"
        "- 如果上一轮是一个失败或已澄清的搜索，并且上一轮有真实主题，而当前用户只补充文件类型或范围，例如 'photos'、'looking for photos'、'PDFs'、'only videos'、'找图片'，应把上一轮主题和当前类型合并后返回 search。例：上一轮 'show all red bicycle'，当前 'looking for photos' -> {\"action\":\"search\",\"params\":{\"query\":\"red bicycle\",\"category\":\"image\"}}。不要再次澄清。\n"
        "- 对上一轮结果的复合问题，例如 'how many items are listed and what categories do they cover'，属于上一轮结果问答；不要把其中名词变成新的全局搜索。\n"
        "- active_paths / candidate scopes 只是可用上下文，不代表用户一定在问选区。不要因为当前有很多 active_paths 就选择 contextual_refine(selected_items)。只有用户明确说 selected/current/this/these/that、选中文件/选中文件夹，或确实是在追问上一轮结果时，才选 selected scope。\n"
        "- 对指向上一轮结果、刚列出的文件夹、当前对话位置或当前主题的数量问题，例如 there / in there / those / them / these results / that folder / previous results / current results / this topic / 里面 / 那里 / 这些 / 它们 / 上面结果 / 那个文件夹 / 这方面 / 这类，选择 count，并设置 params.scope='last_results'，同时保留 category/media_type 过滤。例如 'how many videos are in there?' -> {\"action\":\"count\",\"params\":{\"category\":\"video\",\"media_type\":\"video\",\"scope\":\"last_results\"}}。这类追问绝不能统计全库。\n"
        "- 对带新主题或文件类型的明确 find/search/retrieve/locate/show/list/display/找/搜索/检索/列出/展示 请求，如果没有 selected/current/this/these/这些/选中 这类范围词，选择 search，不要选择 contextual_refine。\n"
        "- 媒体操作不是文件清单。如果用户要求 inspect/extract/export/clip/convert/transcribe/caption/screenshot，或询问音视频某个时间点/时间段的内容，选择 media_export；即使句子以 show/find/get/显示/找 开头，或目标文件还需要先定位，也不要退化为普通 search。\n"
        "- media_export 参数：单一时间点用 sub_intent='point_lookup'，时间范围/前 N 秒/从 A 到 B 用 sub_intent='range_summary'，整个媒体概览用 sub_intent='media_summary'；能判断时填写 target_type=audio_content/video_visual/video_audio。\n"
        "- 如果用户询问音频/视频/录音/录屏/clip 中关于某主题说了什么、展示了什么、提到什么、包含什么、里面有什么，选择 media_content_search。例如 'what is in the video about refund policy' -> {\"action\":\"media_content_search\",\"params\":{\"query\":\"refund policy\",\"media_type\":\"video\"}}。媒体文件清单，如 'show my videos'、'find videos about product launch'，走普通 search。\n"
        "- 媒体主题要看动词：'find/show/list videos about X' 是检索视频文件；'which videos mention X' 或 'what does the video say/show about X' 才是检索媒体内部内容。\n"
        "- 媒体文件名查找是普通 search，不是 media_content_search。用户说 find/show/list/locate/找/展示 某个视频、音频、图片、clip、recording、photo，且后面是像文件名、相机编号、扩展名或标题的文本时，选择 search，params.query 保留文件名/编号/标题，必要时用 category/media_type 过滤。media_content_search 只用于按主题检索音频/视频内容内部。\n"
        "- 媒体清单请求即使带主题限定，也仍然是 search。像“找带雨声的录音”“鸟叫自然录音”“关于产品发布的视频”“含有海浪声的 wav 音频”这类请求，本质是按主题筛选媒体文件清单，而不是去媒体内部做内容问答。\n"
        "- 文档检索即使主题里有媒体领域词也走 search。例如 'search for papers about audio deep learning'、'find reports on video models'、'PDFs about speech recognition'、'找语音识别相关论文'，目标是文档，audio/video/speech 只是主题词。\n"
        "- 如果搜索语句先说一个大类/主题，再用 like / such as / including / featuring / 比如 / 例如 等给例子，params.query 必须保留这个大类/主题，把例子当作提示而不是硬性约束；必要时可以补入同类近义提示词来提升召回。\n"
        "- 对宽泛的媒体文件搜索，params.query 应该是紧凑的检索关键词，而不是整句自然语言。保留主题主干和最强的例子词即可，例如 'ambient sounds rain traffic wav' 这种“主题 + 代表性例子”的形式。\n"
        "- 静态图片不是音视频内容检索。如果上一轮只有一张图片结果，用户问 'what is in this image/photo/picture?'、'describe this image'、'这张图里有什么/描述这张图'，选择 process_previous/contextual_refine 或 selected 范围 summary，不要选 media_content_search。\n"
        "- 如果用户要求 summarize/overview/digest/analyze/explain all/everything/my files/my documents/my docs，选择 summarize_all。不要把 'summarize all my documents' 当成 search query。\n"
        "- 对 'show/list/display everything/files under/in/inside a folder' 或“列出/展示某文件夹下的所有文件”，选择限定到该文件夹的 search；只有明确问数量时才选 count。\n"
        "- 如果文件类别后面带内容限定，例如 of / with / containing / contains / about / that mention / featuring / showing，或中文的“关于/包含/有/带有/里面有/出现/展示”，请选择 search。params.query 保留限定目标，能判断类别时设置 params.category；不要选 count，也不要走全量清单。\n"
        "- 如果查询里有文件夹范围，例如 in/inside/under/from the folder ...，或中文“在/里面/文件夹下/来自...文件夹”，能写 params.folder 时保留文件夹范围，同时 params.query 仍保留内容目标。文件夹范围 + 类别过滤不等于全库类别清单。\n"
        "- 如果用户明确说了扩展名或格式词，如 pdf、csv、xlsx、wav、mp3、mp4、mov、png、jpg、jpeg，params.query 保留主题词，同时在合适时填写 params.file_extensions；不要因为有扩展名就丢掉主题。\n"
        "- 如果用户同时说了文件类型名和主题，例如 papers/articles/reports/photos/images/audio/video/spreadsheets/invoices/resumes/manuals/books/code，或论文/文章/报告/图片/音频/视频/表格/发票/简历/手册/书籍/代码，必须同时保留：主题放进 params.query，音频用 category='audio'，视频用 category='video'。\n"
        "- file/files/document/documents/文件/文档 通常只是泛容器词，除非用户明确是在要文档类清单或具体文档格式，否则不要仅因为这些词出现就设置 category='document'。标题式、目录式、列表/表格/catalog 请求里，category 留空通常比把数据型 PDF 误过滤掉更安全。\n"
        "- 如果查询包含专名、像文件名的短语、发票/订单号、金额、产品型号、或像标题的短语，把它当成检索锚点。不要因为同时出现 file/document/report/image/video/data 等泛词就丢掉锚点。\n"
        "- 示例必须保持 JSON 形状：'find all images of a red bicycle' -> {\"action\":\"search\",\"params\":{\"query\":\"red bicycle\",\"category\":\"image\"}}；'show all photos with bicycles' -> {\"action\":\"search\",\"params\":{\"query\":\"bicycles\",\"category\":\"image\"}}；'find images of a red bicycle' -> {\"action\":\"search\",\"params\":{\"query\":\"red bicycle\",\"category\":\"image\"}}。\n"
        "- params.category 是过滤桶，不是业务主题。只在明确是可过滤的文件/媒介/文档类型时使用，例如 image、document、data/spreadsheet、presentation、audio、video、resume、invoice、manual、report、paper、book、code；不确定时留空，把词保留在 params.query。\n"
        "- 如果查询里已经有很强的检索锚点，例如产品/型号代码、编号、精确标题短语、或像文件名的文本，不要再轻易缩成 category=document/manual/report/paper，除非用户明确是在要一个宽泛的类型清单。优先把锚点完整留在 params.query；category 留空通常比误过滤更安全。\n"
        "- params.query 必须保留所有有意义的主题词和类型词，绝不能只剩命令动词。例如 'retrieve all my pdf invoices' -> {\"action\":\"search\",\"params\":{\"query\":\"pdf invoices\",\"file_extensions\":\"pdf\"}}；'find the business plan about smart home sensors' -> {\"action\":\"search\",\"params\":{\"query\":\"business plan smart home sensors\"}}，除非用户明确要窄的 report/document 清单，否则不要把 business plan 塞进 category。\n"
        "- 对像标题的文档检索，params.query 保留完整标题短语。用户说 'market expansion analysis report' 时，保留完整短语，不要缩成 'market expansion' 或只给 category='document'；明确说 report 时，category='report' 通常比 category='document' 更合适。\n"
        "- 真正的清单/库存查询没有内容目标：'find all images'、'show my screenshots'、'all PDFs'。这类选择 search + category/extension；只有明确问数量（how many / count / number of / total / 有多少 / 统计）才选择 count。\n"
        "- 如果当前只选中 1 个文件，用户说 this file / it / current file / 这个文件 / 这份，并要求 show/tell/read/describe/summarize/compare/explain/给我看/讲讲/总结/解释，选择 summarize_selected 或 summarize_all 且限定在 selected；绝不要选 count 或 search。\n"
        "- 只有真正问选中项清单时才用 contextual_refine list，例如 'list selected files'、'which selected files'、'show selected files'、'列出选中文件'。\n"
        "- 保留介词/限定词后的关键词，不要把 'images of a red bicycle' 改写成只有 'images'。\n"
    )


def _build_routing_examples(lang: str) -> str:
    use_en = str(lang or "en").lower().startswith("en")
    if use_en:
        return (
            "[Routing Examples]\n"
            "- \"look for WAV recordings of ambient sounds like rain or traffic\" -> {\"action\":\"search\",\"params\":{\"query\":\"ambient sounds rain traffic wav\",\"category\":\"audio\",\"media_type\":\"audio\",\"file_extensions\":\"wav\"}}\n"
            "- \"find recordings with river sounds\" -> {\"action\":\"search\",\"params\":{\"query\":\"river sounds ambience\",\"category\":\"audio\",\"media_type\":\"audio\"}}\n"
            "- \"搜索带有雨声的wav录音\" -> {\"action\":\"search\",\"params\":{\"query\":\"rain ambience wav\",\"category\":\"audio\",\"media_type\":\"audio\",\"file_extensions\":\"wav\"}}\n"
            "- \"find the video IMG 2048\" -> {\"action\":\"search\",\"params\":{\"query\":\"IMG 2048\",\"category\":\"video\",\"media_type\":\"video\"}}\n"
            "- \"find the audio file 04 rooftop rain\" -> {\"action\":\"search\",\"params\":{\"query\":\"04 rooftop rain\",\"category\":\"audio\",\"media_type\":\"audio\"}}\n"
            "- \"find videos about product launch\" -> {\"action\":\"search\",\"params\":{\"query\":\"product launch\",\"category\":\"video\",\"media_type\":\"video\"}}\n"
            "- \"which videos mention product launch\" -> {\"action\":\"media_content_search\",\"params\":{\"query\":\"product launch\",\"media_type\":\"video\"}}\n"
            "- \"what happens at 1:20 in this video\" -> {\"action\":\"media_export\",\"params\":{\"query\":\"what happens at 1:20 in this video\",\"sub_intent\":\"point_lookup\",\"media_type\":\"video\",\"target_type\":\"video_visual\",\"time_sec\":80}}\n"
            "- \"summarize the first 30 seconds of this audio\" -> {\"action\":\"media_export\",\"params\":{\"query\":\"summarize the first 30 seconds of this audio\",\"sub_intent\":\"range_summary\",\"media_type\":\"audio\",\"target_type\":\"audio_content\",\"time_sec\":0,\"time_end_sec\":30}}\n"
            "- \"extract a frame at 10 seconds from the video\" -> {\"action\":\"media_export\",\"params\":{\"query\":\"extract a frame at 10 seconds from the video\",\"sub_intent\":\"point_lookup\",\"media_type\":\"video\",\"target_type\":\"video_visual\",\"time_sec\":10}}\n"
            "- \"what does the video say about refund policy\" -> {\"action\":\"media_content_search\",\"params\":{\"query\":\"refund policy\",\"media_type\":\"video\"}}\n"
            "- after prior file results, \"which one sounds most relaxing based on the description\" -> {\"action\":\"process_previous\",\"params\":{}}\n"
            "- \"find all images of a red bicycle\" -> {\"action\":\"search\",\"params\":{\"query\":\"red bicycle\",\"category\":\"image\"}}\n"
        )
    return (
        "[路由示例]\n"
        "- \"look for WAV recordings of ambient sounds like rain or traffic\" -> {\"action\":\"search\",\"params\":{\"query\":\"ambient sounds rain traffic wav\",\"category\":\"audio\",\"media_type\":\"audio\",\"file_extensions\":\"wav\"}}\n"
        "- \"find recordings with river sounds\" -> {\"action\":\"search\",\"params\":{\"query\":\"river sounds ambience\",\"category\":\"audio\",\"media_type\":\"audio\"}}\n"
        "- \"搜索带有雨声的wav录音\" -> {\"action\":\"search\",\"params\":{\"query\":\"rain ambience wav\",\"category\":\"audio\",\"media_type\":\"audio\",\"file_extensions\":\"wav\"}}\n"
        "- \"find the video IMG 2048\" -> {\"action\":\"search\",\"params\":{\"query\":\"IMG 2048\",\"category\":\"video\",\"media_type\":\"video\"}}\n"
        "- \"find the audio file 04 rooftop rain\" -> {\"action\":\"search\",\"params\":{\"query\":\"04 rooftop rain\",\"category\":\"audio\",\"media_type\":\"audio\"}}\n"
        "- \"find videos about product launch\" -> {\"action\":\"search\",\"params\":{\"query\":\"product launch\",\"category\":\"video\",\"media_type\":\"video\"}}\n"
        "- \"which videos mention product launch\" -> {\"action\":\"media_content_search\",\"params\":{\"query\":\"product launch\",\"media_type\":\"video\"}}\n"
        "- \"what happens at 1:20 in this video\" -> {\"action\":\"media_export\",\"params\":{\"query\":\"what happens at 1:20 in this video\",\"sub_intent\":\"point_lookup\",\"media_type\":\"video\",\"target_type\":\"video_visual\",\"time_sec\":80}}\n"
        "- \"summarize the first 30 seconds of this audio\" -> {\"action\":\"media_export\",\"params\":{\"query\":\"summarize the first 30 seconds of this audio\",\"sub_intent\":\"range_summary\",\"media_type\":\"audio\",\"target_type\":\"audio_content\",\"time_sec\":0,\"time_end_sec\":30}}\n"
        "- \"extract a frame at 10 seconds from the video\" -> {\"action\":\"media_export\",\"params\":{\"query\":\"extract a frame at 10 seconds from the video\",\"sub_intent\":\"point_lookup\",\"media_type\":\"video\",\"target_type\":\"video_visual\",\"time_sec\":10}}\n"
        "- \"what does the video say about refund policy\" -> {\"action\":\"media_content_search\",\"params\":{\"query\":\"refund policy\",\"media_type\":\"video\"}}\n"
        "- 上一轮已有文件结果时，\"which one sounds most relaxing based on the description\" -> {\"action\":\"process_previous\",\"params\":{}}\n"
        "- \"find all images of a red bicycle\" -> {\"action\":\"search\",\"params\":{\"query\":\"red bicycle\",\"category\":\"image\"}}\n"
    )


def _build_skill_block(ctx: Any, lang: str) -> str:
    """Render active skills as a compact block for the prompt."""
    from tools.intent_registry import IntentRegistry
    return IntentRegistry.render_compact_block(language=lang, ctx=ctx)


class SkillDispatcher:
    """
    Single LLM call intent router using IntentRegistry-driven skill descriptions.

    Usage:
        result = SkillDispatcher.dispatch(ctx)
        # result = {"action": "search", "params": {"query": "..."}}
    """

    @classmethod
    def dispatch(cls, ctx: Any) -> dict:
        """
        Main entry point. Makes ONE LLM call with dynamic skill context.

        Args:
            ctx: IntentContext with .question, .history, .last_results,
                 .active_paths, .prompt_language, .llm_service

        Returns:
            Intent dict: {"action": str, "params": dict, "confidence": float}
        """
        qn = (ctx.question or "").strip()
        # Keep the middle-layer routing contract in English across user languages.
        lang = "en"

        if not qn:
            return {"action": "chat", "params": {}, "confidence": 0.5}

        if _looks_like_non_action_utterance(qn):
            logger.info("[SkillDispatcher] pure non-action utterance → clarify")
            return {
                "action": "clarify",
                "params": {
                    "question": _non_action_clarify_message(ctx, qn),
                    "_clarify_kind": "non_action_utterance",
                },
                "confidence": 0.99,
            }

        # ── Pre-LLM fast gate: deterministic continuation commands ──
        if getattr(ctx, "history", None) or getattr(ctx, "last_results", None):
            from core.intent.continuation_agent import ExplicitContinuationExpert
            explicit = ExplicitContinuationExpert.analyze(qn, list(getattr(ctx, "history", None) or []))
            if explicit is not None:
                logger.info(f"[SkillDispatcher] explicit continuation fast-gate → {explicit.get('action')}")
                return explicit

        # Build prompt components
        context_hint = _build_context_hint(ctx, lang=lang)
        skill_block = _build_skill_block(ctx, lang)

        # Assemble the final prompt
        prompt = cls._build_prompt(lang, context_hint, skill_block, qn)

        logger.info(
            f"[SkillDispatcher] prompt ({len(prompt)} chars), "
            f"query_chars={len(qn)}, lang={lang}"
        )

        # Single LLM call
        llm_service = getattr(ctx, "llm_service", None)
        if not llm_service:
            logger.warning("[SkillDispatcher] no llm_service → search fallback")
            return {"action": "search", "params": {"query": qn}, "confidence": 0.5}

        try:
            response = llm_service.generate(
                prompt,
                history=[],
                system_prompt=None,
            )
            raw = (response or "").strip()
            result = cls._parse_response(raw, qn)
            result = cls._annotate_result(ctx, result)
            result["confidence"] = max(float(result.get("confidence", 0.0)), 0.82)
            logger.info(
                f"[SkillDispatcher] → action={result.get('action')!r} "
                f"params={result.get('params')!r}"
            )
            return result
        except Exception as e:
            logger.error(f"[SkillDispatcher] LLM failed: {e}", exc_info=True)
            return {"action": "search", "params": {"query": qn}, "confidence": 0.5}

    @classmethod
    def _build_prompt(cls, lang: str, context_hint: str, skill_block: str, query: str) -> str:
        """Assemble the skill dispatch prompt."""
        lang = "en"
        return (
            "You are a file retrieval assistant. Choose ONE skill from the available skills.\n\n"
            "Use English as the internal routing language even when the user writes in another language.\n\n"
            f"[Context]\n{context_hint}\n\n"
            f"[Available Skills]\n{skill_block}\n\n"
            f"{_build_decision_contract(lang)}\n\n"
            f"{_build_routing_examples(lang)}\n\n"
            f"[User] {query}\n\n"
            "[Execution Contract]\n"
            "- Return JSON only: {\"action\": \"...\", \"params\": {...}, \"reason\": \"...\"}\n"
            "- Do not output prose, bullet points, tool-call syntax, or shorthand. The entire response must be one valid JSON object.\n"
            "- Do NOT invent new skill names.\n"
            "- For action=contextual_refine, fill params.scope with one candidate scope when possible, and params.operation with one of: list, summary, qa, rewrite, support.\n"
            "- For action=media_followup, fill params.operation with one of: topic_search, time_lookup, range_summary, summary, rewrite.\n"
            "- For action=media_export, fill params.sub_intent, media_type, target_type, and time_sec/time_end_sec when the user gave a timestamp or interval.\n"
            "- For action=search, params.query should be concise English retrieval keywords when the user asked in Chinese.\n"
            "- Preserve the real retrieval anchor in params.query; translate or normalize it into concise English when helpful, but do not drop names, IDs, model numbers, filenames, or topic nouns.\n"
            "- A bare proper name, entity, product, company, or topic is a search request; set params.query to the canonical entity/topic, not clarify.\n"
            "- When the user asks for information about a specific entity/topic, choose search and keep params.query concise.\n"
            "- If the user asks to explain/analyze/describe a specific topic, table, report, product, or entity, choose search, not summarize_all. summarize_all is only for broad all-files overviews.\n"
            "- Audio/video content questions by topic use media_content_search, not search: if the user asks what a video/audio says, shows, mentions, contains, or is about regarding a topic, return media_content_search with the topic in params.query and media_type when clear.\n"
            "- But find/show/list videos or audio files about a topic is file retrieval: return search with category='video' for videos and category='audio' for audio. Use media_content_search only for inside-content wording such as mention/say/show/contain/transcript/scene.\n"
            "- Audio/video operations use media_export, not search: timestamp lookup, first/last N seconds, between A and B, frame/screenshot, transcript/caption, clip/export/convert.\n"
            "- Document retrieval stays on search even when the subject contains media terms: 'search for papers about audio deep learning', 'find reports on video models', or 'PDFs about speech recognition' retrieve documents; media terms are the topic.\n"
                "- Treat 'file', 'files', 'document', and 'documents' as generic container words unless the user asks for a document inventory or a concrete document format. Preserve title/list/catalog/table words in params.query and avoid category='document' when that filter could exclude data-like PDFs.\n"
                "- Filename or inventory lookup for media files is search, not media_content_search. If the user says find/show/list/locate a video/audio/image/photo/clip/recording by filename-like text, camera-style id, extension, or title, return search and preserve that anchor in params.query.\n"
                "- media_content_search is only for audio/video internals. For a still image follow-up like 'what is in this image/photo/picture?' after a prior image result, return process_previous or contextual_refine with operation='qa' instead of media_content_search.\n"
                "- Global overview wording uses summarize_all: 'summarize all my documents', 'overview of all files', 'digest everything', or 'analyze my docs' are not search queries.\n"
                "- Exception: if that specific thing is explicitly the selected/current/this document or file, choose summarize_selected or summarize_all scoped to the selection, not search.\n"
                "- open_file is only for explicit OS/file-viewer commands such as 'open this file' or 'launch the document'. In selected/current/this-file context, 'show me this file' means summarize/describe the selection unless the user explicitly says open/launch.\n"
                "- If the user asks to show/list/display files of a type or category (photos, presentations, data files, CSVs, resumes, invoices), choose search, not count. count is only for explicit quantities such as 'how many'.\n"
                "- When a file-type noun and a topic appear together, keep both dimensions: topic in params.query, file type in params.category when clear. Examples of file-type nouns include papers/articles/reports/images/audio/video/spreadsheets/invoices/resumes/manuals/books/code.\n"
                "- When the user has selected files and asks about their content, comparison, ranking, or relationships ('tell me about the selected document', 'tell me this', 'show me this', 'which selected file is most detailed', 'what connections can you infer between these selected files'), choose summarize_selected or summarize_all scoped to the selection — not search and not count.\n"
            "- For count questions that refer to prior results, a just-listed folder, or the current topic ('how many videos are in there?', 'how many of those are PDFs?', 'how many documents about this topic?', 'count images in these results'), return count with params.scope='last_results' and keep category/media_type filters. Do not count the whole indexed corpus.\n"
                "- If the query uses a pronoun or prior-result reference (his/her/their/this/that/above), preserve the attribute/topic in params.query; downstream search will resolve it with context.\n"
                "- Prefer a context-bound skill over a fresh global search when the user is clearly staying inside the current scope.\n"
                "- After prior results or a prior comparison, qualitative questions such as 'who is more X', 'which one is stronger at Y', 'who fits better', 'compared to last quarter', or 'what differs between them' should use process_previous or contextual_refine with operation='qa'. Do not turn the quality words into a fresh global search unless the user explicitly asks to find/list/search files.\n"
                "- After prior results, ordinal references and short fragment questions such as 'tell me about the first one', 'summarize the first video', 'holiday policy?', or 'any mobile ones?' should use process_previous/contextual_refine, not search or clarify.\n"
                "- In a prior-results context without explicit selected/current wording, ordinary comparison, connection, synthesis, interpretation, conditional follow-up, or QA about those files can use contextual_refine. If the user explicitly says selected/current/these files, prefer summarize_selected or summarize_all scoped to that selected set.\n"
                "- Use media_followup/topic_search only when the user explicitly asks for timestamps, transcript/audio/video evidence, scenes, frames, or what was heard/said/shown.\n"
                "- Pure laughter or non-action acknowledgements such as \"hahaha\", \"lol\", \"no need\", \"never mind\", or \"哈哈哈\" are NOT file requests; choose chat or clarify, never summarize/count/search.\n"
            )
    @classmethod
    def _parse_response(cls, raw: str, fallback_query: str) -> dict:
        """Extract JSON intent from LLM response."""
        if not raw:
            return {"action": "search", "params": {"query": fallback_query}}

        shorthand = cls._parse_shorthand_response(raw)
        if shorthand:
            return shorthand

        # Find JSON object
        start = raw.find("{")
        if start < 0:
            # Try to extract action word from plain text
            word = raw.strip().lower().split()[0] if raw.strip() else ""
            if word in _ACTION_MAP:
                return {"action": _ACTION_MAP[word], "params": {"query": fallback_query}}
            logger.warning(f"[SkillDispatcher] no JSON found: raw_chars={len(raw or '')}")
            return {"action": "search", "params": {"query": fallback_query}}

        try:
            decoder = json.JSONDecoder()
            result, _ = decoder.raw_decode(raw, start)
            if not isinstance(result, dict):
                return {"action": "search", "params": {"query": fallback_query}}

            action = str(result.get("action") or "search").strip()
            params = result.get("params") or {}
            if not isinstance(params, dict):
                params = {}
            reason = str(result.get("reason") or "").strip()
            if reason:
                params.setdefault("_dispatch_reason", reason[:240])

            # Normalize legacy/internal action names
            if action == "fallback_to_file_op":
                action = "search"
                if not params.get("query"):
                    params["query"] = fallback_query

            # Ensure search has a query
            if action == "search" and not params.get("query"):
                params["query"] = fallback_query

            return {"action": action, "params": params}
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[SkillDispatcher] JSON parse failed: {e}, raw_chars={len(raw or '')}")
            shorthand = cls._parse_shorthand_response(raw)
            if shorthand:
                return shorthand
            return {"action": "search", "params": {"query": fallback_query}}

    @classmethod
    def _parse_shorthand_response(cls, raw: str) -> Optional[dict]:
        """Recover compact non-JSON output when the model already chose a valid action."""
        return cls._parse_function_style_response(raw) or cls._parse_colon_style_response(raw)

    @staticmethod
    def _parse_function_style_response(raw: str) -> Optional[dict]:
        """Recover common local-model shorthand like ``search(query="resumes")``.

        The prompt requires JSON, but small local models sometimes emit a
        function-call-looking string. Treating that as unparseable loses the
        model's actual distilled query and falls back to the whole user utterance.
        This parser is intentionally narrow and only accepts known action names
        with simple key=value arguments.
        """
        text = str(raw or "").strip()
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", text, flags=re.DOTALL)
        if not match:
            return None
        action = match.group(1).strip()
        if action not in _KNOWN_ACTIONS:
            return None

        args = match.group(2).strip()
        params = SkillDispatcher._parse_key_value_payload(args)
        if action == "search" and not params.get("query"):
            return None
        return {"action": action, "params": params}

    @staticmethod
    def _parse_colon_style_response(raw: str) -> Optional[dict]:
        """Recover local-model shorthand like ``search: query="dog images"``."""
        text = str(raw or "").strip()
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)\s*$", text, flags=re.DOTALL)
        if not match:
            return None
        action = match.group(1).strip()
        if action not in _KNOWN_ACTIONS:
            return None
        params = SkillDispatcher._parse_key_value_payload(match.group(2).strip())
        if action == "search" and not params.get("query"):
            return None
        return {"action": action, "params": params}

    @staticmethod
    def _parse_key_value_payload(payload: str) -> Dict[str, Any]:
        """Parse a deliberately tiny key=value or key: value payload surface."""
        text = str(payload or "").strip()
        params: Dict[str, Any] = {}
        if not text:
            return params

        if text.startswith("{") and text.endswith("}"):
            try:
                decoded = json.loads(text)
                if isinstance(decoded, dict):
                    inner = decoded.get("params")
                    if isinstance(inner, dict):
                        params = dict(inner)
                    else:
                        params = {str(k): v for k, v in decoded.items() if k != "action"}
                    return SkillDispatcher._normalize_shorthand_params(params)
            except json.JSONDecodeError:
                text = text[1:-1]

        for match in re.finditer(
            r"['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*[:=]\s*"
            r"(\[[^\]]*\]|\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|[^,}\n]+)",
            text,
            flags=re.DOTALL,
        ):
            key = match.group(1).strip()
            value = SkillDispatcher._parse_shorthand_value(match.group(2))
            if value not in (None, ""):
                params[key] = value
        return SkillDispatcher._normalize_shorthand_params(params)

    @staticmethod
    def _parse_shorthand_value(value: str) -> Any:
        text = str(value or "").strip().rstrip(",")
        if not text:
            return ""
        if text.startswith("["):
            try:
                decoded = json.loads(text)
                if isinstance(decoded, list):
                    return decoded
            except json.JSONDecodeError:
                items = re.findall(r"['\"]([^'\"]+)['\"]", text)
                return items
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            inner = text[1:-1]
            return inner.replace(r"\"", "\"").replace(r"\'", "'").strip()
        low = text.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low in {"null", "none"}:
            return None
        return text.strip()

    @staticmethod
    def _normalize_shorthand_params(params: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        inferred_extensions: List[str] = []
        for key, value in (params or {}).items():
            if key in {"file_extensions", "extensions"} and isinstance(value, list):
                cleaned = [str(item).strip().lstrip(".") for item in value if str(item).strip()]
                normalized[key] = ",".join(cleaned)
            elif key == "media_type":
                raw = str(value or "").strip().lower().lstrip(".")
                if raw in _AUDIO_EXTENSION_HINTS:
                    normalized[key] = "audio"
                    inferred_extensions.append(raw)
                elif raw in _VIDEO_EXTENSION_HINTS:
                    normalized[key] = "video"
                    inferred_extensions.append(raw)
                else:
                    normalized[key] = value
            else:
                normalized[key] = value
        if inferred_extensions and not normalized.get("file_extensions"):
            deduped: List[str] = []
            seen = set()
            for ext in inferred_extensions:
                if ext in seen:
                    continue
                seen.add(ext)
                deduped.append(ext)
            normalized["file_extensions"] = ",".join(deduped)
        return normalized

    @classmethod
    def _annotate_result(cls, ctx: Any, result: dict) -> dict:
        from core.skills import ContextualRefineSkill, MediaFollowupSkill

        payload = dict(result or {})
        params = dict(payload.get("params") or {})
        action = str(payload.get("action") or "").strip()
        if action:
            params.setdefault("_skill_name", action)

        if ContextualRefineSkill.supports_ctx(ctx):
            state = ContextualRefineSkill.build_state(ctx)
            scope_ids = [scope.scope_id for scope in state.candidate_scopes]
            if scope_ids:
                params.setdefault("_candidate_scopes", scope_ids)
            if state.focused_file:
                params.setdefault("focused_file", state.focused_file)

        if MediaFollowupSkill.supports_ctx(ctx):
            media_state = MediaFollowupSkill.build_state(ctx)
            if media_state.focused_file:
                params.setdefault("file_hint", media_state.focused_file)

        payload["params"] = params
        return payload
