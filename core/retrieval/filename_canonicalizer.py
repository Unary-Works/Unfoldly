from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

_SPACE_RE = re.compile(r"\s+")
_FILELIKE_RE = re.compile(
    r'(?<![@/])([A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff_\-().（）,，、& ]{0,120}\.[A-Za-z0-9]{2,8})(?![@/])'
)
_MARKED_FILELIKE_PATTERNS = (
    re.compile(
        r'^\s*(?:please\s+|pls\s+|can you\s+|could you\s+|would you\s+)?'
        r'(?:find|show|open|locate|get|search(?:\s+for)?|look for|tell(?:\s+me)?\s+about)\s+'
        r'(?:(?:the|this|that)\s+)?'
        r'(?:(?:file\s+name|filename|file|document|doc|image|photo|picture|spreadsheet|table|audio|video)\s+){0,3}'
        r'([A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff_\-().（）,，、& ]{0,120})\s*$',
        re.IGNORECASE,
    ),
    re.compile(
        r'^\s*(?:帮我|给我|请|麻烦你|麻烦)?\s*(?:找|打开|查找|搜索|看看|看下)\s*'
        r'(?:这张|这个|这份)?(?:图片|照片|文件名|文件|文档|表格|音频|视频)?\s*'
        r'([A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff_\-().（）,，、& ]{0,120})\s*'
        r'(?:这个)?(?:文件名|文件|文档|图片|照片|表格|音频|视频)?\s*$',
        re.IGNORECASE,
    ),
)
_FILELIKE_PREFIX_NOISE = {
    "a", "an", "the", "all", "any", "some", "about", "find", "show", "open", "locate", "get", "search",
    "look", "for", "tell", "me", "compare", "comparison", "please", "pls", "can", "you", "could", "would",
    "help", "now", "switch", "to", "instead",
    "file", "files", "document", "documents", "doc", "docs", "image", "images",
    "photo", "photos", "picture", "pictures", "spreadsheet", "spreadsheets",
    "table", "tables", "audio", "audios", "video", "videos", "找", "打开", "查找", "搜索", "看看", "看下",
    "请", "帮我", "给我", "关于", "文件", "文档", "图片", "照片", "表格", "音频", "视频",
}
_WEAK_FILELIKE_TERMS = {
    "file", "files", "document", "documents", "doc", "docs", "image", "images",
    "photo", "photos", "picture", "pictures", "spreadsheet", "table",
    "tables", "audio", "audios", "video", "videos", "recording", "recordings",
    "sound", "sounds", "music", "song", "songs", "track", "tracks",
    "content", "details", "first one",
    "第一个", "文件", "文档", "图片", "照片", "表格", "音频", "视频",
}
_REFERENCE_PLACEHOLDER_TERMS = {
    "it", "its", "them", "they", "this", "that", "these", "those",
    "him", "her", "his", "their", "previous", "previous one", "same one",
    "same file", "same files", "same document", "same documents",
    "它", "它们", "他", "她", "他们", "她们", "这个", "那个", "这些", "那些",
    "这份", "那份", "这个文件", "那个文件", "这些文件", "那些文件",
    "上一个", "上一份", "上一条", "上面的", "前面的", "同一个", "同一份",
}
_SINGULAR_REFERENCE_PLACEHOLDER_TERMS = {
    "it", "its", "this", "that", "him", "her", "his",
    "它", "他", "她", "这个", "那个", "这份", "那份", "这个文件", "那个文件",
    "上一个", "上一份", "上一条", "同一个", "同一份",
}
_PLURAL_REFERENCE_PLACEHOLDER_TERMS = {
    "them", "they", "these", "those", "their",
    "它们", "他们", "她们", "这些", "那些", "这些文件", "那些文件",
}
_DEICTIC_ACTION_RE = re.compile(
    r'^\s*(?:please\s+|pls\s+)?'
    r'(?:find|show(?:\s+me)?|open|display|browse|see|list|locate|get(?:\s+me)?|'
    r'search(?:\s+for)?|look\s+for|tell(?:\s+me)?\s+about|describe|explain|'
    r'summari(?:ze|se)|summary|overview|recap|read|what(?:\'s|\s+is)\s+(?:in|about)|what\s+are)\b'
    r'|^\s*(?:帮我|给我|请|麻烦你|麻烦)?\s*'
    r'(?:找|打开|查找|搜索|看看|看下|查看|列出|显示|介绍一下|介绍下|总结一下|总结下|概括一下|概括下|讲讲|说明一下|说说)\b',
    re.IGNORECASE,
)
_DEICTIC_NOUN_RE = re.compile(
    r'\b(this|that|these|those|the\s+previous|the\s+last|the\s+same)\s+'
    r'(file|files|document|documents|doc|docs|report|reports|guide|guides|article|articles|'
    r'profile|profiles|resume|resumes|video|videos|audio|audios|image|images|photo|photos|'
    r'result|results|item|items)\b'
    r'|'
    r'(这个|那个|这些|那些|上一个|上一份|同一个|同一份)'
    r'(文件|文档|报告|指南|资料|简历|视频|音频|图片|结果)?',
    re.IGNORECASE,
)
_EXPLICIT_FILENAME_FOCUS_RE = re.compile(
    r"\bfile\s*name\b|\bfilename\b|\bfiles?\s+named\b|\bnamed\b|文件名|文件名字|名为|名字为",
    re.IGNORECASE,
)
_EXPLICIT_FILE_SCOPE_RE = re.compile(
    r"\b(?:file|document|doc|image|photo|picture|spreadsheet|table|audio|video|clip|recording|song|music)\b"
    r"|文件|文档|图片|照片|表格|音频|视频|录音|歌曲|音乐",
    re.IGNORECASE,
)
_THEMATIC_LOOKUP_QUERY_RE = re.compile(
    r'^\s*(?:please\s+|pls\s+)?'
    r'(?:find|show(?:\s+me)?|open|display|browse|see|list|locate|get(?:\s+me)?|'
    r'search(?:\s+for)?|look\s+for|tell\s+me\s+about|describe|explain|summari(?:ze|se)|overview)\b'
    r'|^\s*(?:帮我|给我|请|麻烦你|麻烦)?\s*(?:找|打开|查找|搜索|看看|看下|查看|列出|显示|介绍一下|介绍下|总结一下|总结下|概括一下|概括下|讲讲|说明一下|说说)\b',
    re.IGNORECASE,
)
_RELATIVE_THAT_FOLLOW_RE = re.compile(
    r'^\s+(?:contain|contains|containing|have|has|having|include|includes|including|'
    r'mention|mentions|mentioning|show|shows|showing|display|displays|displaying|'
    r'look|looks|looking|talk|talking|speak|speaking|feature|features|featuring|'
    r'match|matches|matching|is|are|was|were|can|could|should|would|will|may)\b',
    re.IGNORECASE,
)
_THEMATIC_IT_IN_CONTENT_RE = re.compile(
    r"\b(?:find|show(?:\s+me)?|display|search(?:\s+for)?|look\s+for|locate|get(?:\s+me)?)\b"
    r".{0,80}\b(?:files?|documents?|docs?|images?|photos?|pictures?|videos?|audios?|media)\b"
    r".{0,80}\b(?:of|with|about|containing|contains?|featuring|showing|depicting|including)\b"
    r".{0,80}\bin\s+it\b",
    re.IGNORECASE,
)
_THEMATIC_LOOKUP_TERMS_EN = {
    "resume", "resumes", "cv", "cvs", "invoice", "invoices", "report", "reports",
    "contract", "contracts", "agreement", "agreements", "paper", "papers", "article",
    "articles", "document", "documents", "doc", "docs", "manual", "manuals", "guide",
    "guides", "profile", "profiles", "portfolio", "portfolios", "presentation",
    "presentations", "slides", "deck", "decks", "image", "images", "photo", "photos",
    "picture", "pictures", "audio", "audios", "video", "videos", "music", "song",
    "songs", "track", "tracks", "recording", "recordings", "screenshot", "screenshots",
    "chart", "charts",
}
_THEMATIC_LOOKUP_TERMS_ZH = (
    "简历", "履历", "发票", "账单", "报告", "合同", "协议", "论文", "文档",
    "资料", "手册", "指南", "名片", "课件", "演示文稿", "图片", "照片", "视频",
    "音频", "录音", "音乐", "歌曲", "截图", "图表",
)
_EXPLICIT_OPEN_FILE_ACTION_RE = re.compile(
    r"^\s*(?:please\s+|pls\s+|can you\s+|could you\s+|would you\s+)?(?:open|launch|reveal)\b"
    r"|^\s*(?:帮我|给我|请|麻烦你|麻烦)?\s*(?:打开|启动)\b",
    re.IGNORECASE,
)
_THEMATIC_DOT_PREPOSITION_RE = re.compile(
    r"\b(?:about|related\s+to|regarding|concerning)\b|关于|有关|相关",
    re.IGNORECASE,
)
_THEMATIC_DOT_DOC_CONTEXT_RE = re.compile(
    r"\b(?:documentation|docs?|manuals?|guides?|frameworks?|runtimes?|libraries?|sdks?|apis?)\b"
    r"|文档|资料|手册|指南|框架|运行时|库|接口",
    re.IGNORECASE,
)
_TOPIC_DOT_EXTENSIONS = {
    "c", "cc", "cpp", "cxx", "h", "hpp", "hh", "hxx",
    "js", "mjs", "cjs", "jsx", "ts", "tsx", "vue", "svelte",
    "py", "pyw", "go", "rs", "java", "kt", "kts", "scala",
    "cs", "fs", "fsx", "rb", "php", "swift", "m", "mm",
    "sh", "bash", "zsh", "fish", "ps1", "bat", "cmd",
    "lua", "r", "sql", "proto", "graphql", "gql", "wasm",
}
_DOMAIN_LIKE_STEM_SUFFIXES = {
    "com", "net", "org", "io", "ai", "dev", "app", "co",
    "cn", "hk", "tw", "jp", "kr", "uk", "us", "de", "fr",
    "studio", "media", "music", "audio", "video",
}


def clean_filename_space(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text or "").strip())


def compact_filename_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").strip().lower(), flags=re.UNICODE)


def is_structured_filename_fragment_key(key: str) -> bool:
    value = compact_filename_key(key)
    if len(value) < 6 or not any(ch.isdigit() for ch in value):
        return False
    if value.isdigit():
        return True
    return bool(re.search(r"[a-z]", value))


def filename_stem_key_matches_query(candidate_stem_key: str, query_stem_key: str) -> bool:
    candidate_key = compact_filename_key(candidate_stem_key)
    query_key = compact_filename_key(query_stem_key)
    if not candidate_key or not query_key:
        return False
    if candidate_key == query_key:
        return True
    return is_structured_filename_fragment_key(query_key) and query_key in candidate_key


def _collapse_repeated_surface(text: str) -> str:
    value = clean_filename_space(str(text or "").strip())
    if not value:
        return ""
    parts = value.split(" ")
    if len(parts) <= 1:
        return value
    for width in range(1, (len(parts) // 2) + 1):
        if len(parts) % width != 0:
            continue
        unit = parts[:width]
        if unit and all(parts[idx : idx + width] == unit for idx in range(0, len(parts), width)):
            return " ".join(unit)
    return value


def normalize_filename_candidate(text: str) -> str:
    raw = clean_filename_space(str(text or "").strip(" \"'“”‘’.,;:!?}>"))
    if not raw:
        return ""
    raw = os.path.basename(raw.replace("\\", "/"))
    parts = [part for part in raw.split(" ") if part]
    while len(parts) > 1:
        lead = re.sub(r"^[^A-Za-z0-9\u4e00-\u9fff]+|[^A-Za-z0-9\u4e00-\u9fff]+$", "", parts[0]).lower()
        if not lead or lead in _FILELIKE_PREFIX_NOISE:
            parts.pop(0)
            continue
        break
    value = clean_filename_space(" ".join(parts))
    value = re.sub(r'^(?:file\s*name|filename)\s+', '', value, flags=re.IGNORECASE).strip()
    value = re.sub(r'^(?:named|called)\s+', '', value, flags=re.IGNORECASE).strip()
    value = re.sub(r'^(?:file|document|doc)\s+', '', value, flags=re.IGNORECASE).strip()
    value = re.sub(r'^(?:please|pls|can you|could you|would you|help me)\s+', '', value, flags=re.IGNORECASE).strip()
    value = re.sub(r'^(?:now|switch\s+to|instead)\s+', '', value, flags=re.IGNORECASE).strip()
    value = re.sub(r'^(?:tell me about|tell me|about)\s+', '', value, flags=re.IGNORECASE).strip()
    value = re.sub(r'^(?:compare|comparison)\s+', '', value, flags=re.IGNORECASE).strip()
    value = re.sub(r'\s+(?:instead|please|pls)$', '', value, flags=re.IGNORECASE).strip()
    value = re.sub(r'^(?:帮我|给我|请|麻烦你|麻烦)\s*', '', value).strip()
    value = re.sub(r'^(?:找|打开|查找|搜索|看看|看下)\s*', '', value).strip()
    value = re.sub(r'^(?:一下|一下子)\s*', '', value).strip()
    value = re.sub(r'^(?:文件名)\s*', '', value).strip()
    value = re.sub(r'^(?:这张|这个|这份|那个|那张|那份)\s*', '', value).strip()
    value = re.sub(
        r'\s*(?:这个|这份|这张|那个|那份|那张)?(?:文件名?|文档|图片|照片|表格|音频|视频|file|document|image|photo|picture|spreadsheet|table|audio|video)s?\s*$',
        '',
        value,
        flags=re.IGNORECASE,
    ).strip()
    return _collapse_repeated_surface(value)


def has_plausible_filename_extension(candidate: str) -> bool:
    value = str(candidate or "").strip().strip(" \"'“”‘’.,;:!?}>")
    if "." not in value:
        return False
    ext = str(value.rsplit(".", 1)[-1] or "").strip().lower()
    if len(ext) < 2 or len(ext) > 8:
        return False
    if ext.isdigit():
        return False
    return any("a" <= ch <= "z" for ch in ext)


def score_filename_surface_match(
    query_surface: str,
    candidate_name: str,
    aliases: str = "",
) -> Tuple[bool, int]:
    """
    Score how strongly a user-provided filename surface matches a candidate filename.

    This is intentionally generic:
    - exact basename / stem equality stays strongest
    - basename-only queries can strongly match longer stems that contain the basename nucleus
    - aliases can contribute, but only after filename/stem checks
    """
    query_value = normalize_filename_candidate(query_surface)
    candidate_value = clean_filename_space(str(candidate_name or ""))
    aliases_value = clean_filename_space(str(aliases or ""))
    if not query_value or not candidate_value:
        return False, 0

    query_base = os.path.basename(query_value)
    candidate_base = os.path.basename(candidate_value)
    query_has_ext = has_plausible_filename_extension(query_base)
    query_stem = os.path.splitext(query_base)[0] if query_has_ext else query_base
    candidate_stem = os.path.splitext(candidate_base)[0]

    query_key = compact_filename_key(query_stem)
    candidate_key = compact_filename_key(candidate_stem)
    candidate_base_key = compact_filename_key(candidate_base)
    aliases_key = compact_filename_key(aliases_value)
    if not query_key:
        return False, 0

    query_is_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in query_stem)
    query_token_parts = [
        part for part in re.split(r"[\s_\-.]+", query_stem.lower()) if part
    ]
    candidate_token_blob = " ".join([candidate_stem.lower(), aliases_value.lower()])

    if query_has_ext and compact_filename_key(query_base) == candidate_base_key:
        return True, 100
    if query_key == candidate_key:
        return True, 98

    if query_key and query_key in candidate_key:
        starts = candidate_key.startswith(query_key)
        if query_is_cjk or len(query_key) >= 4:
            return False, 94 if starts else 90

    if aliases_key and query_key in aliases_key and (query_is_cjk or len(query_key) >= 4):
        return False, 88

    if query_token_parts:
        overlap = sum(1 for part in query_token_parts if part and part in candidate_token_blob)
        if overlap == len(query_token_parts) and overlap > 0:
            if len(query_token_parts) == 1:
                return False, 84
            return False, min(92, 78 + overlap * 4)
        if overlap > 0:
            return False, min(80, 66 + overlap * 4)

    return False, 0


def is_descriptive_filename_phrase(candidate: str) -> bool:
    value = str(candidate or "").strip()
    if not value:
        return False
    lowered = value.lower()
    if re.search(
        r"\b(?:about|related to|regarding|concerning|with|contain|contains|containing|including|featuring|mentioning|mentioned)\b",
        lowered,
    ):
        return True
    zh_desc_patterns = (
        "关于",
        "有关",
        "相关",
        "我们有",
        "有没有",
        "内容",
        "资料",
        "介绍",
        "总结",
        "分析",
        "对比",
        "讲了什么",
        "提到",
        "说到",
    )
    if any(pat in value for pat in zh_desc_patterns):
        return True
    if value.endswith("的"):
        return True
    if re.search(r"的(?:文档|文件|论文|资料|内容)$", value):
        return True
    return False


def is_reference_filename_placeholder(candidate: str) -> bool:
    value = clean_filename_space(str(candidate or "").strip())
    if not value:
        return False
    lowered = value.lower()
    if lowered in _REFERENCE_PLACEHOLDER_TERMS:
        return True
    collapsed = re.sub(r"[\s_\-]+", " ", lowered).strip()
    if collapsed in _REFERENCE_PLACEHOLDER_TERMS:
        return True
    return False


def reference_placeholder_number(candidate: str) -> str:
    value = clean_filename_space(str(candidate or "").strip())
    if not value:
        return ""
    lowered = value.lower()
    collapsed = re.sub(r"[\s_\-]+", " ", lowered).strip()
    if lowered in _PLURAL_REFERENCE_PLACEHOLDER_TERMS or collapsed in _PLURAL_REFERENCE_PLACEHOLDER_TERMS:
        return "plural"
    if lowered in _SINGULAR_REFERENCE_PLACEHOLDER_TERMS or collapsed in _SINGULAR_REFERENCE_PLACEHOLDER_TERMS:
        return "singular"
    if is_reference_filename_placeholder(value):
        return "unknown"
    return ""


def classify_reference_target(query: str) -> Dict[str, str]:
    text = clean_filename_space(str(query or "").strip())
    if not text:
        return {"kind": "none", "target": "", "number": ""}

    surfaces = extract_filename_query_surfaces(text, max_candidates=1)
    if surfaces:
        candidate = normalize_filename_candidate(surfaces[0])
        if candidate and looks_like_specific_filename_candidate(candidate):
            return {"kind": "explicit", "target": candidate, "number": ""}

    if not _DEICTIC_ACTION_RE.search(text):
        return {"kind": "none", "target": "", "number": ""}

    deictic_target = ""
    number = ""

    for token in sorted(_REFERENCE_PLACEHOLDER_TERMS, key=len, reverse=True):
        token_re = re.compile(r'(?<![\w\u4e00-\u9fff])' + re.escape(token) + r'(?![\w\u4e00-\u9fff])', re.IGNORECASE)
        match = token_re.search(text)
        if match:
            # "that" often introduces a relative clause in broad search queries
            # like "videos that contain text", not a deictic file reference.
            if token.lower() == "that" and _RELATIVE_THAT_FOLLOW_RE.search(text[match.end():]):
                continue
            # "find every picture with a dog in it" uses "it" as part of a
            # content description, not as a reference to the active file.
            if token.lower() == "it" and _THEMATIC_IT_IN_CONTENT_RE.search(text):
                continue
            deictic_target = token
            number = reference_placeholder_number(token)
            break

    if not deictic_target:
        noun_match = _DEICTIC_NOUN_RE.search(text)
        if noun_match:
            deictic_target = clean_filename_space(noun_match.group(0))
            number = reference_placeholder_number(deictic_target)
            if not number:
                lowered = deictic_target.lower()
                if any(term in lowered for term in ("these", "those")) or any(term in deictic_target for term in ("这些", "那些")):
                    number = "plural"
                else:
                    number = "singular"

    if deictic_target:
        return {"kind": "deictic", "target": deictic_target, "number": number or "unknown"}

    return {"kind": "none", "target": "", "number": ""}


def classify_explicit_filename_match_mode(
    query: str,
    explicit_ref: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    text = clean_filename_space(str(query or "").strip())
    raw_name = ""
    search_term = ""
    if isinstance(explicit_ref, dict):
        raw_name = clean_filename_space(str(explicit_ref.get("raw_name") or "").strip())
        search_term = clean_filename_space(str(explicit_ref.get("search_term") or "").strip())

    raw_candidate = os.path.basename(raw_name) if raw_name else ""
    search_candidate = os.path.basename(search_term) if search_term else ""
    candidate = raw_candidate if has_plausible_filename_extension(raw_candidate) else (search_candidate or raw_candidate)
    if not candidate:
        return {"mode": "broad", "target": "", "target_key": "", "stem_key": ""}

    candidate_has_extension = has_plausible_filename_extension(candidate)
    candidate_ext = str(candidate.rsplit(".", 1)[-1] or "").strip().lower() if candidate_has_extension else ""
    domain_suffix_is_stem = bool(
        candidate_has_extension
        and _EXPLICIT_FILE_SCOPE_RE.search(text)
        and candidate_ext in _DOMAIN_LIKE_STEM_SUFFIXES
        and (
            re.search(r"(?i)(?:^|[\s_\-.])www\.", candidate)
            or any(ch.isdigit() for ch in candidate)
        )
    )
    candidate_stem = (
        candidate
        if domain_suffix_is_stem
        else os.path.splitext(candidate)[0] if candidate_has_extension else candidate
    )
    if (
        candidate_has_extension
        and not domain_suffix_is_stem
        and looks_like_thematic_lookup_candidate(text, candidate)
    ):
        return {
            "mode": "broad",
            "target": candidate_stem,
            "target_key": "",
            "stem_key": compact_filename_key(candidate_stem),
        }

    if candidate_has_extension and not domain_suffix_is_stem:
        return {
            "mode": "exact_filename",
            "target": candidate,
            "target_key": compact_filename_key(candidate),
            "stem_key": compact_filename_key(candidate_stem),
        }

    if text and (
        _EXPLICIT_FILENAME_FOCUS_RE.search(text)
        or (
            _EXPLICIT_FILE_SCOPE_RE.search(text)
            and looks_like_specific_filename_candidate(candidate_stem)
        )
    ):
        return {
            "mode": "exact_stem",
            "target": candidate_stem,
            "target_key": "",
            "stem_key": compact_filename_key(candidate_stem),
        }

    return {
        "mode": "broad",
        "target": candidate_stem,
        "target_key": "",
        "stem_key": compact_filename_key(candidate_stem),
    }


def looks_like_thematic_lookup_candidate(query: str, candidate: str) -> bool:
    raw_query = clean_filename_space(str(query or "").strip())
    raw_candidate = normalize_filename_candidate(str(candidate or "").strip())
    if not raw_query or not raw_candidate:
        return False
    if not _THEMATIC_LOOKUP_QUERY_RE.search(raw_query):
        return False
    if _EXPLICIT_FILENAME_FOCUS_RE.search(raw_query):
        return False
    has_thematic_preposition = bool(_THEMATIC_DOT_PREPOSITION_RE.search(raw_query))
    has_doc_context = bool(_THEMATIC_DOT_DOC_CONTEXT_RE.search(raw_query))
    if (
        "." in raw_candidate
        and not has_plausible_filename_extension(raw_candidate)
        and not _EXPLICIT_OPEN_FILE_ACTION_RE.search(raw_query)
        and (has_thematic_preposition or has_doc_context)
        and re.search(r"\s|关于|有关|相关", raw_candidate)
    ):
        return True
    if has_plausible_filename_extension(raw_candidate):
        if _EXPLICIT_OPEN_FILE_ACTION_RE.search(raw_query):
            return False
        candidate_base = os.path.basename(raw_candidate)
        candidate_has_extra_words = clean_filename_space(raw_candidate) != clean_filename_space(candidate_base)
        ext = str(candidate_base.rsplit(".", 1)[-1] or "").lower()
        if candidate_has_extra_words and (has_thematic_preposition or has_doc_context):
            return True
        if ext in _TOPIC_DOT_EXTENSIONS and (
            has_thematic_preposition
            or (has_doc_context and bool(re.search(r"\s", raw_candidate)))
        ):
            return True
        return False
    if any(ch.isdigit() for ch in raw_candidate):
        return False
    if any(ch in raw_candidate for ch in ("_", "-", ".", "/", "\\", "(", ")", "（", "）")):
        return False

    lowered = raw_candidate.lower()
    ascii_tokens = [tok for tok in re.findall(r"[a-z0-9]+", lowered) if tok]
    has_en_thematic = any(tok in _THEMATIC_LOOKUP_TERMS_EN for tok in ascii_tokens)
    has_zh_thematic = any(term in raw_candidate for term in _THEMATIC_LOOKUP_TERMS_ZH)
    if not has_en_thematic and not has_zh_thematic:
        return False

    if ascii_tokens:
        non_thematic_ascii = [tok for tok in ascii_tokens if tok not in _THEMATIC_LOOKUP_TERMS_EN]
        if has_en_thematic and non_thematic_ascii:
            return True

    if has_zh_thematic and len(raw_candidate) >= 3:
        return True
    return False


def looks_like_specific_filename_candidate(candidate: str) -> bool:
    value = str(candidate or "").strip()
    if not value or is_reference_filename_placeholder(value):
        return False
    has_filename_structure = any(ch in value for ch in ("_", "-", ".", "(", ")", "（", "）"))
    if is_descriptive_filename_phrase(value) and not has_filename_structure:
        return False
    lowered = value.lower()
    if lowered in _WEAK_FILELIKE_TERMS:
        return False
    token_parts = [p for p in re.split(r"[\s_\-]+", lowered) if p]
    if not token_parts or all(part in _WEAK_FILELIKE_TERMS for part in token_parts):
        return False
    if any(ch.isdigit() for ch in value):
        return True
    if has_filename_structure:
        return True
    if any("\u4e00" <= ch <= "\u9fff" for ch in value):
        return len(value) >= 2
    return len("".join(token_parts)) >= 4


def _ordered_unique(values: Iterable[str], *, max_items: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        value = clean_filename_space(str(raw or "")).lower()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(clean_filename_space(str(raw or "")))
        if len(out) >= max_items:
            break
    return out


_COMPOUND_FILELIKE_SEPARATOR_RE = re.compile(
    r"\s+(?:and|or|plus|with|vs\.?|versus)\s+"
    r"|[,;，；、&]+"
    r"|(?:以及|或者|和|与|及)"
    r"|(?:\s*\+\s*)",
    re.IGNORECASE,
)
_FILELIKE_EXTENSION_SURFACE_RE = re.compile(
    r"\.[A-Za-z0-9]{2,8}(?=$|[\s,;，；、&)]|以及|或者|和|与|及)",
    re.IGNORECASE,
)


def _expand_compound_filelike_candidate(candidate: str) -> List[str]:
    """
    Split compound filename mentions such as "a.csv and b.csv" into distinct
    surfaces while leaving ordinary single filenames with spaces untouched.
    """
    value = normalize_filename_candidate(candidate)
    if not value:
        return []
    if len(_FILELIKE_EXTENSION_SURFACE_RE.findall(value)) < 2:
        return [value]

    expanded: List[str] = []
    for part in _COMPOUND_FILELIKE_SEPARATOR_RE.split(value):
        normalized = normalize_filename_candidate(part)
        if normalized and looks_like_specific_filename_candidate(normalized):
            expanded.append(normalized)
    return expanded or [value]


def extract_filename_query_surfaces(text: str, *, max_candidates: int = 16) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    candidates: List[str] = []
    for pattern in _MARKED_FILELIKE_PATTERNS:
        match = pattern.search(raw)
        if not match:
            continue
        candidate = normalize_filename_candidate(match.group(1))
        candidate = re.sub(r"(这张|这个|这份|那个|那张|那份)$", "", candidate).strip()
        if looks_like_specific_filename_candidate(candidate) and not looks_like_thematic_lookup_candidate(raw, candidate):
            candidates.extend(_expand_compound_filelike_candidate(candidate))

    for match in _FILELIKE_RE.finditer(raw):
        candidate = normalize_filename_candidate(match.group(1))
        if candidate:
            candidates.extend(_expand_compound_filelike_candidate(candidate))

    return _ordered_unique(candidates, max_items=max_candidates)
