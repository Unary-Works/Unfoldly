from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple

from core.retrieval.filename_canonicalizer import (
    compact_filename_key,
    extract_filename_query_surfaces,
)

_SEG_RE = re.compile(r"[A-Za-z]+|\d+|[\u4e00-\u9fff]+")
_SPACE_RE = re.compile(r"\s+")
_SEP_RE = re.compile(r"[\\/_\-.]+")
_PAPER_ID_RE = re.compile(r"\b(\d{4})\.(\d{4,6})(v\d+)?\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r'["“”\'‘’]([^"“”\'‘’]{1,160})["“”\'‘’]')

_LOOKUP_NOISE = {
    "a", "an", "the", "and", "or", "of", "for", "with", "from", "into", "about",
    "this", "that", "these", "those", "my", "our", "your", "their", "all", "any", "some",
    "find", "show", "search", "look", "list", "open", "get", "give", "tell",
    "file", "files", "document", "documents", "doc", "docs", "data", "report", "reports",
    "plan", "plans",
    "image", "images", "photo", "photos", "picture", "pictures",
    "audio", "audios", "video", "videos", "resume", "resumes", "invoice", "invoices",
    "请", "帮我", "帮忙", "看看", "查看", "找", "搜索", "列出", "显示",
    "文件", "文档", "数据", "内容", "信息", "我的", "关于",
}

_CJK_TOPIC_PREFIXES = (
    "帮我找一下", "帮我找下", "帮我查一下", "帮我查下", "帮我搜一下", "帮我搜下",
    "帮我找", "帮我查", "帮我搜", "找一下", "找下", "查一下", "查下", "搜一下", "搜下",
    "看一下", "看下", "查看一下", "查看", "搜索一下", "搜索", "查找一下", "查找",
    "帮我", "帮忙", "请", "我想找", "我想看", "我想搜", "我想查",
)
_CJK_TOPIC_SUFFIXES = (
    "的视频", "视频", "的音频", "音频", "的录音", "录音", "的图片", "图片", "的照片", "照片",
    "的文件", "文件", "的文档", "文档", "的资料", "资料", "的内容", "内容", "的相关内容", "相关内容",
    "有哪些", "有什么", "在哪里", "在哪", "是什么", "是哪些", "相关", "那个", "这个",
)
_CJK_ANCHOR_SPLIT_RE = re.compile(
    r"(?:关于|有关|相关|以及|或者|或是|还是|和|与|及|或|的|里|里面|中|"
    r"分享|资料|文件|文档|内容|信息|邮箱|邮件|电话|手机|联系方式|联系)"
)
_CJK_LOOKUP_ANCHOR_NOISE = {
    "一下", "找一下", "搜一下", "查一下", "查看", "搜索", "查找", "帮我", "请问",
    "文件", "文档", "资料", "内容", "信息", "相关", "有关", "分享", "邮箱", "邮件",
    "电话", "手机", "联系方式", "当前", "全局", "选中", "上一轮", "结果",
}

_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape"}
_VIDEO_EXTS = {".mp4", ".m4v", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".ts"}

_AUDIO_QUERY_RE = re.compile(
    r"\b(audio|audios|recording|recordings|sound|sounds|music|song|songs|track|tracks|podcast|podcasts)\b"
    r"|音频|录音|声音|音乐|歌曲"
)
_VIDEO_QUERY_RE = re.compile(
    r"\b(video|videos|movie|movies|film|films|clip|clips|footage|reel|reels|screencast|screen recording)\b"
    r"|视频|录像|录屏|影片"
)
_SPEECH_QUERY_RE = re.compile(
    r"\b(speech|spoken|talking|dialogue|conversation|voice|voices|transcript|transcripts)\b"
    r"|说话|语音|对白|对话|讲话"
)
_SCREEN_TEXT_QUERY_RE = re.compile(
    r"\b(text on screen|on-screen text|onscreen text|screen text|ocr|subtitle|subtitles|caption|captions)\b"
    r"|屏幕文字|画面文字|字幕|文字"
)
_DOCUMENT_TARGET_QUERY_RE = re.compile(
    r"\b(?:papers?|articles?|documents?|docs?|pdfs?|reports?|publications?|theses|thesis|whitepapers?)\b"
    r"|论文|文章|文档|报告|资料|PDF|pdf|白皮书",
    re.IGNORECASE,
)
_MEDIA_TOPIC_QUERY_RE = re.compile(
    r"\b(?:audio|video|speech|music|sound|voice|recording|recordings)\b"
    r"|音频|视频|语音|声音|音乐|录音",
    re.IGNORECASE,
)
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{1,24}")

try:
    from pypinyin import lazy_pinyin as _lazy_pinyin  # type: ignore
except Exception:
    _lazy_pinyin = None  # type: ignore


def _clean_space(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text or "").strip())


def _compact_lookup(text: str) -> str:
    return compact_filename_key(text)


def _is_identifierish_term(term: str) -> bool:
    t = str(term or "").strip()
    if not t:
        return False
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in t)
    has_alpha = any(ch.isalpha() for ch in t)
    has_digit = any(ch.isdigit() for ch in t)
    if has_digit and (has_alpha or has_cjk):
        return True
    if has_cjk and has_alpha:
        return True
    if any(ch in t for ch in ("_", "-", "/", "\\")):
        return True
    if "." in t:
        terminal_sentence_dot = (
            t.endswith(".")
            and t.count(".") == 1
            and " " in t
            and not any(ch.isdigit() for ch in t)
        )
        if not terminal_sentence_dot:
            return True
    if re.search(r"[A-Z]{2,}[A-Za-z0-9]*", t):
        return True
    return False


def _ordered_unique(terms: Iterable[str], max_terms: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in terms:
        t = _clean_space(str(raw or "").lower())
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _surface_has_lookup_signal(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    return bool(
        extract_filename_query_surfaces(raw, max_candidates=1)
        or _PAPER_ID_RE.search(raw)
        or _is_identifierish_term(raw)
        or any(ch.isdigit() for ch in raw)
    )


def extract_filelike_candidates(text: str, *, max_candidates: int = 16) -> List[str]:
    return extract_filename_query_surfaces(text, max_candidates=max_candidates)


@lru_cache(maxsize=4096)
def _pinyin_aliases_for_cjk_run(text: str) -> Tuple[str, ...]:
    run = str(text or "").strip()
    if not run or _lazy_pinyin is None:
        return ()
    try:
        parts = [p.strip().lower() for p in _lazy_pinyin(run, errors="ignore") if str(p or "").strip()]
    except Exception:
        return ()
    if not parts:
        return ()
    joined = "".join(parts)
    spaced = " ".join(parts)
    initials = "".join(p[0] for p in parts if p)
    aliases = [spaced, joined, initials]
    return tuple(_ordered_unique(aliases, max_terms=8))


def build_cjk_latin_aliases(*texts: str, max_terms: int = 64) -> str:
    """Return pinyin aliases for CJK names/paths when pypinyin is available."""
    if _lazy_pinyin is None:
        return ""
    aliases: List[str] = []
    for text in texts:
        raw = str(text or "")
        if not raw:
            continue
        for match in _CJK_RUN_RE.finditer(raw):
            run = match.group(0)
            aliases.extend(_pinyin_aliases_for_cjk_run(run))
    return " ".join(_ordered_unique(aliases, max_terms=max_terms))


def extract_lookup_terms(text: str, *, max_terms: int = 64) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    surfaces: List[str] = [raw]
    for match in _QUOTED_RE.finditer(raw):
        quoted = _clean_space(match.group(1))
        if quoted and _surface_has_lookup_signal(quoted):
            surfaces.append(quoted)
    for filelike in extract_filelike_candidates(raw, max_candidates=16):
        if filelike:
            surfaces.append(filelike)
    norm_path = raw.replace("\\", "/")
    if "/" in norm_path:
        parts = [p for p in norm_path.split("/") if p and p != "."]
        if parts:
            surfaces.extend(parts[-4:])
            surfaces.append(parts[-1])
            if len(parts) >= 2:
                surfaces.append(parts[-2])

    terms: List[str] = []

    def _push(term: str) -> None:
        t = _clean_space(str(term or "").lower())
        if not t:
            return
        if t in _LOOKUP_NOISE and not _is_identifierish_term(t):
            return
        if len(t) == 1 and not any("\u4e00" <= ch <= "\u9fff" for ch in t):
            return
        if t.isdigit() and len(t) < 2:
            return
        terms.append(t)

    for surface in surfaces:
        s = str(surface or "").strip()
        if not s:
            continue
        lower = s.lower()
        _push(lower)

        basename = os.path.basename(s)
        stem = os.path.splitext(basename)[0].strip().lower()
        if stem and stem != lower:
            _push(stem)

        sep_norm = _clean_space(_SEP_RE.sub(" ", lower))
        if sep_norm and sep_norm != lower:
            _push(sep_norm)

        compact = _compact_lookup(lower)
        if compact and compact != lower and len(compact) >= 4:
            _push(compact)

        for match in _PAPER_ID_RE.finditer(lower):
            year, number, version = match.groups()
            full = f"{year}.{number}{version or ''}"
            _push(full)
            _push(f"{year}.{number}")
            _push(f"{year} {number}{(' ' + version) if version else ''}".strip())
            _push(year)
            _push(number)
            if version:
                _push(version)

        segs = [seg.lower() for seg in _SEG_RE.findall(sep_norm or lower) if seg]
        filtered_segs: List[str] = []
        for seg in segs:
            if seg in _LOOKUP_NOISE and not _is_identifierish_term(seg):
                continue
            if seg.isdigit() and len(seg) < 2:
                continue
            if len(seg) < 2 and not any("\u4e00" <= ch <= "\u9fff" for ch in seg):
                continue
            filtered_segs.append(seg)
            _push(seg)

        if filtered_segs:
            _push(" ".join(filtered_segs))
            ascii_only = [seg for seg in filtered_segs if seg.isascii() and seg.isalpha()]
            if len(ascii_only) >= 2:
                _push(" ".join(ascii_only))
            for width in (2, 3):
                for idx in range(0, max(0, len(filtered_segs) - width + 1)):
                    _push(" ".join(filtered_segs[idx : idx + width]))

        for cjk_topic in _extract_cjk_topic_terms(s):
            _push(cjk_topic)
            for anchor in _split_cjk_lookup_anchors(cjk_topic):
                _push(anchor)

    return _ordered_unique(terms, max_terms=max_terms)


def _split_cjk_lookup_anchors(text: str) -> List[str]:
    anchors: List[str] = []
    seen = set()
    for raw in _CJK_ANCHOR_SPLIT_RE.split(str(text or "")):
        term = raw.strip()
        if not (2 <= len(term) <= 12):
            continue
        if term in _CJK_LOOKUP_ANCHOR_NOISE or term in _LOOKUP_NOISE:
            continue
        if not any("\u4e00" <= ch <= "\u9fff" for ch in term):
            continue
        if term not in seen:
            seen.add(term)
            anchors.append(term)
    return anchors


def extract_strong_lookup_anchors(text: str, *, max_terms: int = 16) -> List[str]:
    """
    Return compact anchors suitable for metadata recall.

    Unlike extract_lookup_terms(), this keeps only terms that are specific enough
    to scan across summaries without turning broad semantic searches into noise.
    """
    raw = str(text or "").strip()
    if not raw:
        return []

    anchors: List[str] = []

    def _push(term: str) -> None:
        t = _clean_space(str(term or "").strip().lower())
        if not t or t in anchors:
            return
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in t)
        if has_cjk:
            if not (2 <= len(t) <= 12):
                return
            if t in _CJK_LOOKUP_ANCHOR_NOISE or t in _LOOKUP_NOISE:
                return
            if any(marker in t for marker in ("当前全局选", "上一轮结果", "上轮结果")):
                return
            if re.match(r"^(?:find|search|show|list|open|get|look)(?:\s|[\u4e00-\u9fff])", t):
                return
            if re.match(r"^(?:找|搜|查|搜索|查找|查看)\s*[\u4e00-\u9fff]", t):
                return
            anchors.append(t)
            return
        if _is_identifierish_term(t) or any(ch.isdigit() for ch in t):
            anchors.append(t)

    for term in extract_lookup_terms(raw, max_terms=max_terms * 3):
        _push(term)
    for topic in _extract_cjk_topic_terms(raw):
        _push(topic)
        for anchor in _split_cjk_lookup_anchors(topic):
            _push(anchor)

    return anchors[:max_terms]


def build_lookup_blob(*texts: str, max_terms: int = 96) -> str:
    merged: List[str] = []
    for text in texts:
        merged.extend(extract_lookup_terms(text, max_terms=max_terms))
    return " ".join(_ordered_unique(merged, max_terms=max_terms))


def build_candidate_lookup_blob(candidate: Dict[str, Any]) -> str:
    fp = str(candidate.get("file_path") or "")
    fn = str(candidate.get("file_name") or "")
    ds = str(candidate.get("doc_summary") or "")
    aliases = str(candidate.get("lookup_aliases") or "")
    schema = str(candidate.get("table_schema_hint") or "")
    text = str(candidate.get("text") or "")[:800]
    semantic_aliases = _build_candidate_semantic_aliases(candidate)
    return f"{fn} {fp} {ds} {aliases} {schema} {text} {semantic_aliases}".strip()


def _build_candidate_semantic_aliases(candidate: Dict[str, Any]) -> str:
    meta = candidate.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}

    file_path = str(candidate.get("file_path") or meta.get("file_path") or "")
    file_name = str(candidate.get("file_name") or meta.get("file_name") or "")
    doc_summary = str(candidate.get("doc_summary") or meta.get("doc_summary") or "")
    text = str(candidate.get("text") or "")[:800]
    signal_blob = " ".join([file_name, file_path, doc_summary, text]).lower()

    doc_category = str(
        candidate.get("doc_category")
        or candidate.get("doc_category_family")
        or meta.get("doc_category")
        or meta.get("doc_category_family")
        or ""
    ).strip().lower()
    media_type = str(candidate.get("media_type") or meta.get("media_type") or "").strip().lower()
    chunk_type = str(candidate.get("chunk_type") or meta.get("chunk_type") or "").strip().lower()

    aliases: List[str] = []

    if doc_category == "image":
        aliases.append("image photo picture screenshot visual")
    from core.retrieval.category_engine import is_media_category_value

    if is_media_category_value(doc_category) or media_type:
        if media_type == "video":
            aliases.append("video clip movie footage")
        elif media_type == "audio":
            aliases.append("audio recording sound")
        else:
            aliases.append("audio video media recording")

    if any(token in signal_blob for token in ("screenrecording", "screen recording", "screenrecord", "录屏", "录像")):
        aliases.append("screen recording screencast text on screen ocr onscreen text")
    if any(token in signal_blob for token in ("musicgen", "sleep_music", " music ", " song ", "歌曲", "音乐")):
        aliases.append("music song instrumental melody beat soundtrack")
    if any(token in signal_blob for token in ("sleep", "dreamscape", "bedtime", "slumber", "snore", "snoring", "relax", "meditation", "助眠", "睡眠")):
        aliases.append("sleep bedtime relaxation meditation rest calm soothing")
    if any(token in signal_blob for token in ("speech", "spoken", "voice", "asr", "transcript", "vad", "sensevoice", "whisper")):
        aliases.append("speech spoken voice talking transcript dialogue")

    has_asr = bool(
        candidate.get("media_has_asr_transcript")
        or meta.get("media_has_asr_transcript")
        or meta.get("has_asr_transcript")
        or chunk_type in {"asr_transcript", "asr_segment"}
    )
    has_keyframe_ocr = bool(
        candidate.get("has_keyframe_ocr")
        or meta.get("has_keyframe_ocr")
        or meta.get("media_has_keyframe_ocr")
        or meta.get("keyframe_ocr_text")
    )

    if has_asr:
        aliases.append("speech talking spoken conversation dialogue")
    if has_keyframe_ocr:
        aliases.append("text on screen screen text onscreen text ocr")

    return " ".join(dict.fromkeys(part for part in aliases if part))


def _infer_media_query_constraints(query_text: str) -> Dict[str, bool | str]:
    q = str(query_text or "").strip().lower()
    if not q:
        return {"media_type": "", "needs_asr": False, "needs_ocr": False}
    if _DOCUMENT_TARGET_QUERY_RE.search(q) and _MEDIA_TOPIC_QUERY_RE.search(q):
        # "papers about audio" / "reports on video models" are document searches.
        # Media terms are topical anchors there, not a request to keep audio/video files only.
        return {"media_type": "", "needs_asr": False, "needs_ocr": False}

    asks_audio = bool(_AUDIO_QUERY_RE.search(q))
    asks_video = bool(_VIDEO_QUERY_RE.search(q))
    media_type = ""
    if asks_video and not asks_audio:
        media_type = "video"
    elif asks_audio and not asks_video:
        media_type = "audio"

    return {
        "media_type": media_type,
        "needs_asr": bool(_SPEECH_QUERY_RE.search(q)),
        "needs_ocr": bool(_SCREEN_TEXT_QUERY_RE.search(q)),
    }


def _candidate_media_type(candidate: Dict[str, Any]) -> str:
    meta = candidate.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}

    media_type = str(candidate.get("media_type") or meta.get("media_type") or "").strip().lower()
    if media_type in {"audio", "video"}:
        return media_type

    file_path = str(candidate.get("file_path") or meta.get("file_path") or "")
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _VIDEO_EXTS:
        return "video"
    return ""


def _candidate_has_asr(candidate: Dict[str, Any]) -> bool:
    meta = candidate.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    chunk_type = str(candidate.get("chunk_type") or meta.get("chunk_type") or "").strip().lower()
    return bool(
        candidate.get("media_has_asr_transcript")
        or meta.get("media_has_asr_transcript")
        or meta.get("has_asr_transcript")
        or chunk_type in {"asr_transcript", "asr_segment"}
    )


def _candidate_has_keyframe_ocr(candidate: Dict[str, Any]) -> bool:
    meta = candidate.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    return bool(
        candidate.get("has_keyframe_ocr")
        or meta.get("has_keyframe_ocr")
        or meta.get("media_has_keyframe_ocr")
        or meta.get("keyframe_ocr_text")
    )


def apply_media_query_constraints(
    candidates: List[Dict[str, Any]],
    query_text: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Conservatively narrow media candidates using only clear type/evidence intent.

    This helper never returns an empty slice when the original list was non-empty.
    It only narrows when the query clearly prefers audio or video, or when
    ASR/OCR-backed candidates already exist for a speech/text-on-screen request.
    """
    ranked = list(candidates or [])
    if not ranked:
        return ranked, False

    constraints = _infer_media_query_constraints(query_text)
    changed = False

    media_type = str(constraints.get("media_type") or "").strip().lower()
    if media_type:
        typed = [src for src in ranked if _candidate_media_type(src) == media_type]
        if typed and len(typed) < len(ranked):
            ranked = typed
            changed = True

    if bool(constraints.get("needs_asr")):
        with_asr = [src for src in ranked if _candidate_has_asr(src)]
        if with_asr and len(with_asr) < len(ranked):
            ranked = with_asr
            changed = True

    if bool(constraints.get("needs_ocr")):
        with_ocr = [src for src in ranked if _candidate_has_keyframe_ocr(src)]
        if with_ocr and len(with_ocr) < len(ranked):
            ranked = with_ocr
            changed = True

    return ranked, changed


def _extract_cjk_topic_terms(text: str) -> List[str]:
    terms: List[str] = []
    seen = set()
    for raw in re.findall(r"[\u4e00-\u9fff]{2,}", str(text or "")):
        trimmed = raw
        changed = True
        while changed and len(trimmed) >= 2:
            changed = False
            for prefix in _CJK_TOPIC_PREFIXES:
                if trimmed.startswith(prefix) and len(trimmed) - len(prefix) >= 2:
                    trimmed = trimmed[len(prefix):]
                    changed = True
                    break
            for suffix in _CJK_TOPIC_SUFFIXES:
                if trimmed.endswith(suffix) and len(trimmed) - len(suffix) >= 2:
                    trimmed = trimmed[: -len(suffix)]
                    changed = True
                    break
            if trimmed.startswith("的") and len(trimmed) > 2:
                trimmed = trimmed[1:]
                changed = True
            if trimmed.endswith("的") and len(trimmed) > 2:
                trimmed = trimmed[:-1]
                changed = True
        if len(trimmed) >= 2 and trimmed not in seen:
            seen.add(trimmed)
            terms.append(trimmed)
    return terms


def lookup_match_quality(query_text: str, candidate_text: str) -> Tuple[bool, int]:
    q_terms = extract_lookup_terms(query_text, max_terms=32)
    q_norm = _clean_space(str(query_text or "").strip().lower())
    q_stem = os.path.splitext(os.path.basename(q_norm))[0]
    q_compact = _compact_lookup(q_norm)
    c_terms = set(extract_lookup_terms(candidate_text, max_terms=160))
    c_norm = _clean_space(os.path.basename(str(candidate_text or "").strip()).lower())
    c_stem = _clean_space(os.path.splitext(c_norm)[0])
    c_compact = _compact_lookup(candidate_text)
    c_stem_compact = _compact_lookup(c_stem)
    q_focus_surfaces = extract_filelike_candidates(query_text, max_candidates=8)
    q_focus_terms = extract_lookup_terms(" ".join(q_focus_surfaces), max_terms=32)
    q_focus_compacts = {
        compact for compact in (_compact_lookup(surface) for surface in q_focus_surfaces)
        if compact and len(compact) >= 2
    }

    exact = False
    if q_norm and q_norm in {c_norm, c_stem}:
        exact = True
    elif q_stem and q_stem in {c_norm, c_stem}:
        exact = True
    elif q_compact and len(q_compact) >= 2 and q_compact in {c_compact, c_stem_compact}:
        exact = True
    elif q_focus_compacts and any(compact in {c_compact, c_stem_compact} for compact in q_focus_compacts):
        exact = True
    elif (not q_focus_surfaces) and any(term in {c_norm, c_stem} for term in q_focus_terms):
        exact = True
    elif (not q_focus_surfaces) and any(
        (compact := _compact_lookup(term)) and len(compact) >= 2 and compact in {c_compact, c_stem_compact}
        for term in q_focus_terms
    ):
        exact = True
    if not q_terms and not exact:
        return False, 0
    if not q_terms and exact:
        q_terms = [term for term in [q_norm, q_stem] if term]

    score = 0
    for term in q_terms:
        if term in c_terms:
            if exact and term in {q_norm, q_stem}:
                score += 4
            elif exact and term in q_focus_terms:
                score += 4
            elif _is_identifierish_term(term) or any(ch.isdigit() for ch in term):
                score += 3
            elif len(term) >= 6 or any("\u4e00" <= ch <= "\u9fff" for ch in term):
                score += 2
            else:
                score += 1
            continue
        compact_term = _compact_lookup(term)
        if compact_term and len(compact_term) >= 6 and compact_term in c_compact:
            score += 2

    q_data_terms = {"data", "dataset", "datasets", "table", "spreadsheet", "csv", "excel"}
    c_data_terms = {"data", "dataset", "datasets", "table", "spreadsheet", "csv", "tsv", "xlsx", "xls"}
    q_data_signal = q_data_terms.intersection(q_terms) or bool(
        re.search(r"\b(?:data|dataset|datasets|table|spreadsheet|csv|excel)\b", q_norm)
    )
    if q_data_signal and c_data_terms.intersection(c_terms):
        score += 1

    if exact and score < 4:
        score = 4
    return exact, score


def compute_lookup_overlap_score(query_text: str, candidate_text: str) -> int:
    _, score = lookup_match_quality(query_text, candidate_text)
    return score


def annotate_candidates_with_topic_overlap(
    candidates: List[Dict[str, Any]],
    query_text: str,
) -> List[Dict[str, Any]]:
    cjk_focus_terms = _extract_cjk_topic_terms(query_text)
    specific_terms = _specific_topic_terms(query_text)
    annotated: List[Dict[str, Any]] = []
    for src in list(candidates or []):
        item = dict(src)
        blob = build_candidate_lookup_blob(item)
        exact, overlap = lookup_match_quality(query_text, blob)
        focus_hits = 0
        if cjk_focus_terms:
            for term in cjk_focus_terms:
                if term in blob:
                    focus_hits += 1
        specific_hits = _count_specific_topic_hits(specific_terms, blob)
        item["_topic_lookup_exact"] = bool(exact)
        item["_topic_lookup_focus_hits"] = int(focus_hits)
        item["_topic_specific_hits"] = int(specific_hits)
        item["_topic_lookup_overlap"] = int(max(overlap, focus_hits * 2))
        annotated.append(item)
    return annotated


def sort_candidates_by_topic_overlap(
    candidates: List[Dict[str, Any]],
    query_text: str,
) -> List[Dict[str, Any]]:
    annotated = annotate_candidates_with_topic_overlap(candidates, query_text)
    media_constraints = _infer_media_query_constraints(query_text)
    prefer_media_rank = bool(media_constraints.get("media_type")) or bool(media_constraints.get("needs_asr")) or bool(media_constraints.get("needs_ocr"))
    query_is_lookup_heavy = is_lookup_heavy_query(query_text)
    indexed = list(enumerate(annotated))
    if prefer_media_rank:
        if query_is_lookup_heavy:
            indexed.sort(
                key=lambda item: (
                    1 if item[1].get("_topic_lookup_exact") else 0,
                    int(item[1].get("_topic_specific_hits", 0) or 0),
                    1 if item[1].get("_lexical_filename_exact") else 0,
                    float(item[1].get("_bm25_score", 0.0) or 0.0),
                    int(item[1].get("_direct_score", 0) or 0),
                    int(item[1].get("_topic_lookup_focus_hits", 0) or 0),
                    int(item[1].get("_topic_lookup_overlap", 0) or 0),
                    float(item[1].get("rerank_score", 0.0) or 0.0),
                    -item[0],
                ),
                reverse=True,
            )
        else:
            indexed.sort(
                key=lambda item: (
                    1 if item[1].get("_topic_lookup_exact") else 0,
                    int(item[1].get("_topic_specific_hits", 0) or 0),
                    int(item[1].get("_topic_lookup_focus_hits", 0) or 0),
                    int(item[1].get("_topic_lookup_overlap", 0) or 0),
                    float(item[1].get("rerank_score", 0.0) or 0.0),
                    1 if item[1].get("_lexical_filename_exact") else 0,
                    float(item[1].get("_bm25_score", 0.0) or 0.0),
                    int(item[1].get("_direct_score", 0) or 0),
                    -item[0],
                ),
                reverse=True,
            )
    else:
        indexed.sort(
            key=lambda item: (
                1 if item[1].get("_topic_lookup_exact") else 0,
                int(item[1].get("_topic_specific_hits", 0) or 0),
                int(item[1].get("_topic_lookup_focus_hits", 0) or 0),
                int(item[1].get("_topic_lookup_overlap", 0) or 0),
                1 if item[1].get("_lexical_filename_exact") else 0,
                int(item[1].get("_direct_score", 0) or 0),
                float(item[1].get("_bm25_score", 0.0) or 0.0),
                float(item[1].get("rerank_score", 0.0) or 0.0),
                -item[0],
            ),
            reverse=True,
        )
    return [src for _, src in indexed]


def narrow_candidates_by_topic_overlap(
    candidates: List[Dict[str, Any]],
    query_text: str,
    *,
    require_topic: bool = False,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Narrow to candidates with explicit topic overlap when such overlap exists.

    Intended for category-scoped searches like "files about Beijing":
    if some files clearly mention the topical anchor in filename/path/summary/text,
    keep those and drop zero-overlap category noise. If no positive overlap exists,
    preserve the original candidate set unless the caller requires topic evidence.
    """
    ranked = sort_candidates_by_topic_overlap(candidates, query_text)
    specific_terms = _specific_topic_terms(query_text)
    positives = [
        src for src in ranked
        if (
            (bool(src.get("_topic_lookup_exact")) and not specific_terms)
            or int(src.get("_topic_specific_hits", 0) or 0) > 0
            or (not specific_terms and int(src.get("_topic_lookup_overlap", 0) or 0) > 0)
            or bool(src.get("_matched_terms"))
            or bool(src.get("_expanded_terms"))
        )
    ]
    if positives and len(positives) < len(ranked):
        return positives, True
    if require_topic and _has_specific_topic(query_text) and not positives:
        return [], True
    return ranked, False


def _has_specific_topic(query_text: str) -> bool:
    return bool(_specific_topic_terms(query_text))


def _specific_topic_terms(query_text: str) -> List[str]:
    generic = {
        "audio", "audios", "video", "videos", "recording", "recordings",
        "clip", "clips", "movie", "movies", "film", "films", "footage",
        "file", "files", "media", "source", "sources", "find", "search",
        "show", "list", "locate", "which", "where", "what", "that", "this",
        "the", "a", "an", "my", "your", "our", "about", "with", "in", "on",
        "i", "me", "we", "us", "it", "is", "are", "was", "were", "to", "of",
        "saw", "see", "seen", "watch", "watched",
        "音频", "视频", "录像", "录音", "片段", "文件", "媒体", "找", "搜索",
        "查看", "显示", "哪个", "哪些", "那个", "这个", "关于", "有关",
        "里面", "里", "中", "有", "的", "帮我", "一下",
    }
    result: List[str] = []
    seen = set()

    def _push(term: str) -> None:
        value = _clean_space(str(term or "").strip().lower())
        if len(value) < 2 or value in generic or value in _LOOKUP_NOISE:
            return
        if value in seen:
            return
        seen.add(value)
        result.append(value)

    raw_text = str(query_text or "").strip().lower()
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", raw_text):
        topic_candidates = _extract_cjk_topic_terms(run) or [run]
        for topic in topic_candidates:
            anchors = _split_cjk_lookup_anchors(topic)
            if not anchors and not _CJK_ANCHOR_SPLIT_RE.search(topic):
                anchors = [topic]
            for anchor in anchors:
                anchor = str(anchor or "").strip()
                if not (2 <= len(anchor) <= 16):
                    continue
                if anchor in generic or anchor in _CJK_LOOKUP_ANCHOR_NOISE or anchor in _LOOKUP_NOISE:
                    continue
                _push(anchor)
                for alias in _pinyin_aliases_for_cjk_run(anchor):
                    _push(alias)

    words = [
        word
        for word in re.findall(r"[a-z0-9]+", raw_text)
        if word and word not in generic and word not in _LOOKUP_NOISE
    ]
    for word in words:
        _push(word)
    for width in (2, 3):
        for idx in range(0, max(0, len(words) - width + 1)):
            _push(" ".join(words[idx : idx + width]))

    for filelike in extract_filelike_candidates(raw_text, max_candidates=8):
        term = _clean_space(str(filelike or "").strip().lower())
        if not term:
            continue
        if _is_identifierish_term(term) or any(ch.isdigit() for ch in term):
            _push(term)
    return result


def _count_specific_topic_hits(terms: List[str], candidate_text: str) -> int:
    if not terms or not candidate_text:
        return 0
    text_lower = str(candidate_text or "").lower()
    candidate_terms = set(extract_lookup_terms(text_lower, max_terms=256))
    hits = 0
    for term in terms:
        if any("\u4e00" <= ch <= "\u9fff" for ch in term):
            if len(term) >= 2 and term in text_lower:
                hits += 1
            elif term in candidate_terms:
                hits += 1
            continue
        if term in candidate_terms:
            hits += 1
    return hits


def is_lookup_heavy_query(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    compact = _clean_space(raw)
    if (
        len(compact) >= 4
        and " " not in compact
        and any(ch in compact for ch in ("_", "-", ".", "/", "\\"))
    ):
        return True
    if any(ch.isdigit() for ch in raw) and any(ch.isalpha() or ("\u4e00" <= ch <= "\u9fff") for ch in raw):
        return True
    return any(_is_identifierish_term(term) for term in extract_lookup_terms(raw, max_terms=16))
