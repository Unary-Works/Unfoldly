"""
media handlers — extracted from FileAgent media methods.
"""
from __future__ import annotations

import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from core.skills import MediaTimeSkill
from core.retrieval.filename_canonicalizer import (
    compact_filename_key,
    normalize_filename_candidate,
    score_filename_surface_match,
)
from utils.logger import get_logger
logger = get_logger()

_MEDIA_GROUNDING_STOPWORDS = {
    "a", "an", "all", "and", "any", "about", "audio", "audios", "clip", "clips",
    "content", "contents", "file", "files", "find", "for", "from", "in", "is",
    "media", "mention", "mentions", "of", "on", "recording", "recordings", "show",
    "tell", "the", "these", "those", "video", "videos", "what", "which", "with",
}

_DIRECT_PLACEHOLDER_ASR_PATTERNS = (
    re.compile(r'^\(?speaking in (?:a )?foreign language\)?[.!。 ]*$', re.IGNORECASE),
    re.compile(r'^\(?foreign language\)?[.!。 ]*$', re.IGNORECASE),
)

_PLACEHOLDER_SUMMARY_MARKERS = (
    "speaking in foreign language",
    "contains no actual content",
    "unable to perform the requested tasks",
    "non-transcribed audio segments",
    "no discernible english words",
    "only placeholders indicating speech",
)

_MEDIA_FILE_HINT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9_-]{4,}(?![A-Za-z0-9]))"
    r"(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)"
    r"[A-Za-z0-9][A-Za-z0-9_-]{3,}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_GENERIC_MEDIA_STEM_KEYS = {
    compact_filename_key(term)
    for term in (
        "audio", "video", "media", "movie", "recording", "screen recording",
        "clip", "sample", "demo", "file", "voice", "sound",
    )
}
_MEDIA_OVERVIEW_SIGNAL_RE = re.compile(
    r"\b(?:tell\s+me\s+about|describe|summari[sz]e|summary|overview|recap|"
    r"key\s+takeaways?|main\s+points?|what\s+(?:is|was)\s+(?:in|on|about)|"
    r"what\s+does\s+(?:it|this|that|the\s+(?:audio|video|recording|clip|file))\s+(?:contain|say|show))\b"
    r"|介绍|讲讲|总结|概括|概述|主要内容|讲了什么|说了什么|里面有什么|内容是什么",
    re.IGNORECASE,
)


def _media_extensions() -> set[str]:
    try:
        from core.media.media_expert import MEDIA_EXTENSIONS

        return set(MEDIA_EXTENSIONS)
    except Exception:
        return {
            ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
            ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv",
        }


def _infer_media_file_hint_from_paths(query: str, candidate_paths: List[str]) -> str:
    """Infer a named media file from the current context without scanning disk."""
    q = str(query or "").strip()
    if not q or not candidate_paths:
        return ""

    q_key = compact_filename_key(q)
    tokens = [
        compact_filename_key(match.group(0))
        for match in _MEDIA_FILE_HINT_TOKEN_RE.finditer(q)
        if compact_filename_key(match.group(0))
    ]
    media_exts = _media_extensions()
    seen_paths: set[str] = set()
    ranked: List[tuple[int, str, str]] = []

    for raw_path in list(candidate_paths or []):
        fp = str(raw_path or "").strip()
        if not fp or fp in seen_paths:
            continue
        seen_paths.add(fp)
        ext = os.path.splitext(fp)[1].lower()
        if ext not in media_exts:
            continue
        base = os.path.basename(fp)
        stem = os.path.splitext(base)[0]
        stem_key = compact_filename_key(stem)
        base_key = compact_filename_key(base)
        if not stem_key:
            continue

        exact_surface, surface_score = score_filename_surface_match(q, base, "")
        score = 100 if exact_surface else int(surface_score or 0)
        if base_key and len(base_key) >= 5 and base_key in q_key:
            score = max(score, 100)
        if (
            stem_key
            and len(stem_key) >= 4
            and stem_key not in _GENERIC_MEDIA_STEM_KEYS
            and stem_key in q_key
        ):
            score = max(score, 96)
        for token_key in tokens:
            if token_key in {stem_key, base_key}:
                score = max(score, 100)
            elif (
                len(token_key) >= 4
                and token_key not in _GENERIC_MEDIA_STEM_KEYS
                and token_key in stem_key
            ):
                score = max(score, 92)

        if score >= 90:
            ranked.append((score, fp, base))

    if not ranked:
        return ""
    ranked.sort(key=lambda item: (-item[0], item[2].lower(), item[1].lower()))
    best_score = ranked[0][0]
    best_group = [item for item in ranked if item[0] == best_score]
    if len(best_group) > 1:
        best_keys = {compact_filename_key(item[2]) for item in best_group}
        best_paths = {item[1] for item in best_group}
        if len(best_keys) != 1 or len(best_paths) != 1:
            return ""
    return ranked[0][2]


def _score_media_file_hint_match(
    file_hint: str,
    *,
    file_name: str = "",
    file_path: str = "",
    aliases: str = "",
) -> int:
    """Score a user-provided media filename hint against indexed/active media surfaces."""
    hint = normalize_filename_candidate(file_hint)
    if not hint:
        return 0

    surfaces = []
    for raw in (file_name, os.path.basename(str(file_path or "")), file_path):
        surface = str(raw or "").strip()
        if surface and surface not in surfaces:
            surfaces.append(surface)
    if not surfaces:
        return 0

    best = 0
    for surface in surfaces:
        exact, score = score_filename_surface_match(hint, surface, aliases)
        best = max(best, 100 if exact else int(score or 0))

    hint_base = os.path.basename(hint.replace("\\", "/"))
    hint_stem = os.path.splitext(hint_base)[0] if "." in hint_base else hint_base
    hint_base_key = compact_filename_key(hint_base)
    hint_stem_key = compact_filename_key(hint_stem)
    aliases_key = compact_filename_key(aliases)

    for surface in surfaces:
        candidate_base = os.path.basename(surface.replace("\\", "/"))
        candidate_stem = os.path.splitext(candidate_base)[0]
        candidate_base_key = compact_filename_key(candidate_base)
        candidate_stem_key = compact_filename_key(candidate_stem)

        if hint_base_key and hint_base_key == candidate_base_key:
            best = max(best, 100)
        if hint_stem_key and hint_stem_key == candidate_stem_key:
            best = max(best, 98)
        if hint_stem_key and len(hint_stem_key) >= 4 and hint_stem_key in candidate_stem_key:
            best = max(best, 94)
        if candidate_stem_key and len(candidate_stem_key) >= 4 and candidate_stem_key in hint_stem_key:
            best = max(best, 90)

    if aliases_key and hint_stem_key and len(hint_stem_key) >= 4 and hint_stem_key in aliases_key:
        best = max(best, 88)
    return best


def _extract_media_grounding_terms(query: str) -> List[str]:
    q = str(query or "").strip().lower()
    if not q:
        return []

    terms: List[str] = []
    seen = set()
    for raw in re.findall(r"[a-z0-9][a-z0-9._-]{2,}|[\u4e00-\u9fff]{2,}", q):
        term = raw.strip("._-")
        if len(term) >= 4 and term.endswith("s"):
            singular = term[:-1]
            if singular and singular not in _MEDIA_GROUNDING_STOPWORDS:
                term = singular
        if term in _MEDIA_GROUNDING_STOPWORDS:
            continue
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def _media_text_term_matches(text: str, terms: List[str]) -> List[str]:
    haystack = str(text or "").lower()
    if not haystack or not terms:
        return []

    matched: List[str] = []
    for term in terms:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", haystack):
            matched.append(term)
        elif any("\u4e00" <= ch <= "\u9fff" for ch in term) and term in haystack:
            matched.append(term)
    return matched


def _media_hit_grounding_text(hit: dict) -> str:
    return " ".join(
        [
            str(hit.get("file_name") or ""),
            str(hit.get("file_path") or ""),
            str(hit.get("text") or ""),
            str(hit.get("keyframe_description") or ""),
        ]
    )


def _score_media_hit_for_filter(hit: dict, terms: List[str]) -> tuple[int, float, float]:
    matched_terms = _media_text_term_matches(_media_hit_grounding_text(hit), terms)
    distance = float(hit.get("distance") or 1.0)
    if hit.get("chunk_type") == "media_summary":
        kind_bias = 0.0
    elif hit.get("chunk_type") == "media_audio_summary":
        kind_bias = 0.03
    elif hit.get("chunk_type") == "media_visual_summary":
        kind_bias = 0.04
    elif hit.get("chunk_type") == "interval_summary":
        kind_bias = 0.02
    elif hit.get("chunk_type") == "asr_segment":
        kind_bias = 0.05
    elif hit.get("chunk_type") == "asr_transcript":
        kind_bias = 0.1
    elif hit.get("chunk_type") == "keyframe":
        kind_bias = 0.2
    elif hit.get("chunk_type") == "interval_visual":
        kind_bias = 0.22
    else:
        kind_bias = 0.3
    return (-len(set(matched_terms)), distance, kind_bias)


def _apply_media_file_grounding(file_groups: dict, query_topic: str) -> dict:
    terms = _extract_media_grounding_terms(query_topic)
    if not file_groups:
        return file_groups

    ranked_items = []
    for fp, group in file_groups.items():
        hits = list(group.get("hits") or [])
        filter_hits = sorted(hits, key=lambda hit: _score_media_hit_for_filter(hit, terms))
        group["filter_hits"] = filter_hits

        file_text = " ".join(_media_hit_grounding_text(hit) for hit in hits)
        matched_terms = _media_text_term_matches(file_text, terms)
        group["grounding_terms"] = matched_terms
        group["grounding_match_count"] = len(set(matched_terms))
        group["best_distance"] = min(
            (float(hit.get("distance") or 1.0) for hit in hits),
            default=1.0,
        )
        ranked_items.append((fp, group))

    if any(group.get("grounding_match_count", 0) > 0 for _, group in ranked_items):
        ranked_items = [
            (fp, group)
            for fp, group in ranked_items
            if group.get("grounding_match_count", 0) > 0
        ]
        logger.info(
            "[media_content_search] grounding narrowed candidate files to %s using terms=%s",
            len(ranked_items),
            terms,
        )

    ranked_items.sort(
        key=lambda item: (
            -int(item[1].get("grounding_match_count", 0)),
            float(item[1].get("best_distance", 1.0)),
            str(item[1].get("file_name") or ""),
        )
    )
    return type(file_groups)(ranked_items)


def _strip_media_chunk_prefix(text: str) -> str:
    return re.sub(r'^\[.*?\]\s*', '', str(text or '')).strip()


def _looks_like_placeholder_asr_text(text: str, *, chunk_type: str = "") -> bool:
    clean = _strip_media_chunk_prefix(text)
    if not clean:
        return False

    normalized = re.sub(r'\s+', ' ', clean).strip().lower()
    if any(pat.fullmatch(normalized) for pat in _DIRECT_PLACEHOLDER_ASR_PATTERNS):
        return True

    if chunk_type in {"media_summary", "media_audio_summary"}:
        return all(marker in normalized for marker in ("speaking in foreign language", "unable to")) or (
            "speaking in foreign language" in normalized
            and any(marker in normalized for marker in _PLACEHOLDER_SUMMARY_MARKERS[1:])
        )

    return False


def _clean_media_chunk_text(text: str, *, chunk_type: str = "") -> str:
    clean = _strip_media_chunk_prefix(text)
    if not clean:
        return ""
    if _looks_like_placeholder_asr_text(clean, chunk_type=chunk_type):
        return ""
    return clean


def _media_summary_has_foreign_language_notice(summary_docs: List[str]) -> bool:
    for doc in list(summary_docs or []):
        if _looks_like_placeholder_asr_text(str(doc or ""), chunk_type="media_summary"):
            return True
    return False


def _media_hit_time_range(hit: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    ctype = str(hit.get("chunk_type") or "")
    if ctype in {"asr_segment", "asr_transcript"}:
        start = hit.get("asr_start_sec")
        end = hit.get("asr_end_sec")
        return (
            float(start) if start is not None else None,
            float(end) if end is not None else None,
        )
    if ctype == "keyframe":
        t = hit.get("keyframe_time_sec")
        return (float(t), float(t)) if t is not None else (None, None)
    if ctype == "interval_summary":
        start = hit.get("interval_start_sec")
        end = hit.get("interval_end_sec")
        return (
            float(start) if start is not None else None,
            float(end) if end is not None else None,
        )
    if ctype == "interval_visual":
        t = hit.get("interval_visual_time_sec", hit.get("keyframe_time_sec"))
        return (float(t), float(t)) if t is not None else (None, None)
    return (None, None)


def _media_hit_time(h: Dict[str, Any]) -> float:
    start, end = _media_hit_time_range(h)
    if start is not None:
        return float(start)
    if end is not None:
        return float(end)
    return 0.0


def _merge_adjacent_asr_matches(
    matches: List[Dict[str, Any]],
    *,
    max_gap_sec: float = 1.25,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for item in sorted(matches, key=lambda row: (float(row.get("asr_start_sec") or 0.0), float(row.get("asr_end_sec") or 0.0))):
        start = float(item.get("asr_start_sec") or 0.0)
        end = float(item.get("asr_end_sec") or start)
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if not merged:
            merged.append(dict(item))
            continue
        prev = merged[-1]
        prev_end = float(prev.get("asr_end_sec") or prev.get("asr_start_sec") or 0.0)
        if start <= prev_end + max_gap_sec:
            prev["asr_end_sec"] = max(prev_end, end)
            prev_text = str(prev.get("text") or "").strip()
            if text and text not in prev_text:
                prev["text"] = (prev_text + " " + text).strip()
            prev["distance"] = min(float(prev.get("distance") or 1.0), float(item.get("distance") or 1.0))
        else:
            merged.append(dict(item))
    return merged


def _collect_precise_asr_matches(
    metadatas: List[Dict[str, Any]],
    documents: List[str],
    *,
    time_sec: float,
    time_end: Optional[float] = None,
    point_margin_sec: float = 1.5,
    nearest_margin_sec: float = 4.0,
) -> List[Dict[str, Any]]:
    query_end = float(time_end) if time_end is not None else float(time_sec)
    overlap_matches: List[Dict[str, Any]] = []
    nearest_candidates: List[Tuple[float, Dict[str, Any]]] = []

    for meta, doc_text in zip(metadatas or [], documents or []):
        ctype = str((meta or {}).get("chunk_type") or "")
        if ctype not in {"asr_segment", "asr_transcript"}:
            continue
        start_raw = (meta or {}).get("asr_start_sec")
        end_raw = (meta or {}).get("asr_end_sec")
        if start_raw is None:
            continue
        start = float(start_raw)
        end = float(end_raw if end_raw is not None else start_raw)
        clean = _clean_media_chunk_text(doc_text, chunk_type=ctype)
        if not clean:
            continue
        candidate = {
            "chunk_type": ctype,
            "asr_start_sec": round(start, 1),
            "asr_end_sec": round(end, 1),
            "text": clean,
            "distance": 0.0,
            "speaker": str((meta or {}).get("speaker") or ""),
        }

        if time_end is not None:
            if start <= query_end and end >= float(time_sec):
                overlap_matches.append(candidate)
            continue

        if (start - point_margin_sec) <= float(time_sec) <= (end + point_margin_sec):
            overlap_matches.append(candidate)
            continue

        distance = min(abs(start - float(time_sec)), abs(end - float(time_sec)))
        nearest_candidates.append((distance, candidate))

    if overlap_matches:
        return _merge_adjacent_asr_matches(overlap_matches)

    if time_end is None and nearest_candidates:
        nearest_candidates.sort(key=lambda item: item[0])
        best_distance, best = nearest_candidates[0]
        if best_distance <= nearest_margin_sec:
            return [best]

    return []


def _handle_media_export(
    self,
    question: str,
    params: dict,
    *,
    session_id: Optional[str] = None,
    prompt_language: Optional[str] = None,
    active_paths: Optional[List[str]] = None,
):
    """
    Handle time-based audio/video queries.

    Supports:
      - audio_content: "what is said at 30s in audio.mp3" → ASR transcript lookup
      - video_audio: "what is being discussed at 1:20 in video.mp4" → ASR transcript from video's audio
      - video_visual: "what is the scene at 2:30 in video.mp4" → extract frame + VL describe
    """
    import os
    from collections import defaultdict

    time_sec = float(params.get("time_sec", 0)) if params.get("time_sec") is not None else None
    time_end = params.get("time_end_sec")
    target_type = params.get("target_type", "audio_content")
    sub_intent = params.get("sub_intent", "point_lookup")
    file_hint = params.get("file_hint", "")
    media_type_hint = str(params.get("media_type") or "").strip().lower()
    frame_position = str(params.get("frame_position") or "").strip().lower()
    search_concept = params.get("search_concept", "")
    lang = self._resolve_prompt_language(prompt_language, question=question, session_id=session_id)

    operation = str(params.get("operation") or "").strip().lower()
    if str(sub_intent or "").strip().lower() in {"summary", "media_summary", "overview"} or operation == "summary":
        sub_intent = "range_summary"
        params["sub_intent"] = sub_intent

    if not file_hint:
        contextual_paths = list(active_paths or [])
        last_results_getter = getattr(self, "_get_last_search_results_ref", None)
        last_res = last_results_getter(session_id) if callable(last_results_getter) else []
        for item in list(last_res or []):
            fp = str((item or {}).get("file_path") or "").strip()
            if fp:
                contextual_paths.append(fp)
        inferred_hint = _infer_media_file_hint_from_paths(question, contextual_paths)
        if inferred_hint:
            file_hint = inferred_hint
            params["file_hint"] = inferred_hint
            logger.debug("[media_export] inferred file_hint_chars=%s from contextual media paths", len(inferred_hint or ""))

    try:
        from core.media.media_expert import AUDIO_EXTENSIONS as _AUDIO_EXTENSIONS
    except Exception:
        _AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape"}

    hint_ext = os.path.splitext(str(file_hint or ""))[1].lower()
    if media_type_hint == "audio" or hint_ext in _AUDIO_EXTENSIONS:
        target_type = "audio_content" if target_type == "video_visual" else target_type
    if (
        file_hint
        and time_sec is None
        and time_end is None
        and str(sub_intent or "").strip().lower() in {"", "point_lookup"}
        and _MEDIA_OVERVIEW_SIGNAL_RE.search(question or "")
    ):
        sub_intent = "range_summary"
        params["sub_intent"] = sub_intent
    if time_sec is not None and time_end is not None and sub_intent == "point_lookup":
        sub_intent = "range_summary"

    logger.info(
        f"[media_export] sub_intent={sub_intent}, type={target_type}, search_concept={search_concept!r}, "
        f"time={time_sec}s, file_hint={file_hint!r}, lang={lang}"
    )

    def _fmt_time(sec: float) -> str:
        if sec is None: return "0:00"
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        if h > 0: return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _dedupe_timestamps(raw_timestamps: List[float]) -> List[float]:
        seen = set()
        out: List[float] = []
        for raw_ts in raw_timestamps:
            try:
                ts = max(0.0, float(raw_ts))
            except Exception:
                continue
            key = round(ts, 1)
            if key in seen:
                continue
            seen.add(key)
            out.append(ts)
        return out

    # ── Branch A: Content Search (Find where a concept appears in media) ──
    if sub_intent == "content_search" and search_concept:
        yield {"type": "text", "content": f"🔍 Searching media contents for **{search_concept}**...\n\n"}

        try:
            # Use standard search across media chunks
            where_clause = {
                "chunk_type": {"$in": ["asr_segment", "asr_transcript", "keyframe"]}
            }
            if file_hint:
                where_clause["file_name"] = file_hint

            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            # Get embeddings - same as text search
            query_embeds = self.kb.embed_model.get_text_embedding(search_concept) if hasattr(self.kb, "embed_model") and self.kb.embed_model else None

            if query_embeds:
                results = self.kb.collection.query(
                    query_embeddings=[query_embeds],
                    where=where_clause,
                    n_results=10,
                    include=["metadatas", "documents", "distances"]
                )
            else:
                results = self.kb.collection.query(
                    query_texts=[search_concept],
                    where=where_clause,
                    n_results=10,
                    include=["metadatas", "documents", "distances"]
                )

            if not results or not results.get("metadatas") or not results["metadatas"][0]:
                msg = (f"未能在音视频中找到与「{search_concept}」相关的内容。" if lang == "zh" else
                       f"Could not find any media content related to '{search_concept}'.")
                yield {"type": "text", "content": msg}
                yield {"type": "done", "ok": True, "query_type": "media_export", "sources": []}
                return

            # Render matches
            metas = results["metadatas"][0]
            docs = results["documents"][0]
            distances = results["distances"][0]

            # Group by file
            file_matches = defaultdict(list)
            for _meta, _doc, _dist in zip(metas, docs, distances):
                if _dist > 0.5: # Simple threshold logic
                    continue
                fname = _meta.get("file_name", "Unknown File")
                ctype = _meta.get("chunk_type", "")
                sec = float(_meta.get("asr_start_sec", _meta.get("keyframe_time_sec", 0)))
                file_matches[fname].append({"type": ctype, "time": sec, "text": _doc})

            if not file_matches:
                msg = (f"未能在音视频中找到与「{search_concept}」强相关的内容。" if lang == "zh" else
                       f"Could not find strong media matches for '{search_concept}'.")
                yield {"type": "text", "content": msg}
                yield {"type": "done", "ok": True, "query_type": "media_export", "sources": []}
                return

            # Synthesize with LLM
            blocks = []
            for fname, matches in file_matches.items():
                matches.sort(key=lambda x: x["time"])
                blocks.append(f"File: {fname}")
                for m in matches:
                    src_type = "Speech" if m["type"] in {"asr_segment", "asr_transcript"} else "Visual"
                    blocks.append(f"- [{_fmt_time(m['time'])}] ({src_type}): {m['text'][:150]}...")

            ref_text = "\n".join(blocks)

            if lang == "zh":
                prompt = (f"用户想知道「{search_concept}」在音视频中的出现情况。\n"
                          f"以下是找到的相关片段：\n{ref_text}\n\n"
                          "请简要总结在哪些文件的第几秒出现了该内容，具体是如何提及或显示的。")
            else:
                prompt = (f"The user is searching for '{search_concept}' in the media files.\n"
                          f"Here are the occurrences:\n{ref_text}\n\n"
                          "Briefly summarize the files and timestamps where this concept appears and what happens.")

            llm = self._get_llm_service(detailed=False, session_id=session_id, prompt_language=lang)
            for chunk in llm.generate_stream(prompt):
                if self.is_aborted(session_id):
                    yield {"type": "done", "ok": False, "query_type": "interrupted"}
                    return
                yield {"type": "text", "delta": chunk}

            # Sources for UI
            file_paths = []
            for m in metas:
                if m.get("file_path") and m.get("file_path") not in file_paths:
                    file_paths.append(m.get("file_path"))

            sources = []
            for fp in file_paths[:3]:
                sources.append({
                    "file_path": fp,
                    "file_name": os.path.basename(fp),
                    "relevance_score": 1.0,
                })
            yield {"type": "sources", "content": sources, "total_matches": len(sources), "shown_count": len(sources)}
            yield {"type": "done", "ok": True, "query_type": "media_export", "sources": sources}
            return

        except Exception as e:
            logger.error(f"[media_export] Content search failed: {e}", exc_info=True)
            yield {"type": "text", "content": f"搜索失败: {e}"}
            yield {"type": "done", "ok": False, "query_type": "media_export", "sources": []}
            return

    # ── Branch B: Point Lookup (Look up specific timestamp) ──
    target_file_path = None
    target_file_name = None

    def _has_indexed_media_evidence(file_path: str) -> bool:
        fp = str(file_path or "").strip()
        if not fp:
            return False
        try:
            rows = self.kb.collection.get(
                where={"file_path": fp},
                include=["metadatas"],
            )
            return bool((rows or {}).get("metadatas"))
        except Exception as exc:
            logger.debug("[media_export] indexed evidence check failed for %s: %s", fp, exc)
            return False

    def _active_path_matches_file_hint(file_path: str, file_name: str = "") -> bool:
        return _score_media_file_hint_match(
            file_hint,
            file_name=file_name or os.path.basename(str(file_path or "")),
            file_path=file_path,
        ) >= 90

    ambiguous_media_candidates: List[str] = []

    def _pick_active_media_path(*, require_file_hint: bool = False) -> tuple[Optional[str], Optional[str]]:
        if not active_paths:
            return None, None
        try:
            from core.media.media_expert import MEDIA_EXTENSIONS, VIDEO_EXTENSIONS
        except Exception:
            MEDIA_EXTENSIONS = {
                ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
                ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv",
            }
            VIDEO_EXTENSIONS = {
                ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv",
            }

        matches: List[str] = []
        for raw_path in list(active_paths or []):
            fp = str(raw_path or "").strip()
            if not fp:
                continue
            ext = os.path.splitext(fp)[1].lower()
            if ext not in MEDIA_EXTENSIONS:
                continue
            if target_type == "video_visual" and ext not in VIDEO_EXTENSIONS:
                continue
            if require_file_hint and not _active_path_matches_file_hint(fp):
                continue
            if not _has_indexed_media_evidence(fp):
                logger.info(
                    "[media_export] selected media path has no indexed DB evidence; "
                    "skipping query-time local fallback: %s",
                    fp,
                )
                continue
            matches.append(fp)

        if not matches:
            return None, None
        if file_hint:
            hinted_matches = [fp for fp in matches if _active_path_matches_file_hint(fp)]
            if hinted_matches:
                matches = hinted_matches
        if len(matches) > 1 and not file_hint and not require_file_hint:
            logger.info(
                "[media_export] multiple media files available without a file hint; "
                "asking for clarification instead of defaulting to the first match: %s",
                matches[:5],
            )
            ambiguous_media_candidates[:] = matches[:5]
            return None, None
        if len(matches) > 1:
            logger.info("[media_export] multiple hinted media matches; using first match: %s", matches[0])
        chosen = matches[0]
        return chosen, os.path.basename(chosen)

    def _find_indexed_media_by_file_hint() -> tuple[Optional[str], Optional[str]]:
        if not file_hint:
            return None, None
        try:
            from core.retrieval.category_engine import get_compatible_categories

            rows = self.kb.collection.get(
                where={"doc_category": {"$in": sorted(get_compatible_categories("audio/video"))}},
                include=["metadatas"],
            )
        except Exception as exc:
            logger.debug("[media_export] indexed media scan by file hint failed: %s", exc)
            try:
                rows = self.kb.collection.get(include=["metadatas"])
            except Exception:
                rows = {}

        ranked: List[tuple[int, str, str]] = []
        best_by_key: Dict[str, tuple[int, str, str]] = {}
        media_exts = _media_extensions()
        for meta in list((rows or {}).get("metadatas") or []):
            meta = meta or {}
            fp = str(meta.get("file_path") or "").strip()
            fname = str(meta.get("file_name") or os.path.basename(fp) or "").strip()
            ext = os.path.splitext(fname or fp)[1].lower()
            if ext and ext not in media_exts:
                continue
            aliases = " ".join(
                str(meta.get(key) or "")
                for key in ("title", "original_file_name", "source_name", "display_name")
            )
            score = _score_media_file_hint_match(
                file_hint,
                file_name=fname,
                file_path=fp,
                aliases=aliases,
            )
            if score < 90:
                continue
            key = fp or fname
            if not key:
                continue
            previous = best_by_key.get(key)
            candidate = (score, fp, fname or os.path.basename(fp))
            if previous is None or score > previous[0]:
                best_by_key[key] = candidate

        ranked = list(best_by_key.values())
        if not ranked:
            return None, None
        ranked.sort(key=lambda item: (-item[0], item[2].lower(), item[1].lower()))
        top_score = ranked[0][0]
        top = [item for item in ranked if item[0] == top_score]
        top_paths = {item[1] for item in top if item[1]}
        top_names = {compact_filename_key(item[2]) for item in top if item[2]}
        if len(top_paths) > 1 and len(top_names) > 1:
            ambiguous_media_candidates[:] = [item[1] for item in top[:5] if item[1]]
            logger.info("[media_export] file hint matched multiple indexed media candidates: %s", top[:5])
            return None, None
        return ranked[0][1], ranked[0][2]

    # Try file_hint first (exact filename match in KB)
    if file_hint:
        try:
            results = self.kb.collection.get(
                where={"file_name": file_hint},
                include=["metadatas"],
            )
            if results and results.get("metadatas"):
                target_file_path = results["metadatas"][0].get("file_path", "")
                target_file_name = results["metadatas"][0].get("file_name", file_hint)
        except Exception:
            pass

        if not target_file_path:
            target_file_path, target_file_name = _pick_active_media_path(require_file_hint=True)

        if not target_file_path:
            target_file_path, target_file_name = _find_indexed_media_by_file_hint()

        # No source-store fallback here: dialogue-time media QA is DB-only.

        if not target_file_path:
            msg = (
                f"未在索引中找到名为「{file_hint}」的音视频文件，无法基于对话阶段读取本地文件来回答。请先重新建立索引。"
                if lang == "zh"
                else f"No indexed audio/video evidence was found for '{file_hint}', so I cannot answer it by reading the local file at chat time. Please re-index the file first."
            )
            yield {"type": "text", "content": msg}
            yield {"type": "done", "ok": True, "query_type": "media_export", "sources": []}
            return

    # Prefer the actively selected media file(s) when the user asks about
    # "this video/audio" without naming the file explicitly.
    if not target_file_path:
        target_file_path, target_file_name = _pick_active_media_path()

    # Fallback: use the most recent media file from last results
    if not target_file_path:
        last_results_getter = getattr(self, "_get_last_search_results_ref", None)
        last_res = last_results_getter(session_id) if callable(last_results_getter) else []
        if last_res:
            from core.media.media_expert import MEDIA_EXTENSIONS
            prior_media: List[tuple[str, str]] = []
            seen_prior_media: set[str] = set()
            for r in last_res:
                fp = r.get("file_path", "")
                ext = os.path.splitext(fp)[1].lower()
                if ext in MEDIA_EXTENSIONS:
                    key = str(fp or r.get("file_name") or "").strip()
                    if key and key in seen_prior_media:
                        continue
                    if key:
                        seen_prior_media.add(key)
                    prior_media.append((fp, r.get("file_name", os.path.basename(fp))))
            if len(prior_media) == 1:
                target_file_path, target_file_name = prior_media[0]
            elif len(prior_media) > 1:
                ambiguous_media_candidates[:] = [fp for fp, _ in prior_media[:5] if fp]

    if not target_file_path and ambiguous_media_candidates:
        sources = [
            {
                "file_path": fp,
                "file_name": os.path.basename(fp),
                "relevance_score": 1.0,
            }
            for fp in ambiguous_media_candidates[:5]
        ]
        msg = (
            "当前上下文里有多个音视频文件。请先说具体文件名，我再帮你看时间点或时长。"
            if lang == "zh"
            else "There are multiple audio/video files in the current context. Please name the specific file, then I can answer the timestamp or duration question."
        )
        yield {"type": "text", "content": msg}
        yield {"type": "sources", "content": sources, "total_matches": len(sources), "shown_count": len(sources)}
        yield {"type": "done", "ok": True, "query_type": "media_export", "sources": sources}
        return

    # Still no file found → search KB for any media file matching query
    if not target_file_path:
        try:
            from core.retrieval.category_engine import get_compatible_categories

            search_results = self.kb.collection.get(
                where={"doc_category": {"$in": sorted(get_compatible_categories("audio/video"))}},
                include=["metadatas"],
            )
            if search_results and search_results.get("metadatas"):
                # Pick the first media file
                meta = search_results["metadatas"][0]
                target_file_path = meta.get("file_path", "")
                target_file_name = meta.get("file_name", "")
        except Exception:
            pass

    if not target_file_path:
        msg = (
            "未找到相关的音视频文件。请先搜索或指定文件名。"
            if lang == "zh"
            else "No matching audio/video file found. Please search or specify a filename."
        )
        yield {"type": "text", "content": msg}
        yield {"type": "done", "ok": True, "query_type": "media_export", "sources": []}
        return

    def _target_duration() -> float:
        max_duration = 0.0
        try:
            rows = self.kb.collection.get(
                where={"file_path": target_file_path},
                include=["metadatas"],
            )
            for meta in list((rows or {}).get("metadatas") or []):
                meta = meta or {}
                for key in (
                    "media_duration_sec",
                    "duration_sec",
                    "audio_duration_sec",
                    "video_duration_sec",
                ):
                    value = meta.get(key)
                    if value is None:
                        continue
                    try:
                        max_duration = max(max_duration, float(value or 0.0))
                    except Exception:
                        pass
                for key in ("asr_end_sec", "keyframe_time_sec"):
                    value = meta.get(key)
                    if value is None:
                        continue
                    try:
                        max_duration = max(max_duration, float(value or 0.0))
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("[media_export] DB duration lookup failed for %s: %s", os.path.basename(target_file_path), exc)
        if (
            max_duration <= 0
            and target_file_path
            and os.path.exists(target_file_path)
            and os.getenv("UNFOLDLY_ENABLE_PYAV", "0").strip().lower() in {"1", "true", "yes", "on"}
        ):
            try:
                import av as _av

                with _av.open(target_file_path) as container:
                    if container.duration:
                        max_duration = max(max_duration, float(container.duration) / float(_av.time_base))
                    for stream in list(container.streams or []):
                        if stream.duration and stream.time_base:
                            max_duration = max(max_duration, float(stream.duration * stream.time_base))
            except Exception as exc:
                logger.debug("[media_export] PyAV duration probe failed for %s: %s", os.path.basename(target_file_path), exc)
        if max_duration <= 0 and target_file_path and os.path.exists(target_file_path):
            try:
                import json
                import subprocess
                from core.media.media_expert import MediaExpert

                ffprobe = MediaExpert._ffprobe_cmd()
                if not ffprobe:
                    raise FileNotFoundError("ffprobe")
                proc = subprocess.run(
                    [
                        ffprobe,
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "json",
                        target_file_path,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                payload = json.loads(proc.stdout or "{}")
                max_duration = max(max_duration, float((payload.get("format") or {}).get("duration") or 0.0))
            except Exception as exc:
                logger.debug("[media_export] ffprobe duration probe failed for %s: %s", os.path.basename(target_file_path), exc)
        return max_duration

    duration_cache: Optional[float] = None
    if sub_intent == "range_summary" and time_end is None:
        if time_sec is None:
            time_sec = 0.0
        duration_cache = _target_duration()
        if duration_cache > float(time_sec):
            time_end = duration_cache

    if time_sec is None:
        if frame_position in {"middle", "last"}:
            duration = duration_cache if duration_cache is not None else _target_duration()
            if frame_position == "middle" and duration > 0:
                time_sec = max(0.0, duration / 2.0)
            elif frame_position == "last" and duration > 0:
                time_sec = max(0.0, duration - 0.1)
            else:
                time_sec = 0.0
        else:
            time_sec = 0.0

    source_payload = [{
        "file_path": target_file_path,
        "file_name": target_file_name or os.path.basename(target_file_path),
        "relevance_score": 1.0,
    }]

    duration_cache = duration_cache if duration_cache is not None else _target_duration()
    if sub_intent == "duration_lookup":
        if duration_cache > 0:
            msg = (
                f"`{target_file_name}` 的总时长是 {_fmt_time(duration_cache)}（约 {duration_cache:.1f} 秒）。"
                if lang == "zh"
                else f"`{target_file_name}` is {_fmt_time(duration_cache)} long (about {duration_cache:.1f} seconds)."
            )
        else:
            msg = (
                f"我没有在索引或本地媒体元数据中读到 `{target_file_name}` 的准确时长。"
                if lang == "zh"
                else f"I couldn't read an exact duration for `{target_file_name}` from the index or local media metadata."
            )
        yield {"type": "text", "content": msg + "\n"}
        yield {"type": "sources", "content": source_payload, "total_matches": 1, "shown_count": 1}
        yield {"type": "done", "ok": True, "query_type": "media_export", "sources": source_payload}
        return
    if duration_cache > 0 and time_sec is not None:
        requested_start = float(time_sec)
        query_starts_past_end = requested_start > duration_cache + 0.5
        point_query_past_end = time_end is None and query_starts_past_end
        range_query_past_end = time_end is not None and query_starts_past_end
        if point_query_past_end or range_query_past_end:
            if lang == "zh":
                msg = (
                    f"`{target_file_name}` 的总时长只有 {_fmt_time(duration_cache)}"
                    f"（约 {duration_cache:.1f} 秒），没有 {_fmt_time(requested_start)} 这个时间点。"
                )
            else:
                msg = (
                    f"`{target_file_name}` is only {_fmt_time(duration_cache)} long "
                    f"(about {duration_cache:.1f}s), so it does not have the timestamp {_fmt_time(requested_start)}."
                )
            yield {"type": "text", "content": msg + "\n"}
            yield {"type": "sources", "content": source_payload, "total_matches": 1, "shown_count": 1}
            yield {"type": "done", "ok": True, "query_type": "media_export", "sources": source_payload}
            return

    params["time_sec"] = float(time_sec)
    if time_end is not None:
        params["time_end_sec"] = float(time_end)

    logger.info(f"[media_timequery] Target file: {target_file_name}")
    has_foreign_language_notice = False
    media_summary_excerpt = ""
    try:
        media_summary_rows = self.kb.collection.get(
            where={
                "$and": [
                    {"file_path": target_file_path},
                    {"chunk_type": {"$in": ["media_summary", "media_audio_summary"]}},
                ]
            },
            include=["documents"],
        )
        summary_docs = [str(doc or "").strip() for doc in list(media_summary_rows.get("documents") or []) if str(doc or "").strip()]
        has_foreign_language_notice = _media_summary_has_foreign_language_notice(summary_docs)
        for doc in summary_docs:
            cleaned = _clean_media_chunk_text(doc, chunk_type="media_summary")
            if cleaned:
                media_summary_excerpt = cleaned
                break
    except Exception as exc:
        logger.debug("[media_timequery] media summary lookup failed for %s: %s", os.path.basename(target_file_path), exc)

    time_label = _fmt_time(time_sec)
    if time_end:
        time_label = f"{_fmt_time(time_sec)} - {_fmt_time(float(time_end))}"
    visual_anchor_sec = float(time_sec if time_end is None else (float(time_sec) + float(time_end)) / 2.0)
    from core.media.media_expert import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
    media_ext = os.path.splitext(target_file_path)[1].lower()
    media_time_plan = MediaTimeSkill.build_plan(
        question,
        params,
        target_file_name=target_file_name,
        language=lang,
        has_video=media_ext in VIDEO_EXTENSIONS,
        has_audio=(media_ext in VIDEO_EXTENSIONS) or (media_ext in AUDIO_EXTENSIONS),
    )
    yield {
        "type": "thinking",
        "delta": (
            f"Activated media_time_skill in {media_time_plan.query_mode} mode for `{target_file_name}`.\n"
        ),
    }

    def _collect_point_transcript_rows() -> List[Dict[str, Any]]:
        if not media_time_plan.has_audio:
            return []
        try:
            precise_chunks = self.kb.collection.get(
                where={
                    "$and": [
                        {"file_path": target_file_path},
                        {"chunk_type": "asr_segment"},
                    ]
                },
                include=["metadatas", "documents"],
            )
            precise_matches = _collect_precise_asr_matches(
                list(precise_chunks.get("metadatas") or []),
                list(precise_chunks.get("documents") or []),
                time_sec=float(time_sec),
                time_end=float(time_end) if time_end is not None else None,
                point_margin_sec=3.0,
                nearest_margin_sec=8.0,
            )
            if precise_matches:
                return precise_matches

            coarse_chunks = self.kb.collection.get(
                where={
                    "$and": [
                        {"file_path": target_file_path},
                        {"chunk_type": "asr_transcript"},
                    ]
                },
                include=["metadatas", "documents"],
            )
            return _collect_precise_asr_matches(
                list(coarse_chunks.get("metadatas") or []),
                list(coarse_chunks.get("documents") or []),
                time_sec=float(time_sec),
                time_end=float(time_end) if time_end is not None else None,
                point_margin_sec=8.0,
                nearest_margin_sec=12.0,
            )
        except Exception as exc:
            logger.warning(f"[media_timequery] indexed ASR lookup failed: {exc}")
            return []

    def _format_transcript_rows(rows: List[Dict[str, Any]]) -> str:
        return "\n\n".join(
            f"**[{_fmt_time(row['asr_start_sec'])} - {_fmt_time(row['asr_end_sec'])}]**\n{row['text']}"
            for row in rows
            if str(row.get("text") or "").strip()
        )

    def _collect_indexed_point_visual_entries(
        *,
        window_sec: float = 12.0,
        nearest_fallback_sec: float = 15.0,
        max_items: int = 5,
    ) -> List[Tuple[float, str]]:
        if not media_time_plan.has_video:
            return []
        try:
            kf_results = self.kb.collection.get(
                where={
                    "$and": [
                        {"file_path": target_file_path},
                        {"chunk_type": "keyframe"},
                    ]
                },
                include=["metadatas", "documents"],
            )
        except Exception as exc:
            logger.debug(f"[media_timequery] indexed point keyframe lookup failed: {exc}")
            return []

        entries: List[Tuple[float, str, float]] = []
        for meta, doc_text in zip(
            list((kf_results or {}).get("metadatas") or []),
            list((kf_results or {}).get("documents") or []),
        ):
            try:
                ts = float((meta or {}).get("keyframe_time_sec"))
            except Exception:
                continue
            desc = str((meta or {}).get("keyframe_description") or doc_text or "").strip()
            if not desc:
                continue
            dist = abs(ts - visual_anchor_sec)
            if dist <= window_sec:
                entries.append((ts, desc, dist))

        if not entries:
            all_entries: List[Tuple[float, str, float]] = []
            for meta, doc_text in zip(
                list((kf_results or {}).get("metadatas") or []),
                list((kf_results or {}).get("documents") or []),
            ):
                try:
                    ts = float((meta or {}).get("keyframe_time_sec"))
                except Exception:
                    continue
                desc = str((meta or {}).get("keyframe_description") or doc_text or "").strip()
                if desc:
                    all_entries.append((ts, desc, abs(ts - visual_anchor_sec)))
            nearest_entries = sorted(all_entries, key=lambda item: item[2])[:1]
            entries = [
                item
                for item in nearest_entries
                if float(item[2]) <= float(nearest_fallback_sec)
            ]

        entries = sorted(entries, key=lambda item: (item[0], item[2]))
        if len(entries) > max_items:
            nearest = sorted(entries, key=lambda item: item[2])[:max_items]
            entries = sorted(nearest, key=lambda item: item[0])
        return [(ts, desc) for ts, desc, _dist in entries]

    def _cleanup_frame_path(frame_path: str) -> None:
        try:
            if frame_path and os.path.exists(frame_path):
                os.unlink(frame_path)
            frame_dir = os.path.dirname(frame_path or "")
            if frame_dir and os.path.basename(frame_dir).startswith("media_frame_"):
                shutil.rmtree(frame_dir, ignore_errors=True)
        except Exception:
            pass

    def _collect_realtime_point_visual_entries(
        *,
        window_sec: float = 4.0,
        count: int = 3,
    ) -> List[Tuple[float, str]]:
        if not media_time_plan.has_video:
            return []
        if not target_file_path or not os.path.exists(target_file_path):
            logger.info("[media_timequery] cannot extract query-time frames; file missing: %s", os.path.basename(target_file_path))
            return []

        try:
            from core.media.media_expert import MediaExpert
        except Exception as exc:
            logger.warning("[media_timequery] MediaExpert unavailable for query-time frame extraction: %s", exc)
            return []

        anchor = max(0.0, float(visual_anchor_sec))
        total = max(1, int(count or 1))
        if total <= 1:
            raw_times = [anchor]
        else:
            start = max(0.0, anchor - float(window_sec) / 2.0)
            step = float(window_sec) / max(1, total - 1)
            raw_times = [start + idx * step for idx in range(total)]
        if duration_cache and duration_cache > 0:
            upper = max(0.0, float(duration_cache) - 0.05)
            raw_times = [min(max(0.0, ts), upper) for ts in raw_times]
        sample_times = _dedupe_timestamps(raw_times)
        if not sample_times:
            sample_times = [anchor]

        expert = MediaExpert()
        frame_items: List[Tuple[float, str]] = []
        for ts in sample_times:
            try:
                frame_path = expert.extract_frame_at(target_file_path, float(ts))
            except Exception as exc:
                logger.warning("[media_timequery] query-time frame extraction failed at %.2fs: %s", float(ts), exc)
                frame_path = None
            if frame_path:
                frame_items.append((float(ts), frame_path))

        if not frame_items:
            return []

        descriptions: Dict[float, str] = {}
        try:
            batch_fn = getattr(self.kb, "_generate_video_frame_batch_summaries", None)
            if callable(batch_fn):
                labeled = [(f"F{idx}", path) for idx, (_ts, path) in enumerate(frame_items, 1)]
                try:
                    batch_result = batch_fn(labeled)
                except TypeError:
                    batch_result = batch_fn(labeled, prev_description="")
                for idx, (ts, _path) in enumerate(frame_items, 1):
                    desc = str((batch_result or {}).get(f"F{idx}") or "").strip()
                    if desc:
                        descriptions[float(ts)] = desc

            single_fn = getattr(self.kb, "_generate_video_frame_summary", None)
            prev_desc = ""
            for ts, path in frame_items:
                if float(ts) in descriptions:
                    prev_desc = descriptions[float(ts)]
                    continue
                desc = ""
                if callable(single_fn):
                    try:
                        desc = str(single_fn(path, prev_description=prev_desc) or "").strip()
                    except TypeError:
                        desc = str(single_fn(path) or "").strip()
                if desc:
                    descriptions[float(ts)] = desc
                    prev_desc = desc
        finally:
            for _ts, path in frame_items:
                _cleanup_frame_path(path)

        return [
            (ts, desc)
            for ts, desc in sorted(descriptions.items(), key=lambda item: item[0])
            if str(desc or "").strip()
        ]

    if media_time_plan.query_mode == "range_summary":
        span_sec = max(0.0, float(time_end) - float(time_sec))
        recommended_frame_count = MediaTimeSkill.recommended_interval_frame_count(
            span_sec,
            has_audio=media_time_plan.has_audio,
        )
        dense_frame_count = MediaTimeSkill.recommended_interval_frame_count(
            span_sec,
            has_audio=False,
            prefer_dense=True,
        )
        cached_interval = self.kb.get_cached_media_interval_analysis(
            target_file_path,
            float(time_sec),
            float(time_end),
            language=lang,
        )
        cached_visual_entries = MediaTimeSkill.sample_timeline_entries(
            list(cached_interval.get("visual_entries") or []),
            max_items=min(max(recommended_frame_count, 6), 10),
        )

        yield {
            "type": "status",
            "phase": "running",
            "message": MediaTimeSkill.stage_message(
                "lock_interval",
                plan=media_time_plan,
                time_label=time_label,
            ),
        }
        yield {
            "type": "thinking",
            "delta": (
                f"Locked interval {time_label}; preparing interval-level video/audio analysis.\n"
            ),
        }
        cached_frame_count = int(cached_interval.get("frame_count") or 0)
        cache_is_sufficient = bool(cached_interval.get("summary_text")) and (
            (not media_time_plan.has_video)
            or cached_frame_count >= max(3, recommended_frame_count // 2)
        )
        if cache_is_sufficient:
            yield {
                "type": "status",
                "phase": "running",
                "message": MediaTimeSkill.stage_message(
                    "reuse_interval_cache",
                    plan=media_time_plan,
                    time_label=time_label,
                ),
            }
            yield {
                "type": "thinking",
                "delta": (
                    "Found cached interval analysis with "
                    f"{cached_frame_count} frames and "
                    f"{int(cached_interval.get('transcript_count') or 0)} transcript spans; "
                    "reusing it directly.\n"
                ),
            }
            yield {"type": "text", "content": str(cached_interval.get("summary_text") or "").strip() + "\n"}
            yield {"type": "sources", "content": source_payload, "total_matches": 1, "shown_count": 1}
            yield {"type": "done", "ok": True, "query_type": "media_export", "sources": source_payload}
            return

        def _collect_range_transcript_rows() -> List[Dict[str, Any]]:
            try:
                precise_chunks = self.kb.collection.get(
                    where={
                        "$and": [
                            {"file_path": target_file_path},
                            {"chunk_type": "asr_segment"},
                        ]
                    },
                    include=["metadatas", "documents"],
                )
                rows = _collect_precise_asr_matches(
                    list(precise_chunks.get("metadatas") or []),
                    list(precise_chunks.get("documents") or []),
                    time_sec=float(time_sec),
                    time_end=float(time_end) if time_end is not None else None,
                )
                if rows:
                    return rows

                coarse_chunks = self.kb.collection.get(
                    where={
                        "$and": [
                            {"file_path": target_file_path},
                            {"chunk_type": "asr_transcript"},
                        ]
                    },
                    include=["metadatas", "documents"],
                )
                return _collect_precise_asr_matches(
                    list(coarse_chunks.get("metadatas") or []),
                    list(coarse_chunks.get("documents") or []),
                    time_sec=float(time_sec),
                    time_end=float(time_end) if time_end is not None else None,
                    point_margin_sec=8.0,
                    nearest_margin_sec=12.0,
                )
            except Exception as exc:
                logger.warning(f"[media_range_summary] indexed transcript lookup failed: {exc}")
                return []

        def _merge_visual_entries(*entry_groups: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
            merged: Dict[float, str] = {}
            for group in entry_groups:
                for ts, desc in group or []:
                    clean_desc = str(desc or "").strip()
                    if not clean_desc:
                        continue
                    key = round(float(ts), 1)
                    if key not in merged or len(clean_desc) > len(merged[key]):
                        merged[key] = clean_desc
            return sorted(((ts, desc) for ts, desc in merged.items()), key=lambda item: item[0])

        def _combine_evidence_source(*labels: str) -> str:
            ordered: List[str] = []
            for label in labels:
                clean = str(label or "").strip()
                if clean and clean != "none" and clean not in ordered:
                    ordered.append(clean)
            if not ordered:
                return "none"
            if len(ordered) == 1:
                return ordered[0]
            return "+".join(ordered)

        def _collect_visual_range_entries(sample_count: int) -> Tuple[List[Tuple[float, str]], str]:
            target_total = min(max(sample_count, 6), 10)
            indexed_entries: List[Tuple[float, str]] = []
            cached_entries: List[Tuple[float, str]] = list(cached_visual_entries or [])
            source_label = "none"
            try:
                kf_results = self.kb.collection.get(
                    where={
                        "$and": [
                            {"file_path": target_file_path},
                            {"chunk_type": "keyframe"},
                        ]
                    },
                    include=["metadatas", "documents"],
                )
                for meta, doc_text in zip(
                    list(kf_results.get("metadatas") or []),
                    list(kf_results.get("documents") or []),
                ):
                    ts = meta.get("keyframe_time_sec")
                    if ts is None:
                        continue
                    tsf = float(ts)
                    if not (float(time_sec) <= tsf <= float(time_end)):
                        continue
                    desc = str(meta.get("keyframe_description") or doc_text or "").strip()
                    if desc:
                        indexed_entries.append((tsf, desc))
            except Exception as exc:
                logger.warning(f"[media_range_summary] indexed keyframe lookup failed: {exc}")
            indexed_entries = _merge_visual_entries(indexed_entries)
            cached_entries = _merge_visual_entries(cached_entries)
            existing_entries = _merge_visual_entries(indexed_entries, cached_entries)
            if indexed_entries:
                source_label = _combine_evidence_source(source_label, "indexed_keyframes")
            if cached_entries:
                source_label = _combine_evidence_source(source_label, "cached_interval_visual")

            missing_indexed_slots = MediaTimeSkill.remaining_interval_frame_budget(
                target_total,
                len(existing_entries),
            )
            logger.info(
                "[media_range_summary] visual_plan file=%s interval=%s target_total=%s indexed=%s cached=%s existing=%s missing_indexed_slots=%s",
                target_file_name,
                time_label,
                target_total,
                len(indexed_entries),
                len(cached_entries),
                len(existing_entries),
                missing_indexed_slots,
            )
            if missing_indexed_slots > 0:
                logger.info(
                    "[media_range_summary] no query-time frame extraction; "
                    "using indexed/cached visual evidence only"
                )
            combined = _merge_visual_entries(existing_entries)
            logger.info(
                "[media_range_summary] visual_result file=%s interval=%s indexed=%s cached=%s combined=%s",
                target_file_name,
                time_label,
                len(indexed_entries),
                len(cached_entries),
                len(combined),
            )
            return (
                MediaTimeSkill.sample_timeline_entries(
                    combined,
                    max_items=target_total,
                ),
                source_label,
            )

        visual_entries: List[Tuple[float, str]] = []
        visual_source = "none"
        try:
            if media_time_plan.has_video:
                yield {
                    "type": "status",
                    "phase": "running",
                    "message": MediaTimeSkill.stage_message(
                        "lookup_interval_visual",
                        plan=media_time_plan,
                        time_label=time_label,
                    ),
                }
                yield {
                    "type": "thinking",
                    "delta": (
                        f"Loading indexed visual evidence across {time_label}.\n"
                    ),
                }
                visual_entries, visual_source = _collect_visual_range_entries(recommended_frame_count)
        except Exception:
            pass

        yield {
            "type": "status",
            "phase": "running",
            "message": MediaTimeSkill.stage_message(
                "collect_audio",
                plan=media_time_plan,
                time_label=time_label,
            ),
        }
        transcript_rows = _collect_range_transcript_rows()
        transcript_source = "indexed_asr" if transcript_rows else "none"
        if not transcript_rows:
            logger.info(
                "[media_range_summary] no indexed transcript evidence; "
                "dialogue-time ASR is disabled"
            )

        if (
            media_time_plan.has_video
            and not transcript_rows
            and len(visual_entries) < max(4, recommended_frame_count // 2)
        ):
            yield {
                "type": "thinking",
                "delta": (
                    "Speech evidence is weak in this interval; increasing visual sampling "
                    f"to about {dense_frame_count} frames.\n"
                ),
            }
            denser_entries, denser_source = _collect_visual_range_entries(dense_frame_count)
            visual_entries = MediaTimeSkill.sample_timeline_entries(
                _merge_visual_entries(visual_entries, denser_entries),
                max_items=min(max(dense_frame_count, 6), 10),
            )
            visual_source = _combine_evidence_source(visual_source, denser_source)

        logger.info(
            "[media_range_summary] file=%s interval=%s transcript_rows=%s transcript_source=%s "
            "visual_entries=%s visual_source=%s",
            target_file_name,
            time_label,
            len(transcript_rows),
            transcript_source,
            len(visual_entries),
            visual_source,
        )
        yield {
            "type": "thinking",
            "delta": (
                f"Collected {len(visual_entries)} visual snapshots ({visual_source}) and "
                f"{len(transcript_rows)} transcript spans ({transcript_source}).\n"
            ),
        }

        if not transcript_rows and not visual_entries:
            if cached_interval.get("summary_text"):
                yield {
                    "type": "thinking",
                    "delta": "Fresh evidence was limited, so I fell back to the cached interval summary.\n",
                }
                yield {"type": "text", "content": str(cached_interval.get("summary_text") or "").strip() + "\n"}
                yield {"type": "sources", "content": source_payload, "total_matches": 1, "shown_count": 1}
                yield {"type": "done", "ok": True, "query_type": "media_export", "sources": source_payload}
                return
            msg = (
                f"在 `{target_file_name}` 的 **{time_label}** 范围内没有找到足够的语音或画面证据来做总结。"
                if lang == "zh"
                else f"I couldn't find enough speech or visual evidence in `{target_file_name}` within **{time_label}** to summarize that range."
            )
            yield {"type": "text", "content": msg + "\n"}
            yield {"type": "sources", "content": source_payload, "total_matches": 1, "shown_count": 1}
            yield {"type": "done", "ok": True, "query_type": "media_export", "sources": source_payload}
            return

        prompt = MediaTimeSkill.build_range_summary_prompt(
            plan=media_time_plan,
            time_label=time_label,
            transcript_rows=transcript_rows,
            visual_entries=visual_entries,
            format_time=_fmt_time,
        )

        yield {
            "type": "status",
            "phase": "running",
            "message": MediaTimeSkill.stage_message(
                "generate_range_summary",
                plan=media_time_plan,
                time_label=time_label,
            ),
        }
        yield {
            "type": "thinking",
            "delta": (
                "正在理解视频内容并组织连贯回答，这一步通常会比抽帧更久一些。\n"
                if lang == "zh"
                else "Understanding the video content and composing a coherent answer; this step can take longer than frame extraction.\n"
            ),
        }
        llm = self._get_llm_service(detailed=True, session_id=session_id, prompt_language=lang)
        answer_parts: List[str] = []
        for chunk in llm.generate_stream(prompt):
            if self.is_aborted(session_id):
                yield {"type": "done", "ok": False, "query_type": "interrupted", "sources": source_payload}
                return
            if chunk:
                answer_parts.append(chunk)
                yield {"type": "text", "delta": chunk}
        yield {"type": "text", "delta": "\n"}
        final_summary = "".join(answer_parts).strip()
        if final_summary:
            try:
                yield {
                    "type": "status",
                    "phase": "running",
                    "message": MediaTimeSkill.stage_message(
                        "cache_interval",
                        plan=media_time_plan,
                        time_label=time_label,
                    ),
                }
                stored_chunks = self.kb.persist_media_interval_analysis(
                    target_file_path,
                    float(time_sec),
                    float(time_end),
                    summary_text=final_summary,
                    visual_entries=visual_entries,
                    transcript_rows=transcript_rows,
                    source_label=_combine_evidence_source(visual_source, transcript_source),
                    answer_language=lang,
                )
                if stored_chunks:
                    yield {
                        "type": "thinking",
                        "delta": (
                            f"Stored {stored_chunks} interval evidence chunks for faster follow-up queries.\n"
                        ),
                    }
            except Exception as exc:
                logger.warning(f"[media_range_summary] interval cache upsert failed: {exc}")
        yield {"type": "sources", "content": source_payload, "total_matches": 1, "shown_count": 1}
        yield {"type": "done", "ok": True, "query_type": "media_export", "sources": source_payload}
        return

    # ── Step 2: Handle based on target_type ───────────────────────────

    if target_type in ("audio_content", "video_audio"):
        # ── ASR transcript lookup ─────────────────────────────────────
        yield {
            "type": "status",
            "phase": "running",
            "message": MediaTimeSkill.stage_message(
                "lookup_transcript",
                plan=media_time_plan,
                time_label=time_label,
            ),
        }
        yield {"type": "text", "content": f"🎙️ Looking up transcript at **{time_label}** in `{target_file_name}`...\n\n"}

        transcript_text = _format_transcript_rows(_collect_point_transcript_rows())

        # Try 2 intentionally does not exist: dialogue-time media QA must not launch
        # ASR models. Indexing owns transcript generation; missing transcript
        # evidence should be reported or paired with visual frame analysis.
        if not transcript_text:
            logger.info(
                "[media_timequery] no indexed transcript evidence; dialogue-time ASR is disabled"
            )

        if transcript_text:
            if has_foreign_language_notice and "foreign language" not in transcript_text.lower():
                transcript_text = (
                    transcript_text
                    + (
                        "\n\n注：该音频被索引为包含 foreign language speech（外语语音）。"
                        if lang == "zh"
                        else "\n\nNote: this file is indexed as containing foreign language speech."
                    )
                )
            header = (
                f"📝 **{target_file_name}** 在 **{time_label}** 处的内容：\n\n"
                if lang == "zh"
                else f"📝 Content of **{target_file_name}** at **{time_label}**:\n\n"
            )
            video_visual_entries: List[Tuple[float, str]] = []
            if media_time_plan.has_video:
                yield {
                    "type": "status",
                    "phase": "running",
                    "message": MediaTimeSkill.stage_message(
                        "lookup_visual",
                        plan=media_time_plan,
                        time_label=time_label,
                    ),
                }
                yield {
                    "type": "thinking",
                    "delta": (
                        f"Collecting nearby visual frames for `{target_file_name}` around {time_label} "
                        "from indexed evidence to ground the timestamp answer.\n"
                    ),
                }
                video_visual_entries = _collect_indexed_point_visual_entries()
                if not video_visual_entries:
                    yield {
                        "type": "status",
                        "phase": "running",
                        "message": MediaTimeSkill.stage_message(
                            "extract_point_visual",
                            plan=media_time_plan,
                            time_label=time_label,
                        ),
                    }
                    yield {
                        "type": "thinking",
                        "delta": (
                            "No indexed keyframe was close enough to the requested timestamp; "
                            "extracting frames directly from the local video file.\n"
                        ),
                    }
                    video_visual_entries = _collect_realtime_point_visual_entries()

            if video_visual_entries:
                yield {"type": "text", "content": header}
                synth_prompt = MediaTimeSkill.build_point_audio_visual_prompt(
                    plan=media_time_plan,
                    time_label=time_label,
                    transcript_text=transcript_text,
                    visual_entries=video_visual_entries,
                    format_time=_fmt_time,
                )
                llm_synth = self._get_llm_service(
                    detailed=True,
                    session_id=session_id,
                    prompt_language=lang,
                )
                for _chunk in llm_synth.generate_stream(synth_prompt):
                    if self.is_aborted(session_id):
                        yield {"type": "done", "ok": False, "query_type": "interrupted", "message": "生成已中断"}
                        return
                    yield {"type": "text", "delta": _chunk}
                yield {"type": "text", "delta": "\n"}
            else:
                yield {"type": "text", "content": header + transcript_text + "\n"}

        else:
            # --- Fallback: If no transcript but it's a video, try visual extraction ---
            from core.media.media_expert import VIDEO_EXTENSIONS
            is_video = os.path.splitext(target_file_path)[1].lower() in VIDEO_EXTENSIONS

            if is_video:
                yield {"type": "text", "content":
                    ("未找到对应语音转录，正在查看已索引的画面证据...\n" if lang == "zh"
                     else "No matching transcript was found. Checking indexed visual evidence...\n")
                }
                # Set target_type to video_visual to allow the next block to execute
                target_type = "video_visual"
            else:
                if lang == "zh":
                    if has_foreign_language_notice:
                        msg = (
                            f"在 `{target_file_name}` 的 {time_label} 处未找到转录内容。"
                            "该文件已索引的摘要提示它可能包含外语语音。"
                        )
                    elif media_summary_excerpt:
                        msg = (
                            f"在 `{target_file_name}` 的 {time_label} 处未找到转录内容。"
                            f"已索引摘要提示：{media_summary_excerpt[:240]}"
                        )
                    else:
                        msg = (
                            f"在 `{target_file_name}` 的 {time_label} 处未找到转录内容。"
                            "可能该文件尚未进行 ASR 转录，或该时间段无语音。"
                        )
                else:
                    if has_foreign_language_notice:
                        msg = (
                            f"No transcript found at {time_label} in `{target_file_name}`. "
                            "The file is indexed as containing foreign-language speech."
                        )
                    elif media_summary_excerpt:
                        msg = (
                            f"No transcript found at {time_label} in `{target_file_name}`. "
                            f"The indexed media summary suggests: {media_summary_excerpt[:240]}"
                        )
                    else:
                        msg = (
                            f"No transcript found at {time_label} in `{target_file_name}`. "
                            "The file may not have been ASR-processed yet, or there may be no speech at that timestamp."
                        )
                yield {"type": "text", "content": msg}

    if target_type == "video_visual":
        # ── Video point lookup: indexed keyframes first, realtime frames if needed ──
        yield {
            "type": "status",
            "phase": "running",
            "message": MediaTimeSkill.stage_message(
                "lookup_visual",
                plan=media_time_plan,
                time_label=time_label,
            ),
        }
        yield {
            "type": "text",
            "content": (
                f"🎬 正在查看 `{target_file_name}` 在 **{time_label}** 附近的画面和音频证据...\n\n"
                if lang == "zh"
                else f"🎬 Checking visual and audio evidence around **{time_label}** in `{target_file_name}`...\n\n"
            ),
        }

        visual_entries = _collect_indexed_point_visual_entries()
        visual_source = "indexed"
        if not visual_entries:
            yield {
                "type": "status",
                "phase": "running",
                "message": MediaTimeSkill.stage_message(
                    "extract_point_visual",
                    plan=media_time_plan,
                    time_label=time_label,
                ),
            }
            yield {
                "type": "text",
                "content": (
                    "索引里没有足够接近这个时间点的画面，正在直接从原视频抽帧识别...\n\n"
                    if lang == "zh"
                    else "No indexed frame was close enough to that timestamp, so I am extracting frames directly from the local video...\n\n"
                ),
            }
            visual_entries = _collect_realtime_point_visual_entries()
            visual_source = "realtime" if visual_entries else "none"

        transcript_rows = _collect_point_transcript_rows()
        transcript_text = _format_transcript_rows(transcript_rows)
        if not transcript_text and media_time_plan.has_audio:
            logger.info("[media_timequery] no nearby indexed audio evidence for %s at %s", target_file_name, time_label)

        if visual_entries or transcript_text:
            yield {
                "type": "status",
                "phase": "running",
                "message": MediaTimeSkill.stage_message(
                    "generate_point_answer",
                    plan=media_time_plan,
                    time_label=time_label,
                ),
            }

        if visual_entries and transcript_text:
            header = (
                f"🖼️ **{target_file_name}** 在 **{time_label}** 附近的画面和音频：\n\n"
                if lang == "zh"
                else f"🖼️ Scene and audio in **{target_file_name}** around **{time_label}**:\n\n"
            )
            yield {"type": "text", "content": header}
            synth_prompt = MediaTimeSkill.build_point_audio_visual_prompt(
                plan=media_time_plan,
                time_label=time_label,
                transcript_text=transcript_text,
                visual_entries=visual_entries,
                format_time=_fmt_time,
            )
            llm_synth = self._get_llm_service(
                detailed=True,
                session_id=session_id,
                prompt_language=lang,
            )
            for _chunk in llm_synth.generate_stream(synth_prompt):
                if self.is_aborted(session_id):
                    yield {"type": "done", "ok": False, "query_type": "interrupted", "message": "生成已中断"}
                    return
                yield {"type": "text", "delta": _chunk}
            yield {"type": "text", "delta": "\n"}
        elif visual_entries:
            source_note = "实时抽帧" if visual_source == "realtime" else "来自索引"
            header = (
                f"🖼️ **{target_file_name}** 在 **{time_label}** 附近的画面（{source_note}）：\n\n"
                if lang == "zh"
                else (
                    f"🖼️ Scene in **{target_file_name}** around **{time_label}** "
                    f"({'direct frame extraction' if visual_source == 'realtime' else 'from index'}):\n\n"
                )
            )
            yield {"type": "text", "content": header}
            synth_prompt = MediaTimeSkill.build_point_visual_prompt(
                plan=media_time_plan,
                time_label=time_label,
                frame_descriptions=visual_entries,
                format_time=_fmt_time,
            )
            llm_synth = self._get_llm_service(
                detailed=True,
                session_id=session_id,
                prompt_language=lang,
            )
            for _chunk in llm_synth.generate_stream(synth_prompt):
                if self.is_aborted(session_id):
                    yield {"type": "done", "ok": False, "query_type": "interrupted", "message": "生成已中断"}
                    return
                yield {"type": "text", "delta": _chunk}
            yield {"type": "text", "delta": "\n"}
        elif transcript_text:
            header = (
                f"🎙️ 没能取得 `{target_file_name}` 在 **{time_label}** 附近的画面，但找到附近音频：\n\n"
                if lang == "zh"
                else f"🎙️ I could not get a visual frame around **{time_label}**, but found nearby audio in `{target_file_name}`:\n\n"
            )
            yield {"type": "text", "content": header + transcript_text + "\n"}
        else:
            msg = (
                f"没有在 `{target_file_name}` 的 {time_label} 附近取得可用画面或音频证据；已尝试索引画面和直接抽帧。"
                if lang == "zh"
                else (
                    f"I could not get usable visual or audio evidence around {time_label} in `{target_file_name}`; "
                    "I tried both indexed frames and direct frame extraction."
                )
            )
            yield {"type": "text", "content": msg + "\n"}

    # Emit source file info
    yield {
        "type": "sources",
        "content": source_payload,
        "total_matches": 1,
        "shown_count": 1,
    }
    yield {
        "type": "done",
        "ok": True,
        "query_type": "media_export",
        "sources": source_payload,
    }


def _handle_media_content_search(
    self,
    question: str,
    params: dict,
    *,
    session_id: Optional[str] = None,
    prompt_language: Optional[str] = None,
    active_paths: Optional[List[str]] = None,
):
    """
    Search indexed media content by topic.

    Audio-backed media is recalled from indexed ASR first, then refined to
    precise timestamped moments. Video results are enriched with the nearest
    indexed frame around each matched audio moment.
    """
    import os
    from collections import OrderedDict

    query_topic = str(params.get("query") or question).strip()
    media_type = str(params.get("media_type") or "all").strip().lower()
    file_hint = str(params.get("file_hint") or "").strip()
    lang = self._resolve_prompt_language(prompt_language, question=question, session_id=session_id)
    selected_file_hint = file_hint
    if not selected_file_hint:
        selected_candidates = [
            os.path.basename(str(raw_path or "").strip())
            for raw_path in list(active_paths or [])
            if str(raw_path or "").strip()
        ]
        if len(selected_candidates) == 1:
            selected_file_hint = selected_candidates[0]

    logger.debug(
        f"[media_content_search] topic_chars={len(query_topic or '')}, "
        f"media_type={media_type}, file_hint_chars={len(file_hint or '')}, lang={lang}"
    )

    yield {"type": "thinking", "delta": f"Mode: media_content_search, searching media content for: {query_topic}\n"}

    def _scoped_media_source_payload(limit: int = 3) -> List[Dict[str, Any]]:
        try:
            from core.media.media_expert import AUDIO_EXTENSIONS, MEDIA_EXTENSIONS, VIDEO_EXTENSIONS
        except Exception:
            AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".ape"}
            VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv", ".wmv", ".ts"}
            MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

        candidates: List[str] = []
        for raw_path in list(active_paths or []):
            fp = os.path.abspath(os.path.expanduser(str(raw_path or "").strip()))
            if fp:
                candidates.append(fp)

        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for fp in candidates:
            if not fp or fp in seen:
                continue
            seen.add(fp)
            ext = os.path.splitext(fp)[1].lower()
            if ext not in MEDIA_EXTENSIONS:
                continue
            if media_type == "audio" and ext not in AUDIO_EXTENSIONS:
                continue
            if media_type == "video" and ext not in VIDEO_EXTENSIONS:
                continue
            if file_hint:
                if _score_media_file_hint_match(
                    file_hint,
                    file_name=os.path.basename(fp),
                    file_path=fp,
                ) < 90:
                    continue
            try:
                indexed_rows = self.kb.collection.get(
                    where={"file_path": fp},
                    include=["metadatas"],
                )
                if not (indexed_rows or {}).get("metadatas"):
                    continue
            except Exception:
                continue
            rows.append(
                {
                    "file_path": fp,
                    "file_name": os.path.basename(fp),
                    "relevance_score": 1.0,
                }
            )
            if len(rows) >= limit:
                break
        return rows

    # ── Translate query for embedding model ────────────────────────────
    retrieval_query = query_topic
    if any("\u4e00" <= ch <= "\u9fff" for ch in retrieval_query):
        retrieval_query = self._augment_query_for_retrieval(
            query_topic, prompt_language=lang, session_id=session_id,
        )
    if selected_file_hint:
        hint_stem = os.path.splitext(selected_file_hint)[0]
        hint_bits = [bit for bit in [hint_stem, selected_file_hint] if bit]
        if hint_bits:
            retrieval_query = f"{retrieval_query} {' '.join(dict.fromkeys(hint_bits))}"
    logger.debug("[media_content_search] retrieval_query_chars=%s", len(retrieval_query or ""))

    kb = None
    query_embedding = None
    media_hits: List[Dict[str, Any]] = []
    try:
        from core.kb.knowledge_base import FileKnowledgeBase
        from core.retrieval.category_engine import get_compatible_categories
        kb: FileKnowledgeBase = self.kb

        with kb._embed_context("media_content_search"):
            query_embedding = kb._embed_doc_text(retrieval_query)

        media_categories = sorted(get_compatible_categories("audio/video"))
        def _run_media_query(where_filter: Dict[str, Any], *, n_results: int) -> List[Dict[str, Any]]:
            hits: List[Dict[str, Any]] = []
            results = kb.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
                where=where_filter,
            )
            if not (results and results.get("ids") and results["ids"][0]):
                return hits

            for idx, _doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][idx] if results.get("metadatas") else {}
                raw_text = results["documents"][0][idx] if results.get("documents") else ""
                distance = results["distances"][0][idx] if results.get("distances") else 1.0
                ctype = str(meta.get("chunk_type", ""))
                file_path = str(meta.get("file_path") or "")
                file_name = str(meta.get("file_name") or os.path.basename(file_path))
                if active_paths and file_path not in active_paths:
                    continue
                if file_hint and _score_media_file_hint_match(
                    file_hint,
                    file_name=file_name,
                    file_path=file_path,
                ) < 90:
                    continue

                text = (
                    _clean_media_chunk_text(raw_text, chunk_type=ctype)
                    if ctype in {"asr_transcript", "asr_segment", "media_summary", "media_audio_summary"}
                    else str(raw_text or "")
                )
                if ctype in {"asr_transcript", "asr_segment", "media_summary", "media_audio_summary"} and not text:
                    continue
                hits.append({
                    "file_path": file_path,
                    "file_name": file_name,
                    "chunk_type": ctype,
                    "keyframe_time_sec": meta.get("keyframe_time_sec"),
                    "keyframe_description": str(meta.get("keyframe_description") or ""),
                    "asr_start_sec": meta.get("asr_start_sec"),
                    "asr_end_sec": meta.get("asr_end_sec"),
                    "media_type": str(meta.get("media_type") or ""),
                    "has_asr_transcript": bool(
                        meta.get("has_asr_transcript")
                        or meta.get("media_has_asr_transcript")
                    ),
                    "text": text,
                    "distance": float(distance),
                })
            return hits

        def _audio_first_where(chunk_types: List[str]) -> Dict[str, Any]:
            clauses: List[Dict[str, Any]] = [
                {"chunk_type": {"$in": chunk_types}},
                {"doc_category": {"$in": media_categories}},
            ]
            if media_type in {"audio", "video"}:
                clauses.append({"media_type": media_type})
            return {"$and": clauses}

        primary_hits = _run_media_query(
            _audio_first_where(["asr_transcript", "media_audio_summary", "media_summary"]),
            n_results=80,
        )
        media_hits.extend(primary_hits)

        # Silent videos still need a visual-only fallback, but only after the
        # audio-first recall does not surface any speech-backed evidence.
        if media_type != "audio" and not any(h["chunk_type"] == "asr_transcript" for h in primary_hits):
            visual_fallback_hits = _run_media_query(
                _audio_first_where(["keyframe", "media_visual_summary", "media_summary"]),
                n_results=40,
            )
            seen = {
                (
                    hit["file_path"],
                    hit["chunk_type"],
                    hit.get("asr_start_sec"),
                    hit.get("asr_end_sec"),
                    hit.get("keyframe_time_sec"),
                )
                for hit in media_hits
            }
            for hit in visual_fallback_hits:
                dedupe_key = (
                    hit["file_path"],
                    hit["chunk_type"],
                    hit.get("asr_start_sec"),
                    hit.get("asr_end_sec"),
                    hit.get("keyframe_time_sec"),
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                media_hits.append(hit)

        logger.info(
            f"[media_content_search] {len(media_hits)} hits "
            f"(asr_window={sum(1 for h in media_hits if h['chunk_type']=='asr_transcript')}, "
            f"asr_segment={sum(1 for h in media_hits if h['chunk_type']=='asr_segment')}, "
            f"kf={sum(1 for h in media_hits if h['chunk_type']=='keyframe')}, "
            f"summary={sum(1 for h in media_hits if h['chunk_type']=='media_summary')}, "
            f"audio_summary={sum(1 for h in media_hits if h['chunk_type']=='media_audio_summary')}, "
            f"visual_summary={sum(1 for h in media_hits if h['chunk_type']=='media_visual_summary')})"
        )
    except Exception as e:
        logger.error(f"[media_content_search] query failed: {e}", exc_info=True)

    if not media_hits:
        scoped_sources = _scoped_media_source_payload(limit=1)
        msg = (
            f"未在已索引的视频/音频中找到与「{query_topic}」相关的内容。\n\n"
            "可能原因：相关文件尚未索引，或内容中未涉及该主题。"
            if lang == "zh"
            else f"No matching content found in indexed videos/audios for \"{query_topic}\".\n\n"
            "The relevant files may not be indexed yet, or the topic was not found in any transcript or visual content."
        )
        yield {"type": "text", "content": msg}
        if scoped_sources:
            yield {
                "type": "sources",
                "content": scoped_sources,
                "total_matches": len(scoped_sources),
                "shown_count": len(scoped_sources),
            }
        yield {"type": "done", "ok": True, "query_type": "media_content_search", "sources": scoped_sources}
        return

    # ── Group by file, sort each group by timestamp ────────────────────
    file_groups: dict = OrderedDict()
    for hit in media_hits:
        fp = hit["file_path"]
        if fp not in file_groups:
            file_groups[fp] = {"file_name": hit["file_name"], "file_path": fp, "hits": []}
        file_groups[fp]["hits"].append(hit)

    for fp in file_groups:
        file_groups[fp]["hits"].sort(key=_media_hit_time)

    file_groups = _apply_media_file_grounding(file_groups, query_topic)

    # ── LLM Filtering for Media Hits ────────────────────────────────────
    if file_groups:
        filter_prompt_blocks = []
        file_keys = list(file_groups.keys())
        for idx, fp in enumerate(file_keys, 1):
            group = file_groups[fp]
            summary_texts = []
            for hit in (group.get("filter_hits") or group["hits"])[:4]:
                t = hit.get("text") or hit.get("keyframe_description") or ""
                if len(t) > 120: t = t[:117] + "..."
                summary_texts.append(t)
            combined_snips = " | ".join(summary_texts)
            filter_prompt_blocks.append(f"[{idx}] {group['file_name']}: {combined_snips}")

        filter_prompt = (
            f"You are a relevance filter. Evaluate the following media files against the USER NEED: '{query_topic}'.\n"
            f"Filter out irrelevant files and keep only the relevant ones.\n\n"
            f"RULES:\n"
            f"1. If a candidate's file name or content snippet is relevant to the user need, include it.\n"
            f"2. If no candidates are relevant, return exactly: {{\"selected_indices\": [0]}}\n"
            f"3. Output ONLY a valid JSON object with key `selected_indices` (array of 1-based indices, or [0] if none match).\n"
            f"4. DO NOT output any thinking process, <think> tags, or explanation. Output ONLY the JSON.\n\n"
            f"Candidates:\n"
            + "[0] None of the above candidates are relevant.\n"
            + "\n".join(filter_prompt_blocks)
        )

        try:
            filter_llm = self._get_llm_service(detailed=False, session_id=session_id, prompt_language=lang)
            if hasattr(filter_llm, "force_text_model"):
                filter_llm.force_text_model = True

            raw_filter = str(filter_llm.generate(filter_prompt) or "").strip()
            import json, re
            match = re.search(r'\{.*?\}', raw_filter, re.DOTALL)
            parsed_json = json.loads(match.group(0)) if match else json.loads(raw_filter)
            selected_indices = parsed_json.get("selected_indices", [])

            if isinstance(selected_indices, list) and len(selected_indices) > 0:
                if selected_indices == [0] or (len(selected_indices) == 1 and str(selected_indices[0]) == "0"):
                    file_groups = OrderedDict()
                else:
                    new_file_groups = OrderedDict()
                    for i in selected_indices:
                        try:
                            i_int = int(i)
                            if 1 <= i_int <= len(file_keys):
                                fp = file_keys[i_int - 1]
                                new_file_groups[fp] = file_groups[fp]
                        except Exception:
                            pass
                    if new_file_groups:
                        file_groups = new_file_groups
        except Exception as e:
            logger.warning(f"[media_content_search] LLM filtering failed: {e}")

    if not file_groups:
        scoped_sources = _scoped_media_source_payload(limit=1)
        msg = (
            f"未在已索引的视频/音频中找到与「{query_topic}」相关的内容。\n\n"
            "可能原因：经过内容筛选后，所有候选文件均与该主题无关。"
            if lang == "zh"
            else f"No matching content found in indexed videos/audios for \"{query_topic}\".\n\n"
            "After content filtering, all candidates were deemed irrelevant."
        )
        yield {"type": "text", "content": msg}
        if scoped_sources:
            yield {
                "type": "sources",
                "content": scoped_sources,
                "total_matches": len(scoped_sources),
                "shown_count": len(scoped_sources),
            }
        yield {"type": "done", "ok": True, "query_type": "media_content_search", "sources": scoped_sources}
        return

    keyframe_cache: Dict[str, List[Dict[str, Any]]] = {}
    def _load_keyframe_hits(file_path: str) -> List[Dict[str, Any]]:
        if file_path in keyframe_cache:
            return keyframe_cache[file_path]
        keyframe_cache[file_path] = []
        try:
            kf_results = kb.collection.get(
                where={
                    "$and": [
                        {"file_path": file_path},
                        {"chunk_type": "keyframe"},
                    ]
                },
                include=["metadatas", "documents"],
            )
            for meta, doc_text in zip(
                list(kf_results.get("metadatas") or []),
                list(kf_results.get("documents") or []),
            ):
                if meta.get("keyframe_time_sec") is None:
                    continue
                keyframe_cache[file_path].append({
                    "time_sec": float(meta.get("keyframe_time_sec") or 0.0),
                    "text": str(meta.get("keyframe_description") or doc_text or "").strip(),
                })
        except Exception as exc:
            logger.debug("[media_content_search] keyframe cache load failed for %s: %s", os.path.basename(file_path), exc)
        keyframe_cache[file_path].sort(key=lambda item: float(item.get("time_sec") or 0.0))
        return keyframe_cache[file_path]

    def _lookup_visual_context(file_path: str, anchor_sec: float) -> Optional[Dict[str, Any]]:
        indexed_keyframes = _load_keyframe_hits(file_path)
        if indexed_keyframes:
            best = min(indexed_keyframes, key=lambda item: abs(float(item.get("time_sec") or 0.0) - float(anchor_sec)))
            if abs(float(best.get("time_sec") or 0.0) - float(anchor_sec)) <= 10.0 and str(best.get("text") or "").strip():
                return {
                    "time_sec": float(best.get("time_sec") or anchor_sec),
                    "text": str(best.get("text") or "").strip(),
                    "source": "indexed_keyframe",
                }

        logger.info("[media_content_search] no indexed keyframe near %.1fs; DB-only visual lookup", anchor_sec)
        return None

    def _query_precise_audio_hits(file_path: str) -> List[Dict[str, Any]]:
        precise_hits = _run_media_query(
            {
                "$and": [
                    {"file_path": file_path},
                    {"chunk_type": "asr_segment"},
                ]
            },
            n_results=8,
        )
        if precise_hits:
            return precise_hits
        return _run_media_query(
            {
                "$and": [
                    {"file_path": file_path},
                    {"chunk_type": "asr_transcript"},
                ]
            },
            n_results=4,
        )

    def _query_visual_hits(file_path: str) -> List[Dict[str, Any]]:
        return _run_media_query(
            {
                "$and": [
                    {"file_path": file_path},
                    {"chunk_type": "keyframe"},
                ]
            },
            n_results=4,
        )

    for fp, group in file_groups.items():
        group_hits = list(group.get("hits") or [])
        file_media_type = str(group_hits[0].get("media_type") or "")
        has_audio_evidence = any(
            h.get("chunk_type") == "asr_transcript" or h.get("has_asr_transcript")
            for h in group_hits
        )
        moments: List[Dict[str, Any]] = []

        if has_audio_evidence:
            precise_audio_hits = _query_precise_audio_hits(fp)
            precise_audio_hits = _merge_adjacent_asr_matches(precise_audio_hits, max_gap_sec=2.0)
            for hit in precise_audio_hits[:3]:
                start, end = _media_hit_time_range(hit)
                if start is None:
                    continue
                moment = {
                    "start_sec": float(start),
                    "end_sec": float(end if end is not None else start),
                    "speech_text": str(hit.get("text") or "").strip(),
                    "visual_time_sec": None,
                    "visual_text": "",
                    "distance": float(hit.get("distance") or 1.0),
                }
                if file_media_type == "video":
                    visual = _lookup_visual_context(fp, (moment["start_sec"] + moment["end_sec"]) / 2.0)
                    if visual:
                        moment["visual_time_sec"] = float(visual.get("time_sec") or moment["start_sec"])
                        moment["visual_text"] = str(visual.get("text") or "").strip()
                moments.append(moment)

        if not moments and file_media_type == "video":
            for hit in _query_visual_hits(fp)[:3]:
                visual_ts = hit.get("keyframe_time_sec")
                visual_desc = str(hit.get("keyframe_description") or hit.get("text") or "").strip()
                if visual_ts is None or not visual_desc:
                    continue
                moments.append({
                    "start_sec": float(visual_ts),
                    "end_sec": float(visual_ts),
                    "speech_text": "",
                    "visual_time_sec": float(visual_ts),
                    "visual_text": visual_desc,
                    "distance": float(hit.get("distance") or 1.0),
                })

        group["moments"] = moments

    # ── Emit file list as sources ──────────────────────────────────────
    source_files = []
    for fp, group in file_groups.items():
        moments = list(group.get("moments") or [])
        n_asr = sum(1 for m in moments if m.get("speech_text"))
        n_kf = sum(1 for m in moments if m.get("visual_text"))
        if not moments:
            n_asr = sum(1 for h in group["hits"] if h["chunk_type"] == "asr_transcript")
            n_kf = sum(1 for h in group["hits"] if h["chunk_type"] == "keyframe")
        summary_parts = []
        if n_kf:
            summary_parts.append(f"{n_kf} matching frame{'s' if n_kf > 1 else ''}")
        if n_asr:
            summary_parts.append(f"{n_asr} audio moment{'s' if n_asr > 1 else ''}")
        source_files.append({
            "file_name": group["file_name"],
            "file_path": fp,
            "doc_category": str(group["hits"][0].get("media_type") or "audio/video"),
            "doc_summary": ", ".join(summary_parts) or "matched",
        })
    try:
        if session_id and source_files and hasattr(self, "_set_last_search_results"):
            self._set_last_search_results(session_id, source_files[:50])
        if session_id and source_files and hasattr(self, "_set_followup_hint"):
            self._set_followup_hint(
                session_id,
                action="process_previous",
                params={},
                ttl_turns=2,
                uses=2,
            )
        if session_id and source_files and hasattr(self, "_clear_count_scope_context"):
            self._clear_count_scope_context(session_id, reason="media_content_search_results_updated")
    except Exception as exc:
        logger.warning("[media_content_search] failed to remember result context: %s", exc)
    yield {
        "type": "sources",
        "content": source_files,
        "total_matches": len(source_files),
        "shown_count": len(source_files),
    }

    # ── Build structured prompt for LLM synthesis ──────────────────────
    def _fmt_ts(sec) -> str:
        if sec is None:
            return "?s"
        s = int(float(sec))
        m, s = divmod(s, 60)
        return f"{m}:{s:02d}" if m else f"{s}s"

    def _fmt_ts_range(s0, s1) -> str:
        if s0 is None:
            return "?s"
        if s1 is None:
            return _fmt_ts(s0)
        if abs(float(s1) - float(s0)) < 0.2:
            return _fmt_ts(s0)
        return f"{_fmt_ts(s0)}-{_fmt_ts(s1)}"

    snippet_lines = []
    for fp, group in file_groups.items():
        snippet_lines.append(f"### {group['file_name']}")
        moments = list(group.get("moments") or [])
        if moments:
            for moment in moments[:4]:
                ts = _fmt_ts_range(moment.get("start_sec"), moment.get("end_sec"))
                speech_text = str(moment.get("speech_text") or "").strip()
                visual_text = str(moment.get("visual_text") or "").strip()
                if speech_text:
                    if len(speech_text) > 320:
                        speech_text = speech_text[:317] + "..."
                    snippet_lines.append(f"  🎙️ [{ts}] {speech_text}")
                if visual_text:
                    visual_ts = _fmt_ts(moment.get("visual_time_sec") or moment.get("start_sec"))
                    if len(visual_text) > 280:
                        visual_text = visual_text[:277] + "..."
                    snippet_lines.append(f"  🖼️ [{visual_ts}] {visual_text}")
        else:
            for hit in group["hits"][:10]:
                ctype = hit["chunk_type"]
                if ctype in {"asr_transcript", "asr_segment"}:
                    ts = _fmt_ts_range(hit.get("asr_start_sec"), hit.get("asr_end_sec"))
                    clean = str(hit.get("text") or "").strip()
                    if len(clean) > 300:
                        clean = clean[:297] + "..."
                    snippet_lines.append(f"  🎙️ [{ts}] {clean}")
                elif ctype == "keyframe":
                    ts = _fmt_ts(hit["keyframe_time_sec"])
                    desc = hit.get("keyframe_description") or hit["text"]
                    if len(desc) > 300:
                        desc = desc[:297] + "..."
                    snippet_lines.append(f"  🖼️ [{ts}] {desc}")
                else:
                    text = str(hit.get("text") or "")
                    if len(text) > 200:
                        text = text[:197] + "..."
                    snippet_lines.append(f"  📄 {text}")
        snippet_lines.append("")

    snippets_block = "\n".join(snippet_lines)

    if lang == "zh":
        synth_prompt = (
            f"用户问：「{question}」\n\n"
            f"以下是在已索引的音视频中找到的与「{query_topic}」相关的内容：\n"
            f"（🎙️ = 先由音频召回、再定位到精确时间段的语音片段；🖼️ = 对应时间点的视频画面；📄 = 文件摘要）\n\n"
            f"{snippets_block}\n\n"
            "【指令】请用中文回答。\n"
            "- 优先根据语音内容说明该主题出现在哪些时间段，以及当时具体在讲什么。\n"
            "- 如果是视频且提供了画面描述，请补充这些时间点画面里正在展示什么，并明确写出最相关的时间点（例如 5s 附近）。\n"
            "- 时间点请至少保留一次精确写法，例如 `5s`、`15s` 或 `0:05`，不要只写成模糊说法。\n"
            "- 区分语音提到的内容和画面显示的内容，但不要重复啰嗦。\n"
            "- 用自然流畅的叙述（不要机械罗列时间戳），按阶段归纳。\n"
            "- 最后给出整体总结：一共有几个文件涉及，主要内容是什么。\n"
            "- 引用文件时使用 Markdown 链接格式并附上编号（例如 [文件名](1)）。"
        )
    else:
        synth_prompt = (
            f'User asked: "{question}"\n\n'
            f'Matches found across indexed audio/video files for "{query_topic}":\n'
            f"(🎙️ = audio-first matched speech moment, 🖼️ = video frame around that moment, 📄 = file summary)\n\n"
            f"{snippets_block}\n\n"
            "[Instruction] Answer in English.\n"
            "- Lead with the speech-based timestamps and what is being discussed there.\n"
            "- For videos, add what is visible on screen around the same matched moments when visual evidence is provided, and explicitly name the strongest timestamp(s), such as `5s`, `15s`, or `0:05`.\n"
            "- Distinguish spoken content from visual content when both are present.\n"
            "- Use natural flowing narrative (not mechanical timestamp lists), group by phases.\n"
            "- End with an overall summary: how many files match and what the main content is.\n"
            "- When citing files, use markdown link format with index (e.g. [Filename](1))."
        )

    # ── Stream LLM answer ──────────────────────────────────────────────
    _prev_max = os.environ.get("FILEAGENT_MAX_OUTPUT_TOKENS")
    os.environ["FILEAGENT_MAX_OUTPUT_TOKENS"] = "3200"
    try:
        llm = self._get_llm_service(detailed=True, session_id=session_id, prompt_language=lang)
        for chunk in llm.generate_stream(synth_prompt):
            if self.is_aborted(session_id):
                yield {"type": "done", "ok": False, "query_type": "interrupted", "message": "生成已中断"}
                return
            yield {"type": "text", "delta": chunk}
    finally:
        if _prev_max is None:
            os.environ.pop("FILEAGENT_MAX_OUTPUT_TOKENS", None)
        else:
            os.environ["FILEAGENT_MAX_OUTPUT_TOKENS"] = _prev_max

    yield {"type": "text", "delta": "\n"}
    yield {"type": "done", "ok": True, "query_type": "media_content_search"}
