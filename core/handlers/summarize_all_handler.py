"""
summarize_all handler — extracted from FileAgent._handle_summarize_all.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
logger = get_logger()

from core.llm.builder import get_llm
from langchain_core.messages import HumanMessage
from tools.document_tools import get_kb_instance


_LOW_SIGNAL_SUMMARY_PATTERNS = (
    "blank audio",
    "only blank audio",
    "silent audio",
    "empty audio",
    "empty transcript",
    "no speech",
    "no spoken",
    "no dialogue",
    "without speech",
    "without dialogue",
    "untranslatable speech",
    "non-translatable speech",
    "cannot be translated",
    "unable to translate",
    "placeholder",
    "text extraction unavailable",
    "possibly scanned/protected",
    "contains only music",
    "only music",
    "music only",
    "contains only singing",
    "only singing",
    "contains only music and singing",
    "only music and singing",
    "background music only",
    "instrumental music only",
    "静音",
    "无语音",
    "没有语音",
    "空白音频",
    "未检测到语音",
    "无可翻译",
    "无法翻译",
    "不可翻译",
    "占位符",
    "仅包含音乐",
    "只有音乐",
    "仅有音乐",
    "仅包含歌声",
    "只有歌声",
    "仅有歌声",
    "仅包含音乐和歌声",
    "只有音乐和歌声",
)


def _is_low_signal_summary_text(text: str) -> bool:
    tl = str(text or "").strip().lower()
    if not tl:
        return True
    return any(pat in tl for pat in _LOW_SIGNAL_SUMMARY_PATTERNS)


def _clean_summary_for_global_prompt(summary: str) -> str:
    raw = str(summary or "").strip()
    if not raw:
        return ""

    kept_lines: List[str] = []
    for line in raw.splitlines():
        line = str(line or "").strip()
        if not line:
            continue
        if _is_low_signal_summary_text(line):
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines).strip()
    if not cleaned or _is_low_signal_summary_text(cleaned):
        return ""
    return cleaned


def _build_large_scope_summary_context(
    self,
    *,
    files: List[Dict[str, Any]],
    lang: str,
) -> Tuple[str, Dict[str, Any]]:
    from collections import defaultdict

    def _safe_int_env(name: str, default: str) -> int:
        try:
            return max(1, int(os.getenv(name, default)))
        except Exception:
            return max(1, int(default))

    category_counter: Dict[str, int] = defaultdict(int)
    grouped_files: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in list(files or []):
        cat = self._normalize_category_name(f.get("doc_category", "other"))
        category_counter[cat] += 1
        grouped_files[cat].append(f)

    sorted_categories = sorted(category_counter.items(), key=lambda item: (-int(item[1]), str(item[0])))
    top_cats_limit = _safe_int_env("SUMMARIZE_ALL_TOP_CATEGORIES", "12")
    tail_cats_limit = _safe_int_env("SUMMARIZE_ALL_TAIL_CATEGORIES", "4")
    sample_limit_per_cat = _safe_int_env("SUMMARIZE_ALL_SAMPLES_PER_CATEGORY", "6")
    summary_chars = _safe_int_env("SUMMARIZE_ALL_SAMPLE_SUMMARY_CHARS", "320")

    top_categories = list(sorted_categories[:top_cats_limit])
    tail_candidates = list(sorted_categories[top_cats_limit:])
    tail_categories: List[Tuple[str, int]] = []
    if tail_candidates:
        seen = {cat for cat, _ in top_categories}
        for cat, cnt in reversed(tail_candidates):
            if cat in seen:
                continue
            tail_categories.append((cat, cnt))
            seen.add(cat)
            if len(tail_categories) >= tail_cats_limit:
                break
        tail_categories.reverse()

    selected_categories = top_categories + [item for item in tail_categories if item not in top_categories]

    category_lines: List[str] = []
    sample_lines: List[str] = []
    for cat, cnt in selected_categories:
        if lang == "zh":
            category_lines.append(f"- {cat}: {cnt} 份")
        else:
            category_lines.append(f"- {cat}: {cnt} file(s)")
        sample_lines.append(f"[{cat}]")
        for f in list(grouped_files.get(cat) or [])[:sample_limit_per_cat]:
            name = str(f.get("file_name") or "").strip()
            summary = _clean_summary_for_global_prompt(str(f.get("doc_summary") or "").strip())
            if summary:
                sample_lines.append(f"- {name}: {summary[:summary_chars]}")
        remain = max(0, len(list(grouped_files.get(cat) or [])) - sample_limit_per_cat)
        if remain > 0:
            sample_lines.append(
                f"... 以及另外 {remain} 份同类文件"
                if lang == "zh"
                else f"... plus {remain} more file(s) in this category"
            )
        sample_lines.append("")

    all_category_count = len(sorted_categories)
    context_text = (
        f"你正在总结当前选中范围内的全部 {len(files)} 份文件。\n"
        f"共识别出 {all_category_count} 个类别。\n\n"
        if lang == "zh"
        else f"You are summarizing all {len(files)} selected files in scope.\n"
             f"There are {all_category_count} detected categories.\n\n"
    )
    context_text += (
        "【类别分布】\n" if lang == "zh" else "[Category distribution]\n"
    )
    context_text += "\n".join(category_lines) if category_lines else ("- other: unknown" if lang == "en" else "- other: unknown")
    context_text += "\n\n" + ("【类别代表样本】\n" if lang == "zh" else "[Representative samples by category]\n")
    context_text += "\n".join(sample_lines) if sample_lines else ("(no representative samples)" if lang == "en" else "(无代表样本)")

    meta = {
        "total_files": len(files),
        "total_categories": all_category_count,
        "selected_categories": selected_categories,
        "top_categories": top_categories,
        "tail_categories": tail_categories,
        "sample_limit_per_cat": sample_limit_per_cat,
    }
    return context_text, meta

def _handle_summarize_all(
    self,
    question: str,
    params: dict,
    active_paths: Optional[List[str]],
    session_id: Optional[str],
    emit_status: bool,
    prompt_language: Optional[str] = None,
):
    lang = self._resolve_prompt_language(prompt_language, question=question, session_id=session_id)
    if emit_status:
        msg = "正在整理所有文件摘要…" if lang == "zh" else "Preparing a global summary of all files..."
        yield {"type": "status", "phase": "thinking", "message": msg}

    kb = get_kb_instance()

    file_extensions = []
    raw_exts = (params or {}).get("file_extensions", "")
    if raw_exts:
        for e in raw_exts.split(","):
            e = e.strip().lower()
            if not e: continue
            if not e.startswith("."): e = "." + e
            file_extensions.append(e)
    # Also fall back to question-based detection
    preserve_selected_scope = bool((params or {}).get("_preserve_selected_scope"))
    if not file_extensions and not preserve_selected_scope:
        file_extensions = self._extract_ext_filters_simple(question)

    if emit_status:
        if file_extensions:
            ext_label = "/".join(e.lstrip(".").upper() for e in file_extensions)
            msg = f"正在整理所有 {ext_label} 文件摘要…" if lang == "zh" else f"Preparing summary of all {ext_label} files..."
        else:
            msg = "正在整理所有文件摘要…" if lang == "zh" else "Preparing a global summary of all files..."
        yield {"type": "status", "phase": "thinking", "message": msg}

    category_stats = kb.count_by_category(allowed_paths=active_paths, file_extensions=file_extensions or None)
    files = category_stats.get("files", [])

    if not files:
        yield {"type": "files", "total": 0, "preview": [], "all": []}
        if file_extensions:
            ext_label = "/".join(e.lstrip(".").upper() for e in file_extensions)
            empty_msg = (f"当前数据源中没有找到任何 {ext_label} 文件。" if lang == "zh"
                         else f"No {ext_label} files were found in the currently selected sources.")
        else:
            empty_msg = "当前选中的数据源中没有找到任何文件。" if lang == "zh" else "No files were found in the currently selected sources."
        yield {"type": "text", "delta": empty_msg}
        yield {"type": "done", "ok": True, "query_type": "summarize_all", "sources": [], "trace": []}
        return

    if len(files) == 1:
        progress_msg = (
            "找到 1 个选中文件，正在生成更详细的单文件总结...\n"
            if lang == "zh"
            else "Found 1 selected file. Generating a detailed single-file summary...\n"
        )
    else:
        progress_msg = (
            f"找到 {len(files)} 个文件，正在生成整体总结...\n"
            if lang == "zh"
            else f"Found {len(files)} files. Generating global summary...\n"
        )
    yield {"type": "thinking", "delta": progress_msg}
    yield {"type": "sources", "content": files}
    # Persist files into session so subsequent process_previous turns can reference them.
    self._set_last_search_results(session_id, files)

    # When there are only a few files, do a detailed per-file summary
    # and skip the category-breakdown — that only adds value when there
    # are many files spread across multiple categories.
    _FEW_FILES_THRESHOLD = 5
    few_files_mode = len(files) <= _FEW_FILES_THRESHOLD

    focus = str((params or {}).get("focus") or "").strip()

    _MEDIA_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".ts",
                   ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}

    def _get_media_full_context(file_path: str) -> str:
        """Fetch all keyframe/transcript chunks and format as a coherent narrative."""
        try:
            all_chunks = kb.collection.get(
                where={"file_path": file_path},
                include=["documents", "metadatas"],
            )
            meta_lines = []
            kf_entries = []   # (time_sec, description)
            asr_lines = []
            for chunk_doc, chunk_meta in zip(
                all_chunks.get("documents") or [],
                all_chunks.get("metadatas") or [],
            ):
                ctype = chunk_meta.get("chunk_type", "")
                chunk_doc = str(chunk_doc or "").strip()
                if not chunk_doc:
                    continue
                if ctype == "media_summary":
                    for line in chunk_doc.splitlines():
                        line = str(line or "").strip()
                        if not line:
                            continue
                        if _is_low_signal_summary_text(line):
                            continue
                        if not line.startswith("内容摘要:"):
                            meta_lines.append(line)
                elif ctype == "media_audio_summary":
                    for line in chunk_doc.splitlines():
                        line = str(line or "").strip()
                        if not line or _is_low_signal_summary_text(line):
                            continue
                        if line.startswith("音频摘要:"):
                            asr_lines.append(line[len("音频摘要:"):].strip())
                        elif line.lower().startswith("audio summary:"):
                            asr_lines.append(line.split(":", 1)[-1].strip())
                elif ctype == "media_visual_summary":
                    for line in chunk_doc.splitlines():
                        line = str(line or "").strip()
                        if not line:
                            continue
                        if line.startswith("画面摘要:"):
                            kf_entries.append((-1.0, line[len("画面摘要:"):].strip()))
                        elif line.lower().startswith("visual summary:"):
                            kf_entries.append((-1.0, line.split(":", 1)[-1].strip()))
                elif ctype == "keyframe":
                    t = chunk_meta.get("keyframe_time_sec", 0)
                    kf_entries.append((float(t), chunk_doc))
                elif ctype == "asr_transcript":
                    if not _is_low_signal_summary_text(chunk_doc):
                        asr_lines.append(chunk_doc)

            # Sort keyframes by time
            kf_entries.sort(key=lambda x: x[0])

            parts = []
            if meta_lines:
                label = "【视频信息】" if lang == "zh" else "[Media info] "
                parts.append(label + " | ".join(l for l in meta_lines if l.strip()))

            if kf_entries:
                kf_block = "【画面内容（按时间顺序）】\n" if lang == "zh" else "[Visual timeline]\n"
                kf_lines = []
                for t, desc in kf_entries:
                    if float(t) < 0:
                        kf_lines.append(("【画面摘要】" if lang == "zh" else "[visual summary] ") + desc)
                    else:
                        kf_lines.append(f"[{t:.0f}s] {desc}")
                kf_block += "\n\n".join(kf_lines)
                parts.append(kf_block)

            if asr_lines:
                label = "【语音内容】\n" if lang == "zh" else "[Audio/speech]\n"
                parts.append(label + "\n".join(asr_lines))

            result = "\n\n".join(parts)
            logger.info(
                f"[summarize_all] media context for {os.path.basename(file_path)}: "
                f"{len(result)} chars, {len(kf_entries)} keyframes, {len(asr_lines)} asr chunks"
            )
            return result
        except Exception as e:
            logger.warning(f"[summarize_all] _get_media_full_context failed: {e}")
            return ""

    single_file = files[0] if len(files) == 1 else None
    single_name = str((single_file or {}).get("file_name") or "").strip()
    single_path = str((single_file or {}).get("file_path") or "").strip()
    single_ext = os.path.splitext(single_name)[1].lower()
    single_media_mode = bool(single_file and single_ext in _MEDIA_EXTS and single_path)
    if single_media_mode and session_id and hasattr(self, "_set_followup_hint"):
        try:
            self._set_followup_hint(
                session_id,
                action="process_previous",
                params={
                    "_skill_name": "contextual_refine",
                    "scope": "last_results",
                    "operation": "qa",
                    "_single_media_summary": True,
                },
                ttl_turns=3,
                uses=3,
            )
        except Exception as exc:
            logger.warning("[summarize_all] failed to set single-media follow-up hint: %s", exc)

    if single_media_mode:
        media_context = _get_media_full_context(single_path) or str(single_file.get("doc_summary") or "").strip()
        media_context = _clean_summary_for_global_prompt(media_context)
        if not media_context:
            media_context = f"File name: {single_name}" if lang == "en" else f"文件名：{single_name}"

        if lang == "zh":
            prompt = (
                f"用户正在询问一个已选中的媒体文件，不是在询问文件集合。\n"
                f"文件名：{single_name}\n"
                f"文件路径：{single_path}\n\n"
                f"可用索引信息：\n{media_context}\n\n"
                "请生成一份信息更充实的单文件媒体解读。\n"
                "要求：\n"
                "1. 直接以文件名开头，不要写“媒体文件主题分布”“1 个文件的主题分布”这类集合总结话术。\n"
                "2. 先用 2-3 句话概括这个视频/音频大概是什么；如果文件名暗示主题，可以作为“文件名线索”提及，但要和画面证据区分开。\n"
                "3. 单独写“画面内容”，按时间或关键帧说明看到了什么，至少覆盖主要可见场景、物体、环境、变化。\n"
                "4. 单独写“音频/语音”，只根据已索引语音内容说明；如果没有有效语音，不要把它当核心结论，简短说明即可。\n"
                "5. 不要添加“可追问问题”“后续问题”“你还可以问”之类的结尾；如果证据不足，直接说明不能可靠判断。\n"
                "6. 使用 Markdown，小标题清晰；答案要比主题列表更丰富，但不要编造索引里没有的内容。\n"
                "7. [重要] 不要将总结结果包裹在 Markdown 代码块中。"
            )
        else:
            prompt = (
                "The user is asking about one selected media file, not a collection of files.\n"
                f"File name: {single_name}\n"
                f"File path: {single_path}\n\n"
                f"Available indexed evidence:\n{media_context}\n\n"
                "Generate a richer single-file media explanation.\n"
                "Requirements:\n"
                "1. Start with the file name/title. Do NOT write a corpus/category answer such as "
                "\"topic distribution\", \"media files (1 files)\", or a bare list of topics.\n"
                "2. Give a 2-3 sentence overview of what this media appears to contain. If the filename suggests a topic, "
                "you may mention it as a filename clue, but clearly distinguish it from visible/audio evidence.\n"
                "3. Include a dedicated \"Visual content\" section that explains the key frames or timeline in concrete detail: "
                "visible scene, objects, environment, motion/change, and notable visual clues.\n"
                "4. Include a dedicated \"Audio/speech\" section based only on indexed audio evidence. If there is no meaningful speech transcript, "
                "state that briefly and keep the focus on visuals.\n"
                "5. Do NOT add a \"follow-up questions\", \"questions you could ask\", or \"next steps\" section. If evidence is insufficient, say what cannot be reliably determined.\n"
                "6. Use clear Markdown and make the answer substantially richer than a topic list, while staying grounded in the evidence.\n"
                "7. [IMPORTANT] Do NOT wrap the answer in a Markdown code block.\n"
                "8. [IMPORTANT] You MUST answer in English."
            )
    elif few_files_mode:
        context_text = (
            f"你是文件总结助手。用户选中了以下 {len(files)} 份文件：\n\n"
            if lang == "zh"
            else f"You are a file summarization assistant. The user has selected the following {len(files)} file(s):\n\n"
        )
        for f in files:
            name = f.get("file_name", "")
            file_path = f.get("file_path", "")
            ext = os.path.splitext(name)[1].lower()
            if ext in _MEDIA_EXTS and file_path:
                summary = _get_media_full_context(file_path) or f.get("doc_summary", "")
            else:
                summary = f.get("doc_summary", "")
            summary = _clean_summary_for_global_prompt(summary)
            if summary:
                context_text += f"- 《{name}》：{summary}\n" if lang == "zh" else f'- "{name}": {summary}\n'
            else:
                context_text += f"- 《{name}》\n" if lang == "zh" else f'- "{name}"\n'
        context_text += "\n"

        focus_instruction = (
            (f"用户特别关注的重点是：\"{focus}\"，请在总结时着重关注这方面的信息。"
             if lang == "zh"
             else f'The user focus is "{focus}". Emphasize this in the summary.')
            if focus
            else ("请对每份文件的核心内容做详细总结，重点突出关键信息、主要观点和实际价值。"
                  if lang == "zh"
                  else "Provide a detailed summary of each file's core content, highlighting key information, main points, and practical value.")
        )
        if lang == "zh":
            prompt = (
                f"{context_text}"
                "请根据以上文件内容，为用户生成详细的文件总结。\n"
                "要求：\n"
                "1. 对每份文件的核心内容进行详细介绍，突出关键信息和主要价值。\n"
                f"2. {focus_instruction}\n"
                "3. 语气专业、清晰，使用 markdown 格式。\n"
                "4. [重要] 不要将总结结果包裹在 Markdown 代码块（```markdown ... ```）中。\n"
                "\n【重要指令】请务必使用中文进行概括。即便文件摘要中包含英文，回复内容也必须保持纯中文。"
            )
        else:
            prompt = (
                f"{context_text}"
                "Generate a detailed summary of the file(s) above.\n"
                "Requirements:\n"
                "1. Provide a thorough summary of each file's core content, highlighting key information and value.\n"
                f"2. {focus_instruction}\n"
                "3. Professional and clear markdown output.\n"
                "4. [IMPORTANT] Do NOT wrap the summary in a Markdown code block (```markdown ... ```)."
                "\n[IMPORTANT] You MUST generate the entire summary in English. Even if the input summaries are in Chinese, your final response must be purely in English."
            )
    else:
        enriched_files: List[Dict[str, Any]] = []
        large_scope_media_limit = max(1, int(os.getenv("SUMMARIZE_ALL_MEDIA_SAMPLE_LIMIT", "3")))
        media_enriched = 0
        for f in list(files or []):
            copied = dict(f)
            name = copied.get("file_name", "")
            file_path = copied.get("file_path", "")
            ext = os.path.splitext(name)[1].lower()
            if ext in _MEDIA_EXTS and file_path and media_enriched < large_scope_media_limit:
                media_ctx = _get_media_full_context(file_path)
                if media_ctx:
                    copied["doc_summary"] = _clean_summary_for_global_prompt(media_ctx[:1800])
                    media_enriched += 1
            enriched_files.append(copied)

        context_text, large_scope_meta = _build_large_scope_summary_context(
            self,
            files=enriched_files,
            lang=lang,
        )

        focus_instruction = (
            (f"用户特别关注的重点是：\"{focus}\"，请在总结时着重关注这方面的信息。"
             if lang == "zh"
             else f'The user focus is "{focus}". Emphasize this in the summary.')
            if focus
            else ("用户未指定单一侧重点；请覆盖主要类别，并指出各类文件的主题分布、共性和明显差异。"
                  if lang == "zh"
                  else "No single focus is specified. Cover the major categories and explain their themes, common patterns, and notable differences.")
        )

        selected_category_count = len(list(large_scope_meta.get("selected_categories") or []))
        if lang == "zh":
            prompt = (
                f"{context_text}\n\n"
                "请根据以上“全量类别统计 + 每类代表样本”，为用户生成一份质量更高、信息更充实的整体总结。\n"
                "要求：\n"
                "1. 先用 2-4 句话给出总览，说明这批文件整体是什么类型的数据、主要围绕哪些主题。\n"
                f"2. 然后覆盖当前展示到的 {selected_category_count} 个重点类别，逐类概括其典型内容，不要只讲第一类。\n"
                "3. 每个重点类别尽量提炼该类文件中反复出现的关键信息、人物/品牌/业务线/研究主题，而不是只复述文件名。\n"
                "4. 再额外总结 3-6 条跨类别观察，例如：哪些类别占比最高、哪些主题贯穿多个类别、哪些类别明显不同或是长尾补充。\n"
                "5. 在适合的时候点名少量代表文件作为例子，但整体必须以“全局总结”为主，而不是逐文件罗列。\n"
                f"6. {focus_instruction}\n"
                "7. 语气专业、清晰，使用 markdown 格式。\n"
                "8. 输出可以稍微详细一些，优先保证覆盖全面和信息密度，不要为了简短而牺牲质量。\n"
                "9. 忽略低信息提取噪音，例如空白音频、无语音、仅有音乐/歌声、无法翻译的语音、占位符等；不要把这些噪音当作主要内容来总结。\n"
                "10. [重要] 不要将总结结果包裹在 Markdown 代码块（```markdown ... ```）中。\n"
                "11. 不要先说“好的，我将为你总结”之类前言，直接给总结内容。\n"
                "\n【重要指令】请务必使用中文进行概括。即便文件摘要中包含英文，回复内容也必须保持纯中文。"
            )
        else:
            prompt = (
                f"{context_text}\n\n"
                "Generate a richer and better global summary using the full category distribution plus representative samples above.\n"
                "Requirements:\n"
                "1. Start with a 2-4 sentence overview explaining what kind of corpus this is and the main themes it contains.\n"
                f"2. Then cover the {selected_category_count} highlighted categories with a short but substantive summary for each; do not focus on only the first category.\n"
                "3. For each category, extract recurring topics, entities, brands, business lines, or research themes instead of merely repeating file names.\n"
                "4. Add 3-6 cross-category observations, such as dominant categories, themes spanning multiple categories, and notable outliers/long-tail areas.\n"
                "5. You may cite a few representative files as examples, but the answer must stay a true global summary rather than a file-by-file list.\n"
                f"6. {focus_instruction}\n"
                "7. Use professional, clear markdown.\n"
                "8. It is okay to be somewhat more detailed; prioritize coverage and insight density over being overly short.\n"
                "9. Ignore low-information extraction noise such as blank audio, no speech, music-only/singing-only clips, translation failures, or placeholder text; do not present those artifacts as substantive findings.\n"
                "10. [IMPORTANT] Do NOT wrap the summary in a Markdown code block (```markdown ... ```).\n"
                "11. Do not start with preambles like 'Okay, I will summarize'; go straight to the summary.\n"
                "\n[IMPORTANT] You MUST generate the entire summary in English. Even if the input summaries are in Chinese, your final response must be purely in English."
            )


    llm = get_llm(streaming=True, session_id=session_id)
    final_answer = ""
    for ch in llm.stream([HumanMessage(content=prompt)]):
        if self.is_aborted(session_id):
            yield {"type": "done", "ok": False, "query_type": "interrupted", "message": "生成已中断"}
            return
        delta = getattr(ch, "content", "") or ""
        if delta:
            yield {"type": "text", "delta": delta}
            final_answer += delta

    hist_ref = self._get_history_ref(session_id)
    try:
        hist_ref[-1]["a"] = final_answer
    except Exception:
        pass
    yield {"type": "done", "ok": True, "query_type": "summarize_all", "sources": files, "trace": []}
