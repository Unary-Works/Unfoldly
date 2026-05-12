"""
Repo-local skill helpers for time-based media analysis.

This module keeps the "what to do for video/audio + time" policy in one place:
  - choose point lookup vs interval summary
  - emit stage copy for the UI
  - build dynamic prompts from collected evidence
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_RANGE_SUMMARY_SIGNAL = re.compile(
    r"\b(summary|summarize|overview|recap|key\s+takeaways?|main\s+points?)\b"
    r"|\bwhat\s+(?:is|was)\s+discussed\b"
    r"|\bwhat\s+(?:is|was)\s+(?:being\s+)?described\b"
    r"|\btell\s+me\s+about\b"
    r"|总结|概括|归纳|概要|要点|主要内容|讲了什么|在讲什么|讨论了什么|描述了什么|在描述什么",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MediaTimeSkillPlan:
    question: str
    query_mode: str
    target_type: str
    time_sec: float
    time_end_sec: Optional[float]
    target_file_name: str
    language: str
    has_video: bool
    has_audio: bool
    response_style: str


class MediaTimeSkill:
    """Skill-like planner and prompt builder for media+time requests."""

    @staticmethod
    def recommended_interval_frame_count(
        span_sec: float,
        *,
        has_audio: bool,
        prefer_dense: bool = False,
    ) -> int:
        span = max(0.0, float(span_sec or 0.0))
        if span <= 90:
            count = 4
        elif span <= 300:
            count = 6
        elif span <= 900:
            count = 8
        else:
            count = 10
        if not has_audio:
            count += 1
        if prefer_dense:
            count += 2
        return max(4, min(count, 12))

    @staticmethod
    def remaining_interval_frame_budget(
        target_total: int,
        existing_visual_count: int,
    ) -> int:
        return max(0, int(target_total or 0) - max(0, int(existing_visual_count or 0)))

    @staticmethod
    def recommended_visual_batch_size(
        frame_count: int,
        *,
        fast_motion: bool = False,
    ) -> int:
        total = max(1, int(frame_count or 0))
        if total <= 3:
            size = 2
        elif total <= 8:
            size = 3
        else:
            size = 4
        if fast_motion and size > 2:
            size -= 1
        return max(2, min(size, 4))

    @staticmethod
    def chunk_sequence(
        items: Sequence[Tuple[float, str]],
        *,
        chunk_size: int,
    ) -> List[List[Tuple[float, str]]]:
        size = max(1, int(chunk_size or 1))
        return [list(items[idx: idx + size]) for idx in range(0, len(items), size)]

    @classmethod
    def supports(cls, params: Optional[Dict]) -> bool:
        if not params:
            return False
        return params.get("time_sec") is not None and str(params.get("target_type") or "").strip() in {
            "audio_content",
            "video_audio",
            "video_visual",
        }

    @classmethod
    def build_plan(
        cls,
        question: str,
        params: Dict,
        *,
        target_file_name: str,
        language: str,
        has_video: bool,
        has_audio: bool,
    ) -> MediaTimeSkillPlan:
        target_type = str(params.get("target_type") or "audio_content")
        time_sec = float(params.get("time_sec") or 0.0)
        time_end_raw = params.get("time_end_sec")
        time_end_sec = float(time_end_raw) if time_end_raw is not None else None
        query_mode = "range_summary" if cls.is_range_summary_request(question, params) else "point_lookup"
        if query_mode == "range_summary":
            response_style = "coherent_interval_summary"
        elif target_type == "video_visual":
            response_style = "visual_point_lookup"
        else:
            response_style = "content_point_lookup"

        return MediaTimeSkillPlan(
            question=str(question or ""),
            query_mode=query_mode,
            target_type=target_type,
            time_sec=time_sec,
            time_end_sec=time_end_sec,
            target_file_name=str(target_file_name or ""),
            language=str(language or "en"),
            has_video=bool(has_video),
            has_audio=bool(has_audio),
            response_style=response_style,
        )

    @staticmethod
    def is_range_summary_request(question: str, params: Optional[Dict]) -> bool:
        if not params:
            return False
        time_end = params.get("time_end_sec")
        if time_end is None:
            return False
        sub_intent = str((params or {}).get("sub_intent") or "").strip()
        if sub_intent == "range_summary":
            return True
        return bool(_RANGE_SUMMARY_SIGNAL.search(str(question or "")))

    @staticmethod
    def sample_timeline_entries(
        entries: Sequence[Tuple[float, str]],
        *,
        max_items: int = 8,
    ) -> List[Tuple[float, str]]:
        cleaned = [(float(ts), str(text or "").strip()) for ts, text in entries if str(text or "").strip()]
        if len(cleaned) <= max_items:
            return cleaned
        if max_items <= 1:
            return [cleaned[0]]

        last_idx = len(cleaned) - 1
        picked = {0, last_idx}
        for idx in range(1, max_items - 1):
            picked.add(int(round(idx * last_idx / max(1, max_items - 1))))
        return [cleaned[idx] for idx in sorted(picked)[:max_items]]

    @classmethod
    def stage_message(
        cls,
        stage: str,
        *,
        plan: MediaTimeSkillPlan,
        time_label: str,
    ) -> str:
        lang = plan.language
        file_name = plan.target_file_name or "media file"
        zh = lang == "zh"
        if stage == "lock_interval":
            return (
                f"正在解析 `{file_name}` 的 {time_label} 区间..."
                if zh
                else f"Parsing `{file_name}` for the interval {time_label}..."
            )
        if stage == "lookup_interval_visual":
            return (
                f"正在读取 {time_label} 区间已索引的代表画面证据..."
                if zh
                else f"Loading indexed representative visual evidence across {time_label}..."
            )
        if stage == "collect_audio":
            return (
                f"正在读取 {time_label} 区间已索引的音频内容..."
                if zh
                else f"Looking up indexed speech evidence within {time_label}..."
            )
        if stage == "generate_range_summary":
            return (
                "正在理解这段视频内容，并生成连贯描述..."
                if zh
                else "Understanding the video segment and generating a coherent description..."
            )
        if stage == "reuse_interval_cache":
            return (
                f"正在复用 `{file_name}` 这个区间的历史解析结果..."
                if zh
                else f"Reusing cached interval analysis for `{file_name}`..."
            )
        if stage == "cache_interval":
            return (
                f"正在保存 `{file_name}` 这段区间的综合结果..."
                if zh
                else f"Saving the synthesized interval result for `{file_name}`..."
            )
        if stage == "lookup_transcript":
            return (
                f"正在定位 `{file_name}` 在 {time_label} 附近的音频内容..."
                if zh
                else f"Looking up transcript evidence around {time_label} in `{file_name}`..."
            )
        if stage == "lookup_visual":
            return (
                f"正在查看 `{file_name}` 在 {time_label} 附近的已索引画面..."
                if zh
                else f"Looking up indexed visual evidence around {time_label} in `{file_name}`..."
            )
        if stage == "extract_point_visual":
            return (
                f"正在从 `{file_name}` 直接抽取 {time_label} 附近的画面..."
                if zh
                else f"Extracting visual frames around {time_label} directly from `{file_name}`..."
            )
        if stage == "generate_point_answer":
            return (
                "正在结合画面和附近音频生成回答..."
                if zh
                else "Combining the visual frames and nearby speech into an answer..."
            )
        return ""

    @classmethod
    def build_range_summary_prompt(
        cls,
        *,
        plan: MediaTimeSkillPlan,
        time_label: str,
        transcript_rows: Sequence[Dict[str, object]],
        visual_entries: Sequence[Tuple[float, str]],
        format_time: Callable[[float], str],
    ) -> str:
        transcript_block = "\n".join(
            f"[{format_time(float(row['asr_start_sec']))} - {format_time(float(row['asr_end_sec']))}] {str(row.get('text') or '').strip()}"
            for row in transcript_rows[:18]
            if str(row.get("text") or "").strip()
        )
        visual_block = "\n".join(
            f"[{format_time(float(ts))}] {desc}"
            for ts, desc in visual_entries[:8]
            if str(desc or "").strip()
        )

        if plan.language == "zh":
            return (
                f"用户想了解媒体文件 `{plan.target_file_name}` 在 {time_label} 这段区间里的连续内容。\n\n"
                "请把这个区间当作一段连续视频来描述，而不是零散列点。\n"
                "你的回答要像自然讲述这段视频一样连贯，优先用自然段，不要像机器摘录。\n"
                "请说明这段时间里画面的大致推进、场景或界面的变化、动作/操作流程，以及如果有音频时这段音频在讲什么、如何和画面对应。\n"
                "只描述这个区间，不要扩展到区间外；证据不足时要明确说明，但不要编造。\n\n"
                f"<用户问题>\n{plan.question}\n</用户问题>\n\n"
                f"<区间音频证据>\n{transcript_block or '(none)'}\n</区间音频证据>\n\n"
                f"<区间画面证据>\n{visual_block or '(none)'}\n</区间画面证据>\n\n"
                "请输出一段连贯、具体、像口头讲解一样自然的中文描述，优先写成自然段，必要时可以补 2-3 个要点。"
            )

        return (
            f"The user wants to understand `{plan.target_file_name}` as a continuous segment during {time_label}.\n\n"
            "Describe that interval as a coherent narrative rather than disconnected bullet points.\n"
            "Explain the visible progression across the interval, what changes or stays stable on screen, the actions or workflow shown, and, if speech exists, weave the audio content naturally into the description.\n"
            "Write it like a natural spoken walkthrough of the segment, not like fragmented extraction notes.\n"
            "Answer entirely in English. If speech or visual evidence is Chinese or another language, translate or paraphrase it into natural English while preserving proper names and visible text when useful.\n"
            "Stay strictly within the requested interval. If evidence is limited, say so briefly instead of inventing details.\n\n"
            f"<User question>\n{plan.question}\n</User question>\n\n"
            f"<Interval speech evidence>\n{transcript_block or '(none)'}\n</Interval speech evidence>\n\n"
            f"<Interval visual evidence>\n{visual_block or '(none)'}\n</Interval visual evidence>\n\n"
            "Write a coherent, specific description in natural prose. You may add a few short takeaways only if they help clarity."
        )

    @classmethod
    def build_point_visual_prompt(
        cls,
        *,
        plan: MediaTimeSkillPlan,
        time_label: str,
        frame_descriptions: Sequence[Tuple[float, str]],
        format_time: Callable[[float], str],
    ) -> str:
        frames_block = "\n\n".join(
            (
                f"帧 {format_time(float(ts))}：{desc}"
                if plan.language == "zh"
                else f"Frame at {format_time(float(ts))}: {desc}"
            )
            for ts, desc in frame_descriptions
        )

        if plan.language == "zh":
            return (
                f"用户问：「{plan.question}」\n\n"
                f"以下是视频「{plan.target_file_name}」在 {time_label} 附近连续提取的 "
                f"{len(frame_descriptions)} 帧画面描述：\n\n"
                f"{frames_block}\n\n"
                "请根据以上帧画面内容，详细、具体地回答用户的问题。"
                "描述应涵盖：屏幕上显示的主要内容（界面元素、文字、代码、窗口等）、"
                "用户可见的操作行为、各帧之间的变化或过渡，以及整体场景氛围。"
                "请用中文回复，内容翔实，不少于 150 字。"
            )

        return (
            f'User asked: "{plan.question}"\n\n'
            f"Below are {len(frame_descriptions)} consecutive frame descriptions extracted from "
            f"'{plan.target_file_name}' around {time_label}:\n\n"
            f"{frames_block}\n\n"
            "Based on these frames, provide a detailed and specific answer to the user's question. "
            "Answer entirely in English; translate any Chinese or other non-English visible text when useful. "
            "Cover: what is visible on screen (UI elements, text, code, windows, etc.), "
            "any user actions or interactions visible between frames, transitions, and the overall scene context. "
            "Be comprehensive — at least 3-5 detailed sentences."
        )

    @classmethod
    def build_point_audio_visual_prompt(
        cls,
        *,
        plan: MediaTimeSkillPlan,
        time_label: str,
        transcript_text: str,
        visual_entries: Sequence[Tuple[float, str]],
        format_time: Callable[[float], str],
    ) -> str:
        visual_block = "\n".join(
            f"[{format_time(float(ts))}] {desc}"
            for ts, desc in visual_entries[:8]
            if str(desc or "").strip()
        )

        if plan.language == "zh":
            return (
                f"用户问：「{plan.question}」\n\n"
                f"媒体文件：{plan.target_file_name}\n"
                f"目标时间：{time_label}\n\n"
                f"<附近语音证据>\n{transcript_text or '(none)'}\n</附近语音证据>\n\n"
                f"<附近画面证据>\n{visual_block or '(none)'}\n</附近画面证据>\n\n"
                "请把这个时间点当作一小段视频场景来复原，而不是做证据审计。要求：\n"
                "1. 不要以“好的，我将...”或“作为专业文档分析助手...”开头，直接回答。\n"
                "2. 先写“15 秒附近的画面”：用 2-4 句话具体描述画面里正在显示什么、主体位置、颜色/形状/界面元素、动作或变化。若有多帧证据，要按时间顺序串起来。\n"
                "3. 再写“附近听到的内容”：说明可用转录覆盖的时间窗和语音大意。若转录是 [0:00-1:00] 这种粗窗口，称它为“覆盖 0:15 的粗转录窗口”，不要说“没有直接语音证据”后就结束。\n"
                "4. 最后写“串起来理解”：尝试把画面和语音放在同一个场景里解释，例如旁白可能在描述目标/剧情，而画面显示绘制、生成或展示过程；如果二者明显不一致，要说明可能是 ASR 粗粒度、转录错位或识别误差，不能硬编。\n"
                "5. 不要把画面描述冒充成说话内容；语音和画面证据要区分清楚。\n"
                "6. 整体回答应像自然的视频讲解，优先自然段，可以少量项目符号；不要输出代码块。"
            )

        return (
            f'User asked: "{plan.question}"\n\n'
            f"Media file: {plan.target_file_name}\n"
            f"Target time: {time_label}\n\n"
            f"<Nearby speech evidence>\n{transcript_text or '(none)'}\n</Nearby speech evidence>\n\n"
            f"<Nearby visual evidence>\n{visual_block or '(none)'}\n</Nearby visual evidence>\n\n"
            "Reconstruct this timestamp as a short video scene, not as an evidence audit. Requirements:\n"
            "1. Do not start with preambles like 'Sure' or 'As a document analysis assistant'; answer directly.\n"
            "2. First describe the visual scene around the timestamp in 2-4 concrete sentences: main subject, position, colors/shapes, UI elements, actions, and changes. If multiple frame descriptions exist, connect them chronologically.\n"
            "3. Then describe the nearby speech. If the transcript is a broad window such as [0:00-1:00], call it a broad transcript window covering the timestamp; do not lead with 'there is no direct speech evidence' unless there is no transcript at all.\n"
            "4. Finally, synthesize the scene: explain how the speech may relate to what is visible. If speech and visuals clearly do not align, say the ASR may be coarse, offset, or mistaken; do not force a fabricated connection.\n"
            "5. Keep speech and visual evidence distinct; do not present visual details as spoken words.\n"
            "6. Answer entirely in English. If speech or visual evidence is Chinese or another language, translate or paraphrase it into natural English while preserving proper names and visible text when useful.\n"
            "7. Use natural prose like a video walkthrough. No code block."
        )
