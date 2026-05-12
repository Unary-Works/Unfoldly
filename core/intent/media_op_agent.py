import json
import logging
import re
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def _generate_media_llm_text(llm_service: any, prompt: str) -> str:
    if hasattr(llm_service, "run_local_llm"):
        return llm_service.run_local_llm(prompt)
    if hasattr(llm_service, "generate"):
        return llm_service.generate(prompt, temperature=0.1, max_tokens=256)
    raise AttributeError("LLM service does not support run_local_llm or generate")

class MediaOpAgent:
    """
    Subagent that classifies and extracts parameters for deep media intent queries.
    Focuses on understanding what the user wants to do with a video or audio file.
    
    Sub-intents:
      - point_lookup: Look up what happens at a specific timestamp.
      - range_summary: Summarize a bounded interval in the media.
      - content_search: Search for a concept across the media to find timestamps.
      - media_summary: Summarize the whole video or audio.
    """

    @classmethod
    def analyze(cls, query: str, llm_service: any) -> Optional[Dict]:
        """Use a fast LLM prompt to classify the media query."""
        
        prompt = (
            "You are a media query parameters extractor. Analyze the user's video/audio request.\n"
            f"Query: \"{query}\"\n\n"
            "Return valid JSON ONLY. Output shape:\n"
            "{\n"
            "  \"sub_intent\": \"point_lookup\" | \"range_summary\" | \"content_search\" | \"media_summary\",\n"
            "  \"target_type\": \"audio_content\" | \"video_visual\" | \"video_audio\",\n"
            "  \"time_sec\": float or null,\n"
            "  \"time_end_sec\": float or null,\n"
            "  \"file_hint\": \"filename.ext\" or \"\",\n"
            "  \"search_concept\": \"what the user is looking for\" or \"\"\n"
            "}\n\n"
            "Rules:\n"
            "1. sub_intent:\n"
            "   - 'point_lookup' if the user provides a specific time point (e.g. at 1:20, near the 20-minute mark).\n"
            "   - 'range_summary' if the user asks what happens/discussed/described in a bounded interval (e.g. between 20 and 30 minutes, from 00:20:00 to 00:22:00, first 20 minutes, 前20分钟).\n"
            "   - 'content_search' if the user asks WHEN or WHERE something happens without giving a specific time (e.g., '第几秒出现', 'which second', 'find when').\n"
            "   - 'media_summary' if the user asks for a general summary of the file.\n"
            "2. target_type:\n"
            "   - 'video_visual' if they ask about scenes, frames, visual content, or 'what happens'.\n"
            "   - 'audio_content' if they ask what is said, speech, transcripts.\n"
            "   - 'video_audio' if they ask what is discussed/described in a video segment without explicitly focusing on visuals only.\n"
            "3. time_sec / time_end_sec: extract floats in seconds if specified, else null.\n"
            "4. If there is a start and end time, prefer 'range_summary' over 'point_lookup'.\n"
            "5. search_concept: only used for 'content_search'. It's the keyword/concept they want to find (e.g. 'Unfoldly'). Leave it empty for point/range/summary requests.\n"
            "6. file_hint: extract filename if present, otherwise empty string.\n"
            "Examples:\n"
            "   - 'What is discussed between 20 and 30 minutes in demo.mp4?' -> {\"sub_intent\":\"range_summary\",\"target_type\":\"video_audio\",\"time_sec\":1200,\"time_end_sec\":1800,\"file_hint\":\"demo.mp4\",\"search_concept\":\"\"}\n"
            "   - 'Summarize the first 20 minutes of sample_video.mp4' -> {\"sub_intent\":\"range_summary\",\"target_type\":\"video_audio\",\"time_sec\":0,\"time_end_sec\":1200,\"file_hint\":\"sample_video.mp4\",\"search_concept\":\"\"}\n"
            "   - 'What is shown at 20:00 in clip.mp4?' -> {\"sub_intent\":\"point_lookup\",\"target_type\":\"video_visual\",\"time_sec\":1200,\"time_end_sec\":null,\"file_hint\":\"clip.mp4\",\"search_concept\":\"\"}\n"
            "   - 'Which second mentions OpenAI in the audio?' -> {\"sub_intent\":\"content_search\",\"target_type\":\"audio_content\",\"time_sec\":null,\"time_end_sec\":null,\"file_hint\":\"\",\"search_concept\":\"OpenAI\"}\n"
        )
        
        try:
            # Use a fast, low-temperature generation
            resp = _generate_media_llm_text(llm_service, prompt)
            if not resp:
                return None
            
            clean_resp = resp.strip()
            if "```json" in clean_resp:
                clean_resp = clean_resp.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_resp:
                clean_resp = clean_resp.split("```")[1].split("```")[0].strip()
            
            # Additional cleanup in case there's text around the JSON block
            match = re.search(r'\{.*\}', clean_resp, re.DOTALL)
            if match:
                clean_resp = match.group(0)
                
            data = json.loads(clean_resp)
            
            sub_intent = cls._normalize_sub_intent(
                str(data.get("sub_intent") or "point_lookup").strip(),
                time_sec=data.get("time_sec"),
                time_end_sec=data.get("time_end_sec"),
            )
            target_type = data.get("target_type", "video_visual")
            
            # ── Route to correct action based on sub_intent ──────────────────────
            # content_search: user asks WHEN/WHERE something appears → media_content_search
            # point_lookup: user asks about a specific timestamp → media_export
            # media_summary: user asks for overview of file → media_export (summary mode)
            if sub_intent == "content_search":
                search_concept = str(data.get("search_concept") or "").strip()
                if not search_concept:
                    # Fall back to point_lookup if no concept extracted
                    sub_intent = "point_lookup"
                else:
                    params = {
                        "query": search_concept or query,
                        "media_type": "all",
                        "sub_intent": "content_search",
                    }
                    if data.get("file_hint"):
                        params["file_hint"] = str(data["file_hint"])
                    return {
                        "action": "media_content_search",
                        "params": params,
                        "confidence": 0.95,
                    }

            # point_lookup / media_summary → media_export
            params = {
                "query": query,
                "target_type": target_type,
                "sub_intent": sub_intent,
            }
            
            if data.get("time_sec") is not None:
                params["time_sec"] = float(data["time_sec"])
            if data.get("time_end_sec") is not None:
                params["time_end_sec"] = float(data["time_end_sec"])
            if data.get("file_hint"):
                params["file_hint"] = str(data["file_hint"])
            if data.get("search_concept"):
                params["search_concept"] = str(data["search_concept"])

            return {
                "action": "media_export",
                "params": params,
                "confidence": 0.98,
            }
            
        except Exception as e:
            logger.error(f"[MediaOpAgent] LLM parsing failed: {e}")
            return None

    @staticmethod
    def _normalize_sub_intent(
        sub_intent: str,
        *,
        time_sec: object,
        time_end_sec: object,
    ) -> str:
        normalized = str(sub_intent or "").strip().lower() or "point_lookup"
        has_time = time_sec is not None
        has_range = time_sec is not None and time_end_sec is not None

        if has_range and normalized in {"point_lookup", "media_summary", "range_summary"}:
            return "range_summary"
        if has_time and normalized == "media_summary":
            return "point_lookup"
        if normalized not in {"point_lookup", "range_summary", "content_search", "media_summary"}:
            return "range_summary" if has_range else "point_lookup"
        return normalized
