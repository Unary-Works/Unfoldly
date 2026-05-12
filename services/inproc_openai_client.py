
from __future__ import annotations
import base64
import json
import os
import time
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from services.local_llm import get_local_llm_manager
from services.preference_manager import PreferenceManager
from utils.logger import get_child_logger

logger = get_child_logger(__name__)

_UNSUPPORTED_TEMPLATE_KWARG_KEYS = ("enable_" "thinking", "add_generation_" "prompt")


def _strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    try:
        import re
        # remove <think>...</think> blocks
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
        # remove <|channel>thought...<channel|> blocks (Gemma 4 etc.)
        text = re.sub(r"<\|channel\|?>?thought[\s\S]*?</?channel\|?>?", "", text, flags=re.IGNORECASE)
    except Exception:
        pass
    # remove stray tags
    return (text or "").replace("<think>", "").replace("</think>", "").replace("<|channel>thought", "").replace("<channel|>", "").replace("</channel>", "")


def _compress_data_url_image(url: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
        return url, None

    try:
        from PIL import Image, ImageOps
    except Exception:
        return url, None

    try:
        header, b64 = url.split(",", 1)
        mime = header[5:].split(";", 1)[0].strip().lower()
        raw = base64.b64decode(b64)
    except Exception:
        return url, None

    try:
        max_edge = int(os.getenv("FILEAGENT_VL_IMAGE_MAX_EDGE", "896") or "896")
    except Exception:
        max_edge = 896
    try:
        jpeg_quality = int(os.getenv("FILEAGENT_VL_IMAGE_JPEG_QUALITY", "80") or "80")
    except Exception:
        jpeg_quality = 80

    max_edge = max(256, min(max_edge, 2048))
    jpeg_quality = max(40, min(jpeg_quality, 95))

    try:
        with Image.open(BytesIO(raw)) as src:
            src = ImageOps.exif_transpose(src)
            orig_w, orig_h = src.size

            if max(orig_w, orig_h) <= max_edge:
                return url, None

            if src.mode == "RGB":
                base_img = src.copy()
            elif src.mode in {"RGBA", "LA"} or (src.mode == "P" and "transparency" in src.info):
                rgba = src.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.getchannel("A"))
                base_img = bg
            else:
                base_img = src.convert("RGB")

        if max(base_img.size) > max_edge:
            ratio = min(float(max_edge) / float(base_img.size[0]), float(max_edge) / float(base_img.size[1]))
            new_size = (
                max(1, int(base_img.size[0] * ratio)),
                max(1, int(base_img.size[1] * ratio)),
            )
            base_img = base_img.resize(new_size, Image.Resampling.LANCZOS)

        buf = BytesIO()
        base_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        encoded = buf.getvalue()
        new_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("utf-8")
        return new_url, {
            "orig_bytes": len(raw),
            "new_bytes": len(encoded),
            "orig_size": f"{orig_w}x{orig_h}",
            "new_size": f"{base_img.size[0]}x{base_img.size[1]}",
        }
    except Exception:
        return url, None


@dataclass
class _Delta:
    content: Optional[str] = None


@dataclass
class _Msg:
    content: Optional[str] = None


@dataclass
class _Choice:
    message: _Msg
    delta: _Delta
    finish_reason: Optional[str] = None


@dataclass
class _Resp:
    choices: List[_Choice]


class _ChatCompletions:
    def __init__(self, base_dir: str):
        self._base_dir = base_dir
        self._pref = PreferenceManager(base_dir)
        self._mgr = get_local_llm_manager(base_dir)

    def _needs_vision(self, messages: Any, model: Optional[str]) -> bool:
        try:
            if model and "vl" in str(model).lower():
                return True
        except Exception:
            pass

        try:
            for m in (messages or []):
                c = m.get("content")
                if isinstance(c, list):
                    for item in c:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            return True
        except Exception:
            return False
        return False

    def _select_model_id(self, messages: Any, model: Optional[str], force_text_model: bool = False) -> str:
        if model is not None:
            mid = str(model).strip()
            if mid:
                return mid
            raise ValueError("Empty model id was provided to local LLM client")
        mid = self._pref.get_selected_model_id()
        return mid or "qwen3-4b-gguf"

    def _compress_vision_inputs(self, messages: Any) -> Any:
        if not isinstance(messages, list):
            return messages

        changed_any = False
        compressed_count = 0
        compressed_stats: List[str] = []
        new_messages: List[Any] = []

        for message in messages:
            if not isinstance(message, dict):
                new_messages.append(message)
                continue

            content = message.get("content")
            if not isinstance(content, list):
                new_messages.append(message)
                continue

            changed_message = False
            new_content: List[Any] = []
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image_url":
                    new_content.append(item)
                    continue

                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    raw_url = image_url.get("url")
                else:
                    raw_url = image_url

                new_url, stats = _compress_data_url_image(str(raw_url or ""))
                if stats is None:
                    new_content.append(item)
                    continue

                changed_message = True
                changed_any = True
                compressed_count += 1
                compressed_stats.append(
                    f"{stats['orig_size']} {stats['orig_bytes']}B -> {stats['new_size']} {stats['new_bytes']}B"
                )

                new_item = dict(item)
                if isinstance(image_url, dict):
                    new_image_url = dict(image_url)
                    new_image_url["url"] = new_url
                    new_item["image_url"] = new_image_url
                else:
                    new_item["image_url"] = new_url
                new_content.append(new_item)

            if changed_message:
                new_message = dict(message)
                new_message["content"] = new_content
                new_messages.append(new_message)
            else:
                new_messages.append(message)

        if changed_any:
            logger.info(
                f"[chat] resized {compressed_count} oversized vision input(s) before local VL: "
                + "; ".join(compressed_stats[:3])
            )
        return new_messages if changed_any else messages

    def create(
        self,
        *,
        model: Optional[str] = None,
        messages: Any = None,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> Union[_Resp, Iterator[_Resp]]:
        start_ts = time.time()
        force_text_model = kwargs.pop("force_text_model", False)
        msg_count = len(messages) if isinstance(messages, list) else 0
        needs_vision = self._needs_vision(messages, model)
        model_id = self._select_model_id(messages, model, force_text_model)
        
        if needs_vision:
            cfg = self._mgr.get_target_model_config(model_id) or {}
            
            supports_vision = cfg.get("supports_vision")
            if supports_vision is None:
                mtype = str(cfg.get("type", "")).lower()
                model_files = cfg.get("files") or []
                has_mmproj = any("mmproj" in str(f).lower() for f in model_files)
                supports_vision = ("image" in mtype or "multimodal" in mtype or "vl" in model_id.lower() or has_mmproj)

            if not supports_vision:
                logger.warning(f"[chat] model {model_id} does not support vision but image_url was provided. Skipping.")
                raise ValueError(f"Selected model '{model_id}' does not support vision/image tasks.")
                
        logger.info(
            f"[chat] create start model_hint={model or '<auto>'} selected_model={model_id} "
            f"stream={bool(stream)} force_text_model={bool(force_text_model)} needs_vision={bool(needs_vision)} "
            f"message_count={msg_count}"
        )

        qf = None
        try:
            qf = self._pref.get_selected_quantization_file(model_id)
        except Exception:
            qf = None
        logger.info(f"[chat] quantization selected_model={model_id} quantization={qf or '<default>'}")

        # Current llama_cpp runtime in this app does not reliably support direct
        # extra chat-template kwargs. Keep this layer defensive:
        # - Gemma 4 should use stop/logit_bias + output cleaning
        # - Qwen-family may use prompt suffixes upstream
        # - neither path should pass unsupported raw kwargs through to llama_cpp
        for _tk in _UNSUPPORTED_TEMPLATE_KWARG_KEYS:
            kwargs.pop(_tk, None)

        formatted_messages = []
        system_content = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                system_content += m.get("content", "") + "\n\n"
            else:
                formatted_messages.append(m)
                
        if system_content.strip():
            # Inject system_content into the first user message
            injected = False
            for fm in formatted_messages:
                if isinstance(fm, dict) and fm.get("role") == "user":
                    fm["content"] = system_content.strip() + "\n\n" + str(fm.get("content", ""))
                    injected = True
                    break
            if not injected:
                formatted_messages.insert(0, {"role": "user", "content": system_content.strip()})

        if needs_vision:
            formatted_messages = self._compress_vision_inputs(formatted_messages)

        def _call_mgr(extra: Dict[str, Any]) -> Any:
            return self._mgr.create_chat_completion(
                model_id=model_id,
                preferred_quantization_file=qf,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=float(temperature),
                stream=bool(stream),
                needs_vision=needs_vision,
                **kwargs,
                **extra,
            )

        try:
            out = _call_mgr({})
        except TypeError as e:
            logger.error(f"[chat] create failed selected_model={model_id} error={e}")
            raise

        if not stream:
            text = ""
            finish_reason = None
            try:
                choice0 = (out.get("choices") or [{}])[0]
                text = (choice0.get("message") or {}).get("content") or ""
                fr = choice0.get("finish_reason")
                if fr is not None:
                    finish_reason = str(fr)
            except Exception:
                text = ""
            text = _strip_think_blocks(str(text))
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.info(
                f"[chat] create done selected_model={model_id} stream=False text_len={len(text)} elapsed_ms={elapsed_ms}"
            )
            return _Resp(
                choices=[
                    _Choice(
                        message=_Msg(content=text),
                        delta=_Delta(content=None),
                        finish_reason=finish_reason,
                    )
                ]
            )


        def _iter() -> Iterator[_Resp]:
            in_think = False        # inside <think>...</think> block
            in_g4_think = False     # inside Gemma4 <|channel>thought...<channel|> block
            carry = ""
            max_carry = 20          # covers "</think>" and "<channel|>" boundary splits
            chunk_count = 0
            emitted_chars = 0
            first_chunk_ms: Optional[int] = None
            closing_early = False

            # Gemma4 thinking markers (plain-text, not special tokens)
            _G4_STARTS = ["<|channel>thought", "<channel|>thought", "<|channel|>thought"]
            _G4_END = "<channel|>"

            try:
                for ch in out:  # type: ignore
                    delta_text = ""
                    finish_reason = None
                    try:
                        choice0 = (ch.get("choices") or [{}])[0]
                        delta_text = (choice0.get("delta") or {}).get("content") or ""
                        fr = choice0.get("finish_reason")
                        if fr is not None:
                            finish_reason = str(fr)
                    except Exception:
                        delta_text = ""

                    if not delta_text and not carry:
                        # Preserve terminal metadata (e.g. finish_reason=length) for upper layers.
                        if finish_reason:
                            yield _Resp(
                                choices=[
                                    _Choice(
                                        message=_Msg(content=None),
                                        delta=_Delta(content=None),
                                        finish_reason=finish_reason,
                                    )
                                ]
                            )
                        continue

                    buf = carry + str(delta_text)
                    carry = ""

                    # ── Gemma4 thinking-block filter ──────────────────────────────────────
                    # Gemma4 format: <|channel>thought\n[thinking]\n<channel|>[answer]
                    # Discard everything between start markers and the <channel|> separator.
                    if in_g4_think:
                        end_g4 = buf.find(_G4_END)
                        if end_g4 >= 0:
                            buf = buf[end_g4 + len(_G4_END):]
                            in_g4_think = False
                        else:
                            carry = buf[-max_carry:] if len(buf) > max_carry else buf
                            buf = ""
                    if not in_g4_think and buf:
                        best_p, best_l = -1, 0
                        for _m in _G4_STARTS:
                            _p = buf.find(_m)
                            if _p >= 0 and (best_p < 0 or _p < best_p):
                                best_p, best_l = _p, len(_m)
                        if best_p >= 0:
                            _before = buf[:best_p]
                            _after = buf[best_p + best_l:]
                            _end2 = _after.find(_G4_END)
                            if _end2 >= 0:
                                buf = _before + _after[_end2 + len(_G4_END):]
                            else:
                                buf = _before
                                in_g4_think = True
                    # ─────────────────────────────────────────────────────────────────────

                    # incremental filter: remove <think>...</think> blocks without breaking streaming
                    out_parts: List[str] = []
                    while buf:
                        if in_think:
                            end = buf.find("</think>")
                            if end < 0:
                                # keep tail in case tag splits across chunks
                                if len(buf) > max_carry:
                                    carry = buf[-max_carry:]
                                else:
                                    carry = buf
                                buf = ""
                                break
                            buf = buf[end + len("</think>") :]
                            in_think = False
                            continue

                        start = buf.find("<think>")
                        if start < 0:
                            # no think tag: emit but keep a small carry for boundary cases
                            cleaned = buf.replace("</think>", "").replace("<think>", "")
                            if len(cleaned) > max_carry:
                                out_parts.append(cleaned[:-max_carry])
                                carry = cleaned[-max_carry:]
                            else:
                                carry = cleaned
                            buf = ""
                            break

                        # found start tag
                        before = buf[:start].replace("</think>", "").replace("<think>", "")
                        if before:
                            out_parts.append(before)
                        buf = buf[start + len("<think>") :]
                        in_think = True

                    emit = "".join(out_parts)
                    if emit:
                        if first_chunk_ms is None:
                            first_chunk_ms = int((time.time() - start_ts) * 1000)
                            logger.info(
                                f"[chat] first_chunk selected_model={model_id} stream=True first_chunk_ms={first_chunk_ms}"
                            )
                        chunk_count += 1
                        emitted_chars += len(emit)
                        yield _Resp(
                            choices=[
                                _Choice(
                                    message=_Msg(content=None),
                                    delta=_Delta(content=emit),
                                    finish_reason=finish_reason,
                                )
                            ]
                        )
            except GeneratorExit:
                closing_early = True
                raise
            finally:
                try:
                    if hasattr(out, "close"):
                        out.close()  # type: ignore
                except Exception:
                    pass

                # flush remaining carry if not in any think block
                if (not closing_early) and carry and not in_think and not in_g4_think:
                    tail = carry.replace("</think>", "").replace("<think>", "")
                    if tail:
                        chunk_count += 1
                        emitted_chars += len(tail)
                        yield _Resp(
                            choices=[
                                _Choice(
                                    message=_Msg(content=None),
                                    delta=_Delta(content=tail),
                                    finish_reason=None,
                                )
                            ]
                        )
                elapsed_ms = int((time.time() - start_ts) * 1000)
                logger.info(
                    f"[chat] create done selected_model={model_id} stream=True chunks={chunk_count} "
                    f"text_len={emitted_chars} elapsed_ms={elapsed_ms} first_chunk_ms={first_chunk_ms or 0}"
                )

        return _iter()


class _Chat:
    def __init__(self, base_dir: str):
        self.completions = _ChatCompletions(base_dir)


class InProcOpenAI:

    def __init__(self, base_dir: Optional[str] = None):
        if not base_dir:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.chat = _Chat(base_dir)


_CLIENT: Optional[InProcOpenAI] = None


def get_inproc_openai_client(base_dir: Optional[str] = None) -> InProcOpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = InProcOpenAI(base_dir=base_dir)
    return _CLIENT


def clear_inproc_openai_client():
    global _CLIENT
    _CLIENT = None
