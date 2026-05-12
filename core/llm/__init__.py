"""LLM services package."""
from .utils import (
    _model_supports_system_role,
    _estimate_tokens,
    _chunk_text,
    _chunk_text_by_newlines,
    _approx_tokens_from_text,
    _approx_tokens_from_messages,
    summarize_long_tool_result,
    convert_messages_for_gemma,
    stream_replace_markdown_links,
    build_messages_for_model,
)
from .builder import get_llm, get_llm_with_tools, ToolAgentState

__all__ = [
    "get_llm", "get_llm_with_tools", "ToolAgentState",
    "_chunk_text", "build_messages_for_model", "stream_replace_markdown_links",
]
