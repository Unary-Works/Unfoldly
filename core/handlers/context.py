"""HandlerContext — unified context for all handler functions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class HandlerContext:
    """Unified context passed to all handler functions."""
    question: str
    params: Dict[str, Any]
    active_paths: List[str]
    session_id: Optional[str]
    lang: str                        # "zh" | "en"
    kb: Any                          # FileKnowledgeBase instance
    llm_service: Any                 # _LocalTextService instance  
    prompt_formatter: Any            # callable(prompt_name, lang) -> str
    normalize_category: Any          # callable(str) -> str
    is_generic_category: Any         # callable(str) -> bool
    get_category_keywords: Any       # callable(str) -> List[str]
    history: List[Dict] = field(default_factory=list)
    last_results: List[Dict] = field(default_factory=list)
    abort_checker: Any = None        # callable() -> bool
    log_fn: Any = None               # callable(msg) -> None
