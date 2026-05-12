"""
core.kb package — FileKnowledgeBase and KB instance accessors.
"""
from .knowledge_base import FileKnowledgeBase
from tools.document_tools import get_kb_instance, set_kb_instance

__all__ = ["FileKnowledgeBase", "get_kb_instance", "set_kb_instance"]
