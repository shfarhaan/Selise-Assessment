"""
Agentic RAG System for Domain Knowledge QA
"""

from importlib import import_module
from typing import Any

__version__ = "1.0.0"
__author__ = "RAG Development Team"

__all__ = [
    "DocumentProcessor",
    "EmbeddingManager",
    "FAISSVectorStore",
    "RAGRetriever",
    "AgenticRAG"
]


def __getattr__(name: str) -> Any:
    """Lazy-load heavy submodules to avoid import-time side effects."""
    if name == "DocumentProcessor":
        return getattr(import_module(".document_processor", __name__), name)
    if name in {"EmbeddingManager", "FAISSVectorStore"}:
        return getattr(import_module(".embeddings", __name__), name)
    if name == "RAGRetriever":
        return getattr(import_module(".retriever", __name__), name)
    if name == "AgenticRAG":
        return getattr(import_module(".agent", __name__), name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
