"""Multimodal RAG over a PDF with text, tables, charts and diagrams."""

__all__ = ["Settings", "MultimodalRAG", "RagResult"]


def __getattr__(name):
    # Lazy imports keep `multimodal_rag.utils` usable without the heavy dependencies.
    if name == "Settings":
        from .config import Settings

        return Settings
    if name in ("MultimodalRAG", "RagResult"):
        from . import rag

        return getattr(rag, name)
    raise AttributeError(name)
