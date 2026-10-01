"""Small helpers with no heavy dependencies, so they are easy to test."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Iterable

from PIL import Image


def image_to_data_uri(image_path: str | Path, max_side: int = 1600, quality: int = 85) -> str:
    """Load an image, shrink it to fit max_side, and return a base64 JPEG data URI."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def clean_page(page):
    """Pinecone hands numeric metadata back as floats (6.0), so turn whole numbers back into ints."""
    if isinstance(page, float) and page.is_integer():
        return int(page)
    return page


def format_context(docs: Iterable) -> str:
    """Turn retrieved documents into one labelled context string for the prompt."""
    parts = []
    for doc in docs:
        page = clean_page(doc.metadata.get("page"))
        modality = (doc.metadata.get("modality") or "unknown").upper()
        parts.append(f"[Page {page} | {modality}]\n{doc.page_content}")
    return "\n\n".join(parts)


def collect_visual_paths(docs: Iterable) -> list[str]:
    """Return the image files behind any retrieved visual documents, in rank order."""
    paths: list[str] = []
    for doc in docs:
        if doc.metadata.get("modality") != "visual":
            continue
        path = doc.metadata.get("image_path")
        if path and Path(path).exists() and path not in paths:
            paths.append(path)
    return paths
