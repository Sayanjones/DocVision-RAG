"""Central configuration. Everything can be overridden from the environment or a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class Settings:
    pdf_path: Path = field(
        default_factory=lambda: _path(
            "PDF_PATH", ROOT / "data" / "NovaCore_Multimodal_Company_Report_2026.pdf"
        )
    )
    image_dir: Path = field(
        default_factory=lambda: _path("IMAGE_DIR", ROOT / "extracted_images")
    )

    text_model: str = field(
        default_factory=lambda: os.getenv("TEXT_MODEL", "openai/gpt-oss-20b")
    )
    vision_model: str = field(
        default_factory=lambda: os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    pinecone_index: str = field(
        default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "novacore-multimodal-rag")
    )
    pinecone_namespace: str = field(
        default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "fy2026-demo")
    )
    pinecone_cloud: str = field(default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws"))
    pinecone_region: str = field(
        default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1")
    )

    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    max_images_per_answer: int = field(
        default_factory=lambda: int(os.getenv("MAX_IMAGES_PER_ANSWER", "3"))
    )

    def require_keys(self) -> None:
        missing = [k for k in ("GROQ_API_KEY", "PINECONE_API_KEY") if not os.getenv(k)]
        if missing:
            raise RuntimeError(
                f"Missing environment variable(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )
