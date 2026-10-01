"""Vision model calls: summarise a visual for indexing, and answer questions with the real image."""

from __future__ import annotations

from pathlib import Path

from groq import Groq

from .utils import image_to_data_uri

SUMMARY_PROMPT = """\
This visual was extracted from page {page} of the NovaCore FY2026 company report.

Describe the useful business information visible in the visual.

If it is a chart or graph:
- mention important values
- mention highest/lowest values
- mention the main trend

If it is a diagram:
- identify important components
- explain the flow or relationships

If it is a normal business image:
- describe the useful factual information

Keep the description concise and factual.
"""

ANSWER_PROMPT = """\
You are answering questions about the NovaCore Systems FY2026 report.

Use ONLY the retrieved context and the attached retrieved visuals.

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Answer factually.
- Use the attached visuals when relevant.
- Mention page numbers.
- If the information is missing, say you could not find it.
"""


def _image_part(image_path: str | Path) -> dict:
    return {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}}


def summarize_visual(client: Groq, model: str, image_path: str | Path, page: int) -> str:
    """Ask the vision model for a short factual description of one image."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SUMMARY_PROMPT.format(page=page)},
                    _image_part(image_path),
                ],
            }
        ],
        temperature=0,
        max_completion_tokens=500,
    )
    return response.choices[0].message.content.strip()


def answer_with_vision(
    client: Groq,
    model: str,
    question: str,
    context: str,
    image_paths: list[str],
    max_images: int = 3,
) -> str:
    """Answer a question using the retrieved text context plus the original images."""
    content = [
        {"type": "text", "text": ANSWER_PROMPT.format(context=context, question=question)}
    ]
    content.extend(_image_part(p) for p in image_paths[:max_images])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_completion_tokens=900,
    )
    return response.choices[0].message.content.strip()
