"""Command line entry point.

    python main.py ingest
    python main.py ask "Which quarter had the highest revenue?"
    python main.py demo
    python main.py chat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from multimodal_rag.config import Settings  # noqa: E402
from multimodal_rag.utils import clean_page  # noqa: E402

DEMO_QUESTIONS = [
    "What does NovaCore Systems do and where is the company headquartered?",
    "Which region had the highest year-over-year revenue growth?",
    "According to the revenue graph, which quarter had the highest revenue?",
    "According to the supply-chain diagram, what is the critical quality-control point?",
    "How did average support resolution time change from January to August?",
    "What percentage of electricity at the Penang facility came from solar?",
    "Give me a short FY2026 performance summary. Include total revenue, the "
    "fastest-growing region, the support-resolution improvement, and the solar "
    "share at the Penang facility.",
]


def cmd_ingest(args: argparse.Namespace) -> None:
    from groq import Groq

    from multimodal_rag.ingest import extract_documents
    from multimodal_rag.vectorstore import index_documents
    from multimodal_rag.vision import summarize_visual

    settings = Settings()
    settings.require_keys()
    pdf_path = Path(args.pdf) if args.pdf else settings.pdf_path
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")

    client = Groq()
    documents = extract_documents(
        pdf_path,
        settings.image_dir,
        summarize=lambda path, page: summarize_visual(client, settings.vision_model, path, page),
    )

    counts: dict[str, int] = {}
    for doc in documents:
        counts[doc.metadata["modality"]] = counts.get(doc.metadata["modality"], 0) + 1
    print(f"\nExtracted {len(documents)} documents: {counts}")

    index_documents(settings, documents)


def _print_result(question: str, result) -> None:
    print(f"\nQ: {question}")
    print(f"A: {result.answer}")
    print(f"\n   route: {result.route}")
    print("   sources:")
    for i, doc in enumerate(result.documents, start=1):
        print(f"     {i}. page {clean_page(doc.metadata.get('page'))} | {doc.metadata.get('modality')}")
    for path in result.image_paths:
        print(f"   image: {path}")


def cmd_ask(args: argparse.Namespace) -> None:
    from multimodal_rag import MultimodalRAG

    rag = MultimodalRAG()
    _print_result(args.question, rag.ask(args.question))


def cmd_demo(_: argparse.Namespace) -> None:
    from multimodal_rag import MultimodalRAG

    rag = MultimodalRAG()
    for question in DEMO_QUESTIONS:
        _print_result(question, rag.ask(question))
        print("\n" + "-" * 80)


def cmd_chat(_: argparse.Namespace) -> None:
    from multimodal_rag import MultimodalRAG

    rag = MultimodalRAG()
    print("Ask about the report. Empty line or Ctrl+C to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            break
        _print_result(question, rag.ask(question))


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal RAG over a PDF.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Extract the PDF and load it into Pinecone.")
    p.add_argument("--pdf", help="Path to a PDF (defaults to the NovaCore report).")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="Ask one question.")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    sub.add_parser("demo", help="Run the sample questions.").set_defaults(func=cmd_demo)
    sub.add_parser("chat", help="Interactive question loop.").set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
