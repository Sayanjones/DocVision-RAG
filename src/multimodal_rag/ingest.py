"""PDF -> LangChain Documents. One Document per page of text, per table, and per unique image."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pymupdf
from langchain_core.documents import Document

# (image_path, page_number) -> text description
Summarizer = Callable[[Path, int], str]


def extract_documents(
    pdf_path: Path,
    image_dir: Path,
    summarize: Summarizer,
) -> list[Document]:
    pdf_path = Path(pdf_path)
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    seen_xrefs: set[int] = set()

    with pymupdf.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            print(f"Processing page {page_number}...")

            # 1. Text
            text = page.get_text("text").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": page_number,
                            "modality": "text",
                            "source": pdf_path.name,
                        },
                    )
                )

            # 2. Tables, kept as Markdown so rows and columns stay together
            try:
                for table_number, table in enumerate(page.find_tables().tables, start=1):
                    df = table.to_pandas()
                    if df.empty:
                        continue
                    documents.append(
                        Document(
                            page_content=df.to_markdown(index=False),
                            metadata={
                                "page": page_number,
                                "modality": "table",
                                "table_number": table_number,
                                "source": pdf_path.name,
                            },
                        )
                    )
            except Exception as error:  # table detection can fail on odd layouts
                print(f"  table extraction warning on page {page_number}: {error}")

            # 3. Images, charts and diagrams
            for image_number, info in enumerate(page.get_images(full=True), start=1):
                xref = info[0]
                if xref in seen_xrefs:  # the same embedded image can appear on several pages
                    continue
                seen_xrefs.add(xref)

                extracted = pdf.extract_image(xref)
                image_path = image_dir / f"page_{page_number}_image_{image_number}.{extracted['ext']}"
                image_path.write_bytes(extracted["image"])

                try:
                    summary = summarize(image_path, page_number)
                except Exception as error:
                    print(f"  vision warning on page {page_number}: {error}")
                    continue

                documents.append(
                    Document(
                        page_content=summary,
                        metadata={
                            "page": page_number,
                            "modality": "visual",
                            "image_path": str(image_path),
                            "source": pdf_path.name,
                        },
                    )
                )

    return documents
