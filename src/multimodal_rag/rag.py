"""Query-time pipeline: retrieve, pick a route, answer."""

from __future__ import annotations

from dataclasses import dataclass

from groq import Groq
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .config import Settings
from .utils import collect_visual_paths, format_context
from .vectorstore import get_embeddings, get_vectorstore
from .vision import answer_with_vision

TEXT_PROMPT = ChatPromptTemplate.from_template(
    """\
You are a helpful assistant answering questions about the
NovaCore Systems FY2026 company report.

Use ONLY the retrieved context below.

If the answer is not available in the context, say:
"I could not find that information in the report."

Mention page numbers when possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)


@dataclass
class RagResult:
    answer: str
    route: str  # "text" or "vision"
    documents: list[Document]
    image_paths: list[str]


class MultimodalRAG:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.settings.require_keys()

        self.groq = Groq()  # reads GROQ_API_KEY from the environment
        self.text_llm = ChatGroq(model=self.settings.text_model, temperature=0)
        self.text_chain = TEXT_PROMPT | self.text_llm | StrOutputParser()

        embeddings = get_embeddings(self.settings)
        self.retriever = get_vectorstore(self.settings, embeddings).as_retriever(
            search_kwargs={"k": self.settings.top_k}
        )

    def ask(self, question: str) -> RagResult:
        docs = self.retriever.invoke(question)
        context = format_context(docs)
        image_paths = collect_visual_paths(docs)

        if image_paths:
            answer = answer_with_vision(
                client=self.groq,
                model=self.settings.vision_model,
                question=question,
                context=context,
                image_paths=image_paths,
                max_images=self.settings.max_images_per_answer,
            )
            route = "vision"
        else:
            answer = self.text_chain.invoke({"context": context, "question": question})
            route = "text"

        return RagResult(answer=answer, route=route, documents=docs, image_paths=image_paths)
