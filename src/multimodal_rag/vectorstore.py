"""Embeddings and Pinecone plumbing."""

from __future__ import annotations

import time

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import Settings


def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def ensure_index(settings: Settings, dimension: int) -> None:
    """Create the serverless index if it does not exist yet, and wait until it is ready."""
    pc = Pinecone()  # reads PINECONE_API_KEY from the environment

    if pc.has_index(settings.pinecone_index):
        print(f"Pinecone index '{settings.pinecone_index}' already exists.")
        return

    print(f"Creating Pinecone index '{settings.pinecone_index}' (dimension {dimension})...")
    pc.create_index(
        name=settings.pinecone_index,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )
    while not pc.describe_index(settings.pinecone_index).status["ready"]:
        time.sleep(2)


def get_vectorstore(settings: Settings, embeddings: HuggingFaceEmbeddings) -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=settings.pinecone_index,
        embedding=embeddings,
        namespace=settings.pinecone_namespace,
    )


def index_documents(settings: Settings, documents: list[Document]) -> None:
    """Wipe the namespace and upload fresh documents so re-running never creates duplicates."""
    embeddings = get_embeddings(settings)
    dimension = len(embeddings.embed_query("dimension check"))
    ensure_index(settings, dimension)

    pc = Pinecone()
    index = pc.Index(settings.pinecone_index)
    try:
        index.delete(delete_all=True, namespace=settings.pinecone_namespace)
        print(f"Cleared namespace '{settings.pinecone_namespace}'.")
    except Exception as error:  # an empty or missing namespace can raise
        print(f"Namespace was probably empty, continuing ({error}).")

    get_vectorstore(settings, embeddings).add_documents(documents)
    print(f"Uploaded {len(documents)} documents.")
