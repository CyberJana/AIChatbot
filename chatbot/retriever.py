"""Retriever logic for the AI FAQ Chatbot."""

from __future__ import annotations

from chatbot.embeddings import EmbeddingService
from chatbot.utils import RetrievedChunk, TextChunk
from chatbot.vector_store import FAISSVectorStore


class FAQRetriever:
    """Build and query the PDF knowledge base."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        """Store the services needed to embed text and search it later."""

        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def build_index(self, chunks: list[TextChunk]) -> None:
        """Create a fresh FAISS index from the provided text chunks."""

        if not chunks:
            raise ValueError("At least one chunk is required to build the index.")

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.encode_texts(texts)

        self.vector_store.clear()
        self.vector_store.add(embeddings=embeddings, chunks=chunks)

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Return the most relevant chunks for a user question."""

        if not query.strip():
            raise ValueError("The question cannot be empty.")

        query_embedding = self.embedding_service.encode_query(query)
        return self.vector_store.search(query_embedding=query_embedding, top_k=top_k)
