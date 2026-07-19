"""FAISS vector storage for the AI FAQ Chatbot."""

from __future__ import annotations

import faiss
import numpy as np

from chatbot.utils import RetrievedChunk, TextChunk


class FAISSVectorStore:
    """Store embeddings in a FAISS index and search similar chunks."""

    def __init__(self, embedding_dimension: int) -> None:
        """Create an in-memory FAISS index for cosine similarity search."""

        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be greater than 0.")

        self.embedding_dimension = embedding_dimension
        self.index = faiss.IndexFlatIP(embedding_dimension)
        self._chunks: list[TextChunk] = []

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks currently stored in the index."""

        return len(self._chunks)

    def clear(self) -> None:
        """Remove all chunks and embeddings from the index."""

        self.index.reset()
        self._chunks.clear()

    def add(self, embeddings: np.ndarray, chunks: list[TextChunk]) -> None:
        """Add chunk embeddings and their metadata to the FAISS index."""

        if len(chunks) == 0:
            raise ValueError("At least one chunk is required.")

        if len(chunks) != len(embeddings):
            raise ValueError("The number of embeddings must match the number of chunks.")

        prepared_embeddings = np.asarray(embeddings, dtype="float32")
        self.index.add(prepared_embeddings)
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[RetrievedChunk]:
        """Search for the most similar chunks to a user query embedding."""

        if self.chunk_count == 0:
            raise ValueError("The vector store is empty. Process PDFs first.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_vector = np.asarray([query_embedding], dtype="float32")
        limited_top_k = min(top_k, self.chunk_count)
        scores, indices = self.index.search(query_vector, limited_top_k)

        results: list[RetrievedChunk] = []

        for score, chunk_index in zip(scores[0], indices[0]):
            if chunk_index < 0:
                continue

            results.append(
                RetrievedChunk(
                    chunk=self._chunks[int(chunk_index)],
                    score=float(score),
                )
            )

        return results
