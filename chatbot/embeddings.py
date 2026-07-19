"""Embedding generation for the AI FAQ Chatbot."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Wrap the SentenceTransformer model used by the chatbot."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load the sentence transformer model once for reuse."""

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Convert a list of texts into normalized embedding vectors."""

        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """Convert one user query into a single embedding vector."""

        if not query.strip():
            raise ValueError("The query cannot be empty.")

        return self.encode_texts([query])[0]
