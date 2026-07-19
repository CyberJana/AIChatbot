"""Text chunking utilities for the AI FAQ Chatbot."""

from __future__ import annotations

from chatbot.utils import PDFPage, TextChunk, make_chunk_id


class TextSplitter:
    """Split long PDF text into smaller overlapping chunks."""

    def __init__(self, chunk_size: int = 120, chunk_overlap: int = 30) -> None:
        """Store the chunking configuration used across the project."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(self, pages: list[PDFPage]) -> list[TextChunk]:
        """Split extracted PDF pages into searchable text chunks."""

        chunks: list[TextChunk] = []

        for page in pages:
            chunk_texts = self._split_text(page.text)

            for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
                chunks.append(
                    TextChunk(
                        chunk_id=make_chunk_id(
                            source_name=page.source_name,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                        ),
                        source_name=page.source_name,
                        page_number=page.page_number,
                        text=chunk_text,
                    )
                )

        if not chunks:
            raise ValueError("No chunks were created from the uploaded PDFs.")

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split one block of text into overlapping word-based chunks."""

        words = text.split()
        if not words:
            return []

        step_size = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []

        for start_index in range(0, len(words), step_size):
            end_index = start_index + self.chunk_size
            chunk_words = words[start_index:end_index]

            if chunk_words:
                chunks.append(" ".join(chunk_words))

            if end_index >= len(words):
                break

        return chunks
