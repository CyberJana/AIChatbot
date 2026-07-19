"""Utility helpers and shared data structures for the chatbot project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


@dataclass(slots=True)
class PDFPage:
    """Stores the extracted text for one page from a PDF file."""

    source_name: str
    page_number: int
    text: str


@dataclass(slots=True)
class TextChunk:
    """Represents one searchable chunk of text from a PDF page."""

    chunk_id: str
    source_name: str
    page_number: int
    text: str


@dataclass(slots=True)
class RetrievedChunk:
    """Stores a retrieved chunk together with its similarity score."""

    chunk: TextChunk
    score: float


@dataclass(slots=True)
class ChatAnswer:
    """Stores the chatbot answer and the chunks used to create it."""

    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)


def ensure_directory(path: Path) -> Path:
    """Create a directory when it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace so extracted PDF text is easier to process."""

    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(text: str) -> list[str]:
    """Split text into simple sentence-like pieces for answer generation."""

    clean_text = normalize_whitespace(text)
    if not clean_text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def extract_keywords(text: str) -> set[str]:
    """Extract lightweight keywords from a question for sentence matching."""

    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return {
        word
        for word in words
        if len(word) > 2 and word not in STOP_WORDS
    }


def make_chunk_id(source_name: str, page_number: int, chunk_index: int) -> str:
    """Create a stable chunk identifier that is easy to read in logs and tests."""

    safe_name = re.sub(r"[^a-zA-Z0-9]+", "-", source_name).strip("-").lower()
    return f"{safe_name}-p{page_number}-c{chunk_index}"


def format_source_label(chunk: TextChunk) -> str:
    """Create a user-friendly label for displaying a source chunk."""

    return f"{chunk.source_name} - Page {chunk.page_number}"
