"""Unit tests for the AI FAQ Chatbot project."""

from __future__ import annotations

from pathlib import Path

import app

from chatbot.chat_engine import ChatEngine
from chatbot.text_splitter import TextSplitter
from chatbot.utils import PDFPage, RetrievedChunk, TextChunk, format_source_label


class FakeRetriever:
    """Simple retriever used to test chat answers without external services."""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        """Store the retrieval results that should be returned in tests."""

        self.results = results

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Return the prepared results for the current test case."""

        _ = query
        return self.results[:top_k]


def test_text_splitter_creates_overlapping_chunks() -> None:
    """The splitter should create readable overlapping chunks in word space."""

    words = [f"word{index}" for index in range(12)]
    pages = [
        PDFPage(
            source_name="faq.pdf",
            page_number=1,
            text=" ".join(words),
        )
    ]

    splitter = TextSplitter(chunk_size=5, chunk_overlap=2)
    chunks = splitter.split_pages(pages)

    assert len(chunks) == 4
    assert chunks[0].text == "word0 word1 word2 word3 word4"
    assert chunks[1].text == "word3 word4 word5 word6 word7"
    assert chunks[0].chunk_id == "faq-pdf-p1-c1"


def test_chat_engine_uses_retrieved_sentences_in_answer() -> None:
    """The answer should be grounded in the retrieved FAQ content."""

    results = [
        RetrievedChunk(
            chunk=TextChunk(
                chunk_id="refund-p1-c1",
                source_name="company_faq.pdf",
                page_number=1,
                text=(
                    "Refund requests are accepted within 14 days of purchase. "
                    "Customers must include the original order number."
                ),
            ),
            score=0.92,
        ),
        RetrievedChunk(
            chunk=TextChunk(
                chunk_id="refund-p2-c1",
                source_name="company_faq.pdf",
                page_number=2,
                text="Refunds are processed within five business days after approval.",
            ),
            score=0.88,
        ),
    ]

    engine = ChatEngine(retriever=FakeRetriever(results), min_score=0.2)
    answer = engine.answer_question("What is the refund policy?")

    assert "14 days" in answer.answer
    assert "order number" in answer.answer
    assert len(answer.sources) == 2


def test_chat_engine_returns_fallback_message_when_no_results_exist() -> None:
    """The chatbot should fail gracefully when nothing relevant is found."""

    engine = ChatEngine(retriever=FakeRetriever([]), min_score=0.2)
    answer = engine.answer_question("Do you offer weekend support?")

    assert "could not find a relevant answer" in answer.answer.lower()
    assert answer.sources == []


def test_format_source_label_includes_file_and_page() -> None:
    """Source labels should stay easy for users to understand."""

    chunk = TextChunk(
        chunk_id="support-p3-c1",
        source_name="support_guide.pdf",
        page_number=3,
        text="Support is available Monday to Friday.",
    )

    assert format_source_label(chunk) == "support_guide.pdf - Page 3"


def test_get_upload_directory_is_repo_relative(monkeypatch, tmp_path: Path) -> None:
    """The uploads directory should not depend on the current working directory."""

    monkeypatch.chdir(tmp_path)

    assert app.get_upload_directory() == Path(app.__file__).resolve().parent / "data" / "uploads"
