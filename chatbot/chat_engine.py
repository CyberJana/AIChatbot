"""Answer generation logic for the AI FAQ Chatbot."""

from __future__ import annotations

from typing import Protocol

from chatbot.utils import ChatAnswer, RetrievedChunk, extract_keywords, split_into_sentences


class RetrieverProtocol(Protocol):
    """Define the small retriever interface required by the chat engine."""

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Return relevant text chunks for a user question."""


class ChatEngine:
    """Generate grounded answers from retrieved PDF chunks."""

    def __init__(self, retriever: RetrieverProtocol, min_score: float = 0.30) -> None:
        """Store the retriever and the score threshold for confident matches."""

        self.retriever = retriever
        self.min_score = min_score

    def answer_question(self, question: str, top_k: int = 4) -> ChatAnswer:
        """Retrieve relevant chunks and build an answer from them."""

        if not question.strip():
            raise ValueError("Please enter a question before sending it.")

        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)

        if not retrieved_chunks:
            return ChatAnswer(
                question=question,
                answer=(
                    "I could not find a relevant answer in the uploaded PDFs. "
                    "Try rephrasing your question or upload more detailed documents."
                ),
                sources=[],
            )

        confident_chunks = [
            chunk for chunk in retrieved_chunks if chunk.score >= self.min_score
        ]
        answer_sources = confident_chunks or retrieved_chunks
        answer_text = self._compose_answer(question=question, results=answer_sources)

        if not confident_chunks:
            answer_text = (
                "I found related information, but the match confidence is low. "
                f"{answer_text}"
            )

        return ChatAnswer(
            question=question,
            answer=answer_text,
            sources=answer_sources,
        )

    def _compose_answer(self, question: str, results: list[RetrievedChunk]) -> str:
        """Build a short answer using the most relevant source sentences."""

        keywords = extract_keywords(question)
        selected_sentences: list[str] = []
        seen_sentences: set[str] = set()

        for result in results:
            sentences = split_into_sentences(result.chunk.text)
            matched_in_chunk = False

            for sentence in sentences:
                normalized_sentence = sentence.lower()
                contains_keyword = any(
                    keyword in normalized_sentence for keyword in keywords
                )

                if keywords and not contains_keyword:
                    continue

                if normalized_sentence in seen_sentences:
                    continue

                selected_sentences.append(sentence)
                seen_sentences.add(normalized_sentence)
                matched_in_chunk = True

                if len(selected_sentences) == 3:
                    break

            if matched_in_chunk:
                for sentence in sentences:
                    normalized_sentence = sentence.lower()

                    if normalized_sentence in seen_sentences:
                        continue

                    selected_sentences.append(sentence)
                    seen_sentences.add(normalized_sentence)

                    if len(selected_sentences) == 3:
                        break

            if len(selected_sentences) == 3:
                break

        if selected_sentences:
            return " ".join(selected_sentences)

        fallback_sentences: list[str] = []

        for result in results[:2]:
            sentences = split_into_sentences(result.chunk.text)
            fallback_sentences.append(sentences[0] if sentences else result.chunk.text)

        return " ".join(sentence for sentence in fallback_sentences if sentence.strip())
