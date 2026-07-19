"""Streamlit entry point for the AI FAQ Chatbot project."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import streamlit as st

from chatbot.chat_engine import ChatEngine
from chatbot.embeddings import EmbeddingService
from chatbot.paths import ENV_FILE, UPLOAD_DIR
from chatbot.pdf_loader import PDFLoader
from chatbot.retriever import FAQRetriever
from chatbot.text_splitter import TextSplitter
from chatbot.utils import ChatAnswer, ensure_directory, format_source_label
from chatbot.vector_store import FAISSVectorStore


load_dotenv(ENV_FILE)


@dataclass(slots=True)
class AppConfig:
    """Store user-configurable settings loaded from environment variables."""

    model_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_score: float


def load_config() -> AppConfig:
    """Load chatbot settings from the environment with safe defaults."""

    return AppConfig(
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "120")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "30")),
        top_k=int(os.getenv("TOP_K_RESULTS", "4")),
        min_score=float(os.getenv("MIN_SIMILARITY_SCORE", "0.30")),
    )


def initialize_session_state() -> None:
    """Create the Streamlit session keys used across reruns."""

    default_values = {
        "chat_engine": None,
        "retriever": None,
        "chat_history": [],
        "knowledge_ready": False,
        "processed_files": [],
        "chunk_count": 0,
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_or_create_chat_engine(config: AppConfig) -> ChatEngine:
    """Create the embedding, retrieval, and chat services once per session."""

    if st.session_state["chat_engine"] is None:
        embedding_service = EmbeddingService(model_name=config.model_name)
        vector_store = FAISSVectorStore(embedding_dimension=embedding_service.dimension)
        retriever = FAQRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
        st.session_state["retriever"] = retriever
        st.session_state["chat_engine"] = ChatEngine(
            retriever=retriever,
            min_score=config.min_score,
        )

    return st.session_state["chat_engine"]


def build_unique_file_path(upload_dir: Path, file_name: str) -> Path:
    """Create a non-conflicting file path for an uploaded PDF."""

    candidate_path = upload_dir / file_name
    stem = candidate_path.stem
    suffix = candidate_path.suffix
    counter = 1

    while candidate_path.exists():
        candidate_path = upload_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    return candidate_path


def save_uploaded_files(uploaded_files: list[Any], upload_dir: Path) -> list[Path]:
    """Save Streamlit uploaded files into the local data directory."""

    ensure_directory(upload_dir)
    saved_paths: list[Path] = []

    for uploaded_file in uploaded_files:
        destination = build_unique_file_path(
            upload_dir=upload_dir,
            file_name=uploaded_file.name,
        )
        destination.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(destination)

    return saved_paths


def process_documents(uploaded_files: list[Any], config: AppConfig) -> None:
    """Run the full PDF ingestion pipeline and store the search index."""

    if not uploaded_files:
        raise ValueError("Please upload at least one PDF file before processing.")

    saved_paths = save_uploaded_files(uploaded_files=uploaded_files, upload_dir=UPLOAD_DIR)

    pdf_loader = PDFLoader()
    pages = pdf_loader.load_files(saved_paths)

    splitter = TextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunks = splitter.split_pages(pages)

    get_or_create_chat_engine(config=config)
    retriever: FAQRetriever = st.session_state["retriever"]
    retriever.build_index(chunks=chunks)

    st.session_state["knowledge_ready"] = True
    st.session_state["processed_files"] = [path.name for path in saved_paths]
    st.session_state["chunk_count"] = len(chunks)
    st.session_state["chat_history"] = []


def clear_chat_history() -> None:
    """Remove all chat messages without deleting the processed documents."""

    st.session_state["chat_history"] = []


def serialize_answer(answer: ChatAnswer) -> dict[str, Any]:
    """Convert a ChatAnswer object into a Streamlit-friendly dictionary."""

    return {
        "role": "assistant",
        "content": answer.answer,
        "sources": [
            {
                "label": format_source_label(item.chunk),
                "score": item.score,
                "text": item.chunk.text,
            }
            for item in answer.sources
        ],
    }


def render_chat_history() -> None:
    """Render all previous chat messages stored in the user session."""

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Source text used"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source['label']}**  \n"
                            f"Similarity score: `{source['score']:.3f}`"
                        )
                        st.write(source["text"])


def ask_question(question: str, config: AppConfig) -> ChatAnswer:
    """Submit a user question to the chat engine and return the response."""

    if not st.session_state["knowledge_ready"]:
        raise ValueError("Please process your PDF files before asking questions.")

    chat_engine: ChatEngine = st.session_state["chat_engine"]
    return chat_engine.answer_question(question=question, top_k=config.top_k)


def render_sidebar(config: AppConfig) -> None:
    """Render sidebar controls for uploads, processing, and chat reset."""

    st.sidebar.header("Knowledge Base Setup")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload FAQ documents, manuals, policies, or any PDF content you want to search.",
    )

    if st.sidebar.button("Process PDFs", use_container_width=True):
        try:
            with st.spinner("Reading PDFs, splitting text, and building the FAISS index..."):
                process_documents(uploaded_files=uploaded_files or [], config=config)
            st.sidebar.success("Your documents are ready for questions.")
        except Exception as exc:
            st.sidebar.error(str(exc))

    if st.sidebar.button("Clear Chat", use_container_width=True):
        clear_chat_history()
        st.sidebar.success("Chat history cleared.")

    st.sidebar.divider()
    st.sidebar.subheader("Current Settings")
    st.sidebar.write(f"**Embedding model:** {config.model_name}")
    st.sidebar.write(f"**Chunk size:** {config.chunk_size} words")
    st.sidebar.write(f"**Chunk overlap:** {config.chunk_overlap} words")
    st.sidebar.write(f"**Top results:** {config.top_k}")

    if st.session_state["knowledge_ready"]:
        st.sidebar.divider()
        st.sidebar.subheader("Knowledge Base Status")
        st.sidebar.write(
            f"**Documents loaded:** {len(st.session_state['processed_files'])}"
        )
        st.sidebar.write(f"**Chunks indexed:** {st.session_state['chunk_count']}")
        st.sidebar.write(
            "**Files:** " + ", ".join(st.session_state["processed_files"])
        )

def main() -> None:
    """Render the full Streamlit application."""

    st.set_page_config(
        page_title="AI FAQ Chatbot",
        page_icon="🤖",
        layout="wide",
    )

    config = load_config()
    initialize_session_state()
    render_sidebar(config=config)

    st.title("🤖 AI FAQ Chatbot")
    st.caption(
        "Ask questions about your uploaded PDFs using semantic search powered by Sentence Transformers and FAISS."
    )

    if not st.session_state["knowledge_ready"]:
        st.info(
            "Upload one or more PDF files in the sidebar, click **Process PDFs**, and then start chatting."
        )

    render_chat_history()

    question = st.chat_input("Ask a question about the uploaded PDFs...")
    if not question:
        return

    st.session_state["chat_history"].append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Searching the most relevant PDF content..."):
            answer = ask_question(question=question, config=config)
    except Exception as exc:
        with st.chat_message("assistant"):
            st.error(str(exc))
        st.session_state["chat_history"].pop()
        return

    assistant_message = serialize_answer(answer=answer)
    st.session_state["chat_history"].append(assistant_message)

    with st.chat_message("assistant"):
        st.markdown(assistant_message["content"])
        if assistant_message["sources"]:
            with st.expander("Source text used"):
                for source in assistant_message["sources"]:
                    st.markdown(
                        f"**{source['label']}**  \n"
                        f"Similarity score: `{source['score']:.3f}`"
                    )
                    st.write(source["text"])


if __name__ == "__main__":
    main()
