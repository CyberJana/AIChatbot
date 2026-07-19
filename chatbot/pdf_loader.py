"""PDF loading helpers for the AI FAQ Chatbot."""

from __future__ import annotations

from pathlib import Path

from PyPDF2 import PdfReader

from chatbot.utils import PDFPage, normalize_whitespace


class PDFLoader:
    """Load PDF files and extract cleaned text page by page."""

    def load_files(self, file_paths: list[Path]) -> list[PDFPage]:
        """Load many PDF files and combine all extracted pages into one list."""

        pages: list[PDFPage] = []

        for file_path in file_paths:
            pages.extend(self.load_file(file_path))

        if not pages:
            raise ValueError("No readable text was found in the selected PDF files.")

        return pages

    def load_file(self, file_path: Path) -> list[PDFPage]:
        """Load one PDF file and return each page that contains text."""

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported file type: {file_path.name}")

        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:
            raise ValueError(f"Could not open PDF file: {file_path.name}") from exc

        pages: list[PDFPage] = []

        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            clean_text = normalize_whitespace(raw_text)

            if clean_text:
                pages.append(
                    PDFPage(
                        source_name=file_path.name,
                        page_number=page_number,
                        text=clean_text,
                    )
                )

        if not pages:
            raise ValueError(
                f"The PDF '{file_path.name}' does not contain extractable text."
            )

        return pages
