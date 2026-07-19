"""Shared filesystem paths for the AI FAQ Chatbot project."""

from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = APP_DIR / ".env"
UPLOAD_DIR = APP_DIR / "data" / "uploads"
