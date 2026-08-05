"""Central configuration for the Alstom Standards RAG chatbot."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR: Path = PROJECT_ROOT / "standards"
CHROMA_DIR: Path = PROJECT_ROOT / "vectorstore" / "chroma"
INGEST_STATE_FILE: Path = PROJECT_ROOT / "vectorstore" / ".ingested.json"

COLLECTION_NAME: str = "alstom_standards"

EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DEVICE: str = os.getenv("EMBED_DEVICE", "cpu")

LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150

RETRIEVER_K: int = 6
RETRIEVER_FETCH_K: int = 20

MEMORY_K: int = 6


def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Add it to your .env file at the project root."
        )
    return key