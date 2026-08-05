"""Ingest Alstom standards PDFs into a persistent ChromaDB vector store.

Idempotent: skips files whose (path, mtime, size) hash is already in the
ingest state file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

# Bootstrap: pre-load torch + sentence_transformers so that the eager
# `import sentence_transformers` inside langchain_text_splitters does not
# trigger a fresh c10.dll load (which fails on this Python 3.12 + CPU torch
# combo with WinError 1114).
import torch  # noqa: F401
import sentence_transformers  # noqa: F401

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_DEVICE,
    EMBED_MODEL_NAME,
    INGEST_STATE_FILE,
)
from src.utils import hash_file


def _load_state() -> Dict[str, str]:
    if INGEST_STATE_FILE.exists():
        return json.loads(INGEST_STATE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_state(state: Dict[str, str]) -> None:
    INGEST_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    INGEST_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _discover_pdfs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.pdf") if p.is_file()])


def _build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_and_split(pdf_path: Path, splitter: RecursiveCharacterTextSplitter) -> List:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    for page in pages:
        page.metadata["source"] = pdf_path.name
        page.metadata["doc_type"] = "alstom-standard"
    return splitter.split_documents(pages)


def run(verbose: bool = True) -> Dict[str, int]:
    """Ingest all PDFs under DATA_DIR. Returns stats."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Standards folder not found: {DATA_DIR}. "
            "Create it and add Alstom standard PDFs."
        )

    pdfs = _discover_pdfs(DATA_DIR)
    state = _load_state()
    new_chunks: List = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    files_total = len(pdfs)
    files_new = 0

    for pdf in pdfs:
        h = hash_file(pdf)
        if state.get(str(pdf.name)) == h:
            if verbose:
                print(f"  skip (unchanged): {pdf.name}")
            continue
        if verbose:
            print(f"  ingest: {pdf.name}")
        chunks = _load_and_split(pdf, splitter)
        new_chunks.extend(chunks)
        state[str(pdf.name)] = h
        files_new += 1

    chunks_added = 0
    if new_chunks:
        embeddings = _build_embeddings()
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        Chroma.from_documents(
            documents=new_chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_name=COLLECTION_NAME,
        )
        chunks_added = len(new_chunks)

    _save_state(state)

    stats = {
        "files_total": files_total,
        "files_new": files_new,
        "chunks_added": chunks_added,
    }
    if verbose:
        print(
            f"Done. total={files_total} new={files_new} chunks_added={chunks_added}"
        )
    return stats


def wipe() -> None:
    """Remove the persisted Chroma store and ingest state."""
    import shutil

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    if INGEST_STATE_FILE.exists():
        INGEST_STATE_FILE.unlink()


if __name__ == "__main__":
    run()