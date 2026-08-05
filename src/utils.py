"""Utility helpers for hashing, formatting, and history management."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document


def hash_file(path: Path) -> str:
    """Stable content hash for dedup of unchanged PDFs."""
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode("utf-8"))
    h.update(str(path.stat().st_mtime_ns).encode("utf-8"))
    h.update(str(path.stat().st_size).encode("utf-8"))
    return h.hexdigest()


_WHITESPACE_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def format_sources(docs: Iterable[Document], snippet_chars: int = 220) -> List[dict]:
    """Format retrieved docs for UI display."""
    out: List[dict] = []
    seen = set()
    for d in docs:
        meta = d.metadata or {}
        source = Path(meta.get("source", "unknown")).name
        page = meta.get("page")
        if page is not None:
            page = int(page) + 1
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        snippet = _clean(d.page_content)[:snippet_chars]
        if len(_clean(d.page_content)) > snippet_chars:
            snippet = snippet.rsplit(" ", 1)[0] + "…"
        out.append({"source": source, "page": page, "snippet": snippet})
    return out


def trim_history(messages: List, k: int) -> List:
    """Keep only the last k message pairs (user/assistant)."""
    if k <= 0:
        return []
    return list(messages[-2 * k :])