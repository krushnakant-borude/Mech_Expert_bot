"""CLI: wipe the vector store and rebuild from `standards/`."""
from __future__ import annotations

import torch  # noqa: F401  (DLL bootstrap)
import sentence_transformers  # noqa: F401

from src.ingest import run, wipe


def main() -> None:
    print("Wiping existing vector store...")
    wipe()
    print("Re-ingesting all PDFs in 'standards/'...")
    stats = run(verbose=True)
    print(
        f"\nDone. files={stats['files_total']} "
        f"new={stats['files_new']} chunks_added={stats['chunks_added']}"
    )


if __name__ == "__main__":
    main()