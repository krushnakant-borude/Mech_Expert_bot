# 🚆 Alstom Standards Assistant

A local RAG (Retrieval-Augmented Generation) chatbot that lets anyone in your team
chat with your Alstom train design & development standards PDFs and get
grounded, cited answers.

## Stack

- **LLM**: OpenAI `gpt-4o-mini` (configurable via `LLM_MODEL` in `.env`)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (local, free, via HuggingFace)
- **Vector store**: ChromaDB (persistent on disk)
- **UI**: Streamlit chat
- **PDF loading**: `PyPDFLoader`

## Project layout

```
.
├── standards/                 # put Alstom standard PDFs here
├── vectorstore/chroma/        # persistent ChromaDB (auto-created)
├── src/
│   ├── config.py              # paths, models, env
│   ├── prompts.py             # system + rephrase prompts
│   ├── utils.py               # hashing, source formatting
│   ├── ingest.py              # PDF → chunks → embeddings → Chroma
│   └── retriever.py           # RAG chain (rephrase → retrieve → answer)
├── app.py                     # Streamlit chat UI
├── rebuild_index.py           # CLI: wipe + re-ingest
├── requirements.txt
└── .env                       # OPENAI_API_KEY=...
```

## Setup

```powershell
python -m pip install -r requirements.txt
```

Make sure your `.env` has:

```
OPENAI_API_KEY=sk-...
```

Optional overrides:

```
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1
EMBED_MODEL=BAAI/bge-small-en-v1.5
EMBED_DEVICE=cpu        # or 'cuda' if you have a GPU
```

## First-run / adding new standards

1. Drop one or more PDFs into the `standards/` folder.
2. Ingest them into Chroma:

   ```powershell
   python -m src.ingest
   ```

   (Or use the **Rebuild index** button in the Streamlit sidebar.)

The indexer is **idempotent** — only changed files (by path + mtime + size) are
re-ingested. Each PDF is chunked into ~1000-char pieces with 150-char overlap and
embedded with `bge-small-en-v1.5`. Retrieved chunks are kept diverse via MMR
(`k=6`, `fetch_k=20`).

## Run the chat UI

```powershell
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## What the assistant does

- Rewrites follow-up questions using chat history (last 6 turns) so retrieval
  works on standalone queries.
- Retrieves the top 6 most-relevant chunks via MMR.
- Forces the LLM to answer **only** from the retrieved context, in precise
  technical terminology.
- Requires inline citations in the format `[Source: filename, p.X]`.
- Admits when it doesn't know, rather than hallucinating.

Below each assistant turn, an expander lists the actual chunks it used, with
filename, page, and a snippet so engineers can verify back to the standard.

## CLI smoke test

```powershell
python -c "from src.retriever import answer; print(answer('What is ISO 2768 about?'))"
```

## Notes / known quirks

- The first run will download the `bge-small-en-v1.5` model (~33 MB) into your
  HF cache.
- On Windows + Python 3.12 + CPU torch, the chain pre-imports `torch` and
  `sentence_transformers` to dodge a `c10.dll` initialization race. This is
  benign; ignore the harmless `DeprecationWarning` from `langchain-community`.
- Scanned (image-only) PDFs are not supported yet (no OCR). Convert them to
  text-based PDFs first or open an issue to add OCR support.