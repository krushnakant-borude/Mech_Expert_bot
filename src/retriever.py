"""Build and expose the RAG chain over the persisted Chroma store."""
from __future__ import annotations

import torch  # noqa: F401  (DLL bootstrap; see src/ingest.py)
import sentence_transformers  # noqa: F401

from typing import Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_DEVICE,
    EMBED_MODEL_NAME,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MEMORY_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_K,
    require_openai_key,
)
from src.prompts import build_qa_prompt, build_rephrase_prompt
from src.utils import format_sources


def _build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=_build_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def get_retriever():
    vs = get_vectorstore()
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K},
    )


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        page_label = f"p.{int(page) + 1}" if page is not None else "p.?"
        parts.append(f"[Source: {src}, {page_label}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


class InMemoryHistory:
    """Simple session-keyed message history (single-user local app)."""

    def __init__(self) -> None:
        self._store: Dict[str, List[BaseMessage]] = {}

    def get(self, session_id: str) -> List[BaseMessage]:
        return self._store.setdefault(session_id, [])

    def append(self, session_id: str, role: str, content: str) -> None:
        msg = AIMessage(content=content) if role == "ai" else HumanMessage(content=content)
        msgs = self._store.setdefault(session_id, [])
        msgs.append(msg)
        self._store[session_id] = msgs[-2 * MEMORY_K :]

    def clear(self, session_id: str) -> None:
        self._store[session_id] = []


_HISTORY = InMemoryHistory()


def get_session_history(session_id: str) -> List[BaseMessage]:
    return _HISTORY.get(session_id)


def clear_session(session_id: str = "default") -> None:
    _HISTORY.clear(session_id)


def build_chain() -> Runnable:
    """Build the conversational RAG chain (history-aware)."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=require_openai_key(),
    )
    retriever = get_retriever()

    rephrase_llm = build_rephrase_prompt() | llm | StrOutputParser()

    def _retrieval_step(x: Dict) -> Dict:
        docs = retriever.invoke(x["standalone_question"])
        return {
            "context": _format_docs(docs),
            "sources": format_sources(docs),
            "chat_history": x["chat_history"],
            "question": x["standalone_question"],
        }

    # Parallel branch 1: rewrite question; forward chat_history verbatim.
    parallel_rephrase = RunnablePassthrough.assign(
        standalone_question=RunnableLambda(
            lambda x: rephrase_llm.invoke(
                {"input": x["input"], "chat_history": x["chat_history"]}
            )
        )
    )

    # Parallel branch 2: from (standalone_question, chat_history) -> context, sources.
    retrieve_step = RunnableLambda(_retrieval_step)

    qa_chain = build_qa_prompt() | llm | StrOutputParser()

    chain: Runnable = (
        parallel_rephrase
        | retrieve_step
        | RunnablePassthrough.assign(
            answer=RunnableLambda(
                lambda x: qa_chain.invoke(
                    {
                        "context": x["context"],
                        "question": x["question"],
                        "chat_history": x["chat_history"],
                    }
                )
            )
        )
    )
    return chain


def answer(question: str, session_id: str = "default") -> Tuple[str, List[dict]]:
    """One-shot answer (used by CLI smoke tests)."""
    chain = build_chain()
    result = chain.invoke(
        {"input": question, "chat_history": _HISTORY.get(session_id)}
    )
    _HISTORY.append(session_id, "human", question)
    _HISTORY.append(session_id, "ai", result["answer"])
    return result["answer"], result["sources"]


def stream_answer(question: str, session_id: str = "default"):
    """Generator that yields incremental answer tokens and final sources.

    We can't easily stream the whole chain end-to-end while still returning
    sources, so we: (1) retrieve once, (2) stream the QA LLM call.
    """
    import langchain_chroma

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=require_openai_key(),
    )
    retriever = get_retriever()
    rephrase_llm = build_rephrase_prompt() | llm | StrOutputParser()
    qa_llm = build_qa_prompt() | llm | StrOutputParser()

    history = _HISTORY.get(session_id)
    standalone = rephrase_llm.invoke({"input": question, "chat_history": history})
    docs = retriever.invoke(standalone)
    context = _format_docs(docs)
    sources = format_sources(docs)

    full = []
    for token in qa_llm.stream(
        {"context": context, "question": standalone, "chat_history": history}
    ):
        full.append(token)
        yield token

    _HISTORY.append(session_id, "human", question)
    _HISTORY.append(session_id, "ai", "".join(full))
    yield {"__sources__": sources}