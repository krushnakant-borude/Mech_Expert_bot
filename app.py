"""Streamlit chat UI for the Alstom Standards RAG chatbot."""
from __future__ import annotations

import streamlit as st

# DLL bootstrap (must run before langchain imports).
import torch  # noqa: F401
import sentence_transformers  # noqa: F401

from src.config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR
from src.ingest import run as run_ingest, wipe as wipe_index
from src.retriever import (
    _HISTORY,
    build_chain,
    clear_session,
    stream_answer,
)


def _index_stats() -> dict:
    import json
    from pathlib import Path

    state_file = Path("vectorstore") / ".ingested.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    return {
        "files": len(state),
        "names": sorted(state.keys()),
    }


@st.cache_resource(show_spinner=False)
def _get_chain():
    return build_chain()


def main() -> None:
    st.set_page_config(
        page_title="Alstom Standards Assistant",
        page_icon="🚆",
        layout="wide",
    )

    st.title("🚆 Alstom Standards Assistant")
    st.caption(
        "Chat with your Alstom train design & development standards. "
        "Answers are grounded in the PDFs under the `standards/` folder and "
        "include citations like `[Source: filename, p.X]`."
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("Index")
        stats = _index_stats()
        st.metric("Documents indexed", stats["files"])
        with st.expander("Indexed files", expanded=False):
            for n in stats["names"]:
                st.write(f"- `{n}`")
            if not stats["names"]:
                st.info("No PDFs indexed yet.")

        st.divider()
        st.header("Maintenance")
        if st.button("Rebuild index", use_container_width=True):
            with st.spinner("Re-ingesting standards..."):
                wipe_index()
                result = run_ingest(verbose=False)
            st.success(
                f"Indexed {result['files_new']}/{result['files_total']} files "
                f"({result['chunks_added']} new chunks)."
            )
            _get_chain.clear()
            st.rerun()

        if st.button("Clear chat", use_container_width=True):
            clear_session("default")
            st.session_state.messages = []
            st.rerun()

        st.divider()
        with st.expander("Standards folder", expanded=False):
            st.code(str(DATA_DIR))
            st.caption(
                "Drop new PDFs in this folder, then click 'Rebuild index'."
            )

    # --- Welcome ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm your Alstom standards assistant. Ask me anything "
                    "about train design, tolerances, structural requirements, "
                    "inspection methods, etc. I'll cite the source document and page."
                ),
                "sources": [],
            }
        )

    # --- Render chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(
                    f"Sources ({len(msg['sources'])})", expanded=False
                ):
                    for s in msg["sources"]:
                        st.markdown(
                            f"- **`{s['source']}`**, p.{s['page']} — {s['snippet']}"
                        )

    # --- Chat input ---
    prompt = st.chat_input("Ask about Alstom train standards...")
    if not prompt:
        return

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Generate ---
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        sources: list = []
        try:
            for chunk in stream_answer(prompt, session_id="default"):
                if isinstance(chunk, dict) and "__sources__" in chunk:
                    sources = chunk["__sources__"]
                else:
                    full_text += chunk
                    placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)
        except Exception as e:  # surface errors clearly in the UI
            full_text = f"**Error:** {e}"
            placeholder.error(full_text)
            sources = []

        if sources:
            with st.expander(f"Sources ({len(sources)})", expanded=False):
                for s in sources:
                    st.markdown(
                        f"- **`{s['source']}`**, p.{s['page']} — {s['snippet']}"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": full_text, "sources": sources}
    )


if __name__ == "__main__":
    main()