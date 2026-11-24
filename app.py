"""Streamlit UI for the Document Q&A Assistant (RAG).

Run with:  streamlit run app.py
"""

import os

import streamlit as st

from config import APP_TITLE, CHUNK_OVERLAP, CHUNK_SIZE, TOP_K
from embeddings import get_embedding_provider
from qa import RAGAssistant
from vector_store import VectorStore


@st.cache_resource
def get_assistant() -> RAGAssistant:
    provider = get_embedding_provider()
    # Try the persistent ChromaDB store; fall back to in-memory if the
    # package is not installed (vector_store handles that internally).
    store = VectorStore(use_chroma=True)
    return RAGAssistant(provider=provider, store=store)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄")
    st.title(f"📄 {APP_TITLE}")
    st.write(
        "Upload PDF or text documents, then ask questions. "
        "Every answer comes only from your documents and shows its source page."
    )

    assistant = get_assistant()
    st.sidebar.header("Settings")
    chunk_size = st.sidebar.number_input("Chunk size", value=CHUNK_SIZE, min_value=100, step=100)
    chunk_overlap = st.sidebar.number_input("Chunk overlap", value=CHUNK_OVERLAP, min_value=0, step=50)
    top_k = st.sidebar.number_input("Chunks to retrieve (top-k)", value=TOP_K, min_value=1, max_value=10)
    assistant.top_k = int(top_k)

    mode = assistant.provider.name
    if mode == "openai":
        st.sidebar.success("Embeddings + answers: OpenAI")
    else:
        st.sidebar.warning(
            "Offline mode: local hash embeddings + extractive answers. "
            "Set OPENAI_API_KEY for OpenAI embeddings and generated answers."
        )
    st.sidebar.info(f"Vector store: {assistant.store.backend} — {assistant.store.count()} chunks indexed")
    if st.sidebar.button("Clear indexed documents"):
        assistant.store.clear()
        st.sidebar.success("Cleared.")

    st.header("1. Upload documents")
    uploads = st.file_uploader(
        "PDF or TXT files", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if uploads and st.button("Index uploaded documents"):
        total = 0
        os.makedirs("uploads", exist_ok=True)
        for uploaded in uploads:
            path = os.path.join("uploads", uploaded.name)
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            total += assistant.ingest_file(path, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
        st.success(f"Indexed {total} chunks from {len(uploads)} document(s).")

    st.header("2. Ask a question")
    question = st.text_input("Your question", placeholder="e.g. What is the default chunk size?")
    if st.button("Ask") and question.strip():
        with st.spinner("Retrieving relevant chunks…"):
            answer = assistant.ask(question)
        st.subheader("Answer")
        st.write(answer.text)
        if answer.grounded:
            st.caption(f"Mode: {answer.mode} — grounded in retrieved text")
            with st.expander("Sources"):
                for chunk in answer.sources:
                    st.markdown(f"**{chunk.citation()}**")
                    st.write(chunk.text)
        else:
            st.caption("No supporting text was found in your documents.")


if __name__ == "__main__":
    main()
