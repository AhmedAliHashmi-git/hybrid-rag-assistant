"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from config import OLLAMA_MODEL
from store import index_exists

st.set_page_config(page_title="RAG Assistant", page_icon="📄")
st.title("📄 RAG Assistant")
st.caption(f"Hybrid search (vector + BM25) over your PDFs · answers by Ollama `{OLLAMA_MODEL}`")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve", 1, 10, 5)
    alpha = st.slider("Vector weight (0 = BM25 only, 1 = vector only)", 0.0, 1.0, 0.5, 0.05)
    if st.button("Rebuild index"):
        with st.spinner("Loading PDFs and building embeddings..."):
            from embed_and_store_02 import build_index
            n = build_index()
        st.success(f"Index rebuilt with {n} chunks.")

if not index_exists():
    with st.spinner("First run: building the index from PDFs in data/ ..."):
        from embed_and_store_02 import build_index
        build_index()

query = st.text_input("Ask a question about your documents")

if query:
    from llm_answer_07 import ask
    try:
        with st.spinner("Thinking..."):
            response = ask(query, top_k=top_k, alpha=alpha)
    except Exception as e:
        st.error(str(e))
    else:
        st.subheader("Answer")
        st.write(response["answer"])
        st.subheader("Sources")
        for src in response["sources"]:
            st.markdown(f"- {src}")
        with st.expander("Retrieved chunks"):
            for c in response["chunks"]:
                st.markdown(f"**{c['citation']}** (score {c['score']:.3f})")
                st.text(c["text"])
