# RAG Assistant

Ask questions about your PDFs. Pipeline: PDF load → chunk → embed → hybrid retrieval (vector + BM25) → citations → local LLM (Ollama).

## Setup
1. `python -m venv venv && source venv/bin/activate` (Windows: `venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. Install [Ollama](https://ollama.com), then: `ollama pull llama3`
4. Put your PDFs in `data/`
5. (Optional) `cp .env.example .env` to change models

## Run
- Web app: `streamlit run app.py` (builds the index automatically on first run)
- Or step by step:
  - `python embed_and_store_02.py` – build the index (rerun when PDFs change)
  - `python hybrid_fusion_05.py` – test retrieval
  - `python llm_answer_07.py "your question"` – full answer with citations

## Files
| File | Purpose |
|---|---|
| config.py | paths, models, chunk settings |
| store.py | loads/caches embeddings, chunks, embedding model |
| load_and_chunk_01.py | read PDFs, split into chunks |
| embed_and_store_02.py | embed chunks, save index |
| vector_retrieval_03.py | semantic search |
| bm25_retrieval_04.py | keyword search |
| hybrid_fusion_05.py | blended search (`alpha` weight) |
| citations_06.py | `[file.pdf, page N]` citations |
| llm_answer_07.py | prompt + Ollama answer |
| app.py | Streamlit UI |


# Hybrid RAG Assistant for Document Intelligence

An end-to-end, privacy-preserving Retrieval-Augmented Generation (RAG) assistant that indexes local PDF documents and answers complex questions using hybrid retrieval (Dense Semantic Embeddings + Sparse BM25) and local LLM inference via Ollama.

---

## Architecture & Pipeline

```text
PDF Documents (data/)
         │
         ▼
[1] Document Loading & Text Extraction (PyPDFLoader)
         │
         ▼
[2] Recursive Chunking (500 tokens, 50 overlap)
         │
         ▼
[3] Embeddings & Index Storage (all-MiniLM-L6-v2 -> chunk_embeddings.npy & chunks.json)
         │
         ├──────────────────────────────────────────┐
         ▼                                          ▼
[4] Dense Vector Search (Cosine Similarity)   [5] Lexical Search (BM25Okapi)
         │                                          │
         └────────────────────┬─────────────────────┘
                              ▼
           [6] Hybrid Fusion (Score Normalization & α-Weighting)
                              │
                              ▼
           [7] Citation Extraction ([filename, page N])
                              │
                              ▼
           [8] Local LLM Inference (Llama 3 via Ollama)
                              │
                              ▼
             Interactive Web UI (Streamlit) / CLI
