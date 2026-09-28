"""Step 2: embed all chunks and save the index to disk."""
import json

import numpy as np

from config import CHUNKS_FILE, EMBEDDINGS_FILE
from load_and_chunk_01 import chunk_documents, load_pdfs


def build_index():
    documents = load_pdfs()
    print(f"Loaded {len(documents)} pages")

    chunks = chunk_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    from store import clear_cache, get_model
    model = get_model()
    texts = [c.page_content for c in chunks]
    # normalized vectors -> cosine similarity becomes a simple dot product
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"Created embeddings with shape: {embeddings.shape}")

    np.save(EMBEDDINGS_FILE, embeddings)
    records = [
        {
            "text": c.page_content,
            "source": c.metadata.get("source", "unknown"),
            "page": c.metadata.get("page", "unknown"),
        }
        for c in chunks
    ]
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    clear_cache()
    print("Saved embeddings and chunks successfully!")
    return len(chunks)


if __name__ == "__main__":
    build_index()
