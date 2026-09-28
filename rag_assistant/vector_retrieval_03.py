"""Step 3: semantic (vector) search."""
import numpy as np

from store import get_model, load_index


def search(query, top_k=3):
    embeddings, chunks = load_index()
    query_embedding = get_model().encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_embedding  # cosine similarity (vectors are normalized)
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[-top_k:][::-1]

    return [
        {
            "text": chunks[i]["text"],
            "score": float(scores[i]),
            "source": chunks[i]["source"],
            "page": chunks[i]["page"],
        }
        for i in top_indices
    ]


if __name__ == "__main__":
    query = "candidate name in the CV"
    results = search(query, top_k=3)
    print(f"\nQuery: {query}\nTop {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. Score: {r['score']:.4f}")
        print(f"   Source: {r['source']} (page {r['page']})")
        print(f"   Text: {r['text'][:200]}...\n")
