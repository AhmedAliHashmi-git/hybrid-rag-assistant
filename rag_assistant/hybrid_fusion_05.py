"""Step 5: hybrid search = weighted blend of vector and BM25 scores."""
import numpy as np

from bm25_retrieval_04 import bm25_scores
from store import get_model, load_index


def normalize(scores):
    """Min-max scale to 0-1 so BM25 and cosine scores are comparable."""
    scores = np.asarray(scores, dtype=float)
    span = scores.max() - scores.min()
    if span == 0:
        return np.zeros_like(scores)
    return (scores - scores.min()) / span


def hybrid_search(query, top_k=5, alpha=0.5):
    """alpha=1.0 -> pure vector, alpha=0.0 -> pure BM25, 0.5 -> equal blend."""
    embeddings, chunks = load_index()

    query_embedding = get_model().encode([query], normalize_embeddings=True)[0]
    norm_vector = normalize(embeddings @ query_embedding)
    norm_bm25 = normalize(bm25_scores(query))

    final_scores = alpha * norm_vector + (1 - alpha) * norm_bm25
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(final_scores)[-top_k:][::-1]

    return [
        {
            "text": chunks[i]["text"],
            "final_score": float(final_scores[i]),
            "vector_score": float(norm_vector[i]),
            "bm25_score": float(norm_bm25[i]),
            "source": chunks[i]["source"],
            "page": chunks[i]["page"],
        }
        for i in top_indices
    ]


if __name__ == "__main__":
    query = "Which Institute did Ahmed Ali Hashmi do his bachelors from"
    print(f"\nQuery: {query}")
    for i, r in enumerate(hybrid_search(query, top_k=5), 1):
        print(f"{i}. Final: {r['final_score']:.4f}  (vector: {r['vector_score']:.4f}, bm25: {r['bm25_score']:.4f})")
        print(f"   Source: {r['source']} (page {r['page']})")
        print(f"   Text: {r['text'][:200]}...\n")
