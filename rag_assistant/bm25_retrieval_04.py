"""Step 4: keyword (BM25) search."""
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from config import tokenize
from store import load_index


@lru_cache(maxsize=1)
def get_bm25():
    _, chunks = load_index()
    return BM25Okapi([tokenize(c["text"]) for c in chunks])


def bm25_scores(query):
    return np.asarray(get_bm25().get_scores(tokenize(query)), dtype=float)


def bm25_search(query, top_k=5):
    _, chunks = load_index()
    scores = bm25_scores(query)
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
    query = "Which Institute Ahmed Ali Hashmi did his bachelors"
    print(f"\nQuery: {query}")
    for i, r in enumerate(bm25_search(query, top_k=5), 1):
        print(f"{i}. Score: {r['score']:.4f}")
        print(f"   Source: {r['source']} (page {r['page']})")
        print(f"   Text: {r['text'][:300]}...\n")
