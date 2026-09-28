"""Step 6: attach readable citations to retrieved chunks."""
import os

from hybrid_fusion_05 import hybrid_search


def format_citation(source, page):
    """'data/CV.pdf' + page 0 -> '[CV.pdf, page 1]' (PDF pages are 0-indexed internally)."""
    filename = os.path.basename(str(source).replace("\\", "/"))
    page_number = page + 1 if isinstance(page, int) else page
    return f"[{filename}, page {page_number}]"


def get_answer_with_citations(query, top_k=5, alpha=0.5):
    results = hybrid_search(query, top_k=top_k, alpha=alpha)
    chunks = [
        {
            "text": r["text"],
            "citation": format_citation(r["source"], r["page"]),
            "score": r["final_score"],
        }
        for r in results
    ]
    unique_sources = sorted({c["citation"] for c in chunks})
    return {"chunks": chunks, "unique_sources": unique_sources}


if __name__ == "__main__":
    query = "Which Institute did Ahmed Ali Hashmi do his bachelors from"
    result = get_answer_with_citations(query, top_k=5)
    print(f"\nQuery: {query}\n")
    for i, c in enumerate(result["chunks"], 1):
        print(f"{i}. {c['citation']}  (score: {c['score']:.4f})")
        print(f"   {c['text'][:200]}...\n")
    print("--- Unique sources used ---")
    for src in result["unique_sources"]:
        print(f"  {src}")
