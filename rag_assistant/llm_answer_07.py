"""Step 7: generate a grounded answer with a local Ollama LLM."""
from functools import lru_cache

from citations_06 import get_answer_with_citations
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


@lru_cache(maxsize=1)
def get_llm():
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


def build_prompt(query, chunks):
    context = "\n\n".join(f"[Source: {c['citation']}]\n{c['text']}" for c in chunks)
    return f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the provided documents."
Cite the source in brackets after your answer, like [filename, page N].

Context:
{context}

Question: {query}

Answer:"""


def ask(query, top_k=5, alpha=0.5):
    result = get_answer_with_citations(query, top_k=top_k, alpha=alpha)
    prompt = build_prompt(query, result["chunks"])
    try:
        answer = get_llm().invoke(prompt)
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Ollama model '{OLLAMA_MODEL}' at {OLLAMA_BASE_URL}. "
            f"Make sure Ollama is running and you ran `ollama pull {OLLAMA_MODEL}`. ({e})"
        ) from e
    return {
        "answer": answer.strip(),
        "sources": result["unique_sources"],
        "chunks": result["chunks"],
    }


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Which Institute did Ahmed Ali Hashmi do his bachelors from"
    response = ask(query)
    print(f"\nQuery: {query}\n\n--- Answer ---\n{response['answer']}\n\n--- Sources ---")
    for src in response["sources"]:
        print(f"  {src}")
