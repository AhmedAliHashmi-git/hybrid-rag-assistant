"""Loads the saved index (embeddings + chunks) once and shares it."""
import json
import os
from functools import lru_cache

import numpy as np

from config import CHUNKS_FILE, EMBEDDINGS_FILE, EMBEDDING_MODEL


def index_exists():
    return os.path.exists(EMBEDDINGS_FILE) and os.path.exists(CHUNKS_FILE)


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def load_index():
    """Returns (embeddings, chunks). chunks = list of {text, source, page}."""
    if not index_exists():
        raise FileNotFoundError(
            "Index not found. Run `python embed_and_store_02.py` first "
            "(or click 'Rebuild index' in the app)."
        )
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if len(chunks) != len(embeddings):
        raise ValueError("Index is corrupted (chunks/embeddings mismatch). Rebuild it.")
    return embeddings, chunks


def clear_cache():
    load_index.cache_clear()
