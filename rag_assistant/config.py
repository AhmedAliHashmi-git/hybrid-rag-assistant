"""Shared settings and helpers used by every step of the RAG pipeline."""
import os
import re

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "chunk_embeddings.npy")
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def tokenize(text):
    """Lowercase word tokenizer that ignores punctuation (used by BM25)."""
    return re.findall(r"\w+", text.lower())
