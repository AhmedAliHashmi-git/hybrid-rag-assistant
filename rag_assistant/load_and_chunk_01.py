"""Step 1: load PDFs from data/ and split them into chunks."""
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_FOLDER


def load_pdfs(data_folder=DATA_FOLDER):
    """Loads all PDFs from a folder and returns a list of page documents."""
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    pdf_files = sorted(f for f in os.listdir(data_folder) if f.lower().endswith(".pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_folder}")

    all_documents = []
    for filename in pdf_files:
        file_path = os.path.join(data_folder, filename)
        try:
            pages = PyPDFLoader(file_path).load()
        except Exception as e:  # keep going if one PDF is broken
            print(f"WARNING: could not read {filename}: {e}")
            continue
        # skip pages with no extractable text (e.g. scanned images)
        all_documents.extend(p for p in pages if p.page_content.strip())

    if not all_documents:
        raise ValueError("No text could be extracted from the PDFs.")
    return all_documents


def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_pdfs()
    print(f"Loaded {len(docs)} pages")
    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    print(docs[0].page_content[:200])
    print(docs[0].metadata)
