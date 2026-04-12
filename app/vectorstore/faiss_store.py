import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def build_faiss_index(chunks,embedding_model):
    # Convert string chunks to Document objects if necessary
    if isinstance(chunks[0], str):
        documents = [Document(page_content=chunk) for chunk in chunks]
    else:
        documents = chunks

    vector_store = FAISS.from_documents(documents, embedding_model)
    return vector_store

def save_faiss_index(vector_store, save_path):
    """Save the FAISS index to the specified path."""
    save_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(save_path)

def save_metadata(metadata: list[dict], file_path: Path):
    """Save metadata to a JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
