from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORS_DIR = DATA_DIR / "vectordb"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = VECTORS_DIR / "faiss_index"
METADATA_PATH = VECTORS_DIR / "chunk_metadata.json"
CHUNK_PATH = PROCESSED_DATA_DIR/"K4Ze-Sp6aUE_chunks.json"
