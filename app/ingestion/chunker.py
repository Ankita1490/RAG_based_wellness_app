from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(transcript: str) -> list:
    """
    Splits the input text into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        text (str): The input text to be chunked.

    Returns:
        List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.create_documents([transcript])
    return chunks