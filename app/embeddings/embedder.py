from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL_NAME
from langchain.vectorstores import FAISS

def load_embedding_model():
    """
    Load the HuggingFace embedding model specified in the configuration.
    
    Returns:
        HuggingFaceEmbeddings: An instance of the embedding model.
    """
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print(f"Successfully loaded embedding model: {EMBEDDING_MODEL_NAME}")
        return embedding_model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        raise

