from langchain_community.vectorstores import FAISS

def load_vectorstore(index_path, embedding_model):

    return FAISS.load_local(
        folder_path=str(index_path), 
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

def retrieve_relevant_chunks( vector_store,query, top_k=3):
    results = vector_store.similarity_search(query, k=top_k)
    return results


