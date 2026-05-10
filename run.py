from app.config import (
    FAISS_INDEX_PATH,
    TOP_K,
    LLM_MODEL_NAME,
    MAX_NEW_TOKENS
)
from app.embeddings.embedder import load_embedding_model
from app.retrival.retriever import load_vectorstore, retrieve_relevant_chunks_with_scores
from app.llm.prompt import build_rag_prompt
from app.llm.generator import load_generation_pipeline, generate_answer


def main():
    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Loading FAISS vector store...")
    vector_store = load_vectorstore(FAISS_INDEX_PATH, embedding_model)

    print("Loading LLM...")
    generator, tokenizer = load_generation_pipeline(LLM_MODEL_NAME)

    print("\n RAG Assistant is ready! You can ask questions about the video content. Type 'exit' to quit.")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit", "q", "bye"]:
            print("Exiting RAG Assistant. Goodbye!")
            break

        if not query:
            print("Please enter a question. \n")
            continue

        results = retrieve_relevant_chunks_with_scores(
            vector_store, 
            query, 
            top_k=TOP_K
        )

        prompt = build_rag_prompt(query, results)
        answer = generate_answer(
            generator, 
            tokenizer, 
            prompt, 
            max_new_tokens=MAX_NEW_TOKENS
        )

        print(f"\nRAG Assistant: {answer}\n")
        print()

if __name__ == "__main__":
    main()
