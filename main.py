from app.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNK_PATH,FAISS_INDEX_PATH, TOP_K
from app.ingestion.youtube_loader import fetch_youtube_transcript
from app.ingestion.cleaner import extract_text_from_transcript, clean_text
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import load_embedding_model
from app.vectorstore.faiss_store import build_faiss_index, save_faiss_index, save_metadata
from app.utils.file_handler import save_json, save_text, load_json
from app.retrival.retriever import load_vectorstore, retrieve_relevant_chunks


def main():
    video_id = "K4Ze-Sp6aUE"  

    # Step 1: Fetch the transcript
    transcript = fetch_youtube_transcript(video_id)
    print(f"Fetched transcript entries: {len(transcript)}")

    # Save raw transcript
    raw_file_path = RAW_DATA_DIR / f"{video_id}_transcript.json"
    save_json(transcript, raw_file_path)
    print(f"Saved raw transcript to: {raw_file_path}")

    # # Step 2: Extract and clean the text
    # raw_text = extract_text_from_transcript(transcript)
    # cleaned_text = clean_text(raw_text)

    # # save cleaned transcript
    # cleaned_file_path = PROCESSED_DATA_DIR / f"{video_id}_cleaned.txt"
    # save_text(cleaned_text, cleaned_file_path)
    # print(f"Saved cleaned transcript to: {cleaned_file_path}")

    # Step 3: Chunk the cleaned text
    chunks = chunk_text(transcript)
    print(f"Created {len(chunks)} chunks from the transcript.")
    print(f"Sample chunk content: {chunks[1].page_content[:200]}...")

    # Step 4: Save the cleaned text and chunks
    #save_text(cleaned_text, PROCESSED_DATA_DIR / f"{video_id}_cleaned.txt")
    save_json([chunk.page_content for chunk in chunks], PROCESSED_DATA_DIR / f"{video_id}_chunks.json")

    # Load chunked data
    chunks_data = load_json(CHUNK_PATH)
    print(f"Loaded {len(chunks_data)} chunks from saved file.")

    #generate embeddings
    embedding_model =  load_embedding_model()
    print(f"loaded embedding model: {embedding_model}")

    # # build FAISS index
    vector_store= build_faiss_index(chunks_data, embedding_model)
    print(f"vector store created: {vector_store.index_to_docstore_id}")
    print("Built FAISS index successfully.")

    # Save the FAISS index
    save_faiss_index(vector_store, FAISS_INDEX_PATH)

    query = "What does the video say about a balanced diet?"
    # load embedding model
    embedding_model = load_embedding_model()
    print(f"Loaded embedding model: {embedding_model}")

    #load FAISS vector store
    vector_store = load_vectorstore(FAISS_INDEX_PATH, embedding_model)
    print(f"Loaded FAISS vector store")

    # retrieve k-top similar chunks
    results = retrieve_relevant_chunks(vector_store, query, top_k=TOP_K)
    print(f"Retrived top {TOP_K} relevant chunks for the query: '{query}'")

    for i , doc in enumerate(results, start=0):
        print(f"\n--- Result {i+1} ---")
        print(f"Content: {doc.page_content[:500]}")  # Print the first 500 characters of the chunk
        print(f"Metadata: {doc.metadata}")
        print()

if __name__ == "__main__":
    main()