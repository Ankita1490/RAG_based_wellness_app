from app.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNK_PATH,FAISS_INDEX_PATH
from app.ingestion.youtube_loader import fetch_youtube_transcript
from app.ingestion.cleaner import extract_text_from_transcript, clean_text
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import load_embedding_model
from app.vectorstore.faiss_store import build_faiss_index, save_faiss_index
from app.utils.file_handler import save_json, load_json


def main():    
    video_id = "K4Ze-Sp6aUE"  

    # Step 1: Fetch the transcript
    transcript = fetch_youtube_transcript(video_id)
    print(f"Fetched transcript entries: {len(transcript)}")

    raw_file_path = RAW_DATA_DIR / f"{video_id}_transcript.json"
    save_json(transcript, raw_file_path)
    print(f"Saved raw transcript to: {raw_file_path}")

    chunks = chunk_text(transcript)
    print(f"Created {len(chunks)} chunks from the transcript.")
    print(f"Sample chunk content: {chunks[1].page_content[:200]}...")

    chunks_path = PROCESSED_DATA_DIR / f"{video_id}_chunks.json"
    save_json([chunk.page_content for chunk in chunks], chunks_path)
    print(f"Saved chunks to: {chunks_path}")

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


if __name__ == "__main__":
    main()
