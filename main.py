from app.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.ingestion.youtube_loader import fetch_youtube_transcript
from app.ingestion.cleaner import extract_text_from_transcript, clean_text
from app.ingestion.chunker import chunk_text
from app.utils.file_handler import save_json, save_text


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
    print(f"Sample chunk content: {chunks[0].page_content[:200]}...")

    # Step 4: Save the cleaned text and chunks
    #save_text(cleaned_text, PROCESSED_DATA_DIR / f"{video_id}_cleaned.txt")
    save_json([chunk.page_content for chunk in chunks], PROCESSED_DATA_DIR / f"{video_id}_chunks.json")

if __name__ == "__main__":
    main()