"""Responsible for cleaning and preprocessing raw data before ingestion."""
from typing import List, Dict
import re

def extract_text_from_transcript(transcript: List[Dict]) -> str:
    """
    Extracts only the test field from the YouTube transcript entries and concatenates them into a single string.

    Args:
        transscript: List of transcript dictionaries

    Returns:
        Combined transcript text as a single string.
   """
    
    text_parts = [entry["text"] for entry in transcript if 'text' in entry]
    return " ".join(text_parts)

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing special characters, extra whitespace, and normalizing it.

    Args:
        text: The raw text to be cleaned.
    Returns: Clean text.
    """
    # Replace multiple spaces/newlines with a single space
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text