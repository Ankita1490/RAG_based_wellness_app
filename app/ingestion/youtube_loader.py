"""
Responsible for loading YouTube video transcripts into the system.
"""
from typing import List, Dict
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

def fetch_youtube_transcript(video_id: str) -> List[Dict[str, str]]:
    """
    Fetches the transcript for a given YouTube video ID.

    Args:
        video_id (str): The ID of the YouTube video.

    Returns:
        List of transcript entries. Each entry is usually a dict with keys like ' text', 'start', and 'duration'.
    """

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=['en'])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        print(f"Transcripts are disabled for video ID: {video_id}")
        return []

