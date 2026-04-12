# RAG Project for Wellness
chat bot assistant

# Phase 1 Goal: Create a RAG Project using single video transcription 
The goal of Phase 1 is to build a basic Retrieval-Augmented Generation (RAG) pipeline using the transcript of a single YouTube video.  
This phase focuses on validating the core idea of grounding LLM responses using external data instead of relying only on the model’s internal knowledge.

The pipeline includes:
- Fetching the YouTube transcript using an API
- Cleaning and preprocessing the transcript text
- Splitting the text into smaller chunks
- Generating embeddings for each chunk
- Storing embeddings in a vector database
- Retrieving relevant chunks based on a user query
- Generating responses using an LLM with retrieved context

## Phase 2 Goal: Refactoring the Notebook into a Modular RAG Project

In Phase 2, the goal is to convert the initial notebook-based RAG prototype into a clean, modular, and reusable Python project structure.  
This phase focuses on separating the core pipeline into independent components such as transcript ingestion, text cleaning, chunking, embeddings, vector storage, retrieval, and answer generation.  
The objective is to make the project easier to maintain, test, scale, and extend in future phases.
