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

## Phase 3.1 — CLI Interactive Chat Assistant

In this phase, the project was extended into an interactive command-line chatbot that allows users to ask questions about the YouTube transcript in real time.

The system now supports:

- loading a prebuilt FAISS vector index
- accepting dynamic user queries from the terminal
- retrieving top-k relevant transcript chunks
- generating grounded answers using an instruction-tuned LLM
- continuous interaction without restarting the application

This phase transforms the project from a static RAG pipeline into an interactive conversational assistant.

## Current RAG Pipeline Architecture

```text
User Query
    ↓
FAISS Similarity Search
    ↓
Retrieve Relevant Transcript Chunks
    ↓
Prompt Construction
    ↓
Qwen LLM Generation
    ↓
Grounded Response

## Current Tech Stack

- Python
- LangChain
- FAISS
- Hugging Face Transformers
- Qwen2.5 Instruct Model
- YouTube Transcript API
- RecursiveCharacterTextSplitter
- Sentence Transformers Embeddings


## Current Models

### Embedding Model
- `sentence-transformers/all-MiniLM-L6-v2`

### Generation Model
- `Qwen/Qwen2.5-0.5B-Instruct`