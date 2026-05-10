def build_rag_prompt(query:str, retrieved_docs:list):
    """
    Build a grounded RAG prompt using the user query and retrieved documents.

    Args:
        query: User's question
        retrieved_docs: List of retrieved LangChain Document objects

    Returns:
        Formatted prompt string
    """
    context_parts = []
    for i, (doc, score) in enumerate(retrieved_docs, start=1):
        context_parts.append(f"Chunk {i}:\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful assistant for question answering over a YouTube transcript.

Use only the retrieved context below to answer the user's question.
Do not use outside knowledge.
If the answer is not clearly present in the context, say:
"I could not find the answer in the retrieved transcript."

User Question:
{query}

Retrieved Context:
{context}

Answer:
""".strip()

    return prompt