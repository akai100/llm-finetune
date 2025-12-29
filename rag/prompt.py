def build_rag_prompt(query, docs):
    context = "\n".join(docs)
    return f"""You are a helpful assistant.

Context:
{context}

Question:
{query}

Answer:"""

