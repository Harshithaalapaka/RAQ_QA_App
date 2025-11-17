
#The variable client refers to your LLM API client — that is, the object that actually sends your prompt to the model and returns the response.
from src.rag_retriever import RAGretriever

def rag_advanced(client, query: str, retriever: RAGretriever, top_k=5, min_score=0.05, return_context=False):
    """Full RAG pipeline: retrieval + LLM answer generation with fallback."""
    
    # Retrieve top documents
    results = retriever.retrieve(query, top_k=top_k, Score_threshold=min_score)

    # -------------------------------------------------------
    # 1️⃣ FALLBACK CASE — No results from vector DB
    # -------------------------------------------------------
    if not results:
        # No context → normal LLM response
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer using your general knowledge."},
                {"role": "user", "content": query}
            ],
            max_tokens=512,
            temperature=0.3
        )
        return {
            "answer": response.choices[0].message.content,
            "sources": [],
            "confidence": 0.0,
            "context": "" if return_context else None
        }

    # -------------------------------------------------------
    # 2️⃣ CONTEXT AVAILABLE → Proceed with RAG
    # -------------------------------------------------------

    # Combine retrieved content
    context = "\n\n".join([doc['content'] for doc in results])

    # Metadata for UI
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]

    # Confidence = highest similarity score
    confidence = max([doc['similarity_score'] for doc in results])

    # --------------------
    # Build prompt for RAG
    # --------------------
    prompt = f"""
Use the following context to answer the question.

If the context does NOT contain the answer, then answer using your own general knowledge.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    # Ask LLM with RAG context
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.3
    )

    answer = response.choices[0].message.content

    # Build response
    output = {
        'answer': answer,
        'sources': sources,
        'confidence': confidence
    }

    if return_context:
        output['context'] = context

    return output
