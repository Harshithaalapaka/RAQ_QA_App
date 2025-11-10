
#The variable client refers to your LLM API client — that is, the object that actually sends your prompt to the model and returns the response.
from src.rag_retriever import RAGretriever

def rag_advanced( client,query:str, retriever:RAGretriever, top_k=5, min_score=0.05, return_context=False):
    """Full RAG pipeline: retrieval + LLM answer generation (same as first version)."""
    
    # Retrieve top documents based on similarity score
    results = retriever.retrieve(query, top_k=top_k, Score_threshold=min_score)

    # Handle empty retrievals
    if not results:
        return {
            'answer': "no relevant content found",
            'sources': [],
            'confidence': 0.0,
            'content': ''
        }

    # Combine all retrieved text into one context
    context = "\n\n".join([doc['content'] for doc in results])

    # Extract metadata (file, page, score, preview)
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]

    #  Compute confidence
    confidence = max([doc['similarity_score'] for doc in results])

    # Build the final prompt
    prompt = f"""Use the following context to answer the question correctly.
Context:
{context}

Question: {query}

Answer:"""

    #  Generate answer using the LLM client
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.3
    )
    #print(response.usage)->prints tokens in ur ip prompt,tokens in ur op prompt and sum of both tokens

    #  Extract the answer text
    answer = response.choices[0].message.content

    # Build final structured output
    output = {
        'answer': answer,
        'sources': sources,
        'confidence': confidence
        
    }

    # Optionally return full context
    if return_context:
        output['context'] = context

    return output
