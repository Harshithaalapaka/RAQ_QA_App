"""from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it",temperature=0.1,max_tokens=1024)"""

"""def rag_simple(query,retriever,llm,top_k=3):
    results=retriever.retrieve(query,top_k=top_k)
    context:"\n\n".join([doc['content'] for doc in results]) if results else ""
    #Build a prompt using context + query.
    if not context: 
        return "no relevant context found to answer the question"
    prompt=\"""use the following context to answer question concisely
    context:{context}
    question:{query}
    answer:\"""
    response=llm.invoke([prompt.format(context=context,query=query)])
    return response.content"""
from rag_retriever import RAGretriever

def rag_advanced(self, query:str, retriever:RAGretriever, top_k=5, min_score=0.2, return_context=False):
    """Full RAG pipeline: retrieval + LLM answer generation (same as first version)."""
    
    # 1) Retrieve top documents based on similarity score
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    # 2) Handle empty retrievals
    if not results:
        return {
            'answer': "no relevant content found",
            'sources': [],
            'confidence': 0.0,
            'content': ''
        }

    # 3) Combine all retrieved text into one context
    context = "\n\n".join([doc['content'] for doc in results])

    # 4) Extract metadata (file, page, score, preview)
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]

    # 5) Compute confidence
    confidence = max([doc['similarity_score'] for doc in results])

    # 6) Build the final prompt
    prompt = f"""Use the following context to answer the question correctly.
Context:
{context}

Question: {query}

Answer:"""

    # 7) Generate answer using the LLM client
    response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.3
    )
    #print(response.usage)->prints tokens in ur ip prompt,tokens in ur op prompt and sum of both tokens

    # 8) Extract the answer text
    answer = response.choices[0].message.content

    # 9) Build final structured output
    output = {
        'answer': answer,
        'sources': sources,
        'confidence': confidence
        
    }

    # Optionally return full context
    if return_context:
        output['context'] = context

    return output
"""The OpenAI client sends your prompt to the server and gets back a structured Python object (a response dictionary)
response.choices → list of all model outputs (usually only one).

response.choices[0] → the first item in that list.

response.choices[0].message → the message object containing role and content.

response.choices[0].message.content → the actual text answer from the model."""
