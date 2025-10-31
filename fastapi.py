from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_pipeline import rag_advanced
from src.rag_retriever import RAGretriever
retriever=RAGretriever()
#creating fastapi app and it is the main server
app=FastAPI(title="RAG QA SYSTEM")
# Define what input (request) your API will accept
class queryrequest(BaseModel):
    query:str
class queryresponse(BaseModel):
    answer:str
    confidence:float
    sources:list
#Define your endpoint ("/ask") that the client will call
#POST endpoint → user sends query
@app.post("/ask", responsemodel=queryresponse)
def ask_question(request: queryrequest):
    result = rag_advanced(request.query, retriever=retriever)

    # Return clean structured result
    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources": result["sources"]
    }
  

"""
  
  main.py (your FastAPI backend)

This runs the RAG pipeline and returns a JSON result.

 test_api.py (your client/test script)

This sends a request to the backend and prints what comes back.



  def ask_question(request: QueryRequest):
    
    When a POST request is made to /ask, this function runs.
    It takes the user's query, runs the RAG pipeline, and returns the answer.
    
    try:
        # Call your RAG pipeline with the user's question
        result = rag_advanced(request.query, retriever)

        # Return the full result (FastAPI automatically converts it to JSON)
        return result

    except Exception as e:
        # If something goes wrong, return an error response
        return {"answer": f"Error: {str(e)}", "confidence": 0.0, "sources": []}"""
