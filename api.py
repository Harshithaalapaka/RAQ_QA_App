from fastapi import FastAPI
from pydantic import BaseModel
from src.data_loader import load_documents
from src.text_splitter import split_documents
from src.rag_pipeline import rag_advanced
from src.rag_retriever import RAGretriever
from src.vector_store import Vectorstore
from src.embeddings import Embeddingmanager 
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create OpenAI client (LLM)
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)
# Step 1: Load documents
docs = load_documents("data/pdf")

# Step 2: Split into chunks
chunks = split_documents(docs)
embedding_manager = Embeddingmanager()
vector_store = Vectorstore()

# 2️⃣ Initialize retriever with dependencies
retriever = RAGretriever(vector_store=vector_store, embedding_manager=embedding_manager)

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
@app.post("/ask", response_model=queryresponse)
async def ask_question(request: queryrequest):
    result = rag_advanced(client,request.query,retriever=retriever)
    print(result)
    # Return clean structured result
    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources": result["sources"]
    }
  
