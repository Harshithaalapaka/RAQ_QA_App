from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware#Allows frontend (HTML/JS) to call backend,avoids cors errorsin the browser,Browser blocks frontend → backend calls by default.✔ This allows:HTML / JSReactAny frontend
from pydantic import BaseModel#Used to define request & response formats ensures data validation
from src.data_loader import load_documents
from src.text_splitter import split_documents
from src.rag_pipeline import rag_advanced
from src.rag_retriever import RAGretriever
from src.vector_store import Vectorstore
from src.embeddings import Embeddingmanager
from openai import OpenAI
from dotenv import load_dotenv
from groq import Groq
import os
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

embedding_manager = Embeddingmanager()
vector_store = Vectorstore()  # Persistent store


if vector_store.collection.count() == 0:
    print("Vectorstore empty — loading and embedding documents...")
    docs = load_documents("data/pdf")
    chunks = split_documents(docs)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)
else:
    print(f"Vectorstore already has {vector_store.collection.count()} documents. Skipping load.")

retriever = RAGretriever(vector_store=vector_store, embedding_manager=embedding_manager)


app = FastAPI(title="RAG QA SYSTEM")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: list

# API Endpoint

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    result = rag_advanced(client, request.query, retriever=retriever)
    return {
        "answer": result.get("answer", "No answer found"),
        "confidence": result.get("confidence", 0.0),
        "sources": result.get("sources", []),
    }


