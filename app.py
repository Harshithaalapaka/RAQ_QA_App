#These lines import other Python files (modules) and libraries you created or installed.Chunking is important because embedding models have token limits.
from src.data_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import Embeddingmanager
from src.vector_store import Vectorstore
from src.rag_retriever import RAGretriever
from src.rag_pipeline import rag_advanced#combines retrieval + LLM to produce final answers.
from openai import OpenAI#The OpenAI class is like a client object that lets your code connect to OpenAI’s servers (for example, to use GPT models or embedding models).
from dotenv import load_dotenv
import os

"""here we are importing clsnames so that we can create objs when ever we needed

Imports load_dotenv() which reads environment variables from a .env file and loads the variables inside it into the system environment. so secrets like API keys stay out of code.
"""

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
#Initialize OpenAI client
"""Creates a connection to OpenAI’s API using your secret key.

client → This object lets you send requests to OpenAI (for embeddings, text completion, etc.).
each openai mode has a fixed context window(max no of tokens(1 token=4 chars) it can handle(input+output))"""

client = OpenAI(api_key=openai_api_key)
"""to check context window(model's token limit)
model_info=client.models.retrieve("modelname")
print(model_info)

"""



# Step 1: Load documents
docs = load_documents("../data/pdf")

# Step 2: Split into chunks
chunks = split_documents(docs)

"""texts = [chunk.page_content for chunk in chunks]
Creates a list of only text strings (not document objects).
Each chunk.page_content extracts the text from each chunk.

python
Copy code
embeddings = embed_manager.generate_embeddings(texts)
Sends the list of texts to OpenAI’s embedding model.

Returns a NumPy array of vector embeddings.

Each embedding represents a text’s meaning numerically.
add_documents() stores both the chunks and their embeddings.

This lets you perform similarity search later when a user asks a question
The retriever uses:

vstore → to find the most relevant chunks.

embed_manager → to embed the user’s query for comparison."""

# Step 3: Generate embeddings
embed_manager = Embeddingmanager()
#texts should be a list of strings OpenAI accepts a list .OpenAI’s embedding endpoint expects a list of strings.
texts = [chunk.page_content for chunk in chunks]
embeddings = embed_manager.generate_embeddings(texts)

# Step 4: Store embeddings
vstore = Vectorstore()
vstore.add_documents(chunks, embeddings)

# Step 5: Create retriever
retriever = RAGretriever(vstore, embed_manager)

# Step 6: Interactive query
while True:
    query = input("\nAsk a question (or type 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    result = rag_advanced(client,query, retriever, top_k=3, min_score=0.1, return_context=True)
    """Converts your question into embeddings.

Retrieves the top 3 most similar chunks (top_k=3).

Filters out chunks below a similarity threshold (min_score=0.1).

Combines those chunks and passes them to an LLM (via OpenAI) to generate a final answer.

If return_context=True, it also returns the text passages used to answer."""
    print("\nANSWER:", result['answer'])
    print("\nCONFIDENCE:", result['confidence'])
    print("\nCONTEXT PREVIEW:", result['context'][:300])
