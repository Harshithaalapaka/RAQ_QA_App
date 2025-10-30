from typing import List, Dict, Any
from vector_store import Vectorstore
from embeddingsopenaimodel import Embeddingmanager
class RAGretriever:
    def __init__(self, vector_store: Vectorstore, embedding_manager: Embeddingmanager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, Score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"retrieving docs for query:'{query}'")
        print(f"top_k:{top_k}, scorethreshold:{Score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                """ # convert distance to similarity depending on your DB metric
                similarity_score = 1 - distance  # only if distance is 1 - cosine"""

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    if similarity_score >= Score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarityscore': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                print(f"retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("no docs found")

            return retrieved_docs

        except Exception as e:
            print(f"error during retrieval: {e}")
            return []

"""ragretriever = RAGretriever(vectorstore, embeddingmanager)
ragretriever
"""