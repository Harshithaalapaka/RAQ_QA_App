import os
import uuid
import numpy as np
from typing import List, Any
import chromadb
class Vectorstore:
    def __init__(self,collection_name:str="pdf_docs",persist_directory:str="data/vectorstore"):
        self.collection_name=collection_name
        self.persist_directory=persist_directory
        self.client=None
        self.collection=None
        self._initialise_store()

    
    def _initialise_store(self):
        try:
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client=chromadb.PersistentClient(path=self.persist_directory)
            self.collection=self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description":"pdf document embeddings for rag"
                }
            )
            print(f"vector store initialised.Collection:{self.collection_name}")
            print(f"existing documents in collection:{self.collection.count()}")
        except Exception as e:
            print(f"error initialising vector store:{e}")
            raise

    def add_documents(self,documents:List[Any],embeddings:np.ndarray):
        if len(documents)!=len(embeddings):
            raise ValueError("no of documents must match no of embeddings ")
        print(f"adding{len(documents)} documents to vector store...")

        ids=[]
        metadatas=[]
        documents_text=[]
        embeddings_list=[]

        for i,(doc,embedding) in enumerate(zip(documents,embeddings)):
            doc_id=f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata=dict(doc.metadata)
            metadata['doc_index']=i
            metadata['content_length']=len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"successfully added {len(documents)} documents to vectorstore")
            print(f"total docs in collection:{self.collection.count()}")

        except Exception as e:
            print(f"error adding docs to vectorstore:{e}")
            raise
