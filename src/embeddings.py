from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
class Embeddingmanager:
        def __init__(self,model_name:str="all-MiniLM-L6-V2"):
                self.model_name=model_name
                self.model=None
                self._load_model()   #automatically load at initialization
       
        def _load_model(self):
                try:
                        print(f"loading embedding model:{self.model_name}")
                        self.model=SentenceTransformer(self.model_name)
                        print(f"model loaded successfully.Embedding dimension:{self.model.get_sentence_embedding_dimension()}")
                except Exception as e:
                        print(f"error loading model{self.model_name}:{e}")
                        raise

        def generate_embeddings(self,texts:List[str]) -> np.ndarray:
                     if not self.model:
                             raise ValueError("model not loaded")
                     print(f"generating embeddings for{len(texts)}texts...")
                     embeddings=self.model.encode(texts,show_progress_bar=True)
                     print(f"generated embeddings with shape:{embeddings.shape}")
                     return embeddings
'''embeddingmanager=Embeddingmanager()
embeddingmanager'''

