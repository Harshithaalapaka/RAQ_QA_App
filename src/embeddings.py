from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
class Embeddingmanager:
        def __init__(self,model_name:str="all-MiniLM-L6-V2"):
                self.model_name=model_name
                self.model=None
                self._load_model()
        """  __init__ is a special method in Python.

It runs automatically when you create an object of this class.

model_name: str = "all-MiniLM-L6-V2" means:
"all-MiniLM-L6-V2" is a pre-trained model available in the sentence-transformers library.

You can pass a model name when creating the object.

If you don’t, it will use "all-MiniLM-L6-V2" by default.
self.model → creates an attribute to hold the actual model once it’s loaded.

None means “there is no model yet.”
Calls the internal method _load_model() to actually load the embedding model.
After this line runs, self.model will contain the pre-trained embedding model ready to generate embeddings."""

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

