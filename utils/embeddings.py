"""
SafeEmbeddings: A direct SentenceTransformer wrapper that bypasses all LangChain
HuggingFaceEmbeddings wrappers. These wrappers use the transformers library's
device_map/meta tensor initialization which crashes on Windows with newer torch versions.

This implementation directly calls SentenceTransformer which handles CPU inference
reliably without triggering the 'Cannot copy out of meta tensor' error.
"""
from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class SafeEmbeddings(Embeddings):
    """A stable CPU-only embeddings class that bypasses LangChain's HuggingFace wrappers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Explicitly avoid passing device="cpu" in the constructor as it can trigger 
        # 'Cannot copy out of meta tensor' errors on some Windows environments 
        # during the internal __init__ call to self.to(device).
        self._model = SentenceTransformer(model_name)
        self._model.to("cpu")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()
