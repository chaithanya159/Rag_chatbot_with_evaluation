"""
rag/embeddings.py
------------------
Generates vector embeddings for text using a local sentence-transformers
model — no API key or cloud embedding service needed. Same model is used
for document chunks and user queries so they share a vector space.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, 384-dim, free


class EmbeddingManager:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Returns (N, dim) float32 L2-normalized embeddings."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def generate_embedding(self, text: str) -> np.ndarray:
        return self.generate_embeddings([text])[0]
