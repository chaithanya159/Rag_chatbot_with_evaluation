"""
rag/vector_store.py
--------------------
FAISS-backed vector store (in place of the reference repo's ChromaDB store).
Same interface shape: add_documents(), get_count(), reset(), persisted to
disk under `persist_directory` so it survives server restarts.
"""

import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np

from loaders.document_loader import Document


class VectorStore:
    def __init__(self, persist_directory: str, dim: int = 384):
        self.persist_directory = persist_directory
        self.dim = dim
        self.index_path = os.path.join(persist_directory, "index.faiss")
        self.docs_path = os.path.join(persist_directory, "documents.pkl")

        os.makedirs(persist_directory, exist_ok=True)
        self.documents: List[Document] = []
        self.index = faiss.IndexFlatIP(dim)

        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.docs_path, "rb") as f:
                self.documents = pickle.load(f)

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        assert len(documents) == embeddings.shape[0]
        self.index.add(embeddings)
        self.documents.extend(documents)
        self._persist()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        if self.index.ntotal == 0:
            return []
        query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.documents[idx], float(score)))
        return results

    def get_count(self) -> int:
        return self.index.ntotal

    def reset(self) -> None:
        """Clears all stored documents/vectors — used before loading a new document."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.documents = []
        for path in (self.index_path, self.docs_path):
            if os.path.exists(path):
                os.remove(path)
