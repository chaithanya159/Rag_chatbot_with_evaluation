"""
rag/retriever.py
-----------------
Combines EmbeddingManager + VectorStore to retrieve the most relevant
document chunks for a user query.
"""

from typing import List, Tuple

from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore
from loaders.document_loader import Document


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self, query: str, top_k: int = 5, min_score: float = 0.0
    ) -> List[Tuple[Document, float]]:
        query_embedding = self.embedding_manager.generate_embedding(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return [(doc, score) for doc, score in results if score >= min_score]
