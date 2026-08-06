"""
Embedding Generation Module.

Responsible for generating dense vector embeddings from processed clauses 
for semantic search, vector storage, and RAG pipelines.
"""

from src.legal_nlp.embeddings.generator import EmbeddingGenerator

__all__ = ["EmbeddingGenerator"]