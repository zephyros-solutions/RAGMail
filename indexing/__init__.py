"""
Indexing Module
Handles email retrieval, indexing, and search across multiple strategies.
- Milvus: Dense + sparse hybrid semantic search
- Elasticsearch: BM25 relevance-based search
"""

from indexing.retriever import RMClient, my_embedder
from indexing.es import ElSearch

__all__ = ['RMClient', 'my_embedder', 'ElSearch']
