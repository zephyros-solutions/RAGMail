"""
Configuration Module
Centralized configuration for Milvus, Elasticsearch, LLM, and chunking.
"""

from config.globals import (
    OLLAMA_API_BASE,
    OLLAMA_API_KEY,
    GEN_MODELS,
    DENSE_EMB_MODELS,
    SPARSE_EMB_FUNS,
    MILVUS_DYN,
    MILVUS_MAX_LENGTH,
    MILVUS_LEN_CTX,
    MAX_CHUNK_LEN,
    MAX_CHUNK_EXCESS,
    TOK2CHAR,
    RANKER
)

__all__ = [
    'OLLAMA_API_BASE',
    'OLLAMA_API_KEY',
    'GEN_MODELS',
    'DENSE_EMB_MODELS',
    'SPARSE_EMB_FUNS',
    'MILVUS_DYN',
    'MILVUS_MAX_LENGTH',
    'MILVUS_LEN_CTX',
    'MAX_CHUNK_LEN',
    'MAX_CHUNK_EXCESS',
    'TOK2CHAR',
    'RANKER'
]
