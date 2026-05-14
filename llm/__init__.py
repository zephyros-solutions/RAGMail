"""
LLM Module
Handles language model integration, prompt templates, and RAG orchestration.
- prompts: Prompt templates with few-shot examples and chain-of-thought
- rag: RAG system with specialized analysis modes (perspective, conflict, comparison)
"""

from llm.rag import RAG
from llm.prompts import (
    perspective_analysis_cot,
    perspective_comparison_cot,
    conflict_analysis_cot,
    summarize_with_perspective,
    extract_key_claims
)

__all__ = [
    'RAG', 
    'perspective_analysis_cot',
    'perspective_comparison_cot',
    'conflict_analysis_cot',
    'summarize_with_perspective',
    'extract_key_claims'
]
