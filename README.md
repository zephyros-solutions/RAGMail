# RAGMail

A privacy-preserving AI assistant for exploring personal email archives. It helps you analyze correspondences to understand perspectives, conflicts, and relationships between people in your email history — entirely on your machine.

## What It Does

Feed it your Mac Mail mailbox (tens of thousands of `.emlx` files), and you can ask natural-language questions like:

- "What is Person A's perspective on the project conflict?"
- "Compare the viewpoints of Person A and Person B on topic X"
- "Who disagrees with whom about the timeline?"

The system retrieves relevant email fragments, feeds them to a local LLM, and returns structured, evidence-based analysis in Italian.

## Architecture

RAGMail is a classic retrieval-augmented generation pipeline with a mail-specific twist. The data flows through three stages:

**1. Email Processing** — Raw `.emlx` files are parsed, threaded into conversations, decoded (UTF-8, ISO-8859-1, quoted-printable), cleaned of replies/quotes/signatures, and sender names are consolidated from 140+ variants down to canonical people using an alias map. The result is a set of clean `Mail` objects with resolved sender names and normalized text.

**2. Retrieval** — Clean email text is indexed using one of four strategies:
- **blob**: Loads the entire corpus into context. Works for small mailboxes but hits LLM context limits quickly.
- **grep**: Extracts entities from your question via the LLM, then does keyword matching. Fast, no infrastructure, but blind to synonyms and semantics.
- **Elasticsearch**: Full-text BM25 with an Italian stemmer. Sophisticated ranking but requires a running ES instance.
- **Milvus (recommended)**: Hybrid dense+sparse vector search. Dense embeddings capture semantic similarity; sparse embeddings (via Milvus's built-in BM25) handle exact keyword matching. Results are reranked with weighted fusion. This is the most capable strategy and what most queries should use.

**3. LLM Analysis** — Retrieved chunks are assembled into context and routed to one of four specialized analyzers built with DSPy:
- **Generic**: Open-ended summary of retrieved context
- **Perspective**: Analyzes one person's viewpoint on a topic with chain-of-thought reasoning
- **Conflict**: Identifies the root cause of a disagreement between two people
- **Comparison**: Contrasts two people's positions, identifying convergences and divergences

All LLM inference runs locally via Ollama (default: `llama3.2`, 3.2B params, 131K context).

## Design Decisions

**Why Milvus over a cloud vector DB?** Privacy is the core constraint. This tool is for personal email archives containing sensitive information. Milvus with a SQLite backend keeps everything local.

**Why DSPy?** Prompt engineering is fragile. DSPy's `ChainOfThought` modules give us reproducible prompting with type signatures, and the framework makes it straightforward to swap in optimizers later when we have ground truth data.

**Why offline?** The entire system is designed around the constraint that no data leaves the machine. This means:
- Ollama for inference (no API keys, no data leakage)
- Local Milvus index (no cloud dependencies)
- No telemetry, no analytics, no external calls beyond local services

**Why four retrieval strategies?** Different use cases call for different tradeoffs. For a quick sanity check, grep is fastest. For small corpora, blob avoids retrieval errors entirely. Milvus is the serious option for large mailboxes where semantic understanding matters. ES is a fallback for teams already running Elasticsearch.

## Strengths

- **Solid email processing pipeline**: Handles Mac Mail's `.emlx` format, conversation threading, multi-encoding, and Italian-specific text patterns. The alias system resolves 140+ email/name variants into canonical people (~95% resolution rate).
- **Multi-strategy retrieval**: Four indexing strategies covering the full spectrum from lightweight (grep) to sophisticated (hybrid Milvus). Each is independently usable.
- **Structured analysis**: The four RAG modes (generic, perspective, conflict, comparison) with few-shot examples and chain-of-thought prompts produce more consistent, evidence-grounded responses than raw prompt-and-generate.
- **Offline-first by design**: Everything runs locally. No secrets, no telemetry, no hidden dependencies.
- **Configurable**: `config/globals.py` centralizes every parameter — chunk sizes, embedding models, Milvus settings, available LLMs. Easy to swap models or tune behavior.

## Weaknesses & Known Issues

- **Critical: `breakpoint()` in `llm/rag.py:67`** — Blocks execution in the RAG forward path. Must be removed before this code can run.
- **Elasticsearch is half-wired**: `es.py` and `do_es()` in main.py exist and work, but ES is not integrated into the CLI strategy selection. It's ready but dormant.
- **Sparse embedding is disabled**: `SPARSE_EMB_FUNS = None` in globals.py, falling back to Milvus's built-in BM25. The hybrid strategy is incomplete without the sparse signal.
- **Alias map is manual**: `alias.py` grows ad-hoc as new name variants appear. No automated extraction from the corpus. Maintenance burden increases with corpus size.
- **Evaluation harness uses mocks**: `test_evaluation.py` substitutes all retrieval strategies with keyword matching. You get latency numbers but not real quality metrics.
- **Ground truth is heuristic-only**: Relevance is determined by "document contains 2+ query keywords." This misses semantic matches and creates false positives.
- **No response caching**: Every query re-embeds from scratch. No persistent Milvus collections across runs.
- **Context window management is crude**: The `reduce_context()` function in main.py splits and summarizes with the LLM when context exceeds limits, but there's no smarter truncation strategy.

## Running It

```bash
# Setup
conda create -n rag python=3.11 -y && conda activate rag
pip install -r pip_requirements.txt

# Start Ollama and pull models
ollama pull llama3.2
ollama pull nomic-embed-text

# Process mailbox and query
python main.py -g llama3.2 -d nomic -u $USER -m ~/Library/Mail --method milvus
python main.py -g llama3.2 -u $USER -m ~/Library/Mail --method grep
python main.py -g llama3.2 -u $USER -m ~/Library/Mail --method blob
```

## Current TODOs (from main.py)

- Contextual retrieval with Milvus
- Integrate Milvus with DSPy (official docs exist)
- DSPy optimization pipeline
- Explore ColBERTv2 embeddings vs Milvus native embeddings
- Try different generation/embedding models from Ollama library
- Wrap embedders in `dspy.Embedder` for automatic batching
