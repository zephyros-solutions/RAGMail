#!/usr/bin/env python3
"""
DSPy Optimization Experiments for RAGMail

This module provides a complete pipeline for experimenting with DSPy's optimization
capabilities. It follows the pattern from the DSPy tutorial: build a RAG pipeline,
evaluate it, compile with a teleprompter, and compare before/after results.

Usage:
    python tests/test_dspy.py                          # Run full optimization pipeline
    python tests/test_dspy.py --baseline-only          # Only baseline evaluation
    python tests/test_dspy.py --teleprompter random    # Use random search teleprompter
    python tests/test_dspy.py --teleprompter none      # Skip optimization, just show compiled program
    python tests/test_dspy.py --demo                   # Run single example (no optimization)
"""

import os
import dspy
import ujson
import orjson
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ragqa_arena_tech_500():
    """Load the ragqa_arena_tech_500 dataset, split into train/val/dev/test."""
    with open('ragqa_arena_tech_500.json') as f:
        data = ujson.load(f)
    data = [dspy.Example(**d).with_inputs('question') for d in data]
    trainset, valset, devset, testset = data[:50], data[50:150], data[150:300], data[300:500]
    print(f"Loaded: train={len(trainset)}, val={len(valset)}, dev={len(devset)}, test={len(testset)}")
    return trainset, valset, devset, testset


def load_corpus():
    """Load the ragqa_arena_tech_corpus.jsonl as the document corpus for retrieval."""
    corpus_path = Path("ragqa_arena_tech_corpus.jsonl")
    if not corpus_path.exists():
        print(f"Corpus file not found: {corpus_path}")
        return None
    max_chars = 6000
    docs = []
    with open(corpus_path) as f:
        for line in f:
            doc = orjson.loads(line)
            docs.append({"doc_id": doc["doc_id"], "text": doc["text"][:max_chars]})
    print(f"Loaded {len(docs)} corpus documents")
    return docs


# ---------------------------------------------------------------------------
# Embedder / retriever
# ---------------------------------------------------------------------------

def create_embedder(dimension: int = 512):
    """Create an embedding function using Ollama's nomic-embed-text model."""
    import ollama
    model = 'nomic-embed-text:latest'
    batch_size = 64

    def embedder(texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError(f"texts must be list, got {type(texts)}")
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Truncate each text to dimension to avoid Ollama errors
            truncated = [t[:dimension] for t in batch]
            resp = ollama.embed(model, input=truncated)
            results.extend(resp["embeddings"])
        return results

    return model, embedder


def create_retriever(docs: list, embedder, k: int = 5, saved_dir: str = "./saved_embeddings"):
    """Create a DSPy Embeddings retriever from a document corpus."""
    saved_path = Path(saved_dir)
    if saved_path.exists():
        print(f"Loading saved embeddings from {saved_dir}")
        search = dspy.Embeddings.from_saved(str(saved_path), embedder)
    else:
        corpus = [d["text"] for d in docs]
        print(f"Building retriever with {len(corpus)} documents, saving to {saved_dir}")
        search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=k)
        search.save(str(saved_path))
    return search


# ---------------------------------------------------------------------------
# The RAG pipeline — unoptimized
# ---------------------------------------------------------------------------

class GenerateAnswer(dspy.Signature):
    """Answer questions using the provided context."""
    context = dspy.InputField(desc="relevant facts from retrieval")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="concise, accurate answer")


class UnoptimizedRAG(dspy.Module):
    """RAG pipeline with no optimization — just ChainOfThought on raw context."""

    def __init__(self, retriever, max_context_length: int = 512):
        super().__init__()
        self.retrieve = retriever
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_context_length = max_context_length

    def forward(self, question: str):
        retrieved = self.retrieve(question)
        context = [c[:self.max_context_length] for c in retrieved.passages]
        answer = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=answer.answer)


# ---------------------------------------------------------------------------
# Optimized RAG — same architecture, but compiled via teleprompter
# ---------------------------------------------------------------------------

class OptimizedRAG(dspy.Module):
    """Same RAG architecture as UnoptimizedRAG, but will be compiled with
    a teleprompter that optimizes the few-shot examples in ChainOfThought."""

    def __init__(self, retriever, max_context_length: int = 512):
        super().__init__()
        self.retrieve = retriever
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_context_length = max_context_length

    def forward(self, question: str):
        retrieved = self.retrieve(question)
        context = [c[:self.max_context_length] for c in retrieved.passages]
        answer = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=answer.answer)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def semantic_f1_metric(example, pred, question: str = None):
    """Semantic F1 metric: compares predicted answer with gold answer using
    token-level F1 between their embeddings."""
    from dspy.evaluate import SemanticF1
    metric = SemanticF1()
    return metric(example, pred)


def answer_exact_match_metric(example, pred):
    """Exact match: does the predicted answer exactly match the gold answer?"""
    return dspy.evaluate.answer_exact_match(example, pred)


def combined_metric(example, pred):
    """Combined metric: penalizes both wrong answers and wrong context."""
    from dspy.evaluate import SemanticF1
    score_f1 = SemanticF1()(example, pred)
    score_em = dspy.evaluate.answer_exact_match(example, pred)
    return score_f1


# ---------------------------------------------------------------------------
# Validation functions for teleprompters
# ---------------------------------------------------------------------------

def validate_context_and_answer(example, pred, trace=None):
    """Validate that: (1) answer matches gold, and (2) context actually contains the answer."""
    answer_match = dspy.evaluate.answer_exact_match(example, pred)
    context_match = dspy.evaluate.answer_passage_match(example, pred)
    return answer_match and context_match


def validate_answer_only(example, pred, trace=None):
    """Lighter validation: only check if the answer is roughly correct (Semantic F1 > 0)."""
    score = SemanticF1()(example, pred)
    return score > 0.1  # Accept if any semantic overlap


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def evaluate_pipeline(pipeline, testset, metric_fn, name: str = "Pipeline",
                      num_examples: Optional[int] = None, display_table: int = 5):
    """Run a pipeline on a test set and return the average metric score."""
    if num_examples:
        dev = testset[:num_examples]
    else:
        dev = testset

    evaluator = dspy.Evaluate(
        devset=dev,
        metric=metric_fn,
        num_threads=16,
        display_progress=False,
        display_table=display_table,
        return_all_scores=True,
        return_delegate=True,
    )

    results = evaluator(pipeline)

    # results is a list of (example, pred, score) tuples when return_all_scores=True
    scores = [r[2] for r in results] if isinstance(results, list) else results
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'='*60}")
    print(f"{name}: {avg_score:.3f} avg score over {len(scores)} examples")
    print(f"{'='*60}\n")

    return avg_score, scores


# ---------------------------------------------------------------------------
# The full optimization pipeline
# ---------------------------------------------------------------------------

def run_optimization(teleprompter: str = "fewshot"):
    """
    Run the full DSPy optimization pipeline:

    1. Build the RAG pipeline (unoptimized)
    2. Evaluate baseline
    3. Compile with teleprompter
    4. Evaluate optimized version
    5. Show before/after comparison

    teleprompter choices:
        - "fewshot"      : BootstrapFewShot (default, follows the Medium tutorial)
        - "random"       : BootstrapFewShotWithRandomSearch (tries random perturbations)
        - "none"         : Show what the compiled program looks like without running it
    """

    # 1. Setup LLM
    print("=" * 60)
    print("DSPy RAG Optimization Pipeline")
    print("=" * 60)
    model = 'ollama_chat/llama3.2:latest'
    lm = dspy.LM(model)
    dspy.configure(lm=lm)
    print(f"LLM: {model}")

    # 2. Load data and build retriever
    trainset, valset, devset, testset = load_ragqa_arena_tech_500()
    docs = load_corpus()
    if docs is None:
        print("Cannot proceed without corpus. Please ensure ragqa_arena_tech_corpus.jsonl exists.")
        return

    emb_model, embedder = create_embedder(dimension=512)
    retriever = create_retriever(docs, embedder)
    print(f"Embedder: {emb_model}")

    # 3. Build the unoptimized RAG pipeline
    print("\n--- Building UNOPTIMIZED RAG pipeline ---")
    unoptimized = UnoptimizedRAG(retriever=retriever, max_context_length=512)

    # 4. Evaluate baseline
    print("\n>>> Evaluating BASELINE (unoptimized)")
    baseline_score, baseline_scores = evaluate_pipeline(
        unoptimized, testset, combined_metric,
        name="Baseline", num_examples=50, display_table=3
    )

    # Show a few individual examples for inspection
    print("\n--- Baseline: individual examples ---")
    for i, ex in enumerate(testset[:5]):
        pred = unoptimized(question=ex.question)
        score = semantic_f1_metric(ex, pred)
        print(f"\nQ{i+1}: {ex.question[:70]}...")
        print(f"  Gold:    {ex.response[:100]}")
        print(f"  Predicted: {pred.answer[:100]}")
        print(f"  Semantic F1: {score:.3f}")

    if teleprompter == "none":
        print("\nSkipping compilation (teleprompter=none). Showing unoptimized program.")
        print("\n=== Unoptimized ChainOfThought prompt ===")
        print(unoptimized.generate_answer.demos)
        return

    # 5. Compile with teleprompter
    print(f"\n>>> Compiling with {teleprompter} teleprompter...")
    print(f"   Training examples: {len(trainset)}")
    print(f"   Validation examples: {len(valset)}")

    if teleprompter == "fewshot":
        from dspy.teleprompt import BootstrapFewShot

        teleprompter_obj = BootstrapFewShot(
            metric=validate_context_and_answer,
            max_labeled_demos=8,       # max number of labeled demos to use
            max_rounds=3,              # max refinement rounds
            max_errors=50,             # tolerate errors (LLM timeouts, etc.)
        )

    elif teleprompter == "random":
        from dspy.teleprompt import BootstrapFewShotWithRandomSearch

        teleprompter_obj = BootstrapFewShotWithRandomSearch(
            metric=validate_answer_only,  # looser metric for search
            max_bootstrap_demos=6,
            num_candidate_programs=4,     # try 4 candidate programs
            num_threads=8,
            max_errors=50,
            output_buffer_size=6,
        )

    else:
        raise ValueError(f"Unknown teleprompter: {teleprompter}")

    # Compile: this is where the magic happens
    # The teleprompter:
    #   1. Runs the unoptimized program on training examples
    #   2. Collects successful traces as few-shot demos
    #   3. Refines the demos over multiple rounds
    #   4. Returns a "compiled" program with optimized prompts
    compiled = teleprompter_obj.compile(
        UnoptimizedRAG(retriever=retriever, max_context_length=512),
        trainset=trainset,
    )

    print(f"\nCompiled program has {len(compiled.generate_answer.demos)} labeled demos")

    # Show what the teleprompter learned
    if compiled.generate_answer.demos:
        print("\n=== Teleprompter-learned demos (first 2) ===")
        for i, demo in enumerate(compiled.generate_answer.demos[:2]):
            print(f"\n--- Demo {i+1} ---")
            print(f"Context: {demo.context[:200]}...")
            print(f"Question: {demo.question}")
            print(f"Answer: {demo.answer}")

    # 6. Evaluate optimized pipeline
    print("\n>>> Evaluating OPTIMIZED pipeline")
    optimized_score, optimized_scores = evaluate_pipeline(
        compiled, testset, combined_metric,
        name="Optimized", num_examples=50, display_table=3
    )

    # 7. Compare
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Baseline score:     {baseline_score:.4f}")
    print(f"Optimized score:    {optimized_score:.4f}")
    delta = optimized_score - baseline_score
    pct = (delta / baseline_score * 100) if baseline_score > 0 else 0
    direction = "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
    print(f"Change:             {delta:+.4f} ({pct:+.1f}%) — {direction}")
    print("=" * 60)

    # 8. Show individual optimized predictions
    print("\n--- Optimized: individual examples ---")
    for i, ex in enumerate(testset[:5]):
        pred = compiled(question=ex.question)
        score = semantic_f1_metric(ex, pred)
        print(f"\nQ{i+1}: {ex.question[:70]}...")
        print(f"  Gold:    {ex.response[:100]}")
        print(f"  Predicted: {pred.answer[:100]}")
        print(f"  Semantic F1: {score:.3f}")

    # 9. Run on a human-readable example
    print("\n" + "=" * 60)
    print("HUMAN-READABLE EXAMPLE")
    print("=" * 60)
    test_q = testset[0].question
    print(f"\nQuestion: {test_q}")

    pred_base = unoptimized(question=test_q)
    print(f"\n[BASELINE]  {pred_base.answer[:300]}")

    pred_opt = compiled(question=test_q)
    print(f"\n[OPTIMIZED] {pred_opt.answer[:300]}")

    # 10. Inspect the full compiled program structure
    print("\n" + "=" * 60)
    print("COMPILED PROGRAM STRUCTURE")
    print("=" * 60)
    print(f"generate_answer type: {type(compiled.generate_answer).__name__}")
    print(f"Number of demos: {len(compiled.generate_answer.demos)}")
    if compiled.generate_answer.demos:
        first_demo = compiled.generate_answer.demos[0]
        print(f"Demo context keys: {first_demo.context.keys() if hasattr(first_demo.context, 'keys') else 'N/A'}")
        if hasattr(first_demo.context, 'keys'):
            for k in list(first_demo.context.keys())[:5]:
                print(f"  {k}: {str(first_demo.context[k])[:100]}")

    return {
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "delta": delta,
        "pct_change": pct,
        "unoptimized": unoptimized,
        "compiled": compiled,
        "baseline_scores": baseline_scores,
        "optimized_scores": optimized_scores,
    }


# ---------------------------------------------------------------------------
# Demo mode — single example, no optimization
# ---------------------------------------------------------------------------

def run_demo():
    """Run a single RAG example with both unoptimized and ChainOfThought."""
    model = 'ollama_chat/llama3.2:latest'
    lm = dspy.LM(model)
    dspy.configure(lm=lm)

    trainset, valset, devset, testset = load_ragqa_arena_tech_500()
    docs = load_corpus()
    if docs is None:
        return

    emb_model, embedder = create_embedder(dimension=512)
    retriever = create_retriever(docs, embedder)

    # Unoptimized
    unopt = UnoptimizedRAG(retriever=retriever)
    ex = testset[0]

    print("=" * 60)
    print("DSPy RAG Demo (single example)")
    print("=" * 60)
    print(f"\nQuestion: {ex.question}")
    print(f"Gold answer: {ex.response[:200]}")

    pred = unopt(question=ex.question)
    print(f"\nPredicted answer:\n{pred.answer}")

    # Show retrieved context
    print(f"\n--- Retrieved context ({len(pred.context)} passages) ---")
    for i, ctx in enumerate(pred.context):
        print(f"\n[Passage {i+1}]: {ctx[:300]}")

    # Compare with plain ChainOfThought (no retrieval)
    print("\n--- Plain ChainOfThought (no retrieval) ---")
    cot = dspy.ChainOfThought('question -> answer')
    plain = cot(question=ex.question)
    print(f"Plain CoT answer:\n{plain.answer}")


# ---------------------------------------------------------------------------
# Teleprompter comparison mode
# ---------------------------------------------------------------------------

def compare_teleprompters(teleprompters: list[str] = None):
    """Compare multiple teleprompters on the same RAG pipeline."""
    if teleprompters is None:
        teleprompters = ["fewshot", "random"]

    model = 'ollama_chat/llama3.2:latest'
    lm = dspy.LM(model)
    dspy.configure(lm=lm)

    trainset, valset, devset, testset = load_ragqa_arena_tech_500()
    docs = load_corpus()
    if docs is None:
        return

    emb_model, embedder = create_embedder(dimension=512)
    retriever = create_retriever(docs, embedder)

    results = {}

    for tp_name in teleprompters:
        print(f"\n{'='*60}")
        print(f"Testing teleprompter: {tp_name}")
        print(f"{'='*60}")

        if tp_name == "fewshot":
            from dspy.teleprompt import BootstrapFewShot
            tp = BootstrapFewShot(metric=validate_context_and_answer, max_labeled_demos=8)

        elif tp_name == "random":
            from dspy.teleprompt import BootstrapFewShotWithRandomSearch
            tp = BootstrapFewShotWithRandomSearch(
                metric=validate_answer_only,
                num_candidate_programs=4,
                max_bootstrap_demos=6,
            )

        elif tp_name == "none":
            results[tp_name] = None
            continue

        compiled = tp.compile(
            OptimizedRAG(retriever=retriever),
            trainset=trainset,
        )

        score, _ = evaluate_pipeline(compiled, testset, combined_metric,
                                     name=f"Teleprompter: {tp_name}", num_examples=50)
        results[tp_name] = score

    # Summary table
    print("\n" + "=" * 60)
    print("TELEPROMPTER COMPARISON")
    print("=" * 60)
    print(f"{'Teleprompter':<20} {'Score':>10}")
    print("-" * 30)
    for name, score in results.items():
        if score is not None:
            print(f"{name:<20} {score:>10.4f}")
        else:
            print(f"{name:<20} {'(skipped)':>10}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DSPy optimization experiments for RAGMail")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline evaluation")
    parser.add_argument("--teleprompter", choices=["fewshot", "random", "none"], default="fewshot",
                        help="Teleprompter to use for optimization (default: fewshot)")
    parser.add_argument("--demo", action="store_true", help="Run single example demo")
    parser.add_argument("--compare-tps", nargs="*", metavar="TP", help="Compare teleprompters")
    parser.add_argument("--num-examples", type=int, default=50, help="Number of examples for evaluation")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.compare_tps:
        compare_teleprompters(args.compare_tps)
    elif args.baseline_only:
        # Quick baseline only
        model = 'ollama_chat/llama3.2:latest'
        lm = dspy.LM(model)
        dspy.configure(lm=lm)
        trainset, valset, devset, testset = load_ragqa_arena_tech_500()
        docs = load_corpus()
        if docs:
            emb_model, embedder = create_embedder(dimension=512)
            retriever = create_retriever(docs, embedder)
            unopt = UnoptimizedRAG(retriever=retriever)
            evaluate_pipeline(unopt, testset, combined_metric, name="Baseline",
                            num_examples=args.num_examples)
    else:
        run_optimization(teleprompter=args.teleprompter)
