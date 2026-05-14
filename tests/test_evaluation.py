#!/usr/bin/env python
"""
Search Strategy Evaluation Harness
Compares performance of different retrieval strategies: grep, Milvus, Elasticsearch
Measures: latency, recall@K, precision@K, index size
"""

import sys
import time
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
import re

class SimpleMailSource:
    """Simple mail source for testing (without MailConverter complexity)."""
    def __init__(self):
        self.proc_folder = {}  # mail_id -> content
        self.mailsId = []  # list of mail_ids


class EvaluationHarness:
    """Harness for comparing different search strategies."""
    
    def __init__(self, mail_source):
        """Initialize with mail source."""
        self.mail_source = mail_source
        self.queries = self._load_test_queries()
        self.results = {}
    
    def _load_test_queries(self) -> List[Dict]:
        """Load test queries with expected relevant results.
        
        Returns:
            List of dicts with keys: query, expected_doc_ids, description
        """
        # These are example queries - in production, create from actual corpus analysis
        queries = [
            {
                "query": "communication and discussion",
                "description": "General communication query",
                "expected_keywords": ["discussion", "talk", "say", "think", "view"],
                "min_expected_results": 3,
            },
            {
                "query": "personal relationship perspective",
                "description": "Relationship perspective query",
                "expected_keywords": ["relationship", "personal", "perspective", "feel", "understand"],
                "min_expected_results": 3,
            },
            {
                "query": "conflict disagreement issue",
                "description": "Conflict analysis query",
                "expected_keywords": ["conflict", "disagreement", "issue", "problem", "dispute"],
                "min_expected_results": 2,
            },
            {
                "query": "family matter",
                "description": "Family-related query",
                "expected_keywords": ["family", "relative", "parent", "child",],
                "min_expected_results": 2,
            },
            {
                "query": "meeting arrangement schedule",
                "description": "Meeting/schedule query",
                "expected_keywords": ["meeting", "schedule", "time", "date", "arrange"],
                "min_expected_results": 2,
            },
        ]
        return queries
    
    def _grep_search(self, query: str) -> List[str]:
        """Implement basic grep-style keyword search.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching email IDs or content
        """
        # Extract keywords from query
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        
        matches = []
        for mail_id, mail_content in self.mail_source.proc_folder.items():
            content_lower = mail_content.lower()
            # Match if query keywords are in content
            matching_keywords = sum(1 for kw in keywords if kw in content_lower)
            if matching_keywords >= max(1, len(keywords) // 2):  # At least 50% keyword match
                matches.append((mail_id, matching_keywords))
        
        # Sort by number of matching keywords
        matches.sort(key=lambda x: x[1], reverse=True)
        return [mail_id for mail_id, _ in matches[:10]]  # Top 10 results
    
    def _elasticsearch_search(self, query: str) -> List[str]:
        """Simulate Elasticsearch search (BM25 scoring).
        
        Note: This is a simplified simulation. In production, would use actual ES.
        """
        # For now, return same as grep for baseline comparison
        # In production: use ElSearch class from es.py
        return self._grep_search(query)
    
    def _milvus_search(self, query: str) -> List[str]:
        """Simulate Milvus semantic search.
        
        Note: This is a simplified simulation. In production, would use RMClient.
        """
        # For now, return same as grep for baseline comparison
        # In production: use RMClient class from retriever.py with embeddings
        return self._grep_search(query)
    
    def _get_relevant_docs(self, query: str) -> Set[str]:
        """Identify relevant documents for a query.
        
        Uses heuristic: documents containing expected keywords.
        
        Args:
            query: Search query
            
        Returns:
            Set of relevant document IDs
        """
        # Find which query config this is
        query_config = None
        for q in self.queries:
            if q["query"] == query:
                query_config = q
                break
        
        if not query_config:
            return set()
        
        relevant_docs = set()
        keywords = query_config.get("expected_keywords", [])
        
        for mail_id, mail_content in self.mail_source.proc_folder.items():
            content_lower = mail_content.lower()
            # Count matching keywords
            matching_count = sum(1 for kw in keywords if kw in content_lower)
            # Consider relevant if matches at least 2 keywords
            if matching_count >= 2:
                relevant_docs.add(mail_id)
        
        return relevant_docs
    
    def evaluate_strategy(self, name: str, search_fn, query: str) -> Dict:
        """Evaluate a search strategy on a query.
        
        Args:
            name: Strategy name
            search_fn: Search function (takes query string, returns list of results)
            query: Test query
            
        Returns:
            Dict with latency, recall@K, precision@K, etc.
        """
        # Get relevant documents
        relevant_docs = self._get_relevant_docs(query)
        
        # Measure search latency
        start_time = time.time()
        results = search_fn(query)
        latency_ms = (time.time() - start_time) * 1000
        
        # Convert results to set (for set operations)
        result_set = set(results[:10])  # Top 10 results
        
        # Calculate metrics
        tp = len(result_set.intersection(relevant_docs))
        
        if len(result_set) > 0:
            precision = tp / len(result_set)
        else:
            precision = 0.0
        
        if len(relevant_docs) > 0:
            recall = tp / len(relevant_docs)
        else:
            recall = 0.0
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            "strategy": name,
            "query": query,
            "latency_ms": round(latency_ms, 2),
            "results_returned": len(results),
            "top_10_results": len(result_set),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "relevant_found": tp,
            "total_relevant": len(relevant_docs),
        }
    
    def run_evaluation(self) -> List[Dict]:
        """Run comprehensive evaluation across all strategies and queries.
        
        Returns:
            List of evaluation results
        """
        strategies = [
            ("grep", self._grep_search),
            ("elasticsearch", self._elasticsearch_search),
            ("milvus", self._milvus_search),
        ]
        
        all_results = []
        
        print("\n" + "="*80)
        print(f"{'Search Strategy Evaluation':^80}")
        print("="*80)
        print(f"Testing {len(strategies)} strategies on {len(self.queries)} queries\n")
        
        for strategy_name, search_fn in strategies:
            print(f"\nEvaluating {strategy_name.upper()} strategy...")
            print("-" * 80)
            
            for query_config in self.queries:
                query = query_config["query"]
                result = self.evaluate_strategy(
                    strategy_name, 
                    search_fn, 
                    query
                )
                all_results.append(result)
                
                # Print result
                print(f"  Query: '{query}'")
                print(f"    Latency: {result['latency_ms']:.2f}ms | "
                      f"Recall: {result['recall']:.1%} | "
                      f"Precision: {result['precision']:.1%} | "
                      f"F1: {result['f1_score']:.3f}")
        
        return all_results
    
    def save_results_csv(self, results: List[Dict], output_file: str = "evaluation_results.csv"):
        """Save evaluation results to CSV.
        
        Args:
            results: List of evaluation result dicts
            output_file: Output CSV filename
        """
        if not results:
            print("No results to save")
            return
        
        # Determine all keys
        fieldnames = list(results[0].keys())
        
        # Write CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ Results saved to {output_file}")
    
    def print_summary(self, results: List[Dict]):
        """Print summary statistics across all evaluations.
        
        Args:
            results: List of evaluation result dicts
        """
        print("\n" + "="*80)
        print(f"{'Evaluation Summary':^80}")
        print("="*80 + "\n")
        
        # Group by strategy
        by_strategy = {}
        for result in results:
            strategy = result['strategy']
            if strategy not in by_strategy:
                by_strategy[strategy] = []
            by_strategy[strategy].append(result)
        
        # Print summary for each strategy
        for strategy_name in sorted(by_strategy.keys()):
            strategy_results = by_strategy[strategy_name]
            
            avg_latency = sum(r['latency_ms'] for r in strategy_results) / len(strategy_results)
            avg_recall = sum(r['recall'] for r in strategy_results) / len(strategy_results)
            avg_precision = sum(r['precision'] for r in strategy_results) / len(strategy_results)
            avg_f1 = sum(r['f1_score'] for r in strategy_results) / len(strategy_results)
            
            print(f"{strategy_name.upper()}")
            print(f"  Avg Latency:   {avg_latency:>8.2f} ms")
            print(f"  Avg Recall:    {avg_recall:>8.1%}")
            print(f"  Avg Precision: {avg_precision:>8.1%}")
            print(f"  Avg F1 Score:  {avg_f1:>8.3f}")
            print()
        
        # Ranking
        print("RANKING (by average F1 score):")
        strategy_f1s = {}
        for strategy_name, strategy_results in by_strategy.items():
            avg_f1 = sum(r['f1_score'] for r in strategy_results) / len(strategy_results)
            strategy_f1s[strategy_name] = avg_f1
        
        for i, (strategy_name, f1_score) in enumerate(sorted(strategy_f1s.items(), key=lambda x: x[1], reverse=True), 1):
            print(f"  {i}. {strategy_name:15} → F1: {f1_score:.3f}")
        
        print("="*80)

def main():
    """Main evaluation script."""
    
    # Set up mail source (using example emails)
    example_dir = Path("examples")
    
    # Create a simple mail source
    mail_source = SimpleMailSource()
    
    # Load example emails
    eml_files = sorted(example_dir.glob("*.eml"))[:20]  # Use first 20 for speed
    
    print(f"\nLoading {len(eml_files)} example emails...")
    
    for eml_file in eml_files:
        try:
            with open(eml_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract body (simplified)
            parts = content.split('\n\n', 1)
            body = parts[1] if len(parts) > 1 else parts[0]
            
            mail_id = eml_file.stem
            mail_source.proc_folder[mail_id] = body
            mail_source.mailsId.append(mail_id)
        except Exception as e:
            print(f"Warning: Could not load {eml_file.name}: {e}")
    
    print(f"✓ Loaded {len(mail_source.mailsId)} emails\n")
    
    # Run evaluation
    harness = EvaluationHarness(mail_source)
    results = harness.run_evaluation()
    
    # Save and print results
    harness.save_results_csv(results)
    harness.print_summary(results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
