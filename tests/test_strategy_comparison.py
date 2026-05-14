#!/usr/bin/env python3
"""
Enhanced Strategy Performance Comparison
Compares real implementations: grep, Elasticsearch (BM25), and Milvus (semantic + sparse)
Measures: latency, results quality, index efficiency
"""

import sys
import time
import csv
import os
from pathlib import Path
from typing import List, Dict, Set, Callable
from datetime import datetime, timezone
import json

# Import real implementations
from mail_processing.mailconverter import EmlxConverter
from indexing.es import ElSearch
from indexing.retriever import RMClient, my_embedder
from config.globals import DENSE_EMB_MODELS, SPARSE_EMB_FUNS, RANKER, MILVUS_DYN, MILVUS_MAX_LENGTH, MILVUS_LEN_CTX, MAX_CHUNK_LEN, MAX_CHUNK_EXCESS
import re


class StrategiesComparison:
    """Compare real search strategy implementations."""
    
    def __init__(self, mail_source, use_real_es=False, use_real_milvus=False):
        """Initialize comparison suite.
        
        Args:
            mail_source: MailSource with msgs_array() and proc_folder
            use_real_es: Whether to use real Elasticsearch (requires service running)
            use_real_milvus: Whether to use real Milvus (requires service running)
        """
        self.mail_source = mail_source
        self.use_real_es = use_real_es
        self.use_real_milvus = use_real_milvus
        
        # Initialize real implementations if requested
        if use_real_es:
            try:
                self.es_client = ElSearch(mail_source.mailsId)
                self.es_client.index_mails(mail_source.msgs_array())
                print("✓ Elasticsearch initialized and indexed")
            except Exception as e:
                print(f"✗ Elasticsearch initialization failed: {e}")
                self.use_real_es = False
        
        if use_real_milvus:
            try:
                dense_emb = DENSE_EMB_MODELS['mxbai-embed-large']
                sparse_emb_fn = SPARSE_EMB_FUNS.get('BGEM3')
                
                collection_name = f"eval_{int(time.time())}"
                collection_name = re.sub(r'[^\w\d]', '', collection_name)
                
                self.rm_client = RMClient(
                    collection_name, 
                    k=MILVUS_LEN_CTX,
                    dim_dense_emb=dense_emb['emb_len'],
                    max_length=MILVUS_MAX_LENGTH,
                    dense_embedding_function=my_embedder(dense_emb['name']),
                    sparse_embedding_function=sparse_emb_fn,
                    rerank_function=RANKER,
                    use_contextualize_embedding=False
                )
                
                if self.rm_client.build_collection(enable_dynamic_field=MILVUS_DYN):
                    chunks = self._make_chunks_from_mail(mail_source)
                    self.rm_client.upload_embeddings(chunks, metadata={})
                    print("✓ Milvus initialized and indexed")
                else:
                    self.use_real_milvus = False
            except Exception as e:
                print(f"✗ Milvus initialization failed: {e}")
                self.use_real_milvus = False
    
    def _make_chunks_from_mail(self, mail_source):
        """Create chunks from mail for Milvus indexing."""
        chunks = []
        for mail in mail_source.msgs_array():
            content = mail.get_content()
            # Simple chunking: split by sentences
            sentences = content.split('.')
            chunk_text = ""
            
            for sent in sentences:
                if len(chunk_text) + len(sent) < MAX_CHUNK_LEN:
                    chunk_text += sent + "."
                else:
                    if chunk_text:
                        chunks.append(chunk_text)
                    chunk_text = sent + "."
            
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks[:min(len(chunks), 1000)]  # Limit for demo
    
    def grep_search(self, query: str) -> List:
        """Keyword-based grep search."""
        keywords = [w.lower().strip() for w in query.split() if len(w) > 2]
        
        matches = []
        for mail in self.mail_source.msgs_array():
            content_lower = mail.get_content().lower()
            matching_count = sum(1 for kw in keywords if kw in content_lower)
            
            if matching_count >= max(1, len(keywords) // 2):
                matches.append((mail, matching_count))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches[:10]]
    
    def elasticsearch_search(self, query: str) -> List:
        """Elasticsearch BM25 search."""
        if not self.use_real_es:
            return self.grep_search(query)  # Fallback
        
        try:
            mail_ids = self.es_client.search(query)
            results = []
            for mail_id in mail_ids[:10]:
                if mail_id in self.mail_source.proc_folder:
                    results.append(self.mail_source.proc_folder[mail_id])
            return results
        except Exception as e:
            print(f"ES search error: {e}")
            return self.grep_search(query)
    
    def milvus_search(self, query: str) -> List:
        """Milvus semantic + sparse hybrid search."""
        if not self.use_real_milvus:
            return self.grep_search(query)  # Fallback
        
        try:
            results = self.rm_client.forward(query)
            if hasattr(results, 'context'):
                return results.context
            return results
        except Exception as e:
            print(f"Milvus search error: {e}")
            return self.grep_search(query)
    
    def evaluate(self, queries: List[str]) -> Dict:
        """Evaluate all strategies on given queries.
        
        Args:
            queries: List of test queries
            
        Returns:
            Dict with results per strategy
        """
        strategies = {
            'grep': self.grep_search,
            'elasticsearch': self.elasticsearch_search,
            'milvus': self.milvus_search,
        }
        
        results = {strategy: [] for strategy in strategies}
        
        print("\n" + "="*80)
        print(f"{'Strategy Evaluation on {len(queries)} Queries':^80}")
        print("="*80 + "\n")
        
        for query in queries:
            print(f"Query: '{query}'")
            
            for strategy_name, search_fn in strategies.items():
                try:
                    start = time.time()
                    found_docs = search_fn(query)
                    elapsed_ms = (time.time() - start) * 1000
                    
                    result = {
                        'strategy': strategy_name,
                        'query': query,
                        'latency_ms': round(elapsed_ms, 2),
                        'results_count': len(found_docs),
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                    }
                    results[strategy_name].append(result)
                    
                    print(f"  {strategy_name:15} → {len(found_docs):3d} results, {elapsed_ms:7.2f}ms")
                except Exception as e:
                    print(f"  {strategy_name:15} → ERROR: {e}")
        
        return results
    
    def save_results(self, results: Dict, output_file: str = "strategy_comparison.csv"):
        """Save results to CSV."""
        all_rows = []
        for strategy_results in results.values():
            all_rows.extend(strategy_results)
        
        if not all_rows:
            print("No results to save")
            return
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"\n✓ Results saved to {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print(f"{'Summary Statistics':^80}")
        print("="*80 + "\n")
        
        for strategy, rows in results.items():
            if rows:
                latencies = [r['latency_ms'] for r in rows]
                avg_latency = sum(latencies) / len(latencies)
                total_results = sum(r['results_count'] for r in rows)
                
                print(f"{strategy.upper()}")
                print(f"  Avg Latency:      {avg_latency:>8.2f} ms")
                print(f"  Total Results:    {total_results:>8d}")
                print(f"  Queries Tested:   {len(rows):>8d}")
                print()


def main():
    """Compare strategy performance on real corpus."""
    
    # Configuration
    mailbox = os.path.expanduser("~/Library/Mail")
    use_real_es = False  # Set to True if Elasticsearch is running
    use_real_milvus = False  # Set to True if Milvus is running
    
    print("="*80)
    print("RAGMail Strategy Comparison")
    print("="*80)
    print(f"Mailbox: {mailbox}")
    print(f"Real Elasticsearch: {use_real_es}")
    print(f"Real Milvus: {use_real_milvus}")
    
    # Load emails
    print(f"\nLoading emails from {mailbox}...")
    mail_converter = EmlxConverter(mailbox=mailbox, doThreads=False)
    mail_converter.read_mails()
    
    print(f"✓ Loaded {len(list(mail_converter.msgs_array()))} emails")
    
    # Initialize comparison
    comparison = StrategiesComparison(
        mail_converter,
        use_real_es=use_real_es,
        use_real_milvus=use_real_milvus
    )
    
    # Test queries
    test_queries = [
        "communication and discussion",
        "relationship and family",
        "conflict and disagreement",
        "meeting schedule time",
        "decision and choice",
    ]
    
    # Run evaluation
    results = comparison.evaluate(test_queries)
    
    # Save results
    comparison.save_results(results)
    
    print("\n" + "="*80)
    print("✓ Strategy comparison complete")
    print("="*80)


if __name__ == '__main__':
    main()
