"""
Retrieval evaluation metrics for RAG system.
Implements Precision@k, Recall@k, MRR, and NDCG.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import json
import os


@dataclass
class RetrievalResult:
    """Container for a single retrieval result."""
    query: str
    retrieved_ids: List[str]
    retrieved_scores: List[float]
    relevant_ids: Set[str]


@dataclass 
class EvaluationResults:
    """Container for evaluation metrics."""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    num_queries: int
    
    def to_dict(self) -> Dict:
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "num_queries": self.num_queries
        }
    
    def __str__(self) -> str:
        lines = [
            f"Evaluation Results (n={self.num_queries} queries)",
            "-" * 40,
            f"MRR: {self.mrr:.4f}",
            "",
            "Precision@k:"
        ]
        for k, v in sorted(self.precision_at_k.items()):
            lines.append(f"  P@{k}: {v:.4f}")
        lines.append("")
        lines.append("Recall@k:")
        for k, v in sorted(self.recall_at_k.items()):
            lines.append(f"  R@{k}: {v:.4f}")
        lines.append("")
        lines.append("NDCG@k:")
        for k, v in sorted(self.ndcg_at_k.items()):
            lines.append(f"  NDCG@{k}: {v:.4f}")
        return "\n".join(lines)


def is_relevant(retrieved_id: str, relevant_prefixes: Set[str]) -> bool:
    """
    Check if a retrieved ID matches any relevant prefix.
    Supports both exact matching and prefix matching for flexible citation matching.
    
    Args:
        retrieved_id: The ID of a retrieved document
        relevant_prefixes: Set of relevant IDs/prefixes (ground truth)
        
    Returns:
        True if retrieved_id matches any relevant prefix
    """
    for prefix in relevant_prefixes:
        # Exact match
        if retrieved_id == prefix:
            return True
        # Prefix match (e.g., "Article 5:" matches "Article 5:Prohibited AI practices")
        if retrieved_id.startswith(prefix):
            return True
    return False


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Precision@k: proportion of top-k retrieved documents that are relevant.
    
    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs (ground truth) - supports prefix matching
        k: Number of top results to consider
        
    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if k <= 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if is_relevant(doc_id, relevant_ids))
    return relevant_in_top_k / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Recall@k: proportion of relevant documents found in top-k results.
    
    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs (ground truth) - supports prefix matching
        k: Number of top results to consider
        
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0
    
    top_k = retrieved_ids[:k]
    # Count how many relevant prefixes have at least one match in top_k
    relevant_found = sum(1 for prefix in relevant_ids if any(is_relevant(doc_id, {prefix}) for doc_id in top_k))
    return relevant_found / len(relevant_ids)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Calculate Reciprocal Rank: 1/rank of first relevant document.
    
    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs (ground truth) - supports prefix matching
        
    Returns:
        Reciprocal rank score (0.0 to 1.0)
    """
    for i, doc_id in enumerate(retrieved_ids):
        if is_relevant(doc_id, relevant_ids):
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at k.
    
    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs (ground truth) - supports prefix matching
        k: Number of top results to consider
        
    Returns:
        DCG@k score
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        relevance = 1.0 if is_relevant(doc_id, relevant_ids) else 0.0
        # Using log2(i + 2) to avoid log(1) = 0
        dcg += relevance / np.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    
    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Number of top results to consider
        
    Returns:
        NDCG@k score (0.0 to 1.0)
    """
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    
    # Ideal DCG: all relevant docs at the top
    ideal_retrieved = list(relevant_ids)[:k]
    idcg = dcg_at_k(ideal_retrieved, relevant_ids, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(
    results: List[RetrievalResult],
    k_values: List[int] = [1, 3, 5, 10]
) -> EvaluationResults:
    """
    Evaluate retrieval results across multiple queries.
    
    Args:
        results: List of RetrievalResult objects
        k_values: List of k values to compute metrics for
        
    Returns:
        EvaluationResults with aggregated metrics
    """
    if not results:
        return EvaluationResults(
            precision_at_k={k: 0.0 for k in k_values},
            recall_at_k={k: 0.0 for k in k_values},
            mrr=0.0,
            ndcg_at_k={k: 0.0 for k in k_values},
            num_queries=0
        )
    
    # Collect metrics for each query
    precisions = {k: [] for k in k_values}
    recalls = {k: [] for k in k_values}
    rrs = []
    ndcgs = {k: [] for k in k_values}
    
    for result in results:
        for k in k_values:
            precisions[k].append(precision_at_k(result.retrieved_ids, result.relevant_ids, k))
            recalls[k].append(recall_at_k(result.retrieved_ids, result.relevant_ids, k))
            ndcgs[k].append(ndcg_at_k(result.retrieved_ids, result.relevant_ids, k))
        
        rrs.append(reciprocal_rank(result.retrieved_ids, result.relevant_ids))
    
    # Compute means
    return EvaluationResults(
        precision_at_k={k: np.mean(precisions[k]) for k in k_values},
        recall_at_k={k: np.mean(recalls[k]) for k in k_values},
        mrr=np.mean(rrs),
        ndcg_at_k={k: np.mean(ndcgs[k]) for k in k_values},
        num_queries=len(results)
    )


class RetrievalEvaluator:
    """
    Evaluator class for running retrieval evaluation experiments.
    """
    
    def __init__(self, test_set_path: Optional[str] = None):
        """
        Initialize evaluator with optional test set.
        
        Args:
            test_set_path: Path to JSON file with test questions
        """
        self.test_questions: List[Dict] = []
        if test_set_path and os.path.exists(test_set_path):
            self.load_test_set(test_set_path)
    
    def load_test_set(self, path: str) -> None:
        """Load test questions from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.test_questions = data.get('questions', data)
    
    def save_test_set(self, path: str) -> None:
        """Save test questions to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'questions': self.test_questions}, f, indent=2, ensure_ascii=False)
    
    def add_question(
        self, 
        question: str, 
        relevant_chunk_ids: List[str],
        expected_answer: Optional[str] = None,
        category: Optional[str] = None
    ) -> None:
        """Add a test question with ground truth."""
        self.test_questions.append({
            'question': question,
            'relevant_chunk_ids': relevant_chunk_ids,
            'expected_answer': expected_answer,
            'category': category
        })
    
    def evaluate_retriever(
        self,
        retriever_func,
        k_values: List[int] = [1, 3, 5, 10],
        top_k: int = 10
    ) -> Tuple[EvaluationResults, List[Dict]]:
        """
        Evaluate a retriever function against the test set.
        
        Args:
            retriever_func: Function that takes a query string and returns 
                           list of (chunk_id, score) tuples
            k_values: k values to compute metrics for
            top_k: Number of results to retrieve per query
            
        Returns:
            Tuple of (EvaluationResults, list of per-query details)
        """
        results = []
        details = []
        
        for test_q in self.test_questions:
            question = test_q['question']
            relevant_ids = set(test_q['relevant_chunk_ids'])
            
            # Call retriever
            retrieved = retriever_func(question, top_k)
            
            if isinstance(retrieved[0], tuple):
                retrieved_ids = [r[0] for r in retrieved]
                retrieved_scores = [r[1] for r in retrieved]
            else:
                retrieved_ids = retrieved
                retrieved_scores = list(range(len(retrieved), 0, -1))
            
            result = RetrievalResult(
                query=question,
                retrieved_ids=retrieved_ids,
                retrieved_scores=retrieved_scores,
                relevant_ids=relevant_ids
            )
            results.append(result)
            
            # Per-query details for error analysis
            hits = [rid for rid in retrieved_ids if is_relevant(rid, relevant_ids)]
            misses = [rel_id for rel_id in relevant_ids if not any(is_relevant(rid, {rel_id}) for rid in retrieved_ids)]
            
            details.append({
                'question': question,
                'category': test_q.get('category'),
                'retrieved_ids': retrieved_ids,
                'relevant_ids': list(relevant_ids),
                'hits': hits,
                'misses': misses,
                'rr': reciprocal_rank(retrieved_ids, relevant_ids),
                'p_at_3': precision_at_k(retrieved_ids, relevant_ids, 3),
                'r_at_3': recall_at_k(retrieved_ids, relevant_ids, 3)
            })
        
        eval_results = evaluate_retrieval(results, k_values)
        return eval_results, details


def compare_retrievers(
    evaluator: RetrievalEvaluator,
    retrievers: Dict[str, callable],
    k_values: List[int] = [1, 3, 5],
    top_k: int = 10
) -> Dict[str, EvaluationResults]:
    """
    Compare multiple retriever configurations.
    
    Args:
        evaluator: RetrievalEvaluator with loaded test set
        retrievers: Dict mapping retriever names to retriever functions
        k_values: k values to compute metrics for
        top_k: Number of results to retrieve
        
    Returns:
        Dict mapping retriever names to EvaluationResults
    """
    results = {}
    for name, retriever_func in retrievers.items():
        print(f"Evaluating: {name}")
        eval_results, _ = evaluator.evaluate_retriever(retriever_func, k_values, top_k)
        results[name] = eval_results
        print(f"  MRR: {eval_results.mrr:.4f}, P@3: {eval_results.precision_at_k[3]:.4f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Retrieval Metrics Module")
    print("=" * 40)
    
    # Demo with synthetic data
    retrieved = ["doc1", "doc3", "doc5", "doc2", "doc7"]
    relevant = {"doc1", "doc2", "doc4"}
    
    print(f"Retrieved: {retrieved}")
    print(f"Relevant:  {relevant}")
    print()
    print(f"Precision@1: {precision_at_k(retrieved, relevant, 1):.4f}")
    print(f"Precision@3: {precision_at_k(retrieved, relevant, 3):.4f}")
    print(f"Precision@5: {precision_at_k(retrieved, relevant, 5):.4f}")
    print()
    print(f"Recall@1: {recall_at_k(retrieved, relevant, 1):.4f}")
    print(f"Recall@3: {recall_at_k(retrieved, relevant, 3):.4f}")
    print(f"Recall@5: {recall_at_k(retrieved, relevant, 5):.4f}")
    print()
    print(f"MRR: {reciprocal_rank(retrieved, relevant):.4f}")
    print(f"NDCG@5: {ndcg_at_k(retrieved, relevant, 5):.4f}")
