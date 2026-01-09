"""
Retrieval comparison script for Milestone 2.
Systematically compares naive vs semantic chunking strategies.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
import chromadb

from src.evaluation.retrieval_metrics import RetrievalEvaluator



def create_retriever(collection, embedding_model):
    """Create a retriever function for a ChromaDB collection."""
    def retriever(query: str, top_k: int):
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['metadatas', 'distances']
        )
        
        retrieved = []
        for i, meta in enumerate(results['metadatas'][0]):
            chunk_id = meta.get('citation', results['ids'][0][i])
            score = 1 - results['distances'][0][i]
            retrieved.append((chunk_id, score))
        
        return retrieved
    
    return retriever


def run_comparison():
    """Run full comparison between chunking strategies."""
    # Paths
    data_dir = PROJECT_ROOT / "data"
    chroma_path = data_dir / "chroma_db"
    test_path = data_dir / "evaluation" / "test_questions.json"
    
    # Check prerequisites
    if not chroma_path.exists():
        print("Error: ChromaDB not found. Run vector_store.py first.")
        return
    
    if not test_path.exists():
        print("Error: Test questions not found.")
        return
    
    # Initialize
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    # Get collections
    try:
        naive_collection = client.get_collection("ai_act_naive_chunks")
        semantic_collection = client.get_collection("ai_act_chunks")
        print(f"  Naive collection: {naive_collection.count()} documents")
        print(f"  Semantic collection: {semantic_collection.count()} documents")
    except Exception as e:
        print(f"Error loading collections: {e}")
        return
    
    # Load test set
    print("Loading test questions...")
    evaluator = RetrievalEvaluator(str(test_path))
    print(f"  Loaded {len(evaluator.test_questions)} questions")
    
    # Create retrievers
    retrievers = {
        'Naive': create_retriever(naive_collection, embedding_model),
        'Semantic': create_retriever(semantic_collection, embedding_model)
    }
    
    # Run comparison
    print("\n" + "="*60)
    print("RETRIEVAL COMPARISON: Naive vs Semantic Chunking")
    print("="*60 + "\n")
    
    k_values = [1, 3, 5, 10]
    results = {}
    all_details = {}
    
    for name, retriever_func in retrievers.items():
        print(f"\nEvaluating {name} chunking...")
        eval_results, details = evaluator.evaluate_retriever(
            retriever_func,
            k_values=k_values,
            top_k=10
        )
        results[name] = eval_results
        all_details[name] = details
        print(eval_results)
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"\n{'Metric':<15} {'Naive':>12} {'Semantic':>12} {'Winner':>12}")
    print("-" * 52)
    
    metrics = [
        ('MRR', 'mrr'),
        ('P@1', ('precision_at_k', 1)),
        ('P@3', ('precision_at_k', 3)),
        ('P@5', ('precision_at_k', 5)),
        ('R@3', ('recall_at_k', 3)),
        ('R@5', ('recall_at_k', 5)),
    ]
    
    for metric_name, metric_key in metrics:
        if isinstance(metric_key, tuple):
            naive_val = getattr(results['Naive'], metric_key[0]).get(metric_key[1], 0)
            semantic_val = getattr(results['Semantic'], metric_key[0]).get(metric_key[1], 0)
        else:
            naive_val = getattr(results['Naive'], metric_key)
            semantic_val = getattr(results['Semantic'], metric_key)
        
        winner = 'Naive' if naive_val > semantic_val else 'Semantic' if semantic_val > naive_val else 'Tie'
        print(f"{metric_name:<15} {naive_val:>12.4f} {semantic_val:>12.4f} {winner:>12}")
    
    # Save results
    output_path = data_dir / "evaluation" / "comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_json = {
        name: res.to_dict() for name, res in results.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print failure analysis
    print("\n" + "="*60)
    print("FAILURE ANALYSIS")
    print("="*60)
    
    for name, details in all_details.items():
        failures = [d for d in details if d['rr'] == 0]
        print(f"\n{name} Chunking - {len(failures)} complete failures:")
        for fail in failures[:3]:
            print(f"  - {fail['question'][:60]}...")
            print(f"    Expected: {fail['relevant_ids']}")


if __name__ == "__main__":
    run_comparison()
