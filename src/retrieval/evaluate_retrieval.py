import math
from typing import Any, Dict, List, Union

import numpy as np
from tqdm import tqdm


def evaluate_retrieval(
    engine: Any, golden_set: List[Dict[str, Any]], k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """Runs comprehensive retrieval metrics on a golden dataset.

    This function calculates standard Information Retrieval (IR) metrics to assess
    how well the vector engine finds relevant context chunks for a given set of
    test questions.



    Args:
        engine: The search engine instance (e.g., EmbeddingWithDB) that has a
            `.search(query, k)` method.
        golden_set (List[Dict[str, Any]]): A list of test cases, where each case
            is a dictionary containing 'question' and 'context_id'.
        k_values (List[int]): The specific 'k' thresholds at which to calculate
            metrics. Defaults to [1, 3, 5, 10].

    Returns:
        Dict[str, float]: A dictionary containing calculated metrics:
            - Recall@k: Proportion of queries where the target was in top k.
            - Precision@k: Average precision at rank k.
            - NDCG@k: Ranking quality considering the position of the target.
            - MRR: Mean Reciprocal Rank across all queries.
    """
    total_queries: int = len(golden_set)

    results: Dict[str, Any] = {
        "Recall": {k: 0.0 for k in k_values},
        "Precision": {k: 0.0 for k in k_values},
        "NDCG": {k: 0.0 for k in k_values},
        "MRR": 0.0,
    }

    print(f"Evaluating {total_queries} queries...")

    for item in tqdm(golden_set):
        query: str = item["question"]
        target_id: str = item["context_id"]

        max_k: int = max(k_values)
        search_res: List[Dict[str, Any]] = engine.search(query, k=max_k)
        retrieved_ids: List[str] = [res["chunk_id"] for res in search_res]

        # Calculate MRR (Mean Reciprocal Rank)
        # MRR = 1/n * Σ (1 / rank_i)
        if target_id in retrieved_ids:
            rank: int = retrieved_ids.index(target_id) + 1
            results["MRR"] += 1.0 / rank
        else:
            rank = None

        for k in k_values:
            top_k_ids: List[str] = retrieved_ids[:k]

            if rank is not None and target_id in top_k_ids:
                # Recall@k (Hit Rate)
                results["Recall"][k] += 1

                # Precision@k
                results["Precision"][k] += 1.0 / k

                # NDCG@k (Normalized Discounted Cumulative Gain)
                # For single relevant item, DCG = 1 / log2(rank + 1)
                # Since IDCG is 1 (perfect rank at 1), NDCG = DCG
                results["NDCG"][k] += 1.0 / math.log2(rank + 1)

    final_metrics: Dict[str, float] = {}

    for k in k_values:
        final_metrics[f"Recall@{k}"] = round(results["Recall"][k] / total_queries, 3)
        final_metrics[f"Precision@{k}"] = round(
            results["Precision"][k] / total_queries, 3
        )
        final_metrics[f"NDCG@{k}"] = round(results["NDCG"][k] / total_queries, 3)

    final_metrics["MRR"] = round(results["MRR"] / total_queries, 3)

    return final_metrics
