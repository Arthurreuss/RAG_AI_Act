import math

import numpy as np
from tqdm import tqdm


def evaluate_retrieval(engine, golden_set, k_values=[1, 3, 5, 10]):
    """
    Runs comprehensive retrieval metrics on a golden dataset.

    Metrics Calculated:
    1. Recall@k (Hit Rate): Did the correct chunk appear in top k?
    2. Precision@k: (Relevant Items / k). For single-label RAG, this is 1/k if found, else 0.
    3. MRR (Mean Reciprocal Rank): 1 / Rank of first correct answer.
    4. NDCG@k (Normalized Discounted Cumulative Gain): Measures ranking quality with position decay.
    """
    total_queries = len(golden_set)

    results = {
        "Recall": {k: 0 for k in k_values},
        "Precision": {k: 0 for k in k_values},
        "NDCG": {k: 0 for k in k_values},
        "MRR": 0,
    }

    print(f"Evaluating {total_queries} queries...")

    for item in tqdm(golden_set):
        query = item["question"]
        target_id = item["context_id"]

        max_k = max(k_values)
        search_res = engine.search(query, k=max_k)
        retrieved_ids = [res["chunk_id"] for res in search_res]

        if target_id in retrieved_ids:
            rank = retrieved_ids.index(target_id) + 1
            results["MRR"] += 1.0 / rank
        else:
            rank = None

        for k in k_values:
            top_k_ids = retrieved_ids[:k]

            if target_id in top_k_ids:
                results["Recall"][k] += 1
                results["Precision"][k] += 1.0 / k
                results["NDCG"][k] += 1.0 / math.log2(rank + 1)
            else:
                pass

    final_metrics = {}

    for k in k_values:
        final_metrics[f"Recall@{k}"] = round(results["Recall"][k] / total_queries, 3)
        final_metrics[f"Precision@{k}"] = round(
            results["Precision"][k] / total_queries, 3
        )
        final_metrics[f"NDCG@{k}"] = round(results["NDCG"][k] / total_queries, 3)

    final_metrics["MRR"] = round(results["MRR"] / total_queries, 3)

    return final_metrics
