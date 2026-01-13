import json
import os
from datetime import datetime

import pandas as pd

from src.rag.rag_evaluator import RAGEvaluator
from src.rag.rag_pipeline import RAGChatbot
from src.utils.helper import load_config


def save_results(df: pd.DataFrame, output_dir: str = "results"):
    """
    Saves evaluation results in two formats:
    1. CSV: Numerical metrics only (clean for plotting).
    2. JSON: EVERYTHING (Text + Numbers) for deep analysis.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    target_metric_cols = [
        "hit_rank",
        "rag_score",
        "baseline_score",
        "sources_count",
        "winner",
    ]

    metric_cols = [c for c in target_metric_cols if c in df.columns]

    df_metrics = df[metric_cols].copy()
    df_metrics.reset_index(names=["id"], inplace=True)

    csv_filename = (
        f"rag_eval_metrics_single_turn_queries_reranked_{timestamp}.csv"  # change
    )
    csv_path = os.path.join(output_dir, csv_filename)
    df_metrics.to_csv(csv_path, index=False)

    records = df.to_dict(orient="records")

    json_filename = f"rag_eval_full_single_turn_reranked_{timestamp}.json"  # change
    json_path = os.path.join(output_dir, json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    print(f"\nSaved Metrics (CSV):  {csv_path}")
    print(f"Saved Complete (JSON): {json_path}")

    return csv_path, json_path


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    test_path = cfg["test_set"]["single_turn_path"]  # change

    bot = RAGChatbot(cfg, verbose=False)
    evaluator = RAGEvaluator(bot, test_path)

    print("\n" + "=" * 60)
    print("Starting Incremental Evaluation Pipeline")
    print("   1. RAG Retrieval & Generation")
    print("   2. Baseline (Zero-Shot) Generation")
    print("   3. Combined Judge (RAG vs Baseline vs Truth)")
    print("=" * 60)

    limit = None
    df_results = evaluator.evaluate_all(limit=limit)

    if not df_results.empty:
        hits = df_results[df_results["hit_rank"] > 0]
        hit_rate = (len(hits) / len(df_results)) * 100

        avg_rag = df_results["rag_score"].mean() if "rag_score" in df_results else 0
        avg_base = (
            df_results["baseline_score"].mean() if "baseline_score" in df_results else 0
        )

        print("\n" + "-" * 30)
        print("Evaluation Summary")
        print("-" * 30)
        print(f"Total Queries:       {len(df_results)}")
        print(f"Retrieval Hit Rate:  {hit_rate:.1f}%")
        print(f"Avg. RAG Score:      {avg_rag:.2f} / 10.0")
        print(f"Avg. Baseline Score: {avg_base:.2f} / 10.0")
        print("-" * 30)

        interesting = df_results[
            (df_results["hit_rank"] > 0)
            & (df_results["rag_score"] > df_results["baseline_score"])
        ]

        if not interesting.empty:
            row = interesting.iloc[0]
            print("\nHighlight: RAG Victory (Context Helped!)")
            print(f"Q: {row['question']}")
            print(
                f"Scores: RAG [{row['rag_score']}] vs Baseline [{row['baseline_score']}]"
            )
            print(f"Judge Analysis: {row['judge_analysis']}")
        else:
            print("\nSample Result (First item):")
            cols = ["question", "hit_rank", "rag_score", "baseline_score"]
            existing = [c for c in cols if c in df_results.columns]
            print(df_results[existing].head(1).to_string())

        save_results(df_results)

    else:
        print("No results generated.")
