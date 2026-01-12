from src.rag.rag_evaluator import RAGEvaluator
from src.rag.rag_pipeline import RAGChatbot
from src.utils.helper import load_config

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    bot = RAGChatbot(cfg)
    evaluator = RAGEvaluator(bot, cfg["test_set_path"])

    print("\n" + "=" * 50)
    print("Running RAG Evaluation (Retrieval + Generation)")
    print("=" * 50)

    df_rag = evaluator.evaluate_rag(limit=10, label="RAG")

    if not df_rag.empty and "hit" in df_rag.columns:
        hit_rate = df_rag["hit"].mean() * 100
        print(f"\nRAG Retrieval Hit Rate: {hit_rate:.1f}%")

        print("\nSample RAG Results (First 3):")
        print(
            df_rag[["question", "hit", "target_id", "generated_answer"]]
            .head(3)
            .to_string()
        )
    else:
        print("❌ No RAG results generated.")

    print("\n" + "=" * 50)
    print("Running Baseline Evaluation (Zero-Shot / No Context)")
    print("=" * 50)

    df_baseline = evaluator.evaluate_baseline(limit=10)

    if not df_baseline.empty:
        print("\nSample Baseline Results (First 3):")
        print(df_baseline[["question", "generated_answer"]].head(3).to_string())
    else:
        print("❌ No Baseline results generated.")
