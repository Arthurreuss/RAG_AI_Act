import json
import textwrap

import pandas as pd
from tqdm import tqdm

from src.rag.rag_pipeline import RAGChatbot


class RAGEvaluator:
    def __init__(self, chatbot: RAGChatbot, golden_set_path: str):
        """
        Args:
            chatbot: An initialized RAGChatbot instance.
            golden_set_path: Path to the JSON file with test questions.
        """
        self.bot = chatbot
        with open(golden_set_path, "r") as f:
            self.golden_set = json.load(f)

        print(f"📊 Loaded {len(self.golden_set)} test cases.")

    def evaluate_rag(
        self, limit: int = None, use_full_doc: bool = False, label: str = "RAG"
    ):
        """
        Runs the RAG pipeline on the golden set.
        """
        results = []
        subset = self.golden_set[:limit] if limit else self.golden_set

        print(
            f"🚀 Running {label} evaluation on {len(subset)} queries (Full Doc: {use_full_doc})..."
        )

        for i, item in enumerate(tqdm(subset)):
            question = item["question"]
            ground_truth = item["ground_truth_answer"]
            target_id = item["context_id"]

            try:
                # 1. Reset History
                self.bot.clear_history()

                # 2. Run Chat
                response, sources = self.bot.chat(
                    question, k=5, use_full_doc=use_full_doc
                )

                # 3. Calculate Hit Rate
                retrieved_ids = [s.get("chunk_id", "") for s in sources]
                hit = any(target_id in rid for rid in retrieved_ids)

                results.append(
                    {
                        "question": question,
                        "ground_truth": ground_truth,
                        "generated_answer": response,
                        "retrieved_ids": retrieved_ids,
                        "target_id": target_id,
                        "hit": hit,
                        "model_type": label,
                        "sources_count": len(sources),
                    }
                )

            except Exception as e:
                print(f"❌ Error on item {i}: {e}")

        return pd.DataFrame(results)

    def evaluate_baseline(self, limit: int = None):
        """
        Runs the LLM *without* any retrieved context (Zero-Shot).
        Adapted for Llama.cpp (GGUF).
        """
        results = []
        subset = self.golden_set[:limit] if limit else self.golden_set

        print(f"Running Baseline (No Context) on {len(subset)} queries...")

        for item in tqdm(subset):
            question = item["question"]
            ground_truth = item["ground_truth_answer"]

            # Construct a simple prompt bypassing the RAG context logic
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert on the EU AI Act. Answer the question accurately based on your internal knowledge.",
                },
                {"role": "user", "content": question},
            ]

            # --- FIX: Use llama_cpp logic instead of transformers ---
            try:
                response = self.bot.llm.create_chat_completion(
                    messages=messages, max_tokens=256, temperature=0.1
                )
                answer = response["choices"][0]["message"]["content"]
            except AttributeError:
                # Fallback if self.bot.llm isn't accessible directly, or if using a different backend
                print(
                    "Error: Could not access self.bot.llm. Make sure you are using the GGUF RAGChatbot."
                )
                answer = "Error"
            # --------------------------------------------------------

            results.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated_answer": answer,
                    "retrieved_ids": [],
                    "target_id": "N/A",
                    "hit": False,
                    "model_type": "Baseline (LLM Only)",
                    "sources_count": 0,
                }
            )

        return pd.DataFrame(results)
