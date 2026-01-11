import json
import textwrap
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from src.rag_pipeline import RAGChatbot


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
            target_id = item["context_id"]  # The correct chunk ID from your golden set

            try:
                # 1. Reset History (Critical for independent evaluation)
                self.bot.clear_history()

                # 2. Run Chat
                response, sources = self.bot.chat(
                    question, k=5, use_full_doc=use_full_doc, verbose=False
                )

                # 3. Calculate Hit Rate (Recall)
                # Did any of the retrieved chunks match the target_id?
                # We check if target_id is a substring of retrieved_id to handle cases like "art_5" matching "art_5_para_1"
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
        """
        results = []
        subset = self.golden_set[:limit] if limit else self.golden_set

        print(f"📉 Running Baseline (No Context) on {len(subset)} queries...")

        for item in tqdm(subset):
            question = item["question"]
            ground_truth = item["ground_truth_answer"]

            # Construct a simple prompt bypassing the RAG context logic
            prompt = textwrap.dedent(
                f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are an expert on the EU AI Act. Answer the question accurately based on your internal knowledge.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                {question}
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
            ).strip()

            inputs = self.bot.tokenizer(prompt, return_tensors="pt").to(self.bot.device)

            with torch.no_grad():
                outputs = self.bot.model.generate(
                    **inputs, max_new_tokens=256, temperature=0.1
                )

            answer = self.bot.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            results.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated_answer": answer,
                    "retrieved_ids": [],
                    "target_id": "N/A",
                    "hit": False,  # Baseline cannot "hit" a document
                    "model_type": "Baseline (LLM Only)",
                    "sources_count": 0,
                }
            )

        return pd.DataFrame(results)
