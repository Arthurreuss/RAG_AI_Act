import re

import pandas as pd
from tqdm import tqdm

from src.rag.rag_pipeline import RAGChatbot
from src.utils.helper import load_json


class RAGEvaluator:
    def __init__(self, chatbot: RAGChatbot, golden_set_path: str):
        """
        Args:
            chatbot: An initialized RAGChatbot instance.
            golden_set_path: Path to the JSON file with test questions.
        """
        self.bot = chatbot
        raw_data = load_json(golden_set_path)

        if raw_data and isinstance(raw_data[0], list):
            self.mode = "multi_turn"
            self.golden_set = raw_data
            print(
                f"Detected MULTI-TURN dataset ({len(self.golden_set)} conversations)."
            )
        else:
            self.mode = "single_turn"
            self.golden_set = raw_data
            print(f"Detected SINGLE-TURN dataset ({len(self.golden_set)} questions).")

    def _calculate_hit_metrics(self, retrieved_ids: list, target_ids: list):
        """
        Helper to handle list-of-lists comparison.
        Returns:
            rank (int): 1-based index of the FIRST matching ID found (-1 if none).
            hit (bool): True if at least one target ID was found.
            found_count (int): How many distinct target IDs were found.
        """
        if not isinstance(target_ids, list):
            target_ids = [target_ids]

        best_rank = float("inf")
        found_set = set()

        for tid in target_ids:
            try:
                rank = retrieved_ids.index(tid) + 1
                if rank < best_rank:
                    best_rank = rank
                found_set.add(tid)
            except ValueError:
                continue

        hit = len(found_set) > 0
        final_rank = best_rank if hit else -1

        return final_rank, hit, len(found_set)

    def run_rag_inference(self, limit: int = None, use_full_doc: bool = False):
        """
        Step 1: Run RAG Pipeline.
        Handles nested loops for multi-turn to preserve history context.
        """
        results = []
        subset = self.golden_set[:limit] if limit else self.golden_set

        print(f"[1/3] Running RAG Inference ({self.mode})...")

        iterator_data = (
            subset if self.mode == "multi_turn" else [[item] for item in subset]
        )

        for conv_idx, conversation in enumerate(tqdm(iterator_data)):

            self.bot.clear_history()

            for turn_idx, item in enumerate(conversation):
                question = item["question"]
                target_ids = item["context_id"]
                ground_truth = item["ground_truth_answer"]

                conv_id = f"conv_{conv_idx}"

                try:
                    rag_response, sources = self.bot.chat(
                        question, use_full_doc=use_full_doc
                    )

                    retrieved_ids = [s.get("chunk_id", "") for s in sources]

                    rank, is_hit, found_count = self._calculate_hit_metrics(
                        retrieved_ids, target_ids
                    )

                    results.append(
                        {
                            "conversation_id": conv_id,
                            "turn_number": turn_idx + 1,
                            "question": question,
                            "ground_truth": ground_truth,
                            "target_ids": target_ids,
                            "rag_answer": rag_response,
                            "hit_rank": rank,
                            "is_hit": is_hit,
                            "retrieved_ids": retrieved_ids,
                            "found_count": found_count,
                        }
                    )

                except Exception as e:
                    print(f"❌ Error on {conv_id} turn {turn_idx}: {e}")
                    results.append(
                        {
                            "conversation_id": conv_id,
                            "question": question,
                            "rag_answer": "Error",
                            "hit_rank": -999,
                            "target_ids": target_ids,
                        }
                    )

        return pd.DataFrame(results)

    def run_baseline_inference(self, df_results: pd.DataFrame):
        """
        Step 2: Run Baseline (Zero-Shot).
        Adapted to maintain conversation history for the Baseline model too!
        """
        print(f"[2/3] Running Baseline Inference...")

        baseline_answers = []

        if "conversation_id" in df_results.columns:
            df_sorted = df_results.sort_values(by=["conversation_id", "turn_number"])
        else:
            df_sorted = df_results

        current_conv_id = None
        history_buffer = []

        for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted)):
            question = row["question"]
            row_conv_id = row.get("conversation_id", "single")

            if row_conv_id != current_conv_id:
                history_buffer = []
                current_conv_id = row_conv_id
                history_buffer.append(
                    {
                        "role": "system",
                        "content": "You are an expert. Answer accurately based on your knowledge.",
                    }
                )

            history_buffer.append({"role": "user", "content": question})

            try:
                response = self.bot.llm.create_chat_completion(
                    messages=history_buffer, max_tokens=256, temperature=0.1
                )
                answer = response["choices"][0]["message"]["content"]
            except Exception as e:
                answer = f"Error: {e}"

            history_buffer.append({"role": "assistant", "content": answer})
            baseline_answers.append(answer)

        df_sorted["baseline_answer"] = baseline_answers

        return df_sorted

    def evaluate_against_ground_truth(self, df_results: pd.DataFrame):
        """
        [COMBINED STEP] Matches your previous logic perfectly.
        """
        print(f"[3/3] Judging: RAG vs. Baseline vs. Ground Truth...")

        rag_scores = []
        base_scores = []
        preferences = []
        explanations = []

        for _, row in tqdm(df_results.iterrows(), total=len(df_results)):
            q = row["question"]
            truth = row["ground_truth"]
            rag_ans = row["rag_answer"]
            base_ans = row["baseline_answer"]

            prompt_content = f"""
            You are an expert judge evaluating two AI models.
            
            Question: "{q}"
            
            [Ground Truth / Correct Answer]:
            "{truth}"

            [Model A Answer]:
            "{rag_ans}"
            
            [Model B Answer]:
            "{base_ans}"

            Task:
            1. Rate Model A (1-10) based on accuracy relative to Ground Truth.
            2. Rate Model B (1-10) based on accuracy relative to Ground Truth.
            3. Decide which model is better (or if they are tied).
            
            Scoring Guide:
            1 = Wrong / Irrelevant
            5 = Partially Correct
            10 = Perfect Match with Ground Truth

            Format strictly as:
            Score A: [Number]
            Score B: [Number]
            Winner: [Model A / Model B / Tie]
            Analysis: [One sentence explanation]
            """

            messages = [{"role": "user", "content": prompt_content}]

            try:
                response = self.bot.llm.create_chat_completion(
                    messages=messages, max_tokens=256, temperature=0.0
                )
                content = response["choices"][0]["message"]["content"]

                score_a = 0
                score_b = 0
                winner = "Error"

                match_a = re.search(r"Score A:\s*(\d+)", content)
                match_b = re.search(r"Score B:\s*(\d+)", content)
                match_w = re.search(r"Winner:\s*(.*)", content, re.IGNORECASE)

                if match_a:
                    score_a = int(match_a.group(1))
                if match_b:
                    score_b = int(match_b.group(1))
                if match_w:
                    winner = match_w.group(1).strip()

                rag_scores.append(score_a)
                base_scores.append(score_b)
                preferences.append(winner)
                explanations.append(content)

            except Exception as e:
                rag_scores.append(0)
                base_scores.append(0)
                preferences.append("Error")
                explanations.append(str(e))

        df_results["rag_score"] = rag_scores
        df_results["baseline_score"] = base_scores
        df_results["winner"] = preferences
        df_results["judge_analysis"] = explanations

        return df_results

    def evaluate_all(self, limit: int = None):
        """
        Main entry point.
        """
        df = self.run_rag_inference(limit=limit)
        df = self.run_baseline_inference(df)
        df = self.evaluate_against_ground_truth(df)

        return df
