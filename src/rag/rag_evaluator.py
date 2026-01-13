import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from src.rag.rag_pipeline import RAGChatbot
from src.utils.helper import load_json


class RAGEvaluator:
    """A framework for evaluating RAG chatbot performance against a golden dataset.

    This class facilitates end-to-end evaluation by running RAG inference,
    baseline (zero-shot) inference, and using an LLM as a judge to compare
    outputs against ground truth. It supports both single-turn and
    multi-turn conversation formats.

    Attributes:
        bot (RAGChatbot): The chatbot instance to be evaluated.
        mode (str): Evaluation mode, either 'single_turn' or 'multi_turn'.
        golden_set (List[Any]): The loaded test data for evaluation.
    """

    def __init__(self, chatbot: RAGChatbot, golden_set_path: str) -> None:
        """Initializes the evaluator and detects the dataset format.

        Args:
            chatbot: An initialized RAGChatbot instance.
            golden_set_path: Path to the JSON file containing test questions/conversations.
        """
        self.bot: RAGChatbot = chatbot
        raw_data: List[Any] = load_json(golden_set_path)

        if raw_data and isinstance(raw_data[0], list):
            self.mode: str = "multi_turn"
            self.golden_set = raw_data
            print(
                f"Detected MULTI-TURN dataset ({len(self.golden_set)} conversations)."
            )
        else:
            self.mode: str = "single_turn"
            self.golden_set = raw_data
            print(f"Detected SINGLE-TURN dataset ({len(self.golden_set)} questions).")

    def _calculate_hit_metrics(
        self, retrieved_ids: List[str], target_ids: Union[List[str], str]
    ) -> Tuple[int, bool, int]:
        """Calculates retrieval accuracy metrics for a set of IDs.

        Args:
            retrieved_ids: The list of IDs returned by the retriever.
            target_ids: The expected correct ID or list of IDs.

        Returns:
            A tuple containing:
                - rank (int): 1-based index of the FIRST matching ID found (-1 if none).
                - hit (bool): True if at least one target ID was found in retrieval.
                - found_count (int): Count of distinct target IDs successfully found.
        """
        if not isinstance(target_ids, list):
            target_ids = [target_ids]

        best_rank: float = float("inf")
        found_set: set = set()

        for tid in target_ids:
            try:
                rank: int = retrieved_ids.index(tid) + 1
                if rank < best_rank:
                    best_rank = rank
                found_set.add(tid)
            except ValueError:
                continue

        hit: bool = len(found_set) > 0
        final_rank: int = int(best_rank) if hit else -1

        return final_rank, hit, len(found_set)

    def run_rag_inference(
        self, limit: Optional[int] = None, use_full_doc: bool = False
    ) -> pd.DataFrame:
        """Runs the RAG pipeline on the golden set.

        Handles nested loops for multi-turn conversations to ensure the
        chatbot's internal history is preserved across turns.

        Args:
            limit: Optional number of items to process for a quick test.
            use_full_doc: Whether to pass full document context to the bot.

        Returns:
            pd.DataFrame: A DataFrame containing questions, answers, and
                retrieval metrics.
        """
        results: List[Dict[str, Any]] = []
        subset: List[Any] = self.golden_set[:limit] if limit else self.golden_set

        print(f"[1/3] Running RAG Inference ({self.mode})...")

        iterator_data: List[List[Dict[str, Any]]] = (
            subset if self.mode == "multi_turn" else [[item] for item in subset]
        )

        for conv_idx, conversation in enumerate(tqdm(iterator_data)):
            self.bot.clear_history()

            for turn_idx, item in enumerate(conversation):
                question: str = item["question"]
                target_ids: Union[List[str], str] = item["context_id"]
                ground_truth: str = item["ground_truth_answer"]

                conv_id: str = f"conv_{conv_idx}"

                try:
                    rag_response, sources = self.bot.chat(
                        question, use_full_doc=use_full_doc
                    )

                    retrieved_ids: List[str] = [s.get("chunk_id", "") for s in sources]

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

    def run_baseline_inference(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """Runs a zero-shot baseline inference using only the LLM.

        Maintains conversation history in a buffer to ensure a fair
        comparison with the RAG pipeline in multi-turn scenarios.

        Args:
            df_results: The DataFrame generated by `run_rag_inference`.

        Returns:
            pd.DataFrame: The original DataFrame with an added 'baseline_answer' column.
        """
        print(f"[2/3] Running Baseline Inference...")

        baseline_answers: List[str] = []

        if "conversation_id" in df_results.columns:
            df_sorted: pd.DataFrame = df_results.sort_values(
                by=["conversation_id", "turn_number"]
            )
        else:
            df_sorted = df_results

        current_conv_id: Optional[str] = None
        history_buffer: List[Dict[str, str]] = []

        for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted)):
            question: str = row["question"]
            row_conv_id: str = row.get("conversation_id", "single")

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
                response: Dict[str, Any] = self.bot.llm.create_chat_completion(
                    messages=history_buffer, max_tokens=256, temperature=0.1
                )
                answer: str = response["choices"][0]["message"]["content"]
            except Exception as e:
                answer = f"Error: {e}"

            history_buffer.append({"role": "assistant", "content": answer})
            baseline_answers.append(answer)

        df_sorted["baseline_answer"] = baseline_answers
        return df_sorted

    def evaluate_against_ground_truth(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """Uses an LLM judge to score RAG vs. Baseline answers.

        Parses the judge's response to extract numerical scores and winner
        selection based on a strict formatting prompt.

        Args:
            df_results: DataFrame containing both RAG and Baseline responses.

        Returns:
            pd.DataFrame: DataFrame enriched with 'rag_score', 'baseline_score',
                'winner', and 'judge_analysis'.
        """
        print(f"[3/3] Judging: RAG vs. Baseline vs. Ground Truth...")

        rag_scores: List[int] = []
        base_scores: List[int] = []
        preferences: List[str] = []
        explanations: List[str] = []

        for _, row in tqdm(df_results.iterrows(), total=len(df_results)):
            q: str = row["question"]
            truth: str = row["ground_truth"]
            rag_ans: str = row["rag_answer"]
            base_ans: str = row["baseline_answer"]

            prompt_content: str = f"""
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

            messages: List[Dict[str, str]] = [
                {"role": "user", "content": prompt_content}
            ]

            try:
                response: Dict[str, Any] = self.bot.llm.create_chat_completion(
                    messages=messages, max_tokens=256, temperature=0.0
                )
                content: str = response["choices"][0]["message"]["content"]

                score_a: int = 0
                score_b: int = 0
                winner: str = "Error"

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

    def evaluate_all(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Executes the complete evaluation pipeline.

        Args:
            limit: Optional limit on the number of items to evaluate.

        Returns:
            pd.DataFrame: The final results containing all inference data and scores.
        """
        df: pd.DataFrame = self.run_rag_inference(limit=limit)
        df = self.run_baseline_inference(df)
        df = self.evaluate_against_ground_truth(df)

        return df
