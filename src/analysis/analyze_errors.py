import re
from collections import Counter
from typing import Any, Dict, List, Set, Union

import pandas as pd

from src.utils.helper import load_json


class ErrorAnalyzer:
    """A diagnostic tool for analyzing RAG (Retrieval-Augmented Generation) performance.

    This class processes evaluation results to identify retrieval accuracy,
    categorize system failures, and track performance decay in multi-turn dialogues.

    Attributes:
        data (List[Dict[str, Any]]): The raw JSON data loaded from the file.
        df (pd.DataFrame): The data structured into a pandas DataFrame for analysis.
    """

    def __init__(self, json_path: str) -> None:
        """Initializes the ErrorAnalyzer with data from a JSON file.

        Args:
            json_path (str): The file path to the JSON evaluation results.
        """
        self.data: List[Dict[str, Any]] = load_json(json_path)
        self.df: pd.DataFrame = pd.DataFrame(self.data)
        print(f"Loaded {len(self.df)} records for analysis.")

    def _get_parent_doc(self, chunk_id: str) -> str:
        """Extracts the parent document ID from a granular chunk ID.

        Used to identify 'Near Misses' where the system found the right document
        but the incorrect segment.

        Args:
            chunk_id (str): The specific ID of a text chunk
                (e.g., 'recital_67_part_4').

        Returns:
            str: The base document ID (e.g., 'recital_67').
        """
        base: str = re.sub(
            r"(_part_|_paragraph_|_point_|_subparagraph_).*", "", chunk_id
        )
        return base

    def check_near_misses(self) -> None:
        """Checks if the retriever found the correct document but wrong chunk.

        Identifies instances where the retrieval 'hit_rank' was -1 (a miss),
        but the retrieved IDs shared a parent document with the target IDs.

        Returns:
            None: Prints the summary of exact hits vs. near misses to the console.
        """
        near_misses: int = 0
        total_misses: int = 0

        print("\nRETRIEVAL: Exact Hits vs. Near Misses")
        print("-" * 40)

        for _, row in self.df.iterrows():
            if row["hit_rank"] != -1:
                continue

            total_misses += 1

            targets: Union[List[str], str] = row.get("target_ids", row.get("target_id"))
            if not isinstance(targets, list):
                targets = [targets]

            retrieved: List[str] = row.get("retrieved_ids", [])

            target_parents: Set[str] = {self._get_parent_doc(t) for t in targets}
            retrieved_parents: Set[str] = {self._get_parent_doc(r) for r in retrieved}

            if not target_parents.isdisjoint(retrieved_parents):
                near_misses += 1

        print(f"Total Retrieval Misses: {total_misses}")
        print(f"Near Misses (Correct Doc, Wrong Chunk): {near_misses}")
        if total_misses > 0:
            print(f"-> {near_misses/total_misses:.1%} of misses were actually close!")

    def categorize_failures(self) -> None:
        """Categorizes every query into one of four buckets to understand system behavior.

        The four buckets are:
            1. Success: Retrieval hit and high RAG score.
            2. Lucky Guess: Retrieval missed but high RAG score.
            3. Context Ignored: Retrieval hit but poor RAG score.
            4. System Failure: Retrieval missed and poor RAG score.

        Returns:
            None: Prints the distribution of behavior categories to the console.
        """
        print("\nRAG BEHAVIOR CATEGORIES")
        print("-" * 40)

        categories: List[str] = []

        for _, row in self.df.iterrows():
            is_hit: bool = row["hit_rank"] > 0
            rag_good: bool = row.get("rag_score", 0) >= 7

            if is_hit and rag_good:
                cat = "Success (Retrieval + Good Answer)"
            elif not is_hit and rag_good:
                cat = "Lucky Guess (Rretrieval missed + Good Answer)"
            elif is_hit and not rag_good:
                cat = "Context Ignored (Retrieval + Poor Answer)"
            else:
                cat = "System Failure (Retrieval missed + Poor Answer)"

            categories.append(cat)

        counts: Counter = Counter(categories)
        for cat, count in counts.most_common():
            print(f"{cat}: {count} ({count/len(self.df):.1%})")

    def analyze_multi_turn_decay(self) -> None:
        """Analyzes performance degradation across multiple conversation turns.

        Checks if metrics like RAG score and retrieval hit rate decrease as
        the conversation progresses deeper into turns.

        Returns:
            None: Prints performance stats by turn number and a context
                dependency check. Returns early if 'turn_number' column is missing.
        """
        if "turn_number" not in self.df.columns:
            return

        print("\nMulti-Turn Performance Decay")
        print("-" * 40)

        stats: pd.DataFrame = (
            self.df.groupby("turn_number")
            .agg(
                {
                    "rag_score": "mean",
                    "hit_rank": lambda x: (x > 0).mean(),
                    "baseline_score": "mean",
                }
            )
            .rename(columns={"hit_rank": "hit_rate"})
        )

        print(
            stats.to_string(
                formatters={
                    "rag_score": "{:.2f}".format,
                    "baseline_score": "{:.2f}".format,
                    "hit_rate": "{:.1%}".format,
                }
            )
        )

        print("\nContext Dependency Check:")
        for turn in sorted(self.df["turn_number"].unique()):
            turn_data: pd.DataFrame = self.df[self.df["turn_number"] == turn]
            wins: int = len(
                turn_data[turn_data["rag_score"] > turn_data["baseline_score"]]
            )
            print(f"Turn {turn}: RAG beat Baseline in {wins}/{len(turn_data)} cases")
