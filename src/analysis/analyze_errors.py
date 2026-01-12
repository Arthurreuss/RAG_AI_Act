import re
from collections import Counter

import pandas as pd

from src.utils.helper import load_json


class ErrorAnalyzer:
    def __init__(self, json_path: str):
        self.data = load_json(json_path)

        self.df = pd.DataFrame(self.data)
        print(f"Loaded {len(self.df)} records for analysis.")

    def _get_parent_doc(self, chunk_id: str) -> str:
        """
        Extracts the parent document ID from a chunk ID to check for 'Near Misses'.
        Example: 'recital_67_part_4' -> 'recital_67'
        """
        base = re.sub(r"(_part_|_paragraph_|_point_|_subparagraph_).*", "", chunk_id)
        return base

    def check_near_misses(self):
        """
        Checks if the retriever found the correct *document* but wrong *chunk*.
        """
        near_misses = 0
        total_misses = 0

        print("\nRETRIEVAL: Exact Hits vs. Near Misses")
        print("-" * 40)

        for _, row in self.df.iterrows():
            if row["hit_rank"] != -1:
                continue

            total_misses += 1

            targets = row.get("target_ids", row.get("target_id"))
            if not isinstance(targets, list):
                targets = [targets]

            retrieved = row.get("retrieved_ids", [])

            target_parents = {self._get_parent_doc(t) for t in targets}
            retrieved_parents = {self._get_parent_doc(r) for r in retrieved}

            if not target_parents.isdisjoint(retrieved_parents):
                near_misses += 1

        print(f"Total Retrieval Misses: {total_misses}")
        print(f"Near Misses (Correct Doc, Wrong Chunk): {near_misses}")
        if total_misses > 0:
            print(f"-> {near_misses/total_misses:.1%} of misses were actually close!")

    def categorize_failures(self):
        """
        Categorizes every query into 4 buckets to understand system behavior.
        """
        print("\nRAG BEHAVIOR CATEGORIES")
        print("-" * 40)

        categories = []

        for _, row in self.df.iterrows():
            is_hit = row["hit_rank"] > 0
            rag_good = row.get("rag_score", 0) >= 7

            if is_hit and rag_good:
                cat = "Success (Retrieval + Good Answer)"
            elif not is_hit and rag_good:
                cat = "Lucky Guess (Rretrieval missed + Good Answer)"
            elif is_hit and not rag_good:
                cat = "Context Ignored (Retrieval + Poor Answer)"
            else:
                cat = "System Failure (Retrieval missed + Poor Answer)"

            categories.append(cat)

        counts = Counter(categories)
        for cat, count in counts.most_common():
            print(f"{cat}: {count} ({count/len(self.df):.1%})")

    def analyze_multi_turn_decay(self):
        """
        Specific for Multi-Turn: Checks if performance drops deeper into conversation.
        """
        if "turn_number" not in self.df.columns:
            return

        print("\nMulti-Turn Performance Decay")
        print("-" * 40)

        stats = (
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
            turn_data = self.df[self.df["turn_number"] == turn]
            wins = len(turn_data[turn_data["rag_score"] > turn_data["baseline_score"]])
            print(f"Turn {turn}: RAG beat Baseline in {wins}/{len(turn_data)} cases")
