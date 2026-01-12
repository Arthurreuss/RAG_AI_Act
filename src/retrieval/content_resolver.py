import json
import re
from collections import Counter
from typing import Dict, List, Optional


class ContentResolver:
    def __init__(self, full_data_path: str):
        """
        Args:
            full_data_path: Path to the JSON file containing full Articles/Recitals/Annexes
                            (The one with "children" and full "text" fields)
        """
        self.full_docs_map = self._load_and_index(full_data_path)
        print(f"ContentResolver loaded {len(self.full_docs_map)} parent documents.")

    def _load_and_index(self, path: str) -> Dict[str, Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        index = {}
        for item in data:
            key = item.get("id", "").lower().strip()
            if key:
                index[key] = item
        return index

    def get_parent_id_from_chunk(self, chunk_id: str) -> Optional[str]:
        """
        Heuristic to extract parent ID from chunk ID.
        Example: "article_5_paragraph_1" -> "art_5"
        Example: "recital_10" -> "rec_10"
        """
        cid = chunk_id.lower()

        if "article" in cid:
            match = re.search(r"article_(\d+[a-z]?)", cid)
            if match:
                return f"art_{match.group(1)}"

        elif "recital" in cid:
            match = re.search(r"recital_(\d+)", cid)
            if match:
                return f"rct_{match.group(1)}"

        elif "annex" in cid:
            match = re.search(r"annex_([a-z0-9]+)", cid)
            if match:
                return f"anx_{match.group(1).upper()}"

        return None

    def resolve_to_full_text(self, chunk_ids: List[str]) -> Dict:
        """
        Takes a list of chunk IDs and returns ONLY the full parent object
        that appears most frequently among them (or the first one as fallback).
        """

        if not chunk_ids:
            return []

        valid_parents = []
        for cid in chunk_ids:
            pid = self.get_parent_id_from_chunk(cid)
            if pid and pid in self.full_docs_map:
                valid_parents.append(pid)

        if not valid_parents:
            return []

        most_common_pid = Counter(valid_parents).most_common(1)[0][0]

        return self.full_docs_map[most_common_pid]
