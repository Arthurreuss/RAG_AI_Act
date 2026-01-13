import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional


class ContentResolver:
    """A utility class to resolve granular text chunks back to their parent legal documents.

    This class maintains an index of full legal documents (Articles, Recitals, Annexes)
    and provides heuristics to map chunk identifiers back to their original source
    structure for context retrieval.

    Attributes:
        full_docs_map (Dict[str, Dict[str, Any]]): A mapping of parent document IDs
            to their full structured data.
    """

    def __init__(self, full_data_path: str) -> None:
        """Initializes the ContentResolver by loading and indexing document data.

        Args:
            full_data_path (str): Path to the JSON file containing full Articles,
                Recitals, and Annexes with their hierarchical children.
        """
        self.full_docs_map: Dict[str, Dict[str, Any]] = self._load_and_index(
            full_data_path
        )
        print(f"ContentResolver loaded {len(self.full_docs_map)} parent documents.")

    def _load_and_index(self, path: str) -> Dict[str, Dict[str, Any]]:
        """Loads JSON data and creates a lookup index based on document IDs.

        Args:
            path (str): The file path to the source JSON.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary indexed by lowercase document IDs.
        """
        with open(path, "r", encoding="utf-8") as f:
            data: List[Dict[str, Any]] = json.load(f)

        index: Dict[str, Dict[str, Any]] = {}
        for item in data:
            key: str = item.get("id", "").lower().strip()
            if key:
                index[key] = item
        return index

    def get_parent_id_from_chunk(self, chunk_id: str) -> Optional[str]:
        """Extracts a parent document ID from a granular chunk ID using heuristics.

        Args:
            chunk_id (str): The identifier for a specific text chunk
                (e.g., "article_5_paragraph_1").

        Returns:
            Optional[str]: The normalized parent ID (e.g., "art_5") if a match
                is found, otherwise None.
        """
        cid: str = chunk_id.lower()

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

    def resolve_to_full_text(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """Resolves a list of chunk IDs to the single most relevant parent document.

        Calculates which parent document is most frequently referenced in the
        provided list of chunks and returns that document's full data.

        Args:
            chunk_ids (List[str]): A list of chunk identifiers retrieved
                during a search.

        Returns:
            Dict[str, Any]: The full data object for the most frequent parent
                document. Returns an empty list (as per original logic) if
                no valid parents are resolved.
        """
        if not chunk_ids:
            return []

        valid_parents: List[str] = []
        for cid in chunk_ids:
            pid: Optional[str] = self.get_parent_id_from_chunk(cid)
            if pid and pid in self.full_docs_map:
                valid_parents.append(pid)

        if not valid_parents:
            return []

        most_common_pid: str = Counter(valid_parents).most_common(1)[0][0]

        return self.full_docs_map[most_common_pid]
