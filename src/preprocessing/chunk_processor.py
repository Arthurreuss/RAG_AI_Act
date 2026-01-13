import copy
import json
import os
import traceback
from typing import Any, Dict, List, Optional, Union


class ChunkProcessor:
    """A processor for transforming hierarchical legal documents into flat chunks.

    This class handles recursive processing of nested document structures (like JSON
    exports of legislation), manages context propagation, and performs text
    splitting with overlap for oversized chunks.

    Attributes:
        split_threshold (int): Maximum number of words allowed before a chunk is split.
        chunk_size (int): Target word count for each sub-chunk.
        overlap (int): Number of overlapping words between consecutive sub-chunks.
    """

    def __init__(self, split_threshold: int, chunk_size: int, overlap: int) -> None:
        """Initializes the ChunkProcessor with splitting parameters.

        Args:
            split_threshold (int): The word count limit to trigger splitting.
            chunk_size (int): The number of words in each split part.
            overlap (int): The number of words to overlap between parts.
        """
        self.split_threshold: int = split_threshold
        self.chunk_size: int = chunk_size
        self.overlap: int = overlap

    def process_recursive(
        self,
        node: Dict[str, Any],
        parent_id_chain: str,
        parent_context_text: str,
        citation_chain: List[str],
        meta_info: Dict[str, Any],
        chunks_accumulator: List[Dict[str, Any]],
        root_type: Optional[str] = None,
    ) -> None:
        """Processes document nodes recursively to build flattened chunks.

        This method handles hierarchical ID generation, citation building, sibling
        context aggregation (e.g., text blocks leading into points), and special
        handling for aggregated annexes.

        Args:
            node (Dict[str, Any]): The current node in the hierarchy.
            parent_id_chain (str): The concatenated ID string from parent levels.
            parent_context_text (str): Accumulated text context from parent nodes.
            citation_chain (List[str]): List of identifiers for citation building.
            meta_info (Dict[str, Any]): Metadata for the current branch (e.g., chapter).
            chunks_accumulator (List[Dict[str, Any]]): List to store generated chunks.
            root_type (Optional[str]): The high-level type (e.g., 'article') to preserve.
        """
        node_id: str = node.get("id", "")
        node_type: str = node.get("type", "")
        node_text: str = node.get("text", "").strip()
        node_number: str = str(node.get("number", "")).strip()

        effective_root_type: Optional[str] = root_type if root_type else node_type

        # SPECIAL EDGE CASE: ANNEX II (Aggregate Everything)
        if node_id == "anx_II" or (
            node_type == "annex"
            and "criminal offences" in node.get("title", "").lower()
        ):
            full_text_blob: List[str] = [node_text]

            if "children" in node:
                for child in node["children"]:
                    child_text: str = child.get("text", "").strip()
                    if child_text:
                        full_text_blob.append(child_text)

            combined_blob: str = " ".join(full_text_blob)
            chunk: Dict[str, Any] = {
                "chunk_id": node_id,
                "text_to_embed": f"{node.get('title', 'Annex II')}: {combined_blob}",
                "metadata": {
                    "type": "annex_aggregated",
                    "citation": node.get("title", "Annex II"),
                    "chunk_number": "",
                    "chapter": meta_info.get("chapter", "Unknown"),
                    "section": meta_info.get("section", "Unknown"),
                },
            }
            chunks_accumulator.append(chunk)
            return

        # EXTRACT ANNEX NUMBER FROM TITLE IF MISSING
        if node_type == "annex" and not node_number:
            title: str = node.get("title", "").strip()
            title_identifier: str = title.split(" - ")[0].strip()
            if title_identifier.upper().startswith("ANNEX"):
                parts: List[str] = title_identifier.split()
                if len(parts) > 1:
                    node_number = parts[1]

        # 1. MANAGE CITATION
        local_citation_chain: List[str] = list(citation_chain)
        if node_number:
            if not local_citation_chain or not local_citation_chain[-1].endswith(
                node_number
            ):
                local_citation_chain.append(node_number)

        final_citation_string: str = " ".join(local_citation_chain)

        # 2. ID GENERATION
        if node_number:
            clean_num: str = (
                node_number.replace("(", "")
                .replace(")", "")
                .replace(".", "")
                .strip()
                .lower()
            )
            id_segment: str = f"{node_type}_{clean_num}"
        else:
            id_segment = node_type

        full_chunk_id: str = (
            f"{parent_id_chain}_{id_segment}" if parent_id_chain else id_segment
        )

        # 3. CONTEXT STRATEGY (Parent -> Node)
        if node_type in ["article", "annex"]:
            combined_text: str = ""
            text_for_embedding: str = ""
        else:
            if parent_context_text:
                combined_text = f"{parent_context_text} {node_text}"
            else:
                combined_text = node_text
            text_for_embedding = f"{final_citation_string}: {combined_text}"

        # 4. SAVE CHUNK
        if node_type not in ["article", "annex"] and node_text:
            chunk = {
                "chunk_id": full_chunk_id,
                "text_to_embed": text_for_embedding,
                "metadata": {
                    "type": effective_root_type,
                    "citation": final_citation_string,
                    "chunk_number": node_number,
                    "chapter": meta_info.get("chapter", "Unknown"),
                    "section": meta_info.get("section", "Unknown"),
                },
            }
            chunks_accumulator.append(chunk)

        # 5. RECURSE WITH SIBLING CONTEXT
        if "children" in node and isinstance(node["children"], list):
            sibling_intro_context: str = ""

            for idx, child in enumerate(node["children"]):
                if node_type in ["article", "annex"]:
                    base_context_for_child: str = ""
                else:
                    base_context_for_child = combined_text

                context_to_pass: str = (
                    f"{base_context_for_child} {sibling_intro_context}"
                    if sibling_intro_context
                    else base_context_for_child
                )

                child_id_suffix: str = f"item_{idx}" if not child.get("number") else ""
                next_id_chain: str = (
                    f"{full_chunk_id}_{child_id_suffix}"
                    if child_id_suffix
                    else full_chunk_id
                )

                self.process_recursive(
                    node=child,
                    parent_id_chain=next_id_chain,
                    parent_context_text=context_to_pass,
                    citation_chain=local_citation_chain,
                    meta_info=meta_info,
                    chunks_accumulator=chunks_accumulator,
                    root_type=effective_root_type,
                )

                if child.get("type") == "text_block":
                    sibling_intro_context = child.get("text", "").strip()

                if child.get("type") == "section":
                    sibling_intro_context = ""

    def process_legal_json(
        self, input_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Main entry point for processing the raw hierarchical JSON data.

        Args:
            input_data (List[Dict[str, Any]]): List of hierarchical document nodes.

        Returns:
            List[Dict[str, Any]]: A flattened list of chunk dictionaries.
        """
        chunks_to_embed: List[Dict[str, Any]] = []

        for item in input_data:
            try:
                root_meta: Dict[str, Any] = item.get("metadata", {})
                title: str = item.get("title", "")
                root_type_start: Optional[str] = item.get("type")

                if " - " in title:
                    short_title: str = title.split(" - ")[0]
                else:
                    short_title = f"{item.get('type', 'Item').capitalize()} {item.get('number', '')}".strip()

                initial_chain: List[str] = [short_title]

                self.process_recursive(
                    node=item,
                    parent_id_chain="",
                    parent_context_text="",
                    citation_chain=initial_chain,
                    meta_info=root_meta,
                    chunks_accumulator=chunks_to_embed,
                    root_type=root_type_start,
                )

            except Exception as e:
                print(f"[ERROR] Failed on item {item.get('id', 'Unknown')}: {e}")
                traceback.print_exc()
                continue

        return chunks_to_embed

    def split_text_with_overlap(
        self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> List[str]:
        """Splits a long string into smaller segments based on word count and overlap.

        Args:
            text (str): The text to be split.
            chunk_size (Optional[int]): Word limit for each part. Defaults to self.chunk_size.
            overlap (Optional[int]): Word overlap between parts. Defaults to self.overlap.

        Returns:
            List[str]: A list of text segments.
        """
        c_size: int = chunk_size if chunk_size is not None else self.chunk_size
        o_lap: int = overlap if overlap is not None else self.overlap

        words: List[str] = text.split()

        if len(words) <= c_size:
            return [text]

        chunks: List[str] = []
        start: int = 0
        while start < len(words):
            end: int = start + c_size
            chunk_words: List[str] = words[start:end]
            chunk_str: str = " ".join(chunk_words)
            chunks.append(chunk_str)

            start += c_size - o_lap
            if start >= len(words):
                break

        return chunks

    def process_chunks(
        self, data: List[Dict[str, Any]], split_threshold: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Scans chunks and splits those exceeding the word threshold.

        Args:
            data (List[Dict[str, Any]]): The list of flattened chunks to check.
            split_threshold (Optional[int]): The word threshold for splitting.
                Defaults to self.split_threshold.

        Returns:
            List[Dict[str, Any]]: A new list of chunks including the split segments.
        """
        thresh: int = (
            split_threshold if split_threshold is not None else self.split_threshold
        )

        new_data: List[Dict[str, Any]] = []
        split_count: int = 0

        print(f"Scanning {len(data)} chunks for size (Threshold: {thresh} words)...")

        for item in data:
            original_text: str = item.get("text_to_embed", "")
            citation: str = item.get("metadata", {}).get("citation", "Unknown")
            word_count: int = len(original_text.split())

            if word_count <= thresh:
                new_data.append(item)
            else:
                split_count += 1
                sub_texts: List[str] = self.split_text_with_overlap(original_text)

                for i, sub_text in enumerate(sub_texts):
                    new_item: Dict[str, Any] = copy.deepcopy(item)
                    new_item["chunk_id"] = f"{item['chunk_id']}_part_{i+1}"

                    if i > 0 and not sub_text.startswith(citation):
                        new_item["text_to_embed"] = (
                            f"{citation} (Part {i+1}): ...{sub_text}"
                        )
                    elif i > 0:
                        new_item["text_to_embed"] = f"{sub_text} (Part {i+1})"
                    else:
                        new_item["text_to_embed"] = sub_text

                    new_item["metadata"]["split_part"] = i + 1
                    new_item["metadata"]["total_splits"] = len(sub_texts)

                    new_data.append(new_item)

        print(f"Processing Complete.")
        print(f"  - Original Chunks: {len(data)}")
        print(f"  - Oversized Chunks Found: {split_count}")
        print(f"  - Final Chunk Count: {len(new_data)}")

        return new_data
