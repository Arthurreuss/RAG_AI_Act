import copy
import json
import os
import traceback


class ChunkProcessor:
    def __init__(self, split_threshold=250, chunk_size=150, overlap=30):
        """
        Initialize with default settings for chunk splitting.
        """
        self.split_threshold = split_threshold
        self.chunk_size = chunk_size
        self.overlap = overlap

    # -------------------------------------------------
    # PART 1: HIERARCHICAL STRUCTURE PROCESSING
    # -------------------------------------------------
    def process_recursive(
        self,
        node,
        parent_id_chain,
        parent_context_text,
        citation_chain,
        meta_info,
        chunks_accumulator,
    ):
        """
        Processes nodes recursively.
        Handles SIBLING context (Text Block -> Point) and Annex II aggregation.
        Appends results directly to chunks_accumulator list.
        """

        node_id = node.get("id", "")
        node_type = node.get("type")
        node_text = node.get("text", "").strip()
        node_number = str(node.get("number", "")).strip()

        # 0. SPECIAL EDGE CASE: ANNEX II (Aggregate Everything)
        if node_id == "anx_II" or (
            node_type == "annex"
            and "criminal offences" in node.get("title", "").lower()
        ):
            full_text_blob = [node_text]

            if "children" in node:
                for child in node["children"]:
                    child_text = child.get("text", "").strip()
                    if child_text:
                        full_text_blob.append(child_text)

            combined_blob = " ".join(full_text_blob)
            chunk = {
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
            return  # STOP RECURSION FOR THIS NODE

        # 1. MANAGE CITATION
        local_citation_chain = list(citation_chain)
        if node_number:
            if not local_citation_chain or not local_citation_chain[-1].endswith(
                node_number
            ):
                local_citation_chain.append(node_number)

        final_citation_string = " ".join(local_citation_chain)

        # 2. ID GENERATION
        if node_number:
            clean_num = (
                node_number.replace("(", "")
                .replace(")", "")
                .replace(".", "")
                .strip()
                .lower()
            )
            id_segment = f"{node_type}_{clean_num}"
        else:
            id_segment = node_type

        if parent_id_chain:
            full_chunk_id = f"{parent_id_chain}_{id_segment}"
        else:
            full_chunk_id = id_segment

        # 3. CONTEXT STRATEGY (Parent -> Node)
        if node_type in ["article", "annex"]:
            combined_text = ""
            text_for_embedding = ""
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
                    "type": node_type,
                    "citation": final_citation_string,
                    "chunk_number": node_number,
                    "chapter": meta_info.get("chapter", "Unknown"),
                    "section": meta_info.get("section", "Unknown"),
                },
            }
            chunks_accumulator.append(chunk)

        # 5. RECURSE WITH SIBLING CONTEXT
        if "children" in node and isinstance(node["children"], list):
            sibling_intro_context = ""

            for idx, child in enumerate(node["children"]):
                if node_type in ["article", "annex"]:
                    base_context_for_child = ""
                else:
                    base_context_for_child = combined_text

                if sibling_intro_context:
                    context_to_pass = (
                        f"{base_context_for_child} {sibling_intro_context}"
                    )
                else:
                    context_to_pass = base_context_for_child

                child_id_suffix = f"item_{idx}" if not child.get("number") else ""
                next_id_chain = (
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
                )

                if child.get("type") == "text_block":
                    sibling_intro_context = child.get("text", "").strip()

                if child.get("type") == "section":
                    sibling_intro_context = ""

    def process_legal_json(self, input_data):
        """
        Main entry point for processing the raw hierarchical JSON.
        """
        chunks_to_embed = []

        for item in input_data:
            try:
                root_meta = item.get("metadata", {})
                title = item.get("title", "")

                if " - " in title:
                    short_title = title.split(" - ")[0]
                else:
                    short_title = f"{item.get('type', 'Item').capitalize()} {item.get('number', '')}".strip()

                initial_chain = [short_title]

                self.process_recursive(
                    node=item,
                    parent_id_chain="",
                    parent_context_text="",
                    citation_chain=initial_chain,
                    meta_info=root_meta,
                    chunks_accumulator=chunks_to_embed,
                )

            except Exception as e:
                print(f"[ERROR] Failed on item {item.get('id', 'Unknown')}: {e}")
                traceback.print_exc()
                continue

        return chunks_to_embed

    # -------------------------------------------------
    # PART 2: CHUNK SPLITTING AND REFINEMENT
    # -------------------------------------------------
    def split_text_with_overlap(self, text, chunk_size=None, overlap=None):
        """
        Splits a string into a list of overlapping strings based on word count.
        Allows overriding defaults.
        """
        # Use defaults from init if not provided
        c_size = chunk_size if chunk_size is not None else self.chunk_size
        o_lap = overlap if overlap is not None else self.overlap

        words = text.split()

        if len(words) <= c_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = start + c_size
            chunk_words = words[start:end]
            chunk_str = " ".join(chunk_words)
            chunks.append(chunk_str)

            start += c_size - o_lap
            if start >= len(words):
                break

        return chunks

    def process_chunks(self, data, split_threshold=None):
        """
        Scans existing chunks and splits those that exceed the threshold.
        """
        # Use default from init if not provided
        thresh = (
            split_threshold if split_threshold is not None else self.split_threshold
        )

        new_data = []
        split_count = 0

        print(f"Scanning {len(data)} chunks for size (Threshold: {thresh} words)...")

        for item in data:
            original_text = item.get("text_to_embed", "")
            citation = item.get("metadata", {}).get("citation", "Unknown")
            word_count = len(original_text.split())

            # CASE 1: Keep as is
            if word_count <= thresh:
                new_data.append(item)

            # CASE 2: Split
            else:
                split_count += 1
                sub_texts = self.split_text_with_overlap(original_text)

                for i, sub_text in enumerate(sub_texts):
                    # Deep copy to avoid reference issues
                    new_item = copy.deepcopy(item)

                    new_item["chunk_id"] = f"{item['chunk_id']}_part_{i+1}"

                    # Update Text with context heuristic
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
