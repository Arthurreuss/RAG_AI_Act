import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional


def count_words(text: str) -> int:
    """Counts the number of words in a given string.

    Args:
        text (str): The string to be analyzed.

    Returns:
        int: The number of words in the text. Returns 0 if text is empty or None.
    """
    if not text:
        return 0
    return len(text.split())


def get_bin_label(count: int, max_threshold: int, bin_size: int) -> str:
    """Calculates the bin label for a given word count.

    Args:
        count (int): The word count to categorize.
        max_threshold (int): The count at which values are capped into a '+' category.
        bin_size (int): The numerical range size for each bin.

    Returns:
        str: A string label representing the range (e.g., '0-50') or the max cap (e.g., '500+').
    """
    if count >= max_threshold:
        return f"{max_threshold}+"
    lower = (count // bin_size) * bin_size
    return f"{lower}-{lower + bin_size}"


def analyze_chunks(file_path: str, max_threshold: int, bin_size: int) -> None:
    """Analyzes text chunks from a JSON file and prints statistical distributions.

    Reads a JSON file containing text data and metadata, categorizes chunks by type
    and word count bins, and outputs a formatted summary table including averages
    and a list of chunks exceeding the specified word threshold.

    Args:
        file_path (str): Path to the JSON file containing the data.
        max_threshold (int): The word count threshold for 'oversized' categorization.
        bin_size (int): The size of the increments for the distribution table.

    Returns:
        None: The function prints results directly to the console.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Reading from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    oversized_items: List[Dict[str, Any]] = []
    type_totals: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"words": 0, "chars": 0, "count": 0}
    )

    print(f"Analyzing {len(data)} chunks...\n")

    for item in data:
        text: str = item.get("text_to_embed", "")

        meta: Dict[str, Any] = item.get("metadata", {})
        item_type: str = meta.get("section", "unknown").replace("_", " ").title()
        item_type = item_type.replace("Enacting Terms", "Articles").replace(
            "Preamble", "Recital"
        )

        citation: str = meta.get("citation", item.get("chunk_id", "Unknown ID"))

        w_count: int = count_words(text)
        c_count: int = len(text)

        bin_label: str = get_bin_label(w_count, max_threshold, bin_size)
        stats[item_type][bin_label] += 1

        type_totals[item_type]["words"] += w_count
        type_totals[item_type]["chars"] += c_count
        type_totals[item_type]["count"] += 1

        if w_count >= max_threshold:
            oversized_items.append(
                {"type": item_type, "id": citation, "words": w_count, "chars": c_count}
            )

    bins: List[str] = [
        f"{i}-{i+bin_size}" for i in range(0, max_threshold, bin_size)
    ] + [f"{max_threshold}+"]

    header: str = (
        f"{'TYPE':<25} | " + " | ".join([f"{b:<9}" for b in bins]) + " | TOTAL"
    )
    print(header)
    print("-" * len(header))

    sorted_categories: List[str] = sorted(stats.keys())

    for category in sorted_categories:
        row_str: str = f"{category:<25} | "
        total_cat: int = 0
        counts: Dict[str, int] = stats[category]

        for b in bins:
            val: int = counts.get(b, 0)
            total_cat += val
            val_str: str = str(val) if val > 0 else "."
            row_str += f"{val_str:<9} | "

        row_str += f"{total_cat}"
        print(row_str)

    print("\n" + "=" * 60)
    print(f"AVERAGE LENGTHS BY TYPE")
    print("=" * 60)
    print(f"{'TYPE':<25} | {'AVG WORDS':<10} | {'AVG CHARS':<10}")
    print("-" * 60)

    for category in sorted_categories:
        t_data: Dict[str, float] = type_totals[category]
        avg_w: float = round(t_data["words"] / t_data["count"], 1)
        avg_c: float = round(t_data["chars"] / t_data["count"], 1)
        print(f"{category:<25} | {avg_w:<10} | {avg_c:<10}")

    print("\n" + "=" * 80)
    print(f"    OVERSIZED CHUNKS (> {max_threshold} words)")
    print("=" * 80)

    if not oversized_items:
        print("Good news! All chunks are within the limit.")
    else:
        oversized_items.sort(key=lambda x: x["words"], reverse=True)

        print(f"{'TYPE':<20} | {'ID / CITATION':<40} | {'WORDS':<8} | {'CHARS':<8}")
        print("-" * 85)

        for item in oversized_items:
            ref: str = (item["id"][:37] + "..") if len(item["id"]) > 40 else item["id"]
            print(
                f"{item['type']:<20} | {ref:<40} | {item['words']:<8} | {item['chars']:<8}"
            )
