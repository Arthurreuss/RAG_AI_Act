import json
import os
from collections import defaultdict


def count_words(text):
    if not text:
        return 0
    return len(text.split())


def get_bin_label(count, max_threshold, bin_size):
    if count >= max_threshold:
        return f"{max_threshold}+"
    lower = (count // bin_size) * bin_size
    return f"{lower}-{lower + bin_size}"


def analyze_chunks(file_path, max_threshold, bin_size):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Reading from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = defaultdict(lambda: defaultdict(int))
    oversized_items = []
    type_totals = defaultdict(lambda: {"words": 0, "chars": 0, "count": 0})

    print(f"Analyzing {len(data)} chunks...\n")

    for item in data:
        text = item.get("text_to_embed", "")

        meta = item.get("metadata", {})
        item_type = meta.get("section", "unknown").replace("_", " ").title()
        item_type = item_type.replace("Enacting Terms", "Articles").replace(
            "Preamble", "Recital"
        )

        citation = meta.get("citation", item.get("chunk_id", "Unknown ID"))

        w_count = count_words(text)
        c_count = len(text)

        bin_label = get_bin_label(w_count, max_threshold, bin_size)
        stats[item_type][bin_label] += 1

        type_totals[item_type]["words"] += w_count
        type_totals[item_type]["chars"] += c_count
        type_totals[item_type]["count"] += 1

        if w_count >= max_threshold:
            oversized_items.append(
                {"type": item_type, "id": citation, "words": w_count, "chars": c_count}
            )

    bins = [f"{i}-{i+bin_size}" for i in range(0, max_threshold, bin_size)] + [
        f"{max_threshold}+"
    ]

    header = f"{'TYPE':<25} | " + " | ".join([f"{b:<9}" for b in bins]) + " | TOTAL"
    print(header)
    print("-" * len(header))

    sorted_categories = sorted(stats.keys())

    for category in sorted_categories:
        row_str = f"{category:<25} | "
        total_cat = 0
        counts = stats[category]

        for b in bins:
            val = counts.get(b, 0)
            total_cat += val
            val_str = str(val) if val > 0 else "."
            row_str += f"{val_str:<9} | "

        row_str += f"{total_cat}"
        print(row_str)

    print("\n" + "=" * 60)
    print(f"AVERAGE LENGTHS BY TYPE")
    print("=" * 60)
    print(f"{'TYPE':<25} | {'AVG WORDS':<10} | {'AVG CHARS':<10}")
    print("-" * 60)

    for category in sorted_categories:
        t_data = type_totals[category]
        avg_w = round(t_data["words"] / t_data["count"], 1)
        avg_c = round(t_data["chars"] / t_data["count"], 1)
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
            ref = (item["id"][:37] + "..") if len(item["id"]) > 40 else item["id"]
            print(
                f"{item['type']:<20} | {ref:<40} | {item['words']:<8} | {item['chars']:<8}"
            )
