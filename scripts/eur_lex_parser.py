import json
import os
import re
import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Suppress XML parsing warnings from BS4
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# ==========================================
# 1. UTILITIES & CLEANING
# ==========================================


def clean_text(text):
    """Normalizes whitespace and removes non-breaking spaces."""
    if not text:
        return ""
    text = text.replace("\u00a0", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def get_chapter_info(tag):
    """Finds the chapter context for a given tag/article."""
    chapter_div = tag.find_parent("div", id=re.compile(r"^cpt_[IVX\d]+$"))
    if not chapter_div:
        tag_id = tag.get("id", "")
        if tag_id.startswith("rct_"):
            return "Preamble"
        if tag_id.startswith("anx_"):
            return "Annexes"
        return "General Provisions"

    num_tag = chapter_div.find("p", class_="oj-ti-section-1")
    chapter_num = clean_text(num_tag.get_text()) if num_tag else ""
    title_tag = chapter_div.find("p", class_="oj-ti-section-2")
    chapter_title = clean_text(title_tag.get_text()) if title_tag else ""
    return f"{chapter_num} - {chapter_title}".strip(" -")


# ==========================================
# 2. TEXT STRUCTURE PARSERS (REGEX LOGIC)
# ==========================================


def extract_inline_roman_points(text):
    """
    Level 3: Detects inline roman numerals (i) to (vii).
    """
    # Regex matches (i), (ii), (iii)... (vii)
    pattern = r"(?:\s|^)(\((?:vii|vi|v|iv|iii|ii|i)\))"

    if not re.search(pattern, text):
        return text, []

    parts = re.split(pattern, text)
    if len(parts) < 3:
        return text, []

    main_text = parts[0].strip().rstrip(":")
    sub_children = []

    # re.split keeps delimiters, so we iterate in steps of 2
    for k in range(1, len(parts), 2):
        num = parts[k]
        content = parts[k + 1].strip().strip(";")
        sub_children.append(
            {"type": "point", "number": num, "text": content, "children": []}
        )

    return main_text, sub_children


def extract_inline_alpha_points(text):
    """
    Level 2: Detects inline letters (a), (b)... (z).

    CRITICAL LOGIC:
    This uses a sequence check to distinguish between Letter (i) and Roman (i).
    - If we see (i) after (h), it is accepted as a Letter.
    - If we see (i) after (a), it is ignored (left for the Roman parser).
    """
    # 1. Match ANY letter token inside parentheses
    pattern = r"(?:\s|^)(\(([a-z])\))"

    matches = list(re.finditer(pattern, text))
    if not matches:
        return text, []

    # 2. Filter matches based on Alphabetical Sequence (a -> b -> c)
    valid_indices = []
    expected_char_ord = ord("a")  # Start expecting 'a'

    for m in matches:
        char_found = m.group(2)  # The letter inside, e.g., 'a', 'b', 'i'
        char_ord = ord(char_found)

        if char_ord == expected_char_ord:
            valid_indices.append(m)
            expected_char_ord += 1
        else:
            # If the letter breaks the sequence (e.g. found (i) when expecting (b)),
            # we ignore it here. It will remain in the text and be caught by the Roman parser later.
            pass

    if not valid_indices:
        return text, []

    # 3. Construct the children based on valid splits
    children = []

    # Text before the first (a)
    main_text = text[: valid_indices[0].start()].strip().rstrip(":")

    for k, match in enumerate(valid_indices):
        current_num = match.group(1)  # e.g. "(a)"
        start_content = match.end()

        # End of content is start of next match, or end of string
        if k + 1 < len(valid_indices):
            end_content = valid_indices[k + 1].start()
        else:
            end_content = len(text)

        content = text[start_content:end_content].strip().strip(";")

        # Recursively look for Roman numerals inside this content
        child_main_text, child_subs = extract_inline_roman_points(content)

        children.append(
            {
                "type": "point",
                "number": current_num,
                "text": child_main_text,
                "children": child_subs,
            }
        )

    return main_text, children


# ==========================================
# 3. HTML PARSERS (TABLES & ARTICLES)
# ==========================================


def parse_points_from_table(table_tag):
    """Extracts structured points from HTML tables."""
    points = []
    search_root = table_tag.find("tbody") if table_tag.find("tbody") else table_tag
    rows = search_root.find_all("tr", recursive=False)

    for row in rows:
        cells = row.find_all("td", recursive=False)
        if len(cells) >= 2:
            pt_num = clean_text(cells[0].get_text())
            content_cell = cells[1]

            sub_children = []

            # Recursively handle nested tables
            nested_tables = content_cell.find_all("table", recursive=False)
            for nt in nested_tables:
                sub_children.extend(parse_points_from_table(nt))

            # Extract direct text
            text_bits = []
            for child in content_cell.children:
                if child.name != "table":
                    text_bits.append(
                        clean_text(child.get_text() if child.name else child)
                    )

            raw_text = " ".join([t for t in text_bits if t])

            # If no table sub-children, look for inline regex sub-children
            if not sub_children:
                main_text, inline_subs = extract_inline_roman_points(raw_text)
            else:
                main_text = raw_text
                inline_subs = []

            final_children = sub_children + inline_subs

            # Check if it looks like a numbered point
            is_point = (
                pt_num.startswith("(") and pt_num.endswith(")")
            ) or pt_num.endswith(".")

            if is_point:
                points.append(
                    {
                        "type": "point",
                        "number": pt_num,
                        "text": main_text,
                        "children": final_children,
                    }
                )
            elif raw_text:
                points.append({"type": "text_block", "text": raw_text})

    return points


def parse_article_3_definitions(art_tag, header_text, title_text, chapter_info):
    """Specific parser for Article 3 (Definitions)."""
    full_text_blocks = []
    for child in art_tag.find_all(["p", "table", "div"], recursive=False):
        if "oj-ti-art" in child.get("class", []) or "eli-title" in child.get(
            "class", []
        ):
            continue
        text = clean_text(child.get_text())
        if text:
            full_text_blocks.append(text)

    full_text = "\n".join(full_text_blocks)
    split_match = re.search(r"(?:^|\n)\s*(\(1\))", full_text)

    children = []
    intro_text = full_text.replace("\n", " ")

    if split_match:
        intro_text = full_text[: split_match.start()].strip().replace("\n", " ")
        rest_text = full_text[split_match.start() :]

        def_parts = re.split(r"(?:^|\n)\s*(\(\d+\))", rest_text)

        for k in range(1, len(def_parts), 2):
            num = def_parts[k]
            content = def_parts[k + 1].strip().replace("\n", " ").rstrip(";")

            # Use the smart alpha parser here too
            def_main_text, def_subs = extract_inline_alpha_points(content)

            # Fallback check for Roman if no Alpha found (rare but possible)
            if not def_subs:
                def_main_text, def_subs = extract_inline_roman_points(def_main_text)

            children.append(
                {
                    "type": "definition",
                    "number": num,
                    "text": def_main_text,
                    "children": def_subs,
                }
            )

    return {
        "id": art_tag.get("id"),
        "type": "article",
        "number": "3",
        "title": title_text,
        "text": intro_text,
        "children": children,
        "metadata": {
            "section": "Enacting Terms",
            "chapter": chapter_info,
            "citation": header_text,
        },
    }


def parse_paragraph_div(para_div):
    """Parses standard numbered paragraph divs (e.g., 066.001)."""
    para_id = para_div.get("id")
    intro_parts = []
    children = []

    for child in para_div.children:
        if child.name == "table":
            children.extend(parse_points_from_table(child))
        elif child.name == "p" or isinstance(child, str):
            t = clean_text(child.get_text() if child.name else child)
            if t:
                intro_parts.append(t)

    full_text = " ".join(intro_parts)
    main_text, inline_subs = extract_inline_alpha_points(full_text)

    # If we found alpha points, we use them. If not, check for roman.
    if not inline_subs:
        main_text, inline_subs = extract_inline_roman_points(full_text)

    children.extend(inline_subs)
    num_match = re.match(r"^(\d+)\.", full_text)

    return {
        "type": "paragraph",
        "id": para_id,
        "number": num_match.group(1) if num_match else None,
        "text": main_text,
        "children": children,
    }


def parse_annex_structure(annex_div):
    """Parses content within Annex divs."""
    children = []
    current_section = None

    for child in annex_div.find_all(["p", "table", "div"], recursive=False):
        # Section Headers
        if child.name == "p" and (
            "oj-ti-grseq-1" in child.get("class", [])
            or "oj-ti-section-1" in child.get("class", [])
        ):
            if current_section:
                children.append(current_section)
            current_section = {
                "type": "section",
                "title": clean_text(child.get_text()),
                "children": [],
            }
            continue

        # Tables (lists)
        if child.name == "table":
            points = parse_points_from_table(child)
            target = current_section["children"] if current_section else children
            target.extend(points)

        # Standard Text
        elif child.name == "p" and "oj-normal" in child.get("class", []):
            text = clean_text(child.get_text())
            if text:
                item = {"type": "text_block", "text": text}
                if current_section:
                    current_section["children"].append(item)
                else:
                    children.append(item)

    if current_section:
        children.append(current_section)
    return children


# ==========================================
# 4. MAIN EXECUTION
# ==========================================


def fetch_and_parse(url, output_json_path):
    print(f"Fetching content from: {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        html_content = response.text
    except Exception as e:
        print(f"Failed to fetch: {e}")
        return

    print("Parsing HTML content...")
    soup = BeautifulSoup(html_content, "lxml")
    structured_data = []

    # --- 1. RECITALS ---
    print("--- Processing Recitals ---")
    recitals = soup.find_all("div", id=re.compile(r"^rct_\d+$"))
    for rct in recitals:
        rct_id = rct.get("id")
        rct_text, rct_num = "", ""
        table = rct.find("table")
        if table:
            cells = table.find_all("td")
            if len(cells) >= 2:
                rct_num = clean_text(cells[0].get_text()).strip("()")
                rct_text = clean_text(cells[1].get_text())
        else:
            text_raw = clean_text(rct.get_text())
            match = re.match(r"\((\d+)\)\s*(.*)", text_raw)
            if match:
                rct_num, rct_text = match.group(1), match.group(2)
            else:
                rct_text = text_raw

        if rct_text:
            structured_data.append(
                {
                    "id": rct_id,
                    "type": "recital",
                    "number": rct_num,
                    "title": f"Recital {rct_num}",
                    "text": rct_text,
                    "children": [],
                    "metadata": {
                        "section": "Preamble",
                        "chapter": "Preamble",
                        "citation": f"Recital {rct_num}",
                    },
                }
            )

    # --- 2. ARTICLES ---
    print("--- Processing Articles ---")
    articles = soup.find_all("div", id=re.compile(r"^art_\d+$"))
    for art in articles:
        art_id = art.get("id")
        header_tag = art.find("p", class_="oj-ti-art")
        header_text = clean_text(header_tag.get_text()) if header_tag else art_id

        art_number_match = re.search(r"\d+", header_text)
        art_number = art_number_match.group(0) if art_number_match else "0"

        title_div = art.find("div", class_="eli-title")
        title_text = clean_text(title_div.get_text()) if title_div else "No Title"
        chapter_info = get_chapter_info(art)

        # Special Case: Article 3 (Definitions)
        if art_number == "3":
            structured_data.append(
                parse_article_3_definitions(art, header_text, title_text, chapter_info)
            )
            continue

        article_children = []
        paras = art.find_all("div", id=re.compile(r"^\d{3}\.\d{3}$"))

        # Case A: No numbered paragraphs (Article text is direct)
        if not paras:
            loose_text = []

            # [FIX]: Only grab 'p' tags that are NOT inside a table to avoid double-counting lists
            for child in art.find_all("p", class_="oj-normal"):
                if child != header_tag and not child.find_parent("table"):
                    loose_text.append(clean_text(child.get_text()))

            loose_points = []
            for tbl in art.find_all("table", recursive=False):
                loose_points.extend(parse_points_from_table(tbl))

            # Try regex parsing on the loose text
            main_loose, inline_subs = extract_inline_alpha_points(" ".join(loose_text))
            loose_points.extend(inline_subs)

            if loose_text or loose_points:
                # If we have points but main_loose is empty, format clean
                if not main_loose and loose_points:
                    main_loose = ""

                article_children.append(
                    {
                        "type": "paragraph",
                        "number": "1",
                        "text": main_loose,
                        "children": loose_points,
                    }
                )

        # Case B: Standard Numbered Paragraphs
        else:
            for p_div in paras:
                article_children.append(parse_paragraph_div(p_div))

        # Create Flattened Text for Search/Embeddings
        flat_text_blocks = [title_text]

        def flatten_node(node):
            if "text" in node:
                flat_text_blocks.append(node["text"])
            if "children" in node:
                for c in node["children"]:
                    flat_text_blocks.append(
                        f"{c.get('number', '')} {c.get('text', '')}"
                    )
                    flatten_node(c)

        for child in article_children:
            flat_text_blocks.append(child["text"])
            flatten_node(child)

        structured_data.append(
            {
                "id": art_id,
                "type": "article",
                "number": art_number,
                "title": title_text,
                "text": " ".join(flat_text_blocks),
                "children": article_children,
                "metadata": {
                    "section": "Enacting Terms",
                    "chapter": chapter_info,
                    "citation": f"Article {art_number}",
                },
            }
        )

    # --- 3. ANNEXES ---
    print("--- Processing Annexes ---")
    annexes = soup.find_all("div", id=re.compile(r"^anx_[IVX]+$"))
    for anx in annexes:
        anx_id = anx.get("id")
        title_tags = anx.find_all("p", class_="oj-doc-ti")
        title_text = " - ".join([clean_text(t.get_text()) for t in title_tags])
        if not title_text:
            title_text = anx_id

        annex_children = parse_annex_structure(anx)

        flat_anx_parts = [title_text]

        def recursive_flatten(nodes):
            for node in nodes:
                text = node.get("text", "") or node.get("title", "")
                num = node.get("number", "")
                flat_anx_parts.append(f"{num} {text}")
                if "children" in node:
                    recursive_flatten(node["children"])

        recursive_flatten(annex_children)

        structured_data.append(
            {
                "id": anx_id,
                "type": "annex",
                "title": title_text,
                "text": " ".join(flat_anx_parts),
                "children": annex_children,
                "metadata": {
                    "section": "Annexes",
                    "chapter": "Annexes",
                    "citation": title_text.split("-")[0].strip(),
                },
            }
        )

    # --- 4. SAVE ---
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
    print(f"Done! Saved {len(structured_data)} items to {output_json_path}")


if __name__ == "__main__":
    URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"
    FILE = "data/processed/ai_act_complete.json"
    fetch_and_parse(URL, FILE)
