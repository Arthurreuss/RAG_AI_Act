import json
import os
import re
import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Silence XML warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def clean_text(text):
    if not text:
        return ""
    text = text.replace("\u00a0", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def get_chapter_info(tag):
    """Traverses up to find the Chapter."""
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


# --- Shared Parsing Logic ---


def parse_points_from_table(table_tag):
    """Extracts points (a), (b) from standard tables."""
    points = []
    rows = table_tag.find_all("tr", recursive=False)
    for row in rows:
        cells = row.find_all("td", recursive=False)
        # Standard Article Table: col 1 = (a), col 2 = text
        if len(cells) >= 2:
            pt_num = clean_text(cells[0].get_text())
            pt_text = clean_text(cells[1].get_text())
            if pt_num.startswith("(") and pt_num.endswith(")"):
                points.append({"type": "point", "number": pt_num, "text": pt_text})
            else:
                if pt_text:
                    points.append({"type": "text_block", "text": pt_text})
    return points


def parse_paragraph_div(para_div):
    """Parses Article Paragraphs."""
    para_id = para_div.get("id")
    intro = []
    children = []
    for child in para_div.children:
        if child.name == "table":
            children.extend(parse_points_from_table(child))
        elif child.name == "p" or isinstance(child, str):
            t = clean_text(child.get_text() if child.name else child)
            if t:
                intro.append(t)

    full_text = " ".join(intro)
    num_match = re.match(r"^(\d+)\.", full_text)

    return {
        "type": "paragraph",
        "id": para_id,
        "number": num_match.group(1) if num_match else None,
        "text": full_text,
        "children": children,
    }


# --- NEW: Specific Logic for Annexes ---


def parse_annex_table_row(row):
    """
    Parses a row from an Annex table.
    Annex tables often have 3 columns: [Empty/Spacing] | [Number (1., 3.1.)] | [Content]
    """
    cells = row.find_all("td", recursive=False)

    # Case 1: Standard 3-column Annex layout
    if len(cells) >= 3:
        # Cell 0 is often spacer
        number = clean_text(cells[1].get_text())
        content_cell = cells[2]

        # The content cell might contain nested tables for sub-points (a), (b)
        content_text_parts = []
        sub_children = []

        for child in content_cell.children:
            if child.name == "table":
                # Recursively extract points from nested table
                sub_children.extend(parse_points_from_table(child))
            else:
                t = clean_text(child.get_text() if child.name else child)
                if t:
                    content_text_parts.append(t)

        return {
            "type": "point",
            "number": number.strip("."),
            "text": " ".join(content_text_parts),
            "children": sub_children,
        }

    # Fallback
    return {"type": "text_block", "text": clean_text(row.get_text())}


def parse_annex_structure(annex_div):
    """
    State-machine parser for Annexes.
    Groups content under Sections (e.g., 'Section A', '1. Introduction').
    """
    children = []
    current_section = None

    # Iterate over direct children of the Annex div
    for child in annex_div.find_all(["p", "table", "div"], recursive=False):

        # 1. Detect Section Headers
        # They usually have class 'oj-ti-grseq-1' (e.g. "Section A")
        # OR they are 'oj-doc-ti' (Titles) - we might skip the main title if parsed earlier
        if child.name == "p" and (
            "oj-ti-grseq-1" in child.get("class", [])
            or "oj-ti-section-1" in child.get("class", [])
        ):
            # Save previous section
            if current_section:
                children.append(current_section)

            # Start new section
            section_title = clean_text(child.get_text())
            current_section = {
                "type": "section",
                "title": section_title,
                "children": [],
            }
            continue

        # 2. Detect Content (Tables)
        if child.name == "table":
            # Annex content is almost always in tables
            rows = child.find_all("tr", recursive=False)
            for row in rows:
                item = parse_annex_table_row(row)
                if item["text"]:  # Only add if text exists
                    if current_section:
                        current_section["children"].append(item)
                    else:
                        children.append(item)

        # 3. Detect Loose Text
        elif child.name == "p" and "oj-normal" in child.get("class", []):
            text = clean_text(child.get_text())
            if text:
                item = {"type": "text_block", "text": text}
                if current_section:
                    current_section["children"].append(item)
                else:
                    children.append(item)

    # Append the last section
    if current_section:
        children.append(current_section)

    return children


def fetch_and_parse(url, output_json_path):
    print(f"Fetching content from: {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        html_content = response.text
    except requests.RequestException as e:
        print(f"Failed: {e}")
        return

    print("Parsing HTML content...")
    soup = BeautifulSoup(html_content, "lxml")
    structured_data = []

    # ==========================================
    # 1. PARSE RECITALS (Preamble)
    # ==========================================
    print("--- Processing Recitals ---")
    recitals = soup.find_all("div", id=re.compile(r"^rct_\d+$"))
    count_recitals = 0
    for rct in recitals:
        rct_id = rct.get("id")
        rct_text = ""
        rct_num = ""

        # Check table
        table = rct.find("table")
        if table:
            cells = table.find_all("td")
            if len(cells) >= 2:
                rct_num = clean_text(cells[0].get_text()).strip("()")
                rct_text = clean_text(cells[1].get_text())

        # Check paragraph
        if not rct_text:
            text_raw = clean_text(rct.get_text())
            match = re.match(r"\((\d+)\)\s*(.*)", text_raw)
            if match:
                rct_num = match.group(1)
                rct_text = match.group(2)
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
            count_recitals += 1
    print(f"  -> {count_recitals} Recitals.")

    # ==========================================
    # 2. PARSE ARTICLES (Enacting Terms)
    # ==========================================
    print("--- Processing Articles ---")
    articles = soup.find_all("div", id=re.compile(r"^art_\d+$"))
    count_articles = 0

    for art in articles:
        art_id = art.get("id")

        # Headers
        header_tag = art.find("p", class_="oj-ti-art")
        header_text = clean_text(header_tag.get_text()) if header_tag else art_id
        art_number = (
            re.search(r"\d+", header_text).group(0)
            if re.search(r"\d+", header_text)
            else "0"
        )
        title_div = art.find("div", class_="eli-title")
        title_text = clean_text(title_div.get_text()) if title_div else "No Title"

        # Hierarchy
        article_children = []
        paras = art.find_all("div", id=re.compile(r"^\d{3}\.\d{3}$"))

        if not paras:
            loose_text = []
            for child in art.find_all("p", class_="oj-normal"):
                if child != header_tag:
                    loose_text.append(clean_text(child.get_text()))

            # Tables in un-numbered articles
            loose_points = []
            for tbl in art.find_all("table", recursive=False):
                loose_points.extend(parse_points_from_table(tbl))

            if loose_text or loose_points:
                article_children.append(
                    {
                        "type": "paragraph_loose",
                        "text": " ".join(loose_text),
                        "children": loose_points,
                    }
                )
        else:
            for p_div in paras:
                article_children.append(parse_paragraph_div(p_div))

        # Flatten
        flat_text = f"{title_text} "
        for child in article_children:
            flat_text += child["text"] + " "
            for sub in child.get("children", []):
                flat_text += f"{sub.get('number', '')} {sub.get('text', '')} "

        structured_data.append(
            {
                "id": art_id,
                "type": "article",
                "number": art_number,
                "title": title_text,
                "text": clean_text(flat_text),
                "children": article_children,
                "metadata": {
                    "section": "Enacting Terms",
                    "chapter": get_chapter_info(art),
                    "citation": f"Article {art_number}",
                },
            }
        )
        count_articles += 1
    print(f"  -> {count_articles} Articles.")

    # ==========================================
    # 3. PARSE ANNEXES (Improved)
    # ==========================================
    print("--- Processing Annexes ---")
    annexes = soup.find_all("div", id=re.compile(r"^anx_[IVX]+$"))
    count_annexes = 0

    for anx in annexes:
        anx_id = anx.get("id")

        # 1. Get Annex Title (e.g. "ANNEX I")
        title_tags = anx.find_all("p", class_="oj-doc-ti")
        # Combine multiple title lines (e.g. "ANNEX I" + "List of...")
        title_text = " - ".join([clean_text(t.get_text()) for t in title_tags])
        if not title_text:
            title_text = anx_id

        # 2. Parse Internal Structure (Sections -> Points)
        annex_children = parse_annex_structure(anx)

        # 3. Flatten for embedding
        flat_anx_parts = [title_text]
        for child in annex_children:
            if child["type"] == "section":
                flat_anx_parts.append(child["title"])
                for sub in child["children"]:
                    flat_anx_parts.append(
                        f"{sub.get('number', '')} {sub.get('text', '')}"
                    )
            else:
                flat_anx_parts.append(
                    f"{child.get('number', '')} {child.get('text', '')}"
                )

        flat_anx_text = " ".join(flat_anx_parts)

        structured_data.append(
            {
                "id": anx_id,
                "type": "annex",
                "title": title_text,
                "text": clean_text(flat_anx_text),
                "children": annex_children,
                "metadata": {
                    "section": "Annexes",
                    "chapter": "Annexes",
                    "citation": title_text.split("-")[0].strip(),  # e.g. "ANNEX I"
                },
            }
        )
        count_annexes += 1
    print(f"  -> {count_annexes} Annexes.")

    # Save
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(structured_data)} total items to: {output_json_path}")


if __name__ == "__main__":
    URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"
    FILE = "data/processed/ai_act_complete.json"
    fetch_and_parse(URL, FILE)
