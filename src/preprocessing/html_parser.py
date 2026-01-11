import json
import os
import re
import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Suppress XML parsing warnings from BS4
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


class AIActParser:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}

    # ==========================================
    # 1. UTILITIES & CLEANING
    # ==========================================

    def _clean_text(self, text):
        """Normalizes whitespace and removes non-breaking spaces."""
        if not text:
            return ""
        text = text.replace("\u00a0", " ").replace("\n", " ").replace("\t", " ")
        return re.sub(r"\s+", " ", text).strip()

    def _get_chapter_info(self, tag):
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
        chapter_num = self._clean_text(num_tag.get_text()) if num_tag else ""
        title_tag = chapter_div.find("p", class_="oj-ti-section-2")
        chapter_title = self._clean_text(title_tag.get_text()) if title_tag else ""
        return f"{chapter_num} - {chapter_title}".strip(" -")

    # ==========================================
    # 2. TEXT STRUCTURE PARSERS (REGEX LOGIC)
    # ==========================================

    def _extract_inline_roman_points(self, text):
        """
        Level 3: Detects inline roman numerals (i) to (vii).
        Added: Context-aware guard to prevent splitting citations like "Article 5(i)".
        """
        pattern = r"(?:\s|^)(\((?:vii|vi|v|iv|iii|ii|i)\))"

        matches = list(re.finditer(pattern, text))
        if not matches:
            return text, []

        valid_indices = []

        for m in matches:
            # --- CITATION GUARD (UPDATED) ---
            # Look closer at the prefix (last 20 chars)
            # We want to SKIP if we see: "Article 5(i)", "point 1(i)", "paragraph 2(i)"
            prefix = text[max(0, m.start() - 25) : m.start()]

            # Regex Explanation:
            # \b(points?|...) -> The keywords
            # \s* -> Optional space
            # (?:[\d\w\.]+\s*)? -> Optional number/alphanumeric (e.g. "1", "5a", "4.1") followed by optional space
            # $ -> End of the prefix (right before the match)
            is_citation = re.search(
                r"\b(?:points?|articles?|paragraphs?|sections?|regulations?|annex(?:es)?)\s*(?:[\d\w\.]+\s*)?$",
                prefix,
                re.IGNORECASE,
            )

            if is_citation:
                continue

            valid_indices.append(m)

        if not valid_indices:
            return text, []

        # Logic to split text based on valid indices
        parts = []
        last_end = 0

        # 1. Main Text (before first valid point)
        main_text = text[: valid_indices[0].start()].strip().rstrip(":")

        # 2. Extract Children
        children = []
        for i, m in enumerate(valid_indices):
            num = m.group(1)  # e.g., (i)
            start_content = m.end()

            if i + 1 < len(valid_indices):
                end_content = valid_indices[i + 1].start()
            else:
                end_content = len(text)

            content = text[start_content:end_content].strip().strip(";")
            children.append(
                {"type": "point", "number": num, "text": content, "children": []}
            )

        return main_text, children

    def _extract_inline_alpha_points(self, text):
        """
        Level 2: Detects inline letters (a), (b)... (z).
        Context-aware:
        1. Distinguishes between Letter (i) and Roman (i).
        2. Ignores citations like "Article 5(a)" or "point 1 (a)".
        """
        pattern = r"(?:\s|^)(\(([a-z])\))"

        matches = list(re.finditer(pattern, text))
        if not matches:
            return text, []

        valid_indices = []
        expected_char_ord = ord("a")

        for m in matches:
            char_found = m.group(2)
            char_ord = ord(char_found)

            # --- CITATION GUARD (UPDATED) ---
            prefix = text[max(0, m.start() - 25) : m.start()]

            # Checks for: "point (a)", "point 1(a)", "Article 40 (a)"
            is_citation = re.search(
                r"\b(?:points?|articles?|paragraphs?|sections?|regulations?|annex(?:es)?)\s*(?:[\d\w\.]+\s*)?$",
                prefix,
                re.IGNORECASE,
            )

            if is_citation:
                continue

            # --- SEQUENCE CHECK ---
            if char_ord == expected_char_ord:
                valid_indices.append(m)
                expected_char_ord += 1
            else:
                pass

        if not valid_indices:
            return text, []

        children = []
        main_text = text[: valid_indices[0].start()].strip().rstrip(":")

        for k, match in enumerate(valid_indices):
            current_num = match.group(1)
            start_content = match.end()

            if k + 1 < len(valid_indices):
                end_content = valid_indices[k + 1].start()
            else:
                end_content = len(text)

            content = text[start_content:end_content].strip().strip(";")

            # Recurse for Roman numerals inside the Alpha point
            child_main_text, child_subs = self._extract_inline_roman_points(content)

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

    def _parse_points_from_table(self, table_tag):
        """
        Advanced Table Parser:
        Handles 3-col (Annex I), 2-col (Standard), and Nested Tables (Annex III).
        """
        points = []
        search_root = table_tag.find("tbody") if table_tag.find("tbody") else table_tag
        rows = search_root.find_all("tr", recursive=False)

        for row in rows:
            cells = row.find_all("td", recursive=False)

            pt_num = ""
            content_cell = None

            # PATTERN A: 3 Columns (Spacer | Number | Text) -> e.g., Annex I
            if len(cells) == 3:
                pt_num = self._clean_text(cells[1].get_text())
                content_cell = cells[2]

            # PATTERN B: 2 Columns (Number/Dash | Text) -> e.g., Annex II, III
            elif len(cells) == 2:
                pt_num = self._clean_text(cells[0].get_text())
                content_cell = cells[1]

            # Process the row if we identified content
            if content_cell:
                sub_children = []

                # 1. RECURSION: Check for tables *inside* the content cell (Annex III)
                nested_tables = content_cell.find_all("table", recursive=False)
                for nt in nested_tables:
                    sub_children.extend(self._parse_points_from_table(nt))

                # 2. EXTRACT TEXT: careful not to duplicate text from nested tables
                text_bits = []
                for child in content_cell.children:
                    if child.name != "table":
                        text_bits.append(
                            self._clean_text(child.get_text() if child.name else child)
                        )

                raw_text = " ".join([t for t in text_bits if t])

                # --- FLATTEN DASH LISTS (Annex II) ---
                if pt_num == "—":
                    combined_text = f"— {raw_text}"
                    points.append({"type": "text_block", "text": combined_text})
                    # If there are sub_children (nested tables) in a dash list, we append them after
                    if sub_children:
                        points.extend(sub_children)
                    continue

                # 3. INLINE PARSING: If no table children, check for regex points
                if not sub_children:
                    main_text, inline_subs = self._extract_inline_alpha_points(raw_text)
                    if not inline_subs:
                        main_text, inline_subs = self._extract_inline_roman_points(
                            main_text
                        )
                else:
                    main_text = raw_text
                    inline_subs = []

                final_children = sub_children + inline_subs

                points.append(
                    {
                        "type": "point",
                        "number": pt_num,
                        "text": main_text,
                        "children": final_children,
                    }
                )

        return points

    def _parse_article_3_definitions(
        self, art_tag, header_text, title_text, chapter_info
    ):
        full_text_blocks = []
        for child in art_tag.find_all(["p", "table", "div"], recursive=False):
            if "oj-ti-art" in child.get("class", []) or "eli-title" in child.get(
                "class", []
            ):
                continue
            text = self._clean_text(child.get_text())
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
                def_main_text, def_subs = self._extract_inline_alpha_points(content)
                if not def_subs:
                    def_main_text, def_subs = self._extract_inline_roman_points(
                        def_main_text
                    )
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

    def _parse_paragraph_div(self, para_div):
        """
        Parses a paragraph container that may contain multiple 'logical' blocks.
        Handles cases like Article 43(1) where the structure is:
        Text -> List -> Text (New Sub-Para) -> List -> Text (New Sub-Para)
        """
        para_id = para_div.get("id")

        # 1. Determine the main number (e.g., "1") from the first text node
        first_p = para_div.find("p", class_="oj-normal")
        full_text_start = self._clean_text(first_p.get_text()) if first_p else ""
        num_match = re.match(r"^(\d+)\.", full_text_start)
        main_number = num_match.group(1) if num_match else None

        # 2. Linear Scan to group content into logical blocks
        sub_blocks = []
        current_block = {"text_parts": [], "children": []}

        for child in para_div.children:
            # --- Case A: It is a Table (List of Points) ---
            if child.name == "table":
                points = self._parse_points_from_table(child)
                current_block["children"].extend(points)

            # --- Case B: It is Text ---
            elif child.name == "p" or isinstance(child, str):
                text = self._clean_text(child.get_text() if child.name else child)
                if not text:
                    continue

                # CRITICAL LOGIC:
                # If we encounter text AND the current block already has children (points),
                # it means the previous logical block (Intro + List) is finished.
                # We must start a NEW block for this new text.
                if current_block["children"]:
                    sub_blocks.append(current_block)
                    current_block = {"text_parts": [], "children": []}

                current_block["text_parts"].append(text)

        # Append the final block currently in progress
        if current_block["text_parts"] or current_block["children"]:
            sub_blocks.append(current_block)

        # 3. Construct the Final Object
        # The first block is the "Root" Paragraph (e.g., "1")
        if not sub_blocks:
            return None  # Should not happen given HTML structure

        root_block = sub_blocks[0]
        root_text = " ".join(root_block["text_parts"])

        # Clean inline regex points for the root text
        root_text, root_inline = self._extract_inline_alpha_points(root_text)
        if not root_inline:
            root_text, root_inline = self._extract_inline_roman_points(root_text)

        final_children = root_block["children"] + root_inline

        # Process subsequent blocks as "Paragraph 1.x"
        for i, block in enumerate(sub_blocks[1:], start=1):
            block_text = " ".join(block["text_parts"])

            # Clean inline regex points for sub-block text
            b_text, b_inline = self._extract_inline_alpha_points(block_text)
            if not b_inline:
                b_text, b_inline = self._extract_inline_roman_points(b_text)

            # Calculate sub-number (e.g., 1.1, 1.2)
            sub_num = f"{main_number}.{i}" if main_number else f"{i}"

            sub_node = {
                "type": "paragraph",  # Explicitly requested type
                "number": sub_num,
                "text": b_text,
                "children": block["children"] + b_inline,
            }
            final_children.append(sub_node)

        return {
            "type": "paragraph",
            "id": para_id,
            "number": main_number,
            "text": root_text,
            "children": final_children,
        }

    def _parse_annex_structure(self, annex_div):
        """
        Parses complex Annex structures.
        FIXES:
        1. Merges consecutive section titles (Annex XI).
        2. Treats subtitles as text blocks (Annex XII).
        3. Handles 'oj-enumeration-spacing' divs (Annex VI) as Points.
        """
        children = []
        current_section = None

        for child in annex_div.find_all(["p", "table", "div"], recursive=False):

            # --- 1. SECTIONS ---
            if child.name == "p" and (
                "oj-ti-grseq-1" in child.get("class", [])
                or "oj-ti-section-1" in child.get("class", [])
            ):
                text = self._clean_text(child.get_text())

                if current_section and not current_section["children"]:
                    current_section["text"] += " " + text
                else:
                    if current_section:
                        children.append(current_section)
                    current_section = {
                        "type": "section",
                        "text": text,
                        "children": [],
                    }
                continue

            # --- 2. LISTS (TABLES) ---
            if child.name == "table":
                points = self._parse_points_from_table(child)
                if current_section:
                    current_section["children"].extend(points)
                else:
                    children.extend(points)

            # --- 3. LISTS (DIV ENUMERATION - Annex VI fix) ---
            elif child.name == "div" and "oj-enumeration-spacing" in child.get(
                "class", []
            ):
                sub_ps = child.find_all("p")
                if len(sub_ps) >= 2:
                    pt_num = self._clean_text(sub_ps[0].get_text())
                    raw_text = self._clean_text(sub_ps[1].get_text())

                    # Check for inline sub-points (a), (b)
                    main_text, inline_subs = self._extract_inline_alpha_points(raw_text)

                    point_data = {
                        "type": "point",
                        "number": pt_num,
                        "text": main_text,
                        "children": inline_subs,
                    }

                    if current_section:
                        current_section["children"].append(point_data)
                    else:
                        children.append(point_data)

            # --- 4. LOOSE TEXT ---
            elif child.name == "p" and "oj-normal" in child.get("class", []):
                text = self._clean_text(child.get_text())
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

    def fetch_and_parse(self, url):
        print(f"Fetching content from: {url}")
        try:
            response = requests.get(url, headers=self.headers)
            html_content = response.text
        except Exception as e:
            print(f"Failed to fetch: {e}")
            return

        print("Parsing HTML content...")
        soup = BeautifulSoup(html_content, "lxml")
        structured_data = []

        # --- RECITALS ---
        print("--- Processing Recitals ---")
        recitals = soup.find_all("div", id=re.compile(r"^rct_\d+$"))
        for rct in recitals:
            rct_id = rct.get("id")
            rct_text, rct_num = "", ""
            table = rct.find("table")
            if table:
                cells = table.find_all("td")
                if len(cells) >= 2:
                    rct_num = self._clean_text(cells[0].get_text()).strip("()")
                    rct_text = self._clean_text(cells[1].get_text())
            else:
                text_raw = self._clean_text(rct.get_text())
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

        # --- ARTICLES ---
        print("--- Processing Articles ---")
        articles = soup.find_all("div", id=re.compile(r"^art_\d+$"))
        for art in articles:
            art_id = art.get("id")
            header_tag = art.find("p", class_="oj-ti-art")
            header_text = (
                self._clean_text(header_tag.get_text()) if header_tag else art_id
            )

            art_number_match = re.search(r"\d+", header_text)
            art_number = art_number_match.group(0) if art_number_match else "0"

            if (
                art_number
                in [  # TODO: add logic to add what is in them to the original articles and annexes
                    "102",
                    "103",
                    "104",
                    "105",
                    "106",
                    "107",
                    "108",
                    "109",
                    "110",
                ]
            ):
                print(f"Skipping Article {art_number}...")
                continue

            title_div = art.find("div", class_="eli-title")
            title_text = (
                self._clean_text(title_div.get_text()) if title_div else "No Title"
            )
            chapter_info = self._get_chapter_info(art)

            if art_number == "3":
                structured_data.append(
                    self._parse_article_3_definitions(
                        art, header_text, title_text, chapter_info
                    )
                )
                continue

            article_children = []
            paras = art.find_all("div", id=re.compile(r"^\d{3}\.\d{3}$"))

            if not paras:
                loose_text = []
                for child in art.find_all("p", class_="oj-normal"):
                    if child != header_tag and not child.find_parent("table"):
                        loose_text.append(self._clean_text(child.get_text()))

                loose_points = []
                for tbl in art.find_all("table", recursive=False):
                    loose_points.extend(self._parse_points_from_table(tbl))

                main_loose, inline_subs = self._extract_inline_alpha_points(
                    " ".join(loose_text)
                )
                loose_points.extend(inline_subs)

                if loose_text or loose_points:
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
            else:
                for p_div in paras:
                    article_children.append(self._parse_paragraph_div(p_div))

            flat_text_blocks = [title_text]

            def flatten_node(node):
                # 1. Add the text of the CURRENT node
                if "text" in node and node["text"]:
                    # Optional: prepend number if it's not the root Article text
                    prefix = (
                        f"{node.get('number', '')} "
                        if node.get("number") and node.get("type") != "paragraph"
                        else ""
                    )
                    flat_text_blocks.append(f"{prefix}{node['text']}".strip())

                # 2. Recurse children
                if "children" in node:
                    for c in node["children"]:
                        flatten_node(c)

            # Process top-level children (Paragraphs)
            for child in article_children:
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

        # --- ANNEXES ---
        print("--- Processing Annexes ---")
        annexes = soup.find_all("div", id=re.compile(r"^anx_[IVX]+$"))
        for anx in annexes:
            anx_id = anx.get("id")
            title_tags = anx.find_all("p", class_="oj-doc-ti")
            title_text = " - ".join(
                [self._clean_text(t.get_text()) for t in title_tags]
            )
            if not title_text:
                title_text = anx_id

            annex_children = self._parse_annex_structure(anx)

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

        return structured_data
