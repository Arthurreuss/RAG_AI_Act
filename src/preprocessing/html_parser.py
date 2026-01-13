import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

# Suppress XML parsing warnings from BS4
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


class AIActParser:
    """A parser for the European Union AI Act legal text.

    This class fetches the official HTML version of the AI Act and parses it into
    a structured JSON-like format, handling hierarchical structures such as
    Recitals, Articles, Paragraphs, Points, and Annexes.
    """

    def __init__(self) -> None:
        """Initializes the parser with standard request headers."""
        self.headers: Dict[str, str] = {"User-Agent": "Mozilla/5.0"}

    def _clean_text(self, text: Optional[str]) -> str:
        """Normalizes whitespace and removes non-breaking spaces.

        Args:
            text: The raw string to be cleaned.

        Returns:
            A string with normalized spaces and no tabs or newlines.
        """
        if not text:
            return ""
        text = text.replace("\u00a0", " ").replace("\n", " ").replace("\t", " ")
        return re.sub(r"\s+", " ", text).strip()

    def _get_chapter_info(self, tag: Tag) -> str:
        """Finds the chapter context for a given HTML tag or article.

        Args:
            tag: The BeautifulSoup Tag representing the current article or segment.

        Returns:
            A string describing the chapter number and title, or a general
            section name (e.g., 'Preamble', 'Annexes').
        """
        chapter_div = tag.find_parent("div", id=re.compile(r"^cpt_[IVX\d]+$"))
        if not chapter_div:
            tag_id: str = tag.get("id", "")
            if tag_id.startswith("rct_"):
                return "Preamble"
            if tag_id.startswith("anx_"):
                return "Annexes"
            return "General Provisions"

        num_tag = chapter_div.find("p", class_="oj-ti-section-1")
        chapter_num: str = self._clean_text(num_tag.get_text()) if num_tag else ""
        title_tag = chapter_div.find("p", class_="oj-ti-section-2")
        chapter_title: str = self._clean_text(title_tag.get_text()) if title_tag else ""
        return f"{chapter_num} - {chapter_title}".strip(" -")

    def _extract_inline_roman_points(
        self, text: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Detects and extracts inline roman numerals (i) to (vii) from a text block.

        Includes context-aware guards to prevent splitting legal citations
        like 'Article 5(i)'.

        Args:
            text: The text string to analyze for inline points.

        Returns:
            A tuple containing:
                - The lead-in text before the first point.
                - A list of dictionary objects representing the extracted points.
        """
        pattern = r"(?:\s|^)(\((?:vii|vi|v|iv|iii|ii|i)\))"
        matches = list(re.finditer(pattern, text))
        if not matches:
            return text, []

        valid_indices = []
        for m in matches:
            prefix = text[max(0, m.start() - 25) : m.start()]
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

        main_text = text[: valid_indices[0].start()].strip().rstrip(":")
        children = []
        for i, m in enumerate(valid_indices):
            num = m.group(1)
            start_content = m.end()
            end_content = (
                valid_indices[i + 1].start()
                if i + 1 < len(valid_indices)
                else len(text)
            )
            content = text[start_content:end_content].strip().strip(";")
            children.append(
                {"type": "point", "number": num, "text": content, "children": []}
            )

        return main_text, children

    def _extract_inline_alpha_points(
        self, text: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Detects and extracts inline alphabetic points (a), (b)... (z).

        Distinguishes between alphabetic letters and roman numerals (like 'i')
        and ignores citations such as 'point 1(a)'.

        Args:
            text: The text string to analyze.

        Returns:
            A tuple containing the lead-in text and a list of point objects.
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
            prefix = text[max(0, m.start() - 25) : m.start()]
            is_citation = re.search(
                r"\b(?:points?|articles?|paragraphs?|sections?|regulations?|annex(?:es)?)\s*(?:[\d\w\.]+\s*)?$",
                prefix,
                re.IGNORECASE,
            )

            if is_citation:
                continue

            if char_ord == expected_char_ord:
                valid_indices.append(m)
                expected_char_ord += 1

        if not valid_indices:
            return text, []

        children = []
        main_text = text[: valid_indices[0].start()].strip().rstrip(":")

        for k, match in enumerate(valid_indices):
            current_num = match.group(1)
            start_content = match.end()
            end_content = (
                valid_indices[k + 1].start()
                if k + 1 < len(valid_indices)
                else len(text)
            )
            content = text[start_content:end_content].strip().strip(";")

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

    def _parse_points_from_table(self, table_tag: Tag) -> List[Dict[str, Any]]:
        """Parses list points structured inside HTML tables.

        Handles 3-column (Annex I), 2-column (Standard), and nested table formats
        found in Annex III.

        Args:
            table_tag: The BeautifulSoup Tag representing the table.

        Returns:
            A list of dictionary objects representing points or text blocks.
        """
        points = []
        search_root = table_tag.find("tbody") if table_tag.find("tbody") else table_tag
        rows = search_root.find_all("tr", recursive=False)

        for row in rows:
            cells = row.find_all("td", recursive=False)
            pt_num = ""
            content_cell = None

            if len(cells) == 3:
                pt_num = self._clean_text(cells[1].get_text())
                content_cell = cells[2]
            elif len(cells) == 2:
                pt_num = self._clean_text(cells[0].get_text())
                content_cell = cells[1]

            if content_cell:
                sub_children = []
                nested_tables = content_cell.find_all("table", recursive=False)
                for nt in nested_tables:
                    sub_children.extend(self._parse_points_from_table(nt))

                text_bits = []
                for child in content_cell.children:
                    if child.name != "table":
                        text_bits.append(
                            self._clean_text(child.get_text() if child.name else child)
                        )

                raw_text = " ".join([t for t in text_bits if t])

                if pt_num == "—":
                    points.append({"type": "text_block", "text": f"— {raw_text}"})
                    if sub_children:
                        points.extend(sub_children)
                    continue

                if not sub_children:
                    main_text, inline_subs = self._extract_inline_alpha_points(raw_text)
                    if not inline_subs:
                        main_text, inline_subs = self._extract_inline_roman_points(
                            main_text
                        )
                else:
                    main_text = raw_text
                    inline_subs = []

                points.append(
                    {
                        "type": "point",
                        "number": pt_num,
                        "text": main_text,
                        "children": sub_children + inline_subs,
                    }
                )
        return points

    def _parse_article_3_definitions(
        self, art_tag: Tag, header_text: str, title_text: str, chapter_info: str
    ) -> Dict[str, Any]:
        """Specialized parser for Article 3 which contains dozens of definitions.

        Args:
            art_tag: The HTML tag for Article 3.
            header_text: The string representing the article header (e.g., 'Article 3').
            title_text: The title of the article.
            chapter_info: Chapter metadata.

        Returns:
            A structured dictionary for Article 3 and its definitions.
        """
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

    def _parse_paragraph_div(self, para_div: Tag) -> Optional[Dict[str, Any]]:
        """Parses a paragraph container that may contain multiple logical blocks.

        Args:
            para_div: The HTML div containing one or more paragraph blocks.

        Returns:
            A dictionary representing the paragraph and its nested structure,
            or None if no content is found.
        """
        para_id = para_div.get("id")
        first_p = para_div.find("p", class_="oj-normal")
        full_text_start = self._clean_text(first_p.get_text()) if first_p else ""
        num_match = re.match(r"^(\d+)\.", full_text_start)
        main_number = num_match.group(1) if num_match else None

        sub_blocks = []
        current_block = {"text_parts": [], "children": []}

        for child in para_div.children:
            if child.name == "table":
                current_block["children"].extend(self._parse_points_from_table(child))
            elif child.name == "p" or isinstance(child, str):
                text = self._clean_text(child.get_text() if child.name else child)
                if not text:
                    continue
                if current_block["children"]:
                    sub_blocks.append(current_block)
                    current_block = {"text_parts": [], "children": []}
                current_block["text_parts"].append(text)

        if current_block["text_parts"] or current_block["children"]:
            sub_blocks.append(current_block)

        if not sub_blocks:
            return None

        root_block = sub_blocks[0]
        root_text, root_inline = self._extract_inline_alpha_points(
            " ".join(root_block["text_parts"])
        )
        if not root_inline:
            root_text, root_inline = self._extract_inline_roman_points(root_text)

        final_children = root_block["children"] + root_inline

        for i, block in enumerate(sub_blocks[1:], start=1):
            b_text, b_inline = self._extract_inline_alpha_points(
                " ".join(block["text_parts"])
            )
            if not b_inline:
                b_text, b_inline = self._extract_inline_roman_points(b_text)

            sub_num = f"{main_number}.{i}" if main_number else f"{i}"
            final_children.append(
                {
                    "type": "paragraph",
                    "number": sub_num,
                    "text": b_text,
                    "children": block["children"] + b_inline,
                }
            )

        return {
            "type": "paragraph",
            "id": para_id,
            "number": main_number,
            "text": root_text,
            "children": final_children,
        }

    def _parse_annex_structure(self, annex_div: Tag) -> List[Dict[str, Any]]:
        """Parses the complex and varied structures of the document Annexes.

        Args:
            annex_div: The HTML div representing an Annex section.

        Returns:
            A list of sections, points, and text blocks within the Annex.
        """
        children = []
        current_section = None

        for child in annex_div.find_all(["p", "table", "div"], recursive=False):
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
                    current_section = {"type": "section", "text": text, "children": []}
                continue

            if child.name == "table":
                points = self._parse_points_from_table(child)
                if current_section:
                    current_section["children"].extend(points)
                else:
                    children.extend(points)

            elif child.name == "div" and "oj-enumeration-spacing" in child.get(
                "class", []
            ):
                sub_ps = child.find_all("p")
                if len(sub_ps) >= 2:
                    pt_num = self._clean_text(sub_ps[0].get_text())
                    raw_text = self._clean_text(sub_ps[1].get_text())
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

    def fetch_and_parse(self, url: str) -> List[Dict[str, Any]]:
        """Main entry point to fetch and parse the AI Act from a EUR-Lex URL.

        Args:
            url: The direct URL to the HTML version of the AI Act.

        Returns:
            A list of dictionaries containing the full structured data of
            the Act.
        """
        print(f"Fetching content from: {url}")
        try:
            response = requests.get(url, headers=self.headers)
            html_content = response.text
        except Exception as e:
            print(f"Failed to fetch: {e}")
            return []

        print("Parsing HTML content...")
        soup = BeautifulSoup(html_content, "lxml")
        structured_data: List[Dict[str, Any]] = []

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
            art_id: str = art.get("id", "")
            header_tag = art.find("p", class_="oj-ti-art")
            header_text = (
                self._clean_text(header_tag.get_text()) if header_tag else art_id
            )
            art_number_match = re.search(r"\d+", header_text)
            art_number = art_number_match.group(0) if art_number_match else "0"

            if art_number in [
                "102",
                "103",
                "104",
                "105",
                "106",
                "107",
                "108",
                "109",
                "110",
            ]:
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
                    article_children.append(
                        {
                            "type": "paragraph",
                            "number": "1",
                            "text": main_loose or "",
                            "children": loose_points,
                        }
                    )
            else:
                for p_div in paras:
                    res = self._parse_paragraph_div(p_div)
                    if res:
                        article_children.append(res)

            flat_text_blocks = [title_text]

            def flatten_node(node: Dict[str, Any]) -> None:
                if node.get("text"):
                    prefix = (
                        f"{node.get('number', '')} "
                        if node.get("number") and node.get("type") != "paragraph"
                        else ""
                    )
                    flat_text_blocks.append(f"{prefix}{node['text']}".strip())
                for c in node.get("children", []):
                    flatten_node(c)

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
            title_text = (
                " - ".join([self._clean_text(t.get_text()) for t in title_tags])
                or anx_id
            )
            annex_children = self._parse_annex_structure(anx)

            flat_anx_parts = [title_text]

            def recursive_flatten(nodes: List[Dict[str, Any]]) -> None:
                for node in nodes:
                    text = node.get("text", "") or node.get("title", "")
                    num = node.get("number", "")
                    flat_anx_parts.append(f"{num} {text}")
                    recursive_flatten(node.get("children", []))

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
