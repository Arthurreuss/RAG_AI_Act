import fitz
import re
import json
import os
from typing import List, Dict

def clean_text(text: str) -> str:
    text = text.replace('\n', ' ')
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_header_or_footer(block_rect, page_rect, threshold_pct = 0.08) -> bool:
    x0, y0, x1, y1 = block_rect
    page_width, page_height = page_rect.width, page_rect.height

    if y1 < (page_height * threshold_pct) or y0 > (page_height * (1 - threshold_pct)):
        return True
    return False

def extract_structured_content_from_pdf(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    full_text = list()

    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            if b[6] != 0:
                continue

            if is_header_or_footer(fitz.Rect(b[:4]), page.rect):
                continue

            text = b[4]
            full_text.append(text)

    document_content = "\n".join(full_text)
    
    chunks = list()
    
    split_pattern = re.compile(r"(Article\s+\d+|ANNEX\s+[IVXLCDM]+)", re.IGNORECASE)
    parts = split_pattern.split(document_content)
    if parts[0].strip():
        chunks.append({
            "id":"Recital/Preamble",
            "type":"recital",
            "text": clean_text(parts[0])
        })
    for i in range(1, len(parts), 2):
        marker = parts[i].strip()
        content = parts[i + 1] if (i + 1) < len(parts) else ""
        title_match = re.match(r"\s*[:\-\.]?\s*([^\n\.]+)", content)
        title = title_match.group(1).strip() if title_match else "General"
        chunk_type = "annex" if "ANNEX" in marker.upper() else "article"
        chunks.append({
            "id": f"{marker}:{title}",
            "type": chunk_type,
            "text": clean_text(content)
        })
    return chunks

def save_to_json(data: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Successfully processed {len(data)} chunks into {output_path}")

if __name__ == "__main__":
    INPUT_PDF = "data/raw/OJ_L_202401689_EN_TXT.pdf"
    OUTPUT_JSON = "data/processed/ai_act_structured.json"

    if not os.path.exists(INPUT_PDF):
        print(f"Error: File not found - {INPUT_PDF}")
    else:
        print(f"Parsing {INPUT_PDF}...")
        structured_data = extract_structured_content_from_pdf(INPUT_PDF)
        save_to_json(structured_data, OUTPUT_JSON)