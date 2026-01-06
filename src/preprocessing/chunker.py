import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def create_naive_chunks(data):
    print("Generating naive chunks...")
    full_text = "\n\n".join([item['text'] for item in data if item.get('text')])
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    text_chunks = splitter.split_text(full_text)

    structured_chunks = list()
    for i, text in enumerate(text_chunks):
        structured_chunks.append({
            "id": f"chunk_{i}",
            "text": text,
            "type": "naive_chunk",
            "metadata": {"source": "AI Act (Naive)"},
        })
    
    return structured_chunks
    
def create_semantic_chunks(data):
    print("Generating semantic chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )

    structured_chunks = []
    chunk_counter = 0

    for item in data:
        if len(item['text']) < CHUNK_SIZE:
            structured_chunks.append({
                "id": f"chunk_{chunk_counter}",
                "text": item['text'],
                "metadata": {
                    "source": f"AI Act (Semantic)",
                    "citation":item["id"],
                    "type": item["type"]
                    },
            })
            chunk_counter += 1
        else:
            sub_chunks = splitter.split_text(item['text'])
            for sub_text in sub_chunks:
                structured_chunks.append({
                    "id": f"sem_{chunk_counter}",
                    "text": sub_text,
                    "metadata": {
                        "source": f"AI Act (Semantic)",
                        "citation":item["id"],
                        "type": item["type"]
                    },
                })
                chunk_counter += 1
    return structured_chunks

def save_to_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    print(f"Successfully saved {len(data)} chunks to {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "data/processed/ai_act_structured.json"
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found - {INPUT_FILE}")
    else:
        raw_data = load_data(INPUT_FILE)

        naive_chunks = create_naive_chunks(raw_data)
        save_to_json(naive_chunks, "data/processed/ai_act_naive_chunks.json")

        semantic_chunks = create_semantic_chunks(raw_data)
        save_to_json(semantic_chunks, "data/processed/ai_act_semantic_chunks.json")