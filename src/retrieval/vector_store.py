import chromadb
import chromadb.errors
from chromadb.utils import embedding_functions
import json
import os
import time

CHROMA_PATH = "data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_chorma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

def index_data(collection_name: str, json_path: str):
    if not os.path.exists(json_path):
        print(f"Skipping {collection_name}: File {json_path} does not exist.")
        return
    
    print(f"Indexing {collection_name} from {json_path}...")

    with open(json_path, 'r', encoding='utf-8') as file:
        chunks = json.load(file)

    client = get_chorma_client()
    ef = get_embedding_function()

    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except (ValueError, chromadb.errors.NotFoundError):
        pass
    except Exception as e:
        print(f"Error deleting collection {collection_name}: {e}")
    
    collection = client.create_collection(
        name=collection_name,
        embedding_function=ef
    )

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c.get("metadata", {}) for c in chunks]

    BATCH_SIZE = 100
    total = len(chunks)

    start_time = time.time()

    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        print(f" Processing batch {i} to {end}...")
        
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )

    duration = time.time() - start_time
    print(f"Successfully indexed {total} chunks into {collection_name} in {duration:.2f} seconds.")

if __name__ == "__main__":
    
    index_data(
        collection_name="ai_act_naive_chunks",
        json_path="data/processed/ai_act_naive_chunks.json"
    )

    index_data(
        collection_name="ai_act_chunks",
        json_path="data/processed/ai_act_semantic_chunks.json"
    )