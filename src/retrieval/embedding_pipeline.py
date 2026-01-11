import gc

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from src.utils.helper import (  # Assuming you saved the helper function
    get_device,
    load_json,
)


class EmbeddingWithDB:
    def __init__(self, collection_name="ai_act_legal"):
        """
        Initializes the Engine with a Persistent Vector DB.
        """
        self.device = get_device(verbose=True)
        self.current_model_name = None
        self.model = None

        # 1. Initialize Persistent ChromaDB
        # This creates a folder 'chroma_db' in your project directory
        self.client = chromadb.PersistentClient(path="data/chroma_db")

        # 2. Get or Create Collection
        # We use cosine distance for similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        print(
            f"💽 Vector Database loaded. Collection '{collection_name}' has {self.collection.count()} chunks."
        )

    def load_model(self, model_name):
        """
        Loads the embedding model (Same as before).
        """
        model_map = {
            "bge-m3": "BAAI/bge-m3",
            "e5-mistral": "intfloat/e5-mistral-7b-instruct",
        }
        hf_id = model_map.get(model_name, model_name)

        if self.current_model_name == hf_id:
            return

        # Memory Cleanup
        if self.model is not None:
            del self.model
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()

        print(f"⏳ Loading {model_name}...")
        self.model = SentenceTransformer(
            hf_id, device=self.device, trust_remote_code=True
        )
        if "mistral" in hf_id.lower():
            self.model.half()  # FP16 for Mac

        self.current_model_name = hf_id
        print(f"✅ Loaded {model_name}")

    def embed_and_store(self, chunks_data, reset_db=False):
        """
        Generates embeddings and saves them to ChromaDB.
        """
        if not self.model:
            raise ValueError("No model loaded.")

        if reset_db:
            print("⚠️  Resetting Database Collection...")
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name, metadata={"hnsw:space": "cosine"}
            )

        print(f"⚡ Generating Embeddings for {len(chunks_data)} chunks...")

        texts = [item["text_to_embed"] for item in chunks_data]
        ids = [item["chunk_id"] for item in chunks_data]
        metadatas = [item["metadata"] for item in chunks_data]

        # Generate Vectors
        embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
        )

        # ChromaDB expects lists, not numpy arrays
        embeddings_list = embeddings.tolist()

        print("💾 Saving to Disk (ChromaDB)...")
        # Add to DB (Batching is handled by Chroma, but for huge sets, batch manually)
        self.collection.add(
            documents=texts, embeddings=embeddings_list, metadatas=metadatas, ids=ids
        )
        print(
            f"✅ Stored {len(chunks_data)} chunks. Total DB size: {self.collection.count()}"
        )

    def search(self, query, k=5, filter_criteria=None):
        """
        Semantic Search with optional Metadata Filtering.

        filter_criteria example: {"type": "article"} or {"chapter": "II"}
        """
        if not self.model:
            raise ValueError("No model loaded.")

        # Format query for Mistral if needed
        if "mistral" in self.current_model_name.lower():
            formatted_query = f"Instruct: Retrieve relevant legal documents based on the query.\nQuery: {query}"
        else:
            formatted_query = query

        # 1. Embed Query
        query_vec = self.model.encode([formatted_query], convert_to_numpy=True).tolist()

        # 2. Query DB
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=k,
            where=filter_criteria,  # Magic happens here (Metadata filtering)
        )

        # 3. Format Output (Chroma returns a weird dict structure)
        parsed_results = []
        # Chroma returns lists of lists (because you can query multiple vectors at once)
        for i in range(len(results["ids"][0])):
            parsed_results.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "score": 1
                    - results["distances"][0][i],  # Convert distance to similarity
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
            )

        return parsed_results


# --------------------------------------------------------
# EXECUTION
# --------------------------------------------------------
if __name__ == "__main__":

    chunks = load_json("data/json/ai_act_chunks_split.json")
    engine = EmbeddingWithDB()

    # 1. Load Model
    engine.load_model("bge-m3")

    # 2. Embed & Store (Only do this ONCE. Comment out after first run!)
    engine.embed_and_store(chunks, reset_db=True)

    # 3. Search (Fast!)
    print("\n🔍 Searching for 'banned AI'...")
    results = engine.search("banned AI", k=2)
    for r in results:
        print(f"[{r['score']:.4f}] {r['chunk_id']}")

    # 4. Search with Filter (Example: Only look in Recitals)
    print("\n🔍 Searching with Filter (Type='recital')...")
    filtered = engine.search("high risk", k=2, filter_criteria={"type": "recital"})
    for r in filtered:
        print(f"[{r['score']:.4f}] {r['chunk_id']} ({r['metadata']['type']})")
