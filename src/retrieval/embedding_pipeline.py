import gc

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from src.utils.helper import get_device


class EmbeddingWithDB:
    def __init__(self, collection_name="ai_act_legal"):
        """
        Initializes the Engine with a Persistent Vector DB.
        """
        self.device = get_device(verbose=True)
        self.current_model_name = None
        self.model = None
        self.client = chromadb.PersistentClient(path="data/chroma_db")

        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        print(
            f"Vector Database loaded. Collection '{collection_name}' has {self.collection.count()} chunks."
        )

    def load_model(self, model_name):
        """
        Loads the embedding model based on key.
        """
        model_map = {
            "bge-m3": "BAAI/bge-m3",
            "snowflake-m": "Snowflake/snowflake-arctic-embed-m",
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        }

        hf_id = model_map.get(model_name, model_name)

        if self.current_model_name == hf_id:
            return

        # Cleanup Memory
        if self.model is not None:
            del self.model
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

        print(f"Loading {model_name} ({hf_id})...")

        self.model = SentenceTransformer(
            hf_id, device=self.device, trust_remote_code=True
        )

        self.current_model_name = hf_id
        print(f"Loaded {model_name}")

    def embed_and_store(self, chunks_data, reset_db=False):

        if not self.model:
            raise ValueError("No model loaded.")

        if reset_db:
            print("Resetting Database Collection...")
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name, metadata={"hnsw:space": "cosine"}
            )

        print(f"Generating Embeddings for {len(chunks_data)} chunks...")
        texts = [item["text_to_embed"] for item in chunks_data]
        ids = [item["chunk_id"] for item in chunks_data]
        metadatas = [item["metadata"] for item in chunks_data]

        embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
        )

        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
        )
        print(f"Stored {len(chunks_data)} chunks.")

    def search(self, query, k=5):
        if not self.model:
            raise ValueError("No model loaded.")

        if "snowflake" in self.current_model_name.lower():
            formatted_query = (
                f"Represent this sentence for searching relevant passages: {query}"
            )
        else:
            formatted_query = query

        query_vec = self.model.encode([formatted_query], convert_to_numpy=True).tolist()

        results = self.collection.query(query_embeddings=query_vec, n_results=k)

        parsed = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                parsed.append(
                    {
                        "chunk_id": results["ids"][0][i],
                        "score": 1 - results["distances"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                    }
                )
        return parsed
