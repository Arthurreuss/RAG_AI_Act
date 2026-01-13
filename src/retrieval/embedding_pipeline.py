import gc
from typing import Any, Dict, List, Optional

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from src.utils.helper import get_device


class EmbeddingWithDB:
    """An interface for managing a persistent vector database with embedding models.

    This class handles the initialization of ChromaDB, loading of SentenceTransformer
    models with appropriate memory management, and the core operations of
    embedding, storing, and searching text chunks.

    Attributes:
        device (str): The device used for computation (e.g., 'cuda', 'mps', 'cpu').
        current_model_name (Optional[str]): The HuggingFace ID of the currently loaded model.
        model (Optional[SentenceTransformer]): The active sentence transformer instance.
        client (chromadb.PersistentClient): The persistent ChromaDB client.
        model_map (Dict[str, str]): A mapping of friendly model names to HuggingFace IDs.
        collection (chromadb.Collection): The active ChromaDB collection.
    """

    def __init__(
        self, cfg: Dict[str, Any], collection_name: Optional[str] = None
    ) -> None:
        """Initializes the Engine with a Persistent Vector DB.

        Args:
            cfg (Dict[str, Any]): Configuration dictionary containing vector_store settings.
            collection_name (Optional[str]): Override for the collection name defined in cfg.
        """
        self.device: str = get_device(verbose=True)
        self.current_model_name: Optional[str] = None
        self.model: Optional[SentenceTransformer] = None
        self.client: chromadb.PersistentClient = chromadb.PersistentClient(
            path=cfg["vector_store"]["path"]
        )

        collection_name = (
            cfg["vector_store"]["collection_name"]
            if collection_name is None
            else collection_name
        )
        self.model_map: Dict[str, str] = cfg["vector_store"]["embedding_model_map"]

        self.collection: chromadb.Collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        print(
            f"Vector Database loaded. Collection '{collection_name}' has {self.collection.count()} chunks."
        )

    def load_model(self, model_name: str) -> None:
        """Loads the embedding model based on a key from the model map.

        Handles memory cleanup by deleting old models and clearing the cache
        for CUDA or MPS devices before loading a new model.

        Args:
            model_name (str): The key or HuggingFace ID of the model to load.
        """
        hf_id: str = self.model_map.get(model_name, model_name)

        if self.current_model_name == hf_id:
            return

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

    def embed_and_store(
        self, chunks_data: List[Dict[str, Any]], reset_db: bool = False
    ) -> None:
        """Generates embeddings for a list of chunks and stores them in the vector database.



        Args:
            chunks_data (List[Dict[str, Any]]): A list of dictionaries, each containing
                'text_to_embed', 'chunk_id', and 'metadata'.
            reset_db (bool): If True, deletes the existing collection and recreates it.

        Raises:
            ValueError: If no model has been loaded before calling this method.
        """
        if not self.model:
            raise ValueError("No model loaded.")

        if reset_db:
            print("Resetting Database Collection...")
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name, metadata={"hnsw:space": "cosine"}
            )

        print(f"Generating Embeddings for {len(chunks_data)} chunks...")
        texts: List[str] = [item["text_to_embed"] for item in chunks_data]
        ids: List[str] = [item["chunk_id"] for item in chunks_data]
        metadatas: List[Dict[str, Any]] = [item["metadata"] for item in chunks_data]

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

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Searches the vector database for relevant passages based on a query.



        Args:
            query (str): The search query text.
            k (int): The number of top results to return.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing 'chunk_id',
                'score' (cosine similarity), 'text', and 'metadata'.

        Raises:
            ValueError: If no model has been loaded before calling this method.
        """
        if not self.model:
            raise ValueError("No model loaded.")

        if "snowflake" in (self.current_model_name or "").lower():
            formatted_query: str = (
                f"Represent this sentence for searching relevant passages: {query}"
            )
        else:
            formatted_query = query

        query_vec: List[float] = self.model.encode(
            [formatted_query], convert_to_numpy=True
        ).tolist()

        results = self.collection.query(query_embeddings=query_vec, n_results=k)

        parsed: List[Dict[str, Any]] = []
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
