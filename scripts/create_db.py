from src.retrieval.embedding_pipeline import EmbeddingWithDB
from src.utils.helper import load_config, load_json

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    chunks = load_json(str(cfg["preprocessing"]["file_path_chunks_split"]))
    golden_set = load_json(str(cfg["test_set_path"]))

    engine = EmbeddingWithDB(cfg)
    engine.load_model(cfg["rag_pipeline"]["embedding_model_name"])
    engine.embed_and_store(chunks, reset_db=True)
