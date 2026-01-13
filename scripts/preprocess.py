from src.preprocessing.chunk_processor import ChunkProcessor
from src.preprocessing.html_parser import AIActParser
from src.utils.helper import load_config, load_json, save_json

if __name__ == "__main__":
    config = load_config("config.yaml")

    url = config["preprocessing"]["url_ai_act"]
    file_path_extracted = config["preprocessing"]["file_path_extracted"]
    file_path_chunks = config["preprocessing"]["file_path_chunks"]
    file_path_chunks_split = config["preprocessing"]["file_path_chunks_split"]

    overlap = config["preprocessing"]["chunk_overlap"]
    chunk_size = config["preprocessing"]["chunk_size"]
    split_threshold = config["preprocessing"]["split_threshold"]

    parser = AIActParser()
    chunker = ChunkProcessor(
        split_threshold=split_threshold, chunk_size=chunk_size, overlap=overlap
    )

    extracted = parser.fetch_and_parse(url)
    save_json(extracted, file_path_extracted)
    # extracted = load_json(file_path_extracted)

    structured_chunks = chunker.process_legal_json(extracted)
    save_json(structured_chunks, file_path_chunks)
    # structured_chunks = load_json(file_path_chunks)

    small_chunks = chunker.process_chunks(structured_chunks)
    save_json(small_chunks, file_path_chunks_split)
    # small_chunks = load_json(file_path_chunks_split)
