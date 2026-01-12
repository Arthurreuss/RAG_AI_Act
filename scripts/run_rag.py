import os
from math import e

import huggingface_hub
from dotenv import load_dotenv

from src.rag.rag_pipeline import RAGChatbot
from src.utils.helper import load_config

load_dotenv()

huggingface_hub.login(token=os.getenv("hf_dowload_api_token", ""))

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    bot = RAGChatbot(cfg)

    query1 = "What are the transparency obligations for high-risk AI systems?"
    print(f"\nUser: {query1}")

    answer1, sources1 = bot.chat(query1)

    print(f"Assistant:\n{answer1}")
    print(
        f"\n[Debug] Used {len(sources1)} sources: {[s['chunk_id'] for s in sources1]}"
    )

    query2 = "Does this apply to emotion recognition systems too?"
    print(f"\nUser: {query2}")

    answer2, sources2 = bot.chat(query2)

    print(f"Assistant:\n{answer2}")
