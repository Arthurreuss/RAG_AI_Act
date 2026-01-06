import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

def query_collection(collection_name: str, query_text: str, n_results: int = 3):
    client = get_chroma_client()
    ef = get_embedding_function()

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=ef
        )
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results
    except ValueError:
        print(f"Collection {collection_name} does not exist.")
        return None    

def compare_strategies(query_text: str):
    print(f"\n{"=" * 20} Query: {query_text} {"=" * 20}\n")
    strategies = ["ai_act_naive_chunks", "ai_act_chunks"]

    for strategy in strategies:
        print(f"--- Strategy: {strategy} ---")
        results = query_collection(strategy, query_text)
        
        if results:
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]

                print(f"Result {i + 1} (Distance: {dist:.4f}):")
                if "citation" in meta:
                    print(f" [Source: {meta.get('citation', "N/A")}]")
                else:
                    print(f" [Source: {meta.get('source', "N/A")}]")

                print(f" Text: {doc[:200].replace(chr(10), ' ')}...\n")
                print("-" * 40)
        print("\n")

if __name__ == "__main__":
    test_question = "What are the transparency obligations for high-risk AI systems?"

    while True:
        user_input = input("Enter a question (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        compare_strategies(user_input)