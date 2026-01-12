import textwrap
import warnings
from typing import Any, Dict, List

from llama_cpp import Llama

from src.retrieval.content_resolver import ContentResolver
from src.retrieval.embedding_pipeline import EmbeddingWithDB
from src.utils.helper import get_device, suppress_c_stderr


class RAGChatbot:
    def __init__(self, cfg: Dict):
        """
        Initializes the full RAG pipeline: Vector DB + LLM + Memory.
        """
        self.device = get_device(verbose=True)
        self.cfg = cfg
        self.history = []

        self.retriever = EmbeddingWithDB(cfg)
        self.retriever.load_model(cfg["rag_pipeline"]["embedding_model_name"])

        self.content_resolver = ContentResolver(
            full_data_path=cfg["preprocessing"]["file_path_extracted"]
        )

        model_path = cfg["rag_pipeline"]["llm_model"]
        print(f"Loading GGUF Model from {model_path}...")
        with suppress_c_stderr():
            self.llm = Llama(
                model_path=model_path,
                n_ctx=cfg["rag_pipeline"].get("max_token_context", 8192),
                n_gpu_layers=-1,
                verbose=False,
            )
        print("RAG Pipeline Ready.")

    def get_history(self) -> List[Dict]:
        return self.history

    def load_history(self, history: List[Dict]):
        self.history = history
        print(f"Loaded history with {len(history)} messages.")

    def clear_history(self):
        self.history = []
        print("Conversation history cleared.")

    def _count_tokens(self, text: str) -> int:
        """
        Accurately counts tokens using the Llama model's tokenizer.
        """
        if not text:
            return 0
        return len(self.llm.tokenize(text.encode("utf-8", errors="ignore")))

    def _build_system_prompt(self) -> str:
        return textwrap.dedent(
            """
            You are a legal expert AI assistant for the EU AI Act. 
            Your goal is to answer questions strictly based on the provided context chunks.

            ### CRITICAL GUIDELINES:
            1. **Evidence-Based:** Answer ONLY using the information from the 'Context'.
            2. **Citations:** Every factual claim must be backed by a source ID.
               - Format: "The AI Office is responsible for monitoring [Source: article_56]."
            3. **Uncertainty:** If the provided context does NOT contain the answer, strictly reply: "I cannot answer this based on the provided documents."
            4. **Tone:** Professional, concise, and neutral.
            """
        ).strip()

    def _build_context_window(
        self, items: List[Dict], token_budget: int, use_full_doc: bool
    ) -> str:
        """
        Fits retrieval items into the context window based on strict token budget.
        Returns a formatted string of documents/chunks.
        """
        formatted_parts = []
        current_tokens = 0

        for item in items:
            if use_full_doc:
                title = item.get("title", "")
                number = item.get("number", "")
                type_ = item.get("type", "").title()
                text = item.get("text", "")
                entry = f"=== {type_} {number}: {title} ===\n{text}"
            else:
                source_id = item.get("chunk_id", "unknown")
                text = item.get("text_to_embed", item.get("text", "")).strip()
                entry = f"[Source: {source_id}]\n{text}"

            entry_tokens = self._count_tokens(entry) + 3

            if current_tokens + entry_tokens > token_budget:
                warnings.warn(
                    f"Context budget reached ({current_tokens}/{token_budget} tokens). Stopping context addition."
                )
                break

            formatted_parts.append(entry)
            current_tokens += entry_tokens

        return "\n\n".join(formatted_parts)

    def _rewrite_query(self, user_query: str) -> str:
        """
        Uses Llama.cpp raw completion to rewrite the query.
        """
        if not self.history:
            return user_query

        turns = self.cfg["rag_pipeline"].get("history_length", 4)
        history_text = ""
        for turn in self.history[-turns:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"

        prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Rewrite the last user question to be standalone based on the history. Do not answer it. Just rewrite it.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        History:
        {history_text}
        
        Last Question: {user_query}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        output = self.llm(prompt, max_tokens=128, stop=["<|eot_id|>"], temperature=0.0)

        rewritten = output["choices"][0]["text"].strip()
        print(f"Rewrote: '{user_query}' -> '{rewritten}'")
        return rewritten

    def chat(
        self, user_query: str, k: int, use_full_doc: bool = False, verbose: bool = False
    ):
        search_query = self._rewrite_query(user_query)
        retrieved_items = self.retriever.search(search_query, k=k)

        if use_full_doc:
            chunk_ids = [c["chunk_id"] for c in retrieved_items]
            retrieved_items = self.content_resolver.resolve_to_full_text(chunk_ids)
            if verbose:
                print(f"Resolved to {len(retrieved_items)} full document(s).")

        system_prompt = self._build_system_prompt()
        max_response_tokens = self.cfg["rag_pipeline"].get(
            "generation_max_new_tokens", 512
        )
        total_ctx = self.llm.n_ctx()

        base_cost = (
            self._count_tokens(system_prompt) + self._count_tokens(user_query) + 50
        )

        remaining_budget = total_ctx - max_response_tokens - base_cost

        if remaining_budget <= 0:
            warnings.warn(
                "Query + System prompt too large for context window! Truncating..."
            )
            remaining_budget = 100

        history_messages = []
        history_cost = 0
        history_limit_turns = self.cfg["rag_pipeline"].get("history_length", 6)

        for turn in reversed(self.history[-history_limit_turns:]):
            msg_content = f"{turn['role']}: {turn['content']}"
            msg_cost = self._count_tokens(msg_content) + 5
            if (remaining_budget - history_cost - msg_cost) < 500:
                break

            history_messages.insert(0, turn)
            history_cost += msg_cost

        context_budget = remaining_budget - history_cost
        context_str = self._build_context_window(
            retrieved_items, context_budget, use_full_doc
        )

        if verbose:
            print(
                f"Token Usage -- Hist: {history_cost}, Context: {self._count_tokens(context_str)}, Avail: {remaining_budget}"
            )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_messages)

        final_user_content = (
            f"### CONTEXT:\n{context_str}\n\n### QUESTION:\n{user_query}"
        )
        messages.append({"role": "user", "content": final_user_content})

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_response_tokens,
            temperature=self.cfg["rag_pipeline"].get("generation_temperature", 0.1),
            top_p=self.cfg["rag_pipeline"].get("generation_top_p", 0.9),
        )

        response_text = response["choices"][0]["message"]["content"]

        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": response_text})

        return response_text, retrieved_items
