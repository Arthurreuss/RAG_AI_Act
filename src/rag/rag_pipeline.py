import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from flashrank import Ranker, RerankRequest
from llama_cpp import Llama

from src.retrieval.content_resolver import ContentResolver
from src.retrieval.embedding_pipeline import EmbeddingWithDB
from src.utils.helper import get_device, suppress_c_stderr


class RAGChatbot:
    """A retrieval-augmented generation chatbot for the EU AI Act.

    This class manages the lifecycle of a RAG pipeline, including vector database
    retrieval, reranking, conversation history management, and LLM generation.

    Attributes:
        device (str): The computing device used (e.g., 'cpu', 'cuda').
        verbose (bool): Whether to print detailed logs.
        cfg (Dict[str, Any]): Configuration dictionary for the pipeline.
        history (List[Dict[str, str]]): Conversation history stored as a list of messages.
        retriever (EmbeddingWithDB): The vector search engine.
        ranker (Ranker): The cross-encoder reranker.
        content_resolver (ContentResolver): Utility to resolve chunks to full text.
        llm (Llama): The local LLM instance.
    """

    def __init__(self, cfg: Dict[str, Any], verbose: bool = False) -> None:
        """Initializes the full RAG pipeline: Vector DB + LLM + Memory.

        Args:
            cfg (Dict[str, Any]): Configuration settings for the pipeline.
            verbose (bool): If True, enables diagnostic logging.
        """
        self.device: str = get_device(verbose=verbose)
        self.verbose: bool = verbose
        self.cfg: Dict[str, Any] = cfg
        self.history: List[Dict[str, str]] = []

        self.retriever: EmbeddingWithDB = EmbeddingWithDB(cfg)
        self.retriever.load_model(cfg["rag_pipeline"]["embedding_model_name"])

        self.ranker: Ranker = Ranker(
            model_name=cfg["rag_pipeline"].get(
                "ranker_model", "ms-marco-MiniLM-L-12-v2"
            ),
            log_level="ERROR",
        )

        self.content_resolver: ContentResolver = ContentResolver(
            full_data_path=cfg["preprocessing"]["file_path_extracted"]
        )

        model_path: str = cfg["rag_pipeline"]["llm_model"]
        if self.verbose:
            print(f"Loading GGUF Model from {model_path}...")

        n_ctx: int = cfg["rag_pipeline"].get("max_token_context", 8192)

        with suppress_c_stderr():
            self.llm: Llama = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=-1,
                verbose=False,
            )
        print("RAG Pipeline Ready.")

    def get_history(self) -> List[Dict[str, str]]:
        """Retrieves the current conversation history.

        Returns:
            List[Dict[str, str]]: A list of message dictionaries (role and content).
        """
        return self.history

    def load_history(self, history: List[Dict[str, str]]) -> None:
        """Loads an existing conversation history into the chatbot.

        Args:
            history (List[Dict[str, str]]): A list of previous message dictionaries.
        """
        self.history = history
        if self.verbose:
            print(f"Loaded history with {len(history)} messages.")

    def clear_history(self) -> None:
        """Wipes the current conversation history."""
        self.history = []
        if self.verbose:
            print("Conversation history cleared.")

    def _count_tokens(self, text: str) -> int:
        """Accurately counts tokens using the Llama model's tokenizer.

        Args:
            text (str): The input text to tokenize.

        Returns:
            int: The total number of tokens in the text.
        """
        if not text:
            return 0
        return len(self.llm.tokenize(text.encode("utf-8", errors="ignore")))

    def _clean_source_id(self, source_id: str) -> str:
        """Simplifies granular IDs to the main element for cleaner citations.

        Example: 'article_58_paragraph_2_point_d' -> 'Article 58'.

        Args:
            source_id (str): The raw chunk identifier.

        Returns:
            str: The cleaned, human-readable citation string.
        """
        clean_id: str = re.sub(
            r"(_paragraph_|_point_|_subparagraph_|_part_).*", "", source_id
        )
        clean_id = clean_id.replace("_", " ")
        clean_id = clean_id.capitalize()
        return clean_id

    def _build_system_prompt(self) -> str:
        """Constructs the static system prompt for the LLM.

        Returns:
            str: The dedented system prompt string.
        """
        return textwrap.dedent(
            """
            You are a legal expert AI assistant for the EU AI Act. 
            Your goal is to answer questions strictly based on the provided context chunks.

            ### CRITICAL GUIDELINES:
            1. **Evidence-Based:** Answer ONLY using the information from the 'Context'.
            2. **Citations:** Every factual claim must be backed by a source ID.
               - **Strict Format:** Use only the Article, Recital, or Annex number.
               - Example: "The AI Office is responsible for monitoring [Source: article_56]."
               - Do NOT include paragraph numbers in the citation tag (e.g., avoid [article_56_paragraph_1]).
            3. **Uncertainty:** If the provided context does NOT contain the answer, strictly reply: "I cannot answer this based on the provided documents."
            4. **Tone:** Professional, concise, and neutral.
            """
        ).strip()

    def _build_context_window(
        self,
        items: List[Dict[str, Any]],
        budget: int,
        use_full_doc: bool,
        full_doc: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Fits retrieval items into the context window based on a strict token budget.

        Args:
            items (List[Dict[str, Any]]): Retrieved chunks to include.
            budget (int): The maximum number of tokens allowed for context.
            use_full_doc (bool): Whether to include the full document text.
            full_doc (Optional[Dict[str, Any]]): The resolved full document data.

        Returns:
            str: A formatted string containing the context for the LLM prompt.
        """
        formatted_parts: List[str] = []
        token_count: int = 0

        if use_full_doc and full_doc:
            title: str = full_doc.get("title", "")
            number: str = full_doc.get("number", "")
            type_: str = full_doc.get("type", "Document").title()
            text: str = full_doc.get("text", "")

            clean_id: str = f"{type_}_{number}".lower().replace(" ", "_")
            entry: str = (
                f"=== {type_} {number}: {title} ===\n[Source: {clean_id}]\n{text}"
            )
            entry_tokens: int = self._count_tokens(entry)

            if (token_count + entry_tokens) <= budget:
                formatted_parts.append(entry)
                token_count += entry_tokens

        for item in items:
            raw_id: str = item.get("chunk_id", "unknown")
            clean_id_str: str = self._clean_source_id(raw_id)

            item_text: str = item.get("text_to_embed", item.get("text", "")).strip()
            entry = f"[Source: {clean_id_str}]\n{item_text}"
            entry_tokens = self._count_tokens(entry)

            if (token_count + entry_tokens) > budget:
                break
            formatted_parts.append(entry)
            token_count += entry_tokens

        return "\n\n".join(formatted_parts)

    def _rewrite_query(self, user_query: str) -> str:
        """Rewrites the user query to be standalone using conversation history.

        Args:
            user_query (str): The raw input from the user.

        Returns:
            str: The contextualized, standalone query.
        """
        if not self.history:
            return user_query

        turns: int = self.cfg["rag_pipeline"].get("history_length", 4)
        history_text: str = ""
        for turn in self.history[-turns:]:
            role: str = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"

        prompt: str = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Rewrite the last user question to be standalone based on the history. Do not answer it. Just rewrite it.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        History:
        {history_text}
        
        Last Question: {user_query}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        output: Any = self.llm(
            prompt, max_tokens=128, stop=["<|eot_id|>"], temperature=0.0
        )
        rewritten: str = output["choices"][0]["text"].strip()

        if self.verbose:
            print(f"Rewrote: '{user_query}' -> '{rewritten}'")
        return rewritten

    def _rerank_items(
        self, query: str, items: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Reranks retrieved chunks using FlashRank for higher precision.

        Args:
            query (str): The search query used for ranking.
            items (List[Dict[str, Any]]): The candidate chunks from initial retrieval.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict[str, Any]]: The top-k reranked items.
        """
        if not items:
            return []

        passages: List[Dict[str, Any]] = []
        for item in items:
            passages.append(
                {
                    "id": item.get("chunk_id", "unknown"),
                    "text": item.get("text_to_embed", item.get("text", "")),
                    "meta": item,
                }
            )

        rank_request: RerankRequest = RerankRequest(query=query, passages=passages)
        results: List[Dict[str, Any]] = self.ranker.rerank(rank_request)

        reranked_items: List[Dict[str, Any]] = [
            result["meta"] for result in results[:top_k]
        ]

        if self.verbose:
            print(
                f"Reranked {len(items)} candidates down to {len(reranked_items)} items."
            )
            for i, res in enumerate(results[:3]):
                print(f"  [{i+1}] Score: {res['score']:.4f} - {res['id']}")

        return reranked_items

    def chat(
        self, user_query: str, use_full_doc: bool = False
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Processes a user query through the full RAG cycle.

        Args:
            user_query (str): The question or statement from the user.
            use_full_doc (bool): If True, attempts to resolve and include the full
                legal document associated with the top results.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: A tuple containing the assistant's
                response text and the list of retrieved/reranked context items.
        """
        search_query: str = self._rewrite_query(user_query)

        candidate_k: int = self.cfg["rag_pipeline"].get("rerank_candidate_k", 20)
        candidates: List[Dict[str, Any]] = self.retriever.search(
            search_query, k=candidate_k
        )

        final_k: int = self.cfg["rag_pipeline"].get("retrieval_top_k", 5)
        retrieved_items: List[Dict[str, Any]] = self._rerank_items(
            search_query, candidates, top_k=final_k
        )

        full_doc: Optional[Dict[str, Any]] = None
        if use_full_doc:
            chunk_ids: List[str] = [c["chunk_id"] for c in retrieved_items]
            full_doc = self.content_resolver.resolve_to_full_text(chunk_ids)

        system_prompt: str = self._build_system_prompt()
        max_response_tokens: int = self.cfg["rag_pipeline"].get(
            "generation_max_new_tokens", 512
        )

        total_ctx: int = self.llm.n_ctx()
        base_cost: int = (
            self._count_tokens(system_prompt) + self._count_tokens(user_query) + 100
        )
        available_budget: int = total_ctx - max_response_tokens - base_cost

        context_str: str = self._build_context_window(
            retrieved_items,
            budget=available_budget,
            use_full_doc=use_full_doc,
            full_doc=full_doc if use_full_doc else None,
        )
        context_cost: int = self._count_tokens(context_str)

        history_budget: int = available_budget - context_cost
        history_messages: List[Dict[str, str]] = []
        history_cost: int = 0
        history_limit_turns: int = self.cfg["rag_pipeline"].get("history_length", 6)

        for turn in reversed(self.history[-history_limit_turns:]):
            msg_content: str = f"{turn['role']}: {turn['content']}"
            msg_cost: int = self._count_tokens(msg_content) + 5

            if (history_cost + msg_cost) > history_budget:
                if self.verbose:
                    print(
                        f"Stopping history at {len(history_messages)} turns due to budget."
                    )
                break

            history_messages.insert(0, turn)
            history_cost += msg_cost

        if self.verbose:
            print(
                f"Token Stats: Context={context_cost}, History={history_cost}, "
                f"Remaining={history_budget - history_cost}/{available_budget}"
            )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history_messages)

        final_user_content: str = (
            f"### CONTEXT:\n{context_str}\n\n### QUESTION:\n{user_query}"
        )
        messages.append({"role": "user", "content": final_user_content})

        response: Any = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_response_tokens,
            temperature=self.cfg["rag_pipeline"].get("generation_temperature", 0.1),
            top_p=self.cfg["rag_pipeline"].get("generation_top_p", 0.9),
        )

        response_text: str = response["choices"][0]["message"]["content"]

        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": response_text})

        if use_full_doc and full_doc:
            retrieved_items.append(full_doc)

        return response_text, retrieved_items
