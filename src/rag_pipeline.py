import textwrap
import warnings
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.retrieval.content_resolver import ContentResolver
from src.retrieval.embedding_pipeline import EmbeddingWithDB
from src.utils.helper import get_device


class RAGChatbot:
    def __init__(
        self,
        cfg: Dict,
        use_quantization: bool = True,
    ):
        """
        Initializes the full RAG pipeline: Vector DB + LLM + Memory.
        """
        self.device = get_device(verbose=True)
        self.cfg = cfg

        # Limit context size to prevent crashes (approx 8k tokens)
        self.max_context_chars = cfg["rag_pipeline"].get("max_context_chars", 32000)

        # 1. Load Retriever
        self.retriever = EmbeddingWithDB(cfg)
        self.retriever.load_model(cfg["rag_pipeline"]["embedding_model_name"])

        # 2. Load Content Resolver
        self.content_resolver = ContentResolver(
            full_data_path=cfg["preprocessing"]["file_path_extracted"]
        )

        self.history = []

        # 3. Load LLM
        llm_model_name = cfg["rag_pipeline"]["llm_model"]
        print(f"Loading LLM: {llm_model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        # FIX: Explicitly set pad token for Llama 3 to avoid warnings
        if self.tokenizer.pad_token is None:
            print("Setting pad token to eos token...")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        quantization_config = None
        if use_quantization and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
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

    def _format_context(self, chunks: List[Dict]) -> str:
        formatted = []
        for c in chunks:
            source_id = c.get("chunk_id", "unknown_source")
            text = c.get("text_to_embed", c.get("text", "")).strip()
            formatted.append(f"[Source: {source_id}]\n{text}")
        return "\n\n".join(formatted)

    def _format_full_docs(self, docs: List[Dict]) -> str:
        formatted = []
        current_length = 0

        for d in docs:
            title = d.get("title", "")
            number = d.get("number", "")
            type_ = d.get("type", "").title()
            text = d.get("text", "")

            header = f"=== {type_} {number}: {title} ==="
            entry = f"{header}\n{text}"
            entry_len = len(entry)

            if current_length + entry_len > self.max_context_chars:
                remaining_space = self.max_context_chars - current_length
                truncated_entry = (
                    entry[:remaining_space] + "\n... [TRUNCATED DUE TO CONTEXT LIMIT]"
                )
                formatted.append(truncated_entry)
                warnings.warn(
                    f"Context limit reached! Truncated output at {self.max_context_chars} chars."
                )
                break

            formatted.append(entry)
            current_length += entry_len

        return "\n\n".join(formatted)

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

    def _rewrite_query(self, user_query: str) -> str:
        if not self.history:
            return user_query

        turns = self.cfg["rag_pipeline"].get("history_length", 6)
        conversation_text = ""
        for turn in self.history[-turns:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {turn['content']}\n"

        prompt = textwrap.dedent(
            f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful assistant. Rewrite the last question to be standalone based on the history. 
            Do NOT answer. JUST rewrite.
            <|eot_id|><|start_header_id|>user<|end_header_id|>

            ### HISTORY:
            {conversation_text}

            ### QUERY:
            {user_query}

            ### REWRITE:
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        ).strip()

        # Tokenizer returns dictionary with input_ids and attention_mask
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Greedy decoding for rewriting (Deterministic)
        # Note: No temperature/top_p here because do_sample=False
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        rewritten = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        print(f"Rewrote: '{user_query}' -> '{rewritten}'")
        return rewritten

    def chat(
        self,
        user_query: str,
        k: int,
        use_full_doc: bool = False,
        verbose: bool = False,
    ):
        # 1. Rewrite Query
        search_query = self._rewrite_query(user_query)

        # 2. Retrieve
        retrieved_chunks = self.retriever.search(search_query, k=k)

        # 3. Context Processing
        if use_full_doc:
            chunk_ids = [c["chunk_id"] for c in retrieved_chunks]
            full_docs = self.content_resolver.resolve_to_full_text(chunk_ids)
            if verbose:
                print(
                    f"Mapped {len(retrieved_chunks)} chunks -> {len(full_docs)} full docs."
                )
            context_str = self._format_full_docs(full_docs)
            sources_metadata = full_docs
        else:
            context_str = self._format_context(retrieved_chunks)
            sources_metadata = retrieved_chunks

        # 4. Construct Prompt
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        history_limit = self.cfg["rag_pipeline"].get("history_length", 6)
        messages.extend(self.history[-history_limit:])

        augmented_query = textwrap.dedent(
            f"""
            ### CONTEXT:
            {context_str}

            ### QUESTION:
            {user_query}
        """
        ).strip()
        messages.append({"role": "user", "content": augmented_query})

        # 5. Tokenize (with Attention Mask)
        model_inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        # 6. Generate (Sampling with Low Temperature for Facts)
        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=self.cfg["rag_pipeline"].get(
                "generation_max_new_tokens", 512
            ),
            eos_token_id=terminators,
            do_sample=True,
            # We added these back to ensure low creativity/high groundedness
            temperature=self.cfg["rag_pipeline"].get("generation_temperature", 0.1),
            top_p=self.cfg["rag_pipeline"].get("generation_top_p", 0.9),
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response_text = self.tokenizer.decode(
            outputs[0][model_inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # 7. Update History
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": response_text})

        return response_text, sources_metadata
