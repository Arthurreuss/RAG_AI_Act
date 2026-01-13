import base64
import json
import os
import re
from typing import Any, Dict, List, Optional, Union

import streamlit as st

from src.rag.rag_pipeline import RAGChatbot


@st.cache_resource
def get_chatbot(cfg: Dict[str, Any]) -> RAGChatbot:
    """Initializes and caches the RAGChatbot instance.

    Uses Streamlit's cache_resource to ensure the heavy LLM and vector database
    models are only loaded once across session reruns.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary for the RAG pipeline.

    Returns:
        RAGChatbot: An initialized chatbot instance.
    """
    bot = RAGChatbot(cfg=cfg)
    return bot


def load_chat_history(history_file: str) -> Dict[str, List[Dict[str, str]]]:
    """Loads chat history from a local JSON file.

    Args:
        history_file (str): Path to the JSON file containing history.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary where keys are chat IDs
            and values are lists of message objects.
    """
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_chat_history(
    chats: Dict[str, List[Dict[str, str]]], history_file: str
) -> None:
    """Saves the current session state chats to a JSON file.

    Args:
        chats (Dict[str, List[Dict[str, str]]]): The dictionary of chat sessions to persist.
        history_file (str): Destination path for the JSON file.
    """
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=4)


def switch_chat(chat_id: str, bot: RAGChatbot) -> None:
    """Updates the session state to point to a different chat session.

    Args:
        chat_id (str): The identifier of the chat to switch to.
        bot (RAGChatbot): The chatbot instance to update with the new history.
    """
    st.session_state.current_chat_id = chat_id
    history = st.session_state.all_chats[chat_id]
    bot.load_history(history)


def create_new_chat(cfg: Dict[str, Any], bot: RAGChatbot) -> None:
    """Creates a new unique chat session and switches the UI to it.

    Args:
        cfg (Dict[str, Any]): Configuration containing the history file path.
        bot (RAGChatbot): The chatbot instance to reset.
    """
    base_name = "Chat"
    counter = 1
    while f"{base_name} {counter}" in st.session_state.all_chats:
        counter += 1
    new_id = f"{base_name} {counter}"

    st.session_state.all_chats[new_id] = []
    switch_chat(new_id, bot)
    save_chat_history(st.session_state.all_chats, cfg["streamlit"]["chat_history_file"])


def delete_chat(chat_id: str, history_file: str) -> None:
    """Removes a chat session from session state and updates the local storage.

    Args:
        chat_id (str): The identifier of the chat to delete.
        history_file (str): Path to the JSON history file to update.
    """
    if chat_id in st.session_state.all_chats:
        del st.session_state.all_chats[chat_id]
        save_chat_history(st.session_state.all_chats, history_file)

        if st.session_state.current_chat_id == chat_id:
            remaining = list(st.session_state.all_chats.keys())
            if remaining:
                st.session_state.current_chat_id = remaining[-1]
            else:
                st.session_state.all_chats = {"New Chat": []}
                st.session_state.current_chat_id = "New Chat"


def rename_chat(old_name: str, new_name: str, history_file: str) -> None:
    """Renames an existing chat session key.

    Args:
        old_name (str): The current name of the chat.
        new_name (str): The requested new name.
        history_file (str): Path to the JSON history file to update.
    """
    if not new_name:
        st.warning("Name cannot be empty.")
        return
    if new_name in st.session_state.all_chats:
        st.warning(f"'{new_name}' already exists.")
        return

    st.session_state.all_chats[new_name] = st.session_state.all_chats.pop(old_name)

    if st.session_state.current_chat_id == old_name:
        st.session_state.current_chat_id = new_name

    save_chat_history(st.session_state.all_chats, history_file)
    st.rerun()


def set_background(image_file: str) -> None:
    """Sets a background image for the Streamlit app using CSS injection.

    Args:
        image_file (str): Path to the image file (e.g., .webp, .png).
    """
    with open(image_file, "rb") as f:
        data = f.read()
    b64_data = base64.b64encode(data).decode()

    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64_data}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


def to_roman(n: Union[int, str]) -> str:
    """Converts an integer to a Roman numeral.

    Commonly used for Annex citations (e.g., Annex 4 -> Annex IV).

    Args:
        n (Union[int, str]): The number to convert.

    Returns:
        str: The Roman numeral representation.
    """
    try:
        n = int(n)
    except ValueError:
        return str(n)

    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syb = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ""
    i = 0
    while n > 0:
        for _ in range(n // val[i]):
            roman_num += syb[i]
            n -= val[i]
        i += 1
    return roman_num


def convert_citation_to_key(citation_str: str) -> Optional[str]:
    """Converts human-readable citation strings into normalized database keys.

    Examples:
        "Article 56"  -> "art_56"
        "Recital 120" -> "rct_120"
        "Annex 4"     -> "anx_IV"

    Args:
        citation_str (str): The citation string provided by the LLM or metadata.

    Returns:
        Optional[str]: The normalized key for URL anchoring or ID lookup.
    """
    clean_str = citation_str.lower().strip()

    match = re.search(r"([a-z]+)\s*(\d+)", clean_str)

    if not match:
        return None

    doc_type = match.group(1)
    number = int(match.group(2))

    if "art" in doc_type:
        return f"art_{number}"

    elif "rec" in doc_type:
        return f"rct_{number}"

    elif "ann" in doc_type or "anx" in doc_type:
        return f"anx_{to_roman(number)}"

    return None
