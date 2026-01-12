import base64
import json
import os
import re
from typing import Dict

import streamlit as st

from src.rag.rag_pipeline import RAGChatbot


@st.cache_resource
def get_chatbot(cfg):
    bot = RAGChatbot(cfg=cfg)
    return bot


def load_chat_history(history_file):
    """Loads chat history from local JSON file."""
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_chat_history(chats: Dict, history_file: str):
    """Saves the current session state chats to JSON."""
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=4)


def switch_chat(chat_id, bot: RAGChatbot):
    st.session_state.current_chat_id = chat_id
    history = st.session_state.all_chats[chat_id]
    bot.load_history(history)


def create_new_chat(cfg, bot: RAGChatbot):
    base_name = "Chat"
    counter = 1
    while f"{base_name} {counter}" in st.session_state.all_chats:
        counter += 1
    new_id = f"{base_name} {counter}"

    st.session_state.all_chats[new_id] = []
    switch_chat(new_id, bot)
    save_chat_history(st.session_state.all_chats, cfg["streamlit"]["chat_history_file"])


def delete_chat(chat_id):
    if chat_id in st.session_state.all_chats:
        del st.session_state.all_chats[chat_id]
        save_chat_history(st.session_state.all_chats)

        if st.session_state.current_chat_id == chat_id:
            remaining = list(st.session_state.all_chats.keys())
            if remaining:
                switch_chat(remaining[-1])
            else:
                create_new_chat()


def rename_chat(old_name, new_name):
    """Renames a chat session key."""
    if not new_name:
        st.warning("Name cannot be empty.")
        return
    if new_name in st.session_state.all_chats:
        st.warning(f"'{new_name}' already exists.")
        return

    st.session_state.all_chats[new_name] = st.session_state.all_chats.pop(old_name)

    if st.session_state.current_chat_id == old_name:
        st.session_state.current_chat_id = new_name

    save_chat_history(st.session_state.all_chats)
    st.rerun()


def set_background(image_file):
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


def to_roman(n):
    """Converts an integer to a Roman numeral (covers standard Annex ranges)."""
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


def convert_citation_to_key(citation_str):
    """
    Converts citation strings into ID keys.
    Examples:
    "Article 56"  -> "art_56"
    "Recital 120" -> "rct_120"
    "Annex 4"     -> "anx_IV"
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
