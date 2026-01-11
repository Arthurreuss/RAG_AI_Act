import base64
import json
import os
from typing import Dict

import streamlit as st

from src.rag_pipeline import RAGChatbot


@st.cache_resource
def get_chatbot(cfg):
    bot = RAGChatbot(cfg=cfg, use_quantization=True)
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


def create_new_chat():
    base_name = "Chat"
    counter = 1
    while f"{base_name} {counter}" in st.session_state.all_chats:
        counter += 1
    new_id = f"{base_name} {counter}"

    st.session_state.all_chats[new_id] = []
    switch_chat(new_id)
    save_chat_history(st.session_state.all_chats)


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
    /* 1. Background Image */
    .stApp {{
        background-image: url("data:image/png;base64,{b64_data}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* 2. CUSTOM BUTTON COLOR (#FFCC00) */
    div.stButton > button {{
        background-color: #FFCC00 !important;
        color: black !important;
        border: none !important;
        
        /* Keep the width constraint from previous step */
        width: auto !important;
        padding-left: 20px;
        padding-right: 20px;
    }}
    
    /* Hover effect */
    div.stButton > button:hover {{
        background-color: #E6B800 !important; /* Slightly darker yellow */
        color: black !important;
    }}

    /* 3. Info Box Sizing */
    div[data-testid="stAlert"] {{
        width: fit-content !important;
        max-width: 100%;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
