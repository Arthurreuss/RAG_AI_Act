import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.helper import load_config
from stream_lit.ui.render_chat import render_chat
from stream_lit.ui.render_home import render_home
from stream_lit.utils import get_chatbot, load_chat_history

st.set_page_config(page_title="EU AI Act Assistant", page_icon="⚖️", layout="wide")

cfg = load_config("config.yaml")

if "page" not in st.session_state:
    st.session_state.page = "home"

if "all_chats" not in st.session_state:
    saved_chats = load_chat_history(cfg["streamlit"]["chat_history_file"])
    if saved_chats:
        st.session_state.all_chats = saved_chats
        st.session_state.current_chat_id = list(saved_chats.keys())[-1]
    else:
        st.session_state.all_chats = {"New Chat": []}
        st.session_state.current_chat_id = "New Chat"

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = "New Chat"

try:
    bot = get_chatbot(cfg)
except Exception as e:
    st.error(f"Failed to load RAG Pipeline. Error: {e}")
    st.stop()

if st.session_state.page == "home":
    render_home()
else:
    render_chat(bot)
