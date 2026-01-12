import dis

import streamlit as st

from stream_lit.utils import (
    convert_citation_to_key,
    create_new_chat,
    delete_chat,
    rename_chat,
    save_chat_history,
    switch_chat,
)


def render_chat(cfg, bot):
    with st.sidebar:
        st.header("Chat Sessions")

        col_new, col_clear = st.columns([1, 1])
        with col_new:
            if st.button("➕ New", use_container_width=True):
                create_new_chat(cfg, bot)
                st.rerun()
        with col_clear:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.all_chats = {"New Chat": []}
                switch_chat("New Chat", bot)
                save_chat_history(
                    st.session_state.all_chats, cfg["streamlit"]["chat_history_file"]
                )
                st.rerun()

        st.markdown("---")

        chat_ids = list(st.session_state.all_chats.keys())

        for c_id in reversed(chat_ids):
            col1, col2, col3 = st.columns([0.6, 0.2, 0.2])

            with col1:
                type_ = (
                    "primary"
                    if c_id == st.session_state.current_chat_id
                    else "secondary"
                )
                if st.button(
                    f"💬 {c_id}",
                    key=f"btn_{c_id}",
                    type=type_,
                    use_container_width=True,
                ):
                    switch_chat(c_id, bot)
                    st.rerun()

            with col2:
                with st.popover("✏️", help="Rename this chat"):
                    st.write(f"Rename **{c_id}**")
                    new_name_input = st.text_input(
                        "New Name", value=c_id, key=f"input_{c_id}"
                    )
                    if st.button("Save", key=f"save_{c_id}", type="primary"):
                        if new_name_input != c_id:
                            rename_chat(c_id, new_name_input)
                        else:
                            st.warning("Name hasn't changed.")

            with col3:
                if st.button("✖", key=f"del_{c_id}", help="Delete this chat"):
                    delete_chat(c_id)
                    st.rerun()

        st.markdown("---")
        st.header("Settings")
        use_full_doc = st.toggle("Use Full Document Context", value=False)
        k_retrieval = st.slider("Documents to Retrieve", 1, 10, 3)

    st.subheader(f"{st.session_state.current_chat_id}")

    current_history = st.session_state.all_chats[st.session_state.current_chat_id]

    for msg in current_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask a question about the AI Act..."):
        with st.chat_message("user"):
            st.markdown(user_query)

        st.session_state.all_chats[st.session_state.current_chat_id].append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking & Retrieving..."):
                bot.load_history(
                    st.session_state.all_chats[st.session_state.current_chat_id][:-1]
                )

                response_text, sources = bot.chat(
                    user_query, k=k_retrieval, use_full_doc=use_full_doc
                )

                st.markdown(response_text)

                if sources:
                    displayed = set()
                    with st.expander("View Sources"):
                        for s in sources:
                            sid = s.get("chunk_id", None)
                            if sid is None:
                                clean_id = s["metadata"]["citation"]
                                url_query = s.get("id", "unknown")
                            else:
                                clean_id = bot._clean_source_id(sid)
                                url_query = convert_citation_to_key(clean_id)
                            if url_query not in displayed:
                                url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#{url_query}"
                                st.markdown(f"**[{clean_id}]** - [Link]({url})")
                                displayed.add(url_query)

        st.session_state.all_chats[st.session_state.current_chat_id].append(
            {"role": "assistant", "content": response_text}
        )
        save_chat_history(
            st.session_state.all_chats, cfg["streamlit"]["chat_history_file"]
        )
