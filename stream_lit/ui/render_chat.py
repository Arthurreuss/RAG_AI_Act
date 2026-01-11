import streamlit as st

from stream_lit.utils import (
    create_new_chat,
    delete_chat,
    rename_chat,
    save_chat_history,
    switch_chat,
)


def render_chat(bot):
    with st.sidebar:
        st.header("Chat Sessions")

        col_new, col_clear = st.columns([1, 1])
        with col_new:
            if st.button("➕ New", use_container_width=True):
                create_new_chat()
                st.rerun()
        with col_clear:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.all_chats = {"New Chat": []}
                switch_chat("New Chat", bot)
                save_chat_history(st.session_state.all_chats)
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
                    with st.expander("View Sources"):
                        for s in sources:
                            sid = s.get("chunk_id", s.get("id", "Unknown ID"))
                            st.markdown(f"**[{sid}]**")

        st.session_state.all_chats[st.session_state.current_chat_id].append(
            {"role": "assistant", "content": response_text}
        )
        save_chat_history(st.session_state.all_chats)
