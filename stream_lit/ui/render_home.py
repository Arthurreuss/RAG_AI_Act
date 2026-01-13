import streamlit as st

from stream_lit.utils import set_background


def render_home() -> None:
    """Renders the landing page of the EU AI Act Legal Assistant application.

    This function sets a custom background image, displays the application
    capabilities, and provides a navigation button to enter the chat interface.
    It manages the transition between the home page and the chat page using
    Streamlit's session state.

    Returns:
        None: The function updates the Streamlit UI and may trigger a rerun.
    """
    try:
        set_background("stream_lit/img/EU-AI-ACT.webp")
    except FileNotFoundError:
        pass

    st.title("🇪🇺 EU AI Act Legal Assistant")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
            ### Welcome!
            This AI assistant is specialized in the **EU AI Act**. 
            
            **Capabilities:**
            * 🔍 **Retrieve:** Searches precise Articles, Recitals, and Annexes.
            * 🧠 **Reason:** Answers complex compliance questions.
            * 📚 **Cite:** Provides strict citations [Source: Article X].
            
            *Powered by Llama 3 & Snowflake Embeddings.*
            """
        )

        st.write("")

        if st.button("Start Chatting", type="primary"):
            st.session_state.page = "chat"
            st.rerun()

    with col2:
        st.empty()


if __name__ == "__main__":
    render_home()
