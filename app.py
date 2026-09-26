"""
app.py

Streamlit application for the DHA Building Byelaw
RAG Assistant.
"""

import streamlit as st
from groq import Groq

from config import (
    APP_TITLE,
    GROQ_API_KEY,
    LLM_MODEL,
    MAX_HISTORY_MESSAGES,
    SIMILARITY_THRESHOLD,
    TOP_K,
)

from rag import (
    ask_question,
    load_index,
)


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏗️",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    color: #777;
    font-size: 1rem;
    margin-bottom: 20px;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# GET GROQ API KEY
# =========================================================

def get_groq_api_key() -> str:
    """
    Get Groq API key.

    Priority:

    1. Streamlit Cloud Secrets
    2. Environment variable
    """

    try:

        if "GROQ_API_KEY" in st.secrets:

            return str(
                st.secrets[
                    "GROQ_API_KEY"
                ]
            ).strip()

    except Exception:

        pass

    return GROQ_API_KEY


# =========================================================
# API KEY
# =========================================================

api_key = get_groq_api_key()


if not api_key:

    st.error(
        "GROQ_API_KEY is not configured."
    )

    st.info(
        """
For Streamlit Cloud:

1. Open your app.
2. Click "Manage app".
3. Open Settings → Secrets.
4. Add:

GROQ_API_KEY = "your-groq-api-key"

5. Save.
6. Reboot the app.
"""
    )

    st.stop()


# =========================================================
# GROQ CLIENT
# =========================================================

client = Groq(
    api_key=api_key
)


# =========================================================
# LOAD DOCUMENT INDEX
# =========================================================

try:

    index = load_index()

except Exception as error:

    st.error(
        "Could not load the document index."
    )

    st.code(
        str(error)
    )

    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
<div class="main-title">
🏗️ DHA Building Byelaw AI Assistant
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="subtitle">
Ask questions about the DHA building byelaw document.
The assistant searches the indexed document first and
then uses Groq to generate a grounded answer.
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "📄 Document"
    )

    source_file = index.get(
        "source_file",
        "Unknown",
    )

    chunk_count = index.get(
        "chunk_count",
        len(
            index.get(
                "chunks",
                [],
            )
        ),
    )

    st.write(
        f"**Source:** {source_file}"
    )

    st.write(
        f"**Indexed chunks:** {chunk_count}"
    )

    st.divider()

    st.header(
        "🤖 AI Settings"
    )

    st.write(
        f"**LLM:** `{LLM_MODEL}`"
    )

    st.write(
        f"**Top K:** `{TOP_K}`"
    )

    st.write(
        "**Similarity threshold:** "
        f"`{SIMILARITY_THRESHOLD}`"
    )

    st.divider()

    show_retrieval = st.checkbox(
        "🔍 Show retrieved passages",
        value=False,
    )

    st.divider()

    st.caption(
        "The assistant is restricted to the indexed "
        "building byelaw document."
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# WELCOME MESSAGE
# =========================================================

if not st.session_state.messages:

    st.info(
        """
👋 **Welcome to the DHA Building Byelaw Assistant.**

You can ask questions such as:

- What is the minimum staircase width?
- What are the parking requirements?
- What is the required setback?
- What are the height restrictions?
- What does the document say about residential buildings?

The assistant will only answer when sufficient
information is available in the indexed document.
"""
    )


# =========================================================
# DISPLAY EXISTING CHAT
# =========================================================

for message in st.session_state.messages:

    role = message.get(
        "role"
    )

    content = message.get(
        "content",
        "",
    )

    if role not in {
        "user",
        "assistant",
    }:

        continue

    with st.chat_message(
        role
    ):

        st.markdown(
            content
        )

        # -------------------------------------------------
        # Assistant information
        # -------------------------------------------------

        if role == "assistant":

            sources = message.get(
                "sources",
                [],
            )

            if sources:

                source_text = ", ".join(
                    [
                        f"Page {page}"
                        for page in sources
                    ]
                )

                st.caption(
                    f"📚 Sources: {source_text}"
                )

            # ---------------------------------------------
            # Retrieval information
            # ---------------------------------------------

            if show_retrieval:

                similarity = message.get(
                    "best_similarity"
                )

                if similarity is not None:

                    st.caption(
                        "Best retrieval similarity: "
                        f"{similarity:.3f}"
                    )

                retrieved_chunks = message.get(
                    "retrieved_chunks",
                    [],
                )

                if retrieved_chunks:

                    with st.expander(
                        "🔍 Retrieved document passages"
                    ):

                        for number, chunk in enumerate(
                            retrieved_chunks,
                            start=1,
                        ):

                            page = chunk.get(
                                "page",
                                "Unknown",
                            )

                            similarity = chunk.get(
                                "similarity",
                                0.0,
                            )

                            text = chunk.get(
                                "text",
                                "",
                            )

                            st.markdown(
                                f"**Passage {number}**  \n"
                                f"Page: **{page}**  \n"
                                f"Similarity: **{similarity:.3f}**"
                            )

                            st.write(
                                text
                            )

                            st.divider()


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about the building byelaw..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    question = question.strip()

    if not question:

        st.stop()

    # -----------------------------------------------------
    # DISPLAY USER QUESTION
    # -----------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # -----------------------------------------------------
    # SAVE USER QUESTION
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # -----------------------------------------------------
    # PREVIOUS CONVERSATION
    # -----------------------------------------------------

    history_for_rag = (
        st.session_state.messages[:-1]
    )

    # -----------------------------------------------------
    # GENERATE ANSWER
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🔎 Searching the byelaw and generating answer..."
        ):

            try:

                result = ask_question(
                    client=client,
                    question=question,
                    index=index,
                    conversation_history=history_for_rag,
                )

            except Exception as error:

                st.error(
                    "I couldn't process your question."
                )

                st.exception(
                    error
                )

                st.stop()

        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------

        answer = result.get(
            "answer",
            "I could not generate an answer.",
        )

        st.markdown(
            answer
        )

        # -------------------------------------------------
        # SOURCES
        # -------------------------------------------------

        sources = result.get(
            "sources",
            [],
        )

        if sources:

            source_text = ", ".join(
                [
                    f"Page {page}"
                    for page in sources
                ]
            )

            st.caption(
                f"📚 Sources: {source_text}"
            )

        # -------------------------------------------------
        # RETRIEVAL DEBUGGING
        # -------------------------------------------------

        best_similarity = result.get(
            "best_similarity"
        )

        if (
            show_retrieval
            and best_similarity is not None
        ):

            st.caption(
                "Best retrieval similarity: "
                f"{best_similarity:.3f}"
            )

        retrieved_chunks = result.get(
            "retrieved_chunks",
            [],
        )

        if (
            show_retrieval
            and retrieved_chunks
        ):

            with st.expander(
                "🔍 Retrieved document passages"
            ):

                for number, chunk in enumerate(
                    retrieved_chunks,
                    start=1,
                ):

                    page = chunk.get(
                        "page",
                        "Unknown",
                    )

                    similarity = chunk.get(
                        "similarity",
                        0.0,
                    )

                    text = chunk.get(
                        "text",
                        "",
                    )

                    st.markdown(
                        f"**Passage {number}**  \n"
                        f"Page: **{page}**  \n"
                        f"Similarity: **{similarity:.3f}**"
                    )

                    st.write(
                        text
                    )

                    st.divider()

    # -----------------------------------------------------
    # SAVE ASSISTANT ANSWER
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "best_similarity": best_similarity,
            "retrieved_chunks": retrieved_chunks,
        }
    )

    # -----------------------------------------------------
    # LIMIT CHAT HISTORY
    # -----------------------------------------------------

    max_messages = max(
        2,
        MAX_HISTORY_MESSAGES * 2,
    )

    if len(
        st.session_state.messages
    ) > max_messages:

        st.session_state.messages = (
            st.session_state.messages[
                -max_messages:
            ]
        )
