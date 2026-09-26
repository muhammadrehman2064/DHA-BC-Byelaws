"""
app.py

Streamlit application for the Building Byelaw RAG
Assistant.
"""


import os

import streamlit as st
from groq import Groq

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
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
    page_title="DHA Building Byelaw AI",
    page_icon="🏗️",
    layout="wide",
)


# =========================================================
# GET GROQ API KEY
# =========================================================

def get_groq_api_key() -> str:
    """
    Get the Groq API key.

    Priority:

    1. Streamlit Cloud secrets
    2. Environment variable / .env
    """

    try:

        if "GROQ_API_KEY" in st.secrets:

            return st.secrets[
                "GROQ_API_KEY"
            ]

    except Exception:

        pass

    return GROQ_API_KEY


# =========================================================
# HEADER
# =========================================================

st.title(
    "🏗️ DHA Building Byelaw AI Assistant"
)

st.markdown(
    """
Ask questions about the building byelaw document.

The assistant searches the indexed document first and
then uses the retrieved passages to generate an answer.

If sufficient information cannot be found in the
document, the assistant will say so instead of
answering from general knowledge.
"""
)


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
2. Click Manage app.
3. Open Settings / Secrets.
4. Add:

GROQ_API_KEY = "your-groq-api-key"

Then save and reboot the app.
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
# LOAD INDEX
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
        0,
    )

    st.write(
        f"**Source:** {source_file}"
    )

    st.write(
        f"**Indexed chunks:** {chunk_count}"
    )

    st.divider()

    st.header(
        "⚙️ RAG Settings"
    )

    st.write(
        f"**Top K:** {TOP_K}"
    )

    st.write(
        "**Similarity threshold:** "
        f"{SIMILARITY_THRESHOLD}"
    )

    st.write(
        f"**LLM:** {LLM_MODEL}"
    )

    st.divider()

    st.caption(
        "The assistant is restricted to the indexed "
        "building byelaw document."
    )


# =========================================================
# QUESTION
# =========================================================

st.subheader(
    "Ask a question"
)

question = st.text_area(
    "Your question",
    placeholder=(
        "Example: What is the minimum required "
        "width of a staircase?"
    ),
    height=120,
    label_visibility="collapsed",
)


# =========================================================
# ASK BUTTON
# =========================================================

ask_button = st.button(
    "🔎 Ask Byelaw",
    type="primary",
    use_container_width=True,
)


if ask_button:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # -----------------------------------------------------
    # RUN RAG
    # -----------------------------------------------------

    with st.spinner(
        "Searching the building byelaw..."
    ):

        try:

            result = ask_question(
                client=client,
                question=question,
                index=index,
            )

        except Exception as error:

            st.error(
                "An error occurred while processing "
                "your question."
            )

            st.code(
                str(error)
            )

            st.stop()

    # -----------------------------------------------------
    # ANSWER
    # -----------------------------------------------------

    st.subheader(
        "Answer"
    )

    st.markdown(
        result["answer"]
    )

    # -----------------------------------------------------
    # SOURCES
    # -----------------------------------------------------

    sources = result.get(
        "sources",
        [],
    )

    if sources:

        st.subheader(
            "📚 Source Pages"
        )

        source_text = ", ".join(
            [
                f"Page {page}"
                for page in sources
            ]
        )

        st.write(
            source_text
        )

    # -----------------------------------------------------
    # RETRIEVAL SCORE
    # -----------------------------------------------------

    best_similarity = result.get(
        "best_similarity"
    )

    if best_similarity is not None:

        st.caption(
            "Best retrieval similarity: "
            f"{best_similarity:.3f}"
        )

    # -----------------------------------------------------
    # RETRIEVED PASSAGES
    # -----------------------------------------------------

    retrieved_chunks = result.get(
        "retrieved_chunks",
        [],
    )

    if retrieved_chunks:

        with st.expander(
            "🔍 View retrieved document passages"
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
                    f"### Passage {number}"
                )

                st.write(
                    f"**PDF Page:** {page}"
                )

                st.write(
                    f"**Similarity:** "
                    f"{similarity:.3f}"
                )

                st.write(
                    text
                )

                st.divider()
