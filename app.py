"""
app.py

Streamlit frontend for the Building Byelaw RAG application.
"""

import streamlit as st
from openai import OpenAI

from config import (
    EMBEDDING_MODEL,
    INDEX_PATH,
    LLM_MODEL,
    OPENAI_API_KEY,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from rag import (
    generate_answer,
    load_index,
    retrieve,
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Building Byelaw AI Assistant",
    page_icon="🏗️",
    layout="wide",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🏗️ Building Byelaw AI Assistant")

st.markdown(
    """
Ask questions about the building byelaw document.

The assistant retrieves relevant sections from the
embedded document and generates an answer based only
on those sections.
"""
)


# ---------------------------------------------------------
# API key
# ---------------------------------------------------------

if not OPENAI_API_KEY:

    st.error(
        "OPENAI_API_KEY is not configured."
    )

    st.info(
        """
For local development, create a `.env` file:

OPENAI_API_KEY="your-api-key"

For Streamlit Community Cloud, add the key through
the application's Secrets settings.
"""
    )

    st.stop()


# ---------------------------------------------------------
# Load OpenAI client
# ---------------------------------------------------------

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ---------------------------------------------------------
# Load index
# ---------------------------------------------------------

try:

    index = load_index()

except FileNotFoundError as error:

    st.error(str(error))

    st.stop()

except Exception as error:

    st.error(
        f"Could not load the document index: {error}"
    )

    st.stop()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("Document")

    st.write(
        f"**Source:** "
        f"{index.get('source_file', 'Unknown')}"
    )

    st.write(
        f"**Chunks:** "
        f"{index.get('chunk_count', 0)}"
    )

    st.divider()

    st.header("RAG settings")

    st.write(
        f"Top K: `{TOP_K}`"
    )

    st.write(
        f"Similarity threshold: "
        f"`{SIMILARITY_THRESHOLD}`"
    )

    st.write(
        f"Embedding model: `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"LLM: `{LLM_MODEL}`"
    )

    st.divider()

    st.caption(
        "Answers are restricted to the indexed "
        "building byelaw document."
    )


# ---------------------------------------------------------
# Question input
# ---------------------------------------------------------

question = st.text_area(
    "Ask a question",
    placeholder=(
        "Example: What is the minimum required "
        "width of a residential staircase?"
    ),
    height=120,
)


# ---------------------------------------------------------
# Ask button
# ---------------------------------------------------------

ask_button = st.button(
    "🔎 Ask the Byelaw",
    type="primary",
)


if ask_button:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    with st.spinner(
        "Searching the building byelaw..."
    ):

        try:

            retrieved_chunks = retrieve(
                client=client,
                question=question,
                index=index,
                top_k=TOP_K,
            )

        except Exception as error:

            st.error(
                f"Retrieval failed: {error}"
            )

            st.stop()

    # -----------------------------------------------------
    # Display retrieval information
    # -----------------------------------------------------

    if not retrieved_chunks:

        st.warning(
            "No relevant information was found "
            "in the document."
        )

        st.stop()

    best_score = retrieved_chunks[0]["similarity"]

    # -----------------------------------------------------
    # Confidence gate
    # -----------------------------------------------------

    if best_score < SIMILARITY_THRESHOLD:

        st.warning(
            "I could not find sufficient information "
            "in the building byelaw document to answer "
            "this question."
        )

        st.caption(
            f"Best document similarity: "
            f"{best_score:.3f}"
        )

        st.stop()

    # -----------------------------------------------------
    # Generate grounded answer
    # -----------------------------------------------------

    with st.spinner(
        "Generating a document-grounded answer..."
    ):

        try:

            result = generate_answer(
                client=client,
                question=question,
                retrieved_chunks=retrieved_chunks,
            )

        except Exception as error:

            st.error(
                f"Answer generation failed: {error}"
            )

            st.stop()

    # -----------------------------------------------------
    # Answer
    # -----------------------------------------------------

    st.subheader("Answer")

    st.markdown(
        result["answer"]
    )

    # -----------------------------------------------------
    # Sources
    # -----------------------------------------------------

    if result.get("sources"):

        st.subheader("Retrieved pages")

        pages = result["sources"]

        st.write(
            ", ".join(
                f"Page {page}"
                for page in pages
            )
        )

    # -----------------------------------------------------
    # Retrieved context
    # -----------------------------------------------------

    with st.expander(
        "View retrieved document passages"
    ):

        for number, item in enumerate(
            retrieved_chunks,
            start=1,
        ):

            st.markdown(
                f"""
**Passage {number} — Page {item["page"]}**

Similarity: `{item["similarity"]:.3f}`
"""
            )

            st.write(
                item["text"]
            )

            st.divider()
