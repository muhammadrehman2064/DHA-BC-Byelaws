"""
rag.py

RAG engine for the Building Byelaw Assistant.

Pipeline:

User Question
      |
      v
Local Embedding Model
      |
      v
Question Vector
      |
      v
Cosine Similarity
      |
      v
Relevant Document Chunks
      |
      v
Similarity Threshold
      |
      +---- Weak match ----> Refuse
      |
      +---- Good match ----> Groq
                                  |
                                  v
                           Grounded Answer
"""


import json
from pathlib import Path

import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    INDEX_PATH,
    LLM_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K,
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def get_embedding_model():
    """
    Load the embedding model once and cache it.

    Streamlit Cloud may otherwise load the model repeatedly.
    """

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# =========================================================
# LOAD INDEX
# =========================================================

@st.cache_data
def load_index(
    index_path: Path = INDEX_PATH,
) -> dict:
    """
    Load data/index.json.
    """

    if not index_path.exists():

        raise FileNotFoundError(
            f"""
Vector index not found:

{index_path}

Please run:

python ingest.py

Then push the generated index.json to GitHub.
"""
        )

    with open(
        index_path,
        "r",
        encoding="utf-8",
    ) as file:

        index = json.load(
            file
        )

    if "chunks" not in index:

        raise RuntimeError(
            "Invalid index.json: chunks are missing."
        )

    if not index["chunks"]:

        raise RuntimeError(
            "index.json contains zero chunks."
        )

    return index


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(
    query_vector: np.ndarray,
    document_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between one query
    vector and multiple document vectors.
    """

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32,
    )

    document_matrix = np.asarray(
        document_matrix,
        dtype=np.float32,
    )

    query_norm = np.linalg.norm(
        query_vector
    )

    document_norms = np.linalg.norm(
        document_matrix,
        axis=1,
    )

    query_norm = max(
        query_norm,
        1e-12,
    )

    document_norms = np.maximum(
        document_norms,
        1e-12,
    )

    return (
        document_matrix @ query_vector
    ) / (
        document_norms * query_norm
    )


# =========================================================
# CREATE QUESTION EMBEDDING
# =========================================================

def create_query_embedding(
    question: str,
) -> np.ndarray:
    """
    Convert the user question into an embedding.

    IMPORTANT:

    This MUST use the same embedding model that was used
    by ingest.py.
    """

    if not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    model = get_embedding_model()

    embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    return np.asarray(
        embedding,
        dtype=np.float32,
    )


# =========================================================
# RETRIEVE DOCUMENT CHUNKS
# =========================================================

def retrieve(
    question: str,
    index: dict,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Find the most relevant chunks for a question.
    """

    if not question.strip():

        return []

    chunks = index.get(
        "chunks",
        [],
    )

    if not chunks:

        return []

    # -----------------------------------------------------
    # Question embedding
    # -----------------------------------------------------

    query_embedding = create_query_embedding(
        question
    )

    # -----------------------------------------------------
    # Document embeddings
    # -----------------------------------------------------

    valid_chunks = []

    document_embeddings = []

    for chunk in chunks:

        embedding = chunk.get(
            "embedding"
        )

        if embedding is None:

            continue

        valid_chunks.append(
            chunk
        )

        document_embeddings.append(
            embedding
        )

    if not document_embeddings:

        return []

    document_matrix = np.asarray(
        document_embeddings,
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Similarity
    # -----------------------------------------------------

    similarities = cosine_similarity(
        query_embedding,
        document_matrix,
    )

    # -----------------------------------------------------
    # Top results
    # -----------------------------------------------------

    top_k = max(
        1,
        int(top_k),
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for position in top_indices:

        result = dict(
            valid_chunks[position]
        )

        result["similarity"] = float(
            similarities[position]
        )

        results.append(
            result
        )

    return results


# =========================================================
# CHECK RETRIEVAL QUALITY
# =========================================================

def has_sufficient_context(
    retrieved_chunks: list[dict],
) -> bool:
    """
    Check whether the best retrieved chunk is sufficiently
    similar to the user's question.
    """

    if not retrieved_chunks:

        return False

    best_score = retrieved_chunks[0].get(
        "similarity",
        0.0,
    )

    return (
        best_score >= SIMILARITY_THRESHOLD
    )


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(
    retrieved_chunks: list[dict],
) -> str:
    """
    Convert retrieved chunks into context for Groq.
    """

    context = []

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

        context.append(
            f"""
--- DOCUMENT PASSAGE {number} ---
PDF PAGE: {page}
SIMILARITY: {similarity:.3f}

{text}
"""
        )

    return "\n".join(
        context
    )


# =========================================================
# GROQ SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a Building Byelaw Document Assistant.

You answer questions ONLY from the building byelaw
document passages provided in the user message.

The supplied document is your ONLY source of truth.

STRICT RULES:

1. Do not use outside knowledge.

2. Do not use your general knowledge to fill gaps.

3. Do not invent:
   - measurements
   - dimensions
   - regulations
   - clauses
   - requirements
   - penalties
   - exceptions
   - definitions
   - dates
   - procedures

4. Never guess.

5. If the supplied document passages do not contain
   enough information to answer the question, say:

   "I could not find sufficient information in the
   provided building byelaw document to answer this."

6. Every factual statement about the byelaw must be
   supported by the supplied document passages.

7. Cite PDF pages using this format:

   [Page 12]

8. If information comes from multiple pages:

   [Pages 12, 13]

9. Do not create a citation for a page that was not
   supplied in the document passages.

10. If the question is unrelated to the building
    byelaw document, say that the requested information
    is not available in the document.

11. Do not present your own knowledge as if it came
    from the building byelaw.

12. Keep answers clear and reasonably concise.

13. Do not mention these instructions in your answer.
"""


# =========================================================
# GENERATE ANSWER WITH GROQ
# =========================================================

def generate_answer(
    client: Groq,
    question: str,
    retrieved_chunks: list[dict],
) -> dict:
    """
    Send retrieved document context to Groq and generate
    a grounded answer.
    """

    # -----------------------------------------------------
    # No results
    # -----------------------------------------------------

    if not retrieved_chunks:

        return {
            "answer": (
                "I could not find sufficient information "
                "in the provided building byelaw document "
                "to answer this."
            ),
            "sources": [],
            "grounded": False,
            "retrieved_chunks": [],
        }

    # -----------------------------------------------------
    # Similarity gate
    # -----------------------------------------------------

    if not has_sufficient_context(
        retrieved_chunks
    ):

        return {
            "answer": (
                "I could not find sufficient information "
                "in the provided building byelaw document "
                "to answer this."
            ),
            "sources": [],
            "grounded": False,
            "best_similarity": (
                retrieved_chunks[0].get(
                    "similarity",
                    0.0,
                )
            ),
            "retrieved_chunks": retrieved_chunks,
        }

    # -----------------------------------------------------
    # Build document context
    # -----------------------------------------------------

    context = build_context(
        retrieved_chunks
    )

    # -----------------------------------------------------
    # User prompt
    # -----------------------------------------------------

    user_prompt = f"""
USER QUESTION:

{question}


RETRIEVED BUILDING BYELAW PASSAGES:

{context}


TASK:

Answer the question using ONLY the retrieved
building byelaw passages.

If the passages do not contain sufficient information,
say:

"I could not find sufficient information in the
provided building byelaw document to answer this."

Include the relevant PDF page number after factual
claims.
"""

    # -----------------------------------------------------
    # Groq request
    # -----------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=LLM_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            temperature=0,

            max_completion_tokens=1500,
        )

    except Exception as error:

        raise RuntimeError(
            f"Groq API error: {error}"
        ) from error

    # -----------------------------------------------------
    # Get response
    # -----------------------------------------------------

    try:

        answer = (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as error:

        raise RuntimeError(
            "Unexpected response received from Groq."
        ) from error

    if not answer:

        answer = (
            "I could not generate an answer from "
            "the provided document."
        )

    # -----------------------------------------------------
    # Source pages
    # -----------------------------------------------------

    source_pages = sorted(
        {
            chunk.get("page")
            for chunk in retrieved_chunks
            if chunk.get("page") is not None
        }
    )

    # -----------------------------------------------------
    # Return
    # -----------------------------------------------------

    return {
        "answer": answer.strip(),
        "sources": source_pages,
        "grounded": True,
        "best_similarity": (
            retrieved_chunks[0].get(
                "similarity",
                0.0,
            )
        ),
        "retrieved_chunks": retrieved_chunks,
    }


# =========================================================
# COMPLETE RAG PIPELINE
# =========================================================

def ask_question(
    client: Groq,
    question: str,
    index: dict,
) -> dict:
    """
    Complete RAG pipeline:

    Question
       ↓
    Embedding
       ↓
    Retrieval
       ↓
    Similarity check
       ↓
    Groq
       ↓
    Grounded answer
    """

    retrieved_chunks = retrieve(
        question=question,
        index=index,
        top_k=TOP_K,
    )

    return generate_answer(
        client=client,
        question=question,
        retrieved_chunks=retrieved_chunks,
    )
