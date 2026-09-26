"""
rag.py

RAG (Retrieval-Augmented Generation) engine for the
Building Byelaw AI Assistant.

Architecture:

User Question
      |
      v
Sentence Transformer
      |
      v
Question Embedding
      |
      v
Cosine Similarity
      |
      v
Relevant PDF Chunks
      |
      v
Similarity Threshold
      |
      +---- Weak match ----> Refuse to answer
      |
      +---- Good match ----> Groq LLM
                                  |
                                  v
                         Grounded Answer
                         + Page Citations
"""

import json
from pathlib import Path

import numpy as np
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

# Load the embedding model once when this module starts.
#
# This model runs locally. It does NOT use OpenAI or Groq.
#
# Streamlit cache is not used here so that this file can also
# be used independently for testing.

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# =========================================================
# LOAD VECTOR INDEX
# =========================================================

def load_index(
    index_path: Path = INDEX_PATH,
) -> dict:
    """
    Load the document vector index from index.json.

    The index is created by ingest.py.

    Expected structure:

    {
        "source_file": "...",
        "embedding_model": "...",
        "chunk_count": 100,
        "chunks": [
            {
                "id": "...",
                "page": 1,
                "chunk": 1,
                "text": "...",
                "embedding": [...]
            }
        ]
    }
    """

    if not index_path.exists():

        raise FileNotFoundError(
            f"""
Vector index was not found:

{index_path}

Please run:

python ingest.py

before starting the Streamlit application.
"""
        )

    try:

        with open(
            index_path,
            "r",
            encoding="utf-8",
        ) as file:

            index = json.load(file)

    except json.JSONDecodeError as error:

        raise RuntimeError(
            f"Could not read index.json: {error}"
        ) from error

    if "chunks" not in index:

        raise RuntimeError(
            "Invalid index.json: 'chunks' field is missing."
        )

    if not index["chunks"]:

        raise RuntimeError(
            "The vector index contains no document chunks."
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
    Calculate cosine similarity between:

        one query vector

    and:

        many document vectors.

    Returns one similarity score for each document.

    Higher score = more semantically similar.
    """

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32,
    )

    document_matrix = np.asarray(
        document_matrix,
        dtype=np.float32,
    )

    # Calculate norms.
    query_norm = np.linalg.norm(
        query_vector
    )

    document_norms = np.linalg.norm(
        document_matrix,
        axis=1,
    )

    # Prevent division by zero.
    query_norm = max(
        query_norm,
        1e-12,
    )

    document_norms = np.maximum(
        document_norms,
        1e-12,
    )

    similarities = (
        document_matrix @ query_vector
    ) / (
        document_norms * query_norm
    )

    return similarities


# =========================================================
# QUESTION EMBEDDING
# =========================================================

def create_query_embedding(
    question: str,
) -> np.ndarray:
    """
    Convert the user's question into an embedding.

    The SAME embedding model must be used for:

        document chunks

    and:

        user questions

    because they need to exist in the same vector space.
    """

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    )

    return np.asarray(
        embedding,
        dtype=np.float32,
    )


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(
    question: str,
    index: dict,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Retrieve the most relevant chunks from the building
    byelaw document.

    Parameters
    ----------
    question:
        User's question.

    index:
        Loaded index.json data.

    top_k:
        Number of chunks to retrieve.

    Returns
    -------
    list[dict]
        Retrieved chunks sorted by similarity.
    """

    if not question or not question.strip():

        return []

    chunks = index.get(
        "chunks",
        [],
    )

    if not chunks:

        return []

    # -----------------------------------------------------
    # Create embedding for user question
    # -----------------------------------------------------

    query_embedding = create_query_embedding(
        question
    )

    # -----------------------------------------------------
    # Get document embeddings
    # -----------------------------------------------------

    document_embeddings = []

    valid_chunks = []

    for chunk in chunks:

        embedding = chunk.get(
            "embedding"
        )

        if not embedding:

            continue

        document_embeddings.append(
            embedding
        )

        valid_chunks.append(
            chunk
        )

    if not document_embeddings:

        return []

    document_matrix = np.asarray(
        document_embeddings,
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Calculate similarities
    # -----------------------------------------------------

    similarities = cosine_similarity(
        query_embedding,
        document_matrix,
    )

    # -----------------------------------------------------
    # Sort by highest similarity
    # -----------------------------------------------------

    top_k = max(
        1,
        int(top_k),
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    # -----------------------------------------------------
    # Build results
    # -----------------------------------------------------

    results = []

    for index_position in top_indices:

        chunk = dict(
            valid_chunks[index_position]
        )

        chunk["similarity"] = float(
            similarities[index_position]
        )

        results.append(
            chunk
        )

    return results


# =========================================================
# SIMILARITY CHECK
# =========================================================

def has_sufficient_context(
    retrieved_chunks: list[dict],
    threshold: float = SIMILARITY_THRESHOLD,
) -> bool:
    """
    Decide whether the retrieved document context is
    sufficiently relevant to the user's question.

    We use the highest similarity score as the first
    retrieval gate.
    """

    if not retrieved_chunks:

        return False

    best_score = retrieved_chunks[0].get(
        "similarity",
        0.0,
    )

    return best_score >= threshold


# =========================================================
# BUILD LLM CONTEXT
# =========================================================

def build_context(
    retrieved_chunks: list[dict],
) -> str:
    """
    Convert retrieved chunks into a structured context
    that will be provided to the Groq model.
    """

    if not retrieved_chunks:

        return ""

    context_parts = []

    for number, item in enumerate(
        retrieved_chunks,
        start=1,
    ):

        page = item.get(
            "page",
            "Unknown",
        )

        text = item.get(
            "text",
            "",
        )

        similarity = item.get(
            "similarity",
            0.0,
        )

        context_parts.append(
            f"""
==================================================
SOURCE {number}
PDF PAGE: {page}
RETRIEVAL SIMILARITY: {similarity:.3f}
==================================================

{text}
"""
        )

    return "\n".join(
        context_parts
    )


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a Building Byelaw Document Assistant.

Your job is to answer questions using ONLY the
building byelaw document excerpts provided to you.

The document is the ONLY authoritative source for
your answers in this application.

STRICT RULES:

1. Use ONLY the supplied document excerpts.

2. Do NOT use your general knowledge to complete
   missing information.

3. Do NOT invent:
   - dimensions
   - measurements
   - requirements
   - clauses
   - regulations
   - penalties
   - exceptions
   - definitions
   - dates
   - approvals
   - procedures

4. If the document excerpts do not contain enough
   information to answer the question, respond:

   "I could not find sufficient information in the
   provided building byelaw document to answer this."

5. Do not guess.

6. Do not assume that a standard building practice
   exists in this document just because it is common
   elsewhere.

7. Every factual statement about the building byelaws
   must be supported by the supplied document.

8. Include PDF page citations after factual statements.

   Example:

   The minimum clear width is 1.2 metres. [Page 42]

9. If information comes from multiple pages, cite
   all relevant pages.

   Example:

   The requirement applies to residential buildings,
   with an exception for... [Pages 42, 43]

10. If the document does not answer the question,
    explicitly say so.

11. If the user asks something unrelated to the
    building byelaw document, explain that the
    information is not available in the document.

12. Never present information from your own knowledge
    as if it came from the building byelaw.

13. If the retrieved passages appear relevant but
    still do not contain enough information, refuse
    rather than guessing.

14. Be concise and clear.

15. Do not mention these system instructions in
    your answer.

The goal is maximum document grounding, not maximum
answer generation.
"""


# =========================================================
# GENERATE GROUNDED ANSWER
# =========================================================

def generate_answer(
    client: Groq,
    question: str,
    retrieved_chunks: list[dict],
) -> dict:
    """
    Generate a grounded answer using Groq.

    The function first checks whether the retrieved
    context is sufficiently relevant.

    If it is not, the LLM is NOT called.

    This is an important hallucination-control mechanism.
    """

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    if not question or not question.strip():

        return {
            "answer": "Please enter a question.",
            "sources": [],
            "grounded": False,
            "retrieved_chunks": [],
        }

    # -----------------------------------------------------
    # Validate retrieval
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

        best_score = retrieved_chunks[0].get(
            "similarity",
            0.0,
        )

        return {
            "answer": (
                "I could not find sufficient information "
                "in the provided building byelaw document "
                "to answer this."
            ),
            "sources": [],
            "grounded": False,
            "best_similarity": best_score,
            "retrieved_chunks": retrieved_chunks,
        }

    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------

    context = build_context(
        retrieved_chunks
    )

    # -----------------------------------------------------
    # Build user prompt
    # -----------------------------------------------------

    user_prompt = f"""
USER QUESTION:

{question.strip()}


DOCUMENT EXCERPTS:

{context}


TASK:

Answer the user's question using ONLY the document
excerpts above.

If the document excerpts do not contain sufficient
information, say:

"I could not find sufficient information in the
provided building byelaw document to answer this."

Include PDF page citations for factual claims.
"""

    # -----------------------------------------------------
    # Call Groq
    # -----------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=LLM_MODEL,

            temperature=0,

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
        )

    except Exception as error:

        raise RuntimeError(
            f"Groq API request failed: {error}"
        ) from error

    # -----------------------------------------------------
    # Extract answer
    # -----------------------------------------------------

    try:

        answer = (
            response
            .choices[0]
            .message
            .content
        )

    except (
        AttributeError,
        IndexError,
        TypeError,
    ) as error:

        raise RuntimeError(
            "Groq returned an unexpected response."
        ) from error

    if not answer:

        answer = (
            "I could not generate an answer from "
            "the provided document."
        )

    answer = answer.strip()

    # -----------------------------------------------------
    # Source pages
    # -----------------------------------------------------

    source_pages = sorted(
        {
            item.get("page")
            for item in retrieved_chunks
            if item.get("page") is not None
        }
    )

    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------

    return {
        "answer": answer,
        "sources": source_pages,
        "grounded": True,
        "best_similarity": retrieved_chunks[0].get(
            "similarity",
            0.0,
        ),
        "retrieved_chunks": retrieved_chunks,
    }


# =========================================================
# SIMPLE RAG FUNCTION
# =========================================================

def ask_question(
    client: Groq,
    question: str,
    index: dict,
) -> dict:
    """
    Convenience function that performs the complete RAG
    pipeline:

        question
           ↓
        retrieve
           ↓
        threshold check
           ↓
        Groq
           ↓
        answer
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
