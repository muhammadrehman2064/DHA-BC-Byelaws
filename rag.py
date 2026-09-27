"""
rag.py

RAG engine for the DHA Building Byelaw AI Assistant.

Pipeline:

User Question
      ↓
Embedding
      ↓
Semantic Retrieval
      ↓
Relevant Byelaw Context
      ↓
Groq LLM
      ↓
Natural-language grounded answer
      ↓
Page citations
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
    MAX_COMPLETION_TOKENS,
    MAX_CONTEXT_CHUNKS,
    MAX_HISTORY_MESSAGES,
    SIMILARITY_THRESHOLD,
    TEMPERATURE,
    TOP_K,
)


# =========================================================
# FALLBACK
# =========================================================

FALLBACK_ANSWER = (
    "I could not find sufficient information in the "
    "provided building byelaw document to answer this."
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def get_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# =========================================================
# LOAD INDEX
# =========================================================

@st.cache_data
def load_index(
    index_path: Path = INDEX_PATH,
):

    if not index_path.exists():

        raise FileNotFoundError(
            f"Vector index not found:\n\n"
            f"{index_path}\n\n"
            f"Please make sure data/index.json exists."
        )

    with open(
        index_path,
        "r",
        encoding="utf-8",
    ) as file:

        index = json.load(file)

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
    query_vector,
    document_matrix,
):

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
        float(query_norm),
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
# QUERY EMBEDDING
# =========================================================

def create_query_embedding(
    question,
):

    question = question.strip()

    if not question:

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
# RETRIEVAL
# =========================================================

def retrieve(
    question,
    index,
    top_k=TOP_K,
):

    question = question.strip()

    if not question:

        return []

    chunks = index.get(
        "chunks",
        [],
    )

    if not chunks:

        return []

    query_embedding = create_query_embedding(
        question
    )

    valid_chunks = []

    document_embeddings = []

    for chunk in chunks:

        embedding = chunk.get(
            "embedding"
        )

        text = chunk.get(
            "text",
            "",
        )

        if embedding is None:
            continue

        if not text.strip():
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

    similarities = cosine_similarity(
        query_embedding,
        document_matrix,
    )

    top_k = max(
        1,
        int(top_k),
    )

    top_k = min(
        top_k,
        len(valid_chunks),
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
# CONTEXT QUALITY
# =========================================================

def has_sufficient_context(
    retrieved_chunks,
):

    if not retrieved_chunks:

        return False

    best_score = float(
        retrieved_chunks[0].get(
            "similarity",
            0.0,
        )
    )

    return (
        best_score
        >= SIMILARITY_THRESHOLD
    )


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(
    retrieved_chunks,
):

    context = []

    for number, chunk in enumerate(
        retrieved_chunks[:MAX_CONTEXT_CHUNKS],
        start=1,
    ):

        page = chunk.get(
            "page",
            "Unknown",
        )

        text = chunk.get(
            "text",
            "",
        ).strip()

        context.append(
            f"""
SOURCE {number}
PDF PAGE: {page}

{text}
"""
        )

    return "\n".join(
        context
    )


# =========================================================
# CHAT HISTORY
# =========================================================

def build_history(
    conversation_history,
):

    if not conversation_history:

        return "No previous conversation."

    recent_messages = conversation_history[
        -MAX_HISTORY_MESSAGES:
    ]

    history = []

    for message in recent_messages:

        role = message.get(
            "role",
            "",
        )

        content = message.get(
            "content",
            "",
        )

        if not content:

            continue

        if role == "user":

            history.append(
                f"User: {content}"
            )

        elif role == "assistant":

            history.append(
                f"Assistant: {content}"
            )

    if not history:

        return "No previous conversation."

    return "\n".join(
        history
    )


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are an intelligent AI assistant specializing in
building byelaws.

Your task is to answer the user's question by
UNDERSTANDING and SYNTHESIZING the supplied building
byelaw passages.

You are NOT a document copier.

You must read the retrieved passages, understand them,
and then explain the relevant information naturally,
similar to how a high-quality AI assistant would answer.

=========================================================
CORE BEHAVIOUR
=========================================================

1. Use ONLY the supplied building byelaw passages
   as factual evidence.

2. Do NOT use outside knowledge for building byelaw
   requirements.

3. Do NOT simply copy and paste the retrieved text.

4. Rewrite the information into a clear, natural,
   easy-to-understand answer.

5. Combine information from multiple passages when
   necessary.

6. Remove unnecessary repetition.

7. Answer the actual question directly.

8. If the user asks "what is", explain the requirement.

9. If the user asks "why", explain the reason ONLY if
   the document provides that reason.

10. If the user asks for comparison, clearly compare
    the relevant requirements from the document.

11. If the document provides multiple conditions,
    exceptions or categories, organize them clearly.

12. Use bullet points or a small table when that makes
    the answer easier to understand.

=========================================================
NO HALLUCINATION
=========================================================

Never invent:

- dimensions
- measurements
- percentages
- setbacks
- floor areas
- heights
- parking requirements
- clauses
- regulations
- penalties
- fees
- dates
- exceptions
- definitions
- procedures

If the retrieved passages do not contain enough
information to answer the question, respond:

"I could not find sufficient information in the
provided building byelaw document to answer this."

Do NOT try to complete the answer from general knowledge.

=========================================================
CITATIONS
=========================================================

Cite the PDF page after factual statements.

Example:

The minimum required staircase width is 1.2 metres.
[Page 25]

For information supported by multiple pages:

[Pages 25, 26]

Never invent a page number.

=========================================================
STYLE
=========================================================

Answer like a helpful professional AI assistant.

Do NOT say:

"According to the retrieved passage..."

Do NOT say:

"The provided context says..."

Do NOT dump the document text.

Instead say the answer directly.

Use concise explanations.

If useful, structure the answer as:

### Requirement

...

### Conditions

- ...
- ...

### Exception

...

### Source

[Page XX]

=========================================================
FOLLOW-UP QUESTIONS
=========================================================

Use conversation history to understand references such as:

"what about commercial?"

"what about the previous requirement?"

"and for corner plots?"

However, previous assistant responses are NOT authoritative
evidence.

Only the current retrieved document passages can support
factual claims.
"""


# =========================================================
# GENERATE LLM ANSWER
# =========================================================

def generate_answer(
    client,
    question,
    retrieved_chunks,
    conversation_history=None,
):

    # -----------------------------------------------------
    # No retrieval
    # -----------------------------------------------------

    if not retrieved_chunks:

        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "grounded": False,
            "best_similarity": 0.0,
            "retrieved_chunks": [],
        }


    # -----------------------------------------------------
    # Similarity gate
    # -----------------------------------------------------

    best_similarity = float(
        retrieved_chunks[0].get(
            "similarity",
            0.0,
        )
    )

    if not has_sufficient_context(
        retrieved_chunks
    ):

        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "grounded": False,
            "best_similarity": best_similarity,
            "retrieved_chunks": retrieved_chunks,
        }


    # -----------------------------------------------------
    # Context
    # -----------------------------------------------------

    context = build_context(
        retrieved_chunks
    )


    # -----------------------------------------------------
    # History
    # -----------------------------------------------------

    history = build_history(
        conversation_history
    )


    # -----------------------------------------------------
    # LLM PROMPT
    # -----------------------------------------------------

    user_prompt = f"""
CONVERSATION HISTORY
====================

{history}


CURRENT USER QUESTION
=====================

{question}


BUILDING BYELAW SOURCES
=======================

{context}


INSTRUCTIONS
============

Answer the user's current question.

First understand the relevant information in the
sources.

Then synthesize it into a natural, helpful answer.

DO NOT copy the source passages verbatim.

DO NOT dump the source text.

DO NOT add facts that are not supported by the sources.

If several sources contain related information, combine
them into one coherent explanation.

If the answer requires information that is not present
in the sources, say:

"I could not find sufficient information in the
provided building byelaw document to answer this."

Include PDF page citations for factual claims.

Answer directly.
"""


    # -----------------------------------------------------
    # GROQ
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

            temperature=TEMPERATURE,

            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )

    except Exception as error:

        raise RuntimeError(
            f"Groq API error: {error}"
        ) from error


    # -----------------------------------------------------
    # RESPONSE
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
            "Could not read Groq response."
        ) from error


    if not answer:

        answer = FALLBACK_ANSWER


    answer = answer.strip()


    # -----------------------------------------------------
    # SOURCE PAGES
    # -----------------------------------------------------

    source_pages = sorted(
        {
            chunk.get("page")
            for chunk in retrieved_chunks
            if chunk.get("page") is not None
        }
    )


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    return {

        "answer": answer,

        "sources": source_pages,

        "grounded": True,

        "best_similarity": best_similarity,

        "retrieved_chunks": retrieved_chunks,
    }


# =========================================================
# COMPLETE RAG PIPELINE
# =========================================================

def ask_question(
    client,
    question,
    index,
    conversation_history=None,
):

    question = question.strip()

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )


    # -----------------------------------------------------
    # RETRIEVE
    # -----------------------------------------------------

    retrieved_chunks = retrieve(

        question=question,

        index=index,

        top_k=TOP_K,
    )


    # -----------------------------------------------------
    # GENERATE
    # -----------------------------------------------------

    return generate_answer(

        client=client,

        question=question,

        retrieved_chunks=retrieved_chunks,

        conversation_history=conversation_history,
    )
