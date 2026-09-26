"""
ingest.py

Run this file whenever you add or replace the source PDF.

Example:

    python ingest.py

It creates:

    data/index.json
"""

import json
import re
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INDEX_PATH,
    OPENAI_API_KEY,
    PDF_PATH,
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_configuration():
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Create a .env file and add your API key."
        )

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Place your building byelaw PDF at that location."
        )


# ---------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean PDF-extracted text while preserving useful structure.
    """

    if not text:
        return ""

    # Normalize different whitespace characters.
    text = text.replace("\xa0", " ")

    # Remove excessive spaces.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.

    This simple character-based splitter is intentionally
    beginner-friendly. Later you can replace it with a
    token-based or semantic chunker.
    """

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ---------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------

def extract_pdf_chunks(pdf_path: Path) -> list[dict]:
    """
    Extract text from every PDF page and create chunks.

    Each chunk retains its page number so that the application
    can provide citations such as [Page 23].
    """

    reader = PdfReader(str(pdf_path))

    all_chunks = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        text = clean_text(text)

        if not text:
            print(
                f"Warning: Page {page_number} contains no extractable text."
            )
            continue

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(chunks, start=1):

            all_chunks.append(
                {
                    "id": (
                        f"page-{page_number}-"
                        f"chunk-{chunk_number}"
                    ),
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk,
                }
            )

    return all_chunks


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------

def create_embeddings(
    client: OpenAI,
    chunks: list[dict],
    batch_size: int = 100,
) -> list[dict]:
    """
    Create embeddings in batches.
    """

    total = len(chunks)

    for start in range(0, total, batch_size):

        batch = chunks[start:start + batch_size]

        texts = [item["text"] for item in batch]

        print(
            f"Embedding chunks "
            f"{start + 1}-{min(start + batch_size, total)} "
            f"of {total}..."
        )

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        # The API returns embeddings in input order.
        for item, embedding_data in zip(
            batch,
            response.data,
        ):
            item["embedding"] = embedding_data.embedding

    return chunks


# ---------------------------------------------------------
# Save index
# ---------------------------------------------------------

def save_index(chunks: list[dict]) -> None:

    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_data = {
        "source_file": PDF_PATH.name,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    with open(
        INDEX_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            index_data,
            file,
            ensure_ascii=False,
        )

    print()
    print(f"Index saved to: {INDEX_PATH}")
    print(f"Total chunks: {len(chunks)}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("Building Byelaw RAG - Ingestion")
    print("=" * 60)

    validate_configuration()

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    print()
    print(f"Reading PDF: {PDF_PATH}")

    chunks = extract_pdf_chunks(PDF_PATH)

    if not chunks:
        raise RuntimeError(
            "No text could be extracted from the PDF."
        )

    print(f"Created {len(chunks)} text chunks.")

    chunks = create_embeddings(
        client=client,
        chunks=chunks,
    )

    save_index(chunks)

    print()
    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()
