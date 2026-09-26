"""
ingest.py

Creates the searchable vector index from the building
byelaw PDF.

Run:

    python ingest.py

Input:

    data/building_bylaws.pdf

Output:

    data/index.json
"""


import json
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INDEX_PATH,
    PDF_PATH,
)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.
    """

    if not text:
        return ""

    # Replace non-breaking spaces.
    text = text.replace(
        "\xa0",
        " ",
    )

    # Normalize Windows line endings.
    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    # Remove excessive spaces.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Remove excessive blank lines.
    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text,
    )

    return text.strip()


# =========================================================
# TEXT CHUNKING
# =========================================================

def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.
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

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end >= text_length:

            break

        start = end - overlap

    return chunks


# =========================================================
# EXTRACT PDF
# =========================================================

def extract_pdf_chunks() -> list[dict]:
    """
    Extract every PDF page and divide it into chunks.

    Each chunk keeps its PDF page number.
    """

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"""
PDF file not found:

{PDF_PATH}

Please put your building byelaw PDF here:

data/building_bylaws.pdf
"""
        )

    print()
    print("=" * 60)
    print("BUILDING BYELAW RAG - PDF INGESTION")
    print("=" * 60)
    print()

    print(
        f"Reading PDF: {PDF_PATH}"
    )

    reader = PdfReader(
        str(PDF_PATH)
    )

    print(
        f"Total PDF pages: {len(reader.pages)}"
    )

    all_chunks = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        text = page.extract_text()

        if not text:

            print(
                f"Warning: Page {page_number} "
                "has no extractable text."
            )

            continue

        text = clean_text(
            text
        )

        if not text:

            continue

        chunks = create_chunks(
            text
        )

        for chunk_number, chunk in enumerate(
            chunks,
            start=1,
        ):

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


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

def create_embeddings(
    chunks: list[dict],
) -> list[dict]:
    """
    Generate local embeddings using Sentence Transformers.
    """

    print()
    print(
        "Loading embedding model:"
    )

    print(
        EMBEDDING_MODEL
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    texts = [
        item["text"]
        for item in chunks
    ]

    print()
    print(
        f"Creating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for item, embedding in zip(
        chunks,
        embeddings,
    ):

        item["embedding"] = (
            embedding.tolist()
        )

    return chunks


# =========================================================
# SAVE INDEX
# =========================================================

def save_index(
    chunks: list[dict],
) -> None:
    """
    Save chunks and embeddings to index.json.
    """

    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index = {
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
            index,
            file,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print("INDEX CREATED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Index: {INDEX_PATH}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print()


# =========================================================
# MAIN
# =========================================================

def main():

    chunks = extract_pdf_chunks()

    if not chunks:

        raise RuntimeError(
            """
No text could be extracted from the PDF.

Possible reason:
Your PDF may be scanned/image-based and require OCR.
"""
        )

    print()
    print(
        f"Created {len(chunks)} text chunks."
    )

    chunks = create_embeddings(
        chunks
    )

    save_index(
        chunks
    )

    print(
        "Ingestion completed."
    )


if __name__ == "__main__":
    main()
