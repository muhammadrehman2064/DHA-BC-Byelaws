from pathlib import Path
import os

from dotenv import load_dotenv


# =========================================================
# LOAD LOCAL .ENV
# =========================================================

load_dotenv()


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

PDF_PATH = DATA_DIR / "building_bylaws.pdf"

INDEX_PATH = DATA_DIR / "index.json"


# =========================================================
# GROQ
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Current Groq production model suitable for this project.
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-20b",
)


# =========================================================
# EMBEDDINGS
# =========================================================

# This model runs locally.
# It does NOT require an API key.

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# =========================================================
# RAG SETTINGS
# =========================================================

# Number of document chunks retrieved.
TOP_K = int(
    os.getenv(
        "TOP_K",
        "5",
    )
)


# Minimum similarity required before sending
# retrieved context to the LLM.
SIMILARITY_THRESHOLD = float(
    os.getenv(
        "SIMILARITY_THRESHOLD",
        "0.35",
    )
)


# Approximate character size of each chunk.
CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1200",
    )
)


# Number of overlapping characters.
CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "200",
    )
)
