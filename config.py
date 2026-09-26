from pathlib import Path
import os

from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT VARIABLES
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
# APP
# =========================================================

APP_TITLE = os.getenv(
    "APP_TITLE",
    "DHA Building Byelaw AI Assistant",
)


# =========================================================
# GROQ
# =========================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "",
).strip()


LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-20b",
).strip()


# =========================================================
# EMBEDDING MODEL
# =========================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()


# =========================================================
# RAG SETTINGS
# =========================================================

TOP_K = int(
    os.getenv(
        "TOP_K",
        "5",
    )
)


SIMILARITY_THRESHOLD = float(
    os.getenv(
        "SIMILARITY_THRESHOLD",
        "0.35",
    )
)


MAX_CONTEXT_CHUNKS = int(
    os.getenv(
        "MAX_CONTEXT_CHUNKS",
        "5",
    )
)


# =========================================================
# TEXT CHUNKING
# =========================================================

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1200",
    )
)


CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "200",
    )
)


# =========================================================
# LLM GENERATION
# =========================================================

TEMPERATURE = float(
    os.getenv(
        "TEMPERATURE",
        "0.1",
    )
)


MAX_COMPLETION_TOKENS = int(
    os.getenv(
        "MAX_COMPLETION_TOKENS",
        "1500",
    )
)


# =========================================================
# CHAT HISTORY
# =========================================================

MAX_HISTORY_MESSAGES = int(
    os.getenv(
        "MAX_HISTORY_MESSAGES",
        "6",
    )
)
