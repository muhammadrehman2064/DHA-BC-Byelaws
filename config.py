from pathlib import Path
import os

from dotenv import load_dotenv

=========================================================
LOAD ENVIRONMENT VARIABLES
=========================================================

load_dotenv()

=========================================================
PROJECT PATHS
=========================================================

BASE_DIR = Path(file).resolve().parent

DATA_DIR = BASE_DIR / "data"

INDEX_PATH = DATA_DIR / "index.json"

PDF_PATH = DATA_DIR / "building_bylaws.pdf"

=========================================================
GROQ
=========================================================

GROQ_API_KEY = os.getenv(
"GROQ_API_KEY",
""
).strip()

Groq production model.
GPT-OSS 20B is currently available on Groq and supports
reasoning controls.
You can change this from Streamlit Secrets/environment
variables without changing the code.

LLM_MODEL = os.getenv(
"LLM_MODEL",
"openai/gpt-oss-20b",
)

=========================================================
EMBEDDINGS
=========================================================
IMPORTANT:
This MUST match the embedding model used when the
index.json was created in Google Colab.

EMBEDDING_MODEL = os.getenv(
"EMBEDDING_MODEL",
"sentence-transformers/all-MiniLM-L6-v2",
)

=========================================================
RAG SETTINGS
=========================================================
Number of candidate passages retrieved from the index.

TOP_K = int(
os.getenv(
"TOP_K",
"6",
)
)

Minimum similarity required for the RAG system
to consider the retrieval useful.

SIMILARITY_THRESHOLD = float(
os.getenv(
"SIMILARITY_THRESHOLD",
"0.35",
)
)

Maximum number of previous chat messages sent to
the LLM as conversational context.

MAX_HISTORY_MESSAGES = int(
os.getenv(
"MAX_HISTORY_MESSAGES",
"8",
)
)

Maximum number of document passages actually sent
to the LLM after retrieval filtering.

MAX_CONTEXT_CHUNKS = int(
os.getenv(
"MAX_CONTEXT_CHUNKS",
"5",
)
)

Maximum output tokens for the LLM.

MAX_COMPLETION_TOKENS = int(
os.getenv(
"MAX_COMPLETION_TOKENS",
"2000",
)
)

=========================================================
OPTIONAL GENERATION SETTINGS
=========================================================

TEMPERATURE = float(
os.getenv(
"TEMPERATURE",
"0.2",
)
)

=========================================================
APPLICATION
=========================================================

APP_TITLE = os.getenv(
"APP_TITLE",
"DHA Building Byelaw AI Assistant",
)
