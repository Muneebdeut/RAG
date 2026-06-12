"""
config.py — Centralized configuration for AetherRAG.

All shared constants, paths, and defaults live here.
Import from this module instead of repeating literals across files.
"""

import os
from dotenv import load_dotenv

# Load .env file (silently ignored if it doesn't exist)
load_dotenv()

# Disable ChromaDB telemetry to prevent slow network calls on startup
# This must be set BEFORE chromadb is imported anywhere.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_SERVER_NOFILE", "65535")

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_DB_PATH: str = os.path.join(os.path.dirname(__file__), "chroma_db")
CHROMA_COLLECTION_NAME: str = "rag_docs"

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# ── Chunking Defaults ─────────────────────────────────────────────────────────
# CHUNK_MODE: "token" uses word-token-aware sliding window (spec: 500 tokens, 10% overlap)
#             "char"  uses raw character counts (legacy behaviour)
#             "para"  uses paragraph-structure-aware grouping (keeps paragraphs whole)
CHUNK_MODE: str = "token"
DEFAULT_CHUNK_SIZE: int = 500   # words (token/para mode) | characters (char mode)
DEFAULT_OVERLAP: int = 50       # 10% of 500
MIN_CHUNK_SIZE: int = 50
MAX_CHUNK_SIZE: int = 4000
MIN_OVERLAP: int = 0
MAX_OVERLAP: int = 1000

# ── Retrieval Defaults ────────────────────────────────────────────────────────
DEFAULT_K: int = 3
MIN_K: int = 1
MAX_K: int = 15

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_DIR: str = os.path.join(os.path.dirname(__file__), "dataset")
DATASET_WIKIPEDIA_DIR: str = os.path.join(DATASET_DIR, "wikipedia")
SUPPORTED_EXTENSIONS: tuple = (".txt", ".md")

# ── LLM Providers ─────────────────────────────────────────────────────────────
SUPPORTED_PROVIDERS: list = ["ollama", "mock"]

# Valid Gemini model names (as of 2025-06)
VALID_GEMINI_MODELS: list = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Valid OpenAI model names
VALID_OPENAI_MODELS: list = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo",
]

# Default model names per provider
DEFAULT_GEMINI_MODEL: str = "gemini-2.0-flash"
DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL: str = "llama3"
DEFAULT_OLLAMA_URL: str = "http://localhost:11434"

# ── API Keys (from environment) ───────────────────────────────────────────────
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
