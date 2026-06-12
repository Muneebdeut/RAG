"""
rag_engine.py — Core RAG pipeline for AetherRAG.

Responsibilities:
  - Embedding model management (lazy singleton)
  - ChromaDB client / collection management
  - Text chunking
  - Document retrieval
  - RAG prompt construction
  - LLM streaming (Gemini, OpenAI, Ollama, Mock)
  - Document ingestion
  - Collection management helpers
"""

import re
import time
import logging
import threading
from typing import Optional, Generator

# Must be set before chromadb is imported to suppress slow telemetry calls
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_MODE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_K,
    MIN_K,
    MAX_K,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    MAX_OVERLAP,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
)

# -- Logging -------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -- Thread-safe singletons ----------------------------------------------------
_lock = threading.RLock()
_model = None
_client = None  # chromadb.PersistentClient, typed loosely to avoid top-level import
_collection = None
_reranker = None


# -----------------------------------------------------------------------------
# Singleton accessors
# -----------------------------------------------------------------------------

def get_embedding_model():
    """Lazily load and cache the SentenceTransformer model (thread-safe)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:  # double-checked locking
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL_NAME)
                try:
                    _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                    logger.info("Embedding model loaded successfully.")
                except Exception as exc:
                    logger.error("Failed to load embedding model: %s", exc)
                    raise RuntimeError(
                        f"Could not load embedding model '{EMBEDDING_MODEL_NAME}'. "
                        f"Ensure 'sentence-transformers' is installed.\nDetail: {exc}"
                    ) from exc
    return _model


def get_reranker_model():
    """Lazily load and cache the CrossEncoder re-ranker model (thread-safe)."""
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                logger.info("Loading re-ranker model 'cross-encoder/ms-marco-MiniLM-L-6-v2'...")
                try:
                    _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                    logger.info("Re-ranker model loaded successfully.")
                except Exception as exc:
                    logger.error("Failed to load re-ranker model: %s", exc)
                    raise RuntimeError(
                        f"Could not load re-ranker model 'cross-encoder/ms-marco-MiniLM-L-6-v2'.\n"
                        f"Detail: {exc}"
                    ) from exc
    return _reranker


def get_chroma_client():
    """Lazily create and cache the ChromaDB persistent client (thread-safe)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                # Import here so ANONYMIZED_TELEMETRY env var is already set
                import chromadb
                logger.info("Connecting to ChromaDB at '%s'...", CHROMA_DB_PATH)
                try:
                    settings = chromadb.Settings(anonymized_telemetry=False)
                    _client = chromadb.PersistentClient(
                        path=CHROMA_DB_PATH,
                        settings=settings,
                    )
                    logger.info("ChromaDB client ready.")
                except Exception as exc:
                    logger.error("Failed to connect to ChromaDB: %s", exc)
                    raise RuntimeError(
                        f"Could not connect to ChromaDB at '{CHROMA_DB_PATH}'.\n"
                        f"Detail: {exc}"
                    ) from exc
    return _client


def get_chroma_collection():
    """Lazily get-or-create and cache the ChromaDB collection (thread-safe)."""
    global _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                client = get_chroma_client()
                logger.info("Opening collection '%s'...", CHROMA_COLLECTION_NAME)
                _collection = client.get_or_create_collection(
                    name=CHROMA_COLLECTION_NAME,
                )
                logger.info("Collection '%s' ready (%d docs).", CHROMA_COLLECTION_NAME, _collection.count())
    return _collection


def _reset_collection_cache():
    """Invalidate the cached collection reference (call after delete/recreate)."""
    global _collection
    with _lock:
        _collection = None


# -----------------------------------------------------------------------------
# Collection management helpers
# -----------------------------------------------------------------------------

def clear_collection() -> None:
    """Delete and recreate the RAG collection, effectively wiping all documents."""
    client = get_chroma_client()
    logger.warning("Clearing collection '%s'...", CHROMA_COLLECTION_NAME)
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        logger.info("Collection deleted.")
    except Exception as exc:
        logger.warning("Could not delete collection (may not exist): %s", exc)
    client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    _reset_collection_cache()
    logger.info("Collection '%s' recreated.", CHROMA_COLLECTION_NAME)


def get_collection_stats() -> dict:
    """
    Return a dict with:
      - count (int): total number of stored chunks
      - sources (list[str]): unique source filenames
    """
    try:
        collection = get_chroma_collection()
        count = collection.count()
        sources: list[str] = []
        if count > 0:
            data = collection.get(include=["metadatas"])
            seen: set[str] = set()
            for meta in (data.get("metadatas") or []):
                if meta and "source" in meta and meta["source"] not in seen:
                    seen.add(meta["source"])
                    sources.append(meta["source"])
        return {"count": count, "sources": sources}
    except Exception as exc:
        logger.error("Error fetching collection stats: %s", exc)
        return {"count": 0, "sources": [], "error": str(exc)}


# -----------------------------------------------------------------------------
# Text processing
# -----------------------------------------------------------------------------

# Pre-compile regex patterns for clean_text performance
_RE_HTML_TAG       = re.compile(r"<[^>]+>")                          # <tag ...>
_RE_WIKI_TEMPLATE  = re.compile(r"\{\{[^}]*\}\}")                    # {{template}}
_RE_WIKI_FILE      = re.compile(r"\[\[(?:File|Image|Media):[^\]]*\]\]", re.IGNORECASE)
_RE_WIKI_LINK      = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")   # [[link|text]] -> text
_RE_EXT_LINK       = re.compile(r"\[https?://\S+\s*([^\]]*)\]")      # [url text] -> text
_RE_HEADING        = re.compile(r"^={2,6}\s*(.*?)\s*={2,6}\s*$", re.MULTILINE)
_RE_HTML_ENTITY    = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")  # &amp; etc.
_RE_CONTROL_CHARS  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]") # control chars (keep \t\n)
_RE_MULTI_NEWLINE  = re.compile(r"\n{3,}")
_RE_MULTI_SPACE    = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """
    Remove HTML tags, MediaWiki markup, boilerplate sections, invalid characters,
    and normalise whitespace from raw document text.

    Pipeline:
      1. Strip HTML tags
      2. Remove MediaWiki templates ({{...}}) and file embeds
      3. Resolve wikilinks to their display text
      4. Remove external link markup
      5. Flatten headings to plain text
      6. Strip HTML entities
      7. Remove control characters (non-printable)
      8. Collapse redundant whitespace / blank lines

    Args:
        text: Raw input text (HTML, wikitext, or plain text).

    Returns:
        Cleaned, normalised plain-text string.
    """
    import unicodedata

    if not text:
        return ""

    # 1. Remove HTML tags
    text = _RE_HTML_TAG.sub(" ", text)

    # 2. Remove MediaWiki templates and file/image embeds
    text = _RE_WIKI_TEMPLATE.sub("", text)
    text = _RE_WIKI_FILE.sub("", text)

    # 3. Resolve wikilinks [[link|display]] -> display; [[link]] -> link
    text = _RE_WIKI_LINK.sub(r"\1", text)

    # 4. Remove external link markup [http://... label] -> label
    text = _RE_EXT_LINK.sub(r"\1", text)

    # 5. Flatten section headings (== Title ==) -> Title
    text = _RE_HEADING.sub(r"\1", text)

    # 6. Strip common HTML entities
    text = _RE_HTML_ENTITY.sub(" ", text)

    # 7. Normalise Unicode: replace surrogate / replacement characters
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    text = unicodedata.normalize("NFC", text)

    # 8. Remove non-printable control characters (keep newlines and tabs)
    text = _RE_CONTROL_CHARS.sub("", text)

    # 9. Collapse multiple blank lines and trailing whitespace per line
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    text = _RE_MULTI_SPACE.sub(" ", text)

    return text.strip()


# Lazy NLTK word tokeniser (falls back to str.split on ImportError)
_nltk_ready: bool = False
_nltk_checked: bool = False


def _get_word_tokens(text: str) -> list[str]:
    """Tokenise *text* into words using NLTK punkt tokeniser if available,
    otherwise fall back to whitespace splitting."""
    global _nltk_ready, _nltk_checked
    if not _nltk_checked:
        _nltk_checked = True
        try:
            import nltk
            # Download 'punkt_tab' resource quietly if missing
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
            _nltk_ready = True
            logger.info("NLTK punkt tokeniser available for token-aware chunking.")
        except Exception:
            logger.warning(
                "NLTK not available — falling back to whitespace tokenisation. "
                "Install with: pip install nltk"
            )
    if _nltk_ready:
        try:
            import nltk
            return nltk.word_tokenize(text)
        except Exception:
            pass
    return text.split()

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    mode: str = CHUNK_MODE,
) -> list[str]:
    """
    Split *text* into overlapping chunks using a sliding-window strategy.

    Modes:
      ``"token"`` — word-token-aware (default, per specification).
                    *chunk_size* and *overlap* are measured in **words**.
                    500-word chunks with 50-word overlap = 10% boundary context.
      ``"para"``  — paragraph-aware (keeps paragraphs whole, measures size in **words**).
      ``"char"``  — character-level (legacy). *chunk_size* and *overlap* are
                    measured in **characters**.

    Args:
        text: Raw or pre-cleaned document text.
        chunk_size: Chunk size in words (token/para mode) or chars (char mode).
        overlap: Overlap between consecutive chunks (same unit as chunk_size).
        mode: ``"token"``, ``"para"``, or ``"char"``.

    Returns:
        List of non-empty chunk strings.
    """
    if not text or not text.strip():
        logger.debug("chunk_text: received empty text, returning [].")
        return []

    # Clamp parameters to safe ranges
    chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, MAX_CHUNK_SIZE))
    overlap = max(0, min(overlap, MAX_OVERLAP, chunk_size - 1))

    if mode == "token":
        return _chunk_by_tokens(text, chunk_size, overlap)
    elif mode == "para":
        return _chunk_by_paragraphs(text, chunk_size, overlap)
    return _chunk_by_chars(text, chunk_size, overlap)


def _chunk_by_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window chunker operating on word tokens."""
    tokens = _get_word_tokens(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start += chunk_size - overlap

    logger.debug(
        "chunk_text(token): %d tokens → %d chunks (size=%d, overlap=%d).",
        n, len(chunks), chunk_size, overlap,
    )
    return chunks


def _chunk_by_paragraphs(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Structure-aware chunker operating on paragraph boundaries.
    
    Splits text by double newlines (\\n\\n), and groups consecutive paragraphs
    into chunks up to *chunk_size* word tokens. If a single paragraph exceeds
    *chunk_size*, it is split using the token sliding window.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_chunk_parts: list[str] = []
    current_size = 0

    for p in paragraphs:
        p_tokens = _get_word_tokens(p)
        p_len = len(p_tokens)

        # If a single paragraph exceeds chunk size, split it
        if p_len > chunk_size:
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
                current_chunk_parts = []
                current_size = 0
            # Split the oversized paragraph using token sliding window
            split_chunks = _chunk_by_tokens(p, chunk_size, overlap)
            chunks.extend(split_chunks)
            continue

        # If adding this paragraph exceeds chunk size, save current chunk
        if current_size + p_len > chunk_size and current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))

            # Handle overlap: keep suffix of paragraphs that sum up to <= overlap tokens
            overlap_parts = []
            overlap_size = 0
            for part in reversed(current_chunk_parts):
                part_len = len(_get_word_tokens(part))
                if overlap_size + part_len <= overlap:
                    overlap_parts.insert(0, part)
                    overlap_size += part_len
                else:
                    break
            current_chunk_parts = overlap_parts
            current_size = overlap_size

        current_chunk_parts.append(p)
        current_size += p_len

    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    logger.debug(
        "chunk_text(para): %d paragraphs → %d chunks (size=%d, overlap=%d).",
        len(paragraphs), len(chunks), chunk_size, overlap,
    )
    return chunks


def _chunk_by_chars(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window chunker operating on raw characters (legacy)."""
    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += chunk_size - overlap

    logger.debug(
        "chunk_text(char): %d chars → %d chunks (size=%d, overlap=%d).",
        text_len, len(chunks), chunk_size, overlap,
    )
    return chunks


# -----------------------------------------------------------------------------
# Query Rewriting / Condensation
# -----------------------------------------------------------------------------

def call_llm_non_streaming(
    prompt: str,
    provider: str = "mock",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    ollama_url: Optional[str] = None,
) -> str:
    """Make a non-streaming LLM call to get a complete response."""
    provider = provider.strip().lower()

    if provider == "gemini":
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key is required.")
        import google.generativeai as genai
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel(model_name or DEFAULT_GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()

    elif provider == "openai":
        if not api_key or not api_key.strip():
            raise ValueError("OpenAI API key is required.")
        from openai import OpenAI
        client = OpenAI(api_key=api_key.strip())
        response = client.chat.completions.create(
            model=model_name or DEFAULT_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    elif provider == "ollama":
        from ollama import Client as OllamaClient
        client = OllamaClient(host=(ollama_url or DEFAULT_OLLAMA_URL).strip())
        response = client.generate(model=model_name or DEFAULT_OLLAMA_MODEL, prompt=prompt)
        return response.get("response", "").strip()

    else:
        # Mock doesn't rewrite, returns input prompt context (caller handles fallback)
        return ""


def condense_query(
    query: str,
    chat_history: list[dict],
    provider: str = "mock",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    ollama_url: Optional[str] = None,
) -> str:
    """
    Rewrite the user's query using chat history to make it a standalone search query.
    If there is no history, or the provider is 'mock', returns the original query.
    """
    if not chat_history or provider == "mock":
        return query

    # Format chat history
    history_str = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        history_str += f"[{role}]: {content}\n"

    prompt = (
        "Given the conversation history and the follow-up question below, rewrite it as a standalone, self-contained search query. "
        "The rewritten query should contain all the necessary context so it can be used for document retrieval.\n"
        "If the question is already self-contained or does not relate to the history, output the original question exactly.\n"
        "Do not include any intro, explanation, or notes. Output ONLY the final rewritten question.\n\n"
        f"CHAT HISTORY:\n{history_str}\n"
        f"FOLLOW-UP QUESTION: {query}\n\n"
        "STANDALONE QUERY:"
    )

    try:
        rewritten = call_llm_non_streaming(
            prompt=prompt,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            ollama_url=ollama_url,
        )
        if rewritten:
            logger.info("Rewrote query: '%s' -> '%s'", query, rewritten)
            return rewritten
    except Exception as exc:
        logger.warning("Query rewriting failed (falling back to original): %s", exc)

    return query


# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int = DEFAULT_K,
    source_filters: Optional[list[str]] = None,
    rerank: bool = False,
) -> dict:
    """
    Embed *query* and return the top-*k* nearest chunks from ChromaDB.

    Args:
        query: User question string.
        k: Number of results to retrieve. Clamped to [MIN_K, MAX_K].
        source_filters: Optional list of source filenames to filter results.
        rerank: If True, uses CrossEncoder to re-rank chunks.

    Returns:
        ChromaDB query result dict with keys: ids, documents, metadatas, distances.

    Raises:
        ValueError: If query is empty.
        RuntimeError: If embedding or DB query fails.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    k = max(MIN_K, min(k, MAX_K))

    model = get_embedding_model()
    collection = get_chroma_collection()

    # Guard: can't retrieve more than what's stored
    stored_count = collection.count()
    if stored_count == 0:
        logger.warning("retrieve: collection is empty, returning empty results.")
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Prepare where filter for source filtering
    where_filter = None
    if source_filters:
        if len(source_filters) == 1:
            where_filter = {"source": source_filters[0]}
        else:
            where_filter = {"source": {"$in": source_filters}}

    # If filtering/re-ranking, retrieve candidate_k pool
    if rerank:
        candidate_k = min(stored_count, max(k * 3, 10))
    else:
        candidate_k = min(k, stored_count)

    try:
        query_embedding = model.encode([query.strip()]).tolist()[0]
    except Exception as exc:
        logger.error("Failed to encode query: %s", exc)
        raise RuntimeError(f"Embedding failed for query: {exc}") from exc

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        logger.debug("retrieve: got %d candidate results from ChromaDB query.", len(results["documents"][0]))
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        raise RuntimeError(f"Vector DB query failed: {exc}") from exc

    # Perform Cross-Encoder re-ranking if requested
    if rerank and len(results["documents"][0]) > 1:
        try:
            reranker = get_reranker_model()
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0]

            pairs = [[query.strip(), doc] for doc in docs]
            scores = reranker.predict(pairs)

            import math
            # Sigmoid conversion to standard [0, 1] similarity
            similarities = [1.0 / (1.0 + math.exp(-s)) for s in scores]

            # Zip and sort descending by similarity
            combined = list(zip(ids, docs, metas, similarities))
            combined.sort(key=lambda x: x[3], reverse=True)

            # Take top k
            top_k_slice = combined[:k]

            # Convert similarities back to L2 distances so that UI's _l2_to_similarity evaluates to sim
            dists = [(1.0 / item[3]) - 1.0 for item in top_k_slice]

            results = {
                "ids": [[item[0] for item in top_k_slice]],
                "documents": [[item[1] for item in top_k_slice]],
                "metadatas": [[item[2] for item in top_k_slice]],
                "distances": [dists],
            }
            logger.info("Re-ranking complete: returned top %d re-ranked results.", len(results["documents"][0]))
        except Exception as exc:
            logger.error("Re-ranking failed, falling back to database rank: %s", exc)
            # Slice results to k in case reranking failed but candidate_k was larger than k
            if len(results["documents"][0]) > k:
                results = {
                    "ids": [results["ids"][0][:k]],
                    "documents": [results["documents"][0][:k]],
                    "metadatas": [results["metadatas"][0][:k]],
                    "distances": [results["distances"][0][:k]],
                }
    elif len(results["documents"][0]) > k:
        results = {
            "ids": [results["ids"][0][:k]],
            "documents": [results["documents"][0][:k]],
            "metadatas": [results["metadatas"][0][:k]],
            "distances": [results["distances"][0][:k]],
        }

    return results


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------

def get_rag_prompt(query: str, results: dict) -> str:
    """
    Build the strict RAG prompt from retrieved context chunks.

    Args:
        query: Original user question.
        results: ChromaDB result dict from :func:`retrieve`.

    Returns:
        Formatted prompt string ready for any LLM.
    """
    chunks: list[str] = (results.get("documents") or [[]])[0]
    context = "\n---\n".join(chunk.strip() for chunk in chunks if chunk.strip())

    if not context:
        context = "(No relevant context was found in the knowledge base.)"

    prompt = (
        "You are a precise and helpful AI assistant.\n"
        "Answer the user's question using ONLY the information provided in the CONTEXT below.\n"
        "If the answer cannot be found in the context, respond with exactly:\n"
        "\"The requested information is not available in the provided dataset.\"\n"
        "Do not make up facts or use prior knowledge outside the context.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{query.strip()}\n\n"
        "ANSWER:"
    )
    return prompt


# -----------------------------------------------------------------------------
# LLM streaming
# -----------------------------------------------------------------------------

def _l2_to_similarity(distance: float) -> float:
    """
    Convert an L2 (Euclidean) distance to a [0, 1] similarity score.

    Formula: similarity = 1 / (1 + distance)
    This is monotonically decreasing in distance and bounded to (0, 1].
    """
    return 1.0 / (1.0 + max(0.0, distance))


def generate_llm_response_stream(
    prompt: str,
    provider: str = "mock",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    ollama_url: Optional[str] = None,
    query: Optional[str] = None,
    context: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Stream an LLM response for the given prompt.

    Args:
        prompt: Full RAG prompt string.
        provider: One of 'gemini', 'openai', 'ollama', 'mock'.
        api_key: API key for cloud providers (Gemini / OpenAI).
        model_name: Model identifier string.
        ollama_url: Base URL for the Ollama server.
        query: Original user query (used by mock generator).
        context: Retrieved context text (used by mock generator).

    Yields:
        String tokens/chunks of the response.

    Raises:
        ValueError: On missing required config (e.g. empty API key).
        RuntimeError: On provider call failures.
    """
    provider = (provider or "mock").strip().lower()

    if provider == "gemini":
        yield from _stream_gemini(prompt, api_key, model_name)

    elif provider == "openai":
        yield from _stream_openai(prompt, api_key, model_name)

    elif provider == "ollama":
        yield from _stream_ollama(prompt, model_name, ollama_url)

    else:
        yield from _stream_mock(query or "", context or "")


def _stream_gemini(prompt: str, api_key: Optional[str], model_name: Optional[str]) -> Generator[str, None, None]:
    """Stream a response from Google Gemini."""
    if not api_key or not api_key.strip():
        raise ValueError(
            "A valid Gemini API key is required. "
            "Set GEMINI_API_KEY in your .env file or enter it in the sidebar."
        )
    model_name = model_name or DEFAULT_GEMINI_MODEL

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "The 'google-generativeai' package is not installed. "
            "Run: pip install google-generativeai"
        ) from exc

    try:
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel(model_name)
        logger.info("Streaming Gemini response (model=%s)...", model_name)
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text
    except Exception as exc:
        error_str = str(exc)
        logger.error("Gemini streaming error: %s", error_str)
        if "API_KEY_INVALID" in error_str or "401" in error_str:
            raise RuntimeError(
                "Invalid Gemini API key. Please check your key at https://aistudio.google.com/"
            ) from exc
        if "404" in error_str or "not found" in error_str.lower():
            raise RuntimeError(
                f"Gemini model '{model_name}' was not found. "
                f"Please select a valid model from the sidebar."
            ) from exc
        if "quota" in error_str.lower() or "429" in error_str:
            raise RuntimeError(
                "Gemini API quota exceeded. Please wait a moment and try again."
            ) from exc
        raise RuntimeError(f"Gemini error: {error_str}") from exc


def _stream_openai(prompt: str, api_key: Optional[str], model_name: Optional[str]) -> Generator[str, None, None]:
    """Stream a response from OpenAI."""
    if not api_key or not api_key.strip():
        raise ValueError(
            "A valid OpenAI API key is required. "
            "Set OPENAI_API_KEY in your .env file or enter it in the sidebar."
        )
    model_name = model_name or DEFAULT_OPENAI_MODEL

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is not installed. Run: pip install openai"
        ) from exc

    try:
        client = OpenAI(api_key=api_key.strip())
        logger.info("Streaming OpenAI response (model=%s)...", model_name)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in response:
            content = (
                chunk.choices[0].delta.content
                if chunk.choices and chunk.choices[0].delta
                else None
            )
            if content:
                yield content
    except Exception as exc:
        error_str = str(exc)
        logger.error("OpenAI streaming error: %s", error_str)
        if "401" in error_str or "Incorrect API key" in error_str:
            raise RuntimeError(
                "Invalid OpenAI API key. Please check your key at https://platform.openai.com/"
            ) from exc
        if "429" in error_str or "quota" in error_str.lower():
            raise RuntimeError(
                "OpenAI rate limit or quota exceeded. Please wait and try again."
            ) from exc
        raise RuntimeError(f"OpenAI error: {error_str}") from exc


def _stream_ollama(
    prompt: str, model_name: Optional[str], ollama_url: Optional[str]
) -> Generator[str, None, None]:
    """Stream a response from a local Ollama server."""
    model_name = model_name or DEFAULT_OLLAMA_MODEL
    ollama_url = ollama_url or DEFAULT_OLLAMA_URL

    try:
        from ollama import Client as OllamaClient
    except ImportError as exc:
        raise RuntimeError(
            "The 'ollama' package is not installed. Run: pip install ollama"
        ) from exc

    try:
        client = OllamaClient(host=ollama_url.strip())
        logger.info("Streaming Ollama response (model=%s, url=%s)...", model_name, ollama_url)
        response = client.generate(model=model_name, prompt=prompt, stream=True)
        for chunk in response:
            text = chunk.get("response", "") if isinstance(chunk, dict) else getattr(chunk, "response", "")
            if text:
                yield text
    except Exception as exc:
        error_str = str(exc)
        logger.error("Ollama streaming error: %s", error_str)
        if "connection" in error_str.lower() or "refused" in error_str.lower():
            raise RuntimeError(
                f"Cannot connect to Ollama at '{ollama_url}'. "
                "Make sure Ollama is running: https://ollama.com/"
            ) from exc
        raise RuntimeError(f"Ollama error: {error_str}") from exc


def _stream_mock(query: str, context: str) -> Generator[str, None, None]:
    """
    Offline mock generator — extracts relevant sentences from context using
    keyword matching and streams them word-by-word for a realistic feel.
    """
    if not query.strip() or not context.strip():
        yield "The requested information is not available in the provided dataset."
        return

    # Extract meaningful keywords (length >= 2, skip stop words)
    _STOP_WORDS = {
        "the", "and", "but", "for", "not", "yes", "you", "your", "him", "her",
        "them", "their", "our", "its", "are", "was", "were", "been", "has", "had",
        "have", "does", "did", "done", "then", "than", "thus", "here", "there",
        "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "too", "very", "just", "into", "over",
        "under", "down", "once", "only", "that", "this", "with", "from", "they",
        "will", "what", "which", "about", "who", "whom", "whose", "which", "shall",
        "should", "would", "could", "must", "can"
    }
    keywords = [
        w.lower()
        for w in re.findall(r"\b\w+\b", query)
        if len(w) >= 2 and w.lower() not in _STOP_WORDS
    ]

    # Split context into sentences
    sentences = re.split(r"(?<=[.!?])\s+", context)

    # Score each sentence by keyword overlap
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        sentence_words = set(re.findall(r"\b\w+\b", sentence.lower()))
        score = sum(1 for kw in keywords if kw in sentence_words) / max(len(keywords), 1)
        if score > 0:
            scored.append((score, sentence.strip()))

    # Sort by relevance and take top 3
    scored.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in scored[:3]]

    if not top_sentences:
        yield "The requested information is not available in the provided dataset."
        return

    response_text = "Based on the available context: " + " ".join(top_sentences)

    # Stream word-by-word for realism
    words = response_text.split()
    for i, word in enumerate(words):
        yield word + ("" if i == len(words) - 1 else " ")
        time.sleep(0.025)


# -----------------------------------------------------------------------------
# Document ingestion
# -----------------------------------------------------------------------------

def add_document_to_rag(
    filename: str,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    mode: str = CHUNK_MODE,
) -> int:
    """
    Chunk, embed, and upsert a document into ChromaDB.

    Uses timestamp-based IDs to avoid collision on re-upload.

    Args:
        filename: Display name / source label for the document.
        text: Full document text content.
        chunk_size: Chunk size in characters.
        overlap: Overlap between chunks in characters.
        mode: Chunking mode ("token", "para", or "char").

    Returns:
        Number of chunks indexed (0 on failure).

    Raises:
        ValueError: If filename or text is empty.
        RuntimeError: On embedding or DB write failure.
    """
    if not filename or not filename.strip():
        raise ValueError("filename must not be empty.")
    if not text or not text.strip():
        raise ValueError(f"Document '{filename}' is empty — nothing to index.")

    model = get_embedding_model()
    collection = get_chroma_collection()

    chunks = chunk_text(text, chunk_size, overlap, mode)
    if not chunks:
        logger.warning("add_document_to_rag: no chunks produced for '%s'.", filename)
        return 0

    logger.info("Encoding %d chunks from '%s'...", len(chunks), filename)
    try:
        embeddings = model.encode(chunks).tolist()
    except Exception as exc:
        logger.error("Embedding failed for '%s': %s", filename, exc)
        raise RuntimeError(f"Failed to encode document '{filename}': {exc}") from exc

    # Timestamp + index ensures uniqueness even across multiple uploads of same file
    ts = int(time.time() * 1000)
    ids = [f"{filename}__{i}__{ts}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    try:
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Successfully indexed %d chunks from '%s'.", len(chunks), filename)
    except Exception as exc:
        logger.error("ChromaDB add failed for '%s': %s", filename, exc)
        raise RuntimeError(f"Failed to save chunks for '{filename}' to vector DB: {exc}") from exc

    return len(chunks)
