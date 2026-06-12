"""
ingest.py — CLI tool to bulk-ingest text/markdown files into the RAG knowledge base.

Usage:
    python ingest.py
    python ingest.py --dir path/to/docs --chunk-size 600 --overlap 80
    python ingest.py --dir path/to/docs --force   # re-ingest even already-seen files
"""

# !! Must be set before ANY chromadb import to prevent slow telemetry network calls
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import sys
import time
import logging
import argparse
import uuid

from config import (
    DATASET_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    SUPPORTED_EXTENSIONS,
    LOG_LEVEL,
    CHUNK_MODE,
)
from rag_engine import (
    chunk_text,
    clean_text,
    get_embedding_model,
    get_chroma_collection,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    stream=sys.stdout,
)
logger = logging.getLogger("ingest")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_indexed_sources(collection) -> set[str]:
    """Return the set of source filenames already in the collection."""
    try:
        count = collection.count()
        if count == 0:
            return set()
        data = collection.get(include=["metadatas"])
        return {
            meta["source"]
            for meta in (data.get("metadatas") or [])
            if meta and "source" in meta
        }
    except Exception as exc:
        logger.warning("Could not fetch existing sources: %s", exc)
        return set()


def _ensure_sample_dataset(directory: str) -> None:
    """Create a sample dataset file if the directory is empty."""
    sample_path = os.path.join(directory, "sample_dataset.txt")
    logger.info("Creating sample dataset at '%s'...", sample_path)
    sample_content = (
        "Artificial Intelligence (AI) is a field of computer science focused on "
        "creating systems capable of performing tasks that normally require human intelligence.\n\n"
        "Machine Learning is a subset of AI that enables systems to learn from data "
        "without being explicitly programmed.\n\n"
        "Deep Learning is a subset of Machine Learning that uses neural networks with many layers "
        "to learn hierarchical representations of data.\n\n"
        "Retrieval-Augmented Generation (RAG) combines information retrieval with language generation "
        "to improve the factual accuracy of AI-generated responses. It retrieves relevant documents "
        "from a knowledge base and passes them as context to the language model.\n\n"
        "ChromaDB is an open-source, AI-native vector database optimised for storing and querying "
        "embedding vectors used in semantic search and RAG pipelines.\n\n"
        "Sentence Transformers is a Python library that provides pre-trained models for generating "
        "dense sentence and document embeddings suitable for semantic similarity tasks.\n"
    )
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample_content)


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion logic
# ─────────────────────────────────────────────────────────────────────────────

def ingest_directory(
    directory_path: str = DATASET_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    force: bool = False,
    apply_cleaning: bool = True,
    mode: str = CHUNK_MODE,
) -> dict:
    """
    Ingest all supported files in *directory_path* into ChromaDB.

    Args:
        directory_path: Path to the directory containing documents.
        chunk_size: Word count per chunk (token mode) or char count (char mode).
        overlap: Overlap between consecutive chunks (same unit as chunk_size).
        force: If True, re-ingest files already present in the DB.
        apply_cleaning: If True (default), run clean_text() to strip HTML,
                        MediaWiki markup, and invalid characters before chunking.

    Returns:
        A summary dict with keys:
          - files_found (int)
          - files_indexed (int)
          - files_skipped (int)
          - files_failed (int)
          - total_chunks (int)
          - errors (list[str])
    """
    summary = {
        "files_found": 0,
        "files_indexed": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "total_chunks": 0,
        "errors": [],
    }

    # ── Ensure directory exists ───────────────────────────────────────────────
    if not os.path.exists(directory_path):
        logger.warning("Directory '%s' not found. Creating it...", directory_path)
        os.makedirs(directory_path, exist_ok=True)
        _ensure_sample_dataset(directory_path)

    # ── Discover supported files ──────────────────────────────────────────────
    all_files = [
        f for f in os.listdir(directory_path)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not all_files:
        logger.warning("No supported files (%s) found in '%s'.", SUPPORTED_EXTENSIONS, directory_path)
        return summary

    summary["files_found"] = len(all_files)
    logger.info("Found %d file(s) in '%s'.", len(all_files), directory_path)

    # ── Load model & collection ───────────────────────────────────────────────
    logger.info("Loading embedding model...")
    model = get_embedding_model()
    collection = get_chroma_collection()

    already_indexed = _get_indexed_sources(collection) if not force else set()
    if already_indexed:
        logger.info(
            "%d file(s) already indexed (use --force to re-ingest): %s",
            len(already_indexed), already_indexed
        )

    # ── Process each file ─────────────────────────────────────────────────────
    for filename in all_files:
        file_path = os.path.join(directory_path, filename)

        if filename in already_indexed:
            logger.info("Skipping '%s' (already in DB). Use --force to re-index.", filename)
            summary["files_skipped"] += 1
            continue

        logger.info("Processing '%s'...", filename)
        t_start = time.perf_counter()

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            if not text.strip():
                logger.warning("'%s' is empty — skipping.", filename)
                summary["files_skipped"] += 1
                continue

            if apply_cleaning:
                text = clean_text(text)
                if not text.strip():
                    logger.warning("'%s' produced empty text after cleaning — skipping.", filename)
                    summary["files_skipped"] += 1
                    continue

            chunks = chunk_text(text, chunk_size, overlap, mode)
            if not chunks:
                logger.warning("'%s' produced no chunks — skipping.", filename)
                summary["files_skipped"] += 1
                continue

            logger.info("  Generated %d chunk(s). Encoding...", len(chunks))
            embeddings = model.encode(chunks).tolist()

            # Use timestamp + UUID suffix to prevent ID collisions on re-ingestion
            ts = int(time.time() * 1000)
            uid = uuid.uuid4().hex[:8]
            ids = [f"{filename}__{i}__{ts}_{uid}" for i in range(len(chunks))]
            metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            elapsed = time.perf_counter() - t_start
            logger.info(
                "  [OK] Indexed %d chunk(s) from '%s' in %.2fs.",
                len(chunks), filename, elapsed
            )
            summary["files_indexed"] += 1
            summary["total_chunks"] += len(chunks)

        except Exception as exc:
            err_msg = f"Error processing '{filename}': {exc}"
            logger.error(err_msg)
            summary["files_failed"] += 1
            summary["errors"].append(err_msg)

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=======================================")
    logger.info("  Ingestion Complete")
    logger.info("  Files found    : %d", summary["files_found"])
    logger.info("  Files indexed  : %d", summary["files_indexed"])
    logger.info("  Files skipped  : %d", summary["files_skipped"])
    logger.info("  Files failed   : %d", summary["files_failed"])
    logger.info("  Total chunks   : %d", summary["total_chunks"])
    logger.info("=======================================")

    if summary["errors"]:
        logger.warning("Errors encountered:")
        for err in summary["errors"]:
            logger.warning("  - %s", err)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the AetherRAG knowledge base.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        default=DATASET_DIR,
        help="Path to the directory containing documents to ingest.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Character size of each text chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Overlap in characters between consecutive chunks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest files even if they are already in the database.",
    )
    parser.add_argument(
        "--mode",
        choices=["token", "para", "char"],
        default=CHUNK_MODE,
        help="Chunking mode: 'token' (words), 'para' (paragraph-aware), 'char' (characters).",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        dest="no_clean",
        help="Skip text cleaning (HTML stripping, markup removal). Use for pre-cleaned files.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    result = ingest_directory(
        directory_path=args.dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        force=args.force,
        apply_cleaning=not args.no_clean,
        mode=args.mode,
    )
    sys.exit(0 if result["files_failed"] == 0 else 1)
