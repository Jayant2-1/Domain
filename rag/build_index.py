"""
Offline index builder — reads DSA documents from rag/corpus/,
chunks them, embeds with sentence-transformers, builds a FAISS index,
and persists both index + metadata to disk.

Usage:
    python -m rag.build_index
    python -m rag.build_index --corpus-dir rag/corpus --output-dir faiss_index --chunk-size 512 --overlap 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CORPUS_DIR = "rag/corpus"
DEFAULT_OUTPUT_DIR = "faiss_index"
DEFAULT_CHUNK_SIZE = 512   # tokens (approx words for simple splitter)
DEFAULT_OVERLAP = 50


# ── Document loading ────────────────────────────────────────────────────────

def load_documents(corpus_dir: str) -> List[Tuple[str, str]]:
    """
    Load all .txt and .md files from corpus_dir.

    Returns a list of (filename, full_text) tuples.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    documents: List[Tuple[str, str]] = []
    extensions = {".txt", ".md"}

    for fpath in sorted(corpus_path.rglob("*")):
        if fpath.suffix.lower() in extensions and fpath.is_file():
            text = fpath.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                documents.append((str(fpath.relative_to(corpus_path)), text))

    if not documents:
        raise ValueError(f"No .txt or .md files found in {corpus_dir}")

    logger.info("Loaded %d documents from %s", len(documents), corpus_dir)
    return documents


# ── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping word-level chunks.

    Parameters
    ----------
    text : str
        Full document text.
    chunk_size : int
        Maximum number of words per chunk.
    overlap : int
        Number of overlapping words between consecutive chunks.

    Returns
    -------
    list[str]
        Non-empty chunks.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step

    return chunks


def build_chunks(
    documents: List[Tuple[str, str]],
    chunk_size: int,
    overlap: int,
) -> Tuple[List[str], List[dict]]:
    """
    Chunk all documents and return (texts, metadata).

    metadata entries: {"source": filename, "chunk_index": int}
    """
    all_texts: List[str] = []
    all_meta: List[dict] = []

    for filename, text in documents:
        chunks = chunk_text(text, chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_meta.append({"source": filename, "chunk_index": idx})

    logger.info("Created %d chunks from %d documents", len(all_texts), len(documents))
    return all_texts, all_meta


# ── Embedding + FAISS index ─────────────────────────────────────────────────

def embed_and_build_index(
    texts: List[str],
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[faiss.Index, np.ndarray]:
    """
    Embed texts and build a FAISS inner-product index (cosine similarity
    on L2-normalised vectors).

    Returns (faiss_index, embeddings_matrix).
    """
    logger.info("Loading sentence-transformer model: %s", model_name)
    model = SentenceTransformer(model_name, device="cpu")

    logger.info("Encoding %d chunks …", len(texts))
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,      # L2-normalise → IP == cosine
    )
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)      # inner product on normalised vectors = cosine
    index.add(embeddings)

    logger.info(
        "FAISS index built: %d vectors, dimension %d", index.ntotal, dim,
    )
    return index, embeddings


# ── Persistence ─────────────────────────────────────────────────────────────

def save_index(
    index: faiss.Index,
    metadata: List[dict],
    texts: List[str],
    output_dir: str,
) -> None:
    """
    Write FAISS index, metadata JSON, and chunk texts to output_dir.

    Files created:
        output_dir/index.faiss
        output_dir/metadata.json   (list of {source, chunk_index, text})
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_path = str(out / "index.faiss")
    meta_path = str(out / "metadata.json")

    faiss.write_index(index, index_path)
    logger.info("FAISS index saved to %s", index_path)

    # Merge text into metadata so retriever can return content
    enriched = []
    for meta, text in zip(metadata, texts):
        enriched.append({**meta, "text": text})

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    logger.info("Metadata saved to %s (%d entries)", meta_path, len(enriched))


# ── Main ────────────────────────────────────────────────────────────────────

def build(
    corpus_dir: str = DEFAULT_CORPUS_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    model_name: str = DEFAULT_MODEL_NAME,
) -> None:
    """End-to-end: load → chunk → embed → save."""
    documents = load_documents(corpus_dir)
    texts, metadata = build_chunks(documents, chunk_size, overlap)
    index, _ = embed_and_build_index(texts, model_name)
    save_index(index, metadata, texts, output_dir)
    logger.info("Index build complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from DSA corpus")
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    build(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
