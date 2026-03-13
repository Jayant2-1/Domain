"""
RAG Retriever — loads a persisted FAISS index from disk, embeds queries
with the same sentence-transformer model, and returns top-k relevant
chunks via cosine similarity.

Includes a keyword-based fallback when FAISS is unavailable or returns
low-confidence results.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = "faiss_index"
DEFAULT_TOP_K = 3
DEFAULT_SCORE_THRESHOLD = 0.3


# ── Result container ────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """A single retrieved chunk with its metadata."""
    text: str
    source: str
    chunk_index: int
    score: float
    method: str  # "faiss" or "keyword"


# ── Keyword fallback ────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenization — strips punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_search(
    query: str,
    metadata: List[dict],
    top_k: int = DEFAULT_TOP_K,
) -> List[RetrievalResult]:
    """
    Simple TF-overlap keyword search as a fallback when FAISS is
    unavailable or returns low-confidence results.

    Scores each chunk by the fraction of query tokens that appear in
    the chunk (case-insensitive). This is not a substitute for
    embedding-based search but prevents returning nothing.
    """
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    scored: List[tuple] = []
    for entry in metadata:
        chunk_tokens = set(_tokenize(entry["text"]))
        overlap = len(query_tokens & chunk_tokens)
        score = overlap / len(query_tokens)
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    results: List[RetrievalResult] = []
    for score, entry in scored[:top_k]:
        if score <= 0.0:
            break
        results.append(
            RetrievalResult(
                text=entry["text"],
                source=entry.get("source", "unknown"),
                chunk_index=entry.get("chunk_index", 0),
                score=round(score, 4),
                method="keyword",
            )
        )
    return results


# ── Retriever class ─────────────────────────────────────────────────────────

class Retriever:
    """
    Embedding-based retriever backed by a persistent FAISS index.

    Lifecycle:
        1. __init__() — lightweight, does not load anything.
        2. load()     — load FAISS index + metadata from disk, load
                        the sentence-transformer model (CPU).
        3. search()   — embed query, FAISS top-k, optional keyword fallback.
    """

    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self.index_dir = index_dir
        self.model_name = model_name
        self.top_k = top_k
        self.score_threshold = score_threshold

        # Populated by load()
        self._index: Optional[faiss.Index] = None
        self._metadata: Optional[List[dict]] = None
        self._model: Optional[SentenceTransformer] = None
        self._loaded: bool = False

    # ── Loading ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load FAISS index, metadata, and embedding model from disk.

        Raises FileNotFoundError if index files are missing.
        """
        index_path = Path(self.index_dir) / "index.faiss"
        meta_path = Path(self.index_dir) / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        logger.info("Loading FAISS index from %s", index_path)
        self._index = faiss.read_index(str(index_path))
        logger.info("FAISS index loaded: %d vectors", self._index.ntotal)

        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)
        logger.info("Metadata loaded: %d entries", len(self._metadata))

        logger.info("Loading embedding model: %s", self.model_name)
        # Try local path first, then fall back to HF Hub name
        import os
        local_path = os.path.join("models", "all-MiniLM-L6-v2")
        if os.path.isdir(local_path):
            logger.info("Using local embedding model from %s", local_path)
            self._model = SentenceTransformer(local_path, device="cpu")
        else:
            self._model = SentenceTransformer(self.model_name, device="cpu")

        self._loaded = True
        logger.info("Retriever ready.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Search ──────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> List[RetrievalResult]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Flow:
            1. If FAISS is loaded → embed query, search, filter by threshold.
            2. If FAISS results are below threshold → fallback to keyword search.
            3. If FAISS is not loaded → keyword-only search.

        Parameters
        ----------
        query : str
            The user's question.
        top_k : int | None
            Override the default top_k for this search.

        Returns
        -------
        list[RetrievalResult]
        """
        k = top_k if top_k is not None else self.top_k

        # If index is not loaded, try keyword fallback over metadata
        if not self._loaded or self._index is None or self._metadata is None:
            logger.warning("FAISS index not loaded — using keyword fallback.")
            if self._metadata is not None:
                return keyword_search(query, self._metadata, k)
            return []

        # Embed query
        query_vec = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # FAISS search (inner product on L2-normalised = cosine similarity)
        scores, indices = self._index.search(query_vec, k)
        scores = scores[0]
        indices = indices[0]

        results: List[RetrievalResult] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue  # FAISS returns -1 for missing entries
            entry = self._metadata[idx]
            results.append(
                RetrievalResult(
                    text=entry["text"],
                    source=entry.get("source", "unknown"),
                    chunk_index=entry.get("chunk_index", 0),
                    score=round(float(score), 4),
                    method="faiss",
                )
            )

        # If all FAISS results are below threshold → keyword fallback
        above_threshold = [r for r in results if r.score >= self.score_threshold]
        if not above_threshold:
            logger.info(
                "All FAISS scores below %.2f — falling back to keyword search.",
                self.score_threshold,
            )
            return keyword_search(query, self._metadata, k)

        return above_threshold

    # ── Convenience ─────────────────────────────────────────────────────

    def get_context_text(self, query: str, top_k: int | None = None) -> str:
        """
        Return concatenated chunk texts as a single context string,
        ready to inject into a prompt.
        """
        results = self.search(query, top_k)
        if not results:
            return ""
        parts = []
        for r in results:
            parts.append(f"[Source: {r.source}]\n{r.text}")
        return "\n\n---\n\n".join(parts)
