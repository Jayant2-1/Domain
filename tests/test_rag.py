"""
Unit tests for the RAG subsystem — build_index + retriever.

Tests are designed to be self-contained: they create temporary corpus
files, build an index in a temp directory, and validate the full
pipeline: load → embed → persist → reload → search → fallback.

Requires: sentence-transformers, faiss-cpu, numpy, pytest.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from rag.build_index import (
    build_chunks,
    chunk_text,
    embed_and_build_index,
    load_documents,
    save_index,
)
from app.retriever import (
    Retriever,
    RetrievalResult,
    keyword_search,
)

# ── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_DOCS = {
    "arrays.txt": (
        "An array is a data structure consisting of a collection of elements, "
        "each identified by an index. Arrays are commonly used because they "
        "allow fast access to elements using an index. Time complexity of "
        "access is O(1). Insertion at the end is amortized O(1) for dynamic "
        "arrays. Searching an unsorted array is O(n)."
    ),
    "linked_lists.md": (
        "A linked list is a linear data structure where each element is a "
        "separate object called a node. Each node contains data and a reference "
        "to the next node. Linked lists allow efficient insertion and deletion "
        "at any position in O(1) if the node reference is known. However, "
        "random access is O(n) because you must traverse from the head."
    ),
    "binary_trees.txt": (
        "A binary tree is a tree data structure in which each node has at most "
        "two children referred to as left child and right child. Binary search "
        "trees maintain the property that left subtree values are less than "
        "the root and right subtree values are greater. This allows O(log n) "
        "search, insert, and delete operations on average."
    ),
}


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory) -> str:
    """Create a temporary corpus directory with sample DSA docs."""
    d = tmp_path_factory.mktemp("corpus")
    for name, content in SAMPLE_DOCS.items():
        (d / name).write_text(content, encoding="utf-8")
    return str(d)


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory, corpus_dir) -> str:
    """Build a FAISS index from the sample corpus and return the output dir."""
    out = tmp_path_factory.mktemp("faiss_index")
    docs = load_documents(corpus_dir)
    texts, metadata = build_chunks(docs, chunk_size=50, overlap=10)
    index, _ = embed_and_build_index(texts)
    save_index(index, metadata, texts, str(out))
    return str(out)


# ── build_index tests ──────────────────────────────────────────────────────

class TestChunkText:
    def test_basic_chunking(self):
        text = " ".join(f"word{i}" for i in range(100))
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) >= 3
        # Each chunk should have at most 30 words
        for c in chunks:
            assert len(c.split()) <= 30

    def test_overlap_present(self):
        """Consecutive chunks should share overlapping words."""
        text = " ".join(f"w{i}" for i in range(50))
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        if len(chunks) >= 2:
            words_0 = set(chunks[0].split())
            words_1 = set(chunks[1].split())
            assert len(words_0 & words_1) >= 5

    def test_empty_text(self):
        assert chunk_text("", chunk_size=10, overlap=2) == []

    def test_short_text(self):
        """Text shorter than chunk_size → single chunk."""
        chunks = chunk_text("hello world", chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"


class TestLoadDocuments:
    def test_loads_all_files(self, corpus_dir):
        docs = load_documents(corpus_dir)
        assert len(docs) == 3

    def test_filenames_present(self, corpus_dir):
        docs = load_documents(corpus_dir)
        names = {name for name, _ in docs}
        assert "arrays.txt" in names
        assert "linked_lists.md" in names
        assert "binary_trees.txt" in names

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path/xyz")


class TestBuildChunks:
    def test_metadata_matches_texts(self, corpus_dir):
        docs = load_documents(corpus_dir)
        texts, meta = build_chunks(docs, chunk_size=30, overlap=5)
        assert len(texts) == len(meta)
        assert all("source" in m for m in meta)
        assert all("chunk_index" in m for m in meta)


class TestEmbedAndIndex:
    def test_index_built(self, corpus_dir):
        docs = load_documents(corpus_dir)
        texts, _ = build_chunks(docs, chunk_size=50, overlap=10)
        index, embeddings = embed_and_build_index(texts)
        assert index.ntotal == len(texts)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 384  # MiniLM-L6-v2 dim


class TestSaveAndReload:
    def test_files_exist(self, index_dir):
        assert (Path(index_dir) / "index.faiss").exists()
        assert (Path(index_dir) / "metadata.json").exists()

    def test_reload_index(self, index_dir):
        idx = faiss.read_index(str(Path(index_dir) / "index.faiss"))
        assert idx.ntotal > 0

    def test_reload_metadata(self, index_dir):
        with open(Path(index_dir) / "metadata.json", "r") as f:
            meta = json.load(f)
        assert len(meta) > 0
        assert "text" in meta[0]
        assert "source" in meta[0]


# ── retriever tests ─────────────────────────────────────────────────────────

class TestKeywordSearch:
    def test_basic_overlap(self):
        meta = [
            {"text": "array data structure index access", "source": "a.txt", "chunk_index": 0},
            {"text": "linked list node pointer next", "source": "b.txt", "chunk_index": 0},
            {"text": "binary tree left right child", "source": "c.txt", "chunk_index": 0},
        ]
        results = keyword_search("array index access", meta, top_k=2)
        assert len(results) >= 1
        assert results[0].source == "a.txt"
        assert results[0].method == "keyword"

    def test_empty_query(self):
        meta = [{"text": "some text", "source": "a.txt", "chunk_index": 0}]
        results = keyword_search("", meta, top_k=3)
        assert results == []

    def test_no_overlap(self):
        meta = [{"text": "xyz abc def", "source": "a.txt", "chunk_index": 0}]
        results = keyword_search("quantum physics relativity", meta, top_k=3)
        assert results == []


class TestRetriever:
    def test_load_and_search(self, index_dir):
        """Full pipeline: load persisted index → search → get results."""
        r = Retriever(index_dir=index_dir, top_k=2)
        r.load()
        assert r.is_loaded

        results = r.search("What is the time complexity of array access?")
        assert len(results) >= 1
        assert all(isinstance(rr, RetrievalResult) for rr in results)
        # Should find array-related content
        combined = " ".join(rr.text for rr in results).lower()
        assert "array" in combined

    def test_search_returns_scores(self, index_dir):
        r = Retriever(index_dir=index_dir, top_k=3)
        r.load()
        results = r.search("binary search tree operations")
        assert all(isinstance(rr.score, float) for rr in results)
        assert all(rr.score > 0 for rr in results)

    def test_search_method_is_faiss(self, index_dir):
        r = Retriever(index_dir=index_dir, top_k=2)
        r.load()
        results = r.search("linked list insertion complexity")
        faiss_results = [rr for rr in results if rr.method == "faiss"]
        assert len(faiss_results) >= 1

    def test_get_context_text(self, index_dir):
        r = Retriever(index_dir=index_dir, top_k=2)
        r.load()
        ctx = r.get_context_text("array data structure")
        assert isinstance(ctx, str)
        assert len(ctx) > 0
        assert "[Source:" in ctx

    def test_unloaded_retriever_returns_empty(self):
        """Search on an unloaded retriever returns empty (no crash)."""
        r = Retriever(index_dir="/nonexistent")
        results = r.search("anything")
        assert results == []

    def test_missing_index_raises_on_load(self):
        r = Retriever(index_dir="/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            r.load()

    def test_reload_gives_same_results(self, index_dir):
        """Loading the same index twice should produce identical results."""
        query = "binary tree search complexity"

        r1 = Retriever(index_dir=index_dir, top_k=2)
        r1.load()
        res1 = r1.search(query)

        r2 = Retriever(index_dir=index_dir, top_k=2)
        r2.load()
        res2 = r2.search(query)

        assert len(res1) == len(res2)
        for a, b in zip(res1, res2):
            assert a.text == b.text
            assert a.score == pytest.approx(b.score)


class TestFallbackBehavior:
    def test_low_threshold_forces_keyword(self, index_dir):
        """With an impossibly high threshold, FAISS results fall below → keyword fallback."""
        r = Retriever(index_dir=index_dir, top_k=2, score_threshold=0.9999)
        r.load()
        results = r.search("array data structure")
        # Should still return results via keyword fallback (if any match)
        if results:
            assert all(rr.method == "keyword" for rr in results)
