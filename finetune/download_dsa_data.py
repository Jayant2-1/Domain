"""
Download supplementary DSA training data from public Hugging Face datasets.

Filters for DSA-relevant records using keyword matching and converts them
to the project's JSONL schema (instruction, context, response).

Usage:
    python -m finetune.download_dsa_data

Output:
    finetune/data/downloaded_supplement.jsonl

This is optional — run_full_training.py works without it, but will
automatically include the supplement if it exists.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

OUTPUT = Path("finetune/data/downloaded_supplement.jsonl")

# DSA keywords for filtering (same spirit as app/domain_validator.py)
DSA_KEYWORDS: Set[str] = {
    # Data structures
    "array", "linked list", "stack", "queue", "tree", "binary tree",
    "bst", "binary search tree", "avl", "trie", "heap", "priority queue",
    "hash table", "hash map", "graph", "matrix", "segment tree",
    "union find", "disjoint set", "fenwick",
    # Algorithms
    "sort", "sorting", "merge sort", "quick sort", "heap sort",
    "binary search", "dfs", "bfs", "depth first", "breadth first",
    "dijkstra", "bellman ford", "floyd warshall", "kruskal", "prim",
    "dynamic programming", "memoization", "tabulation",
    "greedy", "backtracking", "recursion", "divide and conquer",
    "sliding window", "two pointer", "kadane",
    "knapsack", "shortest path", "topological sort",
    "minimum spanning tree",
    # Concepts
    "big o", "time complexity", "space complexity",
    "traversal", "inorder", "preorder", "postorder",
    "cycle detection", "connected component", "bipartite",
    # Patterns
    "leetcode", "algorithm", "data structure",
    "subsequence", "subarray", "substring", "permutation",
    "palindrome", "prefix sum",
}

# Compile a regex pattern for efficient matching
_KEYWORD_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in DSA_KEYWORDS),
    re.IGNORECASE,
)


def is_dsa_related(text: str) -> bool:
    """Check if text contains DSA keywords."""
    return bool(_KEYWORD_PATTERN.search(text))


def process_dataset(
    dataset_name: str,
    instruction_field: str = "instruction",
    response_field: str = "output",
    context_field: str = "input",
) -> List[Dict[str, str]]:
    """Download, filter, and convert a single HF dataset."""
    from datasets import load_dataset

    print(f"Downloading {dataset_name}...", end=" ", flush=True)
    try:
        ds = load_dataset(dataset_name, split="train")
        print(f"OK ({len(ds)} records)")
    except Exception as e:
        print(f"FAILED: {e}")
        return []

    print(f"Filtering for DSA content...", end=" ", flush=True)
    filtered: List[Dict[str, str]] = []

    for record in ds:
        instruction = record.get(instruction_field, "")
        if not instruction:
            continue

        # Check instruction for DSA relevance
        if not is_dsa_related(instruction):
            continue

        response = record.get(response_field, "")
        if not response or len(response) < 50:
            continue

        context = record.get(context_field, "") or ""

        filtered.append({
            "instruction": instruction.strip(),
            "context": context.strip() if context else "",
            "response": response.strip(),
        })

    print(f"OK ({len(filtered)} DSA records found)")
    return filtered


def main() -> None:
    print()
    print("=" * 60)
    print("  DSA Training Data Downloader")
    print("=" * 60)
    print()

    # Check if datasets library is installed
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Install it with: pip install datasets")
        sys.exit(1)

    all_records: List[Dict[str, str]] = []
    seen_instructions: Set[str] = set()

    # Dataset configs: (name, instruction_field, response_field, context_field)
    sources = [
        ("iamtarun/python_code_instructions_18k_alpaca", "instruction", "output", "input"),
        ("TokenBender/code_instructions_122k_alpaca_style", "instruction", "output", "input"),
        ("sahil2801/CodeAlpaca-20k", "instruction", "output", "input"),
    ]

    for name, inst_f, resp_f, ctx_f in sources:
        records = process_dataset(name, inst_f, resp_f, ctx_f)

        # Deduplicate against already-collected records
        new_count = 0
        for rec in records:
            key = rec["instruction"].strip().lower()
            if key not in seen_instructions:
                seen_instructions.add(key)
                all_records.append(rec)
                new_count += 1

        print(f"  Added {new_count} new unique records (total: {len(all_records)})")
        print()

    # Save
    if all_records:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved {len(all_records)} records to {OUTPUT}")
    else:
        print("No DSA records found. Supplement file not created.")

    print()
    print("=" * 60)
    print(f"  Total DSA records downloaded: {len(all_records)}")
    print(f"  Output: {OUTPUT}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
