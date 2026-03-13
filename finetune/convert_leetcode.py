"""
Convert merged_problems.json (LeetCode dataset) into the project's
instruction-tuning JSONL format for LoRA fine-tuning.

Produces finetune/data/dsa_master_training.jsonl

Usage:
    python -m finetune.convert_leetcode
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List

# ── Paths ───────────────────────────────────────────────────────────────────
SRC = Path("finetune/data/merged_problems.json")
DST = Path("finetune/data/dsa_master_training.jsonl")

# ── DSA-relevant topic filter ──────────────────────────────────────────────
DSA_TOPICS = {
    "Array", "String", "Hash Table", "Dynamic Programming", "Sorting",
    "Greedy", "Binary Search", "Depth-First Search", "Breadth-First Search",
    "Matrix", "Bit Manipulation", "Tree", "Two Pointers", "Prefix Sum",
    "Heap (Priority Queue)", "Graph", "Stack", "Linked List", "Queue",
    "Binary Tree", "Recursion", "Divide and Conquer", "Backtracking",
    "Trie", "Segment Tree", "Binary Indexed Tree", "Union Find",
    "Monotonic Stack", "Monotonic Queue", "Sliding Window",
    "Topological Sort", "Shortest Path", "Counting Sort",
    "Bucket Sort", "Radix Sort", "Merge Sort", "Quickselect",
    "Binary Search Tree", "Doubly-Linked List",
    "Minimum Spanning Tree", "Strongly Connected Component",
    "Ordered Set", "Line Sweep", "Counting",
    "Memoization", "Math",
}


def _clean_html(text: str) -> str:
    """Strip HTML tags and clean up LeetCode markdown artefacts."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"\$\$([^$]+)\$\$", r"\1", text)  # strip LaTeX wrappers
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_instruction(q: Dict[str, Any]) -> str:
    """Build a natural instruction from the LeetCode problem."""
    title = q["title"]
    desc = _clean_html(q.get("description", ""))
    examples_raw = q.get("examples", [])
    constraints = q.get("constraints", [])

    parts = [f"**{title}**\n"]
    if desc:
        parts.append(desc)

    if examples_raw:
        parts.append("\n**Examples:**")
        for ex in examples_raw[:3]:  # max 3 examples
            ex_text = ex.get("example_text", "")
            if ex_text:
                parts.append(ex_text)

    if constraints:
        parts.append("\n**Constraints:**")
        for c in constraints:
            parts.append(f"- {_clean_html(c)}")

    return "\n".join(parts)


def _build_response_from_solution(q: Dict[str, Any]) -> str:
    """Build a high-quality response from the editorial solution + code."""
    sol_text = _clean_html(q.get("solution", ""))
    py3_code = q.get("code_snippets", {}).get("python3", "")
    topics = q.get("topics", [])
    difficulty = q.get("difficulty", "")

    parts: List[str] = []

    # Difficulty and topics header
    if difficulty or topics:
        meta = []
        if difficulty:
            meta.append(f"**Difficulty:** {difficulty}")
        if topics:
            meta.append(f"**Topics:** {', '.join(topics)}")
        parts.append(" | ".join(meta))

    # Editorial solution (cleaned)
    if sol_text:
        parts.append(sol_text)

    # Python3 starter code reference
    if py3_code and py3_code.strip():
        parts.append(f"\n**Python3 Code Template:**\n```python\n{py3_code.strip()}\n```")

    return "\n\n".join(parts)


def _build_response_no_solution(q: Dict[str, Any]) -> str:
    """Build a response for problems WITHOUT an editorial, using available info."""
    title = q["title"]
    topics = q.get("topics", [])
    difficulty = q.get("difficulty", "")
    py3_code = q.get("code_snippets", {}).get("python3", "")
    hints = q.get("hints", [])

    parts: List[str] = []

    meta = []
    if difficulty:
        meta.append(f"**Difficulty:** {difficulty}")
    if topics:
        meta.append(f"**Topics:** {', '.join(topics)}")
    if meta:
        parts.append(" | ".join(meta))

    parts.append(f"**Problem:** {title}")

    # Use hints as approach guidance
    if hints:
        parts.append("**Approach Hints:**")
        for i, hint in enumerate(hints, 1):
            parts.append(f"{i}. {_clean_html(hint)}")

    # Add code template
    if py3_code and py3_code.strip():
        parts.append(f"\n**Python3 Code Template:**\n```python\n{py3_code.strip()}\n```")

    topic_str = ", ".join(topics) if topics else "general algorithms"
    parts.append(
        f"\nThis problem can be approached using **{topic_str}** techniques. "
        f"Think about the constraints and choose an approach that fits the expected time complexity."
    )

    return "\n\n".join(parts)


def convert() -> int:
    """Convert LeetCode JSON → training JSONL. Returns record count."""
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", data if isinstance(data, list) else [])
    seen_instructions: set[str] = set()
    records: List[Dict[str, str]] = []

    for q in questions:
        topics = set(q.get("topics", []))
        # Filter: keep only DSA-relevant problems
        if not topics & DSA_TOPICS:
            continue

        instruction = _build_instruction(q)

        # Deduplicate by title (proxy for unique instruction)
        title_key = q["title"].strip().lower()
        if title_key in seen_instructions:
            continue
        seen_instructions.add(title_key)

        sol = q.get("solution", "")
        if sol and len(sol) > 100:
            response = _build_response_from_solution(q)
        else:
            response = _build_response_no_solution(q)

        # Skip if response is too short to be useful
        if len(response) < 80:
            continue

        records.append({
            "instruction": instruction,
            "context": "",
            "response": response,
        })

    # Write JSONL
    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records)


def main() -> None:
    print(f"Reading {SRC} …")
    count = convert()
    print(f"Wrote {count} training records to {DST}")
    # Quick stats
    with open(DST, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lengths = [len(line) for line in lines]
    print(f"  Min record length: {min(lengths)} chars")
    print(f"  Max record length: {max(lengths)} chars")
    print(f"  Avg record length: {sum(lengths) // len(lengths)} chars")


if __name__ == "__main__":
    main()
