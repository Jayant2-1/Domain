"""
Download, filter, and convert two Hugging Face datasets into the
project's instruction-tuning JSONL format, then immediately launch
LoRA training on the merged dataset.

Datasets:
  1. sahil2801/CodeAlpaca-20k   – 20 k general code-instruction pairs
  2. newfacade/LeetCodeDataset  – LeetCode problems with solutions

All responses are reformatted with:
  • <think>...</think>  internal reasoning block (Socratic self-questioning)
  • Structured student-facing answer using Socratic + Feynman teaching methods
  • Code examples, complexity analysis, hint progression

Usage:
    python -m finetune.build_hf_datasets          # build data only
    python -m finetune.build_hf_datasets --train  # build + start training
    python -m finetune.build_hf_datasets --train --multi-round
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "finetune" / "data"
OUT_CODEALPACA = DATA_DIR / "codealpaca_hf.jsonl"
OUT_LEETCODE   = DATA_DIR / "leetcode_hf.jsonl"
OUT_COMBINED   = DATA_DIR / "hf_combined_training.jsonl"
MASTER_MERGE   = DATA_DIR / "combined_v2_training.jsonl"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── DSA keyword filter ────────────────────────────────────────────────────────
DSA_KEYWORDS: Set[str] = {
    "array", "string", "linked list", "tree", "graph", "dynamic programming",
    " dp ", "sorting", "searching", "hash table", "hash map", "stack",
    "queue", "recursion", "backtracking", "binary search", "two pointer",
    "sliding window", "greedy", "trie", "heap", "priority queue",
    "algorithm", "complexity", "big o", "time complexity", "space complexity",
    "pointer", "traversal", "bfs", "dfs", "breadth first", "depth first",
    "merge sort", "quick sort", "binary tree", "bst", "memoization",
    "fibonacci", "subset", "permutation", "combination", "palindrome",
    "anagram", "rotation", "subarray", "subsequence", "kadane", "dijkstra",
    "bellman", "kruskal", "prim", "topological", "union find", "segment tree",
    "fenwick", "suffix", "prefix sum", "bit manipulation", "xor", "bitwise",
    "matrix", "grid", "path", "cycle", "connected component", "lru cache",
    "implement", "write a function", "write a program", "code",
}

def _is_dsa(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in DSA_KEYWORDS)


# ── HTML / markdown cleanup ──────────────────────────────────────────────────
def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Socratic / Teaching response wrapper ─────────────────────────────────────
# The think block coaches the model to reason before answering.
# The student-facing block uses layered revelation: hint → key insight → full solution.

def _wrap_codealpaca(instruction: str, raw_output: str) -> str:
    """Wrap a CodeAlpaca response in Socratic teaching format."""
    code = raw_output.strip()

    think = textwrap.dedent(f"""\
        <think>
        The student is asking: "{instruction[:200]}"

        Let me work through this step by step before answering:
        1. What concept is being tested here?
        2. What common mistakes do learners make with this?
        3. What is the most illuminating way to explain it — can I start
           with a simpler sub-problem and build up?
        4. What is the time and space complexity of the solution?
        5. Are there edge cases the student should be aware of?
        </think>""")

    hint_question = _make_hint_question(instruction)

    body = textwrap.dedent(f"""\
        {hint_question}

        **Step-by-step breakdown:**

        {code}

        **Why this works — key insight:**
        Before jumping to the solution, notice that the problem reduces to a
        simpler sub-problem. Once you identify the invariant (the thing that
        stays true at every step), the code almost writes itself.

        **Complexity analysis:**
        - **Time:** Analyse each loop/recursive call — what's the dominant term?
        - **Space:** Consider auxiliary structures and the call stack.

        **Check your understanding:**
        > Can you modify this solution to handle the edge case where the input
        > is empty or contains duplicates? Try it first, then check your answer
        > against the solution above.""")

    return f"{think}\n\n{body}"


def _make_hint_question(instruction: str) -> str:
    """Generate a Socratic opening question tailored to the instruction."""
    low = instruction.lower()
    if "sort" in low:
        return ("Before we look at the code — *what property of the input* "
                "can we exploit to sort faster than O(n log n) in a special case?")
    if "search" in low or "find" in low:
        return ("Pause for a second: *is the data sorted or structured in any way?* "
                "That single question determines whether we can do better than O(n).")
    if "tree" in low or "bst" in low:
        return ("Think about this: *what invariant does a BST maintain at every node?* "
                "Your answer to that question is the key to every tree algorithm.")
    if "graph" in low or "bfs" in low or "dfs" in low:
        return ("Key question first: *do we need the shortest path, or just reachability?* "
                "That determines whether we reach for BFS or DFS.")
    if "dynamic" in low or " dp" in low:
        return ("Before coding, ask yourself: *does this problem have overlapping "
                "sub-problems and optimal sub-structure?* If yes, DP is your tool.")
    if "linked list" in low:
        return ("Visualise the list as a chain. *Which pointer do you advance first, "
                "and why does order matter?* Draw it out before you code.")
    return ("Take a moment to think: *what is the simplest input for which this "
            "problem is interesting?* Start there, then generalise.")


def _wrap_leetcode(desc: str, starter: str, difficulty: str, solution: str) -> tuple[str, str]:
    """
    Returns (instruction, response) for a LeetCode problem.
    Instruction = full problem spec.
    Response    = Socratic walk-through + full solution.
    """
    instruction = _clean(desc)
    if starter and starter.strip():
        instruction += f"\n\n**Starter code:**\n```python\n{starter.strip()}\n```"

    diff_label = difficulty.strip().capitalize() if difficulty else "Unknown"
    sol_clean = _clean(solution)

    think = textwrap.dedent(f"""\
        <think>
        Problem difficulty: {diff_label}.

        Socratic self-questioning before answering:
        1. What data structure is most natural here — array, hash map, tree, graph?
        2. Can I brute-force first? What's the brute-force complexity?
        3. Where is the bottleneck — and what pattern eliminates it?
           (Two pointers? Sliding window? DP? BFS/DFS?)
        4. What are the constraints? Do they hint at an O(n log n) or O(n) expected solution?
        5. Are there tricky edge cases: empty input, single element, duplicates, negatives?
        </think>""")

    approach_intro = _difficulty_intro(diff_label)

    body = textwrap.dedent(f"""\
        {approach_intro}

        **Guided discovery — try these hints before reading the solution:**

        - **Hint 1:** What happens if you process elements in a specific order?
          Does sorting help? Does a hash map give O(1) lookup?
        - **Hint 2:** Identify the invariant — what stays true after each iteration?
        - **Hint 3:** Write the recurrence (if DP) or the loop body (if iterative)
          in plain English before converting it to code.

        ---

        **Full solution ({diff_label}):**

        {sol_clean}

        ---

        **Complexity recap:**
        - Identify every loop and recursive call.
        - Ask: does this sub-problem repeat? → memoise it.
        - Ask: is this lookup O(1) or O(log n)? → choose your structure accordingly.

        **Challenge yourself:**
        > Can you solve this problem with *half the memory* you used above?
        > What trade-off would that require?""")

    return instruction, f"{think}\n\n{body}"


def _difficulty_intro(diff: str) -> str:
    if diff.lower() == "hard":
        return ("This is a **Hard** problem — expect it to combine at least two "
                "separate techniques. Break it into sub-problems and solve each independently.")
    if diff.lower() == "medium":
        return ("This is a **Medium** problem. The brute-force is usually obvious; "
                "the interview expects you to optimise it with a known pattern.")
    return ("This is an **Easy** problem. Focus on writing clean, readable code "
            "and articulating *why* it works — not just that it works.")


# ── Dataset 1: CodeAlpaca-20k ─────────────────────────────────────────────────

def build_codealpaca(limit: Optional[int] = None) -> List[Dict[str, str]]:
    log.info("Loading sahil2801/CodeAlpaca-20k from Hugging Face …")
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    log.info("  Raw size: %d rows", len(ds))

    records: List[Dict[str, str]] = []
    skipped = 0

    for row in ds:
        instruction: str = (row.get("instruction") or "").strip()
        inp: str         = (row.get("input") or "").strip()
        output: str      = (row.get("output") or "").strip()

        if not instruction or not output:
            skipped += 1
            continue

        combined = instruction + " " + output
        if not _is_dsa(combined):
            skipped += 1
            continue

        # Build context from the optional `input` field
        context = inp if inp else ""

        response = _wrap_codealpaca(instruction, output)

        records.append({
            "instruction": instruction,
            "context": context,
            "response": response,
        })

        if limit and len(records) >= limit:
            break

    log.info("  Kept %d / skipped %d records", len(records), skipped)
    return records


# ── Dataset 2: newfacade/LeetCodeDataset ─────────────────────────────────────

def build_leetcode(limit: Optional[int] = None) -> List[Dict[str, str]]:
    log.info("Loading newfacade/LeetCodeDataset from Hugging Face …")
    from datasets import load_dataset  # type: ignore

    # Try loading; fallback split names
    try:
        ds = load_dataset("newfacade/LeetCodeDataset", split="train")
    except Exception:
        try:
            ds = load_dataset("newfacade/LeetCodeDataset", split="test")
        except Exception:
            all_splits = load_dataset("newfacade/LeetCodeDataset")
            first_split = list(all_splits.keys())[0]
            log.warning("Unexpected split layout – using split '%s'", first_split)
            ds = all_splits[first_split]

    log.info("  Raw size: %d rows", len(ds))

    # Normalise field names (dataset sometimes ships with different casing)
    def _get(row: Any, *keys: str, default: str = "") -> str:
        for k in keys:
            v = row.get(k) or row.get(k.lower()) or row.get(k.upper())
            if v:
                return str(v).strip()
        return default

    records: List[Dict[str, str]] = []
    skipped = 0

    for row in ds:
        desc      = _get(row, "problem_description", "description", "content")
        starter   = _get(row, "starter_code", "starterCode", "code_snippet")
        difficulty = _get(row, "difficulty")
        # newfacade/LeetCodeDataset: explanation is in 'response', raw code in 'completion'
        solution  = _get(row, "response", "completion", "solution",
                         "editorial_solution", "python_solution")

        if not desc or not solution:
            skipped += 1
            continue

        instruction, response = _wrap_leetcode(desc, starter, difficulty, solution)

        if len(instruction) < 20 or len(response) < 50:
            skipped += 1
            continue

        records.append({
            "instruction": instruction,
            "context": f"Difficulty: {difficulty}" if difficulty else "",
            "response": response,
        })

        if limit and len(records) >= limit:
            break

    log.info("  Kept %d / skipped %d records", len(records), skipped)
    return records


# ── Persist helpers ──────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d records → %s", len(records), path)


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def merge_deduplicate(*sources: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: Set[str] = set()
    merged: List[Dict[str, str]] = []
    for recs in sources:
        for r in recs:
            key = r.get("instruction", "").strip().lower()[:300]
            if key and key not in seen:
                seen.add(key)
                merged.append(r)
    return merged


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build HF datasets + optionally train")
    parser.add_argument("--train",        action="store_true",
                        help="Launch LoRA training after building data")
    parser.add_argument("--multi-round",  action="store_true",
                        help="Use 3-round decaying-LR training schedule")
    parser.add_argument("--max-steps",    type=int, default=400,
                        help="Steps for single-round training (ignored with --multi-round)")
    parser.add_argument("--ca-limit",     type=int, default=None,
                        help="Max CodeAlpaca records to keep (default: all DSA ones)")
    parser.add_argument("--lc-limit",     type=int, default=None,
                        help="Max LeetCode records to keep (default: all)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading if JSONL files already exist")
    args = parser.parse_args()

    # ── Step 1: Download & convert ──────────────────────────────────────────
    if args.skip_download and OUT_CODEALPACA.exists() and OUT_LEETCODE.exists():
        log.info("--skip-download: reusing existing JSONL files")
        ca_records = load_jsonl(OUT_CODEALPACA)
        lc_records = load_jsonl(OUT_LEETCODE)
    else:
        ca_records = build_codealpaca(limit=args.ca_limit)
        write_jsonl(OUT_CODEALPACA, ca_records)

        lc_records = build_leetcode(limit=args.lc_limit)
        write_jsonl(OUT_LEETCODE, lc_records)

    # ── Step 2: Merge with existing data ────────────────────────────────────
    existing_sources = [
        DATA_DIR / "dsa_master_training.jsonl",
        DATA_DIR / "seed_dsa_training.jsonl",
        DATA_DIR / "quality_dsa_training.jsonl",
        DATA_DIR / "corpus_training.jsonl",
    ]
    existing: List[Dict[str, str]] = []
    for p in existing_sources:
        chunk = load_jsonl(p)
        if chunk:
            log.info("  Loaded %d records from %s", len(chunk), p.name)
            existing.extend(chunk)

    merged = merge_deduplicate(existing, ca_records, lc_records)
    write_jsonl(MASTER_MERGE, merged)

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Existing DSA records : {len(existing):>6}")
    print(f"  CodeAlpaca-20k (DSA) : {len(ca_records):>6}")
    print(f"  LeetCode dataset     : {len(lc_records):>6}")
    print(f"  ─────────────────────────────")
    print(f"  Merged (deduplicated): {len(merged):>6}")
    print(f"  Output               : {MASTER_MERGE}")
    print("=" * 60)
    print()

    if not args.train:
        print("  Data ready. Run with --train to start LoRA training.")
        return

    # ── Step 3: Launch training ──────────────────────────────────────────────
    print("  Launching LoRA training …")
    print()

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_out = str(ROOT / "adapters" / f"v_{ts}")

    cmd = [
        sys.executable, "-m", "finetune.train_lora",
        "--data", str(MASTER_MERGE),
        "--output", adapter_out,
    ]
    if args.multi_round:
        cmd.append("--multi-round")
    else:
        cmd += ["--max-steps", str(args.max_steps)]

    log.info("Running: %s", " ".join(cmd))
    # Run in a fresh process so GPU memory is clean (avoids WDDM segfault
    # from memory fragmentation after the large dataset build phase)
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if result.returncode == 139:
        log.error("Training crashed with SIGSEGV (exit 139). "
                  "This is usually a CUDA/bitsandbytes memory issue on Windows."
                  " Try closing other GPU-heavy apps and re-running:\n"
                  "  python -m finetune.train_lora --data %s --output %s --multi-round",
                  MASTER_MERGE, adapter_out)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
