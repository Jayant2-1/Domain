#!/usr/bin/env python
"""
Build fine-tuning training data from the local RAG corpus.

Reads the 10 DSA markdown files under rag/corpus/, parses them into
meaningful sections, and generates instruction/response training pairs
that match the inference pipeline format (including <think> tags).

Usage:
    python finetune/build_training_from_corpus.py

Output:
    finetune/data/corpus_training.jsonl   – one JSON object per line
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "rag" / "corpus"
OUTPUT_FILE = ROOT / "finetune" / "data" / "corpus_training.jsonl"

# ── Topic mapping ────────────────────────────────────────────────────────────

TOPIC_MAP = {
    "01_arrays.md": "arrays",
    "02_linked_lists.md": "linked_lists",
    "03_stacks_queues.md": "stacks_queues",
    "04_hash_tables.md": "hash_tables",
    "05_trees.md": "trees",
    "06_graphs.md": "graphs",
    "07_sorting_searching.md": "sorting_searching",
    "08_dynamic_programming.md": "dynamic_programming",
    "09_recursion_backtracking.md": "recursion_backtracking",
    "10_strings.md": "strings",
}

TOPIC_DISPLAY = {k: v.replace("_", " ").title() for k, v in TOPIC_MAP.items()}


# ── Markdown parsing ─────────────────────────────────────────────────────────

def parse_sections(text: str) -> List[Dict]:
    """
    Parse markdown into a flat list of sections.

    Each section dict has: level, title, body (raw text under heading),
    parent_title (title of parent ## section if this is ###/####).
    """
    sections: list[dict] = []
    current: dict | None = None
    parent_h2 = ""

    for line in text.split("\n"):
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            # save previous
            if current and current["body"].strip():
                sections.append(current)
            level = len(m.group(1))
            title = m.group(2).strip()
            if level <= 2:
                parent_h2 = title
            current = {
                "level": level,
                "title": title,
                "body": "",
                "parent": parent_h2 if level > 2 else "",
            }
        elif current is not None:
            current["body"] += line + "\n"

    if current and current["body"].strip():
        sections.append(current)

    return sections


def extract_code_blocks(text: str) -> List[str]:
    """Extract fenced code blocks from markdown."""
    return re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)


def extract_complexities(text: str) -> List[str]:
    """Pull out Big-O expressions."""
    return list(set(re.findall(r"O\([^)]+\)", text)))


def has_table(text: str) -> bool:
    return bool(re.search(r"\|.*\|.*\|", text))


def strip_section_heading(title: str) -> str:
    """Clean up heading text: remove trailing markers like ' — extra info'."""
    # keep the whole thing, but strip leading/trailing whitespace
    return title.strip()


# ── <think> block builder ────────────────────────────────────────────────────

def build_think_block(
    title: str,
    body: str,
    topic_display: str,
) -> str:
    """
    Build a <think> analysis block from section content.
    Matches the format expected by app/reasoning.py.
    """
    # Pattern: use the section title as pattern name
    pattern = title

    # Strategy: first meaningful non-code sentence from the body
    strategy = "Analyze and implement step by step."
    in_code = False
    for line in body.strip().split("\n"):
        line_s = line.strip()
        if line_s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        # Skip empty, table lines, headings, very short lines
        if (not line_s or line_s.startswith("|") 
                or line_s.startswith("#") or len(line_s) < 15):
            continue
        # Strip leading markdown list markers and bold prefixes
        line_s = re.sub(r"^[-*•]\s+", "", line_s)
        line_s = re.sub(r"^\*\*.*?\*\*:?\s*", "", line_s)
        if line_s and len(line_s) >= 15:
            strategy = line_s[:200]
            break
    if len(strategy) > 200:
        strategy = strategy[:197] + "..."

    # Complexity
    complexities = extract_complexities(body)
    if complexities:
        complexity_str = "Time: " + ", ".join(complexities[:2])
    else:
        complexity_str = "Depends on the specific implementation"

    # Edge cases: heuristic extraction
    edge_cases = _extract_edge_cases(body, title, topic_display)

    return (
        "<think>\n"
        f"Pattern: {pattern}\n"
        f"Strategy: {strategy}\n"
        f"Complexity: {complexity_str}\n"
        f"Edge cases: {edge_cases}\n"
        "</think>"
    )


def _extract_edge_cases(body: str, title: str, topic: str) -> str:
    """Heuristically pull edge cases from content."""
    # Look for explicit edge-case mentions
    ec_lines = []
    for line in body.split("\n"):
        lower = line.lower()
        if any(kw in lower for kw in ("edge case", "corner case", "special case",
                                       "empty", "single element", "null",
                                       "overflow", "negative", "duplicate")):
            clean = line.strip().lstrip("-•*").strip()
            if clean:
                ec_lines.append(clean)

    if ec_lines:
        return "; ".join(ec_lines[:3])

    # Fallback heuristics by topic
    topic_l = topic.lower()
    if "array" in topic_l:
        return "Empty array, single element, all duplicates, very large input"
    if "linked" in topic_l:
        return "Empty list, single node, cycle present, null pointers"
    if "tree" in topic_l:
        return "Empty tree, single node, skewed tree, very deep recursion"
    if "graph" in topic_l:
        return "Disconnected graph, self-loops, negative weights, empty graph"
    if "stack" in topic_l or "queue" in topic_l:
        return "Empty structure, single element, overflow conditions"
    if "hash" in topic_l:
        return "Empty input, all collisions, very high load factor, duplicate keys"
    if "sort" in topic_l or "search" in topic_l:
        return "Already sorted, reverse sorted, all equal elements, single element"
    if "dynamic" in topic_l or "dp" in topic_l:
        return "Base cases, zero values, negative numbers, very large inputs"
    if "recursion" in topic_l or "backtrack" in topic_l:
        return "Maximum recursion depth, empty input, no valid solution exists"
    if "string" in topic_l:
        return "Empty string, single character, all same characters, Unicode"
    return "Empty input, boundary values, large inputs"


# ── Question generators ─────────────────────────────────────────────────────

def make_questions(
    section: Dict,
    topic_display: str,
) -> List[Tuple[str, str]]:
    """
    Generate diverse question variants for a section.
    Returns list of (question_text, intent_type) tuples.
    """
    title = section["title"]
    body = section["body"]
    parent = section.get("parent", "")
    questions: list[tuple[str, str]] = []

    # Skip sections that are just headers or too short
    if len(body.strip()) < 50:
        return []

    # Skip meta sections
    if title.lower() in ("practice tips", "key properties", "properties"):
        # Still generate one explain question for these
        if len(body.strip()) > 80:
            questions.append(
                (f"What are the key properties and tips for {topic_display}?", "explain")
            )
        return questions

    title_clean = re.sub(r"\s*[—–-]\s+.*$", "", title).strip()

    # If the title is a question like "What is X?", use it directly
    is_question = title_clean.lower().startswith(("what ", "how ", "why ", "when "))

    # 1. EXPLAIN variant — always
    if is_question:
        questions.append(
            (title_clean if title_clean.endswith("?") else title_clean + "?", "explain")
        )
    elif parent:
        questions.append(
            (f"Explain {title_clean} in the context of {topic_display}.", "explain")
        )
    else:
        questions.append(
            (f"Explain {title_clean}.", "explain")
        )

    # 2. SOLVE variant — if there's code
    code_blocks = extract_code_blocks(body)
    if code_blocks:
        questions.append(
            (f"Implement {title_clean} in Python and explain how it works.", "solve")
        )

    # 3. ANALYZE variant — if complexity info exists
    complexities = extract_complexities(body)
    if complexities:
        questions.append(
            (f"What is the time and space complexity of {title_clean}? Explain why.", "analyze")
        )

    # 4. Additional natural phrasing variants
    lower_title = title.lower()
    if "algorithm" in lower_title or "sort" in lower_title:
        questions.append(
            (f"How does {title_clean} work step by step?", "explain")
        )
    elif "pattern" in lower_title or "technique" in lower_title:
        questions.append(
            (f"When should I use the {title_clean} pattern and why?", "explain")
        )
    elif "problem" in lower_title or "classic" in lower_title:
        # Don't generate for container headings like "Classic Problems"
        pass

    return questions


# ── Response builder ─────────────────────────────────────────────────────────

def build_response(
    section: Dict,
    intent: str,
    topic_display: str,
) -> str:
    """
    Build a training response from section content.

    The response includes a <think> analysis block followed by the section
    content formatted as a proper tutoring answer.
    """
    title = section["title"]
    body = section["body"].strip()

    # Build <think> block
    think = build_think_block(title, body, topic_display)

    # Build the answer portion from the corpus content
    # Clean up the body: keep code blocks, explanations, tables
    answer_parts = [f"## {title}\n"]

    # Add the body content, but clean up markdown artifacts
    answer_parts.append(body)

    # If there's no complexity analysis in the body but we extracted some,
    # add it at the end
    complexities = extract_complexities(body)
    if complexities and "complexity" not in body.lower():
        answer_parts.append(
            f"\n**Complexity:** {', '.join(complexities)}"
        )

    answer = "\n".join(answer_parts).strip()

    # Combine think + answer
    full_response = f"{think}\n\n{answer}"
    return full_response


# ── Context builder ──────────────────────────────────────────────────────────

def build_context(section: Dict, all_sections: List[Dict]) -> str:
    """
    Build a context snippet from related sections (simulates RAG context).
    """
    # Use parent section's intro or neighboring content as context
    title = section["title"]
    parent = section.get("parent", "")

    context_parts = []
    if parent and parent != title:
        # Find the parent section and use its first paragraph
        for s in all_sections:
            if s["title"] == parent and s["level"] <= 2:
                first_para = s["body"].strip().split("\n\n")[0]
                if len(first_para) > 30:
                    context_parts.append(first_para.strip())
                break

    # Include the section content as reference material (first ~300 chars)
    body_preview = section["body"].strip()
    if len(body_preview) > 400:
        body_preview = body_preview[:400] + "..."
    context_parts.append(body_preview)

    return "\n\n".join(context_parts)


# ── Main pipeline ────────────────────────────────────────────────────────────

def process_file(filepath: Path, topic: str, topic_display: str) -> List[Dict]:
    """Process one corpus markdown file into training records."""
    text = filepath.read_text(encoding="utf-8")
    sections = parse_sections(text)
    records = []

    for section in sections:
        questions = make_questions(section, topic_display)
        if not questions:
            continue

        context = build_context(section, sections)

        for question, intent in questions:
            response = build_response(section, intent, topic_display)

            # Skip if response is too short
            if len(response) < 100:
                continue

            record = {
                "instruction": question,
                "context": context,
                "response": response,
            }
            records.append(record)

    return records


def generate_comparison_pairs() -> List[Dict]:
    """
    Generate cross-topic comparison training pairs.
    These teach the model to compare different DSA concepts.
    """
    comparisons = [
        {
            "instruction": "Compare BFS and DFS for graph traversal. When should I use each?",
            "context": "BFS uses a queue and visits nodes level by level. DFS uses a stack (or recursion) and explores as far as possible along each branch before backtracking.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: Graph Traversal Comparison
                Strategy: Compare BFS and DFS across multiple dimensions: data structure used, traversal order, use cases, complexity, and implementation.
                Complexity: Both are O(V + E) time, O(V) space
                Edge cases: Disconnected graphs need multiple traversal starts; infinite graphs need depth limits for DFS
                </think>

                ## BFS vs DFS Comparison

                | Aspect | BFS | DFS |
                |--------|-----|-----|
                | Data Structure | Queue (FIFO) | Stack / Recursion (LIFO) |
                | Traversal Order | Level by level | Branch by branch (deep first) |
                | Shortest Path | Yes (unweighted graphs) | No |
                | Memory | O(w) where w = max width | O(h) where h = max height |
                | Time | O(V + E) | O(V + E) |
                | Implementation | Iterative with queue | Recursive or iterative with stack |

                **When to use BFS:**
                - Finding shortest path in unweighted graphs
                - Level-order traversal of trees
                - Finding all nodes within k distance
                - Peer-to-peer networks, social network connections

                **When to use DFS:**
                - Detecting cycles
                - Topological sorting
                - Finding connected components
                - Solving puzzles (mazes, sudoku)
                - Path existence (not necessarily shortest)

                ```python
                from collections import deque

                def bfs(graph, start):
                    visited = {start}
                    queue = deque([start])
                    order = []
                    while queue:
                        node = queue.popleft()
                        order.append(node)
                        for neighbor in graph[node]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                    return order

                def dfs(graph, start, visited=None):
                    if visited is None:
                        visited = set()
                    visited.add(start)
                    order = [start]
                    for neighbor in graph[start]:
                        if neighbor not in visited:
                            order.extend(dfs(graph, neighbor, visited))
                    return order
                ```

                **Key insight:** BFS guarantees shortest path in unweighted graphs because it explores all nodes at distance k before any node at distance k+1. DFS is better for problems requiring exhaustive exploration of all paths.""").strip(),
        },
        {
            "instruction": "Compare arrays and linked lists. What are the trade-offs?",
            "context": "Arrays store elements in contiguous memory with O(1) random access. Linked lists store elements in nodes connected by pointers, allowing O(1) insertion/deletion at known positions.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: Data Structure Comparison
                Strategy: Compare arrays and linked lists across access patterns, memory layout, insertion/deletion efficiency, and practical use cases.
                Complexity: Array access O(1), LL access O(n); Array insert O(n), LL insert O(1) at known position
                Edge cases: Dynamic arrays amortize resizing; linked lists have pointer overhead per node
                </think>

                ## Arrays vs Linked Lists

                | Operation | Array | Linked List |
                |-----------|-------|-------------|
                | Access by index | O(1) | O(n) |
                | Search | O(n) / O(log n) sorted | O(n) |
                | Insert at beginning | O(n) | O(1) |
                | Insert at end | O(1) amortized | O(1) with tail pointer |
                | Insert at middle | O(n) | O(1) after traversal |
                | Delete | O(n) | O(1) after traversal |
                | Memory | Contiguous, cache-friendly | Scattered, pointer overhead |

                **Use Arrays when:**
                - You need frequent random access by index
                - Data size is known or changes infrequently
                - Cache performance matters (contiguous memory)
                - You need to sort or binary search

                **Use Linked Lists when:**
                - Frequent insertions/deletions at arbitrary positions
                - You don't know the size in advance
                - You need to implement stacks, queues, or LRU caches
                - Memory fragmentation is acceptable

                ```python
                # Array (Python list) — O(1) access, O(n) insert at beginning
                arr = [1, 2, 3, 4, 5]
                arr[2]           # O(1) access
                arr.insert(0, 0) # O(n) — shifts all elements

                # Linked List — O(1) insert at head, O(n) access
                class Node:
                    def __init__(self, val, next=None):
                        self.val = val
                        self.next = next

                head = Node(1, Node(2, Node(3)))
                # Insert at head: O(1)
                new_head = Node(0, head)
                ```

                **Practical note:** In modern hardware, arrays almost always outperform linked lists due to CPU cache locality, even for operations where linked lists have better theoretical complexity.""").strip(),
        },
        {
            "instruction": "Compare merge sort and quicksort. Which one should I use?",
            "context": "Merge sort divides the array in half, recursively sorts each half, then merges. Quicksort picks a pivot, partitions around it, then recursively sorts the partitions.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: Sorting Algorithm Comparison
                Strategy: Compare merge sort and quicksort on time complexity (average, worst), space usage, stability, cache performance, and practical recommendations.
                Complexity: Both O(n log n) average; quicksort O(n²) worst case, merge sort O(n log n) guaranteed
                Edge cases: Quicksort degrades on already-sorted input with bad pivot choice; merge sort needs O(n) extra space
                </think>

                ## Merge Sort vs Quick Sort

                | Aspect | Merge Sort | Quick Sort |
                |--------|-----------|------------|
                | Average Time | O(n log n) | O(n log n) |
                | Worst Time | O(n log n) | O(n²) |
                | Space | O(n) | O(log n) |
                | Stable | Yes | No (default) |
                | In-place | No | Yes |
                | Cache Friendly | Less | More |

                **Merge Sort:**
                - Guaranteed O(n log n) regardless of input
                - Stable — preserves relative order of equal elements
                - Requires O(n) extra space for merging
                - Great for linked lists (no random access needed)
                - Used in Python's `sorted()` (Timsort is merge sort + insertion sort)

                **Quick Sort:**
                - O(n log n) average, but O(n²) worst case (mitigated by random pivot)
                - In-place — only O(log n) stack space
                - Better cache locality (sequential access pattern)
                - Generally faster in practice due to smaller constant factors
                - Used in C's qsort

                ```python
                def merge_sort(arr):
                    if len(arr) <= 1:
                        return arr
                    mid = len(arr) // 2
                    left = merge_sort(arr[:mid])
                    right = merge_sort(arr[mid:])
                    return merge(left, right)

                def merge(left, right):
                    result = []
                    i = j = 0
                    while i < len(left) and j < len(right):
                        if left[i] <= right[j]:
                            result.append(left[i]); i += 1
                        else:
                            result.append(right[j]); j += 1
                    result.extend(left[i:])
                    result.extend(right[j:])
                    return result

                def quicksort(arr):
                    if len(arr) <= 1:
                        return arr
                    pivot = arr[len(arr) // 2]
                    left = [x for x in arr if x < pivot]
                    mid = [x for x in arr if x == pivot]
                    right = [x for x in arr if x > pivot]
                    return quicksort(left) + mid + quicksort(right)
                ```

                **Recommendation:** Use merge sort when stability matters or worst-case guarantees are needed. Use quicksort for general-purpose in-memory sorting where average performance is key.""").strip(),
        },
        {
            "instruction": "Compare dynamic programming and greedy algorithms. How do I decide which to use?",
            "context": "Dynamic programming solves problems by combining solutions to overlapping subproblems. Greedy algorithms make locally optimal choices at each step, hoping to find a global optimum.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: Algorithm Paradigm Comparison
                Strategy: Compare DP and greedy across optimality guarantees, problem characteristics, implementation patterns, and classic examples.
                Complexity: DP typically O(n²) or O(n*W); Greedy typically O(n log n) or O(n)
                Edge cases: Greedy fails when local optimal ≠ global optimal (e.g., 0/1 knapsack); DP may be overkill for problems with greedy solutions
                </think>

                ## Dynamic Programming vs Greedy Algorithms

                | Aspect | Dynamic Programming | Greedy |
                |--------|-------------------|--------|
                | Approach | Try all subproblems, pick best | Make locally optimal choice |
                | Optimality | Always optimal (if applicable) | Not always optimal |
                | Subproblems | Overlapping | No revisiting |
                | Complexity | Usually higher | Usually lower |
                | Implementation | Table/memoization | Simple iteration |

                **Use Dynamic Programming when:**
                - Problem has overlapping subproblems AND optimal substructure
                - You need the guaranteed optimal solution
                - Greedy approach fails (can't prove greedy choice property)
                - Examples: 0/1 Knapsack, Edit Distance, LCS, Coin Change (general)

                **Use Greedy when:**
                - Problem has the greedy choice property (local optimal → global optimal)
                - You can prove the greedy approach works
                - Examples: Activity Selection, Huffman Coding, Fractional Knapsack, Dijkstra's

                **How to decide:**
                1. Can you prove the greedy choice property? → Use Greedy
                2. Does the problem have overlapping subproblems? → Use DP
                3. If neither is obvious, try greedy first (simpler), verify with examples

                ```python
                # Greedy: Activity Selection (works because greedy choice property holds)
                def activity_selection(activities):
                    activities.sort(key=lambda x: x[1])  # sort by end time
                    selected = [activities[0]]
                    for act in activities[1:]:
                        if act[0] >= selected[-1][1]:
                            selected.append(act)
                    return selected

                # DP: 0/1 Knapsack (greedy FAILS here — must use DP)
                def knapsack(weights, values, W):
                    n = len(weights)
                    dp = [0] * (W + 1)
                    for i in range(n):
                        for w in range(W, weights[i] - 1, -1):
                            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
                    return dp[W]
                ```

                **Key insight:** Greedy is faster but only works when you can prove local choices lead to global optimal. DP is more general but slower — it considers all possibilities.""").strip(),
        },
        {
            "instruction": "Compare hash tables and binary search trees for storing key-value pairs.",
            "context": "Hash tables provide O(1) average-case operations using hash functions. Binary search trees maintain sorted order and provide O(log n) operations.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: Data Structure Comparison
                Strategy: Compare hash tables and BSTs on lookup/insert/delete performance, ordering capabilities, memory usage, and use cases.
                Complexity: Hash table O(1) average; BST O(log n) for balanced trees
                Edge cases: Hash table O(n) worst case with all collisions; unbalanced BST degrades to O(n)
                </think>

                ## Hash Tables vs Binary Search Trees

                | Operation | Hash Table (avg) | Hash Table (worst) | Balanced BST |
                |-----------|-----------------|-------------------|--------------|
                | Search | O(1) | O(n) | O(log n) |
                | Insert | O(1) | O(n) | O(log n) |
                | Delete | O(1) | O(n) | O(log n) |
                | Min/Max | O(n) | O(n) | O(log n) |
                | Ordered traversal | O(n log n) | O(n log n) | O(n) |
                | Range query | O(n) | O(n) | O(log n + k) |

                **Use Hash Tables when:**
                - You need fastest average-case lookups
                - Order doesn't matter
                - Key equality checks are sufficient
                - Examples: caching, counting frequencies, deduplication

                **Use BSTs when:**
                - You need ordered operations (min, max, successor, predecessor)
                - You need range queries efficiently
                - You need guaranteed O(log n) worst case (with balancing)
                - Examples: database indexing, interval scheduling, order statistics

                ```python
                # Hash table (Python dict) — O(1) average lookup
                freq = {}
                for char in "hello":
                    freq[char] = freq.get(char, 0) + 1
                # freq = {'h': 1, 'e': 1, 'l': 2, 'o': 1}

                # BST — maintains sorted order
                from sortedcontainers import SortedDict
                sd = SortedDict()
                sd['banana'] = 2
                sd['apple'] = 5
                sd['cherry'] = 1
                print(list(sd.keys()))  # ['apple', 'banana', 'cherry'] — sorted!
                ```

                **Practical note:** In most interview and competitive programming contexts, hash tables (Python dicts) are the default choice unless you specifically need ordering.""").strip(),
        },
        {
            "instruction": "What is the difference between a stack and a recursion call stack? How are they related?",
            "context": "A stack is a LIFO data structure. The call stack is the system's stack that tracks function calls during program execution. Recursion uses the call stack implicitly.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: Concept Relationship — Stack and Recursion
                Strategy: Explain the data structure stack, the system call stack, and how recursion implicitly uses a stack. Show how iterative stack-based solutions mirror recursive ones.
                Complexity: Stack operations O(1); recursion depth = O(stack depth), risk of stack overflow
                Edge cases: Maximum recursion depth limits (Python default 1000), stack overflow with deep recursion
                </think>

                ## Stack vs Recursion Call Stack

                **The Data Structure Stack:**
                A stack is a LIFO (Last-In, First-Out) collection with push, pop, and peek operations, all O(1).

                **The System Call Stack:**
                When a function calls another function (or itself), the system pushes a "stack frame" onto the call stack containing:
                - Return address
                - Local variables
                - Function parameters

                **The Connection:**
                Every recursive algorithm implicitly uses the system call stack. This is why any recursive solution can be converted to an iterative one using an explicit stack.

                ```python
                # Recursive DFS (uses call stack implicitly)
                def dfs_recursive(node, visited=None):
                    if visited is None:
                        visited = set()
                    visited.add(node)
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            dfs_recursive(neighbor, visited)

                # Iterative DFS (uses explicit stack — same behavior!)
                def dfs_iterative(start):
                    visited = set()
                    stack = [start]
                    while stack:
                        node = stack.pop()
                        if node in visited:
                            continue
                        visited.add(node)
                        for neighbor in graph[node]:
                            if neighbor not in visited:
                                stack.append(neighbor)
                ```

                **Why this matters:**
                - Python has a recursion limit (~1000 by default)
                - Deep recursion causes stack overflow
                - Converting to iterative with explicit stack avoids this
                - Understanding this connection helps you solve tree/graph problems both ways""").strip(),
        },
        {
            "instruction": "Compare memoization (top-down) and tabulation (bottom-up) approaches in dynamic programming.",
            "context": "Memoization starts from the original problem and caches subproblem results as they are computed recursively. Tabulation builds up solutions iteratively from the smallest subproblems.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: DP Implementation Comparison
                Strategy: Compare memoization and tabulation on implementation style, performance, space usage, and when to prefer each.
                Complexity: Both have same time complexity for a given problem; tabulation may have better constants due to no recursion overhead
                Edge cases: Memoization can hit recursion limits; tabulation may compute unnecessary subproblems
                </think>

                ## Memoization vs Tabulation

                | Aspect | Memoization (Top-Down) | Tabulation (Bottom-Up) |
                |--------|----------------------|----------------------|
                | Direction | Start from problem, recurse down | Start from base cases, build up |
                | Implementation | Recursive + cache | Iterative + table |
                | Subproblems computed | Only those needed | All subproblems |
                | Stack usage | O(depth) call stack | O(1) |
                | Code clarity | Often more intuitive | Can be harder to derive |
                | Space optimization | Harder | Easier (rolling array) |

                ```python
                # Memoization (Top-Down) — Fibonacci
                from functools import lru_cache

                @lru_cache(maxsize=None)
                def fib_memo(n):
                    if n <= 1:
                        return n
                    return fib_memo(n - 1) + fib_memo(n - 2)

                # Tabulation (Bottom-Up) — Fibonacci
                def fib_tab(n):
                    if n <= 1:
                        return n
                    dp = [0] * (n + 1)
                    dp[1] = 1
                    for i in range(2, n + 1):
                        dp[i] = dp[i-1] + dp[i-2]
                    return dp[n]

                # Space-Optimized Tabulation
                def fib_opt(n):
                    if n <= 1:
                        return n
                    a, b = 0, 1
                    for _ in range(2, n + 1):
                        a, b = b, a + b
                    return b
                ```

                **When to use Memoization:**
                - The recursive structure is natural and easy to write
                - Not all subproblems need to be solved
                - You want quick prototyping

                **When to use Tabulation:**
                - You need to avoid recursion limits
                - You want to optimize space (rolling array technique)
                - The iteration order is clear
                - Production code where stack overflow is a concern

                **Recommendation:** Start with memoization to verify correctness (it's easier to write), then convert to tabulation if needed for performance or space.""").strip(),
        },
        {
            "instruction": "Compare adjacency matrix and adjacency list for graph representation. When should I use each?",
            "context": "An adjacency matrix is a 2D array of size V×V where entry [i][j] indicates an edge between vertices i and j. An adjacency list stores a list of neighbors for each vertex.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: Graph Representation Comparison
                Strategy: Compare adjacency matrix and adjacency list on space, edge lookup, iteration, and suitability for sparse vs dense graphs.
                Complexity: Matrix: O(V²) space, O(1) edge lookup; List: O(V+E) space, O(degree) edge lookup
                Edge cases: Self-loops, parallel edges, weighted vs unweighted, directed vs undirected
                </think>

                ## Adjacency Matrix vs Adjacency List

                | Aspect | Adjacency Matrix | Adjacency List |
                |--------|-----------------|----------------|
                | Space | O(V²) | O(V + E) |
                | Check edge exists | O(1) | O(degree) |
                | Get all neighbors | O(V) | O(degree) |
                | Add edge | O(1) | O(1) |
                | Remove edge | O(1) | O(degree) |
                | Best for | Dense graphs (E ≈ V²) | Sparse graphs (E << V²) |

                **Use Adjacency Matrix when:**
                - Graph is dense (many edges relative to vertices)
                - You need O(1) edge existence checks
                - Floyd-Warshall all-pairs shortest path
                - Small number of vertices

                **Use Adjacency List when:**
                - Graph is sparse (few edges relative to vertices)
                - You need to iterate over neighbors efficiently (BFS/DFS)
                - Memory is a constraint
                - Most real-world graphs (social networks, web graphs)

                ```python
                # Adjacency Matrix
                V = 5
                matrix = [[0] * V for _ in range(V)]
                matrix[0][1] = 1  # edge from 0 to 1
                matrix[1][0] = 1  # undirected: add reverse
                # Check edge: matrix[i][j] — O(1)

                # Adjacency List
                from collections import defaultdict
                adj = defaultdict(list)
                adj[0].append(1)  # edge from 0 to 1
                adj[1].append(0)  # undirected: add reverse
                # Get neighbors: adj[node] — O(1) to access, O(degree) to iterate
                ```

                **Practical note:** For most DSA problems and interviews, adjacency lists are the standard choice because most graphs are sparse. Use `defaultdict(list)` in Python for clean implementation.""").strip(),
        },
    ]
    return comparisons


def generate_hint_pairs() -> List[Dict]:
    """Generate training pairs for the HINT intent type."""
    hints = [
        {
            "instruction": "I'm stuck on the two sum problem. Can you give me a hint?",
            "context": "The two sum problem asks to find two numbers in an array that add up to a target value.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: Hash Map Lookup
                Strategy: Student is stuck, provide progressive hints without revealing the full solution. Guide them toward using a hash map.
                Complexity: O(n) time with hash map, O(n) space
                Edge cases: Duplicate values, target that requires same element twice
                </think>

                Here are some progressive hints:

                **Hint 1:** The brute force approach checks every pair — that's O(n²). Can you do better by avoiding repeated work?

                **Hint 2:** For each number, you know what its complement would be (target - number). How can you quickly check if you've seen that complement before?

                **Hint 3:** Think about using a data structure that gives you O(1) lookups...

                **Hint 4:** As you iterate through the array, store each number in a hash map. For each new number, check if (target - current number) already exists in your map.

                Try implementing this approach! The key insight is trading space for time — using O(n) extra space to reduce time from O(n²) to O(n).""").strip(),
        },
        {
            "instruction": "How should I approach finding if a linked list has a cycle? Give me a hint, don't solve it for me.",
            "context": "Cycle detection in a linked list requires determining if any node is visited twice during traversal.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: Fast and Slow Pointers (Floyd's Algorithm)
                Strategy: Give progressive hints toward the two-pointer approach without revealing the full implementation.
                Complexity: O(n) time, O(1) space
                Edge cases: Empty list, single node, cycle at the head
                </think>

                Great that you want to think it through! Here are progressive hints:

                **Hint 1:** The naive approach uses a hash set to track visited nodes — O(n) time AND O(n) space. Can you detect a cycle with O(1) space?

                **Hint 2:** Imagine two people walking on a circular track at different speeds. What eventually happens?

                **Hint 3:** Use two pointers moving at different speeds through the list. What happens when both are inside a cycle?

                **Hint 4:** The slow pointer moves 1 step at a time, the fast pointer moves 2 steps. If there's a cycle, they must eventually meet (like runners on a track). If the fast pointer reaches null, there's no cycle.

                Try coding it! The key question to answer: why must the two pointers meet if there's a cycle?""").strip(),
        },
        {
            "instruction": "I need to find the longest increasing subsequence. Where do I start?",
            "context": "The longest increasing subsequence (LIS) problem asks for the longest subsequence where elements are in strictly increasing order. The subsequence doesn't need to be contiguous.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: Dynamic Programming — LIS
                Strategy: Guide the student from brute force thinking toward the DP approach, then mention the O(n log n) optimization.
                Complexity: O(n²) DP, O(n log n) with patience sorting + binary search
                Edge cases: All elements same, strictly decreasing, single element, already sorted
                </think>

                Let me guide you step by step:

                **Step 1 — Understand the problem:** Unlike subarray (contiguous), subsequence elements don't need to be adjacent. [3,1,4,1,5,9] has LIS [1,4,5,9] of length 4.

                **Step 2 — Think about each element:** For each element, what's the longest increasing subsequence ending at that element?

                **Step 3 — The DP relationship:** If you know the LIS length ending at every previous element, and a previous element is smaller than the current one, what can you say about the LIS ending at the current element?

                **Step 4:** `dp[i]` = length of LIS ending at index i. For each `j < i`, if `arr[j] < arr[i]`, then `dp[i] = max(dp[i], dp[j] + 1)`.

                **Bonus challenge:** The DP solution is O(n²). There's an O(n log n) approach using a clever "tails" array and binary search. Try the DP version first, then research "patience sorting" for the optimized version!""").strip(),
        },
    ]
    return hints


def generate_conceptual_pairs() -> List[Dict]:
    """Generate conceptual understanding training pairs."""
    concepts = [
        {
            "instruction": "What is amortized analysis and why does it matter for dynamic arrays?",
            "context": "Amortized analysis averages the time per operation over a worst-case sequence of operations. It gives a tighter bound than worst-case analysis for data structures with occasional expensive operations.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: Complexity Analysis Concept
                Strategy: Explain amortized analysis with the dynamic array (ArrayList) resizing example, using the accounting method or aggregate method.
                Complexity: Individual insert can be O(n) when resizing, but amortized O(1) per insert
                Edge cases: Understanding that amortized ≠ average case; it's a worst-case guarantee over sequences
                </think>

                ## Amortized Analysis

                Amortized analysis gives the average performance of each operation in the **worst-case sequence**, not the average case.

                ### Dynamic Array Example

                When you `append()` to a Python list:
                - **Usually:** O(1) — just place element at the end
                - **Sometimes:** O(n) — array is full, allocate 2x space, copy all elements

                The key insight: the expensive O(n) copy happens so rarely that it "pays for" all the cheap O(1) appends.

                **Proof (Aggregate Method):**
                - Start with capacity 1. After n appends, copies happen at sizes 1, 2, 4, 8, ..., n
                - Total copy cost: 1 + 2 + 4 + ... + n = 2n - 1
                - Total cost for n operations: n (placements) + 2n (copies) = 3n
                - Amortized cost per operation: 3n / n = **O(1)**

                ```python
                # Python list handles this automatically
                arr = []
                for i in range(1000000):
                    arr.append(i)  # Amortized O(1) per append
                # Total time: O(n), not O(n²)!
                ```

                **Why it matters:**
                - Without amortized analysis, you'd think dynamic arrays are O(n) per insert
                - Amortized O(1) means dynamic arrays are practical for building lists
                - Same concept applies to: hash table resizing, splay trees, union-find

                **Important:** Amortized ≠ Average case. Average case assumes random input. Amortized is the worst-case cost averaged over any sequence of operations.""").strip(),
        },
        {
            "instruction": "Explain the concept of stable vs unstable sorting algorithms with examples.",
            "context": "A stable sorting algorithm preserves the relative order of elements with equal keys. An unstable algorithm may change their relative order.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: Sorting Property — Stability
                Strategy: Define stability, show a concrete example where it matters, and classify common sorting algorithms.
                Complexity: Stability doesn't affect time complexity — it's a property of the algorithm's behavior
                Edge cases: When all elements are equal (trivially stable); when you sort by multiple keys
                </think>

                ## Stable vs Unstable Sorting

                **Stable sort:** Elements with equal keys maintain their original relative order.
                **Unstable sort:** Equal elements may be rearranged.

                ### Why Stability Matters

                Consider sorting students by grade, where they're already sorted by name:
                ```
                Before: [(Alice, A), (Bob, B), (Charlie, A), (Diana, B)]
                ```

                **Stable sort by grade:**
                ```
                [(Alice, A), (Charlie, A), (Bob, B), (Diana, B)]
                — A's keep name order ✓, B's keep name order ✓
                ```

                **Unstable sort by grade:**
                ```
                [(Charlie, A), (Alice, A), (Diana, B), (Bob, B)]
                — A's and B's may be rearranged ✗
                ```

                ### Classification

                | Algorithm | Stable? | Time (avg) | Space |
                |-----------|---------|-----------|-------|
                | Merge Sort | ✅ Yes | O(n log n) | O(n) |
                | Insertion Sort | ✅ Yes | O(n²) | O(1) |
                | Bubble Sort | ✅ Yes | O(n²) | O(1) |
                | Tim Sort | ✅ Yes | O(n log n) | O(n) |
                | Quick Sort | ❌ No | O(n log n) | O(log n) |
                | Heap Sort | ❌ No | O(n log n) | O(1) |
                | Selection Sort | ❌ No | O(n²) | O(1) |

                **Python's `sorted()` and `.sort()` use TimSort — which is stable!**

                ```python
                # Multi-key sorting relies on stability
                students = [("Bob", 85), ("Alice", 92), ("Charlie", 85)]

                # Sort by score — Python's sort is stable, so equal scores keep name order
                students.sort(key=lambda x: x[1])
                # [("Bob", 85), ("Charlie", 85), ("Alice", 92)]
                ```""").strip(),
        },
        {
            "instruction": "What is the difference between a tree, a binary tree, and a binary search tree?",
            "context": "Trees are hierarchical data structures with nodes and edges. Binary trees restrict each node to at most two children. Binary search trees add an ordering property.",
            "response": textwrap.dedent("""\
                <think>
                Pattern: Data Structure Hierarchy — Trees
                Strategy: Progressively add constraints from general tree → binary tree → BST, showing what each constraint enables.
                Complexity: BST operations O(h) where h is height; balanced BST O(log n); degenerate BST O(n)
                Edge cases: Empty tree, single node, skewed BST (degenerates to linked list)
                </think>

                ## Tree → Binary Tree → Binary Search Tree

                Each is a specialization of the previous:

                ### 1. Tree (General)
                - Each node can have **any number of children**
                - No ordering requirement
                - Examples: file systems, HTML DOM, organization charts
                - Used in: Trie (prefix tree), N-ary trees

                ### 2. Binary Tree
                - Each node has **at most 2 children** (left and right)
                - No ordering requirement on values
                - Properties: max nodes at level i = 2^i, height h tree has at most 2^(h+1)-1 nodes
                - Used in: expression trees, Huffman coding

                ### 3. Binary Search Tree (BST)
                - Binary tree WITH ordering: **left < root < right** for every node
                - This ordering enables efficient search, insert, delete
                - Operations: O(h) where h = height

                ```python
                # Binary Tree Node (no ordering)
                class TreeNode:
                    def __init__(self, val=0, left=None, right=None):
                        self.val = val
                        self.left = left
                        self.right = right

                # BST Search — O(h) time
                def search_bst(root, target):
                    if not root:
                        return None
                    if target == root.val:
                        return root
                    elif target < root.val:
                        return search_bst(root.left, target)
                    else:
                        return search_bst(root.right, target)

                # BST Insert — O(h) time
                def insert_bst(root, val):
                    if not root:
                        return TreeNode(val)
                    if val < root.val:
                        root.left = insert_bst(root.left, val)
                    else:
                        root.right = insert_bst(root.right, val)
                    return root
                ```

                | Feature | General Tree | Binary Tree | BST |
                |---------|-------------|-------------|-----|
                | Max children | Unlimited | 2 | 2 |
                | Ordering | None | None | Left < Root < Right |
                | Search | O(n) | O(n) | O(h) = O(log n) balanced |
                | Inorder traversal | N/A | Any order | Sorted order! |

                **Key insight:** BST's ordering property gives us O(log n) search (when balanced), similar to binary search on a sorted array, but with O(log n) insert/delete too.""").strip(),
        },
    ]
    return concepts


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_records: list[dict] = []

    # 1. Parse corpus files
    print("=" * 60)
    print("Building training data from RAG corpus")
    print("=" * 60)

    for filename, topic in TOPIC_MAP.items():
        filepath = CORPUS_DIR / filename
        if not filepath.exists():
            print(f"  SKIP {filename} (not found)")
            continue

        topic_display = topic.replace("_", " ").title()
        records = process_file(filepath, topic, topic_display)
        print(f"  {filename}: {len(records)} training pairs")
        all_records.extend(records)

    # 2. Add hand-crafted comparison pairs
    comparisons = generate_comparison_pairs()
    print(f"  + {len(comparisons)} comparison pairs")
    all_records.extend(comparisons)

    # 3. Add hint-style pairs
    hints = generate_hint_pairs()
    print(f"  + {len(hints)} hint pairs")
    all_records.extend(hints)

    # 4. Add conceptual pairs
    concepts = generate_conceptual_pairs()
    print(f"  + {len(concepts)} conceptual pairs")
    all_records.extend(concepts)

    # 5. Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print(f"Total: {len(all_records)} training pairs → {OUTPUT_FILE}")

    # Stats
    response_lens = [len(r["response"]) for r in all_records]
    avg_len = sum(response_lens) / len(response_lens) if response_lens else 0
    print(f"Average response length: {avg_len:.0f} chars")
    print(f"Min response: {min(response_lens)} chars")
    print(f"Max response: {max(response_lens)} chars")


if __name__ == "__main__":
    main()
