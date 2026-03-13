"""
Domain Validator - ensures queries are related to Data Structures & Algorithms.

Provides keyword/pattern-based validation so DSA-only questions are answered.
Off-topic requests are rejected with a polite refusal.  System prompts and
internal state are never leaked.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Set

logger = logging.getLogger(__name__)

# ── DSA keywords grouped by category ─────────────────────────────────────────

_DATA_STRUCTURES: Set[str] = {
    "array", "linked list", "stack", "queue", "deque", "tree", "binary tree",
    "bst", "binary search tree", "avl", "red-black", "b-tree", "trie",
    "heap", "min-heap", "max-heap", "priority queue", "hash table",
    "hash map", "hash set", "dictionary", "graph", "adjacency list",
    "adjacency matrix", "disjoint set", "union-find", "segment tree",
    "fenwick tree", "bloom filter", "skip list", "splay tree",
    "linked-list", "doubly linked", "circular list", "matrix",
}

_ALGORITHMS: Set[str] = {
    "sort", "sorting", "bubble sort", "merge sort", "quick sort", "heap sort",
    "insertion sort", "selection sort", "radix sort", "counting sort",
    "bucket sort", "tim sort", "topological sort",
    "search", "searching", "binary search", "linear search", "dfs",
    "bfs", "depth-first", "breadth-first", "dijkstra", "bellman-ford",
    "floyd-warshall", "kruskal", "prim", "a-star", "a*",
    "dynamic programming", "dp", "memoization", "tabulation",
    "greedy", "divide and conquer", "backtracking", "recursion",
    "sliding window", "two pointer", "two-pointer", "fast slow pointer",
    "kadane", "knapsack", "lcs", "lis", "edit distance",
    "shortest path", "minimum spanning tree", "mst",
}

_CONCEPTS: Set[str] = {
    "big-o", "big o", "time complexity", "space complexity", "asymptotic",
    "amortized", "worst case", "best case", "average case",
    "in-place", "stable sort", "comparison sort", "recursion tree",
    "recurrence relation", "master theorem",
    "traversal", "inorder", "preorder", "postorder", "level order",
    "balancing", "rotation", "hashing", "collision", "load factor",
    "rehashing", "open addressing", "chaining",
    "cycle detection", "topological order", "strongly connected",
    "connected component", "bipartite", "eulerian", "hamiltonian",
}

_CODING_PATTERNS: Set[str] = {
    "leetcode", "competitive programming", "coding interview",
    "algorithm", "data structure", "dsa", "pointer", "index",
    "subarray", "subsequence", "substring", "permutation", "combination",
    "palindrome", "anagram", "prefix sum", "suffix array",
    "monotonic stack", "monotonic queue", "bit manipulation",
    "interview", "problem", "data structures and algorithms",
}

ALL_DSA_KEYWORDS: Set[str] = _DATA_STRUCTURES | _ALGORITHMS | _CONCEPTS | _CODING_PATTERNS

# Split into single-word (\b-safe) and multi-word (phrase match) sets
_SINGLE_WORD_KEYWORDS: Set[str] = {kw for kw in ALL_DSA_KEYWORDS if " " not in kw}
_MULTI_WORD_KEYWORDS: Set[str] = {kw for kw in ALL_DSA_KEYWORDS if " " in kw}

# Pre-compile a single pattern from single-word keywords only (word-boundary anchored)
_KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in sorted(_SINGLE_WORD_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Blocklist patterns - attempts to extract system prompts or jailbreak
_BLOCKLIST_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(previous|all|above)\s+(instructions?|prompts?)",
        r"what\s+(is|are)\s+your\s+(system|initial)\s+(prompt|instruction)",
        r"reveal\s+(your|the)\s+(system|hidden)\s+(prompt|instruction)",
        r"repeat\s+(the|your)\s+(system|initial)\s+(prompt|message)",
        r"act\s+as\s+(?!a\s+(dsa|algorithm|data\s+structure))",
        r"you\s+are\s+now\s+a",
        r"pretend\s+(you|to)\s+(are|be)\s+(?!.*tutor)",
    ]
]

REFUSAL_MESSAGE = (
    "I'm a DSA (Data Structures & Algorithms) tutor. "
    "I can only help with topics like arrays, trees, graphs, sorting, "
    "dynamic programming, and other DSA concepts. "
    "Please ask a DSA-related question!"
)

GREETING_RESPONSE = (
    "Hello! I'm your **DSA Tutor**, specialized in Data Structures and Algorithms "
    "for coding interview preparation.\n\n"
    "I can help you with:\n"
    "- **Data Structures**: Arrays, Linked Lists, Trees, Graphs, Hash Tables, Heaps, Tries\n"
    "- **Algorithms**: Sorting, Searching, Dynamic Programming, Greedy, Backtracking\n"
    "- **Techniques**: Two Pointers, Sliding Window, BFS/DFS, Divide & Conquer\n"
    "- **Analysis**: Time/Space Complexity, Big-O notation\n\n"
    "Try asking me something like:\n"
    "- *\"Explain the two-pointer technique with an example\"*\n"
    "- *\"How does Dijkstra's algorithm work?\"*\n"
    "- *\"Solve the two-sum problem\"*\n"
    "- *\"Compare BFS vs DFS\"*\n\n"
    "What would you like to learn about?"
)

VAGUE_QUERY_RESPONSE = (
    "I'd love to help! Could you be more specific about what you'd like to learn?\n\n"
    "I specialize in **Data Structures and Algorithms**. You can ask me about:\n"
    "- Any data structure (arrays, trees, graphs, hash tables, etc.)\n"
    "- Any algorithm (sorting, searching, dynamic programming, etc.)\n"
    "- Coding techniques (two pointers, sliding window, etc.)\n"
    "- Complexity analysis (Big-O, time/space trade-offs)\n\n"
    "For example, try: *\"What is a binary search tree?\"* or *\"Explain dynamic programming\"*"
)

# Pattern to detect greetings
_GREETING_PATTERN = re.compile(
    r"^\s*(h(i|ello|ey|owdy)|good\s+(morning|afternoon|evening)|"
    r"greetings|sup|yo|what'?s?\s+up|hey(\s+there)?|hola|namaste|"
    r"thanks?(\s+you)?|thank\s+you)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

# Pattern to detect question-word starts
_QUESTION_WORD_PATTERN = re.compile(
    r"^\s*(what|how|why|when|where|which|who|explain|describe|tell|"
    r"define|can|could|show|teach|is|are|do|does|give|list|name|walk)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationResult:
    """Result of domain validation."""
    is_valid: bool
    matched_keywords: list[str]
    reason: str


def validate_query(query: str) -> ValidationResult:
    """
    Check whether a user query is DSA-related.

    Returns ValidationResult with is_valid=True if the query passes,
    or is_valid=False with a reason if it's off-topic or blocked.

    Special reasons:
      - "greeting"  → the query is a greeting (hi, hello, etc.)
      - "too_vague" → too short and ambiguous to process
    """
    # Check blocklist first
    for pattern in _BLOCKLIST_PATTERNS:
        if pattern.search(query):
            logger.warning("Blocked query (prompt injection attempt): %.80s...", query)
            return ValidationResult(
                is_valid=False,
                matched_keywords=[],
                reason="blocked",
            )

    # Check for greetings — handled with a pre-built friendly response
    if _GREETING_PATTERN.match(query.strip()):
        logger.info("Greeting detected: %.40s", query)
        return ValidationResult(
            is_valid=False,
            matched_keywords=[],
            reason="greeting",
        )

    # Find DSA keyword matches
    q_lower = query.lower()
    regex_matches = _KEYWORD_PATTERN.findall(query)
    phrase_matches = [kw for kw in _MULTI_WORD_KEYWORDS if kw in q_lower]
    all_matches = list(set(m.lower() for m in regex_matches)) + list(set(phrase_matches))
    unique_matches = list(set(all_matches))

    if unique_matches:
        return ValidationResult(
            is_valid=True,
            matched_keywords=unique_matches,
            reason="dsa_match",
        )

    # Queries starting with question words → likely an educational question
    word_count = len(query.split())
    if word_count >= 2 and _QUESTION_WORD_PATTERN.match(query.strip()):
        logger.info("Question-word query allowed without keywords: %.60s...", query)
        return ValidationResult(
            is_valid=True,
            matched_keywords=[],
            reason="question_allowed",
        )

    # Very short queries with no DSA keywords or question pattern → too vague
    if word_count < 3:
        return ValidationResult(
            is_valid=False,
            matched_keywords=[],
            reason="too_vague",
        )

    # Longer queries without explicit keywords — allow with low confidence
    # (the model's system prompt will still constrain the answer)
    logger.info("No DSA keywords found but allowing long query: %.60s...", query)
    return ValidationResult(
        is_valid=True,
        matched_keywords=[],
        reason="allowed_long_query",
    )


# ── Topic auto-detection ────────────────────────────────────────────────────────────────

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "arrays": [
        "array", "subarray", "two pointer", "two-pointer", "sliding window",
        "kadane", "prefix sum", "matrix", "contiguous",
    ],
    "linked_lists": [
        "linked list", "linked-list", "doubly linked", "circular list",
        "lru cache", "singly linked",
    ],
    "stacks_queues": [
        "stack", "queue", "deque", "monotonic stack", "monotonic queue",
        "parentheses", "bracket", "push", "pop",
    ],
    "hash_tables": [
        "hash map", "hash table", "hash set", "dictionary", "collision",
        "hashing", "load factor",
    ],
    "trees": [
        "tree", "binary tree", "bst", "binary search tree", "avl", "trie",
        "heap", "min-heap", "max-heap", "traversal", "inorder", "preorder",
        "postorder", "level order", "segment tree", "fenwick",
    ],
    "graphs": [
        "graph", "bfs", "dfs", "dijkstra", "bellman", "shortest path",
        "topological", "adjacency", "cycle detection", "mst", "kruskal",
        "prim", "connected component", "bipartite",
    ],
    "sorting": [
        "sort", "sorting", "merge sort", "quick sort", "bubble sort",
        "heap sort", "binary search", "counting sort", "radix sort",
    ],
    "dynamic_programming": [
        "dynamic programming", "dp", "memoization", "tabulation",
        "knapsack", "lcs", "lis", "coin change", "edit distance",
    ],
    "recursion": [
        "recursion", "recursive", "backtracking", "base case",
        "permutation", "combination", "n-queens",
    ],
    "strings": [
        "string", "substring", "subsequence", "palindrome", "anagram",
        "kmp", "pattern matching", "suffix",
    ],
}


def detect_topic(question: str) -> str:
    """Auto-detect the DSA topic from the question text.

    Returns the best-matching topic name, or 'dsa' as a general fallback.
    """
    q_lower = question.lower()
    scores: dict[str, int] = {}
    for topic, keywords in _TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q_lower)
        if score > 0:
            scores[topic] = score
    if not scores:
        return "dsa"
    best = max(scores, key=scores.get)
    logger.debug("Auto-detected topic: %s (score=%d)", best, scores[best])
    return best
