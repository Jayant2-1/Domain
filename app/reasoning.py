"""
Reasoning Pipeline - single-pass inference that produces structured "thinking"
before a polished final answer, similar to DeepSeek's thinking mode.

The pipeline uses a RAG-first approach: high-quality corpus content is used
as the backbone of answers. The LLM's job is to teach/explain the material
using Socratic methods, not to generate content from scratch.

The pipeline is stateless - all state flows through function arguments.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ── Regex to split <think>...</think> from the rest ─────────────────────────

_THINK_PATTERN = re.compile(
    r"<think>\s*(.*?)\s*</think>\s*(.*)",
    re.DOTALL,
)

# Pattern to detect "no code" requests
_NO_CODE_PATTERN = re.compile(
    r"\b(?:no\s+code|without\s+code|don'?t\s+(?:include|give|show|write)\s+code|only\s+explain|explanation\s+only|no\s+implementation)\b",
    re.IGNORECASE,
)

# Pattern to strip code blocks from text
_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.DOTALL)


def _strip_code_from_rag(text: str) -> str:
    """Remove code blocks from RAG content, keeping only prose/explanations."""
    stripped = _CODE_BLOCK_PATTERN.sub("", text)
    # Clean up excess whitespace left by removed blocks
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def _user_wants_no_code(question: str, history: list[dict[str, str]] | None = None) -> bool:
    """Check if the user explicitly asked for no code."""
    if _NO_CODE_PATTERN.search(question):
        return True
    # Also check recent history for follow-up "no code" requests
    if history:
        for msg in history[-3:]:
            if msg.get("role") == "user" and _NO_CODE_PATTERN.search(msg.get("content", "")):
                return True
    return False


# ── Data structures ──────────────────────────────────────────────────────────

class IntentType(Enum):
    """Detected intent of the user's DSA question."""
    SOLVE = "solve"
    EXPLAIN = "explain"
    HINT = "hint"
    COMPARE = "compare"
    ANALYZE = "analyze"


@dataclass(frozen=True)
class ReasoningResult:
    """Complete output from the reasoning pipeline."""
    analysis: str
    answer: str
    intent: IntentType
    thinking_time: float
    total_time: float
    topic: str
    rag_used: bool
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "analysis": self.analysis,
            "answer": self.answer,
            "intent": self.intent.value,
            "thinking_time": round(self.thinking_time, 2),
            "total_time": round(self.total_time, 2),
            "topic": self.topic,
            "rag_used": self.rag_used,
            "metadata": self.metadata,
        }


# ── Intent detection ─────────────────────────────────────────────────────────

_INTENT_KEYWORDS: dict[IntentType, list[str]] = {
    IntentType.SOLVE: [
        "solve", "implement", "code", "write", "solution", "program",
        "find", "compute", "calculate", "return", "output", "function",
        "how to do", "how do i", "how would you",
    ],
    IntentType.EXPLAIN: [
        "explain", "what is", "what are", "how does", "why", "describe",
        "tell me about", "walk me through", "understand", "concept",
        "meaning", "definition", "tutorial",
    ],
    IntentType.HINT: [
        "hint", "clue", "tip", "nudge", "help me think", "guide",
        "approach", "direction", "where to start", "stuck",
    ],
    IntentType.COMPARE: [
        "compare", "difference", "vs", "versus", "better", "worse",
        "trade-off", "tradeoff", "pros and cons", "when to use",
        "which is", "prefer",
    ],
    IntentType.ANALYZE: [
        "complexity", "big o", "time complexity", "space complexity",
        "analyze", "analysis", "efficient", "optimise", "optimize",
        "performance", "benchmark", "worst case", "best case",
    ],
}


def detect_intent(question: str) -> IntentType:
    """Detect the user's intent from their question text."""
    q_lower = question.lower()
    scores: dict[IntentType, int] = {intent: 0 for intent in IntentType}

    # Detect negated keywords (e.g. "no code", "without code", "don't code")
    negated = set()
    neg_pattern = re.compile(r"(?:no|without|don'?t|not|skip|avoid)\s+(\w+)")
    for m in neg_pattern.finditer(q_lower):
        negated.add(m.group(1))

    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                # Skip if the keyword is negated
                if any(neg_word in kw.split() for neg_word in negated):
                    continue
                scores[intent] += 1

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] == 0:
        return IntentType.SOLVE

    logger.debug("Intent detected: %s (score=%d)", best.value, scores[best])
    return best


# ── Output parser ────────────────────────────────────────────────────────────

def parse_thinking_output(raw_text: str) -> tuple[str, str]:
    """
    Parse model output containing <think>...</think> tags.

    Returns (analysis, answer).
    If no tags found, the entire text is treated as the answer.
    """
    match = _THINK_PATTERN.search(raw_text)
    if match:
        analysis = match.group(1).strip()
        answer = match.group(2).strip()
        return analysis, answer

    # Fallback: no <think> tags - whole output is the answer
    logger.warning("No <think> tags found in model output, using full text as answer")
    return "", raw_text.strip()


# ── Socratic follow-up questions ─────────────────────────────────────────────

_SOCRATIC_QUESTIONS: dict[str, list[str]] = {
    "arrays": [
        "What happens to the time complexity if the array is already sorted?",
        "Can you think of a case where this approach would fail?",
        "How would you modify this to handle duplicate elements?",
    ],
    "linked_lists": [
        "Why is a linked list better than an array for frequent insertions?",
        "What is the trade-off between singly and doubly linked lists?",
        "How would you detect if a linked list has a cycle?",
    ],
    "stacks_queues": [
        "Can you think of a real-world scenario that behaves like a stack?",
        "When would you choose a queue over a stack?",
        "How would you implement a stack using two queues?",
    ],
    "hash_tables": [
        "What happens when two keys hash to the same index?",
        "Why is the load factor important for hash table performance?",
        "When would a hash table be a poor choice?",
    ],
    "trees": [
        "What is the difference between a balanced and unbalanced BST?",
        "Why does tree height matter for search performance?",
        "How would you find the lowest common ancestor of two nodes?",
    ],
    "graphs": [
        "When would you choose BFS over DFS, or vice versa?",
        "How do you detect a cycle in a directed graph?",
        "What makes Dijkstra's algorithm fail with negative edge weights?",
    ],
    "sorting": [
        "Why is quicksort generally faster than mergesort in practice?",
        "When would you prefer a stable sort over an unstable one?",
        "Can you sort faster than O(n log n)? Under what conditions?",
    ],
    "dynamic_programming": [
        "How do you identify whether a problem has overlapping subproblems?",
        "What is the difference between top-down and bottom-up DP?",
        "Can you spot the optimal substructure in this problem?",
    ],
    "recursion": [
        "What is the base case here, and why is it important?",
        "How would you convert this recursive solution to an iterative one?",
        "What is the maximum recursion depth, and could it cause a stack overflow?",
    ],
    "strings": [
        "How does the approach change if the string contains Unicode characters?",
        "What is the difference between a substring and a subsequence?",
        "How would you optimize this for very long strings?",
    ],
}


def _get_socratic_questions(topic: str, question: str) -> str:
    """Return 2 Socratic follow-up questions for the topic."""
    questions = _SOCRATIC_QUESTIONS.get(topic, [])
    if not questions:
        questions = [
            "What edge cases should you consider for this problem?",
            "Can you think of a more efficient approach?",
            "How would you test your solution?",
        ]
    # Pick 2 questions (rotate based on question hash to vary)
    idx = hash(question) % max(1, len(questions) - 1)
    selected = questions[idx:idx + 2]
    if len(selected) < 2 and questions:
        selected += questions[:2 - len(selected)]
    return "\n".join(f"- *{q}*" for q in selected[:2])


# ── Quality gate — detect garbage outputs ────────────────────────────────────

def _is_garbage_output(answer: str) -> bool:
    """Detect if the model output is garbage (parroting instructions, too short, etc.)."""
    if len(answer) < 30:
        return True
    lower = answer.lower()
    # Model is parroting format instructions back
    parrot_phrases = [
        "follow these rules",
        "follow the rules above",
        "state time and space complexities clearly",
        "concise definition",
        "bullet list of applicable scenarios",
        "numbered step-by-step walkthrough",
        "time o(…)",
        "space o(…)",
        "complete, correct, well-commented code",
    ]
    parrot_count = sum(1 for p in parrot_phrases if p in lower)
    if parrot_count >= 2:
        return True
    return False


def _build_explanation_from_rag(rag_context: str, question: str, topic: str) -> str:
    """Build a text-only explanation from RAG content (no code blocks)."""
    # Get the raw RAG content and strip code
    text_only = _strip_code_from_rag(rag_context)
    # Clean source prefixes
    text_only = re.sub(r"\[Source: [^\]]+\]\n", "", text_only)
    if len(text_only.strip()) < 50:
        return ""

    topic_display = topic.replace("_", " ").title()
    return (
        f"## {topic_display} — Conceptual Explanation\n\n"
        f"{text_only.strip()}\n"
    )


def _build_rag_fallback_answer(question: str, rag_context: str, topic: str) -> str:
    """Build a structured answer directly from RAG content when model output is garbage."""
    if not rag_context:
        return ""

    # Extract the most relevant RAG chunk
    chunks = rag_context.split("\n\n---\n\n")
    best_chunk = chunks[0] if chunks else rag_context

    # Clean source prefixes
    best_chunk = re.sub(r"\[Source: [^\]]+\]\n", "", best_chunk)

    # Build a structured answer from RAG content
    topic_display = topic.replace("_", " ").title()
    socratic = _get_socratic_questions(topic, question)

    answer = (
        f"{best_chunk}\n\n"
        f"---\n\n"
        f"**🤔 Think deeper:**\n{socratic}"
    )
    return answer


# ── Prompt builder (single pass) ─────────────────────────────────────────────

def build_reasoning_messages(
    question: str,
    topic: str,
    user_rating: float,
    rag_context: str,
    intent: IntentType,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build messages for single-pass reasoning. Kept short for 7B model."""
    topic_display = topic.replace("_", " ")

    if user_rating < 900:
        level = "beginner"
    elif user_rating < 1100:
        level = "intermediate"
    else:
        level = "advanced"

    # Short, focused system prompt — 7B models follow short instructions best
    system = (
        f"You are MLML, a friendly and expert Data Structures & Algorithms tutor. "
        f"You are helping a {level}-level student with {topic_display}.\n"
        f"Think step by step inside <think>...</think> tags, then write your detailed answer after </think>.\n"
        f"Be thorough. Use examples, analogies, and clear step-by-step explanations."
    )

    messages = [{"role": "system", "content": system}]

    # Conversation history for multi-turn context
    if history:
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # User message with optional RAG reference
    user_content = ""
    if rag_context:
        user_content += f"Reference:\n{rag_context}\n\n---\n\n"
    user_content += question

    messages.append({"role": "user", "content": user_content})
    return messages


# ── Pipeline execution ───────────────────────────────────────────────────────

class ReasoningPipeline:
    """
    Single-pass reasoning pipeline with <think> tag parsing.

    Uses RAG-first approach: corpus content is the backbone,
    model enhances with teaching. Falls back to RAG directly
    if model output is garbage.
    """

    def __init__(self, model_loader, retriever) -> None:
        self._loader = model_loader
        self._retriever = retriever

    def run(
        self,
        question: str,
        topic: str,
        user_rating: float,
        max_new_tokens: int | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> ReasoningResult:
        """
        Execute the single-pass pipeline synchronously.

        The model generates <think>analysis</think> answer in one call.
        Output is parsed into analysis + answer.
        If model output is garbage, falls back to RAG content directly.
        """
        intent = detect_intent(question)
        if max_new_tokens is None:
            max_new_tokens = 2048

        logger.info("Intent: %s | Topic: %s | Rating: %.0f", intent.value, topic, user_rating)

        # RAG context
        rag_context = ""
        if self._retriever and self._retriever.is_loaded:
            rag_context = self._retriever.get_context_text(question)

        # Build prompt
        messages = build_reasoning_messages(
            question=question,
            topic=topic,
            user_rating=user_rating,
            rag_context=rag_context,
            intent=intent,
            history=history,
        )

        # Single inference call
        t0 = time.perf_counter()
        raw_output = self._loader.generate(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            repetition_penalty=1.1,
        )
        total_time = time.perf_counter() - t0

        # Parse <think>...</think>
        analysis, answer = parse_thinking_output(raw_output)

        # Safety net: if answer is very short, supplement with RAG
        if len(answer.strip()) < 30 and rag_context:
            logger.warning("Short output (%d chars), supplementing with RAG", len(answer))
            answer = (answer.strip() + "\n\n" + rag_context) if answer.strip() else rag_context

        # Estimate thinking time
        total_chars = len(analysis) + len(answer)
        thinking_time = total_time * (len(analysis) / total_chars) if total_chars > 0 and analysis else 0.0

        logger.info("Pipeline done in %.2fs | analysis=%d | answer=%d chars", total_time, len(analysis), len(answer))

        return ReasoningResult(
            analysis=analysis,
            answer=answer,
            intent=intent,
            thinking_time=thinking_time,
            total_time=total_time,
            topic=topic,
            rag_used=bool(rag_context),
            metadata={"user_rating": user_rating, "max_tokens": max_new_tokens},
        )
