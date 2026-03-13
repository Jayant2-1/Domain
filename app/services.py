"""
Service layer - orchestrates the skill engine, RAG retriever, model loader,
and database for each user interaction.

All public functions are async.  No direct HTTP concerns here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.config import settings
from app.database import (
    get_all_skill_ratings,
    get_or_create_topic,
    get_or_create_user,
    get_skill_rating,
    insert_interaction,
    update_interaction_feedback,
    upsert_skill_rating,
)
from app.model_loader import ModelLoader, ModelOOMError
from app.retriever import Retriever
from app.skill_engine import compute_new_rating

logger = logging.getLogger(__name__)

# -- Singleton-ish references set during lifespan ----------------------------

_model_loader: Optional[ModelLoader] = None
_retriever: Optional[Retriever] = None


def set_model_loader(loader: ModelLoader) -> None:
    global _model_loader
    _model_loader = loader


def get_model_loader() -> ModelLoader:
    if _model_loader is None:
        raise RuntimeError("ModelLoader not initialised.")
    return _model_loader


def set_retriever(retriever: Retriever) -> None:
    global _retriever
    _retriever = retriever


def get_retriever() -> Retriever:
    if _retriever is None:
        raise RuntimeError("Retriever not initialised.")
    return _retriever


def model_is_loaded() -> bool:
    """Public accessor — True when the model loader exists and the model is loaded."""
    return _model_loader is not None and _model_loader.is_loaded


def retriever_is_loaded() -> bool:
    """Public accessor — True when the retriever exists and its index is loaded."""
    return _retriever is not None and _retriever.is_loaded


# -- Prompt building ---------------------------------------------------------

def build_messages(
    question: str,
    topic: str,
    user_rating: float,
    rag_context: str,
) -> list:
    """
    Build the chat messages list for the model.

    Returns a list of {"role": ..., "content": ...} dicts that will be
    formatted by tokenizer.apply_chat_template().
    """
    # Determine difficulty hint based on rating
    if user_rating < 900:
        level = "beginner"
        style = "Use simple language, step-by-step explanations, and analogies. Start from basics."
    elif user_rating < 1100:
        level = "intermediate"
        style = "Assume familiarity with basic concepts. Focus on patterns, trade-offs, and implementation details."
    else:
        level = "advanced"
        style = "Be concise and technical. Discuss edge cases, optimisations, time/space analysis, and advanced variants."

    topic_display = topic.replace('_', ' ')

    system = (
        f"You are an expert Data Structures and Algorithms (DSA) tutor.\n"
        f"Topic: {topic_display}\n"
        f"Student level: {level} (ELO rating: {user_rating:.0f})\n\n"
        f"Instructions:\n"
        f"- {style}\n"
        f"- Give DETAILED, thorough explanations. Do not give one-line answers.\n"
        f"- Include code examples ONLY if the student asks for code or a solution.\n"
        f"- If the student asks for explanation only or says no code, do NOT include code.\n"
        f"- Include time and space complexity analysis with reasoning.\n"
        f"- Give a concrete example with input/output.\n"
        f"- Be accurate. If you are unsure about something, say so.\n"
        f"- Your answer should be comprehensive — at least 200 words."
    )

    messages = [{"role": "system", "content": system}]

    if rag_context:
        user_content = (
            f"Here is some reference material on {topic_display}:\n\n"
            f"{rag_context}\n\n"
            f"Based on the above, answer this question:\n{question}"
        )
    else:
        user_content = question

    messages.append({"role": "user", "content": user_content})
    return messages


# -- Core service functions --------------------------------------------------

async def handle_ask(
    username: str,
    topic: str,
    question: str,
    difficulty: float = 1000.0,
) -> Dict[str, Any]:
    """
    Full ask pipeline:
      1. Resolve user + topic in DB
      2. Get current ELO rating
      3. Retrieve RAG context
      4. Build prompt
      5. Run inference (with timeout)
      6. Store interaction
      7. Return response + metadata
    """
    # 1. DB lookups
    user_id = await get_or_create_user(username)
    topic_id = await get_or_create_topic(topic)

    # 2. Current skill
    skill = await get_skill_rating(user_id, topic_id)
    current_rating = skill["rating"]
    matches = skill["matches"]

    # 3. RAG
    retriever = get_retriever()
    rag_context = ""
    if retriever.is_loaded:
        rag_context = await asyncio.to_thread(retriever.get_context_text, question)

    # 4. Build messages
    messages = build_messages(question, topic, current_rating, rag_context)

    # 5. Inference with timeout
    loader = get_model_loader()
    try:
        response_text = await asyncio.wait_for(
            asyncio.to_thread(loader.generate, None, messages),
            timeout=settings.inference_timeout,
        )
    except asyncio.TimeoutError:
        logger.error("Inference timed out after %.0fs", settings.inference_timeout)
        raise
    except ModelOOMError:
        raise

    # 6. Store interaction (feedback = NULL initially)
    interaction_id = await insert_interaction(
        user_id=user_id,
        topic_id=topic_id,
        question=question,
        rag_context=rag_context or None,
        response=response_text,
        difficulty=difficulty,
        rating_before=current_rating,
    )

    return {
        "interaction_id": interaction_id,
        "response": response_text,
        "topic": topic,
        "rating": current_rating,
        "matches": matches,
        "rag_used": bool(rag_context),
    }


async def handle_feedback(
    interaction_id: int,
    feedback: int,
    username: str | None = None,
    topic: str | None = None,
    difficulty: float = 1000.0,
) -> Dict[str, Any]:
    """
    Process user feedback on an interaction:
      1. Look up interaction details from DB
      2. Look up current rating
      3. Compute ELO update
      4. Persist new rating + feedback
      5. Return delta
    """
    from app.database import get_interaction, get_or_create_user, get_or_create_topic

    # Look up interaction to get user/topic if not provided
    interaction = await get_interaction(interaction_id)
    if interaction is None:
        return {"rating_before": 0, "rating_after": 0, "delta": 0, "matches": 0}

    user_id = interaction["user_id"]
    topic_id = interaction["topic_id"]
    difficulty = interaction.get("difficulty", difficulty) or difficulty

    skill = await get_skill_rating(user_id, topic_id)
    rating_before = skill["rating"]
    matches = skill["matches"]

    answered_correctly = feedback > 0

    new_rating, delta = compute_new_rating(
        user_rating=rating_before,
        question_difficulty=difficulty,
        answered_correctly=answered_correctly,
        matches=matches,
    )

    feedback_val = 1 if answered_correctly else 0
    await update_interaction_feedback(interaction_id, feedback_val, new_rating)
    await upsert_skill_rating(user_id, topic_id, new_rating, matches + 1)

    return {
        "rating_before": round(rating_before, 2),
        "rating_after": round(new_rating, 2),
        "delta": round(delta, 2),
        "matches": matches + 1,
    }


async def handle_get_skills(username: str) -> Dict[str, Any]:
    """Return all skill ratings for a user."""
    user_id = await get_or_create_user(username)
    ratings = await get_all_skill_ratings(user_id)
    return {"username": username, "skills": ratings}
