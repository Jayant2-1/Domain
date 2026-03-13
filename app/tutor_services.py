"""
Tutor Service Layer - orchestrates the reasoning pipeline, domain validation,
skill engine, and database for the /tutor endpoints.

Strict separation from the original services.py - this module handles
the single-pass "thinking mode" flow exclusively.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List

from app.config import settings
from app.database import (
    get_or_create_topic,
    get_or_create_user,
    get_skill_rating,
    insert_interaction,
    update_interaction_feedback,
    upsert_skill_rating,
)
from app.domain_validator import (
    GREETING_RESPONSE,
    REFUSAL_MESSAGE,
    VAGUE_QUERY_RESPONSE,
    detect_topic,
    validate_query,
)
from app.model_loader import ModelOOMError
from app.reasoning import ReasoningPipeline, ReasoningResult
from app.services import get_model_loader, get_retriever
from app.skill_engine import compute_new_rating

logger = logging.getLogger(__name__)


def _get_pipeline() -> ReasoningPipeline:
    """Build a ReasoningPipeline from the current singletons."""
    return ReasoningPipeline(
        model_loader=get_model_loader(),
        retriever=get_retriever(),
    )


# ── Full (non-streaming) tutor ask ──────────────────────────────────────────

async def handle_tutor_ask(
    username: str,
    topic: str,
    question: str,
    difficulty: float = 1000.0,
    thinking_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Single-pass thinking-mode pipeline:
      1. Validate domain
      2. Resolve user + topic
      3. Get current ELO
      4. Run single-pass reasoning (model outputs <think>+answer in one call)
      5. Store interaction
      6. Return analysis + answer + metadata
    """
    # 1. Domain validation (also detects greetings / vague queries)
    validation = validate_query(question)
    if not validation.is_valid:
        if validation.reason == "greeting":
            answer = GREETING_RESPONSE
        elif validation.reason == "too_vague":
            answer = VAGUE_QUERY_RESPONSE
        else:
            answer = REFUSAL_MESSAGE
        return {
            "interaction_id": None,
            "analysis": "",
            "answer": answer,
            "intent": "none",
            "thinking_time": 0,
            "total_time": 0,
            "topic": topic,
            "rating": 0,
            "matches": 0,
            "rag_used": False,
            "rejected": validation.reason == "blocked",
        }

    # 2. Auto-detect topic
    topic = detect_topic(question)

    # 3. DB lookups
    user_id = await get_or_create_user(username)
    topic_id = await get_or_create_topic(topic)

    # 3. Current skill
    skill = await get_skill_rating(user_id, topic_id)
    current_rating = skill["rating"]
    matches = skill["matches"]

    # 4. Single-pass reasoning
    pipeline = _get_pipeline()

    if thinking_enabled:
        try:
            result: ReasoningResult = await asyncio.wait_for(
                asyncio.to_thread(
                    pipeline.run,
                    question=question,
                    topic=topic,
                    user_rating=current_rating,
                ),
                timeout=settings.inference_timeout,
            )
        except asyncio.TimeoutError:
            logger.error("Reasoning pipeline timed out")
            raise
        except ModelOOMError:
            raise

        analysis = result.analysis
        answer = result.answer
        intent = result.intent.value
        thinking_time = result.thinking_time
        total_time = result.total_time
        rag_used = result.rag_used
    else:
        # Fallback: plain inference (like original /ask)
        from app.services import build_messages
        messages = build_messages(question, topic, current_rating, "")
        loader = get_model_loader()
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(loader.generate, None, messages),
                timeout=settings.inference_timeout,
            )
        except asyncio.TimeoutError:
            logger.error("Plain inference timed out")
            raise
        analysis = ""
        intent = "solve"
        thinking_time = 0
        total_time = 0
        rag_used = False

    # 5. Store interaction
    interaction_id = await insert_interaction(
        user_id=user_id,
        topic_id=topic_id,
        question=question,
        rag_context=analysis or None,
        response=answer,
        difficulty=difficulty,
        rating_before=current_rating,
    )

    return {
        "interaction_id": interaction_id,
        "analysis": analysis,
        "answer": answer,
        "intent": intent,
        "thinking_time": round(thinking_time, 2),
        "total_time": round(total_time, 2),
        "topic": topic,
        "rating": current_rating,
        "matches": matches,
        "rag_used": rag_used,
        "rejected": False,
    }


# ── Streaming tutor ask (SSE) ───────────────────────────────────────────────

async def handle_tutor_ask_stream(
    username: str,
    topic: str,
    question: str,
    difficulty: float = 1000.0,
    history: List[Dict[str, str]] | None = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming single-pass pipeline via SSE events.

    The model runs ONCE. We emit thinking_start before inference,
    then parse the output and emit thinking_done + answering_done
    once the single inference completes.

    Yields dicts with an "event" key:
      - {"event": "validation_failed", "message": ...}
      - {"event": "thinking_start"}
      - {"event": "thinking_done", "analysis": ..., "thinking_time": ...}
      - {"event": "answering_done", "answer": ..., "total_time": ...}
      - {"event": "done", ...full metadata...}
      - {"event": "error", "message": ...}
    """
    # 1. Domain validation (also detects greetings / vague queries)
    validation = validate_query(question)
    if not validation.is_valid:
        if validation.reason == "greeting":
            msg = GREETING_RESPONSE
        elif validation.reason == "too_vague":
            msg = VAGUE_QUERY_RESPONSE
        else:
            msg = REFUSAL_MESSAGE
        yield {"event": "validation_failed", "message": msg}
        return

    # 2. Auto-detect topic from question
    topic = detect_topic(question)
    logger.info("Auto-detected topic: %s for question: %.60s", topic, question)

    # 3. DB lookups
    try:
        user_id = await get_or_create_user(username)
        topic_id = await get_or_create_topic(topic)
        skill = await get_skill_rating(user_id, topic_id)
        current_rating = skill["rating"]
        matches = skill["matches"]
    except Exception as exc:
        logger.exception("DB error in streaming pipeline")
        yield {"event": "error", "message": str(exc)}
        return

    pipeline = _get_pipeline()

    # 4. Single inference call
    yield {"event": "thinking_start"}

    try:
        result: ReasoningResult = await asyncio.wait_for(
            asyncio.to_thread(
                pipeline.run,
                question=question,
                topic=topic,
                user_rating=current_rating,
                history=history,
            ),
            timeout=settings.inference_timeout,
        )
    except asyncio.TimeoutError:
        yield {"event": "error", "message": "Inference timed out"}
        return
    except ModelOOMError as exc:
        yield {"event": "error", "message": str(exc)}
        return

    # 4. Emit parsed results
    yield {
        "event": "thinking_done",
        "analysis": result.analysis,
        "thinking_time": round(result.thinking_time, 2),
        "intent": result.intent.value,
    }

    yield {
        "event": "answering_done",
        "answer": result.answer,
        "total_time": round(result.total_time, 2),
    }

    # 5. Store interaction
    try:
        interaction_id = await insert_interaction(
            user_id=user_id,
            topic_id=topic_id,
            question=question,
            rag_context=result.analysis or None,
            response=result.answer,
            difficulty=difficulty,
            rating_before=current_rating,
        )
    except Exception:
        interaction_id = None

    yield {
        "event": "done",
        "interaction_id": interaction_id,
        "total_time": round(result.total_time, 2),
        "topic": topic,
        "rating": current_rating,
        "matches": matches,
        "rag_used": result.rag_used,
    }
