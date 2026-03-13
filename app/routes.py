"""
API routes for the Offline Adaptive DSA Tutor.

All endpoints are async.  Inference runs in a thread via asyncio.to_thread
(handled in services.py) so it never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.limiter import limiter
from app.model_loader import ModelOOMError
from app.services import handle_ask, handle_feedback, handle_get_skills

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ──────────────────────────────────────────────

class AskRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    topic: str = Field(..., min_length=1, max_length=100)
    question: str = Field(..., min_length=1, max_length=4000)
    difficulty: float = Field(default=1000.0, ge=0.0, le=4000.0)


class AskResponse(BaseModel):
    interaction_id: int
    response: str
    topic: str
    rating: float
    new_rating: Optional[float] = None
    matches: int
    rag_used: bool


class FeedbackRequest(BaseModel):
    interaction_id: int = Field(..., gt=0)
    feedback: int = Field(..., ge=-1, le=1)
    # Optional full params for backward compatibility
    username: Optional[str] = None
    topic: Optional[str] = None
    difficulty: float = Field(default=1000.0, ge=0.0, le=4000.0)


class FeedbackResponse(BaseModel):
    rating_before: float
    rating_after: float
    delta: float
    matches: int


class SkillsResponse(BaseModel):
    username: str
    skills: list


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    retriever_loaded: bool
    db: str
    adapter: str | None = None


# ── Endpoints ───────────────────────────────────────────────────────────────

@limiter.limit("10/minute")
@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, req: AskRequest) -> AskResponse:
    """Submit a DSA question — returns model response + skill metadata."""
    try:
        result = await handle_ask(
            username=req.username,
            topic=req.topic,
            question=req.question,
            difficulty=req.difficulty,
        )
        return AskResponse(**result)

    except ModelOOMError as exc:
        logger.error("OOM during /ask: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out.")
    except Exception as exc:
        logger.exception("Unexpected error in /ask")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Submit feedback on an interaction — updates ELO rating."""
    try:
        result = await handle_feedback(
            interaction_id=req.interaction_id,
            feedback=req.feedback,
            username=req.username,
            topic=req.topic,
            difficulty=req.difficulty,
        )
        return FeedbackResponse(**result)
    except Exception as exc:
        logger.exception("Unexpected error in /feedback")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.get("/skills/{username}", response_model=SkillsResponse)
async def skills(username: str) -> SkillsResponse:
    """Get all skill ratings for a user."""
    try:
        result = await handle_get_skills(username)
        return SkillsResponse(**result)
    except Exception as exc:
        logger.exception("Unexpected error in /skills")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check — reports status of model, retriever, and DB.
    Uses public accessors from services to avoid reaching into private state.
    """
    from app.services import model_is_loaded, retriever_is_loaded, get_model_loader
    from app.database import _connect

    model_ok = model_is_loaded()
    retriever_ok = retriever_is_loaded()

    loader = get_model_loader()
    adapter_path = loader._adapter_path if loader else None

    db_status = "healthy"
    try:
        async with _connect() as db:
            cursor = await db.execute("SELECT 1")
            await cursor.fetchone()
    except Exception:
        db_status = "unhealthy"

    overall = "ok" if (model_ok and retriever_ok and db_status == "healthy") else "degraded"

    return HealthResponse(
        status=overall,
        model_loaded=model_ok,
        retriever_loaded=retriever_ok,
        db=db_status,
        adapter=adapter_path,
    )
