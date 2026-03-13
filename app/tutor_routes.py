"""
Tutor API Routes - /tutor/ask and /tutor/ask/stream endpoints.

Provides the "thinking mode" interface with structured reasoning,
domain validation, and SSE streaming support.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.limiter import limiter
from app.model_loader import ModelOOMError
from app.tutor_services import handle_tutor_ask, handle_tutor_ask_stream

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tutor", tags=["tutor"])


# ── Request / Response schemas ───────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: str = Field(..., max_length=4000)


class TutorAskRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    topic: str = Field(default="dsa", max_length=100)
    question: str = Field(..., min_length=1, max_length=4000)
    difficulty: float = Field(default=1000.0, ge=0.0, le=4000.0)
    thinking_enabled: bool = Field(default=True)
    history: List[HistoryMessage] = Field(default_factory=list, max_length=10)


class TutorAskResponse(BaseModel):
    interaction_id: Optional[int]
    analysis: str
    answer: str
    intent: str
    thinking_time: float
    total_time: float
    topic: str
    rating: float
    matches: int
    rag_used: bool
    rejected: bool


# ── Endpoints ────────────────────────────────────────────────────────────────

@limiter.limit("5/minute")
@router.post("/ask", response_model=TutorAskResponse)
async def tutor_ask(request: Request, req: TutorAskRequest) -> TutorAskResponse:
    """
    Two-stage reasoning endpoint.

    Returns structured analysis (thinking) + polished final answer.
    Set thinking_enabled=false to skip the analysis stage.
    """
    try:
        result = await handle_tutor_ask(
            username=req.username,
            topic=req.topic,
            question=req.question,
            difficulty=req.difficulty,
            thinking_enabled=req.thinking_enabled,
        )
        return TutorAskResponse(**result)

    except ModelOOMError as exc:
        logger.error("OOM during /tutor/ask: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Reasoning pipeline timed out.")
    except Exception:
        logger.exception("Unexpected error in /tutor/ask")
        raise HTTPException(status_code=500, detail="Internal server error.")


@limiter.limit("5/minute")
@router.post("/ask/stream")
async def tutor_ask_stream(request: Request, req: TutorAskRequest):
    """
    SSE streaming endpoint for two-stage reasoning.

    Streams events as Server-Sent Events:
      - thinking_start      -> analysis is beginning
      - thinking_done       -> analysis complete (includes analysis text)
      - answering_start     -> final answer generation begins
      - answering_done      -> final answer complete
      - done                -> full metadata (interaction_id, rating, etc.)
      - error               -> if something went wrong
      - validation_failed   -> query rejected by domain validator
    """
    async def event_generator():
        try:
            async for event_data in handle_tutor_ask_stream(
                username=req.username,
                topic=req.topic,
                question=req.question,
                difficulty=req.difficulty,
                history=[{"role": m.role, "content": m.content} for m in req.history],
            ):
                event_type = event_data.get("event", "message")
                payload = json.dumps(event_data, ensure_ascii=False)
                yield f"event: {event_type}\ndata: {payload}\n\n"
        except ModelOOMError as exc:
            error_payload = json.dumps({"event": "error", "message": str(exc)})
            yield f"event: error\ndata: {error_payload}\n\n"
        except asyncio.TimeoutError:
            error_payload = json.dumps({"event": "error", "message": "Pipeline timed out"})
            yield f"event: error\ndata: {error_payload}\n\n"
        except Exception as exc:
            logger.exception("Unexpected error in /tutor/ask/stream")
            error_payload = json.dumps({"event": "error", "message": "Internal server error"})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
