"""
Fine-tuning management endpoints.

POST /finetune/trigger  — Run LoRA fine-tuning in a background thread
POST /finetune/adapter  — Hot-swap a LoRA adapter on the running model
GET  /finetune/status   — Check fine-tuning status
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
finetune_router = APIRouter(prefix="/finetune", tags=["finetune"])


# ── State ───────────────────────────────────────────────────────────────────

class _FinetuneState:
    """Simple in-memory state tracker for fine-tuning jobs."""
    def __init__(self):
        self.running = False
        self.last_status: str = "idle"
        self.last_adapter: Optional[str] = None
        self.last_started: Optional[str] = None
        self.last_finished: Optional[str] = None
        self.last_error: Optional[str] = None
        self.records_exported: int = 0

_state = _FinetuneState()


# ── Schemas ─────────────────────────────────────────────────────────────────

class FinetuneTriggerRequest(BaseModel):
    min_feedback: int = Field(default=1, ge=-1, le=1)
    max_steps: int = Field(default=200, ge=10, le=5000)
    learning_rate: float = Field(default=2e-4, gt=0, lt=1)
    lora_r: int = Field(default=16, ge=4, le=64)
    use_seed_data: bool = Field(default=False, description="Use bundled seed DSA data instead of DB interactions")


class FinetuneTriggerResponse(BaseModel):
    message: str
    job_started: bool


class AdapterLoadRequest(BaseModel):
    adapter_path: str = Field(..., min_length=1)


class AdapterLoadResponse(BaseModel):
    message: str
    adapter_path: str


class FinetuneStatusResponse(BaseModel):
    running: bool
    status: str
    last_adapter: Optional[str]
    last_started: Optional[str]
    last_finished: Optional[str]
    last_error: Optional[str]
    records_exported: int


# ── Background fine-tuning task ─────────────────────────────────────────────

def _run_finetune_sync(
    data_path: str,
    output_dir: str,
    model_id: str,
    max_steps: int,
    learning_rate: float,
    lora_r: int,
) -> str:
    """Synchronous fine-tuning — runs in a thread."""
    from finetune.train_lora import train

    return train(
        data_path=data_path,
        output_dir=output_dir,
        model_id=model_id,
        max_seq_len=512,
        lora_r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        batch_size=1,
        grad_accum=4,
        learning_rate=learning_rate,
        max_steps=max_steps,
    )


async def _finetune_job(
    min_feedback: int,
    max_steps: int,
    learning_rate: float,
    lora_r: int,
    use_seed_data: bool,
) -> None:
    """Async wrapper that runs export + training in a background thread."""
    from app.services import get_model_loader
    loader = get_model_loader()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        _state.running = True
        _state.last_status = "unloading inference model"
        _state.last_started = ts
        _state.last_error = None

        loader.unload()

        _state.last_status = "exporting data"

        if use_seed_data:
            data_path = "finetune/data/seed_dsa_training.jsonl"
            if not Path(data_path).exists():
                raise FileNotFoundError(f"Seed data not found: {data_path}")
            with open(data_path) as f:
                _state.records_exported = sum(1 for line in f if line.strip())
        else:
            # Export from database
            from finetune.prepare_data import export_data
            data_path = f"finetune/data/train_{ts}.jsonl"
            count = await export_data(
                db_path="data/mlml.db",
                output_path=data_path,
                min_feedback=min_feedback,
            )
            _state.records_exported = count
            if count == 0:
                _state.last_status = "no data"
                _state.last_error = "No positive interactions found. Use seed data or collect more feedback."
                return

        output_dir = f"adapters/v_{ts}"
        _state.last_status = "training"

        from app.config import settings
        model_id = settings.model_id

        adapter_path = await asyncio.to_thread(
            _run_finetune_sync,
            data_path, output_dir, model_id,
            max_steps, learning_rate, lora_r,
        )

        _state.last_adapter = adapter_path
        _state.last_status = "completed"
        _state.last_finished = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("Fine-tuning complete. Adapter: %s", adapter_path)

    except Exception as exc:
        logger.exception("Fine-tuning failed")
        _state.last_status = "failed"
        _state.last_error = str(exc)
    finally:
        _state.running = False
        try:
            await asyncio.to_thread(loader.load)
        except Exception as exc:
            logger.warning("Failed to reload inference model after fine-tuning: %s", exc)


# ── Endpoints ───────────────────────────────────────────────────────────────

@finetune_router.post("/trigger", response_model=FinetuneTriggerResponse)
async def trigger_finetune(req: FinetuneTriggerRequest) -> FinetuneTriggerResponse:
    """Trigger a LoRA fine-tuning job in the background."""
    if _state.running:
        raise HTTPException(status_code=409, detail="Fine-tuning already in progress.")

    # Launch background task
    asyncio.create_task(_finetune_job(
        min_feedback=req.min_feedback,
        max_steps=req.max_steps,
        learning_rate=req.learning_rate,
        lora_r=req.lora_r,
        use_seed_data=req.use_seed_data,
    ))

    return FinetuneTriggerResponse(
        message="Fine-tuning job started in background.",
        job_started=True,
    )


@finetune_router.post("/adapter", response_model=AdapterLoadResponse)
async def load_adapter(req: AdapterLoadRequest) -> AdapterLoadResponse:
    """Hot-swap a LoRA adapter on the running model."""
    from app.services import _model_loader

    if _model_loader is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")

    path = Path(req.adapter_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Adapter not found: {req.adapter_path}")

    try:
        if not _model_loader.is_loaded:
            await asyncio.to_thread(_model_loader.load)

        await asyncio.to_thread(_model_loader.load_adapter, req.adapter_path)
        return AdapterLoadResponse(
            message=f"Adapter loaded successfully.",
            adapter_path=req.adapter_path,
        )
    except Exception as exc:
        logger.exception("Failed to load adapter")
        raise HTTPException(status_code=500, detail=str(exc))


@finetune_router.get("/status", response_model=FinetuneStatusResponse)
async def finetune_status() -> FinetuneStatusResponse:
    """Get the current fine-tuning job status."""
    return FinetuneStatusResponse(
        running=_state.running,
        status=_state.last_status,
        last_adapter=_state.last_adapter,
        last_started=_state.last_started,
        last_finished=_state.last_finished,
        last_error=_state.last_error,
        records_exported=_state.records_exported,
    )
