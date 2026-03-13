"""
FastAPI application entry point.

Lifespan hook initialises:
  1. Structured logging
  2. Database pool + schema
  3. FAISS retriever  (graceful degradation if index missing)
  4. Model loader     (lazy — loaded on first /ask request)
  5. Optional LoRA adapter

On shutdown everything is torn down cleanly.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.database import close_db, init_db
from app.limiter import limiter
from app.model_loader import ModelLoader
from app.mongo import close_mongo, connect_mongo
from app.retriever import Retriever
from app.routes import router
from app.tutor_routes import router as tutor_router
from app.finetune_routes import finetune_router
from app.auth_routes import router as auth_router
from app.services import set_model_loader, set_retriever

# ── Structured logging ──────────────────────────────────────────────────────

def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.addHandler(handler)


# ── Middleware classes ──────────────────────────────────────────────────────

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Read or generate X-Request-ID and echo it on the response."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject /api/ requests (except /api/health) when api_key is set and header is wrong."""

    def __init__(self, app, api_key: str = ""):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        if (
            self.api_key
            and request.method != "OPTIONS"
            and request.url.path.startswith("/api/")
            and request.url.path != "/api/health"
        ):
            key = request.headers.get("x-api-key", "")
            if key != self.api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
        return await call_next(request)


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_logging()
    logger = logging.getLogger("app.main")
    logger.info("Starting DSA Tutor …")

    # 1. Database
    await init_db(settings.db_path)
    logger.info("Database ready.")

    # 1b. MongoDB
    try:
        await connect_mongo()
    except Exception as exc:
        logger.warning("MongoDB unavailable: %s — auth features will be disabled.", exc)

    # 2. Retriever
    retriever = Retriever(
        index_dir=settings.faiss_index_dir,
        top_k=settings.rag_top_k,
        score_threshold=settings.rag_score_threshold,
    )
    try:
        retriever.load()
        logger.info("Retriever loaded.")
    except FileNotFoundError:
        logger.warning(
            "FAISS index not found at %s — RAG will be unavailable until index is built.",
            settings.faiss_index_dir,
        )
    set_retriever(retriever)

    # 3. Model loader (lazy — will load on first generate() call)
    loader = ModelLoader(
        model_id=settings.model_id,
        max_new_tokens=settings.max_new_tokens,
    )
    # 4. Optional adapter
    if settings.adapter_dir:
        try:
            loader.load()
            loader.load_adapter(settings.adapter_dir)
            logger.info("Adapter loaded from %s", settings.adapter_dir)
        except Exception as exc:
            logger.warning("Failed to load adapter: %s — continuing with base model.", exc)
    set_model_loader(loader)

    # 5. Start background model preload (don't block startup)
    async def _preload():
        try:
            logger.info("Background model preload starting …")
            await asyncio.to_thread(loader.load)
            logger.info("Model preloaded and ready.")
        except Exception as exc:
            logger.warning("Background preload failed: %s — model will load on first request.", exc)

    asyncio.create_task(_preload())

    logger.info("Startup complete.")
    yield

    # Shutdown
    logger.info("Shutting down …")
    loader.unload()
    await close_db()
    await close_mongo()
    logger.info("Shutdown complete.")


# ── App factory ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Offline Adaptive DSA Tutor",
        version="1.0.0",
        lifespan=lifespan,
    )

    # -- Middleware (added in reverse execution order) --------------------
    # Innermost first, outermost last.
    # Request flow: CorrelationID → APIKey → CORS → route handler
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)
    app.add_middleware(CorrelationIDMiddleware)

    # -- Rate limiter ----------------------------------------------------
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # -- Routers (all under /api/) ---------------------------------------
    app.include_router(auth_router, prefix="/api")
    app.include_router(router, prefix="/api")
    app.include_router(tutor_router, prefix="/api")
    app.include_router(finetune_router, prefix="/api")

    # Serve React build if it exists, else fall back to old vanilla frontend
    import os
    react_dist = os.path.join(os.path.dirname(__file__), "..", "frontend-react", "dist")
    vanilla_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    if os.path.isdir(react_dist):
        app.mount("/", StaticFiles(directory=react_dist, html=True), name="frontend")
    elif os.path.isdir(vanilla_dir):
        app.mount("/", StaticFiles(directory=vanilla_dir, html=True), name="frontend")
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
