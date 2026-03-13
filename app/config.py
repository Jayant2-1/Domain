"""
Application configuration - loaded from environment variables with
sensible defaults via pydantic-settings.
"""

from __future__ import annotations

from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All tuneable knobs for the DSA Tutor application."""

    # -- Database --------------------------------------------------------
    db_path: str = "data/mlml.db"

    # -- Model -----------------------------------------------------------
    model_id: str = "models/Mistral-7B-Instruct-v0.3"
    max_new_tokens: int = 2048
    inference_timeout: float = 600.0   # seconds (includes first-load time for 7B model)

    # -- RAG -------------------------------------------------------------
    faiss_index_dir: str = "faiss_index"
    rag_top_k: int = 3
    rag_score_threshold: float = 0.3

    # -- Adapter ---------------------------------------------------------
    adapter_dir: str = "adapters/v_final"   # LoRA adapter loaded at startup

    # -- Server ----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # -- MongoDB ---------------------------------------------------------
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "mlml"

    # -- Auth / JWT ------------------------------------------------------
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # -- Security --------------------------------------------------------
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    api_key: str = ""   # empty = no auth required (local dev)

    model_config = {"env_prefix": "MLML_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
