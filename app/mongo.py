"""
MongoDB async connection layer using motor.

Provides database access for auth (users, sessions) and optionally
other collections. Falls back gracefully if MongoDB is unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def connect_mongo() -> None:
    """Initialize the MongoDB connection and create indexes."""
    global _client, _db
    _client = AsyncIOMotorClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5000,
    )
    _db = _client[settings.mongodb_db]

    # Ensure indexes
    await _db.users.create_index("email", unique=True)
    await _db.users.create_index("username", unique=True)

    # Verify connection
    await _client.admin.command("ping")
    logger.info("MongoDB connected: %s / %s", settings.mongodb_uri, settings.mongodb_db)


async def close_mongo() -> None:
    """Close the MongoDB connection."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB disconnected.")


def get_db() -> AsyncIOMotorDatabase:
    """Return the active database instance."""
    if _db is None:
        raise RuntimeError("MongoDB not initialised — call connect_mongo() first.")
    return _db
