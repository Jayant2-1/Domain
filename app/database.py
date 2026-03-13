"""
Async database layer — aiosqlite for local SQLite storage.

All public functions are async.  Each function opens its own connection
to avoid concurrency issues with a shared global connection.
Parameterised queries throughout (?-style placeholders).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

# ── Connection management ───────────────────────────────────────────────────

_db_path: str = "data/mlml.db"


@asynccontextmanager
async def _connect() -> AsyncIterator[aiosqlite.Connection]:
    """Open a new connection with WAL mode and foreign keys enabled."""
    db = await aiosqlite.connect(_db_path)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    try:
        yield db
    finally:
        await db.close()


async def init_db(db_path: str = "data/mlml.db") -> None:
    """Set the database path, create directories, and apply the schema."""
    global _db_path
    _db_path = db_path

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.commit()

        sql_path = Path(__file__).resolve().parent.parent / "sql" / "init_db.sql"
        schema = sql_path.read_text(encoding="utf-8")
        await db.executescript(schema)
        await db.commit()

    logger.info("Database initialised at %s", db_path)


async def close_db() -> None:
    """No-op — connections are opened and closed per-function."""
    logger.info("Database close called (no-op; per-function connections).")


# ── Users ───────────────────────────────────────────────────────────────────

async def get_or_create_user(username: str) -> int:
    """Return user id, creating the user if needed."""
    async with _connect() as db:
        cursor = await db.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = await cursor.fetchone()
        if row is not None:
            return row[0]
        cursor = await db.execute("INSERT INTO users (username) VALUES (?)", (username,))
        await db.commit()
        return cursor.lastrowid  # type: ignore[return-value]


# ── Topics ──────────────────────────────────────────────────────────────────

async def get_or_create_topic(name: str) -> int:
    """Return topic id, creating the topic if needed."""
    async with _connect() as db:
        cursor = await db.execute("SELECT id FROM topics WHERE name = ?", (name,))
        row = await cursor.fetchone()
        if row is not None:
            return row[0]
        cursor = await db.execute("INSERT INTO topics (name) VALUES (?)", (name,))
        await db.commit()
        return cursor.lastrowid  # type: ignore[return-value]


# ── Skill ratings ──────────────────────────────────────────────────────────

async def get_skill_rating(user_id: int, topic_id: int) -> Dict[str, Any]:
    """Fetch the skill rating for (user, topic)."""
    async with _connect() as db:
        cursor = await db.execute(
            "SELECT rating, matches FROM skill_ratings WHERE user_id = ? AND topic_id = ?",
            (user_id, topic_id),
        )
        row = await cursor.fetchone()
        if row is not None:
            return {"rating": float(row[0]), "matches": int(row[1])}
        return {"rating": 1000.0, "matches": 0}


async def upsert_skill_rating(
    user_id: int, topic_id: int, rating: float, matches: int
) -> None:
    """Insert or update the skill rating for (user, topic)."""
    async with _connect() as db:
        await db.execute(
            """
            INSERT INTO skill_ratings (user_id, topic_id, rating, matches, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT (user_id, topic_id)
            DO UPDATE SET rating = ?, matches = ?, updated_at = datetime('now')
            """,
            (user_id, topic_id, rating, matches, rating, matches),
        )
        await db.commit()


async def get_all_skill_ratings(user_id: int) -> List[Dict[str, Any]]:
    """Return all topic ratings for a user."""
    async with _connect() as db:
        cursor = await db.execute(
            """
            SELECT t.name AS topic, sr.rating, sr.matches
            FROM skill_ratings sr
            JOIN topics t ON t.id = sr.topic_id
            WHERE sr.user_id = ?
            ORDER BY t.name
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [
            {"topic": r[0], "rating": float(r[1]), "matches": int(r[2])}
            for r in rows
        ]


# ── Interactions ────────────────────────────────────────────────────────────

async def insert_interaction(
    user_id: int,
    topic_id: int,
    question: str,
    rag_context: Optional[str],
    response: str,
    difficulty: float,
    rating_before: float,
) -> int:
    """Insert a new interaction row and return its id."""
    async with _connect() as db:
        cursor = await db.execute(
            """
            INSERT INTO interactions
                (user_id, topic_id, question, rag_context, response,
                 difficulty, rating_before)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, topic_id, question, rag_context, response, difficulty, rating_before),
        )
        await db.commit()
        return cursor.lastrowid  # type: ignore[return-value]


async def update_interaction_feedback(
    interaction_id: int,
    feedback: int,
    rating_after: float,
) -> None:
    """Set feedback and post-update rating on an existing interaction."""
    async with _connect() as db:
        await db.execute(
            "UPDATE interactions SET feedback = ?, rating_after = ? WHERE id = ?",
            (feedback, rating_after, interaction_id),
        )
        await db.commit()


async def get_interaction(interaction_id: int) -> Optional[Dict[str, Any]]:
    """Fetch an interaction by id."""
    async with _connect() as db:
        cursor = await db.execute(
            "SELECT id, user_id, topic_id, question, difficulty, rating_before FROM interactions WHERE id = ?",
            (interaction_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "user_id": row[1],
            "topic_id": row[2],
            "question": row[3],
            "difficulty": float(row[4]) if row[4] else 1000.0,
            "rating_before": float(row[5]) if row[5] else 1000.0,
        }


async def get_positive_interactions(
    min_feedback: int = 1,
    limit: int = 10000,
) -> List[Dict[str, Any]]:
    """Fetch interactions with positive feedback for fine-tuning."""
    async with _connect() as db:
        cursor = await db.execute(
            """
            SELECT i.question, i.rag_context, i.response, t.name AS topic
            FROM interactions i
            JOIN topics t ON t.id = i.topic_id
            WHERE i.feedback >= ?
            ORDER BY i.created_at DESC
            LIMIT ?
            """,
            (min_feedback, limit),
        )
        rows = await cursor.fetchall()
        return [
            {"question": r[0], "rag_context": r[1], "response": r[2], "topic": r[3]}
            for r in rows
        ]
