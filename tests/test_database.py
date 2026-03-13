"""
Unit tests for the async database layer.

Uses a temporary SQLite database via tempfile. Each test gets a fresh
database via the function-scoped db_setup fixture.
"""

from __future__ import annotations

import os

import aiosqlite
import pytest
import pytest_asyncio

from app.database import (
    close_db,
    get_all_skill_ratings,
    get_interaction,
    get_or_create_topic,
    get_or_create_user,
    get_positive_interactions,
    get_skill_rating,
    init_db,
    insert_interaction,
    update_interaction_feedback,
    upsert_skill_rating,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="function")
async def db_setup(tmp_path):
    """Initialise a fresh SQLite database for each test."""
    path = str(tmp_path / "test.db")
    await init_db(path)
    yield path


@pytest_asyncio.fixture(autouse=True)
async def _clean_rows(db_setup):
    """Clean up tables after each test."""
    yield
    async with aiosqlite.connect(db_setup) as db:
        await db.execute("DELETE FROM interactions")
        await db.execute("DELETE FROM skill_ratings")
        await db.execute("DELETE FROM topics")
        await db.execute("DELETE FROM users")
        await db.commit()


# ── User tests ──────────────────────────────────────────────────────────────

class TestUsers:
    @pytest.mark.asyncio
    async def test_create_user(self, db_setup):
        uid = await get_or_create_user("alice")
        assert isinstance(uid, int)
        assert uid > 0

    @pytest.mark.asyncio
    async def test_idempotent_create(self, db_setup):
        uid1 = await get_or_create_user("bob")
        uid2 = await get_or_create_user("bob")
        assert uid1 == uid2


# ── Topic tests ─────────────────────────────────────────────────────────────

class TestTopics:
    @pytest.mark.asyncio
    async def test_create_topic(self, db_setup):
        tid = await get_or_create_topic("arrays")
        assert isinstance(tid, int)

    @pytest.mark.asyncio
    async def test_idempotent_topic(self, db_setup):
        t1 = await get_or_create_topic("trees")
        t2 = await get_or_create_topic("trees")
        assert t1 == t2


# ── Skill rating tests ─────────────────────────────────────────────────────

class TestSkillRatings:
    @pytest.mark.asyncio
    async def test_default_rating(self, db_setup):
        uid = await get_or_create_user("carol")
        tid = await get_or_create_topic("dp")
        sr = await get_skill_rating(uid, tid)
        assert sr["rating"] == 1000.0
        assert sr["matches"] == 0

    @pytest.mark.asyncio
    async def test_upsert_and_read(self, db_setup):
        uid = await get_or_create_user("dave")
        tid = await get_or_create_topic("graphs")
        await upsert_skill_rating(uid, tid, 1050.5, 3)
        sr = await get_skill_rating(uid, tid)
        assert sr["rating"] == pytest.approx(1050.5, abs=0.1)
        assert sr["matches"] == 3

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, db_setup):
        uid = await get_or_create_user("eve")
        tid = await get_or_create_topic("sorting")
        await upsert_skill_rating(uid, tid, 1100.0, 5)
        await upsert_skill_rating(uid, tid, 1120.0, 6)
        sr = await get_skill_rating(uid, tid)
        assert sr["rating"] == pytest.approx(1120.0, abs=0.1)
        assert sr["matches"] == 6

    @pytest.mark.asyncio
    async def test_get_all_ratings(self, db_setup):
        uid = await get_or_create_user("frank")
        t1 = await get_or_create_topic("arrays")
        t2 = await get_or_create_topic("trees")
        await upsert_skill_rating(uid, t1, 1010.0, 1)
        await upsert_skill_rating(uid, t2, 990.0, 2)
        all_sr = await get_all_skill_ratings(uid)
        assert len(all_sr) == 2
        topics = {r["topic"] for r in all_sr}
        assert topics == {"arrays", "trees"}


# ── Interaction tests ───────────────────────────────────────────────────────

class TestInteractions:
    @pytest.mark.asyncio
    async def test_insert_and_feedback(self, db_setup):
        uid = await get_or_create_user("grace")
        tid = await get_or_create_topic("arrays")
        iid = await insert_interaction(
            user_id=uid,
            topic_id=tid,
            question="What is an array?",
            rag_context="An array is a data structure...",
            response="An array stores elements by index.",
            difficulty=1000.0,
            rating_before=1000.0,
        )
        assert isinstance(iid, int)
        assert iid > 0

        await update_interaction_feedback(iid, feedback=1, rating_after=1020.0)

    @pytest.mark.asyncio
    async def test_positive_interactions_export(self, db_setup):
        uid = await get_or_create_user("heidi")
        tid = await get_or_create_topic("trees")
        iid = await insert_interaction(uid, tid, "Q?", "ctx", "A.", 1000.0, 1000.0)
        await update_interaction_feedback(iid, feedback=1, rating_after=1020.0)

        positives = await get_positive_interactions(min_feedback=1)
        assert len(positives) >= 1
        assert positives[0]["question"] == "Q?"
        assert positives[0]["response"] == "A."
