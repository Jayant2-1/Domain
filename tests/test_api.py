"""
Integration tests for auth API endpoints.

Uses FastAPI TestClient with a real MongoDB test database.
Requires MongoDB running on localhost:27017 (or MLML_MONGODB_URI set).
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.mongo import connect_mongo, close_mongo, get_db


@pytest_asyncio.fixture
async def app():
    """Create app and connect to a test database."""
    import os
    os.environ.setdefault("MLML_MONGODB_DB", "mlml_test")
    os.environ.setdefault("MLML_JWT_SECRET", "test-secret-key")

    application = create_app()

    # Initialize MongoDB
    try:
        await connect_mongo()
    except Exception:
        pytest.skip("MongoDB not available")

    # Clean test database
    db = get_db()
    await db.users.delete_many({})

    yield application

    await db.users.delete_many({})
    await close_mongo()


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_register(client):
    res = await client.post("/api/auth/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "password123",
    })
    assert res.status_code == 201
    data = res.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["username"] == "testuser"


@pytest.mark.asyncio
async def test_register_duplicate_email(client):
    await client.post("/api/auth/register", json={
        "username": "user1",
        "email": "dup@example.com",
        "password": "password123",
    })
    res = await client.post("/api/auth/register", json={
        "username": "user2",
        "email": "dup@example.com",
        "password": "password456",
    })
    assert res.status_code == 409


@pytest.mark.asyncio
async def test_register_duplicate_username(client):
    await client.post("/api/auth/register", json={
        "username": "taken",
        "email": "a@example.com",
        "password": "password123",
    })
    res = await client.post("/api/auth/register", json={
        "username": "taken",
        "email": "b@example.com",
        "password": "password456",
    })
    assert res.status_code == 409


@pytest.mark.asyncio
async def test_login(client):
    await client.post("/api/auth/register", json={
        "username": "loginuser",
        "email": "login@example.com",
        "password": "password123",
    })
    res = await client.post("/api/auth/login", json={
        "email": "login@example.com",
        "password": "password123",
    })
    assert res.status_code == 200
    assert "access_token" in res.json()


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    await client.post("/api/auth/register", json={
        "username": "wrongpw",
        "email": "wp@example.com",
        "password": "correct",
    })
    res = await client.post("/api/auth/login", json={
        "email": "wp@example.com",
        "password": "incorrect",
    })
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_me_authenticated(client):
    reg = await client.post("/api/auth/register", json={
        "username": "meuser",
        "email": "me@example.com",
        "password": "password123",
    })
    token = reg.json()["access_token"]
    res = await client.get("/api/auth/me", headers={
        "Authorization": f"Bearer {token}",
    })
    assert res.status_code == 200
    assert res.json()["username"] == "meuser"


@pytest.mark.asyncio
async def test_me_unauthenticated(client):
    res = await client.get("/api/auth/me")
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token(client):
    reg = await client.post("/api/auth/register", json={
        "username": "refreshuser",
        "email": "refresh@example.com",
        "password": "password123",
    })
    refresh = reg.json()["refresh_token"]
    res = await client.post("/api/auth/refresh", json={
        "refresh_token": refresh,
    })
    assert res.status_code == 200
    assert "access_token" in res.json()


@pytest.mark.asyncio
async def test_health_endpoint(client):
    res = await client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
