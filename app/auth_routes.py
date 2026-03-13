"""
Auth routes — register, login, token refresh, current user.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.auth_middleware import get_current_user
from app.mongo import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


# ── Schemas ─────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    username: str


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    username: str
    email: str
    created_at: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    db = get_db()

    # Check uniqueness
    if await db.users.find_one({"email": req.email}):
        raise HTTPException(status_code=409, detail="Email already registered.")
    if await db.users.find_one({"username": req.username}):
        raise HTTPException(status_code=409, detail="Username already taken.")

    user_doc = {
        "username": req.username,
        "email": req.email,
        "password_hash": hash_password(req.password),
        "created_at": datetime.now(timezone.utc),
    }
    await db.users.insert_one(user_doc)
    logger.info("User registered: %s", req.username)

    access = create_access_token({"sub": req.username})
    refresh = create_refresh_token({"sub": req.username})
    return TokenResponse(access_token=access, refresh_token=refresh, username=req.username)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    db = get_db()
    user = await db.users.find_one({"email": req.email})
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    access = create_access_token({"sub": user["username"]})
    refresh = create_refresh_token({"sub": user["username"]})
    return TokenResponse(access_token=access, refresh_token=refresh, username=user["username"])


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(req: RefreshRequest):
    try:
        payload = decode_token(req.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type.")
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token.")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token.")

    db = get_db()
    user = await db.users.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")

    access = create_access_token({"sub": username})
    refresh = create_refresh_token({"sub": username})
    return TokenResponse(access_token=access, refresh_token=refresh, username=username)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        username=current_user["username"],
        email=current_user["email"],
        created_at=current_user["created_at"].isoformat()
        if isinstance(current_user["created_at"], datetime)
        else str(current_user["created_at"]),
    )
