"""
Auth middleware — FastAPI dependency for extracting the current user
from a Bearer JWT in the Authorization header.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth import decode_token
from app.mongo import get_db

_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_scheme),
) -> dict:
    """Decode the access token and return the user document from MongoDB."""
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    try:
        payload = decode_token(creds.credentials)
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type.")
        username: str | None = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token.")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    db = get_db()
    user = await db.users.find_one({"username": username}, {"password_hash": 0})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found.")

    user["_id"] = str(user["_id"])
    return user
