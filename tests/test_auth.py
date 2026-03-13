"""
Tests for auth utilities — password hashing and JWT token creation/verification.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_and_verify(self):
        plain = "secure_password_123"
        hashed = hash_password(plain)
        assert hashed != plain
        assert verify_password(plain, hashed)

    def test_wrong_password_fails(self):
        hashed = hash_password("correct_password")
        assert not verify_password("wrong_password", hashed)

    def test_different_hashes_for_same_password(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2  # bcrypt uses random salt


class TestJWT:
    def test_access_token_roundtrip(self):
        data = {"sub": "testuser"}
        token = create_access_token(data)
        payload = decode_token(token)
        assert payload["sub"] == "testuser"
        assert payload["type"] == "access"

    def test_refresh_token_roundtrip(self):
        data = {"sub": "testuser"}
        token = create_refresh_token(data)
        payload = decode_token(token)
        assert payload["sub"] == "testuser"
        assert payload["type"] == "refresh"

    def test_custom_expiry(self):
        token = create_access_token({"sub": "u"}, expires_delta=timedelta(hours=1))
        payload = decode_token(token)
        assert payload["sub"] == "u"

    def test_token_contains_exp(self):
        token = create_access_token({"sub": "u"})
        payload = decode_token(token)
        assert "exp" in payload

    def test_invalid_token_raises(self):
        from jose import JWTError
        with pytest.raises(JWTError):
            decode_token("invalid.token.here")
