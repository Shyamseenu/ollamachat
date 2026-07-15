"""
auth.py — lightweight cookie-session authentication for OllamaChat.

Design choices (deliberately simple for a small local app):
  - Passwords hashed with bcrypt (passlib).
  - Sessions are signed, tamper-proof cookies (itsdangerous), NOT JWTs.
    There's no client-side token to manage and no server-side session
    store needed — the cookie itself IS the session, verified on every
    request via its signature + expiry.
  - SESSION_SECRET_KEY must be set via environment variable in production.
    If unset, a dev-only default is used and a warning is printed.
"""

import os
import secrets
from datetime import datetime
from typing import Optional

from fastapi import HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from passlib.context import CryptContext
from pymongo.collection import Collection

SESSION_COOKIE_NAME = "ollamachat_session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7  # 7 days

_SESSION_SECRET = os.environ.get("SESSION_SECRET_KEY")
if not _SESSION_SECRET:
    _SESSION_SECRET = secrets.token_hex(32)
    print(
        "[auth] WARNING: SESSION_SECRET_KEY not set in environment — using a "
        "random ephemeral secret. All sessions will be invalidated on restart. "
        "Set SESSION_SECRET_KEY in your .env for production."
    )

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_serializer = URLSafeTimedSerializer(_SESSION_SECRET, salt="ollamachat-session")


# Password hashing

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


# Session cookie helpers

def create_session_token(user_id: str) -> str:
    return _serializer.dumps({"user_id": user_id})


def read_session_token(token: str) -> Optional[str]:
    try:
        data = _serializer.loads(token, max_age=SESSION_MAX_AGE_SECONDS)
        return data.get("user_id")
    except (BadSignature, SignatureExpired):
        return None


def set_session_cookie(response, user_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=create_session_token(user_id),
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,       # not readable by JS — mitigates XSS token theft
        samesite="lax",      # sent on top-level navigation, blocks basic CSRF
        secure=os.environ.get("COOKIE_SECURE", "false").lower() == "true",
        path="/",
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")


# User store (wraps the Mongo `users` collection)

class AuthService:
    def __init__(self, users_col: Collection):
        self.users_col = users_col
        # Unique index — DB-level guarantee against duplicate usernames,
        # even under concurrent registration requests.
        self.users_col.create_index("username", unique=True)

    def register(self, username: str, password: str, email: str = "") -> dict:
        username = (username or "").strip().lower()
        if not username or not password:
            raise ValueError("Username and password are required")
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        if self.users_col.find_one({"username": username}):
            raise ValueError("That username is already taken")

        user_doc = {
            "username": username,
            "email": (email or "").strip().lower(),
            "password_hash": hash_password(password),
            "created_at": datetime.utcnow(),
        }
        try:
            result = self.users_col.insert_one(user_doc)
        except Exception:
            # Most likely the unique index caught a race condition
            raise ValueError("That username is already taken")
        user_doc["_id"] = result.inserted_id
        return user_doc

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        username = (username or "").strip().lower()
        user = self.users_col.find_one({"username": username})
        if not user or not verify_password(password, user["password_hash"]):
            return None
        return user

    def get_by_id(self, user_id: str) -> Optional[dict]:
        from bson import ObjectId
        try:
            return self.users_col.find_one({"_id": ObjectId(user_id)})
        except Exception:
            return None


# FastAPI dependencies

def get_current_user_id(request: Request) -> str:
    """Raises 401 if there's no valid session cookie. Use as a route dependency."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user_id = read_session_token(token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired, please log in again")
    return user_id


def get_optional_user_id(request: Request) -> Optional[str]:
    """Same as above but returns None instead of raising — for pages that
    render differently when logged out (e.g. GET /)."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    return read_session_token(token)