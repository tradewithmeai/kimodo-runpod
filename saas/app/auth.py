"""
Authentication.

Google establishes IDENTITY only. Everything that matters commercially - whether the
account is active, paid, how many credits remain, which jobs are theirs - lives in our
database and is keyed on Google's stable `sub`, never the email (emails can change
hands; `sub` cannot).

The browser receives an ID token from Google Identity Services; the backend verifies its
signature, audience, issuer and expiry server-side, then issues its own HttpOnly session
cookie. The Google token is never used as a session token.
"""

import logging
import time
from typing import Optional

from fastapi import Request, Response, HTTPException

from . import config, db

log = logging.getLogger("linel.auth")


def _verify_google_token(token: str) -> dict:
    """
    Verify a Google ID token server-side.

    Uses google-auth, which fetches and caches Google's public keys and checks the
    signature, expiry, issuer and audience. Do NOT hand-decode the JWT: an unverified
    token is attacker-controlled JSON.
    """
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="google-auth is not installed; cannot verify Google sign-in",
        )

    if not config.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID is not configured")

    try:
        info = google_id_token.verify_oauth2_token(
            token, google_requests.Request(), config.GOOGLE_CLIENT_ID
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=f"invalid Google token: {exc}")

    if info.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        raise HTTPException(status_code=401, detail="unexpected token issuer")
    if not info.get("sub"):
        raise HTTPException(status_code=401, detail="token has no subject")
    return info


def login_with_google(token: str, response: Response) -> dict:
    info = _verify_google_token(token)
    user = db.upsert_google_user(
        sub=info["sub"],
        email=info.get("email", ""),
        name=info.get("name", ""),
        picture=info.get("picture", ""),
    )
    _set_session_cookie(response, db.create_session(user["id"]))
    return user


def login_dev(username: str, response: Response) -> dict:
    """
    Local development login. Refused whenever Google is configured, so it cannot be
    left enabled in production by accident.
    """
    if not config.DEV_LOGIN_ENABLED or config.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=403, detail="dev login is disabled")
    safe = "".join(c for c in username if c.isalnum() or c in "-_.")[:40] or "dev"
    user = db.upsert_google_user(sub=f"dev:{safe}", email=f"{safe}@dev.local", name=safe)
    _set_session_cookie(response, db.create_session(user["id"]))
    return user


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=config.SESSION_COOKIE,
        value=token,
        max_age=config.SESSION_DAYS * 86400,
        httponly=True,              # not readable from JavaScript
        secure=config.COOKIE_SECURE,  # must be True behind HTTPS in production
        samesite="lax",
        path="/",
    )


def logout(request: Request, response: Response) -> None:
    token = request.cookies.get(config.SESSION_COOKIE)
    if token:
        db.delete_session(token)
    response.delete_cookie(config.SESSION_COOKIE, path="/")


def current_user(request: Request) -> Optional[dict]:
    return db.get_session_user(request.cookies.get(config.SESSION_COOKIE, ""))


def require_user(request: Request) -> dict:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="sign in required")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="account is suspended")
    return user


def require_admin(request: Request) -> dict:
    user = require_user(request)
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="admin only")
    return user


def public_user(user: dict) -> dict:
    """The subset safe to send to the browser."""
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "picture": user["picture"],
        "credits": user["credits"],
        "is_admin": bool(user["is_admin"]),
    }
