"""
Configuration, entirely environment-driven so the same image runs locally and on the VPS.

Nothing here reads a hardcoded absolute path: the service must be deployable without
editing source. Defaults are chosen so `python -m app.main` works on a dev box with no
.env at all.
"""

import os
import secrets
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


BASE_DIR = Path(__file__).resolve().parent.parent

# --- storage -----------------------------------------------------------------
DATA_DIR = Path(os.environ.get("LINEL_DATA_DIR", BASE_DIR / "data"))
DB_PATH = Path(os.environ.get("LINEL_DB_PATH", DATA_DIR / "linel.db"))
OUTPUT_DIR = Path(os.environ.get("LINEL_OUTPUT_DIR", DATA_DIR / "outputs"))

# --- model -------------------------------------------------------------------
# MDM_REPO is consumed by motion_server; we pass it through rather than duplicating.
MDM_REPO = os.environ.get("MDM_REPO", "")
MODEL_CHECKPOINT = os.environ.get("LINEL_CHECKPOINT", "")
MODEL_DEVICE = os.environ.get("LINEL_DEVICE", "cpu")

# Cap threads so one generation cannot monopolise a small VPS and starve the web
# process. Measured: 1 thread = 5.85s, 2 threads = 3.49s for a 6s animation.
TORCH_THREADS = _int("LINEL_TORCH_THREADS", 2)

# --- generation limits (also the abuse surface) ------------------------------
MAX_SECONDS = float(os.environ.get("LINEL_MAX_SECONDS", "9.8"))
MIN_SECONDS = float(os.environ.get("LINEL_MIN_SECONDS", "1.0"))
MAX_PROMPT_CHARS = _int("LINEL_MAX_PROMPT_CHARS", 500)
MAX_QUEUE_DEPTH = _int("LINEL_MAX_QUEUE_DEPTH", 50)
MAX_JOBS_PER_USER_QUEUED = _int("LINEL_MAX_JOBS_PER_USER_QUEUED", 1)

# --- credits -----------------------------------------------------------------
SIGNUP_CREDITS = _int("LINEL_SIGNUP_CREDITS", 20)
CREDITS_PER_GENERATION = _int("LINEL_CREDITS_PER_GENERATION", 1)

# --- auth --------------------------------------------------------------------
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
# Dev login exists so the service is testable before Google is configured.
# It is refused whenever a Google client ID is present, so it cannot be left on
# by accident in production.
DEV_LOGIN_ENABLED = _bool("LINEL_DEV_LOGIN", not bool(GOOGLE_CLIENT_ID))
SESSION_COOKIE = "linel_session"
SESSION_DAYS = _int("LINEL_SESSION_DAYS", 30)
# Secret is generated per-process if unset: sessions then die on restart, which is
# safe-by-default rather than silently using a well-known key.
SECRET_KEY = os.environ.get("LINEL_SECRET_KEY") or secrets.token_urlsafe(32)
COOKIE_SECURE = _bool("LINEL_COOKIE_SECURE", False)

# --- server ------------------------------------------------------------------
HOST = os.environ.get("LINEL_HOST", "127.0.0.1")
PORT = _int("LINEL_PORT", 8000)
PUBLIC_URL = os.environ.get("LINEL_PUBLIC_URL", f"http://{HOST}:{PORT}")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def summary() -> dict:
    """Safe to log and to expose on an admin endpoint - contains no secrets."""
    return {
        "data_dir": str(DATA_DIR),
        "device": MODEL_DEVICE,
        "torch_threads": TORCH_THREADS,
        "max_seconds": MAX_SECONDS,
        "signup_credits": SIGNUP_CREDITS,
        "credits_per_generation": CREDITS_PER_GENERATION,
        "google_configured": bool(GOOGLE_CLIENT_ID),
        "dev_login_enabled": DEV_LOGIN_ENABLED,
        "cookie_secure": COOKIE_SECURE,
    }
