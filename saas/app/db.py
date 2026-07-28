"""
SQLite persistence.

SQLite is deliberate for the beta: a single-writer workload with one inference worker
does not need Postgres, and one file is far easier to back up. The schema is kept
Postgres-compatible so migrating later is a connection-string change rather than a
rewrite.

The generation ledger records everything needed to compute real unit economics -
including LLM token counts and cost, which the benchmark showed dominate the cost of a
generation by roughly 60x.
"""

import json
import sqlite3
import secrets
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    google_sub      TEXT UNIQUE,           -- Google's stable subject id, NOT the email
    email           TEXT,
    name            TEXT,
    picture         TEXT,
    credits         INTEGER NOT NULL DEFAULT 0,
    is_active       INTEGER NOT NULL DEFAULT 1,
    is_admin        INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    last_seen_at    REAL
);

CREATE TABLE IF NOT EXISTS sessions (
    token           TEXT PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    created_at      REAL NOT NULL,
    expires_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS jobs (
    id                  TEXT PRIMARY KEY,
    user_id             INTEGER NOT NULL REFERENCES users(id),
    status              TEXT NOT NULL,      -- queued|running|done|failed|cancelled
    prompt              TEXT NOT NULL,      -- what the user typed
    interpreted_prompt  TEXT,               -- what the LLM layer rewrote it to
    seconds             REAL NOT NULL,
    guidance            REAL NOT NULL,
    seed                INTEGER,
    frames              INTEGER,
    output_path         TEXT,
    output_bytes        INTEGER,
    error               TEXT,
    credits_charged     INTEGER NOT NULL DEFAULT 0,
    queued_at           REAL NOT NULL,
    started_at          REAL,
    finished_at         REAL,
    duration_s          REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id, queued_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS llm_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT REFERENCES jobs(id),
    provider        TEXT,
    model           TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cached_tokens   INTEGER,
    cost_usd        REAL,
    latency_s       REAL,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_job ON llm_calls(job_id);

CREATE TABLE IF NOT EXISTS credit_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    delta           INTEGER NOT NULL,
    reason          TEXT NOT NULL,
    job_id          TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_credit_user ON credit_events(user_id, created_at DESC);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL lets the web process read while the worker writes, which is the whole
    # concurrency story for this design.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    config.ensure_dirs()
    with get_db() as conn:
        conn.executescript(SCHEMA)


# --- users -------------------------------------------------------------------

def upsert_google_user(sub: str, email: str, name: str, picture: str = "") -> dict:
    """Find or create a user keyed on Google's stable `sub`, never the email."""
    now = time.time()
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE google_sub = ?", (sub,)).fetchone()
        if row:
            conn.execute(
                "UPDATE users SET email=?, name=?, picture=?, last_seen_at=? WHERE id=?",
                (email, name, picture, now, row["id"]),
            )
            return dict(conn.execute("SELECT * FROM users WHERE id=?", (row["id"],)).fetchone())

        cur = conn.execute(
            "INSERT INTO users (google_sub, email, name, picture, credits, created_at, last_seen_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (sub, email, name, picture, config.SIGNUP_CREDITS, now, now),
        )
        uid = cur.lastrowid
        conn.execute(
            "INSERT INTO credit_events (user_id, delta, reason, created_at) VALUES (?,?,?,?)",
            (uid, config.SIGNUP_CREDITS, "signup", now),
        )
        return dict(conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone())


def get_user(user_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None


def adjust_credits(user_id: int, delta: int, reason: str, job_id: str | None = None) -> int:
    """Apply a credit change and record it. Returns the new balance."""
    with get_db() as conn:
        conn.execute("UPDATE users SET credits = credits + ? WHERE id = ?", (delta, user_id))
        conn.execute(
            "INSERT INTO credit_events (user_id, delta, reason, job_id, created_at) VALUES (?,?,?,?,?)",
            (user_id, delta, reason, job_id, time.time()),
        )
        return conn.execute("SELECT credits FROM users WHERE id=?", (user_id,)).fetchone()["credits"]


# --- sessions ----------------------------------------------------------------

def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?,?,?,?)",
            (token, user_id, now, now + config.SESSION_DAYS * 86400),
        )
    return token


def get_session_user(token: str) -> Optional[dict]:
    if not token:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id"
            " WHERE s.token = ? AND s.expires_at > ?",
            (token, time.time()),
        ).fetchone()
        return dict(row) if row else None


def delete_session(token: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# --- jobs --------------------------------------------------------------------

def new_job_id() -> str:
    return "job_" + secrets.token_hex(10)


def create_job(job_id: str, user_id: int, prompt: str, seconds: float,
               guidance: float, seed: int | None) -> dict:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO jobs (id, user_id, status, prompt, seconds, guidance, seed, queued_at)"
            " VALUES (?,?,'queued',?,?,?,?,?)",
            (job_id, user_id, prompt, seconds, guidance, seed, time.time()),
        )
        return dict(conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    with get_db() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))


def get_job(job_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None


def list_jobs(user_id: int, limit: int = 25) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE user_id=? ORDER BY queued_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def count_active_jobs(user_id: int) -> int:
    with get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) c FROM jobs WHERE user_id=? AND status IN ('queued','running')",
            (user_id,),
        ).fetchone()["c"]


def queue_depth() -> int:
    with get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) c FROM jobs WHERE status IN ('queued','running')"
        ).fetchone()["c"]


def reclaim_orphaned_jobs() -> int:
    """
    On startup, any job left 'running' belongs to a process that died. Nothing is
    executing it, so it would otherwise hang forever. Fail them and refund.
    """
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, user_id, credits_charged FROM jobs WHERE status IN ('running','queued')"
        ).fetchall()
        for r in rows:
            conn.execute(
                "UPDATE jobs SET status='failed', error=?, finished_at=? WHERE id=?",
                ("interrupted by server restart", time.time(), r["id"]),
            )
            if r["credits_charged"]:
                conn.execute("UPDATE users SET credits = credits + ? WHERE id = ?",
                             (r["credits_charged"], r["user_id"]))
                conn.execute(
                    "INSERT INTO credit_events (user_id, delta, reason, job_id, created_at)"
                    " VALUES (?,?,?,?,?)",
                    (r["user_id"], r["credits_charged"], "refund:restart", r["id"], time.time()),
                )
        return len(rows)


# --- llm ledger --------------------------------------------------------------

def record_llm_call(job_id: str, provider: str, model: str, input_tokens: int,
                    output_tokens: int, cached_tokens: int, cost_usd: float,
                    latency_s: float) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO llm_calls (job_id, provider, model, input_tokens, output_tokens,"
            " cached_tokens, cost_usd, latency_s, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (job_id, provider, model, input_tokens, output_tokens, cached_tokens,
             cost_usd, latency_s, time.time()),
        )


def usage_summary() -> dict:
    """Real unit economics rather than estimates."""
    with get_db() as conn:
        j = conn.execute(
            "SELECT COUNT(*) n, AVG(duration_s) avg_s, SUM(duration_s) total_s"
            " FROM jobs WHERE status='done'"
        ).fetchone()
        l = conn.execute(
            "SELECT COUNT(*) n, SUM(cost_usd) cost, SUM(input_tokens) inp, SUM(output_tokens) outp"
            " FROM llm_calls"
        ).fetchone()
        u = conn.execute("SELECT COUNT(*) n FROM users").fetchone()
        return {
            "users": u["n"],
            "generations_done": j["n"] or 0,
            "avg_generation_s": round(j["avg_s"], 3) if j["avg_s"] else None,
            "total_compute_s": round(j["total_s"], 1) if j["total_s"] else 0,
            "llm_calls": l["n"] or 0,
            "llm_cost_usd": round(l["cost"], 6) if l["cost"] else 0.0,
            "llm_input_tokens": l["inp"] or 0,
            "llm_output_tokens": l["outp"] or 0,
        }
