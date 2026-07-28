"""
linel.uk - text-to-motion SaaS.

The raw model is never exposed. Every generation goes through an authenticated,
credit-checked, queued endpoint so that abuse and cost are bounded by construction
rather than by hoping nobody finds the URL.
"""

import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Make the existing motion_server importable without copying it.
_MOTION_APP_DIR = os.environ.get("LINEL_MOTION_APP_DIR", "")
if _MOTION_APP_DIR and _MOTION_APP_DIR not in sys.path:
    sys.path.insert(0, _MOTION_APP_DIR)

from . import auth, config, db, jobs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("linel")

app = FastAPI(title="linel.uk", docs_url=None, redoc_url=None)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# three.js is vendored rather than loaded from a CDN — see web/index.html for why.
app.mount("/vendor", StaticFiles(directory=WEB_DIR / "vendor"), name="vendor")


@app.on_event("startup")
def _startup() -> None:
    config.ensure_dirs()
    db.init_db()
    reclaimed = db.reclaim_orphaned_jobs()
    if reclaimed:
        log.warning("reclaimed %d job(s) interrupted by a previous shutdown", reclaimed)
    jobs.start_worker(preload=True)
    log.info("config: %s", config.summary())


# --- pages -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict:
    ws = jobs.worker_state()
    return {
        "ok": True,
        "model_loaded": ws["model_loaded"],
        "loading": ws["loading"],
        "load_error": ws["load_error"],
        "queue_depth": db.queue_depth(),
        "current_job": ws["current_job"],
    }


# --- auth --------------------------------------------------------------------

class GoogleLogin(BaseModel):
    credential: str


class DevLogin(BaseModel):
    username: str = "dev"


@app.get("/api/me")
def me(request: Request) -> dict:
    user = auth.current_user(request)
    return {
        "user": auth.public_user(user) if user else None,
        "google_client_id": config.GOOGLE_CLIENT_ID or None,
        "dev_login": config.DEV_LOGIN_ENABLED and not config.GOOGLE_CLIENT_ID,
        "credits_per_generation": config.CREDITS_PER_GENERATION,
        "max_seconds": config.MAX_SECONDS,
    }


@app.post("/api/auth/google")
def auth_google(body: GoogleLogin, response: Response) -> dict:
    user = auth.login_with_google(body.credential, response)
    return {"user": auth.public_user(user)}


@app.post("/api/auth/dev")
def auth_dev(body: DevLogin, response: Response) -> dict:
    user = auth.login_dev(body.username, response)
    return {"user": auth.public_user(user)}


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response) -> dict:
    auth.logout(request, response)
    return {"ok": True}


# --- generation --------------------------------------------------------------

class SubmitJob(BaseModel):
    prompt: str = Field(min_length=1)
    seconds: float = 6.0
    guidance: float = 2.5
    seed: int | None = None


@app.post("/api/jobs")
def submit_job(body: SubmitJob, request: Request) -> dict:
    user = auth.require_user(request)

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(400, "prompt is empty")
    if len(prompt) > config.MAX_PROMPT_CHARS:
        raise HTTPException(400, f"prompt exceeds {config.MAX_PROMPT_CHARS} characters")

    seconds = max(config.MIN_SECONDS, min(float(body.seconds), config.MAX_SECONDS))
    guidance = max(1.0, min(float(body.guidance), 10.0))

    # Per-user concurrency: one generation at a time keeps a single worker fair.
    if db.count_active_jobs(user["id"]) >= config.MAX_JOBS_PER_USER_QUEUED:
        raise HTTPException(429, "you already have a generation in progress")
    if db.queue_depth() >= config.MAX_QUEUE_DEPTH:
        raise HTTPException(503, "the queue is full, try again shortly")

    cost = config.CREDITS_PER_GENERATION
    if user["credits"] < cost:
        raise HTTPException(402, "not enough credits")

    job_id = db.new_job_id()
    db.create_job(job_id, user["id"], prompt, seconds, guidance, body.seed)
    # Charge on submit, refund on failure - so a user cannot flood the queue by
    # submitting faster than we can bill.
    db.adjust_credits(user["id"], -cost, "generation", job_id)
    db.update_job(job_id, credits_charged=cost)
    jobs.enqueue(job_id)

    return {"job_id": job_id, "status": "queued", "credits": user["credits"] - cost}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, request: Request) -> dict:
    user = auth.require_user(request)
    job = db.get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "job not found")  # same error for both: no enumeration
    return _public_job(job)


@app.get("/api/jobs")
def list_jobs(request: Request) -> dict:
    user = auth.require_user(request)
    return {"jobs": [_public_job(j) for j in db.list_jobs(user["id"])]}


@app.get("/api/jobs/{job_id}/motion")
def job_motion(job_id: str, request: Request):
    user = auth.require_user(request)
    job = db.get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "job not found")
    if job["status"] != "done" or not job["output_path"]:
        raise HTTPException(409, f"job is {job['status']}")
    path = Path(job["output_path"])
    if not path.exists():
        raise HTTPException(410, "output has been cleaned up")
    return FileResponse(path, media_type="application/json")


def _public_job(job: dict) -> dict:
    waited = None
    if job["started_at"]:
        waited = round(job["started_at"] - job["queued_at"], 2)
    return {
        "job_id": job["id"],
        "status": job["status"],
        "prompt": job["prompt"],
        "seconds": job["seconds"],
        "guidance": job["guidance"],
        "frames": job["frames"],
        "error": job["error"],
        "queued_at": job["queued_at"],
        "waited_s": waited,
        "duration_s": job["duration_s"],
    }


# --- admin -------------------------------------------------------------------

@app.get("/api/admin/usage")
def admin_usage(request: Request) -> dict:
    auth.require_admin(request)
    return {"usage": db.usage_summary(), "config": config.summary(),
            "worker": jobs.worker_state()}


class GrantCredits(BaseModel):
    user_id: int
    amount: int
    reason: str = "manual grant"


@app.post("/api/admin/credits")
def admin_grant(body: GrantCredits, request: Request) -> dict:
    auth.require_admin(request)
    if not db.get_user(body.user_id):
        raise HTTPException(404, "user not found")
    balance = db.adjust_credits(body.user_id, body.amount, body.reason)
    return {"user_id": body.user_id, "credits": balance}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
