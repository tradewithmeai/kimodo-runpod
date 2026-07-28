"""
Job queue and the single inference worker.

Why a queue at all: a generation costs ~5.8 CPU-seconds on one vCPU. Served
synchronously, a handful of concurrent requests would saturate a small VPS and take the
web process - and any co-hosted site - down with it. The browser therefore submits a job,
gets an id back immediately, and polls.

Why exactly one worker: the model holds ~1.2GB resident and is loaded once at worker
start (cold start ~5.4s). A second worker would double the memory for no throughput gain
on a 2-vCPU box, since each generation already uses the capped thread budget.
"""

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from . import config, db

log = logging.getLogger("linel.jobs")

_queue: "queue.Queue[str]" = queue.Queue()
_worker: Optional[threading.Thread] = None
_model = None
_model_lock = threading.Lock()
_state = {"model_loaded": False, "loading": False, "load_error": None, "current_job": None}


def worker_state() -> dict:
    return {**_state, "queue_depth": _queue.qsize()}


def _load_model():
    """
    Import and load lazily, inside the worker thread.

    motion_server performs an os.chdir() at import time (MDM resolves some asset paths
    relative to its own repo root), so importing it at module scope would move the whole
    process's working directory out from under the web app. Doing it here keeps that
    side effect contained to the point where it is actually needed.
    """
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        _state["loading"] = True
        try:
            import torch

            torch.set_num_threads(config.TORCH_THREADS)

            from motion_server import MotionModel  # type: ignore

            ckpt = config.MODEL_CHECKPOINT
            if not ckpt:
                repo = Path(config.MDM_REPO)
                ckpt = repo / "save" / "humanml_enc_512_50steps" / "model000750000.pt"
            t0 = time.perf_counter()
            _model = MotionModel(Path(ckpt), device=config.MODEL_DEVICE)
            log.info("model loaded in %.1fs from %s", time.perf_counter() - t0, ckpt)
            _state["model_loaded"] = True
            _state["load_error"] = None
        except Exception as exc:  # surfaced via /health rather than crashing the worker
            _state["load_error"] = f"{exc.__class__.__name__}: {exc}"
            log.exception("model load failed")
            raise
        finally:
            _state["loading"] = False
    return _model


def _run_job(job_id: str) -> None:
    job = db.get_job(job_id)
    if not job or job["status"] != "queued":
        return  # cancelled or already handled

    db.update_job(job_id, status="running", started_at=time.time())
    _state["current_job"] = job_id
    t0 = time.perf_counter()
    try:
        model = _load_model()

        # The prompt actually sent to the model: the LLM-interpreted form when one
        # exists, otherwise the raw user text.
        prompt = job["interpreted_prompt"] or job["prompt"]

        joints = model.generate(
            prompt,
            seconds=job["seconds"],
            guidance=job["guidance"],
            seed=job["seed"],
        )
        duration = time.perf_counter() - t0

        payload = {
            "job_id": job_id,
            "prompt": job["prompt"],
            "interpreted_prompt": job["interpreted_prompt"],
            "fps": 20,
            "frames": int(joints.shape[0]),
            "joint_names": _joint_names(model),
            "chains": _chains(model),
            "seconds": job["seconds"],
            "guidance": job["guidance"],
            "seed": job["seed"],
            "generated_in_s": round(duration, 3),
            "joints": [[[round(float(v), 4) for v in j] for j in frame] for frame in joints],
        }

        out = config.OUTPUT_DIR / f"{job_id}.json"
        out.write_text(json.dumps(payload), encoding="utf-8")

        db.update_job(
            job_id,
            status="done",
            frames=int(joints.shape[0]),
            output_path=str(out),
            output_bytes=out.stat().st_size,
            finished_at=time.time(),
            duration_s=round(duration, 3),
        )
        log.info("job %s done in %.2fs (%d frames)", job_id, duration, joints.shape[0])

    except Exception as exc:
        log.exception("job %s failed", job_id)
        db.update_job(
            job_id,
            status="failed",
            error=f"{exc.__class__.__name__}: {exc}"[:500],
            finished_at=time.time(),
            duration_s=round(time.perf_counter() - t0, 3),
        )
        # Never charge for work we failed to deliver.
        if job["credits_charged"]:
            db.adjust_credits(job["user_id"], job["credits_charged"], "refund:failed", job_id)
            db.update_job(job_id, credits_charged=0)
    finally:
        _state["current_job"] = None


def _worker_loop() -> None:
    log.info("worker started")
    while True:
        job_id = _queue.get()
        if job_id is None:  # shutdown sentinel
            break
        try:
            _run_job(job_id)
        except Exception:
            log.exception("worker loop error on %s", job_id)
        finally:
            _queue.task_done()


def start_worker(preload: bool = False) -> None:
    global _worker
    if _worker and _worker.is_alive():
        return
    _worker = threading.Thread(target=_worker_loop, name="linel-worker", daemon=True)
    _worker.start()
    if preload:
        threading.Thread(target=_safe_preload, name="linel-preload", daemon=True).start()


def _safe_preload() -> None:
    try:
        _load_model()
    except Exception:
        pass  # already recorded in _state["load_error"]


def enqueue(job_id: str) -> None:
    _queue.put(job_id)


def requeue_pending() -> int:
    """Re-enqueue jobs left 'queued' in the DB (e.g. after a clean restart)."""
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status='queued' ORDER BY queued_at"
        ).fetchall()
    for r in rows:
        _queue.put(r["id"])
    return len(rows)


# --- skeleton metadata -------------------------------------------------------
# Read from the model module so that when the 31-joint Skelator model lands, these
# follow automatically rather than being hardcoded here as well.

def _joint_names(model) -> list[str]:
    try:
        import motion_server  # type: ignore
        return list(motion_server.JOINT_NAMES)
    except Exception:
        return []


def _chains(model) -> list[list[int]]:
    try:
        import motion_server  # type: ignore
        return [list(c) for c in motion_server.KINEMATIC_CHAINS]
    except Exception:
        return []
