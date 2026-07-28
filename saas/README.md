# linel — text-to-motion SaaS

Working vertical slice: sign in → spend a credit → queued generation → 3D playback.
Runs locally today; deploys to a VPS unchanged.

## Why it is shaped this way

**Generation is a queued job, never a synchronous request.** One generation costs ~5.8
CPU-seconds on a single vCPU (measured, `local/bench_cpu_saas.py`). Served synchronously,
a handful of concurrent users would saturate a small VPS and take the web process — and
any co-hosted site — down with it. The browser submits, gets a job id, and polls.

**Exactly one inference worker.** The model holds ~1.2 GB resident and takes ~5.4 s to
load, so it is loaded once in a worker thread and reused. A second worker would double
memory for no throughput gain on a 2-vCPU box.

**The model endpoint is never public.** Every generation goes through an authenticated,
credit-checked, rate-limited endpoint.

## Run it

```bash
pip install -r requirements.txt

set MDM_REPO=...\local\motion-diffusion-model
set LINEL_MOTION_APP_DIR=...\local\app

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and use the dev sign-in. Copy `.env.example` to `.env` for
persistent settings.

## Layout

```
app/
  config.py   env-driven settings; no hardcoded paths
  db.py       SQLite: users, sessions, jobs, llm_calls, credit_events
  auth.py     Google Identity Services verification + own session cookie
  jobs.py     queue + single worker; model loaded once
  main.py     FastAPI routes
web/
  index.html  viewer, auth UI, job polling
  vendor/     three.js, self-hosted (see below)
```

## Auth

Google establishes **identity only**. Everything commercial — active, paid, credits
remaining, job ownership — lives in our database, keyed on Google's stable `sub`, never
the email (emails change hands; `sub` does not).

The browser gets a Google ID token; the backend verifies signature, audience, issuer and
expiry **server-side** via `google-auth`, then issues its own HttpOnly session cookie.
The Google token is never used as a session token.

**Dev login** exists so the service is testable before Google is configured. It refuses
to run whenever `GOOGLE_CLIENT_ID` is set, so it cannot be left enabled in production by
accident.

Before going public, set `LINEL_COOKIE_SECURE=true` (needs HTTPS) and a fixed
`LINEL_SECRET_KEY`.

## Credits

Charged on submit, **refunded automatically on failure** — so a user cannot flood the
queue by submitting faster than we bill, and never pays for work we failed to deliver.
Every change is recorded in `credit_events` with a reason.

Guards in place: one active job per user (429), global queue cap (503), prompt length
cap, clamped duration and guidance.

## Third-party scripts

**three.js is vendored** into `web/vendor/` rather than pulled from a CDN. For a service
handling auth and payments, a third-party script in the page is a supply-chain risk, and
importmap entries cannot carry SRI hashes. Self-hosting removes the problem and makes
the app work without external network access. Pinned at 0.160.0.

**Google Identity Services is the one external script** and is loaded from Google's
origin. It cannot carry an SRI hash — Google rotates the file and documents it as
unpinnable — and it is required for sign-in.

## Verified end to end

| Check | Result |
|---|---|
| Anonymous submit | 401 refused |
| Dev sign-in | user created, 20 signup credits |
| Submit | queued, credits 20 → 19 |
| Job lifecycle | queued → running → done, 0.38 s wait, 5.378 s generation |
| Motion payload | 80 frames, 22 joints, 20 fps, 5 chains |
| Second concurrent submit | 429 blocked |
| Vendored three.js | served locally, zero CDN references |
| Restart recovery | orphaned jobs failed + credits refunded on startup |

## Not built yet

- **LLM interpretation layer.** The `interpreted_prompt` column and `llm_calls` ledger
  exist and the worker already prefers `interpreted_prompt` when present — the call
  itself is the next piece. This is the real cost driver: an LLM call costs roughly
  **60× more than the generation it triggers**.
- **Stripe billing.** Plan is £1/month billed as £12 annually — twelve £1 charges would
  lose ~22% to fees.
- **BVH/FBX export** for Blender and Unity retargeting.
- **Output lifecycle** — nothing deletes old motion JSON yet.

## Deployment notes

Nothing reads a hardcoded absolute path, so the same tree runs on a VPS. Needs:
Nginx terminating TLS in front, `LINEL_COOKIE_SECURE=true`, a persistent
`LINEL_SECRET_KEY`, and a process supervisor (systemd) — the current worker dies with
the process and reclaims interrupted jobs on the next start.
