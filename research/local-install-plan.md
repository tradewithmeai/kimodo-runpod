# Local install plan — RTX 2070 SUPER / Windows 11

(The workflow's planning agent died mid-download; this plan is written from the verified
probe findings plus the pod build history. Target: the same motion service + viewer running
locally, then the chat layer in front of it.)

## Hardware reality check (from measured numbers)

| | Pod (3090) | Local (2070 SUPER, sm_75) |
|---|---|---|
| Peak VRAM (image tower stubbed) | 261 MB | ~261 MB — 3% of the 8 GB card |
| 6 s generation, warm | 0.40 s | ~0.8–1.2 s expected (roughly 40% of 3090 throughput) |
| Arch concerns | — | none found: fp16 fine on Turing, no arch-gated kernels in sampling path; avoid bf16 |

## Steps

1. **Python 3.12** — check `py -0` first; if absent: `winget install Python.Python.3.12`.
   (3.12 matches the pod env exactly; 3.7 from MDM's ancient environment.yml is NOT needed.)
2. **venv + torch** — `py -3.12 -m venv .venv`, then install torch 2.8.0 from the
   `cu126` index (`pip install torch --index-url https://download.pytorch.org/whl/cu126`).
   cu126 wheels definitively include sm_75; cu128 (pod parity) likely fine but cu126 is the
   conservative choice for Turing. Verify: `torch.cuda.get_device_capability()` → (7, 5).
3. **Deps — the exact working set from the pod, chumpy-free:**
   `pip install "numpy" scipy einops smplx blobfile ftfy regex fastapi "uvicorn[standard]" gdown`
   then `pip install git+https://github.com/openai/CLIP.git`.
   Do NOT install chumpy (uninstallable on py3.12; the server stubs the only path needing it).
4. **Clone MDM** — `git clone --depth 1 https://github.com/GuyTevet/motion-diffusion-model`.
   No dataset download, no SMPL/body_models download (server bypasses both).
5. **Weights** — checkpoint via `gdown 1cfadR1eZ116TIdXK7qDX1RugAerEiJXr` (207 MB zip →
   `save/humanml_enc_512_50steps/`). CLIP ViT-B/32 auto-downloads on first run (338 MB,
   cached in `%USERPROFILE%/.cache/clip`).
6. **Server** — `pod/motion_server.py` + `viewer.html`, with the hardcoded `/workspace`
   REPO path made configurable (env var `MDM_REPO`); port stays 8888 locally.
7. **Verify** — walk prompt must travel ~4.8 m / 6 s (proves the std fix is in);
   `stands still` ≤ 0.05 m (proves no treadmill drift).

## Known traps (all hit on the pod; will recur locally)

- chumpy breaks any pip batch containing it — keep it out entirely.
- MDM's `.to()` returns None (`_apply` override) — never chain `.to(device).eval()`.
- `gdown --fuzzy` flag no longer exists — use bare file IDs.
- Deleting `clip_model.visual` outright breaks `encode_text` (CLIP's `.dtype` property
  reads `visual.conv1.weight.dtype`) — use the `_VisualStub` already in the server.
- `y['text_embed']` is silently overwritten when `y['text']` is present — relevant only if
  the embedding-injection path is ever used locally.

## Then: the chat layer

Browser chat panel → local proxy (Node or Python, holds `ANTHROPIC_API_KEY` in `.env`) →
Claude with a `generate_motion(text, seconds, guidance, seed)` tool → local server →
viewer. System prompt seeded from `sens-vocabulary-probe.md`: translate intent to explicit
body-action captions; prefer locomotion/posture vocabulary; avoid the dead fine-gesture
category; sequence multiple clips for multi-step requests.
