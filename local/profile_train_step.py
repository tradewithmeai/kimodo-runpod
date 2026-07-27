"""
Where does an MDM training step actually spend its time?

The question this answers: would renting a much larger GPU speed up a full retrain?
That depends entirely on whether the step is GPU-compute-bound. If pure forward+backward
is a small fraction of the real per-step wall time, a bigger card buys almost nothing —
the time is going to CLIP text encoding, data loading, and Python/kernel-launch overhead,
none of which scale with VRAM or SM count.

Run:  venv\Scripts\python.exe -W ignore profile_train_step.py
"""

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "app"))
import motion_server as ms  # noqa: E402

REPO = Path(os.environ.get("MDM_REPO", r"D:\Documents\11Projects\Kimodo\local\motion-diffusion-model"))
CKPT = REPO / "save" / "humanml_enc_512_50steps" / "model000750000.pt"

BATCH = 64      # MDM's training batch size
FRAMES = 196    # HumanML3D training horizon
NJOINTS = 263   # HML vector width
WARMUP = 3
ITERS = 20


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


def main():
    m = ms.MotionModel(CKPT)
    model, diffusion = m.model, m.diffusion
    dev = m.device
    print(f"\ndevice: {torch.cuda.get_device_name(0)}")
    print(f"batch={BATCH} frames={FRAMES}\n")

    captions = [f"a person walks forward and waves number {i}" for i in range(BATCH)]

    # ---- 1. CLIP text encoding (runs EVERY training step in stock MDM) -------
    def clip_step():
        with torch.no_grad():
            model.encode_text(captions)

    t_clip = bench(clip_step)

    # ---- 2. MDM forward + backward (the part a bigger GPU accelerates) -------
    x = torch.randn(BATCH, NJOINTS, 1, FRAMES, device=dev)
    t = torch.randint(0, diffusion.num_timesteps, (BATCH,), device=dev)
    mask = torch.ones(BATCH, 1, 1, FRAMES, dtype=torch.bool, device=dev)

    # Pre-encode once so this measures the diffusion transformer alone.
    with torch.no_grad():
        emb = model.encode_text(captions)

    for p in model.parameters():
        p.requires_grad_(True)

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        kw = {"y": {"mask": mask, "lengths": torch.tensor([FRAMES] * BATCH, device=dev),
                    "text": captions}}
        losses = diffusion.training_losses(model, x, t, model_kwargs=kw)
        losses["loss"].mean().backward()

    try:
        t_full = bench(fwd_bwd, iters=10)
    except Exception as e:
        print(f"training_losses path failed ({e.__class__.__name__}: {e})")
        print("falling back to raw forward+backward")

        def raw():
            model.zero_grad(set_to_none=True)
            kw = {"y": {"mask": mask, "lengths": torch.tensor([FRAMES] * BATCH, device=dev),
                        "text": captions}}
            out = model(x, t, **kw)
            out.float().pow(2).mean().backward()

        t_full = bench(raw, iters=10)

    peak = torch.cuda.max_memory_allocated() / 1024**2

    print("--- per-step timings (seconds) ---")
    print(f"CLIP text encode only     {t_clip:8.4f}")
    print(f"full train step           {t_full:8.4f}")
    print(f"  of which CLIP           {t_clip / t_full * 100:7.1f}%")
    print(f"  of which MDM compute    {(t_full - t_clip) / t_full * 100:7.1f}%")
    print(f"\npeak VRAM this batch      {peak:8.1f} MB  of 8192 MB")
    print(f"headroom for larger batch {8192 / max(peak, 1):8.1f}x")

    steps = 750_000
    print(f"\n--- extrapolated to {steps:,} steps (the shipped checkpoint) ---")
    print(f"at measured rate          {t_full * steps / 3600:8.1f} GPU-hours")
    print(f"if CLIP were precomputed  {(t_full - t_clip) * steps / 3600:8.1f} GPU-hours")
    print(f"saving                    {t_clip * steps / 3600:8.1f} GPU-hours")


if __name__ == "__main__":
    main()
