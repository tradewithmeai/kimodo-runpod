"""
Local smoke test for the MDM install on the RTX 2070 SUPER.
Run:  set MDM_REPO=D:\Documents\11Projects\Kimodo\local\motion-diffusion-model
      venv\Scripts\python -W ignore smoke_test.py
Measures: CUDA sanity, load time, cold/warm generation time, peak VRAM,
and MPJPE against the pod (3090) reference output at the same seed.
"""
import os, sys, time
from pathlib import Path

LOCAL = Path(r"D:\Documents\11Projects\Kimodo\local")
os.environ.setdefault("MDM_REPO", str(LOCAL / "motion-diffusion-model"))

import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())

sys.path.insert(0, str(LOCAL / "app"))
import numpy as np
import motion_server as ms

ckpt = Path(os.environ["MDM_REPO"]) / "save" / "humanml_enc_512_50steps" / "model000750000.pt"
t0 = time.time()
m = ms.MotionModel(ckpt)
print(f"load: {time.time()-t0:.2f}s")

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
j = m.generate("a person walks forward and then turns around", seconds=6.0, guidance=2.5, seed=1234)
cold = time.time() - t0
t0 = time.time()
j2 = m.generate("a person walks forward and then turns around", seconds=6.0, guidance=2.5, seed=1234)
warm = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024**2
print(f"gen cold: {cold:.2f}s  warm: {warm:.2f}s  peak VRAM: {peak:.1f} MB")
print("output:", j.shape, j.dtype)
print("determinism rerun max|diff|:", float(np.abs(j - j2).max()))

ref = np.load(LOCAL / "ref_walk_turn_1234.npy")
d = np.linalg.norm(j - ref, axis=-1)
print(f"vs 3090 reference: MPJPE {d.mean()*1000:.2f} mm  max {d.max()*1000:.2f} mm")
