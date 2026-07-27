"""
CPU inference benchmark — decides Krystal shared hosting vs VPS, and what a
generation actually costs.

The SaaS roadmap's decisive metric: CPU-seconds and peak process RAM per animation.
A cheap VPS has 1-2 vCPU, so we cap torch threads to simulate that rather than
benchmarking on all local cores and getting a flattering, useless number.

Run:  venv\\Scripts\\python.exe -W ignore bench_cpu_saas.py
"""

import gc
import os
import sys
import time
from pathlib import Path

# Must be set before torch import to actually bind thread counts.
THREADS = int(os.environ.get("BENCH_THREADS", "2"))
os.environ["OMP_NUM_THREADS"] = str(THREADS)
os.environ["MKL_NUM_THREADS"] = str(THREADS)
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

import torch  # noqa: E402

torch.set_num_threads(THREADS)

sys.path.insert(0, str(Path(__file__).parent / "app"))

REPO = Path(os.environ.get("MDM_REPO", r"D:\Documents\11Projects\Kimodo\local\motion-diffusion-model"))
CKPT = REPO / "save" / "humanml_enc_512_50steps" / "model000750000.pt"

try:
    import psutil
    _proc = psutil.Process()

    def rss_mb():
        return _proc.memory_info().rss / 1024**2
except ImportError:
    def rss_mb():
        return float("nan")


def main():
    print(f"threads          {THREADS}")
    print(f"baseline RSS     {rss_mb():8.1f} MB")

    t0 = time.perf_counter()
    import motion_server as ms
    m = ms.MotionModel(CKPT, device="cpu")
    load_s = time.perf_counter() - t0
    peak_after_load = rss_mb()
    print(f"cold start       {load_s:8.1f} s")
    print(f"RSS after load   {peak_after_load:8.1f} MB\n")

    peak = peak_after_load
    print(f"{'seconds':>8} {'frames':>7} {'gen_s':>8} {'RSS_MB':>9}")
    print("-" * 36)

    results = []
    for secs in (2.0, 4.0, 6.0, 9.8):
        t = time.perf_counter()
        j = m.generate("a person walks forward and waves", seconds=secs, seed=1234)
        dt = time.perf_counter() - t
        r = rss_mb()
        peak = max(peak, r)
        results.append((secs, dt))
        print(f"{secs:8.1f} {j.shape[0]:7d} {dt:8.2f} {r:9.1f}")

    # Stability / leak check: repeat the typical job.
    print(f"\nleak check — 5x repeat of the 6s job")
    times = []
    for i in range(5):
        t = time.perf_counter()
        m.generate("a person sits down on a chair", seconds=6.0, seed=i)
        times.append(time.perf_counter() - t)
        peak = max(peak, rss_mb())
        print(f"  run {i+1}  {times[-1]:6.2f} s   RSS {rss_mb():8.1f} MB")

    drift = rss_mb() - peak_after_load
    typical = sorted(times)[len(times) // 2]

    print(f"\n--- summary (threads={THREADS}) ---")
    print(f"cold start            {load_s:8.1f} s")
    print(f"peak RSS              {peak:8.1f} MB")
    print(f"RSS drift over 9 gens {drift:8.1f} MB")
    print(f"typical 6s generation {typical:8.2f} s")

    verdict = (
        "excellent shared SaaS candidate" if typical < 30 else
        "usable with a queue" if typical < 120 else
        "specialist only, tighten credits" if typical < 300 else
        "unsuitable without optimisation"
    )
    print(f"verdict               {verdict}")
    print(f"\ncost per generation at 1 vCPU-hour = $0.01:  "
          f"${typical / 3600 * 0.01:.6f}")


if __name__ == "__main__":
    main()
