# MDM Reproduction Training Run — Technical Plan

**Status:** pre-launch, awaiting review
**Date:** 2026-07-27
**Author:** Kimodo project

---

## 1. Objective

Retrain Motion Diffusion Model (MDM) from scratch on HumanML3D to reproduce the
behaviour of the shipped `humanml_enc_512_50steps` checkpoint.

**The goal is to validate the training pipeline, not to improve the model.** By
targeting a known-good result we get a pass/fail signal: if the reproduction behaves
like the reference, the pipeline is trustworthy and every later experiment inherits
that trust. If it diverges, we learn that now — on a run whose correct answer we
already know — rather than during a run whose output we cannot independently judge.

Secondary outcome: we end up owning a set of weights trained entirely by us, from a
public dataset, rather than depending on a third-party checkpoint. This matters for
the licensing position (see `docs/` licensing notes).

Explicit non-goals for this run: no architecture changes, no hyperparameter search,
no dataset augmentation, no LLM-in-the-loop prompt work. Those come after.

---

## 2. Hardware, cost, and measured throughput

### Measured (not estimated)

| GPU | s/step | it/s | source |
|---|---|---|---|
| RTX 2070 SUPER (local, sm_75) | 0.417 | 2.40 | `local/profile_train_step.py`, batch 64 |
| **RTX 3090 (RunPod, sm_86)** | **0.173** | **5.79** | 383-step live training epoch |

The 2070 SUPER figure independently validates against the MDM paper's "~3 days on a
2080 Ti" (750k × 0.417 s = 86.8 h = 3.6 days), which gives confidence the
extrapolation is sound.

### Projected run cost

| Item | Value |
|---|---|
| Steps | 600,000 (per reference `args.json`) |
| Pure training compute | 600,000 × 0.173 s = **28.8 h** |
| Periodic eval overhead | 12 evals × (1000 samples × 3 reps) — **TBD, measured in validation** |
| Dataset load (per process start) | ~3.5 min, one-off |
| GPU rate | $0.50/hr (RTX 3090, EU-CZ-1, secure cloud) |
| **Estimated total** | **~29–35 h, ~$15–18** |

### Why not a larger GPU

An 80 GB card at $4.99/hr was considered and rejected:

- Extrapolating the measured 2070S→3090 scaling (2.4×), an 80 GB card is likely
  ~2× the 3090 at this batch size → ~14 h, ~$70. **~4× worse value for ~15 h saved.**
- The large card's real advantage is batch size (512+ vs 64). Raising batch size
  changes gradient noise and requires LR re-tuning — i.e. it changes the recipe,
  which directly defeats the objective in §1.
- At batch 64 a large card is badly under-utilised; the speedup would come from clock
  and memory bandwidth only, not from occupancy.

**Decision: RTX 3090 at batch 64.** Revisit large hardware for the *next* run, where
a changed recipe is the point rather than a contaminant.

---

## 3. Exact training recipe

Taken verbatim from the reference checkpoint's `args.json` unless flagged in §6.

### Model architecture
| Parameter | Value |
|---|---|
| `arch` | `trans_enc` (transformer encoder) |
| `latent_dim` | 512 |
| `layers` | 8 |
| `num_heads` | 4 (default) |
| `ff_size` | 1024 (default) |
| `dropout` | 0.1 (default) |
| `text_encoder_type` | `clip` (CLIP ViT-B/32, frozen) |
| `emb_trans_dec` | False |
| Trainable params | 17.9 M (+131.4 M frozen CLIP) |

### Diffusion
| Parameter | Value |
|---|---|
| `diffusion_steps` | 50 |
| `noise_schedule` | `cosine` |
| `sigma_small` | True (default) |
| `cond_mask_prob` | 0.1 (enables classifier-free guidance) |

### Optimisation
| Parameter | Value |
|---|---|
| `batch_size` | 64 |
| `lr` | 1e-4 |
| `weight_decay` | 0.0 |
| `adam_beta2` | 0.999 |
| `num_steps` | 600,000 |
| `seed` | 10 |

### Loss weights
| Parameter | Value | Note |
|---|---|---|
| `lambda_vel` | 0.0 | disabled |
| `lambda_rcxyz` | 0.0 | disabled |
| `lambda_fc` | 0.0 | foot-contact loss disabled |

All three geometric losses are zero, so the loss is plain MSE on the 263-dim HML
vector. **Consequence:** `Rotation2xyz` / SMPL is never invoked in the training math.

### Periodic behaviour
| Parameter | Value |
|---|---|
| `save_interval` | 50,000 steps (→ 12 checkpoints) |
| `log_interval` | 1,000 steps |
| `eval_during_training` | **True** |
| `eval_split` | `test` |
| `eval_batch_size` | 32 |
| `eval_num_samples` | 1,000 |
| `eval_rep_times` | 3 |
| `gen_during_training` | **True** |
| `gen_num_samples` | 3 |
| `gen_num_repetitions` | 2 |
| `gen_guidance_param` | 2.5 |

### Launch command

```bash
python -m train.train_mdm \
  --save_dir /workspace/checkpoints/repro_full \
  --dataset humanml --arch trans_enc \
  --diffusion_steps 50 --noise_schedule cosine \
  --latent_dim 512 --layers 8 \
  --batch_size 64 --lr 1e-4 --weight_decay 0.0 \
  --cond_mask_prob 0.1 --seed 10 \
  --num_steps 600000 --save_interval 50000 --log_interval 1000 \
  --eval_during_training --eval_split test \
  --eval_batch_size 32 --eval_num_samples 1000 --eval_rep_times 3 \
  --gen_during_training --gen_num_samples 3 --gen_num_repetitions 2 \
  --gen_guidance_param 2.5 \
  --train_platform_type TensorboardPlatform \
  --overwrite
```

---

## 4. Data provenance and preparation

### Dataset: HumanML3D

| Component | Count / size | Source |
|---|---|---|
| `new_joint_vecs/*.npy` (263-dim motion vectors) | 29,228 files, 4.30 GB archive | Google Drive folder `1OZrTlAGRvLjXhXwnRiOC-oxYry1vf-Uu` |
| `new_joints/*.npy` (22×3 joint positions) | 29,228 files | same |
| `texts/*.txt` (captions) | 29,232 files | **GitHub** `EricGuo5513/HumanML3D` → `HumanML3D/texts.zip` |
| `Mean.npy` / `Std.npy` | normalisation stats | Drive folder |
| `train.txt` / `val.txt` / `test.txt` | split definitions | Drive folder |
| Train entries actually loaded | **23,384** | verified in live run |

File counts are ~2× the 14,616 base motions because HumanML3D includes left-right
**mirrored** copies (`M######.npy`) as augmentation. This is part of the standard
dataset, not duplication on our side.

`animations.rar` (2.1 GB of preview videos) is **deliberately not downloaded** — it is
never read by training.

### Auxiliary assets
| Asset | Purpose | Source |
|---|---|---|
| `glove/` | tokeniser resources for the eval text encoder | Drive `1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n` |
| `t2m/` | pretrained T2M evaluators (FID / R-precision / diversity) | Drive `1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb` |
| `body_models/smpl/` | SMPL neutral model | MDM `prepare/download_smpl_files.sh` |
| CLIP ViT-B/32 | frozen text encoder | downloaded by OpenAI CLIP package at init |

### Preparation gotchas already solved

1. **RAR5 vs p7zip — silent data loss.** `p7zip 16.02` (the version in Ubuntu 22.04)
   cannot read RAR5. It exits 0 and creates all 29,228 entries as **0-byte files**.
   The dataset loader then swallows every `EOFError` in a bare `except: pass` and
   reports `ValueError: not enough values to unpack (expected 2, got 0)` — an error
   that names nothing relevant. `bsdtar`/libarchive also fails partway
   (~9,912 files, then `Unsupported block header size`).
   **Fix:** official RARLab `unrar` 7.12 from `rarlab.com`. Verify with
   `find new_joint_vecs -name '*.npy' -size 0 | wc -l` → must be 0.
2. **Captions are not in the Drive folder.** They must come from the HumanML3D GitHub
   repo's `texts.zip`.
3. **`chumpy` on Python 3.12.** Its `setup.py` does `import pip`, which fails under
   pip's build isolation. **Fix:** `pip install --no-build-isolation
   git+https://github.com/mattloper/chumpy` — installs and imports cleanly as 0.71.
   (This also retires the `Rotation2xyz` stub used in our inference server.)
4. **Undeclared training-only deps:** `spacy`, `joblib`, `moviepy<2` (v2 removed
   `moviepy.editor`), `imageio-ffmpeg`, `tensorboard`, `wandb`.

---

## 5. Environment

| Component | Version |
|---|---|
| Host | RunPod, RTX 3090 24 GB, 32 vCPU, 125 GB RAM, EU-CZ-1 |
| Image | `runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204` |
| Python | 3.12 |
| torch | 2.8.0+cu128 |
| CUDA | 12.8, driver-compatible with sm_86 |
| venv | `/workspace/envs/mdm` (on the network volume, survives pod loss) |
| MDM repo | `/workspace/repos/motion-diffusion-model` |
| Dataset | `/workspace/datasets/HumanML3D`, symlinked to `repo/dataset/HumanML3D` |
| Checkpoints | `/workspace/checkpoints/repro_full` |

Everything the run depends on lives on the **200 GB network volume**, so a pod loss
does not destroy the environment — only the running process.

---

## 6. Deviations from the reference recipe

Full disclosure of every difference between this run and the original.

| # | Item | Reference | Ours | Justification |
|---|---|---|---|---|
| 1 | `train_platform_type` | `WandBPlatform` | `TensorboardPlatform` | Logging backend only; no effect on training math. Avoids a third-party account dependency and keeps metrics on our volume. |
| 2 | `num_steps` vs checkpoint name | `args.json` says 600,000; shipped file is `model000750000.pt` | 600,000 | **Unresolved discrepancy** — see §9. The reference may have been resumed and extended beyond its recorded config. We follow the recorded config. |
| 3 | Library versions | 2022-era (torch 1.7, Python 3.7) | torch 2.8, Python 3.12 | Original env is not installable on modern CUDA/sm_86. This is the single largest source of possible numerical divergence. |

Nothing else differs. Architecture, diffusion, optimiser, batch size, seed, loss
weights, eval and generation settings are byte-identical to the reference `args.json`.

---

## 7. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Crash at first `save_interval` (eval/gen path untested) | High — would waste ~2.4 h | **Pre-flight validation run** exercising eval + gen + save at `num_steps 60`, `save_interval 50`. Full run launches only if this passes. |
| Pod terminated / preempted mid-run | High | Checkpoints (model + optimizer state) every 50k steps to the network volume. Resume with `--resume_checkpoint <path>`. Non-interruptible (on-demand, not spot) instance chosen specifically to reduce this. |
| SSH disconnect kills the process | High | Launched under `setsid nohup`, fully detached from the session. Verified working for all prior long-running jobs. |
| Divergence from reference due to library versions | Medium | Cannot be eliminated. Detected by the §9 acceptance criteria — this is precisely what the run is designed to surface. |
| Disk exhaustion | Low | 12 checkpoints × 225 MB ≈ 2.7 GB, plus samples. Volume is 200 GB with ~195 GB free. |
| Cost overrun from an idle finished pod | Medium | Run duration is known (~30 h); check on completion and stop the pod. Billing is $0.50/hr = $12/day if forgotten. |
| RTX 3090 stock unavailable on recreate | Medium | 3090 is Low stock and EU-CZ-1 is the only DC carrying it. If the pod is lost we may have to wait or fall back to a 4090 at $0.69/hr (which would also change step timing). |

---

## 8. Execution and monitoring

1. **Pre-flight validation** (in progress): eval + gen + save paths at tiny scale.
2. **Launch** full run detached via `setsid nohup`, logging to
   `/workspace/train_full.log`.
3. **Monitor** without holding a connection:
   - `grep "step\[" train_full.log | tail` — loss trajectory
   - checkpoint files appearing every 50k steps
   - TensorBoard event files on the volume
4. **Checkpoint retention:** keep all 12. They are the evidence of the loss curve and
   allow post-hoc evaluation at any point in training.
5. **On completion:** stop the pod, back up checkpoints, run acceptance tests (§9).

---

## 9. Acceptance criteria — how we decide the run succeeded

The run is a **test**, so it needs a defined pass condition. Proposed:

**Quantitative (primary).** Compare final-checkpoint T2M evaluator metrics against the
reference checkpoint, evaluated identically on the `test` split:
- FID
- R-precision (top-1/2/3)
- Diversity
- MultiModality

Pass if within a defined tolerance of the reference. **Open question: what tolerance?**
Diffusion training has run-to-run variance even with a fixed seed (cuDNN
non-determinism, different library versions). A sensible starting point is ±10 % on
FID and ±0.02 absolute on R-precision, but this should be sanity-checked by review.

**Qualitative (secondary).** Generate a fixed prompt set through both our checkpoint
and the reference, and compare visually in the viewer for behavioural equivalence.

**Diagnostic (continuous).** Loss curve shape should track the reference's known
trajectory. Test-run loss at step 375 was 0.384, which is a plausible early value.

**Note on the 600k/750k discrepancy (§6.2):** if our 600k result underperforms the
reference, the most likely explanation is simply that the reference had 150k more
steps, not that our pipeline is broken. Evaluating our own 550k checkpoint against our
600k one will show whether the curve had flattened — if it had, the step count is not
the explanation and something else differs.

---

## 10. Questions for review

1. **Acceptance tolerance** — what FID / R-precision deltas should count as a
   successful reproduction, given a 4-year library-version gap?
2. **600k vs 750k** — follow the recorded config (current plan) or match the shipped
   checkpoint's step count? The latter costs ~$4 more and ~7 h.
3. **Determinism** — worth setting `torch.use_deterministic_algorithms(True)` and
   `cudnn.deterministic`? It costs throughput but makes the run re-runnable exactly.
   Current plan does not, matching the reference.
4. **Eval overhead** — is 12 full evals (1000 samples × 3 reps) the right cadence, or
   is it worth reducing `eval_rep_times` to 1 during training and doing the full
   3-rep evaluation once, post-hoc, from saved checkpoints?
5. **Anything in §6 that should be considered a material deviation** rather than a
   benign one.
