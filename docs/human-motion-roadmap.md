# Human Movement Simulation — Roadmap

Four separate problem domains, all wanted eventually, none started yet. Each has its own
models, its own GPU profile, and its own output format. They can share one dev pod but not
one environment — CUDA/torch/simulator requirements conflict, so plan on one conda env or
container per domain.

Target host: RunPod interactive dev pod (SSH + Jupyter), on-demand GPU.

---

## 1. Character motion / animation (text-or-music → skeletal motion)

Generative models that produce SMPL / BVH animation from a prompt. Closest to a
Blender/Unity content pipeline.

**Models to evaluate**
| Model | Input | Notes |
|---|---|---|
| MDM (Motion Diffusion Model) | text, action label | The reference baseline; slow sampling, well documented |
| MoMask | text | Masked generative transformer, much faster than MDM, better fidelity |
| MotionGPT | text, dialogue | Treats motion as a language token stream; good for instruction-style control |
| OmniControl | text + joint trajectory | Adds spatial control (put the hand *here*) on top of diffusion |
| EDGE | music | Dance generation, if music-driven movement is ever in scope |

**Data**: HumanML3D, KIT-ML, AMASS (AMASS needs a per-dataset licence registration —
do this early, it gates most of the above).

**Output**: SMPL pose sequences → BVH/FBX. Retarget onto a rig before Unity import.

**GPU**: inference fits in 16–24 GB. Fine-tuning wants 24–48 GB.

**Open questions**
- Which skeleton is canonical for this project — SMPL, Mixamo, or a custom rig?
- Do we need root-motion-correct output (foot sliding is the usual failure mode)?

---

## 2. Physics-based humanoid control (RL policies)

Learned locomotion in a physics simulator. Movement is *simulated*, not sampled — it
reacts to terrain, pushes, and contact. Longest time-to-first-result of the four.

**Stack options**
| Stack | Notes |
|---|---|
| Isaac Gym / Isaac Lab | Massively parallel envs on one GPU; NVIDIA-only, fussy install, the de-facto choice |
| MuJoCo (MJX) | Cleaner API, JAX-accelerated, easier to install, smaller env counts |
| Brax | Pure JAX, fastest to iterate, less physical fidelity |

**Methods to reproduce**
- **AMP** — adversarial motion priors; style comes from a mocap dataset, task from a reward.
- **ASE / CALM** — reusable low-level skill latents, then a high-level controller on top.
- **PHC / PHC+** — physics-based humanoid *tracking*; drives a sim humanoid to follow a
  reference motion. This is the natural bridge from domain 1 and 3 into physics.

**GPU**: this is the expensive one. Long-running (hours to days) single-GPU training,
A100/H100-class ideal, 4090 workable. Needs a persistent network volume for checkpoints —
an interactive pod alone will lose work when it's stopped.

**Open questions**
- Is the goal a controllable agent, or just physically plausible clips?
- Real-time inference in Unity later, or offline baking to animation?

---

## 3. Video → pose estimation (mocap from footage)

Recover human motion from ordinary video. Cheapest to stand up and the most immediately
useful — it produces the reference data the other two domains consume.

**Models to evaluate**
| Model | Notes |
|---|---|
| SMPLer-X | Large-scale whole-body (body + hands + face) SMPL-X regressor; strong general default |
| WHAM | World-grounded motion with global trajectory — handles a moving camera |
| 4D-Humans / HMR2.0 | Tracking over time, robust to occlusion |
| NLF / OSX | Alternatives worth a look for whole-body accuracy |

**Output**: SMPL-X parameters per frame → same retarget path as domain 1.

**GPU**: inference only, 12–24 GB is plenty. No training expected initially.

**Open questions**
- Source footage: single fixed camera, moving camera, or multi-view?
- Does hand/face detail matter, or body-only?

---

## 4. Human-like input simulation (cursor and keystrokes)

Entirely unrelated to the above — no GPU, no deep model needed for a first version.
Generating cursor trajectories and keystroke timings that look human rather than robotic.

**Approaches, cheapest first**
1. **Analytic** — Fitts's law timing + Bézier/minimum-jerk paths with overshoot, dwell,
   and micro-tremor noise. Gets most of the way there for a fraction of the effort.
2. **Statistical** — sample inter-key intervals and path curvature from recorded human
   traces (record your own; public datasets are thin and often ToS-encumbered).
3. **Learned** — a small sequence model (LSTM/tiny transformer) trained on recorded traces.
   Only worth it if 1 and 2 measurably fail a detector.

**Use case must be pinned down before building**: driving your own test suites and
load-testing your own apps is straightforward. Anything aimed at evading a third party's
bot detection is a different conversation and I'd want to know whose system it is.

**GPU**: none. This one does not belong on the RunPod at all — build it locally.

---

## Suggested order

1. **Domain 3 first** — cheapest, fastest to a result, and it generates reference motion.
2. **Domain 1 next** — reuses the same SMPL representation and retarget path.
3. **Domain 2 last** — largest cost and longest debugging tail; PHC consumes the output
   of 1 and 3, so it genuinely benefits from being third.
4. **Domain 4 in parallel, locally** — no shared code with the rest.

## Cross-cutting decisions to make once, not four times

- **Canonical skeleton and retarget path** — SMPL/SMPL-X → engine rig. Decide once.
- **Persistent storage** — a RunPod network volume for weights, AMASS, and checkpoints,
  so stopping a pod doesn't cost a redownload.
- **Env isolation** — Isaac Gym pins old Python/CUDA and will fight the diffusion stack.
  Separate envs from day one.
- **Licences** — AMASS, SMPL, and SMPL-X are research-licence gated and several of the
  models above inherit those terms. Check before anything ships.
