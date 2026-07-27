# SaaS hosting decision — measured, and two corrections to the roadmap

Companion to `docs/saas-roadmap.md`. That document names the right decisive metric;
this one supplies the measurement, answers Krystal-vs-VPS, and corrects two factual
premises — one of which is materially more serious than the issue the roadmap raises.

---

## 1. The benchmark (measured 2026-07-27, not estimated)

Run on the local RTX 2070 SUPER machine with **CUDA disabled and torch threads capped**
to simulate cheap-VPS core counts. Script: `local/bench_cpu_saas.py`.

| Metric | 1 vCPU | 2 vCPU |
|---|---|---|
| Cold start (import + model load) | 5.4 s | 6.2 s |
| **Peak process RSS** | **1177 MB** | **1180 MB** |
| Typical 6 s animation | **5.85 s** | **3.49 s** |
| 2 s animation | 9.9 s* | 5.9 s* |
| 9.8 s animation (max) | ~9 s | 5.1 s |
| RSS drift over 9 consecutive generations | **−200 MB** | **−201 MB** |

\* first timed call includes lazy CUDA/kernel warm-up; steady-state is the leak-check row.

**Steady state at 1 vCPU: 5.81, 5.85, 5.86, 5.86, 5.85 s** — variance under 1%.

### Interpretation against the roadmap's own thresholds

The roadmap says "under 30 seconds per ordinary generation: excellent shared SaaS
candidate". We are at **5.85 s on a single core** — roughly 5× inside the best bracket.

**Memory does not leak.** RSS is flat across nine generations and actually *drops* ~200 MB
after load as allocator arenas settle. The process is stable enough to run indefinitely.

### What this means economically

At 5.85 CPU-seconds per generation, one **1 vCPU** core delivers ~615 generations/hour,
or ~450,000/month at full utilisation. Even at 10% practical utilisation that is
**~45,000 generations per month from a single core**.

**Motion generation is effectively free.** The compute cost per animation is on the order
of $0.00002. Two consequences:

1. The £12/year founder plan is not constrained by inference cost at all.
2. **The real marginal cost is the LLM interpretation call, not the model.** A single
   prompt-rewriting call at Haiku-class pricing (~500 in / 100 out tokens) costs roughly
   $0.001 — about **60× more than the motion generation it triggers**. Cost control
   belongs in the LLM layer: cache aggressively, keep the system prompt short and
   cacheable, and only escalate to a larger model on request.

---

## 2. Krystal shared hosting vs VPS

**Verdict: VPS. Krystal's shared/cPanel hosting cannot run this, and no amount of
optimisation will change that.**

This is not about our numbers being too big — they are small. It is about what shared
hosting fundamentally *is*:

| Requirement | Shared hosting reality |
|---|---|
| A persistent Python process resident for hours | Shared hosting runs short-lived PHP requests; long-running daemons are prohibited |
| ~1.2 GB RSS held continuously | Per-account memory limits are typically a fraction of this, and it's held *permanently*, not per request |
| Install PyTorch + CUDA-less native wheels (~2–3 GB) | No arbitrary native package installation |
| A job queue and background worker | No process supervisor available |
| Model load once, serve many (cold start 5.4 s) | Reloading per request would make every generation ~11 s and thrash the host |

The blocker is the **persistent 1.2 GB process**, not the per-generation cost. A shared
host would either kill the worker or throttle it into uselessness, and it would put your
other Krystal-hosted sites at risk — which the roadmap rightly flags as a test criterion.

### Recommended VPS spec

| Resource | Minimum | Comfortable |
|---|---|---|
| vCPU | 1 | **2** |
| RAM | 2 GB | **4 GB** |
| Disk | 20 GB | 40 GB |

Sizing rationale: the worker needs ~1.2 GB resident. On a 2 GB box that leaves ~800 MB
for OS, Nginx, database and the web app — workable but tight, and it violates the
roadmap's own sensible rule that the worker shouldn't exceed half of usable RAM. **4 GB
satisfies that rule with room for a second worker later.** At UK prices that is roughly
£4–8/month, which the founder plan covers at a handful of subscribers.

**Do not co-locate this on a box already serving your other production sites** for the
first beta, even though it would technically fit. A separate cheap VPS makes the blast
radius zero, and at these prices the isolation is worth more than the saving.

---

## 3. Correction: there is no NVIDIA model in this project

The roadmap's "Ownership wording" section is built on a false premise. It cites
`huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1` and advises the wording *"build upon research
and open components including NVIDIA Kimodo"*. **Publishing that would be inaccurate.**

"Kimodo" is *your project name*. It is not a model, and nothing from NVIDIA is in this
stack. What we actually run:

| Component | Origin | Licence |
|---|---|---|
| Motion Diffusion Model (MDM) | Guy Tevet et al., Tel Aviv University, 2022 | **MIT** (verified in repo `LICENSE`) |
| CLIP ViT-B/32 (frozen text encoder) | OpenAI | MIT |
| HumanML3D dataset | Guo et al. | see §4 — **this is the real constraint** |
| SMPL body model | MPI | research licence, registration required |

MDM being MIT is *better* than the NVIDIA licence the roadmap analyses: MIT permits
commercial use, modification and sublicensing outright, with only attribution required.

---

## 4. The licensing issue that actually matters — and it is not the model

The roadmap misses this entirely, and it is the one item on this page that could
genuinely stop a commercial launch.

**HumanML3D is derived from AMASS. AMASS is licensed for non-commercial research use,
and SMPL carries its own research-use licence.** MDM's own README states plainly:

> "our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses
> datasets that each have their own respective licenses that must also be followed."

So the position is:

- **The code is MIT** — commercially fine.
- **The weights we are training right now are trained on HumanML3D/AMASS** — and that
  data's terms are research-oriented. Selling access to a model trained on it is at
  best unresolved, at worst a breach.

This does **not** stop the current training run — that run's purpose is pipeline
validation, which is research. It stops *commercialising those specific weights*.

### Options, roughly in order of effort

1. **Read the actual AMASS/HumanML3D terms** for the specific sub-datasets used, and
   check whether any permit commercial use. AMASS is an aggregate of many mocap corpora
   with differing terms; some may be permissive.
2. **Obtain a commercial licence** from MPI for AMASS/SMPL. This is a normal request and
   a route several companies have taken.
3. **Train on commercially-clear data.** This is where your own work becomes the moat:
   your anatomically correct skeleton plus your own captured or licensed motion gives a
   model with clean provenance and no third-party constraint at all. Slower, but it
   makes the ownership story genuinely yours rather than caveated.
4. **Ship the tooling, not the weights.** Open-source the pipeline (MIT-clean) and let
   users bring their own weights, while the hosted service waits on option 2 or 3.

**Recommendation:** keep the provenance record the roadmap asks for — starting
checkpoint, datasets, code sources, licences, exactly what was retrained — and resolve
the data licence *before* taking money, not after. The engineering is the easy part here;
this is the part that needs a decision.

---

## 5. Revised immediate plan

1. ✅ **CPU benchmark** — done, decisively positive (this document).
2. **Resolve the data licence question** (§4) — blocking for commercialisation, not for
   development.
3. **Buy `linel.uk`** and a 2 vCPU / 4 GB VPS.
4. **Deploy the inference worker privately**, bound to localhost, with a job queue.
   Re-run this same benchmark on the real VPS to confirm the numbers transfer.
5. **Then** add Google Identity Services auth, credits, and Stripe annual billing.

Steps 3–5 are straightforward given the measurements. Step 2 is the one that decides
whether this is a product or a very good portfolio piece.
