# sens-ood-phrasing — does natural user phrasing break MDM?

**Question:** how much does out-of-distribution (conversational) phrasing degrade MDM output
versus HumanML3D-style captions? This is the direct measure of what a Claude
prompt-rewriting layer would buy.

**Verdict up front:** rewriting buys **one large, unambiguous win and one small one**.

1. **Large win — abstract / meta prompts.** Prompts with no explicit body action
   ("vibes", "make it cooler", "can you do a cool animation for my game please")
   produce motion **statistically indistinguishable from the unconditional idle**.
   53% of those generations are a literal 6-second standing idle. Rewriting them into
   HumanML3D captions takes the idle rate to **0%** and doubles articulation
   (0.206 → 0.410 m/s mean joint speed).
2. **Small win — concrete but conversational prompts.** Rewriting "the guy trips over
   something" → "a person walks forward, stumbles, and catches their balance" reduces
   mean worst-bone length instability from **0.283 → 0.232** (-18%) and raises the
   physically-valid rate from **60% → 70%** of generations. Real but modest.
3. **No win at all** for the model's biggest failure source: **action content**, not
   phrasing. Sits, crouches, jumps and kicks blow the skeleton apart regardless of how
   perfectly the caption is written. Canonical HumanML3D captions
   ("a person jumps forward with both feet") are the *worst* group in the whole study.

---

## Setup

- Model: `/workspace/repos/motion-diffusion-model/save/humanml_enc_512_50steps/model000750000.pt`
  via `MotionModel.generate()`, 6.0 s (120 frames @ 20 fps), guidance 2.5.
- **3 seeds (0,1,2) per prompt**, all prompt groups sharing the same seeds so that
  same-seed comparisons are apples-to-apples. 36 prompts × 3 = **108 generations**.
- Prompt groups:
  - `natural` — 10 conversational prompts.
  - `hml_rewrite` — 10 HumanML3D-style rewrites of the *same intent*, one per natural prompt.
  - `abstract` — 5 prompts with no explicit body action ("vibes", "make it cooler", …).
  - `abstract_rewritten` — 5 HumanML3D-style rewrites of those.
  - `canonical_hml` — 5 textbook HumanML3D captions (ceiling control).
  - `empty` — `text=""`, the unconditional baseline.

### Metrics

| metric | definition | why |
|---|---|---|
| `limb_cv_max` | max over the 21 skeleton bones of (temporal std of bone length / median length) | real skeletons have **constant** bone lengths. Any value >0 is reconstruction going off-manifold. Empty-prompt reference = **0.064**. |
| `d_empty` | mean per-frame root-relative joint L2 to the **same-seed** empty-prompt motion, / √22, in metres | "did the text do anything at all?" Seed-noise floor (empty vs empty, different seeds) = **0.128**. |
| `speed` | mean root-relative joint speed, m/s | static (model gave up) vs moving |
| `head_h_mean` | mean head height, m | upright ≈ 1.58–1.61; collapse ≪ 1.3 |

Visual check: 8-frame stick-figure strips rendered for every clip at seed 0 and
inspected directly — `research/ood-figs/ood_g1..g5.png`.

---

## Result 1 — abstract phrasing is silently ignored

| clip | text | `d_empty` | `speed` | idle gens |
|---|---|---|---|---|
| x_vibes | "vibes" | **0.051** | 0.142 | 3/3 |
| x_thing | "a guy doing the thing" | **0.128** | 0.191 | 1/3 |
| x_meta | "can you do a cool animation for my game please" | **0.155** | 0.065 | 3/3 |
| x_cooler | "make it cooler" | **0.165** | 0.086 | 1/3 |
| x_swag | "he moves with a lot of swagger, real confident energy, you know?" | 0.169 | 0.548 | 0/3 |
| — | EMPTY baseline (seed-noise floor) | 0.128 | 0.099 | — |

`d_empty` for "vibes" is **0.051 m** — less than half the seed-to-seed noise floor of the
empty prompt itself. The text is doing *nothing*. Visually confirmed in `ood_g4.png`:
`x_vibes` and `EMPTY_base` are the same motion frame for frame (arms lift briefly, then
freeze standing). `x_meta` is 6 seconds of a motionless standing figure — `speed` 0.065 m/s,
`limb_cv_max` 0.033, the most "perfect" and most useless clip in the study.

**Rewriting fixes this cleanly:**

| group | idle gens | mean `speed` | mean `d_empty` |
|---|---|---|---|
| abstract (as typed) | **8/15 (53%)** | 0.206 | 0.134 |
| abstract rewritten | **0/15 (0%)** | **0.410** | 0.180 |

e.g. "vibes" → "a person sways from side to side and bobs their head": `speed`
0.142 → 0.308, and `ood_g5.png` shows a genuine weight-shifting sway instead of a frozen
stance.

Note the failure is **silent** — the output is a perfectly clean, anatomically valid,
completely unrelated idle. No error, no jitter. A user typing naturally gets a
convincing-looking nothing.

## Result 2 — concrete natural phrasing: modest, intent-dependent degradation

Mean `limb_cv_max` over 3 seeds, natural vs matched rewrite (**bold = ≥0.30, non-physical**):

| intent | natural prompt | nat | hml rewrite | delta |
|---|---|---|---|---|
| exhausted | "show me someone who is exhausted after a long run" | **0.529** | 0.153 | **-0.376** |
| sneak | "someone sneaking around trying not to get caught" | **0.510** | **0.427** | -0.083 |
| celebrate | "give me a big celebration, they just won" | **0.365** | **0.622** | +0.257 |
| trip | "the guy trips over something" | **0.327** | 0.123 | **-0.204** |
| sofa | "chilling on the sofa" | 0.280 | **0.408** | +0.128 |
| zombie | "a zombie shuffle" | 0.276 | 0.075 | -0.201 |
| baddance | "dancing badly at a wedding" | 0.227 | 0.267 | +0.040 |
| stormoff | "he is angry and storms off" | 0.181 | 0.121 | -0.060 |
| lost | "he is totally lost, no idea where he is" | 0.088 | 0.050 | -0.038 |
| nervous | "make him look nervous" | 0.046 | 0.070 | +0.024 |
| **mean** | | **0.283** | **0.232** | **-18%** |

Per-generation "physically valid" rate (`limb_cv_max` < 0.30 **and** `head_h_mean` > 1.30):
**natural 18/30 (60%)**, **rewritten 21/30 (70%)**.

The averages hide the real structure. Three distinct outcomes:

**(a) Rewriting rescues a hard failure** — `exhausted`, `trip`, `zombie`.
See `ood_g2.png`: `exhausted_nat` and `trip_nat` **collapse into a crumpled non-physical
heap on the floor** (head height falls to 0.94 m and 0.67 m mean; bone lengths swing 33–53%),
while `exhausted_hml` and `trip_hml` are upright, coherent walking/stumbling
(head 1.52/1.55 m, `limb_cv_max` 0.15/0.12). That is 5 of 6 natural generations destroyed
and 6 of 6 rewrites clean. `d_empty` for `trip_nat` is 0.654 — far from the prior, but far
in the *wrong* direction.

**(b) Rewriting makes no difference — the action itself is broken** — `sneak`, `sofa`.
Both phrasings collapse (`ood_g1.png`: `sofa_nat`, `sofa_hml` and `sneak_hml` all fold into
the same tangle of limbs on the ground). Sitting and crouching are simply not reliable in
this checkpoint.

**(c) Rewriting makes it worse** — `celebrate` (0.365 → 0.622). The rewrite
"a person jumps up and raises both arms above their head" is more dynamic and more correct
semantically (hands-above-head 72% of frames vs 35%, `ood_g1.png` shows real jumping vs a
frozen arms-up pose) but the jump is what breaks the skeleton. **More faithful ≠ more stable.**

## Result 3 — the ceiling is low, and phrasing is not what limits it

| group | gens | mean `limb_cv_max` | mean `speed` | mean `d_empty` |
|---|---|---|---|---|
| empty (control) | 3 | 0.064 | 0.099 | — |
| abstract | 15 | 0.139 | 0.206 | 0.134 |
| hml_rewrite | 30 | 0.232 | 0.285 | 0.253 |
| natural | 30 | 0.283 | 0.284 | 0.320 |
| abstract_rewritten | 15 | 0.296 | 0.410 | 0.180 |
| **canonical_hml** | 15 | **0.385** | 0.332 | 0.186 |

Canonical HumanML3D captions — the exact in-distribution phrasing — score **worst** on
skeletal stability:

| canonical caption | `limb_cv_max` | outcome (seed 0, `ood_g5.png`) |
|---|---|---|
| "a person jumps forward with both feet" | **1.288** | jump not executed; limbs blow up transiently |
| "the man kicks with his left leg" | 0.287 | **clean, on-intent leg kick** |
| "a person throws a punch with the right arm" | 0.187 | coherent |
| "a person walks in a circle" | 0.083 | coherent |
| "a person walks forward and waves" | 0.081 | coherent, on-intent |

The ordering is by **action**, not by phrasing style: locomotion / upper-body gestures are
solid; ballistic (jump) and ground-contact (sit, crouch, collapse) actions are unreliable
no matter how the caption is worded. Note `limb_cv_max` >1.0 here is a 1–2 frame transient
blowup, not a sustained collapse — visually the figure momentarily stretches then recovers.

## Other observed failure modes on natural phrasing

- **Chaotic flail.** `zombie_nat` ("a zombie shuffle") spins the character through
  **390° of accumulated yaw** with arms flailing (`speed` 0.543 m/s, ~4× the empty
  baseline) — `ood_g2.png`. The rewrite walks normally.
- **Frozen semantics.** `nervous_nat` ("make him look nervous") is `d_empty` 0.150 with
  `speed` 0.092 — below the empty baseline's own 0.099. It is the unconditional idle with
  a different label (`ood_g4.png`).
- **Static / "gave up".** 12/30 natural generations had second-half joint speed
  < 0.12 m/s, i.e. the motion stops moving partway through and holds a pose.

---

## Caveats — read before quoting these numbers

- **Root translation is pinned near the dataset mean for every prompt.** Net root XZ
  displacement over 6 s is 0.97–1.20 m for *all 36 prompts including the empty one*
  (empty = 0.991 m). Root height is near-frozen (ptp 0.0005–0.03 m). Root-motion metrics
  therefore carry almost no signal in this pipeline and were **excluded** from all
  conclusions above; every claim rests on root-relative articulation and bone lengths.
  Whether that is a checkpoint property or a denormalisation issue in
  `motion_server.generate` is **UNKNOWN** and worth its own probe.
- n = 3 seeds per prompt, 10 pairs. The 60% → 70% validity difference is **not**
  statistically separated from noise at this sample size. The abstract-prompt result
  (53% → 0% idle) is far outside noise and is safe to quote.
- There is **no automated text↔motion alignment score** here. Semantic fidelity claims come
  from direct visual inspection of stick-figure strips at seed 0 only
  (`research/ood-figs/`), plus targeted geometric proxies (hands-above-head fraction,
  accumulated yaw, head height).
- `limb_cv_max` conflates a 1-frame transient blowup with a sustained collapse. Head height
  separates them: sustained collapse always drops mean head height below ~1.1 m.

## Reproduce

Pod-side artefacts (all still present):
- `/workspace/probe_ood.py` — 26 prompts × 3 seeds → `/workspace/ood_npy/*.npy`, `/workspace/ood_metrics.json`
- `/workspace/probe2b.py` — 10 rewrite/control prompts × 3 seeds
- `/workspace/analyze3.py`, `/workspace/analyze5.py` — metric tables
- `/workspace/render.py`, `/workspace/render2.py` — stick-figure strips (needs `pip install matplotlib`)
