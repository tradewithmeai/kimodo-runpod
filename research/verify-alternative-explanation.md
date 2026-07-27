# Adversarial verification — "alternative explanation" lens

Every number below is from a probe I ran myself on the pod (RTX 3090, humanml_enc_512_50steps,
20 fps, guidance 2.5 unless stated). Method: for each claim, state a competing explanation,
then try to kill the claim with it.

---

## 1. [vocabulary-probe] normalisation bug — **CONFIRMED** (mechanism nailed down, magnitude right)

**Claim:** root translation is decoded ~20-30x too small; every clip drifts ~1 m at a fixed
0.19 m/s regardless of text.

**Competing explanations I tested**
- (A) `t2m_std.npy` is a *legitimate* alternative normalisation (text-to-motion's grouped
  per-feature-block std with `feat_bias`), applied consistently at train and decode time,
  so the small root std is correct and the model simply cannot produce translation.
- (B) The near-constant ~1 m drift is a *mean* artifact, not a std artifact, and correcting
  the std would not restore text dependence.

**Test 1 — internal redundancy of the 263-d vector.** The HumanML3D vector encodes root
horizontal velocity **twice**: channels 1:3 (`root_linear_velocity`, std group 0.000677) and
channels 193:196 (`local_vel` of joint 0 = root displacement in the root frame, std group
0.019541). Same physical quantity, different normalisation group — so if the file were
self-consistent, the two decoded signals would agree.

Measured on the model's own raw output (seed 1234, 120 frames):

| prompt | corr(ch2, ch195) | norm-space std ratio ch195/ch2 | norm-space mean ratio |
|---|---|---|---|
| a person walks forward | +1.0000 | 0.871 | 0.867 |
| a person runs forward | +1.0000 | 0.870 | 0.871 |
| a person walks backwards | +1.0000 | 0.869 | 0.872 |
| a person stands still | +0.9753 | 1.033 | 0.881 |
| a person lies down on the floor | +0.9916 | 0.893 | 0.881 |
| (empty) | +0.9931 | 0.865 | 0.874 |

Correlation is 1.0000 to four decimals and the two channels carry the *same normalised
amplitude* (ratio ~0.87). Their true stds therefore differ by ~0.87, not by the 28.87x the
file asserts (`std[193]/std[1] = 0.019541/0.000677 = 28.87`). The means agree already
(`mean[2] = 0.008684`, `mean[195] = 0.008720`). **The file's `std[1:3]` is internally
inconsistent with its own `std[193:196]`.** Explanation (A) is refuted.

**Test 2 — physical calibration by stance-foot slip.** Swept a multiplier `k` on `std[1:3]`,
decoded with `recover_from_ric`, and measured world-space horizontal displacement per frame
of the lower ankle (a physically correct walk has ~0 slip):

| k (std[1:3]) | walks fwd slip / net XZ | stands still slip / net XZ | empty slip / net XZ | walks backwards |
|---|---|---|---|---|
| 1 (as shipped) | 0.0337 m / **1.19 m** | 0.0082 / **0.99 m** | 0.0081 / **0.99 m** | 0.0232 / 0.92 m |
| 8 | 0.0239 / 2.28 | 0.0057 / 0.69 | 0.0057 / 0.71 | 0.0165 / 0.10 |
| 16 | 0.0128 / 3.53 | 0.0029 / 0.35 | 0.0030 / 0.38 | 0.0089 / 0.87 |
| 20 | 0.0077 / 4.16 | 0.0015 / 0.18 | 0.0017 / 0.22 | 0.0053 / 1.34 |
| **24** | **0.0045** / 4.78 | **0.0004** / **0.01** | **0.0006** / **0.05** | **0.0028** / 1.81 |
| 28.9 | 0.0065 / 5.55 | 0.0018 / 0.20 | 0.0016 / 0.15 | 0.0050 / 2.39 |
| 35 | 0.0143 / 6.50 | 0.0040 / 0.46 | 0.0036 / 0.40 | 0.0105 / 3.12 |

Foot slip is minimised at **k = 24** for every prompt tested (walk, run, stand, backwards,
lie down, empty), i.e. true `std[1:3] ~ 0.0163`, **24x the shipped value** — squarely inside
the earlier agent's 20-30x estimate. At k=24 the spurious drift disappears (stands still
0.99 m -> 0.01 m; empty 0.99 -> 0.05 m) and text dependence appears (walks forward 4.78 m =
0.80 m/s; walks backwards 1.81 m; runs forward 2.94 m). Explanation (B) is refuted: the mean
does dominate at k=1, but that is *because* the std is 24x too small, and fixing the std
restores text dependence rather than leaving it flat.

**Verdict: CONFIRMED.** Bug is real, magnitude ~24x (their "20-30x" is right), and the
downstream caveats in `ood-phrasing` and `embedding-substitution` about root metrics being
unusable are justified. Side effect: `runs forward` at 0.49 m/s still lags `walks forward` at
0.80 m/s even after correction, independently confirming vocabulary-probe finding (5).

Independent cross-check (second verifier, probes /workspace/probe_verify_alt2.py): raw
normalised ch2 is text-dependent and healthy (walk +1.94, run +0.96, stand -0.53, empty
-0.51) yet denormalises to 0.0083-0.0100 m/frame for ALL of them; root XZ displacement over
6 s: walk 1.19 m, stand 0.99 m, empty 0.99 m. Same observables, same conclusion.

---

## 2. [conditioning-path] — **CONFIRMED**

Competing explanation: none plausible (structural claims); re-measured anyway.
- Forward hook on `clip_model.visual` during a full generate: **0 calls**, motion returned fine.
- `encode_text("a person walks forward and then sits down")` -> shape (1,1,512) float32,
  norm 9.7251. Determinism: same prompt+seed twice -> MPJPE 0.0 exactly.
- Code read on pod matches all cited lines (mdm.py `clip_encode_text` 22-token context
  zero-padded to 77; `text_embed` cache branch; `emb = text_emb + time_emb`;
  gaussian_diffusion.py:633-635 overwrite when `y['text']` present).

## 3. [embedding-perturbation] — **CONFIRMED**

Competing explanation tested: "10% noise < seed change" could be an artefact of the single
noise direction and single seed-pair yardstick used.
- Seed-change yardstick over 5 seed pairs (1234 vs 7/99/2025/555/31337), injection path:
  MPJPE 0.054/0.121/0.120/0.141/0.126 -> median **0.121** (their 0.134 is typical, not cherry-picked).
- 10% noise in **10 fresh random directions**, same seed: median **0.039**, mean 0.064,
  values 0.010-0.267. 9/10 directions below the seed-change median; 1/10 (0.267) above it.

Headline holds in the typical case; the strong direction-dependence the finding itself
reports is real and is the necessary caveat. Not explained by sampling randomness.

## 4. [embedding-substitution] — **CONFIRMED with one correction (tail understated)**

Confirmed mechanics:
- Injection bit-identity: `y['text_embed']` vs `y['text']` -> max_abs_diff **0.0**.
- Steering is real: two different norm-matched random vectors, same seed -> MPJPE 0.79.
- The MEDIAN random direction is coherent: 16 fresh norm-matched random vectors (cos ~ 0),
  bone-length cv_max median **0.220**; calibration same metric: empty 0.035, walk 0.141,
  sit 0.317, jump 0.643. Root height normal (0.90-0.93 m) in every run.

CORRECTION: "statistically indistinguishable / bone CV max 0.138 over 36 runs" does not
survive a larger sample. Over 16 fresh directions at seed 1234: 4/16 exceed the 0.30
physical-validity threshold, 1/16 hits cv_max **2.13**; a 17th vector (gen seed 42) hit
**2.38**. Alternatives tested:
- Sampling randomness? Partly, but no: dir42 is broken at 2 of 3 seeds (2.38/2.46/0.12),
  dir10 at 1 of 3 (2.13/0.05/0.63), dir6 elevated at 3 of 3 (0.62/0.54/0.33) — a
  direction-by-seed interaction, with worst cases beyond the worst real prompt measured
  anywhere in these findings (canonical "jumps" 1.29). Their n=12 directions missed a
  roughly 10-25% bad tail.
- Guidance blend? No: the broken vector stays broken at g=5.0 (cv_max 1.61) and g=7.5 (1.27),
  so coherence of random vectors at g=2.5 is not an artefact of blending toward the
  unconditional mode.

Net: "CLIP supplies none of the motion prior" is right for the median off-cone vector but
overstated for the tail. Irrelevant for a replacement encoder living near the text cone, but
the phrasing should be weakened.

## 5. [precompute-cache] — **CONFIRMED**

Re-measured in one process: peak during normal generate **456.3 MB** (their 456.3); after
`del clip_model` + empty_cache, generate with injected embedding: peak **95.0 MB** (their
95.0), resident 91.6 MB, output **bit-identical** (0.0). Checkpoint: 108 keys, **0**
`clip_model.*` keys, **81,818,987 bytes** — exact match.

## 6. [footprint-floor] — **CONFIRMED (spot-checked)**

The fp32 baseline (435-456 MB depending on process history) and the ~91-95 MB CLIP-free
floor reproduce; two prior agents and this verification now agree independently. fp16/sm_75
sub-claims not re-run — they are code-scan and kernel-dispatch facts with no
sampling-randomness/guidance/length alternative to test.

## 7. [paraphrase-divergence] — **CONFIRMED in aggregate; the 0.80 ratio is not a constant**

Competing explanation tested: within-group < seed-noise could be specific to seed 1234 and
their paraphrase sets. Re-run with MY OWN paraphrase sets at seed 7:
- walk: within 0.073 vs seed-noise 0.102 (0.72x)
- wave: within 0.010 vs 0.067 (0.15x)
- sit: within 0.236 vs 0.148 (**1.60x**)
- overall means: within 0.107 vs noise 0.105 (~1.0x)

The exact 0.80 is seed- and paraphrase-set-dependent and an intent can flip above 1 with
more OOD wordings. But the operative claim survives: within-intent rewording is the same
order as reseeding and far below between-intent separation. "A rewriting layer for phrasing
is theatre" stands for in-distribution paraphrases, and does not conflict with ood-phrasing,
whose rewrite win is on abstract prompts — a different population.

## 8. [ood-phrasing] — **CONFIRMED (the strong sub-claim)**

Re-measured across seeds 1234/7/99: "vibes" d_empty = 0.074/0.052/0.015 and "can you do a
cool animation for my game please" = 0.083/0.085/0.069, all at/below the same-seed
seed-noise floor (0.085/0.098/0.091), with idle speeds 0.038-0.087 m/s ~ empty (0.062-0.083).
Control "a person walks forward": d_empty 0.083-0.125 with speed 0.30-0.36 m/s (4-5x idle).
Abstract prompts really are silently ignored — not sampling randomness. (The finding itself
already flags its 60%->70% concrete-prompt result as not separated from noise; agreed.)

## 9. [vocabulary-probe] dead gesture zone — **CONFIRMED (robust to length AND guidance)**

Competing explanations tested: clip length (maybe gestures need longer horizons) and
guidance scale. "the man waves his right hand" vs empty vs "the man kicks with his right
leg", 3 seeds each (mean root-relative articulation, m/s):

| condition | wave | empty | kick |
|---|---|---|---|
| 6.0 s, g=2.5 | 0.131 | 0.070 | 0.458 |
| 9.8 s, g=2.5 | 0.096 | 0.017 | 0.484 |
| 6.0 s, g=7.5 | 0.152 | 0.041 | 0.483 |

Wave never leaves the idle band while kick is 3-5x higher under every condition. The dead
category is a model property, not a sampler, seed, guidance, or length artefact.

---

# Overall verdict

**No finding is refuted.** Every headline claim survived direct tests of the
sampling-randomness, guidance-scale, and clip-length alternatives. Two quantitative
statements need weakening:
1. embedding-substitution's "statistically indistinguishable coherence" — the random-direction
   tail is heavier than the n=12 sample showed (~10-25% of directions exceed the 0.30
   validity threshold; catastrophic direction-by-seed cases reach cv_max 2.1-2.5).
2. paraphrase-divergence's 0.80 within/noise ratio — fluctuates ~0.15-1.60 by intent and
   paraphrase set; the aggregate conclusion (within ~ seed noise << between-intent) stands.

Second-verifier probes on the pod: /workspace/probe_verify_alt1.py .. probe_verify_alt5.py.

