# Adversarial verification — methodology lens

Verdict: **REFUTED** (the inferential machinery, not the raw arithmetic).

The individual measurements mostly replicate. What does not survive is the **decision rule**
that three of the findings are built on: *"divergence below the seed-noise floor ⇒ the factor
doesn't matter."* I show below, by direct measurement, that this rule produces false negatives on
motions that are unambiguously different. Two further findings used control samples of n=1.

All numbers below are from probes I ran myself on the pod
(`/workspace/probe_meth_A.py` … `probe_meth_E.py`, results in `/workspace/probe_meth_{A,B,C}.json`,
`probe_meth_D.json`). Checkpoint `humanml_enc_512_50steps`, 6 s / 120 frames, guidance 2.5,
8 seeds `[1234, 7, 99, 2025, 31337, 5, 860, 2]` unless stated.

---

## 0. What I confirm first

Determinism holds: same prompt + same seed reproduces bit-identically, so every divergence number
in the corpus is signal, not sampler jitter. I also reproduced two of the earlier probes' exact
figures to the 4th decimal (their single seed-pair 0.1342; their two noise directions 0.0548 and
0.2611). **The earlier agents' arithmetic is honest.** The problem is what was inferred from it.

---

## 1. The seed-noise floor WAS measured — but it was measured badly, and pooled wrongly

The floor exists in the corpus (`probe_para2.py`), but at n=6 pairs per intent from 4 seeds, and it
was pooled to a single number (0.1628) that is then used as a universal yardstick.

My re-measurement, 8 seeds → **112 pairs per intent** (vs their 6):

| intent | seed floor (raw MPJPE) | sd | their 4-seed value |
|---|---|---|---|
| walk | 0.1340 | 0.0378 | 0.1036 |
| sit  | 0.2502 | 0.1347 | 0.2735 |
| jump | 0.2305 | 0.0729 | 0.2018 |
| wave | 0.0784 | 0.0321 | 0.0704 |
| kick | 0.1864 | 0.0459 | 0.1645 |

The floor spans **3.2x** between intents (wave 0.078 → sit 0.250). A single pooled floor of 0.1628
describes no intent in the set. Any claim of the form "X = 0.13, floor = 0.16, therefore X is
below the floor" is comparing against a number that does not apply.

## 2. The floor is not a valid decision rule — demonstrated

I generated prompt pairs whose motions are **known** to differ, verified each with an independent
root-relative descriptor over the same 8 seeds (paired t across seeds), then asked whether MPJPE
separates them from the floor.

| pair | MPJPE | floor | ratio | descriptor A vs B | paired t |
|---|---|---|---|---|---|
| walk forward vs **in a circle** | 0.1102 | 0.1126 | **0.98 — BELOW FLOOR** | artic 0.322 vs 0.376 | −2.34 |
| kick right vs left leg | 0.1816 | 0.1575 | 1.15 | foot asym −1.441 vs +0.826 | −9.55 |
| walk slowly vs quickly | 0.1422 | 0.1010 | 1.41 | artic 0.224 vs 0.365 | −3.69 |
| walk forward vs backwards | 0.1845 | 0.1168 | 1.58 | artic 0.322 vs 0.348 | −0.69 |
| **wave right vs left hand** | 0.1783 | 0.0766 | 2.33 | L−R wrist −0.689 vs **+0.935** | **−28.43** |
| raise right vs left arm | 0.1197 | 0.0641 | 1.87 | L−R wrist −0.343 vs +0.719 | −11.13 |

Two things kill the rule:

- **"Walks forward" vs "walks in a circle" lands BELOW its own seed floor** (0.1102 < 0.1126) while
  being a different motion. A false negative on a case nobody would dispute.
- **A total left/right inversion** — descriptor sign-flipped, paired t = −28.4, as unambiguous a
  semantic change as this model can express — scores only **0.178**, i.e. inside the range the
  paraphrase probe reports for its *within-group* "no difference" condition (sit 0.2097,
  jump 0.1816).

So MPJPE near or below the floor does **not** license "semantically equivalent". The metric lacks
the resolution for the question it is being asked.

## 3. `paraphrase-divergence`: numbers replicate, per-group numbers do not, conclusion does not follow

**Replicates and is in fact stronger.** With 8 seeds instead of 1 for the within-group term
(48 pairs/intent), paraphrase divergence is below the floor in **5/5** intents, significant in 4:

| intent | within | floor | ratio | d = within − floor, 95% bootstrap CI |
|---|---|---|---|---|
| walk | 0.0686 | 0.1340 | 0.51 | −0.0653 [−0.0795, −0.0507] SIG |
| sit  | 0.2097 | 0.2502 | 0.84 | −0.0405 [−0.0834, +0.0044] ns |
| jump | 0.1816 | 0.2305 | 0.79 | −0.0489 [−0.0768, −0.0193] SIG |
| wave | 0.0362 | 0.0784 | 0.46 | −0.0421 [−0.0511, −0.0325] SIG |
| kick | 0.1200 | 0.1864 | 0.64 | −0.0664 [−0.0819, −0.0510] SIG |

**But their per-group breakdown is not reproducible.** They reported jump 1.16x and kick 0.99x
(the basis for "in 4 of 5 intents rewording moves the motion less than reseeding"). I get 0.79x and
0.64x. Cause: an asymmetric design — the within-group term used **one seed (1234)**, the floor term
used four. Per-seed, within-group divergence swings enormously (walk 0.0370–0.1432; sit
0.0915–0.3450 across the 8 seeds). Their per-intent numbers are single draws from that spread.

**I also extended the paraphrase set, and the aggregate result survives.** Their 4 paraphrases per
intent were all canonical third-person HumanML3D captions ("a person X" / "someone X" / "the figure
X" / "a man X") — the narrowest possible slice of "phrasing", and precisely the style a rewriting
layer *outputs*. I re-ran with 6 styles per intent (canonical, imperative, second-person, terse,
verbose, colloquial), 8 seeds, 120 pairs/intent:

| intent | style divergence | floor | ratio | CI |
|---|---|---|---|---|
| walk | 0.1008 | 0.1143 | 0.88 | [−0.0259, −0.0020] SIG |
| sit  | 0.1874 | 0.2504 | 0.75 | [−0.0916, −0.0324] SIG |
| jump | 0.1868 | 0.2284 | 0.82 | [−0.0562, −0.0275] SIG |
| wave | 0.0787 | 0.0946 | 0.83 | [−0.0266, −0.0045] SIG |
| kick | 0.1609 | 0.1968 | 0.82 | [−0.0556, −0.0157] SIG |

Credit where due: this is a stronger version of their claim than they ran.

**The conclusion "a prompt-rewriting layer for phrasing is theatre" still does not follow**, for
three measured reasons:

1. Per §2, the rule is invalid — a left/right inversion also sits in this band.
2. **Individual styles do exceed the floor**: walk/verbose 1.53x, wave/verbose 1.47x,
   kick/colloquial 1.21x, sit/colloquial 1.07x, jump/imperative 1.04x. The *average* being below
   the floor hides the specific phrasings a rewriter would target.
3. **MPJPE misses the failure phrasing actually causes.** Clearest case I measured:
   `"make the guy boot something with his right foot"` → articulation **0.178 m/s** vs canonical
   `"a person kicks with their right leg"` **0.399 m/s** — a 2.2x collapse, landing near the
   empty-prompt baseline of 0.085. That is the model failing to kick. Its MPJPE to canonical is
   0.2377, a mere 1.21x the floor. The metric registers a shrug; the motion is gone.

This is corroborated *inside the corpus*: the vocabulary probe measured wave articulation 0.131 for
"the man waves his right hand" vs 0.042 for the imperative (3.1x) — while the paraphrase probe
reports the wave group's paraphrase divergence as 0.0425, its smallest number. Two probes measured
the same axis and drew opposite conclusions because they used different metrics. The ood probe
likewise found rewriting rescues `exhausted` (0.529→0.153) and `trip` (0.327→0.123) from skeletal
collapse. "Theatre" is contradicted by the corpus's own measurements.

## 4. `embedding-perturbation`: the yardstick was n=1 and biased high; the bound is a median

Their seed-change yardstick, 0.1342, is **a single pair** (seed 1234 vs 1235). I reproduce it
exactly — and the properly sampled floor for that prompt over 28 pairs is **0.1066 ± 0.0339**
(range 0.0515–0.1658). Their single draw sat 26% above the mean, inflating the bar their noise
levels had to clear.

Their direction sample was n=2, and it is the load-bearing generalisation. Over **20 random
directions** at each magnitude (same prompt, seed 1234):

| rel noise | median MPJPE | mean | min | max | fraction below the true floor 0.1066 | max skeleton-length deviation |
|---|---|---|---|---|---|---|
| 5%  | 0.0208 | 0.0319 | 0.0041 | 0.1005 | **100%** | 1.4% |
| 10% | 0.0482 | 0.0668 | 0.0096 | 0.2668 | **75%** | **14.0%** |
| 25% | 0.1246 | 0.1202 | 0.0185 | 0.2971 | 45% | 10.1% |

"Noise up to ~10% of the embedding norm changes the motion less than a seed change does" is a
**median** statement dressed as a bound: it fails for 1 in 4 directions, worst case 2.5x the floor.
Their reported "total skeleton length moved ≤4.3%" at 10% is direction-1-specific; over 20
directions the worst is **14.0%**, so "coherence held everywhere" is not established either. Their
n=2 sample happened to draw one direction from each side of the floor, which is what produced the
"anisotropic" caveat — the caveat is correct but it is the *headline*, not a footnote. The 5%
result is solid: 100% below floor over 20 directions.

## 5. `ood-phrasing`: direction robust, the "0%" is not, and the criterion is partly inside the noise

The 53%→0% headline uses `idle = (d_empty < 0.17) AND (speed < 0.20)` — two unjustified cut points.
I swept both over the existing `/workspace/ood_npy/` outputs (no regeneration needed):

- **Direction is robust**: abstract > abstract_rewritten at all 24 threshold combinations tested.
  Good.
- **The "0%" is not**: rewritten reaches 13–20% at (0.10–0.13, 0.25–0.30) and 20–40% at dcut ≥ 0.25.
  Abstract itself ranges 20%–80% across the sweep. The specific pair 53%/0% is the most flattering
  cell in the grid.
- **The distance criterion alone is inside the noise**: `d_empty < 0.17` flags **47% of the
  canonical HumanML3D prompts** as idle — prompts that demonstrably work. On that same metric the
  canonical prompts' own seed floor is 0.194 (range 0.082–0.354), i.e. *above* the 0.17 cut. It is
  the conjunction with `speed < 0.20` that saves the classifier (canonical false-positive rate
  13%, 2/15). So the honest statement is 53% vs a 13% false-positive floor, not 53% vs 0.
- **Effective n is 5, not 15**: 15 clips = 5 prompts × 3 seeds, and same-prompt clips are strongly
  correlated. Their own caveat that "the 60%→70% difference is NOT separated from noise" is right;
  the idle result is larger but rests on 5 prompts.

## 6. What survives unharmed

Nothing in this lens touches `conditioning-path`, `precompute-cache`, or `footprint-floor` — those
are architectural/VRAM measurements with deterministic, bit-exact verification and no statistical
inference. The `embedding-substitution` population test (n=36 vs 36, reporting median [min,max] and
explicitly checking overlap) is the best-designed experiment in the corpus and I found no fault
with its controls.

---

## Corrections to apply

1. Stop using a **pooled** seed-noise floor. It varies 3.2x by intent (0.078–0.250).
2. Stop using **"below the seed floor" as evidence of semantic equivalence.** Measured
   counterexample: walk-forward vs walk-in-a-circle, 0.1102 vs floor 0.1126. Pair every MPJPE claim
   with a task-relevant descriptor (articulation, left/right asymmetry, foot alternation).
3. `paraphrase-divergence`: the aggregate result is real and replicates more strongly than reported
   (5/5 intents, and it holds across imperative/terse/verbose/colloquial styles too). Its per-intent
   breakdown is single-seed noise and should be withdrawn. The conclusion "a rewriting layer is
   theatre" should be withdrawn — it is refuted by the corpus's own vocabulary and ood measurements
   and by the kick/colloquial case above (articulation 0.399→0.178 at only 1.21x floor MPJPE).
4. `embedding-perturbation`: restate as "the **median** random direction at 10% noise stays under
   the seed floor (75% of 20 directions); the worst reaches 2.5x floor with 14% skeleton distortion."
   The 5% figure is safe.
5. `ood-phrasing`: report 53% vs the 13% canonical false-positive rate, note effective n = 5 prompts,
   and state that the direction (not the magnitude) is what is threshold-robust.

Probes: `/workspace/probe_meth_A.py` (floor + paraphrase × 8 seeds, 160 gens),
`probe_meth_B.py` (6 phrasing styles × 5 intents × 8 seeds, 248 gens),
`probe_meth_C.py` (20 random embedding directions × 3 magnitudes),
`probe_meth_D.py` (decision-rule validity, 6 known-different pairs × 8 seeds),
`probe_meth_E.py` (threshold sweep, no regeneration).

---

# Second independent verification pass

A second verifier re-ran the core checks from scratch with independently chosen prompts,
paraphrase sets, seeds, and perturbation directions (`/workspace/probe_verify_method.py`,
`probe_verify_artic.py`, `probe_verify_rule.py`; raw JSON `/workspace/verify_method_out.json`,
`/workspace/verify_artic_out.json`). Nothing below was copied from the first pass.

## Corroborations of the REFUTED verdict (independent data)

1. **Decision-rule false negative reproduces.** "a person walks forward" vs "a person walks in a
   circle", 4 seeds: cross-prompt same-seed MPJPE mean **0.1038** vs the two prompts' own seed
   floors **0.1036 / 0.1154** — at/below floor for an undisputed intent change (first pass got
   0.1102 vs 0.1126). And "waves right hand" vs "waves left hand": wrist L−R descriptor sign-flips
   on all 4 seeds (−0.49..−0.80 vs +0.90..+1.02) yet MPJPE is only **0.192** — inside the band the
   paraphrase probe calls within-group noise. The floor is not a semantic-equivalence test.
2. **Floor is prompt-dependent and single pairs are unreliable**: my 6-prompt × 4-seed floors span
   0.076 (wave) to 0.165 (kick), pairs 0.043–0.204 — consistent with the first pass's 3.2x spread.
3. **Perturbation anisotropy is even wider than either probe stated**: my direction at 10% noise
   (cos 0.9950) gives MPJPE **0.0058** vs the original probe's 0.0548 and 0.2611 — a ~45x spread
   across three directions at the same cosine. "≤10% noise < seed change" is a per-direction
   statement, not a bound. Confirms §4.
4. **Independent paraphrase sets swing per-intent results**: my own sit paraphrases give
   within-group 0.280 (max pair 0.454) — at between-intent scale (my between mean 0.317) — vs the
   original's 0.146. Paraphrase-set choice is a large uncontrolled variable at n=4; per-intent
   claims from single paraphrase sets should not be trusted. Aggregate replicates (my within 0.141
   / between 0.317, ratio 0.44 vs their 0.48).
5. **Phrasing measurably matters on the corpus's own articulation metric**: I reproduce wave
   articulation "the man waves his right hand" 0.1310 vs "a person waves hello" 0.0687 — 1.9x from
   phrasing alone, same intent. "A rewriting layer is theatre" is refuted inside the corpus.

## Confirmations (the mechanistic findings stand, now doubly measured)

- **Determinism**: max abs diff 0.0 (same prompt+seed).
- **conditioning-path**: pooled [1,1,512] embedding confirmed by direct call (norm 9.5585 matches
  their 9.558); `emb = text_emb + time_emb` and the `text_embed` hook confirmed in source.
- **embedding-substitution / precompute-cache mechanics**: `y['text_embed']` injection vs text path
  = **0.0 max abs diff** (bit-identical, reproduced). Overwrite gotcha reproduced: with both
  `text_embed` (walk) and `text` (cartwheel) present, output is bit-identical to the cartwheel
  text path — `p_sample_loop` wins, exactly as the caveats state.
- **Random-vector coherence**: fresh random direction (cos −0.006): bone-CV 0.0274 vs walk 0.0267,
  total bone length 4.34 vs 4.36 m, pelvis 0.917 m; rand-vs-walk MPJPE 0.176 > walk floor 0.104
  (the vector steers). Substitution finding confirmed.
- **vocabulary-probe raw data**: articulation numbers reproduce to 4 decimals (empty 0.0703,
  "a person" 0.0447, wave 0.0687, clap 0.0825, stand 0.0219, walk 0.3253, kick 0.3588). The
  dead-gesture-category result is real. One wording caveat: "statistically indistinguishable from
  the empty prompt" holds for articulation energy only — wave vs empty at the SAME seed differs by
  0.129–0.150 root-relative MPJPE, 1.6–2x the empty seed floor (0.053–0.096). Gesture prompts
  produce *different but gestureless* motion; the conditioning is not inert.
- **ood-phrasing "abstract prompts ignored"**: "vibes" vs empty, same seed, root-relative
  0.015–0.074 — at/below the empty cross-seed floor. Direction confirmed (magnitude critique in §5
  stands).

## New: the root-translation bug is attributed (resolves ood-phrasing's UNKNOWN)

`data_loaders/humanml/data/dataset.py:783-796`: `t2m_mean.npy`/`t2m_std.npy` (meta_dir
`{dataset}_mean/std`) are the **T2M evaluator's** normalisation stats ("used by T2M models
(including evaluators) ... this is to translate their norms to ours"). MDM's own models
de-normalise with `Mean.npy`/`Std.npy` from the dataset root — which `motion_server.py`
deliberately avoids downloading, substituting the evaluator files. The evaluator std suppresses
the 4 root channels (first 6 values re-read: 0.000515, 0.000677, 0.000677, 0.00612, 0.1226,
0.1226), so decoded root translation/yaw is ~20-30x too small for every clip. Re-measured: net
root XZ displacement stand 0.991 m / walk 1.190 / run 1.112 / empty 0.993 — text-insensitive.
**This is a motion_server.py de-normalisation bug, not a checkpoint property.** Fix: de-normalise
with the dataset-root `Mean.npy`/`Std.npy` (requires fetching those two files), keeping
`t2m_*.npy` only if feeding the T2M evaluator.

Consequence for methodology: all cross-clip MPJPE comparisons in the corpus remain internally
valid (every clip shares the same decode), but the bug is *why* walk-forward vs walk-in-a-circle
is invisible to MPJPE (yaw frozen — measured path curvature 1.000 vs 1.003), i.e. the §2
decision-rule failure is partly mechanistic and will shrink (not vanish) once the std is fixed.

## Second-pass verdict

Agrees with the first pass. **REFUTED** at the level of the inferential headlines:
paraphrase-divergence's "rewriting layer is theatre" (contradicted by corpus-internal and
re-measured phrasing effects, and built on a metric that cannot see left/right inversions or
turning), embedding-perturbation's "≤10% noise < seed change" (median-per-direction, not a bound),
ood-phrasing's "0%" (threshold-flattered), vocabulary-probe's "indistinguishable from empty"
(metric-specific). **CONFIRMED**: every mechanistic claim tested — pooled-vector conditioning,
bit-exact embedding injection and the text-overwrite gotcha, random-vector coherence and steering,
the dead gesture category, the root-decode bug (now attributed), determinism, and the seed floors
themselves as honest arithmetic.
