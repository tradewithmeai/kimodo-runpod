# Sensitivity: Paraphrase Divergence

**Question:** does prompt PHRASING (as opposed to prompt MEANING) change MDM's output enough
to justify building a Claude-powered prompt-rewriting layer?

**Verdict: NO. A rewriting layer aimed at phrasing is theatre.**
Paraphrasing a prompt moves the output *less* than simply changing the random seed does.

---

## Method

- Model: `humanml_enc_512_50steps/model000750000.pt` via `motion_server.MotionModel`
- `seconds=6.0`, `guidance=2.5`, **fixed `seed=1234`** for the whole paraphrase matrix
- Output: `(120, 22, 3)` joint positions per prompt
- Metric: mean per-joint position error (MPJPE) = `mean(||a-b||)` over all frames x 22 joints,
  reported both raw (world space) and root-relative (joint 0 subtracted, isolates pose from translation)
- 5 intent groups x 4 paraphrases = 20 prompts
  - WITHIN  = 6 pairs per group x 5 groups = **30 pairs**
  - BETWEEN = 16 pairs per group-pair x 10 group-pairs = **160 pairs**
- Noise floor control: same prompt, 4 different seeds (1234/7/99/2025), per intent = 30 pairs

Determinism sanity check: same prompt + same seed twice -> MPJPE **0.000000**. Fully deterministic,
so every number below is signal, not re-sampling jitter.

Probes: `/workspace/probe_paraphrase.py`, `/workspace/probe_para2.py` (on the pod)

### Prompt groups

| group | paraphrases |
|---|---|
| walk | a person walks forward / someone walks ahead / the figure strides forward / a man moves forward on foot |
| sit | a person sits down on a chair / someone lowers themselves into a seat / the figure takes a seat / a man sits down |
| jump | a person jumps up / someone leaps into the air / the figure hops upward / a man performs a jump |
| wave | a person waves their hand / someone waves hello / the figure raises an arm and waves / a man waves his hand in greeting |
| kick | a person kicks with their right leg / someone delivers a kick / the figure swings a leg out in a kick / a man kicks forward |

---

## Headline numbers

| measure | n pairs | raw MPJPE | root-relative MPJPE |
|---|---|---|---|
| **WITHIN-group** (paraphrase, same seed) | 30 | **0.1305** (sd 0.0935) | **0.1296** (sd 0.0877) |
| **BETWEEN-group** (different intent, same seed) | 160 | **0.2696** (sd 0.0913) | **0.2593** (sd 0.0934) |
| **SEED NOISE FLOOR** (same prompt, different seed) | 30 | **0.1628** | 0.1590 |

- within / between ratio: **0.484 raw, 0.500 root-relative**
- within / seed-noise ratio: **0.80** — paraphrasing does *less* than reseeding

## Per-group within-divergence vs that group's own seed-noise floor

| group | within-group paraphrase | seed-noise floor (same prompt) | paraphrase / noise |
|---|---|---|---|
| walk | 0.0676 (min 0.0102, max 0.0964) | 0.1036 | 0.65 |
| sit  | 0.1464 (min 0.1079, max 0.1909) | 0.2735 | 0.54 |
| jump | 0.2335 (min 0.0861, max 0.3629) | 0.2018 | **1.16** |
| wave | 0.0425 (min 0.0204, max 0.0593) | 0.0704 | 0.60 |
| kick | 0.1627 (min 0.0376, max 0.2492) | 0.1645 | 0.99 |

In 4 of 5 intents, rewording the prompt perturbs the motion *less* than changing the seed.
Only `jump` shows paraphrase divergence above its noise floor, and only by 16%.

## Between-group matrix (raw | root-relative)

|  | sit | jump | wave | kick |
|---|---|---|---|---|
| **walk** | 0.3310 \| 0.3423 | 0.2094 \| 0.1953 | 0.2080 \| 0.1744 | 0.1940 \| 0.1762 |
| **sit**  | — | 0.3741 \| 0.3585 | 0.3612 \| 0.3540 | 0.3508 \| 0.3451 |
| **jump** | — | — | 0.2289 \| 0.2187 | 0.2366 \| 0.2279 |
| **wave** | — | — | — | 0.2018 \| 0.2007 |

`sit` is the outlier intent — it sits ~0.35 from everything else (it is the only prompt family
that ends the clip in a fundamentally different body configuration). The standing intents
(walk/jump/wave/kick) are only ~0.19–0.24 apart from each other, i.e. semantic separation between
*standing* actions is itself modest.

## Distribution overlap

- max WITHIN pair: 0.3629 raw (jump group)
- min BETWEEN pair: 0.1269 raw
- The two distributions overlap substantially. Mean separation is real (2.07x) but a single pair's
  MPJPE does not tell you whether two prompts meant the same thing.

---

## Interpretation

1. **Phrasing is a second-order effect.** Mean paraphrase divergence (0.1305) is 48% of mean
   different-intent divergence, which sounds meaningful in isolation — but it is *below* the
   0.1628 divergence you get for free by changing the seed on an unchanged prompt. The model is
   robust to wording; CLIP maps these paraphrases to near-equivalent conditioning.

2. **The variance you can actually control is the seed, not the words.** If a user dislikes an
   output, re-rolling the seed changes it more than any rewrite Claude could produce, and does so
   without risking semantic drift.

3. **Where a rewriting layer might still earn its place (not measured here):**
   - prompts that are *out of distribution* for HumanML3D vocabulary (this test used in-distribution
     phrasings — all four variants of each group are things the training set would recognise)
   - prompts that are under-specified or contain multiple actions
   - injecting duration/direction/limb specifics that change *meaning*, not phrasing
   These are semantic edits, not paraphrase. This experiment shows no value in a layer that merely
   restates the same intent more fluently.

4. **`jump` is the one soft spot.** Its within-group spread (0.0861 to 0.3629) is wider than its own
   noise floor and overlaps walk|jump between-group distance (0.2094). "the figure hops upward" vs
   "someone leaps into the air" plausibly land on genuinely different motions — hop vs leap is
   arguably a meaning difference, not a phrasing one. If a rewriting layer is built, the payoff is in
   normalising *verb intensity* (hop/leap/jump), not sentence structure.

## Recommendation

Do not build a prompt-rewriting layer for phrasing normalisation. Spend the effort on
(a) exposing seed re-roll to the user as the primary "give me another take" control, and
(b) if any text preprocessing is built at all, scope it to canonicalising action verbs and
appending missing specifics — a small deterministic mapping, not an LLM call.

## Caveats / UNKNOWN

- Single seed for the paraphrase matrix. A different seed would shift individual pair numbers;
  the noise-floor control bounds how much (~0.16 raw), which is why the within-vs-noise comparison
  is the robust conclusion and individual pair values are not.
- MPJPE is a geometric metric; it does not measure *perceptual* motion quality. Two clips at
  MPJPE 0.10 could still look different to a human. Not measured.
- All prompts are in-distribution HumanML3D-style English. Behaviour on unusual or non-native
  phrasing: UNKNOWN.
- guidance=2.5 only. Whether higher guidance amplifies phrasing sensitivity: UNKNOWN.
