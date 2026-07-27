# Adversarial verification — replication lens

Every number below was re-measured by me on the pod (RTX 3090) with **different seeds, prompts, and lengths** than the original agents used. Probes: `/workspace/verify_rep1.py` … `verify_rep7.py`.

## Verdict summary

| Finding | Verdict |
|---|---|
| conditioning-path | **REPLICATES** |
| embedding-perturbation | **REPLICATES** (direction-dependence milder here: 1.7x, not 5x) |
| embedding-substitution | **REPLICATES** |
| precompute-cache | **REPLICATES** (95.0 MB peak reproduced to the decimal; bit-identical) |
| footprint-floor | **REPLICATES** (spot-checked: baseline peak, no-visual config, MDM-only floor, dtype gotcha) |
| paraphrase-divergence | **PARTIAL — headline claim does not survive a new intent set** |
| ood-phrasing | **REPLICATES** |
| vocabulary-probe | **REPLICATES** (dead gesture category + root-translation bug both reproduce on fresh seeds) |

## 1. conditioning-path — REPLICATES

New prompt ("a person crawls under a table"), new seed (555), new length (4.5 s = 90 frames):

- `encode_text` → `(1,1,512)` float32 cuda for bs=1, `(1,3,512)` for bs=3 — pooled sentence vector, per-prompt not per-token. Matches.
- Full generate: **visual tower forward calls = 0** (hook on `visual` + all 113 submodules), **CLIP text transformer fired exactly 1 time**, **MDM.forward called 100 times** (2×50 CFG), seqTransEncoder input **(91,1,512)** = 90 motion tokens + 1 combined text/time token. All exactly as claimed, and the 91-token shape confirms the prepend mechanism generalises beyond their 120-frame case.
- All CLIP params `requires_grad=False`. Param counts identical: total 169,157,640 / CLIP 151,277,313 / visual 87,849,216 / MDM-only 17,880,327.
- Static check: mdm.py:212-214 (`text_embed` cache), gaussian_diffusion.py:635 (encode-once + silent overwrite of `y['text_embed']` when `y['text']` present) both present as quoted.

## 2. embedding-perturbation — REPLICATES

New base prompt "a person jumps twice and then walks in a circle", new seed 555, new noise directions (202/303). Emb norm 10.37.

| Perturbation | cos | MPJPE (m) |
|---|---|---|
| noise 1% | 1.0000 | 0.0049 |
| noise 5% | 0.9988 | 0.0218 |
| noise 10% | 0.9951 | 0.0363 |
| noise 25% | 0.9704 | 0.0879 |
| noise 100% | 0.7139 | 0.3261 |
| **seed change (555→556)** | — | **0.1941** |
| zero embedding | — | 0.3074 |
| random same-norm (cos 0.018) | — | 0.3716 |

- 10% noise (0.036) ≪ seed change (0.194): the headline "noise up to ~10% changes motion less than a seed change" replicates with margin.
- Random vector: coherent human-scaled skeleton (pelvis 0.90 m, total bone len 4.83 m vs base 4.44). Replicates.
- Semantic-decay signature replicates: at 100% noise bone-CV mean collapses 0.077 → 0.026, converging on the zero-embed value (0.021).
- Direction dependence: second direction at 10% gave 0.061 vs 0.036 — anisotropic, but only 1.7x here vs their 5x. The *existence* of anisotropy replicates; the magnitude is direction-lottery. Minor caveat only.

## 3. embedding-substitution — REPLICATES

- Injecting `y['text_embed'] = encode_text(prompt)` while **omitting** `y['text']` → **max_abs_diff = 0.0** vs the text path, at a new prompt/seed. The injection path and the gaussian_diffusion.py:635 overwrite hazard are both real.
- Random vector at cos 0.018 produces coherent skeleton (above) that is nothing like the prompt's motion (MPJPE 0.372, worse than zero-cond 0.307) — CLIP selects which motion; the motion prior is MDM's. Replicates.

## 4. precompute-cache — REPLICATES

Fresh process per config, new prompt "a person kicks with their right leg then turns around", seed 777:

| Config | Load | Peak VRAM | Params | vs baseline |
|---|---|---|---|---|
| A: full model, text path | 8.52 s | 456.3 MB | 169.16 M | — |
| B: `load_and_freeze_clip → Identity`, `clip.load` patched to raise, injected 2 KB embedding | 2.03 s | **95.0 MB** | 17.88 M | **bit-identical (0.0)** |

Peak VRAM reproduces their number to the decimal. Load times are noisier than theirs (cold caches) but the direction and rough ratio hold; CLIP weights provably never opened (`clip.load` raises if called — it wasn't).

## 5. footprint-floor — REPLICATES (spot-checked)

- Baseline fp32 peak 456.3 MB in my run vs their 435-456 MB range: consistent.
- MDM-only precompute floor: 95.0 MB peak, bit-exact (their cfg5: 90.6 MB with a different prompt/allocator state — same ballpark).
- `del clip_model.visual` in-process: params 81,308,424 (matches 81.31 M), peak 274.3 MB (their fresh-process 260.6 MB; mine carries same-process allocator arena), **bit-exact 0.0**.
- The claimed dtype gotcha is real: after deleting `visual`, `clip_model.dtype` raises AttributeError; their suggested property override (`transformer.resblocks[0].attn.in_proj_weight.dtype`) returns float16 and everything runs.
- fp16 sub-mm deltas and sm_75 dispatch checks not re-run (secondary to the headline; nothing observed contradicts them).

## 6. paraphrase-divergence — PARTIAL: headline does not survive a new intent set

New groups (run/crawl/throw/squat), 4 fresh paraphrases each, new seed 555; seed control 555/42/2718/9001:

- WITHIN-group mean **0.2032**, BETWEEN-group **0.5153** → ratio 0.394. The robust part — paraphrase ≪ intent change — **replicates**.
- But WITHIN (0.2032) vs SEED-NOISE (0.1776) → ratio **1.15, i.e. the headline direction FLIPS**: on my intent set, rewording moves the motion slightly MORE than reseeding, not less.
- Per group: run 0.83x, crawl 0.96x, throw 0.32x — below noise, consistent with their claim — but **squat 2.60x** (within 0.408 vs noise floor 0.157). Their own data had jump at 1.16x; posture-transition intents break the claim badly.
- Correction: "paraphrase < seed noise" is intent-dependent, not general. For locomotion/gesture intents it holds; for posture-transition intents (squat here, jump marginally in theirs) paraphrase choice moves the output 2-3x the seed-noise floor. The conclusion "a prompt-rewriting layer for phrasing is theatre" is overstated — and is also in tension with the vocabulary-probe's own measurement that "the man waves his right hand" articulates ~3x more than the imperative phrasing. Phrasing is cheap insurance precisely on the intents this model handles worst.

## 7. ood-phrasing — REPLICATES

Fresh seeds (21/22/23), fresh rewrites. Seed-noise floor (empty vs empty, root-relative) = 0.0904.

- Abstract prompts ignored: "vibes" d_empty 0.064 and "can you do a cool animation for my game please" 0.052 — both **below the seed-noise floor** (text literally does nothing); speeds 0.075/0.041 at or below the empty baseline. "make him look nervous" 0.105 ≈ floor, speed 0.059. Rewrites lift d_empty to 0.14-0.24 and speed 4-25x. The silent-idle failure and its rescue by rewriting **replicate**. (One of my four abstract prompts, "something epic and dramatic", was NOT ignored — consistent with their 53%-not-100% idle rate.)
- Action content breaks the skeleton regardless of phrasing: limb_cv_max "a person jumps forward with both feet" 0.996 mean (1.278 on one seed — their 1.288 reproduced), "a person jumps" 0.828, sofa-sit 0.311, vs kick 0.285 (theirs: 0.287) and walk-and-wave 0.097 (theirs: 0.081) clean. Empty 0.045-0.068 (theirs 0.064). Numbers land on top of theirs with different seeds. Only nit: "a person crouches down low" was mostly clean here (0.070), so "crouches break the skeleton" is seed/phrase-dependent; sits and jumps clearly do break.

## 8. vocabulary-probe — REPLICATES

Fresh seeds (11/12/13):

- Normalisation-bug evidence: `t2m_std[:4]` = [0.000515, 0.000677, 0.000677, 0.00612] exactly as claimed. Root XZ displacement over 6 s: empty 0.998 m, "stands still" 0.990 m, "walks forward" 1.196 m, "runs forward" 1.120 m, "runs in a circle" 1.386 m — everything drifts ~1 m regardless of text. The suppressed-root-translation bug is real and prompt-independent.
- Dead gesture category: with a null baseline (empty prompt) articulation of 0.121, ALL upper-body gestures sit at or below it — waves 0.106, salutes 0.106, claps 0.073, drinks 0.055, crosses arms 0.050. Meanwhile runs-in-circle 0.770, walks 0.413, kicks 0.404, sits 0.208, crawls drops root height to 0.710 m. Category structure reproduces exactly on seeds never used before.
- "Run is broken" replicates too: "runs forward" articulation 0.295 < "walks forward" 0.413, and rootXZ barely above standing.

## Overall

NOT REFUTED. Seven of eight findings replicate cleanly on fresh seeds, prompts, and lengths — including every load-bearing engineering claim (single pooled 512-d vector, image tower dead, `text_embed` injection bit-identical, CLIP-free 95 MB / 17.9 M-param inference path, root-translation bug, dead gesture category, silent abstract-prompt idle). One finding needs a correction: paraphrase-divergence's headline "rewording < reseeding" is intent-dependent and flipped on my intent set (squat paraphrases diverge 2.6x the seed-noise floor); its robust core (within-intent ≪ between-intent) still holds.
