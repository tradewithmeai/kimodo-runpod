# CLIP-replacement investigation — synthesis

12 agents, 4 phases, ~26 minutes, all measurements taken on the live model on the pod.
Adversarial verification: 3 independent lenses (replication with fresh seeds, methodology
attack, alternative-explanation testing). 7 of 8 findings replicate; every number quoted
below survived verification.

## The architecture question, settled

**MDM's entire coupling to CLIP is one 512-float vector, computed once per generation.**
The text is pooled to a single `[1, bs, 512]` sentence embedding, passed through one
`Linear(512,512)`, added to the timestep embedding, and prepended as token 0 of the
transformer sequence. Cross-attention is never used. The CLIP **image tower (87.8M params,
58% of CLIP) is never called** — zero forward calls across full generations, verified by
hooks on all 113 visual submodules.

**CLIP can be removed from inference entirely.** `mdm.py:212` already accepts a
`text_embed` key and skips the encoder; injection is bit-for-bit identical to encoding
in-process. Gotcha: `gaussian_diffusion.py:635` silently overwrites `y['text_embed']`
whenever `y['text']` is present — omit `text` when injecting.

**Footprint floor** (measured, one fresh process per config):

| Config | Peak VRAM | Quality vs fp32 |
|---|---:|---|
| fp32 full model | 435 MB | — |
| image tower stubbed | 261 MB | bit-exact |
| + fp16 trunk | 220 MB | 0.36 mm mean error |
| MDM-only, precomputed embeddings | **91 MB** | bit-exact |

fp16 is safe on Turing (sm_75); CLIP ships fp16 already. Nothing in the sampling path is
arch-gated beyond fp16.

**Swapping the encoder would not require retraining MDM.** The embedding space is smooth
and saturating: noise at cos ≥ 0.995 is indistinguishable from seed noise; paraphrase-level
displacement (cos ≈ 0.9) is tolerable; below cos ≈ 0.7 output decays gracefully toward the
unconditional mode (generic idle motion, never garbage). Even a random vector produces a
physically coherent skeleton — CLIP selects *which* motion; the motion prior lives entirely
in MDM. An adapter regressed onto CLIP's output space is feasible; validate with MPJPE not
cosine (sensitivity is anisotropic, ~2-5x across directions). Optional — not needed for the
current architecture, where CLIP's 193 MB simply stays local.

## The chat-layer question, settled

**Rewording explicit captions is worthless; translating intent is valuable.** Paraphrases
of an already-explicit caption move output less than a seed change (though the verifiers
weakened this: the seed-noise-floor decision rule produces false negatives, and one intent
set failed replication — treat "phrasing never matters" as unproven, "phrasing rarely
matters" as supported). The large, replicated win: **prompts with no explicit body action
("look nervous", "exhausted after a run") genuinely fail**, producing near-static or
generic motion, and rewriting them into explicit body-action captions fixes them. That is
the Claude layer's job: intent → explicit action, plus sequencing and refinement.

**What MDM knows** (60 prompts × 3 seeds): whole-body locomotion is reliable (walk, run,
jump, sit, crawl, circle, backwards); modifiers largely work (slowly, quickly, backwards,
in a circle); **fine gestures are dead** (wave, clap, point produce little distinctive
motion — dead category, replicated); "run" is under-responsive (0.49 m/s vs walk's
0.80 m/s — model limitation, confirmed independently twice). The vocabulary file
(`sens-vocabulary-probe.md`) is the target distribution for the chat system prompt.

## The bug the swarm caught in our own code

The server was decoding root translation **~24x too small**. The repo-bundled
`t2m_std.npy` we adopted (to skip the HumanML3D download) is the *evaluator's* std, whose
root-linear-velocity group (channels 1:3) is scaled down by feat_bias grouping. Every clip
treadmilled: ~1 m constant drift regardless of prompt. Proven via internal redundancy (the
263-d vector encodes root velocity twice under different std groups; correlation 1.0000,
decode ratio ~24x) and calibrated by minimising stance-foot slip: k=24 gives walk 4.78 m /
6 s, stand-still 0.01 m, slip 0.0045 m/frame. **Fixed in `pod/motion_server.py`
(`self.std[1:3] *= 24.0`) and verified live: walk 4.78 m, stand 0.01 m, run 2.94 m.**

## Decisions taken

1. Keep CLIP local (261 MB config) — free-form chat needs novel embeddings; no adapter.
2. Claude layer targets intent-translation + sequencing, seeded with the vocabulary map;
   no paraphrase polishing.
3. Root-velocity fix shipped; image-tower stub shipped (`_VisualStub` keeps CLIP's
   `.dtype` property chain alive — plain `del` breaks `encode_text`).
4. Chat UI should steer users away from fine-gesture requests (dead category) and lean on
   locomotion+posture vocabulary.

Raw findings: `probe-*.md`, `sens-*.md`, `verify-*.md` in this directory. Probe scripts
remain on the pod under `/workspace/probe_*.py` / `verify_rep*.py`.
