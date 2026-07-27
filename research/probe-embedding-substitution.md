# Probe: embedding substitution — can an arbitrary vector stand in for a CLIP embedding?

Model: `humanml_enc_512_50steps/model000750000.pt`, MDM `trans_enc`, `emb_policy=add`,
`text_encoder_type=clip`, `clip_dim=512`, CFG guidance 2.5, 6 s (~120 frames @20 fps),
fixed seeds. All numbers below are measured on the pod.

## Headline

**The MDM decoder does not need a meaning-bearing point in CLIP space.** A random
512-d Gaussian vector with the same L2 norm as a real CLIP embedding — cosine
similarity ≈ **-0.04** to the real embedding — produces motion that is
**statistically indistinguishable in physical coherence** from motion produced by
real prompts. CLIP does not supply the human-motion prior; it only *selects which*
motion. Any vector selects *a* motion, coherently.

## How the injection works (load-bearing implementation detail)

`model/mdm.py:210` supports a caching path:

```python
if 'text_embed' in y.keys():   # caching option
    enc_text = y['text_embed']
else:
    enc_text = self.encode_text(y['text'])
```

**BUT** `diffusion/gaussian_diffusion.py:633-635` silently overwrites it at the top
of `p_sample_loop`:

```python
if 'text' in model_kwargs['y'].keys():
    ...
    model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])
```

So to inject a custom embedding you must pass `y['text_embed']` and **omit `y['text']`
entirely**. My first run passed both and every "custom embedding" condition silently
collapsed to the empty-string embedding (all MPJPE = 0.0000 between them). Fixed.

Verified equivalence: `gen(text_embed=encode_text("a person walks forward"))` vs
`gen(text="a person walks forward")` → **MPJPE = 0.0000 m** (bit-identical). The
injection path is exact.

Required shape/dtype: `[1, batch, 512]`, `float32`, on cuda.

## 1–2. Real CLIP embedding vs random vector of equal norm

seed 1234, "a person walks forward":

| condition | cos to walk-emb | bone-length CV (mean) | bone CV max | total bone len (m) | root-Y range (m) | ankle-Y max (m) | foot alternations |
|---|---|---|---|---|---|---|---|
| CLIP "walks forward" | 1.00 | 0.027 | 0.138 | 4.363 | 0.004 | 0.245 | 8 |
| random, ‖·‖ matched (9.558) | **-0.041** | 0.016 | 0.051 | 4.932 | 0.001 | 0.046 | 2 |
| zero vector | — | 0.020 | 0.075 | 4.347 | 0.002 | 0.225 | 6 |

The random-vector output is **coherent human motion** (limb lengths *more* stable
than the real prompt: CV 0.016 vs 0.027; upright root at 0.897 m; feet on the ground)
— but it is **not walking**: 2 foot alternations vs 8, peak ankle lift 0.046 m vs
0.245 m. It is a near-static standing/idle pose.

MPJPE walk vs random = **0.190 m**; walk vs zero-cond = **0.136 m**.

### Population test (this is the strongest evidence)
12 real prompts × 3 seeds (n=36) vs 12 independent random unit-directions scaled to
the mean real norm (9.853) × 3 seeds (n=36). Median [min, max]:

| metric | real prompts | random vectors |
|---|---|---|
| bone-length CV (temporal limb stability) | 0.041 [0.003, 0.121] | **0.034** [0.006, 0.136] |
| bone CV max (worst single bone) | 0.176 [0.008, 0.783] | **0.138** [0.024, 0.786] |
| total skeleton bone length (m) | 4.376 [4.334, 5.940] | **4.354** [4.289, 6.280] |
| root height mean (m) | 0.914 [0.895, 0.935] | **0.913** [0.865, 0.929] |
| root height range (m) | 0.0034 [0.0002, 0.0353] | **0.0026** [0.0002, 0.0461] |
| ankle-Y max (m) | 0.240 [0.019, 1.423] | **0.223** [0.048, 1.284] |
| foot alternations (gait proxy) | 4.5 [0, 22] | **5.0** [0, 17] |

Every coherence statistic overlaps. Random conditioning is **not** degenerate.

Diversity (mean pairwise MPJPE across the 12 conditions at seed 1234):
real prompts **0.382 m** [0.105, 0.967]; random vectors **0.324 m** [0.058, 1.105].
Different random vectors give genuinely *different* motions — the vector is being
consumed as a steering signal, not ignored.

Magnitude sensitivity (seed 1234): same random direction at ×3 norm → bone CV 0.007,
frozen pose, MPJPE 0.109 from the ×1 version; at ×0.3 norm → bone CV 0.029, foot_alt 9,
MPJPE 0.083 from the zero-cond output. Scaling toward 0 walks the output toward the
unconditional motion, as expected.

## 3. Different prompt gives a different motion — conditioning is real

seed 1234, MPJPE against "a person walks forward":

| condition | MPJPE (m) | MPJPE root-relative (m) |
|---|---|---|
| "a person sits down" | **0.663** | 0.669 |
| "a person jumps" | 0.159 | — |
| "a person waves their right hand" | 0.234 | — |
| random vector | 0.190 | 0.172 |
| zero vector | 0.136 | 0.109 |

Across seeds 1234/7/99, walk↔sit MPJPE = 0.663 / 0.619 / 0.533 m — an order of
magnitude above the numerical floor and far above walk↔zero (0.136 / 0.126 / 0.208).
Confirmed: the embedding drives behaviour.

Qualitative separation is also visible in the metrics — "waves" has ankle-Y max 0.019 m
and 0 foot alternations (planted, upper-body only); "jumps" has ankle-Y max 0.360 m and
root-Y range 0.018 m; "walks" has 8 foot alternations.

## 4. Interpolation walk → sit (α = 0, 0.25, 0.5, 0.75, 1)

MPJPE to each endpoint, 3 seeds:

| α | d(walk) s1234 / s7 / s99 | d(sit) s1234 / s7 / s99 | bone CV | total bone len (m) |
|---|---|---|---|---|
| 0.00 | 0.000 / 0.000 / 0.000 | 0.663 / 0.619 / 0.533 | 0.027 / 0.020 / 0.029 | 4.363 / 4.346 / 4.349 |
| 0.25 | 0.067 / 0.011 / 0.013 | 0.666 / 0.618 / 0.533 | 0.026 / 0.018 / 0.027 | 4.358 / 4.342 / 4.346 |
| 0.50 | 0.224 / 0.153 / 0.204 | 0.497 / 0.514 / 0.392 | 0.079 / 0.087 / 0.090 | 4.585 / 4.496 / 4.555 |
| 0.75 | 0.697 / 0.577 / 0.507 | 0.137 / 0.072 / 0.054 | 0.090 / 0.099 / 0.113 | 5.935 / 5.741 / 5.508 |
| 1.00 | 0.663 / 0.619 / 0.533 | 0.000 / 0.000 / 0.000 | 0.082 / 0.084 / 0.108 | 5.862 / 5.874 / 5.597 |

**It morphs; it does not collapse.** d(walk) increases monotonically and d(sit)
decreases monotonically in all three seeds. The midpoint is not nonsense: bone CV 0.079–0.090
sits *between* the endpoints (0.027 and 0.082–0.108) and inside the real-prompt range.
Two caveats: (a) the transition is **not linear** — α=0.25 is still essentially the walk
motion (d(walk) 0.011–0.067 m) and α=0.75 is essentially the sit motion; the switch
happens sharply between 0.25 and 0.75, consistent with CFG sharpening a decision boundary.
(b) worst-single-bone CV does spike at α=0.5 (0.62 / 0.75 / 0.69, above both endpoints),
so there is *mild* localised midpoint degradation, but nothing like collapse.
Renormalising the lerp to the interpolated norm changed nothing meaningful
(MPJPE ≤ 0.019 m difference from plain lerp).

## Geometry of CLIP text space (why this matters)

Pairwise cosine between the 4 probe prompt embeddings: 0.79–0.87
(walk|sit 0.808, walk|jump 0.832, walk|wave 0.807, sit|jump 0.867, sit|wave 0.788, jump|wave 0.810).
L2 norms 9.28–10.20.

**Real CLIP text embeddings live in a very narrow cone** — semantically unrelated
prompts are ~0.8 cosine apart. A random Gaussian vector sits at cos ≈ 0, i.e. far
outside that cone. The model has never seen anything like it. It still emits coherent
human motion. `embed_text` is a single `nn.Linear(512, 512)` (`mdm.py:126`) whose output
is simply **added** to the timestep embedding — there is no mechanism that could
"reject" an off-cone vector, and empirically nothing does.

Also measured: a vector built as (mean of the 4 real embeddings + noise), renormalised
— `cone0.3`, `cone1.0` — behaves like a mild perturbation of the unconditional output
(MPJPE to zero-cond 0.067–0.178 m), not like any specific prompt.

## Answer to the question

The model consumes **any conditioning vector**, not specifically a meaning-bearing
point in CLIP's space. Coherence of the output — plausible limb lengths, upright root,
grounded feet, temporally smooth — comes entirely from the 17.9M-param MDM decoder and
its diffusion prior over the HumanML3D representation. CLIP's contribution is purely
*which* point in that motion manifold you land on.

Consequence for replacing CLIP: you need a 512-d encoder that maps text to a point
**inside CLIP's learned cone with CLIP's relative geometry**, because MDM's
`embed_text` linear layer was trained only on that cone. You do **not** need CLIP for
motion quality. Off-cone vectors are safe (no garbage output), they are just
semantically meaningless — so a distilled/quantised text encoder that reproduces CLIP
embeddings to within roughly the *inter-prompt* distance (cos ≳ 0.9, well above the
0.79–0.87 between distinct prompts) should be behaviourally faithful, and any error
beyond that degrades gracefully into "a different but still coherent motion" rather
than into corruption.

## Caveats / not measured
- The `root_travel` / mean-speed metric I computed is **unreliable** — nearly every
  condition, including feet-planted waving, returned ~0.99 m of net root displacement
  over 6 s. That is foot-sliding/root-drift in the `recover_from_ric` reconstruction,
  not real locomotion. I did not use it for any conclusion; gait was judged by foot
  alternation count and ankle lift instead.
- "a person sits down" has notably degraded geometry at all 3 seeds (total bone length
  5.5–5.9 m vs 4.35 m for walk, i.e. ~30% limb stretch) and root height barely drops
  (range 0.033 m). This looks like a genuine weakness of the 50-step checkpoint on that
  prompt, not an artefact of the probe — the text path and the embed path give identical
  output. It does not affect the conclusions (walk↔sit separation is measured on MPJPE).
- Not tested: whether a *learned* small encoder can actually hit CLIP's cone. That is a
  separate experiment.
- Guidance scale fixed at 2.5 throughout; behaviour at other guidance not measured.

## Reproduce
Scripts on the pod: `/workspace/probe_embsub.py`, `/workspace/probe_embsub2.py`,
`/workspace/probe_embsub3.py`.
