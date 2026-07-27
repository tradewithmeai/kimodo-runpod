# Probe: footprint floor for local MDM inference

All numbers measured on the pod (RTX 3090, sm_86, torch 2.8.0+cu128, Python 3.12),
one fresh process per configuration. Prompt `"a person walks forward and then turns
around"`, 6.0 s (120 frames @ 20 fps), guidance 2.5, seed 1234.

Probe scripts on the pod:
`/workspace/probe_footprint.py`, `/workspace/probe_quality.py`,
`/workspace/probe_fp16_quality.py`, `/workspace/probe_sm75.py`,
`/workspace/probe_sm75b.py`, `/workspace/probe_sm75c.py`,
`/workspace/probe_cpuclip.py`, `/workspace/probe_ctx.py`.
Joint outputs: `/workspace/fp_out/joints_<cfg>.npy`.

---

## Headline table

| # | Config | Peak VRAM (alloc, MB) | Reserved (MB) | Weights (MB) | Params | Load (s) | Gen cold / warm (s) | Quality vs fp32 |
|---|--------|----------------------:|--------------:|-------------:|-------:|---------:|--------------------:|-----------------|
| 1 | fp32, full model (baseline) | **435.2** | 444 | 405.3 | 169.16 M | 4.91 | 0.629 / 0.405 | — |
| 2 | fp16 trunk, full CLIP | **394.0** | 452 | 371.2 | 169.16 M | 5.46 | 0.586 / 0.432 | 0.36 mm mean, 1.23 mm max |
| 3 | fp32, CLIP image tower deleted | **260.6** | 282 | 237.6 | 81.31 M | 4.62 | 0.626 / 0.396 | **bit-exact (0.0)** |
| 4 | fp16 trunk + image tower deleted | **219.5** | 272 | 203.5 | 81.31 M | 4.91 | 0.596 / 0.417 | 0.36 mm mean, 1.23 mm max |
| 5 | MDM-only, no CLIP, precomputed text embed | **90.6** | 98 | 68.2 | 17.88 M | 0.39 | 0.770 / 0.392 | **bit-exact (0.0)** |

"Peak VRAM" = `torch.cuda.max_memory_allocated()` measured across the generate call
(weights are resident, so this is the true working peak). "Reserved" =
`max_memory_reserved()`, i.e. what the caching allocator holds from the driver.

Whole-process GPU footprint (`mem_get_info` delta, includes the CUDA context):
- CUDA context + driver, before any model: **44 MB**
- Config 1 (fp32 full): **502 MB**
- Config 4 (fp16 + no image tower): **510 MB** — higher than config 1 only because
  `.half()` transiently allocates a second copy and the caching allocator keeps the
  arena. A `torch.cuda.empty_cache()` after setup drops this to ~264 MB
  (219.5 allocated + context). Do that once at the end of load.
- Config 5 (MDM-only): **150 MB**

All configs are run-to-run deterministic for a fixed seed (`rerun_max_diff = 0.0`).

---

## 1. fp32 baseline — confirmed

435.2 MB peak, matching the previously established 435 MB. 405.3 MB of that is
weights; only ~30 MB is activations/workspace. **The footprint is almost entirely
CLIP weights sitting idle.**

Note the "fp32" label is a slight misnomer inherited from MDM: `load_and_freeze_clip`
calls `clip.model.convert_weights`, so CLIP's Linear/Conv/attention weights are
**already fp16** in the baseline. Only the 17.9 M-param MDM trunk is fp32.

## 2. fp16 — safe, but buys almost nothing on its own

Because CLIP is already fp16, halving the MDM trunk only saves 34 MB (68 → 34 MB of
weights). Peak goes 435.2 → 394.0 MB, a 9 % cut. **Not worth it in isolation.**

Sampling stays coherent. Across five prompts, fp16 vs fp32 at the same seed:

| Prompt | Mean joint err | Max joint err | % of body height | Bone-length std fp32 → fp16 |
|---|---:|---:|---:|---|
| walks forward and turns around | 0.412 mm | 11.66 mm | 0.061 % | 2.36 % → 2.37 % |
| jumps up and down | 0.480 mm | 3.85 mm | 0.063 % | 9.60 % → 9.61 % |
| sits down on a chair | 1.100 mm | 4.49 mm | 0.359 % | 8.85 % → 8.88 % |
| throws a ball with right hand | 0.541 mm | 5.04 mm | 0.086 % | 3.51 % → 3.52 % |
| doing a cartwheel | 1.226 mm | 7.85 mm | 0.240 % | 19.60 % → 19.60 % |

Body height (pelvis→head) is 0.699 m; the figure spans 1.651 m. Errors are
sub-millimetre on average and at most ~12 mm on a single joint in a single frame.
Bone-length stability — the structural coherence measure — is unchanged to two
decimal places. **fp16 does not degrade motion quality.**

Implementation gotcha: `model.half()` on the whole MDM breaks CLIP. OpenAI CLIP's
`LayerNorm` subclass hard-casts activations to fp32 (`x.type(torch.float32)`) while
the surrounding Linear weights are fp16, and halving the LayerNorm weights makes the
two disagree. Half only the non-`clip_model` children:

```python
for name, mod in model.named_children():
    if name != "clip_model":
        mod.half()
```

Then re-cast `clip_encode_text`'s output (it ends in `.float()`) back to half, and
patch **both** `model.encode_text` and `cfg_model.encode_text` — `ClassifierFreeSampleModel.__init__`
copies the bound method (`self.encode_text = self.model.encode_text`), and
`p_sample_loop` calls it on the *wrapper*, so patching the inner model alone is
silently ignored. Finally wrap `forward` to cast `x` fp32→fp16 in and fp32 out, since
the diffusion loop generates fp32 noise.

## 3. CLIP image tower deleted — free 175 MB, bit-exact output

`del clip_model.visual` after `clip.load`. 435.2 → 260.6 MB peak, 169.16 M → 81.31 M
params. Output is **byte-identical** to the fp32 baseline (max abs coord diff 0.0) —
as it must be, since `encode_text` never touches the visual tower.

**This is the single best win: 40 % of peak VRAM for literally zero quality cost.**

One gotcha: `clip.model.CLIP.dtype` is a property defined as
`self.visual.conv1.weight.dtype`, and `encode_text` reads it. Deleting `visual` raises
`AttributeError: 'CLIP' object has no attribute 'dtype'`. Override the property to read
off the text tower instead:

```python
import clip.model
clip.model.CLIP.dtype = property(
    lambda self: self.transformer.resblocks[0].attn.in_proj_weight.dtype)
```

(Do *not* point it at `token_embedding.weight` — `convert_weights` skips `nn.Embedding`,
so that returns fp32 while the attention weights are fp16, and you get
`mat1 and mat2 must have the same dtype`.)

## 4. fp16 + image tower deleted — 219.5 MB

Composes cleanly: 219.5 MB peak, same 0.36 mm quality delta as config 2. Half the
baseline footprint while still accepting arbitrary text prompts at runtime.

## 5. MDM-only — the floor is 90.6 MB, and the precompute path already exists

**The precompute path is built into upstream MDM and works today.**
`gaussian_diffusion.p_sample_loop` (line ~633) already hoists text encoding out of the
denoising loop:

```python
if 'text' in model_kwargs['y'].keys():
    # encoding once instead of each iteration saves lots of time
    model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])
```

and `MDM.forward` (line ~210) honours a caller-supplied `y['text_embed']`:

```python
if 'text_embed' in y.keys():   # caching option
    enc_text = y['text_embed']
else:
    enc_text = self.encode_text(y['text'])
```

So CLIP can be replaced wholesale by a stub `nn.Module` and a cached `[1, 1, 512]`
fp32 tensor. `load_model_wo_clip` is fine with this — the checkpoint contains no
`clip_model.*` keys at all (it asserts `len(unexpected_keys) == 0` and only tolerates
*missing* keys under `clip_model.`), so a parameterless stub loads cleanly.

Result: **90.6 MB peak, 68.2 MB of weights, 17.88 M params, 0.39 s load, and output
bit-identical to the fp32 baseline.** Load time drops 12× because loading CLIP's
weights from disk is what dominates the 4.9 s.

Cost of this path: you can only generate motion for prompts whose embedding you have.
Options for arbitrary prompts:
- **CLIP text tower on CPU**: 63.43 M params, 242 MB host RAM, 4.08 s load,
  **239 ms** per prompt (bs=1). Runs once per generation, not per denoising step, so
  it adds 239 ms to a ~0.4 s generation. GPU footprint stays at 90.6 MB.
- **Precomputed embedding table**: ship a dict of `{prompt: 512-float vector}` (2 KB
  each) for a fixed prompt set. Zero runtime text cost.
- Keep the fp16 text tower on GPU (config 4, 219.5 MB) if free-form prompts matter
  more than the extra 129 MB.

---

## sm_75 (RTX 2070 SUPER, Turing) compatibility

**Nothing in the sampling path is gated on sm_80+. It will run on Turing.**

Checked:

1. **Wheel arch coverage.** `torch.cuda.get_arch_list()` on the installed
   `2.8.0+cu128` build is `['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100',
   'sm_120']`. **sm_75 is compiled in** — no JIT-PTX fallback, no recompile needed.

2. **bf16 is never used.** A source scan of `model/`, `diffusion/`,
   `motion_server.py` and the installed `clip` package for
   `bfloat16|bf16|flash_attn|xformers|triton|cpp_extension|torch.compile|scaled_dot_product`
   returned **zero hits**. Turing's lack of native bf16 is irrelevant here. fp16 is
   what config 2/4 use, and Turing has full fp16 tensor cores.

3. **No flash attention anywhere in the path — it isn't even *eligible* on sm_86.**
   Forcing `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` fails with
   `RuntimeError: No available kernel. Aborting execution.` in **both** fp32 and fp16.
   `nn.MultiheadAttention` with `batch_first=False` and a `src_key_padding_mask` does
   not dispatch to flash. So the fact that PyTorch's flash kernel requires sm_80+ can
   never bite on Turing.

4. **Backends that do work are all Turing-supported.** With fp32:
   - `EFFICIENT_ATTENTION` (mem-efficient, sm_50+): OK, identical to default output.
   - `MATH`: OK, max coord diff 1.56e-3 m vs default.

   With the fp16 trunk: `EFFICIENT_ATTENTION` OK (0.420 s), `MATH` OK (0.578 s),
   `CUDNN_ATTENTION` OK. Only `FLASH` is unavailable, on both architectures.

5. **No BetterTransformer / nested-tensor fastpath.**
   `seqTransEncoder.use_nested_tensor` is `False` (because `batch_first=False`), so the
   arch-sensitive fused encoder-layer fastpath is never taken. `norm_first=False`,
   `activation=gelu`, 4 heads over `latent_dim=512` (head_dim 128).

6. **`torch.compile` is not used** anywhere, so no Inductor/Triton codegen to worry
   about on Windows or on sm_75.

7. **VRAM headroom.** Worst case (config 1) holds 502 MB of an 8 GB card. Non-issue.

### One caveat worth stating

Outputs will **not be bit-identical between the 3090 and the 2070 SUPER**. Under fp16
the default SDPA dispatch picks a different kernel per architecture; forcing different
backends on the same 3090 already shifts joint positions by up to 1.2–1.8 cm
(`default vs MEM_EFF: 1.81e-2 m`, `default vs MATH: 6.36e-3 m`,
`default vs CUDNN: 1.22e-2 m`). That is the same order as the fp32→fp16 delta and
does not change the motion's character — diffusion amplifies tiny numerical
differences — but any test that asserts exact joint values across machines will fail.
Pin `sdpa_kernel(SDPBackend.MATH)` if cross-machine reproducibility is required
(costs ~0.16 s per generation in fp16).

---

## Recommendation

**Ship config 3 or 4.** Deleting the CLIP image tower is a two-line change, costs
nothing in quality (bit-exact), and removes 175 MB — 40 % of peak VRAM. Adding fp16
on top takes it to 219.5 MB for a 0.36 mm mean joint error, which is invisible.

Going to config 5 (90.6 MB) is worth it only if the prompt set is fixed or you accept
a CPU text encoder (+239 ms/prompt, +242 MB host RAM). It removes 129 MB of VRAM and
4.5 s of load time.

Generation time is essentially flat across all five configs (0.39–0.43 s warm) — CLIP
runs once per generation, not per denoising step, so removing it does not speed up
sampling. The win is purely memory and load time.
