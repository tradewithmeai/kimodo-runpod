# Probe: precompute-cache — can CLIP be removed from inference entirely?

**Answer: YES. Completely, and bit-for-bit losslessly.**

MDM already ships a first-class cache hook. `model/mdm.py:212-214`:

```python
if 'text' in self.cond_mode:
    if 'text_embed' in y.keys():  # caching option
        enc_text = y['text_embed']
    else:
        enc_text = self.encode_text(y['text'])
```

Pass `y['text_embed']` and `encode_text` (and therefore `clip_model`) is never
called. CLIP can be deleted from the GPU, or never constructed at all.

All numbers below were measured on the pod (3090), prompt
`"a person walks forward and waves"`, 6 s / 120 frames, guidance 2.5, seed 1234.

---

## 1. The embedding is 2 KB

`MDM.clip_encode_text` returns `.float().unsqueeze(0)` → shape `[1, bs, 512]`,
float32.

| Artifact | Bytes |
|---|---|
| Raw tensor `[1,1,512]` fp32 | **2048** |
| `torch.save` fp32 `.pt` | 3653 |
| `torch.save` fp16 `.pt` | 2629 |
| `np.save` fp32 `.npy` | 2176 |
| 1000 prompts, `[1000,512]` fp32 `.pt` | 2,049,626 (2.0 MB) |
| 1000 prompts, fp16 `.pt` | 1,025,626 (1.0 MB) |

**2 KB per prompt fp32, 1 KB fp16.** Encoding 1000 prompts took 0.375 s, so
building a cache is trivial. The whole HumanML3D text corpus (~45k captions)
would be ~92 MB fp32 / ~46 MB fp16 — against 354 MB for the CLIP weights.

## 2. CLIP-free generation works — exact working code

```python
@torch.no_grad()
def generate_cached(m, text_embed, seconds=6.0, guidance=2.5, seed=None):
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    n_frames = int(min(max(seconds, 1.0) * ms.FPS, ms.MAX_FRAMES))
    bs = 1
    model = m._cfg_model if guidance != 1.0 else m.model
    model_kwargs = {"y": {
        "mask": torch.ones(bs, 1, 1, n_frames, dtype=torch.bool, device=m.device),
        "lengths": torch.tensor([n_frames] * bs, device=m.device),
        "text_embed": text_embed.to(m.device),   # <-- replaces "text": [str]
        "scale": torch.ones(bs, device=m.device) * guidance,
    }}
    shape = (bs, m.model.njoints, m.model.nfeats, n_frames)
    sample = m.diffusion.p_sample_loop(model, shape, clip_denoised=False,
        model_kwargs=model_kwargs, skip_timesteps=0, init_image=None,
        progress=False, dump_steps=None, noise=None, const_noise=False)
    x = sample.cpu().permute(0, 2, 3, 1).float().numpy() * m.std + m.mean
    joints = ms.recover_from_ric(torch.from_numpy(x).float(), ms.N_JOINTS)[0, 0].numpy()
    joints[:, :, 1] -= joints[:, :, 1].min()
    return joints
```

The **only** change from `MotionModel.generate` is `"text": [text]` →
`"text_embed": tensor[1,1,512]`. Everything else — CFG wrapper, `mask_cond`
uncond zeroing, `p_sample_loop` — is untouched.

Producing the embedding (once, anywhere, on any machine with CLIP):

```python
with torch.no_grad():
    emb = m.model.encode_text(["a person walks forward and waves"])  # [1,1,512] fp32
torch.save(emb.cpu(), "emb.pt")
```

### Two ways to get rid of CLIP

**(a) Delete after load** — works:
```python
del m.model.clip_model
gc.collect(); torch.cuda.empty_cache()
```
`clip_model` is referenced nowhere except `clip_encode_text` / `bert_encode_text`
(grep of `mdm.py`: lines 112, 120-121, 140-151, 178, 184). `ClassifierFreeSampleModel`
holds `self.encode_text = self.model.encode_text` — a bound method on the MDM
module, not an extra CLIP reference — so the delete actually frees.

**(b) Never construct it** — better:
```python
import model.mdm as _mdm
_mdm.MDM.load_and_freeze_clip = lambda self, clip_version: torch.nn.Identity()
```
`load_model_wo_clip` already skips `clip_model.*` keys, so this loads clean.
Verified with `clip.load` and `clip._download` monkeypatched to raise: the run
completed, so **the 354 MB `ViT-B-32.pt` is never opened**.

### Fidelity

Same seed, all three paths (baseline / del-CLIP / never-load-CLIP):

```
max_abs_diff_vs_baseline = 0.0
mean_abs_diff            = 0.0
```

**Bit-identical.** Not "close" — the cached embedding is literally the same
tensor the encoder would have produced, so the denoiser sees identical input.

## 3. Peak VRAM

| Path | Resident after load | Peak during generation |
|---|---|---|
| Normal (CLIP present) | 444.2 MB | **456.3 MB** |
| CLIP deleted after load | 83.1 MB | **95.0 MB** |
| CLIP never loaded | 83.1 MB | **95.0 MB** |

**456 MB → 95 MB, a 79% cut.** Transient peak above resident weights is ~12 MB
in both cases — the diffusion loop itself is tiny; CLIP was ~361 MB of dead
weight sitting on the GPU doing nothing for 50 of the 50 denoising steps.

(Note: `del` frees the memory but the *transient* peak during load is still
444 MB, since CLIP is materialised before being dropped. Only path (b) keeps
peak-load at 83 MB.)

## 4. Load time

| Path | `MotionModel(...)` construction |
|---|---|
| With CLIP | 4.84 s (9.14 s cold-page first run) |
| CLIP deleted after load | 5.03 s |
| **CLIP never loaded** | **0.58 s** |

**8.3x faster startup.** `del`-after-load buys nothing on time (you still pay
the load); you have to skip construction to win.

Generation time is unchanged: 0.46 s baseline vs 0.40 s (del) / 0.41 s (never).
Text encoding was never the bottleneck — the 50-step loop is.

## 5. What actually ships

| Artifact | Bytes |
|---|---|
| Checkpoint `model000750000.pt` as distributed | 81,818,987 (81.8 MB) |
| — clip keys inside it | **0 of 108 keys** |
| Re-saved MDM-only state dict | 81,797,274 |
| Re-saved MDM-only, fp16 | 46,037,395 (46.0 MB) |
| CLIP ViT-B-32.pt (`~/.cache/clip/`) | 353,976,522 (354 MB) |

The shipped checkpoint **already contains zero CLIP weights** — CLIP is fetched
separately by `clip.load()`. So dropping CLIP removes a 354 MB download and
361 MB of VRAM, and costs nothing from the checkpoint.

Param counts confirm: 169,157,640 with CLIP → **17,880,327 without**.

### The one remaining dependency

`model/mdm.py` line 5 is a module-level `import clip`. The pure-python `clip`
package must still be importable even in the never-load path (it is only the
*weights* that are avoided). For a lean local deploy, vendor `mdm.py` and drop
that import along with the `bert` import.

---

## Verdict for local deploy (RTX 2070 SUPER, 8 GB)

A local build can ship **17.9M-param MDM only** — 82 MB fp32 / 46 MB fp16 —
plus a `dict[str, tensor[1,1,512]]` embedding cache at 2 KB per prompt, and get
**bit-identical motion** at **95 MB peak VRAM** and **0.58 s startup**.

The tradeoff is the obvious one: a pure cache only serves prompts you
precomputed. Options, in order of cost:
- **Fixed prompt set** (game with N canned actions): ship the cache, CLIP never
  exists anywhere. ~2 KB/prompt.
- **Open-ended prompts**: CLIP text-tower-only ONNX/int8 on CPU, or a remote
  embedding endpoint returning 2 KB. The motion model never needs it.
- CLIP is a *pure function of the string* here — it is called once per
  generation, before the loop, and takes no gradient. Nothing about the design
  requires it to be co-located.

## Reproduction

Scripts on the pod:
- `/workspace/probe_precompute_extract.py` — extract + save + size + baseline
- `/workspace/probe_precompute_clipfree.py` — modes `base` | `del` | `never`
- `/workspace/probe_precompute_ship.py` — artifact sizes, 1000-prompt cache
- `/workspace/probe_precompute_noclipfile.py` — proves CLIP weights never opened

Cache artifacts in `/workspace/emb_cache/`.
