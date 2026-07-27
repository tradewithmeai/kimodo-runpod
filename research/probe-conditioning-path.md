# Probe: CLIP -> MDM conditioning path

All facts below were measured on the live model on the pod (checkpoint
`humanml_enc_512_50steps/model000750000.pt`), cross-checked against
`/workspace/repos/motion-diffusion-model/model/mdm.py`.

Probe scripts: `/workspace/probe_condpath.py`, `/workspace/probe_condpath2.py`

## Live config (measured)

```
arch                = trans_enc
cond_mode           = text
emb_policy          = add
text_encoder_type   = clip
clip_version        = ViT-B/32
clip_dim            = 512
latent_dim          = 512
emb_trans_dec       = False
dataset             = humanml
njoints/nfeats      = 263 / 1
mask_frames         = False
is_prefix_comp      = False
encode_text bound to: clip_encode_text
embed_text          = Linear(in_features=512, out_features=512, bias=True)
```

## 1. What encodes text, and what shape does it produce

`MDM.encode_text` is aliased to `MDM.clip_encode_text` (mdm.py:113).
It does (mdm.py:163-178):

- `clip.tokenize(raw_text, context_length=22, truncate=True)` — humanml hardcodes
  max 20 tokens + BOS + EOS, then zero-pads back to CLIP's 77 context length.
- `self.clip_model.encode_text(texts).float().unsqueeze(0)`

Measured output:

```
bs=1: torch.Size([1, 1, 512])   float32, cuda:0
bs=2: torch.Size([1, 2, 512])
```

Shape is `[1, batch, 512]` — a SINGLE pooled sentence vector per prompt
(CLIP's EOS-token pooled + text_projection output), not a per-token sequence.
The leading 1 is a sequence axis of length 1.

Then `self.embed_text` (`nn.Linear(512 -> 512)`, 262,656 params) maps it to the
MDM latent dim. Measured `embed_text` output: `torch.Size([1, 1, 512])`.

## 2. Where it enters the diffusion transformer

Path (mdm.py:209-253), confirmed by forward hooks:

1. `time_emb = self.embed_timestep(timesteps)` -> measured `[1, 1, 512]`
2. `text_emb = self.embed_text(self.mask_cond(enc_text, force_mask))` -> `[1, 1, 512]`
3. `emb_policy == 'add'`, so `emb = text_emb + time_emb` -> `[1, 1, 512]`
   (text is ADDED to the timestep embedding, they share one token)
4. `xseq = torch.cat((emb, x), axis=0)` — the combined embedding is
   **prepended as token 0** of the motion sequence.
5. `self.seqTransEncoder(xseq)[1:]` — the prefix token is dropped after the stack.

Measured input to `seqTransEncoder` for a 6 s (120-frame) generation:

```
xseq shape into seqTransEncoder: torch.Size([121, 1, 512])   # 120 frames + 1 cond token
```

So: **NOT cross-attention** (that is the `trans_dec` / BERT branch, unused here).
It is a self-attention encoder where text is summed into the timestep embedding
and that single vector is concatenated as an extra token at position 0.

CFG: `ClassifierFreeSampleModel` runs the model twice per step. Measured
**100 MDM.forward calls** for a 50-step generate. The uncond pass sets
`y['uncond']=True`, which makes `mask_cond` return `torch.zeros_like(enc_text)`
— CLIP is NOT re-run for the uncond branch.

Text is encoded ONCE per generate, not per step:
`diffusion/gaussian_diffusion.py:635` caches
`model_kwargs['y']['text_embed'] = model.encode_text(...)` before the loop, and
`MDM.forward` reuses it (mdm.py:210-213).
Measured: CLIP text transformer fired **1 time** across the whole 50-step,
100-forward generation.

## 3. Is the CLIP IMAGE tower used? NO — proven

Registered forward hooks on `model.clip_model.visual` **and all 113 of its
submodules**, then ran a full `generate("a person walks forward", 6.0, 2.5)`.

```
hooks registered on visual + 113 submodules
generate out shape (120, 22, 3)
VISUAL forward calls: 0    visual-submodule calls: 0
TEXT transformer forward calls: 1
```

Gradient check after generation:

```
any visual param requires_grad: False
any visual param .grad not None: False
any clip param requires_grad: False   # whole CLIP frozen in load_and_freeze_clip (mdm.py:147-149)
```

The image tower is dead weight during text-to-motion sampling: 0 forward calls,
0 gradients, frozen. `logit_scale` and the contrastive head are likewise unused —
only `token_embedding`, `positional_embedding`, `transformer`, `ln_final`,
`text_projection` participate.

## 4. Param counts (measured)

| Group | Params |
|---|---|
| CLIP total | 151,277,313 |
| — `visual` (IMAGE tower) | **87,849,216** |
| — text side total | **63,428,096** |
| — `logit_scale` (contrastive temp) | 1 |
| MDM total (incl. CLIP) | 169,157,640 |
| MDM without CLIP | 17,880,327 |
| `embed_text` linear (512->512) | 262,656 |

Text-side breakdown:

```
transformer          37,828,608   # 12 resblocks, width 512
token_embedding      25,296,896   # 49408 x 512 vocab
text_projection         262,144   # 512 x 512
positional_embedding     39,424   # 77 x 512
ln_final                  1,024
```

## Implications for removing CLIP

- The image tower (87.85 M params, 58% of CLIP, ~168 MB fp16) is provably never
  executed. Deleting `clip_model.visual` is a free win — no code path touches it.
- The remaining text side is 63.43 M params, of which 25.3 M is just the token
  embedding table (49,408-word vocab), most of which a motion prompt never hits.
- The entire interface between CLIP and MDM is **one 512-d float32 vector per
  prompt**, computed once per generation (measured: 1 CLIP call per 50-step
  sample). Any replacement only needs to emit a `[1, bs, 512]` tensor in the same
  embedding space that `embed_text`'s learned 512->512 Linear expects — a cached
  lookup table, a distilled tiny text encoder, or a precomputed embedding
  shipped with the app would all satisfy the contract.
- Because CLIP runs once (~1 of 100 forwards), removing it saves VRAM/disk, not
  meaningful sampling latency.
