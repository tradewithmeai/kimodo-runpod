# Licensing: what we own, what we don't, and how to own it

**Not legal advice** — I'm reporting what the licences say, with sources. For anything that
ships commercially, have a solicitor read the AMASS and CMU terms directly.

## Verdict in one line

The **code** is ours to use commercially. The **weights are not** — and the weights are the
only real problem. Fixing it means training our own, which is affordable and has a large
side benefit.

## Component by component

| Component | Licence | Commercial? | Notes |
|---|---|---|---|
| MDM source code | MIT (Guy Tevet, 2022) | ✅ Yes | Verified in the local clone's LICENSE |
| CLIP (model + weights) | MIT (OpenAI) | ✅ Yes | Text encoder only; we never touch the image tower |
| Our server, viewer, scripts | Ours | ✅ Yes | Written from scratch this project |
| three.js | MIT | ✅ Yes | |
| PyTorch, FastAPI, etc. | BSD/MIT/Apache | ✅ Yes | |
| **`model000750000.pt` checkpoint** | **Inherits AMASS** | ❌ **No** | The blocker |
| **`t2m_mean.npy` / `t2m_std.npy`** | Derived from HumanML3D | ⚠️ Tainted | Statistics computed over AMASS-derived data |
| SMPL body model | Research-only (Max Planck) | ❌ No | **Not used** — I stubbed `Rotation2xyz`; `body_models/` is empty locally |

MDM's own README is explicit that MIT covers the code only: *"Note that our code depends on
other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have
their own respective licenses that must also be followed."*

## The blocker, precisely

The checkpoint was trained on **HumanML3D**, which is a re-annotation of **AMASS**.
HumanML3D's own code is MIT, but it cannot redistribute the motion data — *"Due to the
distribution policy of AMASS dataset, we are not allowed to distribute the data directly."*

AMASS's licence is unambiguous, and one clause kills this outright:

> "Any other use, in particular any use for commercial purposes, is prohibited."

> **"This license also prohibits the use of the Dataset to train methods/algorithms/neural
> networks/etc. for commercial use of any kind."**

That second sentence is the one that matters. It reaches past the data and binds the
*trained model*. So the checkpoint we're running is research-only, and everything it
generates inherits that.

A commercial AMASS licence exists — `ps-license@tue.mpg.de`. Worth an email, but AMASS is
an aggregation of ~20 source datasets with their own terms, so expect friction. Training
our own is cleaner.

## "Can't I just mocap the output?"

Short answer: **re-recording the numbers doesn't launder it; re-performing it with a human
does.** The distinction is real and worth understanding.

- **Recording the generated skeleton** — exporting joints, baking to BVH, re-importing,
  "mocapping" the viewer — is just copying the same data through another format. AMASS's
  restriction is contractual and reaches derived works; a format change isn't a break in
  the chain. This does not clear it.
- **A human performing the motion while wearing a mocap suit** produces a genuinely new
  recording, authored by that performer. Using generated motion as *reference* that a human
  then performs is the same as an animator watching a video and keyframing from it. That's
  much stronger ground — but the value of our stack collapses to "expensive previz", and
  you'd need real mocap kit and a performer.

Your instinct — "leave it behind and use it for mocap only" — is a legitimate fallback.
It's just a worse deal than owning the weights, given what owning them actually costs.

## How we own it: train our own weights

**MDM's code is MIT. Nothing stops us training it on data we're allowed to use.**

The dataset that unlocks this is the **CMU Graphics Lab Motion Capture Database**, whose
terms are about as friendly as it gets:

> "The motion capture data may be copied, modified, or redistributed without permission."

> "You may include this data in commercially-sold products, but you may not resell this
> data directly, even in converted form."

Attribution required: *"The data used in this project was obtained from mocap.cs.cmu.edu.
The database was created with funding from NSF EIA-0196217."* We can't sell the mocap files
as files — irrelevant, we're selling animation and a model.

~2,600 trials from 144 subjects, segmentable into far more training clips.

### What the work actually is

Not a five-minute job — but not a research project either. Four pieces:

1. **Motion conversion** — CMU ships ASF/AMC (and community BVH). We need it in the
   representation MDM consumes. That representation is a *format*, not data: computable
   from any joint stream with HumanML3D's `motion_process.py` (MIT). This is the real
   engineering: retarget CMU's skeleton to our canonical one, resample to 20 fps, compute
   the feature vector, recompute mean/std **from our own data** (which also clears the
   tainted `t2m_*.npy` files).
2. **Text captions** — the hard part, and where CMU is weak: its per-trial descriptions are
   coarse ("subject 8 walks"). HumanML3D's value was crowd-sourced captions. Options:
   expand CMU's metadata into varied captions (I can generate these at scale — it's exactly
   the intent→caption translation the chat layer needs anyway), or hand-caption a subset.
   Caption quality sets the ceiling on prompt understanding.
3. **Training** — MDM's trunk is **17.9M parameters**. Tiny. On the 3090 pod at $0.50/hr,
   a from-scratch run on a CMU-sized corpus is plausibly 1–3 days ≈ **$12–36 of compute**.
   That is the entire cost of owning it outright.
4. **Verification** — regenerate the probe suite from `research/` against the new weights;
   confirm no HumanML3D-derived artefact remains anywhere in the pipeline.

### The side benefit that makes this worth doing anyway

**If we're training our own weights, we choose the skeleton.**

Right now MDM outputs 22 SMPL-topology joints, and getting that onto your anatomically
correct 31-point rig needs a retargeting step that loses fidelity. Train on our own data
and we can define the output topology as **your rig**, directly. No retargeting, no joint
mapping, no lossy conversion — the model emits your skeleton natively.

That turns a licensing chore into a capability upgrade, and it's the strongest argument for
doing it rather than emailing Max Planck.

## Recommendation

1. **Now:** keep using the current stack for R&D and previz. It's research-licensed —
   perfectly legal for what we're doing today. Ship nothing from it.
2. **Next:** build the CMU→our-skeleton conversion pipeline. This is the load-bearing work
   and it's useful regardless of which dataset we end up training on.
3. **Then:** caption, train, verify. Budget a few days and ~$25 of GPU time.
4. **Result:** `solvx-motion-v1` — our weights, our skeleton, our name, commercially clear,
   with CMU attribution in the model card.

Optional parallel track: email `ps-license@tue.mpg.de` about AMASS commercial terms. If
they quote something reasonable it's a shortcut, but don't block on it.

## Sources

- MDM licence: `local/motion-diffusion-model/LICENSE` (MIT) and README §License
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D) — MIT code, cannot redistribute AMASS data
- [AMASS licence](https://amass.is.tue.mpg.de/license.html) — non-commercial; prohibits training for commercial use
- [CMU mocap terms](https://huggingface.co/datasets/gbionics/cmu-fbx) — commercial use permitted with attribution
- KIT Motion-Language: terms not published clearly; would need direct contact with KIT H²T lab
