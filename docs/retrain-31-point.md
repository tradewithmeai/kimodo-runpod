# Retraining to a 31-point skeleton

Direction set 2026-07-27: the current 600k-step run is a **baseline only**. Production
weights will be retrained for (a) a 31-joint anatomically correct skeleton and (b) a
caption distribution matched to the LLM API front end. Neither the baseline weights nor
its 22-joint representation ship.

This changes the technical plan and — more importantly — improves the licensing position.

---

## 1. The representation changes shape: 263 → 371

MDM does not consume joints directly. It consumes the HumanML3D feature vector, whose
width is a function of joint count. Verified against `motion_process.py` and the
hardcoded values in `utils/model_util.py`:

```
dim = 4 + (J-1)*3 + (J-1)*6 + J*3 + 4
```

| Block | Size at J=22 | Size at J=31 | Contents |
|---|---|---|---|
| Root state | 4 | 4 | root angular velocity, linear vel X/Z, height Y |
| Local joint positions (`ric`) | 63 | 90 | `(J-1) × 3`, relative to root |
| Joint rotations (`rot`) | 126 | 180 | `(J-1) × 6`, 6D continuous rotation |
| Local velocities | 66 | 93 | `J × 3` |
| Foot contact | 4 | 4 | binary contact flags |
| **Total** | **263** | **371** | |

The formula reproduces both `njoints = 263` (HumanML3D, J=22) and `njoints = 251`
(KIT, J=21), so it is confirmed rather than inferred.

### Consequences

- **`njoints` in `get_model_args` becomes 371.** This is an input/output dimension, not
  a config knob — it resizes the first embedding layer and the final projection.
- **The 8-layer, 512-dim transformer body is unchanged and transferable.** Only the two
  boundary layers change shape. Warm-starting the body from the baseline is standard
  transfer learning and should converge much faster than 600k steps from scratch —
  *but see §3, because warm-starting inherits the baseline's data provenance.*
- **`t2m_mean.npy` / `t2m_std.npy` must be recomputed** for the new representation. The
  shipped normalisation vectors are 263-wide and specific to HumanML3D's joint set.
- **`recover_from_ric` works unmodified** — it already takes `joints_num` as an argument
  and slices accordingly. Our inference server passes 22; it becomes 31.
- **The kinematic chains change.** `KINEMATIC_CHAINS` in `pod/motion_server.py` is
  hardcoded to SMPL's 22-joint topology; the viewer draws bones from it. Needs replacing
  with the real skeleton's hierarchy.
- **Foot-contact indices change.** `lambda_fc` is 0.0 so this doesn't affect the loss
  today, but the 4 contact flags reference specific joint indices that will move.
- **The evaluators do not transfer.** The pretrained T2M evaluators (FID, R-precision,
  Diversity) were trained on 22-joint HumanML3D features. They cannot score 371-dim
  motion. **This is the biggest hidden cost of the switch** — without them there is no
  quantitative quality metric, only visual inspection. Options: retrain the evaluator on
  the new representation, or evaluate by retargeting generated 31-joint motion back to
  22 joints and scoring that (imperfect but comparable across runs).

---

## 2. Retraining for the LLM front end

Since Claude sits upstream rewriting user language into model-facing captions, the model
should be trained on **the distribution Claude actually emits**, not on HumanML3D's
original human-written captions.

The technique: re-caption the training set by passing each motion's existing caption(s)
through the same prompt pipeline the product uses, and train on those. The model then
sees at training time exactly the phrasing style it will see in production, which is
precisely the mismatch the earlier OOD probes measured.

Two cautions:

- **Re-captioning changes the text, not the motion.** The motion data's licence is
  unaffected by rewriting its captions (§3).
- **Freeze the prompt pipeline before generating the training captions.** If the
  production prompt changes later, the training distribution silently drifts away from
  the serving distribution. Version the prompt alongside the weights.

---

## 3. Licensing: the position genuinely improves, with one trap

Recorded in `docs/saas-hosting-decision.md` §4: HumanML3D derives from AMASS, which is
research-licensed, so **baseline weights are not clearly sellable**. The retrain changes
this — but only if the *motion data* changes, not merely the skeleton.

| Retrain approach | Motion data provenance | Commercially clean? |
|---|---|---|
| Retarget HumanML3D motions to 31 joints, train on those | Still AMASS-derived | **No** — retargeting is a transformation of the same data |
| Warm-start from baseline weights, train on new data | Weights derive from AMASS training | **Contested** — arguably a derivative work |
| Train from scratch on your own captured/licensed motion | Yours | **Yes** |
| Train from scratch on video-derived motion from footage you own or licence | Yours | **Yes** |

**The trap: changing the skeleton does not launder the data.** A 31-joint retarget of an
AMASS motion is still that AMASS motion, expressed differently. Only replacing the source
motion changes the provenance.

**This is where the project roadmap's domain 3 (video → pose estimation) stops being a
"nice to have" and becomes the data engine.** SMPLer-X / WHAM on footage you own or
licence produces motion with clean provenance, in your own skeleton, at whatever scale
you can source video. That is the path to weights with no third-party constraint —
and it is already on the roadmap.

**Practical sequencing:** the baseline validates the pipeline (research use, fine). The
31-point retrain can proceed on retargeted HumanML3D for *development and evaluation*
while the clean-data pipeline is built. Only the weights that ship to paying users need
clean provenance. Keep the provenance record per checkpoint: source data, licence,
starting weights, exactly what changed.

---

## 4. What the baseline is still worth

It is not wasted, and its value does not depend on the weights shipping:

1. **The pipeline is proven** — data prep, training loop, checkpointing, eval, rendering,
   and the five environment fixes are all validated and reproducible.
2. **A known-good reference point.** When the 371-dim model trains, the baseline's loss
   curve and metrics tell you whether the new run is behaving plausibly. Without it,
   a bad 31-point run and a good one look the same.
3. **Throughput numbers transfer approximately.** 0.173 s/step at 263-dim; 371-dim is
   ~1.4× the feature width, so expect a modest increase, not a different order.
4. **It de-risks the expensive run.** Every failure mode found now costs $0.50/hr instead
   of being found during a run whose correct answer nobody knows.
