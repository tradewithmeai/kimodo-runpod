# Vocabulary probe — what MDM actually knows

Agent: `sens-vocabulary-probe`. All numbers below were measured on the pod against
`humanml_enc_512_50steps/model000750000.pt` via `/workspace/app/motion_server.py`.

Scripts (on pod): `/workspace/probe2.py` (60-prompt sweep, 3 seeds each),
`/workspace/probe3.py` (calibration + phrasing experiments).
Raw results: `/workspace/probe_vocab2.json`.

- 60 prompts x 3 seeds (7 / 1234 / 99), 6.0 s (120 frames @ 20 fps), guidance 2.5.
- Plus a 12-prompt run-phrasing sweep, a 4-point guidance sweep, and an 11-template
  phrasing sweep.
- Total wall time for the main sweep: 74.5 s for 180 generations (~0.41 s each).

---

## 0. BLOCKER FOUND FIRST — the root-translation channels are being decoded ~20-30x too small

This has to be stated before any metric, because it invalidates the obvious
"root displacement" measurement and it is a live bug in the current server.

`motion_server.py` de-normalises the 263-d HML vector with
`dataset/t2m_mean.npy` / `dataset/t2m_std.npy`. Measured values of the first four
channels of that std file (ch0 = root yaw velocity, ch1 = lateral velocity,
ch2 = forward velocity, ch3 = root height):

```
mean[:5] = [-1.0e-05, -2.0e-05, 8.68e-03, 0.91752, 0.05658]
std[:5]  = [ 0.00051,  0.00068, 0.00068, 0.00612, 0.12256]
```

The network's raw (normalised) output for "a person walks forward" is healthy —
ch2 has mean +1.42, std 1.29 — but multiplying by std 0.00068 collapses it:

```
DENORM ch2 (fwd vel): mean +0.0096  std 0.0009  min +0.008  max +0.011
DENORM ch3 (root y) : mean +0.9177  std 0.0005
```

Consequence, measured directly on the recovered joints:

| prompt | root XZ displacement over 6 s | root height range |
|---|---|---|
| a person stands still | 0.99 m | 0.000 m |
| a person walks forward | 1.15 m | 0.002 m |
| a person runs forward | 1.09 m | 0.005 m |
| (empty prompt) | 0.99 m | 0.000 m |

**Every clip drifts forward ~1 m at a near-constant 0.19 m/s and the pelvis height
never changes**, regardless of prompt. Root displacement and root height range are
therefore *not usable metrics* against the current pipeline, and any viewer/exporter
built on it will show every character sliding forward at the same speed while
sitting, lying, or standing still.

Calibration (`probe3.py` Part A): regressing measured stance-foot ground slip
(which comes from the *ric* joint channels, which are fine) against ch2 over 16
walking clips gives `speed_mps = 0.263*ch2 + 0.379`, R^2 = 0.78, implying a true
`Std[2] ~= 0.013`. A cleaner two-point estimate from "walks slowly" (0.42 m/s,
ch2 +0.56) vs "walks forward" (0.81 m/s, ch2 +1.59) gives `Std[2] ~= 0.019`.
Either way the file value of 0.00068 is **~20-30x too small**. Channels 4+ (the
root-relative joint positions) are fine — hip height 0.90 m, running foot swing
0.24 m, all physically plausible — so the corruption is confined to ch0-ch3.

Fix direction (not attempted here, out of scope): the HumanML3D *training*
normalisation (`Mean.npy`/`Std.npy` from the HumanML3D dataset) is what
`inv_transform` is supposed to use; `t2m_mean/std` are the evaluator's stats and
have a different scale on the root channels.

### Metrics actually used below (corruption-immune)

- **ch2 / fwd** — raw normalised forward-velocity channel, clip mean. Prompt-driven,
  unit-ish scale, dataset mean = 0. This is the honest locomotion signal.
- **yaw / yaw_sd** — raw normalised ch0, mean and std. Turning.
- **hgt** — raw normalised ch3 mean. Posture height (crouch / sit / lie).
- **artic** — mean root-relative joint speed, m/s. Articulation energy. Valid.
- **head_y / head_rng** — head height above pelvis, m. Posture and posture change.
- **foot_lift** — foot vertical range in root frame, m.
- **div** — mean pairwise RMS pose difference across the 3 seeds. Mode-collapse check.

**Null baselines (essential reference points):** the empty prompt `""` gives
artic **0.070**, and `"a person"` gives artic **0.045**. Anything at or below
~0.10 artic with no posture change is indistinguishable from generating nothing.

---

## 1. Full sweep results

`spd` = stance-foot ground-speed estimator, m/s (reliable only when a foot stays
planted — it under-reads for running and over-reads for in-place bouncing; treat
`fwd` as the authority).

```
cat   | prompt                                         | spd  dist | artic | headY hrng | flift | pstd | yaw±sd       | fwd±sd    | hgt   | div
loco  | a person walks forward                         | 0.81  4.89 | 0.325 |  0.68 0.10 | 0.107 | 0.038 | -0.035/0.040 | +1.59/1.47 | +0.07 | 0.091
loco  | a person walks                                 | 0.79  4.73 | 0.323 |  0.69 0.14 | 0.124 | 0.041 | -0.030/0.040 | +1.56/1.50 | +0.12 | 0.096
loco  | walk                                           | 0.47  2.83 | 0.359 |  0.67 0.22 | 0.111 | 0.044 | -0.001/0.039 | +1.44/1.84 | +0.02 | 0.093
loco  | a person runs forward                          | 0.09  0.57 | 0.280 |  0.65 0.14 | 0.173 | 0.034 | -0.006/0.187 | +0.87/1.87 | -0.12 | 0.061
loco  | a person is running fast                       | 0.15  0.91 | 0.278 |  0.66 0.16 | 0.134 | 0.035 | -0.014/0.190 | +1.03/2.13 | -0.03 | 0.058
loco  | a person jogs in place                         | 0.87  5.23 | 0.749 |  0.71 0.14 | 0.295 | 0.040 | -0.010/0.040 | -0.65/0.32 | +0.38 | 0.088
loco  | a person crawls on the floor                   | 0.32  1.90 | 0.396 | -0.29 0.39 | 0.616 | 0.082 | -0.181/0.649 | +0.24/0.52 | -3.13 | 0.171
loco  | a person jumps                                 | 0.16  0.98 | 0.603 |  0.70 0.48 | 0.216 | 0.048 | -0.003/0.028 | -0.51/0.44 | +0.23 | 0.093
loco  | a person jumps forward with both feet          | 0.35  2.08 | 0.550 |  0.64 0.68 | 0.299 | 0.064 | -0.016/0.032 | +0.58/1.43 | +0.18 | 0.116
loco  | a person hops on one leg                       | 0.05  0.32 | 0.232 |  0.68 0.15 | 0.704 | 0.025 | -0.044/0.041 | -0.41/0.22 | +0.34 | 0.070
loco  | a person climbs stairs                         | 0.18  1.09 | 0.258 |  1.02 0.83 | 0.944 | 0.131 | -0.040/0.289 | -0.09/0.52 | +2.53 | 0.083
loco  | a person walks up the stairs                   | 0.21  1.23 | 0.260 |  1.01 0.79 | 0.905 | 0.126 | -0.050/0.413 | -0.11/0.50 | +2.46 | 0.078
loco  | a person swims                                 | 0.08  0.49 | 0.302 | -0.25 0.31 | 0.329 | 0.056 | +0.075/0.106 | -0.48/0.16 | -1.80 | 0.132
loco  | a person tiptoes forward                       | 0.07  0.43 | 0.168 |  0.51 0.82 | 0.065 | 0.067 | -0.039/0.571 | -0.46/0.33 | +0.00 | 0.093
inter | a person sits down on a chair                  | 0.05  0.30 | 0.156 |  0.14 0.76 | 0.041 | 0.083 | +0.031/0.114 | -0.67/0.28 | -2.58 | 0.172
inter | a person stands up                             | 0.16  0.99 | 0.272 |  0.28 0.88 | 0.258 | 0.135 | -0.005/0.551 | -0.57/0.36 | -1.22 | 0.165
inter | a person picks up an object from the ground    | 0.05  0.29 | 0.182 |  0.09 0.87 | 0.042 | 0.083 | -0.077/0.288 | -0.52/0.26 | -0.97 | 0.109
inter | a person bends down and picks something up     | 0.07  0.40 | 0.195 |  0.27 0.85 | 0.046 | 0.092 | -0.040/0.309 | -0.54/0.43 | -0.42 | 0.080
inter | a person throws a ball                         | 0.09  0.54 | 0.188 |  0.66 0.09 | 0.035 | 0.049 | -0.154/1.556 | -0.54/0.20 | +0.06 | 0.100
inter | a person pushes a heavy object                 | 0.02  0.15 | 0.062 | -0.01 0.40 | 0.021 | 0.034 | -0.015/0.289 | -0.52/0.10 | -0.70 | 0.137
inter | a person pulls something with both hands       | 0.04  0.25 | 0.070 |  0.66 0.07 | 0.022 | 0.033 | -0.125/0.292 | -0.50/0.06 | +0.23 | 0.058
inter | a person kicks with the right leg              | 0.16  0.96 | 0.378 |  0.65 0.16 | 1.293 | 0.076 | -0.055/0.441 | -0.49/0.60 | +0.11 | 0.158
inter | a person punches with both fists               | 0.13  0.77 | 0.378 |  0.61 0.11 | 0.025 | 0.044 | +0.041/0.222 | -0.54/0.19 | -0.12 | 0.109
inter | a person opens a door                          | 0.02  0.09 | 0.059 |  0.68 0.02 | 0.006 | 0.024 | -0.035/0.040 | -0.51/0.03 | +0.11 | 0.064
inter | a person drinks from a cup                     | 0.02  0.14 | 0.071 |  0.69 0.02 | 0.014 | 0.027 | -0.027/0.166 | -0.51/0.06 | +0.11 | 0.073
inter | a person lies down on the floor                | 0.08  0.50 | 0.051 | -0.81 0.10 | 0.026 | 0.032 | -0.007/0.075 | -0.56/0.05 | -5.02 | 0.104
expr  | a person waves with their right hand           | 0.03  0.18 | 0.083 |  0.62 0.01 | 0.014 | 0.014 | -0.002/0.164 | -0.49/0.03 | -0.25 | 0.089
expr  | a person waves hello                           | 0.03  0.17 | 0.069 |  0.65 0.01 | 0.015 | 0.006 | +0.013/0.040 | -0.52/0.03 | -0.13 | 0.059
expr  | a person dances                                | 0.58  3.50 | 0.582 |  0.64 0.27 | 0.206 | 0.085 | -0.689/1.285 | -0.06/0.98 | -0.04 | 0.164
expr  | a person is dancing happily                    | 0.67  3.99 | 0.712 |  0.65 0.31 | 0.291 | 0.088 | -0.552/1.092 | -0.03/0.93 | +0.00 | 0.168
expr  | a person bows                                  | 0.02  0.15 | 0.116 |  0.44 0.71 | 0.025 | 0.055 | -0.037/0.365 | -0.53/0.20 | +0.04 | 0.022
expr  | a person claps their hands                     | 0.03  0.16 | 0.082 |  0.67 0.01 | 0.015 | 0.013 | -0.025/0.025 | -0.51/0.03 | +0.12 | 0.054
expr  | a person stretches their arms                  | 0.03  0.18 | 0.097 |  0.68 0.01 | 0.032 | 0.030 | -0.041/0.129 | -0.51/0.06 | +0.15 | 0.135
expr  | a person salutes                               | 0.02  0.10 | 0.065 |  0.67 0.01 | 0.007 | 0.016 | +0.018/0.294 | -0.52/0.03 | +0.01 | 0.099
expr  | a person shakes their head                     | 0.03  0.20 | 0.088 |  0.66 0.12 | 0.016 | 0.028 | -0.061/0.170 | -0.54/0.07 | +0.09 | 0.122
expr  | a person crosses their arms                    | 0.01  0.09 | 0.049 |  0.66 0.02 | 0.011 | 0.018 | +0.015/0.032 | -0.50/0.04 | +0.10 | 0.046
expr  | a person stands still                          | 0.01  0.03 | 0.022 |  0.39 0.01 | 0.008 | 0.013 | +0.004/0.023 | -0.53/0.01 | -1.73 | 0.046
expr  | a person does a cartwheel                      | 1.39  8.36 | 1.495 | -0.14 1.10 | 2.161 | 0.248 | -0.155/2.086 | +0.32/2.14 | +0.13 | 0.422
mod   | a person walks slowly                          | 0.42  2.51 | 0.194 |  0.70 0.06 | 0.135 | 0.029 | -0.023/0.047 | +0.56/0.89 | +0.23 | 0.080
mod   | a person walks quickly                         | 0.74  4.46 | 0.447 |  0.64 0.21 | 0.128 | 0.048 | -0.002/0.044 | +1.94/2.37 | -0.11 | 0.085
mod   | a person walks backwards                       | 0.53  3.17 | 0.260 |  0.71 0.08 | 0.202 | 0.034 | -0.060/0.047 | -1.91/1.10 | +0.47 | 0.064
mod   | a person walks in a circle                     | 0.94  5.63 | 0.385 |  0.68 0.07 | 0.120 | 0.043 | +1.737/0.727 | +1.95/0.61 | +0.16 | 0.104
mod   | a person walks while turning to the left       | 0.44  2.61 | 0.219 |  0.67 0.07 | 0.095 | 0.035 | -0.502/0.412 | +0.70/1.09 | +0.09 | 0.068
mod   | a person turns around 180 degrees              | 0.10  0.62 | 0.095 |  0.69 0.03 | 0.049 | 0.023 | -0.805/1.252 | -0.36/0.40 | +0.22 | 0.034
mod   | a person walks forward then stops and turns... | 0.33  1.97 | 0.209 |  0.69 0.06 | 0.103 | 0.033 | +0.506/0.349 | +0.73/1.25 | +0.15 | 0.065
mod   | a person runs in a circle                      | 1.90 11.40 | 0.802 |  0.62 0.15 | 0.210 | 0.054 | +2.096/0.515 | +4.75/1.02 | -0.09 | 0.104
mod   | a person walks with their hands raised         | 0.51  3.04 | 0.268 |  0.70 0.11 | 0.248 | 0.071 | +0.019/0.059 | +0.72/0.79 | +0.41 | 0.104
mod   | a person walks sideways to the right           | 0.38  2.31 | 0.229 |  0.67 0.09 | 0.136 | 0.034 | -0.006/0.092 | -0.71/0.31 | +0.19 | 0.067
mod   | a person walks like a zombie                   | 0.83  5.00 | 0.360 |  0.66 0.13 | 0.169 | 0.060 | -0.609/0.998 | +1.53/0.83 | +0.06 | 0.139
mod   | a person walks casually forward for a few s... | 0.66  3.96 | 0.267 |  0.68 0.10 | 0.126 | 0.036 | -0.026/0.050 | +1.15/1.42 | +0.09 | 0.075
mod   | a person walks forward and then sits down      | 0.13  0.80 | 0.253 |  0.34 0.76 | 0.120 | 0.097 | -0.300/0.950 | -0.02/1.30 | -1.25 | 0.074
mod   | a person raises their right arm slowly         | 0.01  0.08 | 0.047 |  0.68 0.02 | 0.009 | 0.017 | -0.048/0.041 | -0.51/0.03 | +0.14 | 0.056
ctrl  | (empty string)                                 | 0.02  0.13 | 0.070 |  0.69 0.03 | 0.021 | 0.027 | -0.016/0.027 | -0.53/0.09 | +0.14 | 0.086
ctrl  | asdf qwerty zxcv                               | 0.06  0.36 | 0.119 |  0.67 0.08 | 0.061 | 0.029 | +0.006/0.126 | -0.71/0.37 | +0.13 | 0.071
ctrl  | a person operates a forklift                   | 0.03  0.15 | 0.031 |  0.26 0.03 | 0.023 | 0.011 | -0.009/0.029 | -0.41/0.03 | -2.33 | 0.041
ctrl  | a person plays the violin                      | 0.01  0.08 | 0.035 |  0.67 0.01 | 0.007 | 0.024 | -0.033/0.080 | -0.50/0.02 | +0.07 | 0.020
ctrl  | a person rides a bicycle                       | 0.19  1.17 | 0.562 |  0.66 0.42 | 0.254 | 0.063 | -0.055/0.111 | -0.54/0.46 | +0.28 | 0.158
ctrl  | a person does a backflip                       | 0.41  2.47 | 0.730 |  0.04 1.34 | 2.149 | 0.216 | -0.132/0.609 | -0.50/1.38 | -0.66 | 0.320
ctrl  | the quick brown fox jumps over the lazy dog    | 0.86  5.15 | 0.606 |  0.23 1.09 | 0.221 | 0.134 | +0.024/0.120 | -0.31/2.13 | -0.88 | 0.200
ctrl  | a person                                       | 0.01  0.09 | 0.045 |  0.68 0.01 | 0.010 | 0.021 | -0.040/0.032 | -0.51/0.03 | +0.11 | 0.083
```

---

## 2. What works — the target distribution for a rewriting layer

### Tier A — strong, distinctive, semantically correct

These produce motion whose measured signature matches the words. These are what
a rewriting layer should aim at.

| prompt family | evidence |
|---|---|
| `a person walks forward` / `a person walks` | fwd +1.56..+1.59, artic 0.32, foot_lift 0.11, head steady 0.68. Clean gait. |
| `a person walks quickly` | fwd +1.94 vs slow +0.56 — the speed adverb is a real 3.5x lever |
| `a person walks slowly` | fwd +0.56, artic 0.194 — correctly damped |
| `a person walks backwards` | fwd **-1.91** — sign flip, unambiguous |
| `a person walks in a circle` | yaw +1.74 (all other prompts sit near 0), fwd +1.95 |
| `a person runs in a circle` | **best locomotion result in the whole sweep**: fwd +4.75, yaw +2.10, artic 0.80 |
| `a person sprints` | fwd +3.80, ~1.6 m/s measured — see section 3 |
| `a person runs across the room` | fwd +2.68, ~1.3 m/s |
| `a person jogs in place` | fwd **-0.65** (no travel) + artic 0.749 (high) — "in place" is understood |
| `a person hops on one leg` | foot_lift 0.704 with fwd -0.41 — one-legged, stationary. Correct. |
| `a person kicks with the right leg` | foot_lift **1.293**, the largest single-limb excursion |
| `a person climbs stairs` / `walks up the stairs` | head_y 1.02 (vs 0.68 standing), hgt +2.5, foot_lift 0.94 — rising, correct |
| `a person crawls on the floor` | head_y **-0.29** (head below pelvis), hgt -3.13, foot_lift 0.62 |
| `a person swims` | head_y -0.25, hgt -1.80 — prone. Correct. |
| `a person lies down on the floor` | hgt **-5.02**, head_y -0.81 — most extreme posture in the set |
| `a person sits down on a chair` | hgt -2.58, head_y 0.14, head_rng 0.76 |
| `a person stands up` | head_rng **0.88** — largest rise; correct inverse of sitting |
| `a person picks up an object from the ground` | head_rng 0.87, head_y 0.09 |
| `a person bends down and picks something up` | head_rng 0.85 |
| `a person jumps` / `jumps forward with both feet` | head_rng 0.48 / 0.68, artic 0.55-0.60 |
| `a person dances` / `is dancing happily` | artic 0.58-0.71, yaw_sd 1.1-1.3 (spinning), div 0.16-0.17 (varied) |
| `a person bows` | head_rng 0.71 with div 0.022 — highly consistent across seeds |
| `a person punches with both fists` | artic 0.378 |
| `a person walks like a zombie` | artic 0.36, yaw_sd 1.0 — stylistic modifier lands |
| `a person walks with their hands raised` | foot_lift 0.25, posestd 0.071 — upper-body modifier composes with gait |

### Tier B — works but weakly / needs help

- `a person walks sideways to the right` — fwd -0.71 (not lateral). Lateral channel
  did not light up; it reads as a shuffle, not a strafe.
- `a person turns around 180 degrees` — yaw -0.805, yaw_sd 1.25, but artic only
  0.095 and no travel. It turns, but the body is nearly frozen.
- `a person walks while turning to the left` — yaw -0.50, half the magnitude of
  "in a circle". "in a circle" is the stronger turning phrase by 3.5x.
- `a person throws a ball` — artic 0.188 but yaw_sd **1.556**, the noisiest turn
  signal in the set. Erratic torso spin rather than a throw.
- `a person tiptoes forward` — head_rng 0.82 but foot_lift 0.065 and fwd -0.46.
  Bobbing in place, not tiptoeing forward.
- `a person rides a bicycle` — artic 0.562, out-of-distribution but not degenerate.
- `a person walks forward and then sits down` — the sit dominates (hgt -1.25,
  head_rng 0.76) and the walk is lost (fwd -0.02). Sequential composition is weak.

### Tier C — degenerate: indistinguishable from generating nothing

Reference: empty prompt artic **0.070**, `"a person"` artic **0.045**.

| prompt | artic | head_rng | verdict |
|---|---|---|---|
| a person stands still | 0.022 | 0.01 | below null; also crouched (head_y 0.39, hgt -1.73) — wrong pose |
| a person operates a forklift | 0.031 | 0.03 | frozen; does read as seated (hgt -2.33) |
| a person plays the violin | 0.035 | 0.01 | frozen idle |
| a person raises their right arm slowly | 0.047 | 0.02 | frozen idle |
| a person crosses their arms | 0.049 | 0.02 | frozen idle |
| a person lies down (motion, not pose) | 0.051 | 0.10 | pose is right, but there is no *transition* |
| a person opens a door | 0.059 | 0.02 | frozen idle |
| a person pushes a heavy object | 0.062 | 0.40 | leans (head_y -0.01) but barely moves |
| a person salutes | 0.065 | 0.01 | frozen idle |
| a person waves hello | 0.069 | 0.01 | == null baseline. Does not wave. |
| a person pulls something with both hands | 0.070 | 0.07 | == null baseline |
| a person drinks from a cup | 0.071 | 0.02 | == null baseline |
| a person claps their hands | 0.082 | 0.01 | barely above null |
| a person waves with their right hand | 0.083 | 0.01 | barely above null |
| a person shakes their head | 0.088 | 0.12 | barely above null |
| a person stretches their arms | 0.097 | 0.01 | barely above null |

**The whole expressive/upper-body category is effectively dead.** Wave, clap,
salute, cross arms, stretch, shake head, drink, open door — all land within noise
of the empty prompt. The only expressive prompts that survive are the ones that
are really whole-body locomotion: `dances`, `bows`, `cartwheel`.

### Tier D — wild / unusable

- `a person does a cartwheel` — artic **1.495**, foot_lift 2.16, yaw_sd 2.09,
  seed-divergence **0.422** (4x the typical 0.09). Different motion every seed.
- `a person does a backflip` — artic 0.730, foot_lift 2.15, div 0.320. Same story.

### Nonsense handling — important and counter-intuitive

- `asdf qwerty zxcv` -> artic 0.119, near-null. Safe collapse to idle. Good.
- `the quick brown fox jumps over the lazy dog` -> artic **0.606**, head_rng 1.09,
  hgt -0.88, div 0.200. CLIP latches onto "quick"/"jumps" and produces large,
  incoherent whole-body motion.

**Unrecognised text does NOT reliably fall back to idle.** Ordinary English
sentences containing motion words produce confident garbage. A rewriting layer
cannot rely on the model to fail safe; it must gate the vocabulary itself.

---

## 3. "run" is broken; "sprint" is not

`probe3.py` Part B, 3 seeds each. `est_speed` from the stance-foot estimator.

```
a person runs forward                        ch2  +0.87  est_speed 0.09 m/s
a person runs                                ch2  +1.05  est_speed 0.14 m/s
run                                          ch2  +0.99  est_speed 0.15 m/s
a person is running                          ch2  +1.10  est_speed 0.24 m/s
a person runs quickly forward                ch2  +0.97  est_speed 0.10 m/s
a person jogs forward                        ch2  +1.00  est_speed 0.14 m/s
a person runs forward in a straight line     ch2  +0.94  est_speed 0.12 m/s
a person walks forward                       ch2  +1.59  est_speed 0.81 m/s   <-- walk beats run
a person walks quickly forward               ch2  +1.60  est_speed 0.43 m/s
a person sprints                             ch2  +3.80  est_speed 1.56 m/s   <-- works
a person runs across the room                ch2  +2.68  est_speed 1.33 m/s   <-- works
a person runs forward then slows to a walk   ch2  +3.88  est_speed 1.47 m/s   <-- works
```

Every plain phrasing of "run" produces **less** forward velocity than "walk".
Adding "quickly", "forward", "in a straight line" does not help — all cluster at
ch2 ~ +0.9 to +1.1. What unlocks real running speed is a *destination or intensity*
word: `sprints` (+3.80), `across the room` (+2.68), or a *transition* clause
(`then slows to a walk`, +3.88). Also note `runs in a circle` at +4.75 from the
main sweep.

**Rewrite rule: never emit a bare "runs" prompt. Map run -> `a person sprints`,
`a person runs across the room`, or `a person runs in a circle`.**

---

## 4. Guidance is a first-class lever for locomotion, not for gesture

`probe3.py` Part C, seed 7.

```
a person walks forward          g=1.0 ch2 +1.03 artic 0.246
                                g=2.5 ch2 +1.42 artic 0.299
                                g=5.0 ch2 +2.46 artic 0.408
                                g=10.  ch2 +2.83 artic 0.452

a person runs forward           g=1.0 ch2 +0.74 artic 0.312
                                g=2.5 ch2 +0.63 artic 0.283
                                g=5.0 ch2 +0.52 artic 0.235   <-- gets WORSE
                                g=10.  ch2 +1.30 artic 0.312

a person waves with right hand  g=1.0 ch2 -0.50 artic 0.045
                                g=2.5 ch2 -0.51 artic 0.120
                                g=5.0 ch2 -0.48 artic 0.139
                                g=10.  ch2 -0.48 artic 0.131
```

- Walking: guidance 1 -> 10 nearly triples forward velocity (+1.03 -> +2.83) and
  doubles articulation. Raising guidance to ~5 is a cheap fix for "the walk looks
  sluggish".
- Running: guidance does not rescue it, and 5.0 is worse than 1.0. Confirms the
  problem is the text embedding, not the sampler.
- Waving: guidance saturates at ~2.5 and never escapes the null band
  (artic 0.13 vs null 0.070). Gesture cannot be fixed by turning guidance up.

---

## 5. Phrasing template effects (same action: wave)

`probe3.py` Part D, 3 seeds. Every one of these is in or near the null band —
the point is the *relative* ordering, which generalises to other actions.

```
bare verb            wave                                        artic 0.104  posestd 0.011
V-ing                waving                                      artic 0.106  posestd 0.012
subject+verb         a person waves                              artic 0.081  posestd 0.007
subject+verb+detail  a person waves with their right hand        artic 0.083  posestd 0.014
the man              the man waves his right hand                artic 0.131  posestd 0.019   <-- best
a man                a man is waving his right hand              artic 0.124  posestd 0.018
caption style        a person raises their right hand and
                     waves it side to side                       artic 0.041  posestd 0.009   <-- worst
2nd person           you wave your right hand                    artic 0.089  posestd 0.019
imperative           wave your right hand                        artic 0.042  posestd 0.012
```

Findings:

1. **`the man` / `a man` beat `a person`** by ~1.5x articulation. HumanML3D
   captions are heavily "a man ..." / "the man ...".
2. **Imperative and 2nd-person are bad.** `wave your right hand` (0.042) is 2.5x
   weaker than `the man waves his right hand` (0.131) and below the null baseline.
   A rewriting layer must convert user imperatives into third-person declaratives.
3. **Long descriptive "caption style" is bad** (0.041) — the worst of the set.
   Adding descriptive sub-clauses dilutes the CLIP embedding rather than enriching
   it. Elsewhere the same effect: `a person runs forward in a straight line`
   (+0.94) is no better than `a person runs` (+1.05).
4. **Bare verbs work.** `walk` gives fwd +1.44 vs `a person walks forward` +1.59;
   `dance` gives artic 0.592 vs `a person dances` 0.582. Sentence framing is
   optional — semantics dominate syntax. But `sit` (0.039) and `wave` (0.104)
   inherit the same category weakness as their sentence forms.

---

## 6. Concrete target distribution for a prompt-rewriting layer

### Emit (verified strong)

```
Locomotion:
  a person walks forward
  a person walks slowly
  a person walks quickly
  a person walks backwards
  a person walks in a circle
  a person walks like a zombie
  a person walks with their hands raised
  a person sprints
  a person runs across the room
  a person runs in a circle
  a person jogs in place
  a person hops on one leg
  a person jumps
  a person jumps forward with both feet
  a person climbs stairs
  a person crawls on the floor
  a person swims

Posture transitions (low energy but correct and distinctive):
  a person sits down on a chair
  a person stands up
  a person lies down on the floor
  a person picks up an object from the ground
  a person bends down and picks something up
  a person bows

Whole-body action:
  a person dances
  a person is dancing happily
  a person kicks with the right leg
  a person punches with both fists
```

### Never emit

- Bare `run` in any plain form — rewrite to `sprints` / `runs across the room` /
  `runs in a circle`.
- Any upper-body-only gesture: wave, clap, salute, cross arms, stretch, shake head,
  drink, open door, play an instrument, raise an arm. These are at or below the
  null-prompt baseline. If the user asks for one, the honest options are (a) return
  a canned/authored clip, or (b) rewrite to the nearest whole-body action
  (e.g. wave -> `a person raises both arms and jumps`, untested), never (c) pass
  it through.
- `a person stands still` — produces a crouched, wrong-height idle. Use an authored
  T-pose/idle instead.
- `a person does a cartwheel` / `a backflip` — 3-4x normal seed divergence, unusable.
- Sequential clauses (`walk forward and then sit down`) — the second clause wins
  and the first is lost.
- Long descriptive sub-clauses — measurably weaker than the short form.
- Imperative or second-person phrasing.

### Phrasing template

```
"<a|the> <man|person> <verb-3rd-person> <short modifier>"
```

- Third person declarative, present tense.
- Prefer `the man` / `a man` over `a person` (~1.5x articulation).
- One modifier maximum. Modifiers that measurably work:
  `slowly`, `quickly`, `backwards`, `in a circle`, `in place`, `like a zombie`,
  `across the room`, `with their hands raised`, `on one leg`, `with both feet`.
- Modifiers that do NOT work: `sideways to the right`, `in a straight line`,
  `while turning to the left` (use `in a circle` instead), `180 degrees`.

### Sampler defaults implied by the measurements

- guidance **5.0** for locomotion prompts (nearly 2x forward velocity vs 2.5).
- guidance **2.5** for posture transitions and everything else (5.0 gives no gain
  and hurts running).

---

## 7. Unknowns

- **Visual plausibility** — everything here is kinematic statistics. No clip was
  rendered or eyeballed. "Distinctive and plausible-looking" is inferred from
  signature-vs-semantics agreement, not from watching.
- **Foot sliding / ground contact quality** — not measured.
- **The exact correct `Std[0..3]` values** — measured only as "20-30x larger than
  the file", via an indirect foot-slip estimator (R^2 0.78). The right fix is to
  source the HumanML3D training `Mean.npy`/`Std.npy`, not to scale by a fudge factor.
- **Whether upper-body gestures are dead in the model or dead in this checkpoint** —
  not separable from these measurements. Guidance sweeps rule out the sampler.
- **Clip lengths other than 6 s** — `MAX_FRAMES` is 196 (9.8 s); not probed.
