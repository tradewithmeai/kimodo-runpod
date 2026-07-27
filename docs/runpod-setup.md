# RunPod Setup

Account: backup account (`richwatson420@gmail.com`). API key lives in `.env` (gitignored) —
**not** the `RUNPOD_API_KEY` in the global Windows environment, which belongs to the other
account. Always source `./.env` explicitly; do not rely on the inherited env var.

## Current state

| Resource | Value |
|---|---|
| Pod | `kimodo-dev` / `irm9vm5a0qdb34` |
| GPU | RTX 3090, 24 GB, sm_86 (Ampere) |
| Rate | $0.50/hr on-demand, secure cloud |
| Host | 32 vCPU, 125 GB RAM, 50 GB container disk |
| Image | `runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204` (torch 2.8.0+cu128) |
| Volume | `kimodo-motion-cz` / `tgdoj7p0au`, 200 GB, mounted at `/workspace` |
| Datacenter | EU-CZ-1 |

Check live state, including the SSH IP/port after a restart:

```bash
node scripts/pod-status.mjs
```

## Datacenter choice: EU-CZ-1

Network volumes are **datacenter-locked** — any pod mounting this volume must run in
EU-CZ-1, so the GPU options are limited to EU-CZ-1 stock. Volumes can grow but never
shrink. Billed ~$0.07/GB/mo (~$14/mo at 200 GB) whether or not a pod is attached.

EU-CZ-1 is the only datacenter carrying the RTX 3090. The tradeoff is a thinner ladder
than EU-RO-1 (5 GPU types vs 11) and **no A100**, so the 80 GB+ tier costs $1.99
(RTX PRO 6000) rather than $1.49.

Upgrade ladder available in EU-CZ-1 — snapshot, re-check before relying on it:

| GPU | VRAM | $/hr | Arch | Use |
|---|---|---|---|---|
| RTX 3090 | 24 GB | 0.50 | Ampere | current — pose estimation, MDM/MoMask sampling |
| RTX 4090 | 24 GB | 0.69 | Ada | faster iteration |
| RTX 5090 | 32 GB | 0.99 | Blackwell | more VRAM |
| RTX PRO 6000 WK | 96 GB | 1.89 | Blackwell | large RL runs |
| RTX PRO 6000 | 96 GB | 1.99 | Blackwell | large RL runs |

**Ampere is a feature, not a compromise.** Blackwell cards (5090, PRO series) require
torch 2.7+ and CUDA 12.8+. Several roadmap repos — MDM, PHC, older SMPL tooling — pin
torch 1.x or 2.0, which will not build for sm_120. The 3090 at sm_86 runs them unmodified.

RTX 3090 stock is Low and it is the only DC offering it, so **prefer stopping the pod over
terminating it** — terminating means competing for a scarce card to get another.

## Pricing gotcha

`lowestPrice` in the GraphQL API returns **community cloud** pricing unless you pass
`secureCloud: true`. For cards with `communityCloud: false` (e.g. RTX PRO 4500) this
reports a rate that can never actually be rented — understating true cost by ~2x. Always
cross-check `securePrice`. `scripts/gpu-availability.mjs` does this correctly.

## SSH

A dedicated key was generated for RunPod rather than reusing an existing one, so revoking
it never affects the Hetzner / Hostinger / stratbot hosts.

| Field | Value |
|---|---|
| Private key | `~/.ssh/id_ed25519_runpod` |
| Comment | `runpod-kimodo-2026-07-27` |
| Passphrase | none |
| Registered | account-wide (`updateUserSettings.pubKey`), applies to every pod |

`~/.ssh/config` has a `kimodo` alias, so:

```bash
ssh kimodo
```

`IdentitiesOnly yes` is set on that block and matters here: there are nine keys in
`~/.ssh`, and without it SSH offers them one by one and can hit "too many authentication
failures" before reaching the right one.

**The IP and port change every time the pod is stopped and restarted.** Re-run
`node scripts/pod-status.mjs` after a restart and update the `kimodo` block.

The account key is injected into pods at **creation** time. Registering a new key later
does not retrofit into running pods — recreate the pod, or append to
`~/.ssh/authorized_keys` inside it.

## Jupyter

Port 8888 is exposed through the RunPod proxy:
`https://irm9vm5a0qdb34-8888.proxy.runpod.net`

## Volume layout

Everything reusable lives on the volume so stopping a pod never costs a redownload.
Anything outside `/workspace` is on the 50 GB container disk and is **lost on restart**.

```
/workspace
  models/        # model weights, per-project subdirs
  datasets/      # AMASS, HumanML3D, KIT-ML
  checkpoints/   # training output
  envs/          # conda envs — Isaac Gym and the diffusion stack must stay separate
  repos/         # cloned model repos
```

## API notes

- REST base: `https://rest.runpod.io/v1` — `pods`, `networkvolumes`, `endpoints`.
  There is no `/v1/user`, `/v1/gputypes`, or `/v1/datacenters`.
- Auth header for REST: `Authorization: Bearer $RUNPOD_API_KEY`.
- GPU stock, pricing, **port mappings, and machine details** are GraphQL only:
  `POST https://api.runpod.io/graphql?api_key=KEY`. REST returns `machine: {}` and
  `portMappings: null` even for a running pod — only the pod-create response populates
  `machine`. Use `pod(input:{podId}) { machine { gpuDisplayName location } runtime { ports { ... } } }`.

## Cost control

- Pod billing runs while the pod is RUNNING, whether or not you are connected.
- **Stop** the pod when idle — keeps the container disk and your scarce 3090 allocation.
- **Terminate** only to release the card for good; the volume survives either way.
- Volume storage (~$14/mo) bills continuously and is the only cost when no pod exists.

## Persistence across pod loss

Pods are fully disposable. Everything needed to serve lives on the network volume:
weights, the `mdm` venv, `/workspace/app`, and `/workspace/onstart.sh` (the idempotent
launcher). `scripts/pod-create.mjs` now does the whole recovery in one command:

```
node scripts/pod-create.mjs
```

creates the pod → waits for SSH → rewrites the `Host kimodo` block in `~/.ssh/config` →
runs `onstart.sh` over SSH → polls the public `/health` endpoint until the server reports
ready → prints the new viewer URL. The proxy URL changes with the pod id on every
recreation; the local deployment in `local/` is unaffected by any of this.

## Status

- [x] Network volume created (EU-CZ-1)
- [x] SSH key generated and registered
- [x] Pod created, SSH verified, torch sees the GPU
- [x] Volume directory layout created
- [ ] Per-domain conda envs
- [ ] First model deployed (suggested: pose estimation, per the roadmap)
