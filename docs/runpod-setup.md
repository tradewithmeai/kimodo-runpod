# RunPod Setup

Account: backup account. API key lives in `.env` (gitignored) — **not** the `RUNPOD_API_KEY`
in the global Windows environment, which belongs to the other account. Always source
`./.env` explicitly; do not rely on the inherited env var.

## Network volume

| Field | Value |
|---|---|
| Name | `kimodo-motion` |
| ID | `vq9ltt8vmm` |
| Datacenter | `EU-RO-1` |
| Size | 200 GB |
| Mount | `/workspace` (default when attached to a pod) |

Volumes are **datacenter-locked**: any pod using this volume must run in EU-RO-1, so the
GPU choice is limited to EU-RO-1 stock. Volumes can grow but never shrink. Billed ~$0.07/GB/mo
(~$14/mo at 200 GB) whether or not a pod is attached.

EU-RO-1 was chosen over EUR-IS-1 because it carries more GPU types (13 vs 9 with stock at
time of choosing), including the ones the roadmap eventually needs: A100 PCIe 80GB, RTX PRO
6000 96GB, and B200.

## Upgrade ladder available in EU-RO-1

Snapshot at setup time — stock moves, re-check before relying on it.

| GPU | VRAM | $/hr | Use |
|---|---|---|---|
| RTX PRO 4500 | 32 GB | 0.34 | starting point — pose estimation, MDM/MoMask sampling |
| RTX 4090 | 24 GB | 0.34 | same price, less VRAM — no reason to prefer it |
| RTX 5090 | 32 GB | 0.69 | faster iteration |
| A100 PCIe | 80 GB | 1.19 | fine-tuning, Isaac Gym parallel envs |
| RTX PRO 6000 | 96 GB | 1.69 | large RL runs |
| B200 | 180 GB | 5.98 | headroom |

Re-check stock any time:

```bash
node scripts/gpu-availability.mjs EU-RO-1 EUR-IS-1
```

## SSH

A dedicated key was generated for RunPod rather than reusing an existing one, so revoking
it never affects the Hetzner / Hostinger / stratbot hosts.

| Field | Value |
|---|---|
| Private key | `~/.ssh/id_ed25519_runpod` |
| Comment | `runpod-kimodo-2026-07-27` |
| Passphrase | none |
| Registered | account-wide (`updateUserSettings.pubKey`), applies to every pod |

`~/.ssh/config` has an `ssh.runpod.io` block pinning that key with `IdentitiesOnly yes`.
That flag matters here: there are eight keys in `~/.ssh`, and without it SSH offers them
one by one and can hit "too many authentication failures" before reaching the right one.

Connect via the proxy once a pod exists:

```bash
ssh <podid>-<hash>@ssh.runpod.io
```

The account key is injected into pods at **creation** time. Registering a new key later
does not retrofit into already-running pods — recreate the pod, or add the key to
`~/.ssh/authorized_keys` inside it manually.

## API notes

- REST base: `https://rest.runpod.io/v1` — works for `pods`, `networkvolumes`, `endpoints`.
  There is no `/v1/user`, `/v1/gputypes`, or `/v1/datacenters`.
- GPU stock and pricing are **GraphQL only**: `POST https://api.runpod.io/graphql?api_key=KEY`,
  querying `gpuTypes { lowestPrice(input:{gpuCount:1, dataCenterId:"..."}) }`.
- Auth header for REST: `Authorization: Bearer $RUNPOD_API_KEY`.

## Volume layout convention

Keep everything reusable on the volume so stopping a pod never costs a redownload:

```
/workspace
  models/        # model weights, per-project subdirs
  datasets/      # AMASS, HumanML3D, KIT-ML
  checkpoints/   # training output
  envs/          # conda envs — Isaac Gym and the diffusion stack must stay separate
  repos/         # cloned model repos
```

## Status

- [x] Network volume created
- [ ] Pod created
- [ ] Base environment set up on volume
