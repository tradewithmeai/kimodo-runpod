# Kimodo

A RunPod-hosted environment for simulating human movement. The working piece today is
text-to-motion: you type a description, a diffusion model samples a 22-joint skeletal
animation on the GPU, and it plays back in a browser. Three further domains — physics-based
humanoid control, video pose estimation, and human-like input simulation — are scoped in
[`docs/human-motion-roadmap.md`](docs/human-motion-roadmap.md) but not yet built.

Infrastructure details, including the API quirks worth knowing before you start poking at
RunPod's API, are in [`docs/runpod-setup.md`](docs/runpod-setup.md).

## Getting access

You need two credentials, and they are separate on purpose: an API key, which lets you
create and destroy pods, and an SSH keypair, which lets you log into them. Neither is
shared between people. If one of us needs to revoke access, or a laptop goes missing, we
want to be able to cut off one person without disrupting the other.

You will not have a console login for the RunPod account, and that is deliberate rather
than an oversight. The account holder is in the UK; a console sign-in from Brazil is the
kind of thing that trips a provider's fraud heuristics and can get an account locked at an
inconvenient moment. API requests and SSH connections carry no such risk, and between them
they cover everything you actually need: creating pods, destroying them, and working on
them. So the two credentials below are issued *to* you rather than created *by* you.

**Your API key** is created by the account holder from the RunPod console, at
Settings → API Keys, and sent to you. Key creation is not exposed over the API, so it
cannot be scripted. Ask for one issued to you specifically rather than a copy of someone
else's — they are revocable individually, and that only helps if they aren't shared. Treat
it as a full-power credential: it can create pods that cost money and delete the volume
that holds all the weights, and RunPod has no read-only mode that would still permit
booting a pod. If it ever ends up somewhere it shouldn't, say so and it gets rotated; that
is a two-minute inconvenience, not a crisis.

Once you have it, create a `.env` file at the root of this repo containing a single line:

```
RUNPOD_API_KEY=rpa_your_key_here
```

`.env` is gitignored and must stay that way. There is an `.env.example` showing the
expected shape. Be aware that if you already have a `RUNPOD_API_KEY` set in your shell
environment for a different RunPod account, every script here deliberately ignores it and
reads `.env` instead — this repo targets a specific account and silently picking up the
wrong one has already caused confusion once.

**Your SSH key** should be generated fresh for RunPod rather than reused from another host,
for the same isolation reason:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_runpod -C "runpod-kimodo-<yourname>"
```

Send the resulting `~/.ssh/id_ed25519_runpod.pub` to the account holder — the `.pub` file
only, obviously; the private key never leaves your machine, and nothing in this workflow
ever asks for it. They register it with:

```bash
node scripts/ssh-key-add.mjs /path/to/your_key.pub
```

RunPod keeps every authorised key in a single account-wide field, one per line, and the
mutation that writes it replaces the whole field — so that script reads the current value
and appends to it. Setting the field directly would silently revoke everyone else. Once
registered, both of you can log into any pod on the account.

One piece of timing matters here. Keys are injected into a pod when it is **created**, not
when it starts. If a pod is already running when your key is added, that pod will not
accept it, and the fix is either to append your key to `/root/.ssh/authorized_keys` inside
the running pod or simply to create a fresh one. Every pod created after your key is
registered picks it up automatically.

The same script lists what is currently authorised, and revokes:

```bash
node scripts/ssh-key-add.mjs --list
node scripts/ssh-key-add.mjs --remove "runpod-kimodo-<name>"
```

Revoking affects pods created afterwards; a pod already running keeps the keys it was
built with, so if you need to cut off access immediately, destroy the pod as well.

Finally, add a host block to `~/.ssh/config` so you are not retyping addresses. The
`IdentitiesOnly` line is not decoration: SSH offers keys one at a time, and if you have a
few in `~/.ssh` you can exhaust the server's authentication attempts before it ever tries
the right one.

```
Host kimodo
    HostName <pod-ip>
    Port <pod-port>
    User root
    IdentityFile ~/.ssh/id_ed25519_runpod
    IdentitiesOnly yes
    ServerAliveInterval 15
    ServerAliveCountMax 10
    TCPKeepAlive yes
```

Leave `HostName` and `Port` as placeholders for now; you'll fill them in below.

## What the account looks like

There is one persistent resource: a 200 GB network volume named `kimodo-motion-cz`, living
in the **EU-CZ-1** datacenter. It holds the model weights, the Python environment, the
cloned repositories and the application code, and it survives pods being destroyed. It
costs roughly $14/month whether or not anything is attached to it, and that is the only
charge that accrues when no pod is running.

Network volumes are locked to their datacenter, and this one constrains everything else:
any pod that mounts it must also run in EU-CZ-1, so the GPU has to be something EU-CZ-1
stocks. That datacenter was chosen because it is the only one carrying the RTX 3090, at
$0.50/hr, which is both the cheapest 24 GB card available to us and — because it is an
Ampere part — the one least likely to fight the older, torch-pinned research code this
project depends on. Newer Blackwell cards require torch 2.7 or later, which several of the
repositories on the roadmap will not build against.

Pods are disposable. Everything worth keeping lives under `/workspace`; anything written
elsewhere is on the container disk and disappears when the pod does.

## Booting a pod

Check what is currently running before creating anything, because a forgotten pod bills
continuously:

```bash
node scripts/pod-status.mjs
```

That prints the volume, any running pods with their hourly rate, and the current SSH
address. To create the standard pod:

```bash
node scripts/pod-create.mjs             # RTX 3090, the default
node scripts/pod-create.mjs --dry-run   # confirm stock without spending anything
```

The script encodes the full specification — datacenter, volume, image, ports, disk — and
refuses to proceed if the volume is missing or the requested GPU is out of stock, rather
than creating something subtly wrong. It waits for SSH to come up and prints the address.
Pass a different GPU id as the first argument to use another card; run it with a bogus id
to have it list the ones currently in stock.

**3090 availability is genuinely tight.** It shows as Low stock most of the time and
EU-CZ-1 is the only datacenter that has it, so creation can fail simply because none are
free. Retry, or fall back to the RTX 4090 at $0.69/hr. Note also that stopping a pod does
not reliably reserve the hardware — you are taking your chances on one being available
whenever you come back, so treat every boot as a fresh allocation.

Once it is up, copy the printed IP and port into the `Host kimodo` block in your SSH
config, and confirm you can reach it:

```bash
ssh kimodo nvidia-smi
```

## Running the motion server

The application code lives at `/workspace/app` on the volume, so it is already there on a
fresh pod. Start it with:

```bash
ssh kimodo 'nohup /workspace/run_server.sh > /workspace/server.log 2>&1 &'
```

Give it about twenty seconds to load CLIP and the checkpoint, then confirm it is ready:

```bash
curl https://<pod-id>-8888.proxy.runpod.net/health
```

A healthy response reports `{"ready":true,"device":"NVIDIA GeForce RTX 3090","steps":50}`.
Open the same URL without the `/health` suffix in a browser and you get the viewer: a
prompt box, a few example prompts, and controls for clip length and guidance. Generating
four seconds of motion takes well under a second, so it is effectively interactive. The
proxy URL is derived from the pod id and therefore changes every time you create a pod.

Port 8888 is used because RunPod already proxies it. Adding another port means recreating
the pod, so it was simpler to serve the application there. The server is currently started
by hand under `nohup` and will not survive a restart on its own; making it a proper service
is outstanding work.

Two limits are worth knowing before you wonder whether something is broken. Clips are
capped at roughly 9.8 seconds because the model was trained on a 196-frame horizon at
20 fps, and longer requests are silently clamped. Guidance defaults to 2.5, which is what
the checkpoint was trained with; raising it makes the model follow the prompt more
literally at the cost of stiffer, less natural movement.

## How the model is wired up

The model is MDM (Motion Diffusion Model), using the 50-step checkpoint rather than the
original 1000-step one — it samples roughly twenty times faster for comparable quality,
which matters when the GPU is metered by the hour.

`pod/motion_server.py` deliberately bypasses MDM's own dataset loader. The upstream
sampling script constructs a full HumanML3D dataset purely to obtain normalisation
statistics and an attribute lookup, which would mean a large download for two small
`.npy` files that already ship inside the repository. Skipping it also means we never
touch the SMPL mesh pipeline: the network emits a 263-dimensional motion vector, and
`recover_from_ric` turns that directly into joint positions, which is all the viewer needs.

That decision has a useful consequence. MDM's constructor eagerly builds a rotation-to-mesh
helper that loads an SMPL pickle containing `chumpy` arrays, and `chumpy` cannot be
installed on Python 3.12 — its `setup.py` imports `pip`, which modern pip rejects. Since
the mesh path is dead weight here, the helper is stubbed out rather than worked around.

A few other sharp edges are documented inline where they bite, but worth flagging so they
don't surprise you: MDM overrides `nn.Module._apply` without returning a value, so `.to()`
evaluates to `None` and must never be chained with `.eval()`; and current versions of
`gdown` have dropped the `--fuzzy` flag that the repository's own download scripts still
use, so bare Google Drive file ids are required.

## Costs, and the habit that matters

Compute bills for as long as a pod exists, whether or not anyone is connected to it, and
whether or not it is doing any work. At $0.50/hr an idle 3090 quietly costs about $12/day.
The volume adds its ~$14/month regardless.

So: run `node scripts/pod-status.mjs` when you sit down and when you finish, and delete the
pod when you are done with it. Everything you care about is on the volume, and
`pod-create.mjs` will rebuild an identical pod in about a minute.
