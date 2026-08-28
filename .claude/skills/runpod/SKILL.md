---
name: runpod
description: Rent a RunPod GPU to run a training sweep, then destroy it — the whole point is that a pod exists only while it is computing. Use whenever work needs a real GPU instead of the local GTX 1650, when the user says "/runpod", "train on the cloud", "spin up a pod", "run this sweep on RunPod", "how much have we spent on RunPod", or asks to check/clean up cloud GPU resources.
allowed_tools: ["Bash", "Read", "Write", "Edit"]
---

# RunPod — rent a GPU, use it, destroy it

The local GTX 1650 is compute-bound at ~165 img/s and serialises every workstream
in the project. RunPod exists to break that. It is **not** a place to keep a
machine.

**The budget is real and small: the account holds ~$8 with no auto-recharge.** A
pod left running overnight costs more than every experiment we have ever run.
Read [Cost discipline](#cost-discipline) before creating anything.

Background and the migration decision:
[docs/reports/2026-08-28-cloud-training-migration.md](../../../docs/reports/2026-08-28-cloud-training-migration.md).

## The one rule

> **A pod is created by a script that also destroys it. Never create a pod you
> are not, in the same command, already committed to deleting.**

Everything below follows from that.

## Setup

The API key lives in the **login keyring**, never on disk in cleartext — this
repo is public, and a plaintext `.env` is one `git add -f` away from being
published. `runpodctl` reads `RUNPOD_API_KEY` from the environment, so a helper
injects it from the keyring for the life of one command:

```bash
tools/runpod/with-key runpodctl user   # sanity check: prints balance
```

For a session with many calls, source the env shim instead of a `.env` file:

```bash
. ./tools/runpod/env.sh            # exports RUNPOD_API_KEY from the keyring
runpodctl user
```

Never `cat`/`echo` the key, and never write it back to a file — not `.env`, not
`~/.runpod/config.toml` (which `runpodctl config --apiKey` creates world-readable;
if it reappears, delete it). To enrol or rotate the key:

```bash
secret-tool store --label='RunPod API key' service runpod key RUNPOD_API_KEY
```

`runpodctl` is v2 — the syntax is `runpodctl <resource> <action>`, and
`-o json` is the default output. The old top-level `create` / `stop` / `get`
verbs are deprecated; do not use them.

## Cost discipline

Ordered by how much money each one saves.

1. **`pod delete`, never `pod stop`.** A stopped pod releases the GPU but keeps
   billing for its volume disk, and it is easy to forget. Deleting is free of
   consequence *because all durable state lives on the network volume*, which
   survives deletion.
2. **`--cloud-type COMMUNITY`.** The flag defaults to `SECURE`, which is roughly
   **2× the price** (A6000: $0.33 vs $0.53; 4090: $0.34 vs $0.74). Forgetting
   this flag is the single most expensive mistake available.
3. **One pod per *sweep*, never per run.** Pod creation and environment setup
   cost minutes; a `run_g`-sized training costs ~3. Queue every config you want
   and run them back-to-back on one pod.
4. **Trap the teardown.** The runner must delete the pod on `EXIT`, `INT` and
   `TERM`, not only on the happy path.
5. **Upload on a CPU pod.** Moving 8 GB to the network volume needs no GPU.
   Attach the volume to a `--compute-type cpu` pod for that.
6. **Never leave the balance topped up.** With ~$8 and no auto-recharge, the
   balance *is* the hard ceiling — a forgotten pod burns at most what is left
   and then RunPod kills it. Do not enable auto-recharge, and consider lowering
   the account `spendLimit` (currently $80) to something near the balance.

There is **no native idle-timeout or TTL** on RunPod pods. Nothing will clean up
after us. That is why the trap and `reap` (below) exist.

## Where to rent

Network volumes are **datacenter-locked** — `runpodctl` has no cross-DC transfer
— so the datacenter choice pins our GPU pool for as long as the volume exists.

**Pin `EU-SE-1`.** It is the only DC carrying two viable GPUs, so a stockout on
the cheap one has a fallback in the same DC:

| GPU | VRAM | Price | Cloud | Stock | Role |
|---|---|---|---|---|---|
| RTX A6000 | 48 GB | **$0.33/hr** | COMMUNITY | Low | primary |
| A40 | 48 GB | $0.44/hr | SECURE | Medium | fallback when A6000 is out |

Both are Ampere: native bf16 tensor cores (unlike the local Turing card, where
fp16 NaNs in the forward pass) and 48 GB, which is what makes unfrozen backbones
and 384 px inputs possible at all.

Avoid pinning `EUR-IS-2` for the RTX 4090 ($0.34): it is the only DC that has
one, stock is Low, and there is no same-DC fallback.

Re-check before every session — stock moves:

```bash
runpodctl gpu list -o json | python3 -c "
import json,sys
for g in json.load(sys.stdin):
    if g['gpuId'] in ('NVIDIA RTX A6000','NVIDIA A40'):
        print(g['displayName'], g.get('communityPricePerHr'), g.get('securePricePerHr'))
        for d in g.get('dataCenterAvailability') or []:
            if d['stockStatus'] not in ('none', None): print('  ', d['dataCenterId'], d['stockStatus'])
"
```

## Pod requirements for this project

Non-negotiable, and each one is a scar:

- **≥32 GB system RAM.** `StyleData` holds ~9 GB of JPEG bytes resident. Torch
  2.14 changed the multiprocessing default to `forkserver`, which turns
  DataLoader workers into full pickled copies — that OOM-killed a `run_f`
  launch. Keep `multiprocessing_context='fork'` *and* give the box headroom.
- **A CUDA image matching the venv recipe** in `tools/train/README.md`. The
  harness needs torch, torchvision, timm, cv2, yaml **and** the full ONNX stack
  (`onnx`, `onnxscript`, `onnxruntime`). `onnxscript` is not optional: from
  torch 2.9 `torch.onnx.export` routes through the dynamo exporter, and without
  it `export.py` dies *after* the GPU time is already spent.
- **Run `tools/train/_env.sh` before any GPU work.** It preflights the
  interpreter and aborts if the ONNX stack or CUDA is missing. It exists because
  a run once trained for 20 minutes against the wrong interpreter and then
  crashed at export. Do not bypass it.
- **Never `SIGSTOP` a CUDA training process.** On resume it can become a
  zombie-spinner: GPU pegged at 100 % util while throughput collapses ~30×, and
  both `nvidia-smi` and liveness checks look healthy. Only a steps/min monitor
  catches it. To pause, checkpoint + kill + resume from checkpoint. A wedged
  CUDA process can survive `kill -9` on the PID — kill the process *group*
  (`kill -9 -PGID`).

## The workflow

### 1. Preflight — before spending anything

```bash
set -a; . ./.env; set +a
runpodctl user                                    # balance covers the plan?
runpodctl pod list                                # MUST be [] — else reap first
runpodctl network-volume list                     # volume exists? right DC?
```

### 2. Create the network volume (once, ever)

Size it for the caches actually needed, not for comfort — it bills at
**$0.07/GB/month whether or not a pod is attached**.

```bash
runpodctl network-volume create --name board-reader --size 20 --data-center-id EU-SE-1
```

### 3. Stage the data (once, on a CPU pod)

Training reads **only** `data/external/_cropcache/` (confirmed in
`dataset.py:load_styles`) — the raw datasets never need to travel. Ship the cache
variant the sweep needs plus `data/eval`, `data/uploads` and `data/images` (~100
MB) so `evaluate.py` can score candidates on the pod.

Do this on a CPU pod with the volume attached. There is no reason to pay for an
idle GPU while 8 GB crawls up a domestic uplink.

### 4. Run a sweep

Always through the runner script, never by hand:

```bash
tools/train/cloud/rp sweep configs/run_g.yaml configs/run_h.yaml
```

The runner must, in this order: preflight → create pod (`--cloud-type COMMUNITY`,
`--network-volume-id …`, `--wait`) → **install the EXIT/INT/TERM trap that
deletes the pod** → `_env.sh` → run each config → pull `best.pt`, `model.onnx`,
`history.json`, `events.jsonl` back → delete the pod.

If the runner does not exist yet, build it before running anything expensive.
Driving pod creation by hand from a chat session is exactly how a pod gets
forgotten.

### 5. Reap — every single time, and whenever in doubt

```bash
runpodctl pod list                     # expect []
runpodctl pod delete <pod-id>          # anything still standing
```

Cheap, idempotent, and the only thing standing between a dropped SSH connection
and the whole budget. Run it after any interrupted session, any crash, any
`Ctrl-C`.

### 6. Check the damage

```bash
runpodctl user       # clientBalance — compare against the last reading
runpodctl billing
```

## Division of labour

The cloud is a GPU, nothing more.

| Stays local | Goes to RunPod |
|---|---|
| Cache building (`cropcache.py` — pure CPU) | Training (`train.py`) |
| Replay harness, curated + settled gates | In-loop candidate scoring (`evaluate.py`) |
| Trial recordings, `truth.json` | |
| Every ship/no-ship decision | |

**The cloud produces candidates; nothing ships without passing the existing local
gates.** Models come back as a 45 MB `best.pt` / `model.onnx`.

## Traps specific to this move

- **cuDNN picks different kernels on different GPUs.** Two of ten seeds already
  failed to reproduce exactly on one local card (±0.x accuracy). Cloud numbers
  are **not** comparable to historical local numbers. Re-baseline `run_e`/`run_f`
  on the pod once (~$0.10) before comparing anything against them.
- **`amp: false` is a Turing-specific measurement**, not a general truth. On
  Ampere, bf16 is native and should win — but benchmark it on the pod rather
  than assuming either way. Same for `channels_last` and `compile`, both of which
  already exist in `Config` and are off for local reasons.
- **The loader ceiling above ~170 img/s is unmeasured.** Locally the JPEG loader
  keeps up and throughput is flat across workers 6/8/12 — but that only proves it
  keeps up *at 170 img/s*. At 3000 img/s it is the prime suspect. Measure it
  before believing any speedup number.
- **Dataset provenance is undocumented** — no acquisition script or source URLs
  exist for `chessrender360`, `chessred` or `chesscog`. A fresh pod cannot
  rebuild `data/external/` from scratch. This is why we ship the cache, and it is
  a reproducibility hole worth closing separately.

## Quick reference

```bash
set -a; . ./.env; set +a                       # load key (never print it)
runpodctl user                                 # balance
runpodctl gpu list                             # prices + per-DC stock
runpodctl datacenter list                      # what each DC carries
runpodctl network-volume list|create|delete
runpodctl pod list                             # [] means nothing is burning
runpodctl pod create --image <img> --gpu-id "NVIDIA RTX A6000" \
  --cloud-type COMMUNITY --data-center-ids EU-SE-1 \
  --network-volume-id <id> --public-ip --wait
runpodctl pod logs <id>
runpodctl pod delete <id>                      # ALWAYS this, never `pod stop`
runpodctl send <file> / runpodctl receive <code>
```

Useful flags: `--wait` blocks until SSH answers (needs `--public-ip` on community
cloud); `--wait-timeout` defaults to 10m and **keeps the pod on timeout**, with
the id in the error — reap it. `--env` takes a JSON object. `--container-disk-in-gb`
defaults to 20.
