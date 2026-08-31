---
name: runpod
description: Rent a RunPod GPU to run a training sweep, then destroy it — the whole point is that a pod exists only while it is computing. Use whenever work needs a real GPU instead of the local GTX 1650, when the user says "/runpod", "train on the cloud", "spin up a pod", "run this sweep on RunPod", "how much have we spent on RunPod", or asks to check/clean up cloud GPU resources.
allowed_tools: ["Bash", "Read", "Write", "Edit"]
---

# RunPod — rent a GPU, use it, destroy it

The local GTX 1650 is compute-bound at ~165 img/s and serialises every workstream
in the project. RunPod exists to break that. It is **not** a place to keep a
machine.

**The budget is real and small: ~$14 as of 2026-08-31, no auto-recharge.** A
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
A viable DC must clear **three** constraints at once: it supports network
volumes, it supports the S3 API (our staging path, §3), and it stocks our GPU
tier. All three were measured 2026-08-31; the method to re-measure is below.

**Pinned: `EUR-IS-1`** (Reykjavík) — the only DC that had *both* our GPU tiers
in stock at the 2026-08-31 exploration, plus volume + S3 support. The previous
pin (`EU-RO-1`) is dead: its volume was destroyed 2026-08-31 (no standing
volume exists — recreate in `EUR-IS-1` on next use, §2). US datacenters were
explicitly surveyed and **none stocked consumer 4090-class cards**; the best US
option is recorded below as the fallback pin.

| GPU | VRAM | Price | Cloud | Role |
|---|---|---|---|---|
| RTX 4090 | 24 GB | **$0.34/hr** | COMMUNITY | primary |
| RTX PRO 6000 | 96 GB | $1.69/hr | COMMUNITY | explicit escalation for VRAM-hungry runs (`rp --fallback-pro6000`) |
| RTX A4000 | 16 GB | $0.17/hr | COMMUNITY | cheap smoke-tests only (16 GB, Ampere) |

Runner-ups from the survey, if `EUR-IS-1` dries up: `EUR-NO-1` (4090 + 5090,
volumes + S3) in Europe; **`US-NE-1`** if a US region is wanted (no 4090s, but
PRO 6000 MIG 24 GB at $0.59/hr SECURE is a fine 24 GB Blackwell substitute,
volumes + S3).

All pinned GPUs are post-Ampere (Ada/Blackwell): native bf16 tensor cores,
unlike the local Turing card where fp16 NaNs in the forward pass. 24 GB covers
everything short of a large unfrozen backbone at 384 px; those runs escalate to
the PRO 6000 explicitly rather than re-pinning the volume.

### Re-detecting the pin (the 2026-08-31 method, ~2 commands)

Stock moves; re-run this before recreating a volume, never for a routine sweep:

```bash
# 1. Which DCs stock which GPUs (NB: this view is secure-cloud-centric;
#    community stock per-DC is not directly visible):
runpodctl gpu list -o json | python3 -c "
import json,sys,collections
by_dc = collections.defaultdict(list)
for g in json.load(sys.stdin):
    for d in (g.get('dataCenterAvailability') or []):
        if d.get('stockStatus') not in ('none', None):
            by_dc[d['dataCenterId']].append((g['displayName'], d['stockStatus']))
for dc, gs in sorted(by_dc.items(), key=lambda kv: -len(kv[1])):
    print(dc, ':', ', '.join(f'{n}({s})' for n, s in gs))
"
# 2. Which DCs support volumes (the create error lists them; creates nothing):
runpodctl network-volume create --name probe --size 10 --data-center-id XX-INVALID-9
# 3. Which DCs support the S3 API: https://docs.runpod.io/storage/s3-api
# Pin the intersection that carries the 4090 (or the best 24GB+ bf16 card).
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
. ./tools/runpod/env.sh                             # key from the keyring, never a file
runpodctl user                                    # balance covers the plan?
runpodctl pod list                                # MUST be [] — else reap first
runpodctl network-volume list                     # volume exists? right DC?
```

### 2. Create the network volume (once, ever)

Size it for the caches actually needed, not for comfort — it bills at
**$0.07/GB/month whether or not a pod is attached**.

```bash
runpodctl network-volume create --name board-reader --size 20 --data-center-id EUR-IS-1
```

(The 2026-08-28 volume in `EU-RO-1` was destroyed 2026-08-31 to stop the
standing $0.07/GB/mo — data must be re-staged (§3) after recreating.)

### 3. Stage the data (once, over the S3 API — no pod at all)

Training reads **only** `data/external/_cropcache/` (confirmed in
`dataset.py:load_styles`) — the raw datasets never need to travel. Ship the cache
variant the sweep needs plus `data/eval`, `data/uploads` and `data/images` (~100
MB) so `evaluate.py` can score candidates on the pod.

RunPod exposes network volumes over an **S3-compatible API**
(https://docs.runpod.io/storage/s3-api) — the bucket is the volume id, the
endpoint is DC-specific (`https://s3api-eur-is-1.runpod.io` for the current
pin). Uploading this way costs nothing, resumes (`aws s3 sync`), and does not
depend on instance stock — which matters, because a DC can have **zero CPU
instances available** (EU-RO-1 did) and
`rp stage`'s CPU-pod path then cannot run at all. Auth is a *separate* S3 API
key (console → Settings → S3 API Keys), stored in the keyring under
`service runpod-s3 part access|secret`:

```bash
. ./tools/runpod/s3-env.sh    # AWS_* from the keyring; neutralizes ~/.aws
aws s3 sync data/external/_cropcache/<variant> \
  "s3://$BR_RP_S3_BUCKET/_cropcache/<variant>" --endpoint-url "$BR_RP_S3_ENDPOINT"
```

Key layout on the volume must match what `rp`'s `link_data` symlinks expect:
`_cropcache/<variant>/`, `data/eval`, `data/uploads`, `data/images`. Files
>500 MB need multipart (aws cli does this automatically; parts cap at 500 MB).
`rp stage` (CPU pod + rsync) remains the fallback when stock allows.

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
. ./tools/runpod/env.sh                        # key from keyring (or: tools/runpod/with-key <cmd>)
runpodctl user                                 # balance
runpodctl gpu list                             # prices + per-DC stock
runpodctl datacenter list                      # what each DC carries
runpodctl network-volume list|create|delete
runpodctl pod list                             # [] means nothing is burning
runpodctl pod create --image <img> --gpu-id "NVIDIA GeForce RTX 4090" \
  --cloud-type COMMUNITY --data-center-ids EUR-IS-1 \
  --network-volume-id <id> --public-ip --wait
runpodctl pod logs <id>
runpodctl pod delete <id>                      # ALWAYS this, never `pod stop`
runpodctl send <file> / runpodctl receive <code>
```

Useful flags: `--wait` blocks until SSH answers (needs `--public-ip` on community
cloud); `--wait-timeout` defaults to 10m and **keeps the pod on timeout**, with
the id in the error — reap it. `--env` takes a JSON object. `--container-disk-in-gb`
defaults to 20.
