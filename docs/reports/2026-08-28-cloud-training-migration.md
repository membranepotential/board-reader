# Cloud training migration — cost analysis and decision

_2026-08-28 · decision: **move GPU training to RunPod**; keep CPU work, eval and
gating local._

## Summary

Training compute for this project is **negligibly cheap** — a full `run_g`-equivalent
is ~2.2 PFLOP, which a rented Ampere card chews through in about three minutes for
roughly **two cents**. The GTX 1650 is not expensive, it is *slow*: 165 img/s,
compute-bound, and it serialises the entire project because one training run owns the
card.

The costs that actually matter are not GPU-hours:

1. **Idle pod time** — leaving a pod up is ~100× the cost of the work it does.
2. **The one-time cache upload** — 8–34 GB over home upstream.
3. **Engineering time** — the harness assumes a Turing card with 4 GB and has
   Turing-specific settings baked into its defaults.

At ~$0.02/run, the funded **$7.91 balance buys roughly 300 experiments**. The binding constraint stops
being the GPU and becomes our ability to design and read experiments.

## Local baseline (measured, not estimated)

From `scratch/runs/run_g/events.jsonl` — the last completed run:

| | |
|---|---|
| Wall time | **2405 s (40.1 min)** for 20 epochs |
| Train throughput | 170 img/s, fp32 (`amp: false`) |
| Peak VRAM | 2.4 GB of 4 GB (val pass; steady-state steps are ~218 MiB) |
| Images seen | 288k train + 320k val |
| Total compute | **≈2.2 PFLOP** |
| Train pool available | `n_train: 663,680` |

Two facts from `.learnings.ndjson` sharpen this:

- The 1650 is **compute-bound**, measured flat across batch 48/64/96 (163/166/170 img/s)
  and workers 6/8/12 (168/163/169). The JPEG loader keeps up *at this speed*. Its
  ceiling above ~170 img/s is unmeasured.
- **Peak VRAM at bs96 is ~250 MiB.** So 4 GB does *not* bind ResNet-18 today.

That second point matters for how we justify this. The VRAM ceiling does not constrain
the runs we do — it constrains **the runs we have never attempted**: unfrozen backbones,
ViT-B fine-tunes, 384 px inputs. What constrains today's runs is pure throughput.

### The under-training finding

`run_g` uses `steps_per_epoch: 300` at `batch_size: 48` = 14,400 images/epoch, against a
pool of 663,680. **We show the model 2.2 % of the available data per epoch**, and 288k
images across the whole run — less than half the dataset, seen once.

That is not a modelling choice. It is a wall-clock choice: a true epoch is 46× larger, so
a real 20-epoch pass is **~22 hours** on the 1650. On a rented A6000 it is ~100 minutes and $0.55.

This is the single highest-value experiment the migration unlocks, and it requires **no
new code** — one config key.

## Cost model

### Per-run

The RTX A6000 has 38.7 TFLOPS fp32 against the 1650's 2.98 — 13× on raw fp32 — plus
native Ampere bf16 tensor cores the Turing card does not have at all. Anchors: NVIDIA's
own ResNet-50 v1.5 PyTorch benchmark gives 2470 img/s on 1× A100 (AMP) vs 938 (TF32), and
ResNet-18 is ~2.25× fewer FLOPs. Derated for a small model's kernel-launch overhead and
loader limits, **10–16× is the honest band**.

**This estimate is replaced by a measurement in Phase 1.6 of the plan. Do not budget
anything off it before then.**

| Workload | Local (1650) | A6000 @ ~2000 img/s | Cost @ $0.33/hr |
|---|---|---|---|
| `run_g` as configured | 40 min | ~3–4 min | **~$0.02** |
| 50-run sweep | 33 h | ~3 h | **~$1** |
| Full-data run (13.3M imgs) | ~22 h | ~100 min | **~$0.55** |

### Provider comparison

Prices below are **live from the RunPod API on 2026-08-28**, not from marketing pages.
Note that `runpodctl pod create` defaults to `--cloud-type SECURE`, which is roughly
double the community price — an easy and expensive mistake.

| Provider / GPU | VRAM | $/hr | Notes |
|---|---|---|---|
| **RunPod RTX A6000 (community, EU-SE-1)** | 48 GB | **$0.33** | chosen — cheapest *and* most VRAM |
| **RunPod A40 (secure, EU-SE-1)** | 48 GB | $0.44 | fallback; stock Medium vs A6000's Low, same DC |
| RunPod RTX 4090 (community) | 24 GB | $0.34 | ~2× the raw fp32, but EUR-IS-2 only, Low stock, no same-DC fallback |
| RunPod L40S (community) | 48 GB | $0.79 | fastest of these; revisit if throughput ever justifies 2.4× |
| Vast.ai 4090 (interruptible) | 24 GB | ~$0.30 | anonymous-host marketplace |
| Scaleway L4-1-24G | 24 GB | €0.79 | EU residency, real VPC — no gain here |
| Lambda A100 40 GB | 40 GB | $1.99 | 1× SKUs often unavailable |

The A6000 choice is not a compromise: it is simultaneously the cheapest option on the
list **and** the one with the most VRAM, which is precisely the axis that gates the
experiments we have never been able to run.

### Storage and transfer

- **Network volume: $0.07/GB/month** under 1 TB. **Billed while the pod is stopped** —
  that is the point (it persists), but it is a standing charge.
- **Egress is free** on RunPod. Ingress is free everywhere.
- Network volumes are **datacenter-locked** — no cross-DC transfer command in
  `runpodctl`. Pinning a DC pins our GPU availability pool. Accepted risk.

What has to travel (training reads **only** `data/external/_cropcache/`, confirmed in
`dataset.py:load_styles` — the raw datasets are never touched):

| Cache variant | Size | Needed for |
|---|---|---|
| `hr1.6_sp0.15_maskgray` | 7.9 GB | run_g / E7 member 1 |
| `pose_h1.9_r0.5_f1_maskgray` | 2.7 GB — **incomplete, 2 of 11 styles** | run_h / E7 member 2 |
| `hr1.6_sp0.15` | 8.8 GB | run_e re-baseline |
| `pose_h1.9_r0.5_f1` | 8.8 GB | run_f re-baseline |
| `data/eval` + `data/uploads` + `data/images` | ~100 MB | `evaluate.py` on-pod scoring |

~34 GB once complete → **$2.40/month** on a 40 GB volume.

### Monthly scenarios

| Pattern | GPU hours | Monthly |
|---|---|---|
| Disciplined — spin up, sweep, tear down | 8 | **~$4** |
| Heavy — 4 h/evening × 20 evenings | 80 | **~$28** |
| **Careless — pod left running** | 720 | **~$240** |

The gap between row 2 and row 3 is the whole risk. Teardown discipline is worth more
than any GPU choice on this list — and RunPod has **no native idle-timeout or TTL**, so
nothing cleans up after us.

One structural protection is already in place: the account holds **$7.91 with no
auto-recharge**, so the balance *is* a hard ceiling. A forgotten pod burns what is left
and then RunPod terminates it. Keeping it that way is worth more than any alert.

## Why not rebuild the cache on the pod instead of uploading

Because we cannot. **Dataset provenance is undocumented** — there is no acquisition
script, no source URL, and no `SOURCES` file anywhere in the repo for `chessrender360`,
`chessred`, or `chesscog`. A fresh pod cannot reconstruct `data/external/` from scratch.

Shipping the cache sidesteps this entirely, and it is the smaller transfer anyway
(34 GB of cache vs 26 GB of raw sources *plus* an undocumented rebuild).

Worth fixing separately: the provenance gap is a genuine reproducibility hole that has
nothing to do with this migration.

## Risks, from the learnings store

| Risk | Mitigation |
|---|---|
| `StyleData` is **~9 GB RAM-resident**; torch 2.14's forkserver default made workers full pickled copies and OOM-killed run_f | Filter pods for **≥32 GB RAM**; keep `multiprocessing_context='fork'` |
| Loader ceiling above ~170 img/s is **unmeasured** — it will likely bind at ~2000 img/s | Explicit loader benchmark in Phase 2.1 before believing any speedup |
| `amp: false` is a **Turing-specific** measurement (fp16 NaNs in the *forward* pass; GradScaler cannot help). Ampere has native bf16 | Re-benchmark AMP on the pod. Do not assume either way |
| cuDNN picks different kernels per GPU; 2 of 10 seeds already failed to reproduce exactly locally (±0.x acc) | Re-baseline run_e/run_f on the pod (~$0.10) before comparing anything to old numbers |
| `_env.sh` preflight exists because a run trained 20 min then died at export on the wrong interpreter | Never bypass it; the pod image must satisfy it |
| Pausing CUDA training with SIGSTOP produces a zombie-spinner (~30× throughput collapse, invisible to `nvidia-smi`) | Pod stop/resume = checkpoint + kill + resume-from-checkpoint |
| Community-cloud hosts are third parties | Data is public datasets plus our own board photos — low sensitivity, but named explicitly. Secure Cloud at $0.69/hr if this ever stops being true |

## What the migration unlocks

Ordered by value-per-unit-effort:

1. **Full-data training** — consume the 663k crops we already have instead of 2.2 % of
   them. Zero new code. ~$0.55.
2. **Unfreeze the backbone.** Our own finding is that *no frozen encoder beats ~80 %
   cross-style*. Full fine-tuning is the untested lever and it is exactly what 4 GB
   forbids. `unfreeze_blocks` / `freeze_backbone` already exist in `Config`.
3. **Architecture bakeoff.** `models.py` currently registers only
   `resnet18 | resnet50 | dinov2`. Adding timm-backed ConvNeXt-T, EfficientNetV2-S,
   ViT-S/16 and Swin-T is contained work, and `timm` is already in the venv recipe.
4. **Resolution.** 224 → 320/384 for the tall-crop retrain (Next #1). Needs new caches.
5. **Parallelism.** N pods = N configs in one wall-clock slot.
6. **Unblocking the local GPU**, which currently serialises every workstream in STATUS.

## Decision

Move GPU training to **RunPod RTX A6000 community pods in `EU-SE-1`** (A40 in the same
datacenter as the stock fallback), with a 20 GB persistent network volume. Keep on the
local box: cache building (CPU), `evaluate.py`, the replay harness, and every
curated/settled gate. **The cloud produces candidates; nothing ships without passing the
existing local gates.**

The datacenter is the load-bearing choice, because network volumes are DC-locked and
`runpodctl` has no cross-DC transfer. `EU-SE-1` is the only datacenter carrying two
viable GPUs, so a stockout on the A6000 has an answer that does not strand the volume.

Budget: **~$5.30 in the first month** against a $7.91 balance.

Execution plan: [docs/plans/2026-08-28-runpod-migration.md](../plans/2026-08-28-runpod-migration.md)
· operating guide: [.claude/skills/runpod/SKILL.md](../../.claude/skills/runpod/SKILL.md).

### Incidental finding

`.env` — holding `RUNPOD_API_KEY` — **was not gitignored, on a repo that is public.**
The key was never committed (`git log --all -- .env` is empty), so no rotation was
required, but the exposure was one careless `git add -A` away.

Resolved by moving the key out of the filesystem entirely: it now lives in the login
keyring, reached through `tools/runpod/env.sh` (source it) or
`tools/runpod/with-key <cmd>` (single command, key scoped to that process). `.env` is
deleted, and `.env`/`.env.*` are gitignored as defence in depth. Note that
`runpodctl config --apiKey` writes a cleartext `~/.runpod/config.toml` — do not use it.

## Sources

Pricing and benchmark figures verified 2026-08-28:

- [RunPod pricing 2026](https://gpuperhour.com/providers/runpod) ·
  [RunPod storage/network-volume pricing](https://www.hivenet.com/post/runpod-pricing-complete-guide-to-gpu-cloud-costs) ·
  [RunPod pod docs](https://docs.runpod.io/pods/choose-a-pod)
- [Lambda pricing](https://lambda.ai/pricing) ·
  [Lambda persistent storage](https://lambda.ai/blog/persistent-storage-for-lambda-cloud-is-expanding)
- [Scaleway GPU pricing](https://www.scaleway.com/en/pricing/gpu/) ·
  [Scaleway storage pricing](https://www.scaleway.com/en/pricing/storage/)
- [Vast.ai pricing 2026](https://gpus.io/en/providers/vast-ai)
- [NVIDIA ResNet-50 v1.5 PyTorch benchmarks](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/resnet50v1.5/README.md)
