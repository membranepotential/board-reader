# Plan — move GPU training to RunPod

_2026-08-28 · analysis: [docs/reports/2026-08-28-cloud-training-migration.md](../reports/2026-08-28-cloud-training-migration.md)
· operating guide: [.claude/skills/runpod/SKILL.md](../../.claude/skills/runpod/SKILL.md)_

**Budget: $7.91** (live account balance, no auto-recharge). The whole plan below
costs **~$5.30 in its first month** and leaves headroom. The balance itself is
the hard ceiling — nothing here can overrun it.

## Ground state (verified 2026-08-28 against the live API)

- Account funded **$7.91**, `spendLimit` $80, no pods, no network volumes.
- `runpodctl v2.0.0-beta.1` installed at `/usr/bin/runpodctl`.
- `RUNPOD_API_KEY` lives in the **login keyring**, reached via
  `tools/runpod/env.sh` (source it) or `tools/runpod/with-key <cmd>` (one
  command). It was briefly in a plaintext `.env` on this public repo; that file
  is now deleted and `.env` is gitignored as defence in depth. The key was never
  committed (`git log --all -- .env` is empty), so no rotation was needed.
- **Datacenter: `EU-SE-1`** — the only DC with two viable GPUs, which matters
  because network volumes are DC-locked:
  - RTX A6000, 48 GB, **$0.33/hr** COMMUNITY, stock Low — primary
  - A40, 48 GB, $0.44/hr SECURE, stock Medium — fallback
- Local baseline to beat: `run_g` = 2405 s, 170 img/s, `best_acc 0.8983`.

Both candidates are Ampere with 48 GB — native bf16 and 2× the VRAM of the 4090
option, at a lower price. Expected speedup **10–16×**, so a `run_g`-equivalent
lands at ~3 min and **~$0.02**.

## Phase 0 — local prep (no spend)

- [ ] **0.1** Complete the `pose_h1.9_r0.5_f1_maskgray` cache. It is missing 9 of
      11 styles (2.7 GB vs the 8.8 GB unmasked variant), so `run_h`/E7 member 2
      cannot run anywhere until it is built. Pure CPU — the local GPU is free
      meanwhile.
- [ ] **0.2** Write `tools/train/cloud/rp` — the runner. Non-negotiable
      properties, in order: preflight (balance, no orphan pods, volume present) →
      create with `--cloud-type COMMUNITY` → **`trap … EXIT INT TERM` that deletes
      the pod** → `_env.sh` → run the config queue → pull artifacts → delete.
      Subcommands: `sweep`, `stage`, `reap`, `cost`.
- [ ] **0.3** Write `tools/train/cloud/setup.sh` — reproduces the venv recipe from
      `tools/train/README.md` on a fresh pod and ends by running `_env.sh`.
      Pin the CUDA wheel index to the pod image's CUDA, not the local cu13x.
- [ ] **0.4** Lower the account `spendLimit` from $80 toward the balance. Belt and
      braces; the balance is already the real cap.
- [ ] **0.5** Measure home upstream and decide whether 7.9 GB goes up in one push
      or overnight. This sets the Phase 1 schedule, nothing else.

**Gate:** `rp` deletes its pod on `Ctrl-C`, on error, and on success. Test this
against a CPU pod (cents) before it ever guards a GPU.

## Phase 1 — beachhead: prove parity (~$0.60)

- [ ] **1.1** Create the network volume: 20 GB in `EU-SE-1` (**$1.40/month**,
      billed whether or not a pod is attached). Sized for
      `hr1.6_sp0.15_maskgray` (7.9 GB) + eval data (~0.1 GB) + run artifacts,
      with room for one more cache. Expand later rather than over-provision now.
- [ ] **1.2** Stage data on a **CPU pod** with the volume attached — no reason to
      pay for an idle GPU while 8 GB uploads. Ship
      `_cropcache/hr1.6_sp0.15_maskgray/`, `data/eval`, `data/uploads`,
      `data/images`, and the repo.
- [ ] **1.3** Bring up an A6000, run `setup.sh`, confirm `_env.sh` passes.
- [ ] **1.4** Run `configs/smoke.yaml` — proves the loop end to end for pennies.
- [ ] **1.5** **The parity gate.** Re-run `run_g` *exactly as configured* —
      `amp: false`, batch 48, `steps_per_epoch: 300`, `seed: 0`. Compare to the
      local `best_acc 0.8983`.
- [ ] **1.6** Record wall time. **This replaces the 10–16× estimate with a
      measurement; every budget downstream is recomputed from it.**

**Gate:** `best_acc` within ~±0.005 of 0.8983. cuDNN kernel selection differs
across GPUs, so byte-identical is not the bar and never will be — but a larger
gap means something is wrong with the data or the geometry, not with cuDNN, and
must be chased before spending more.

## Phase 2 — tune the box, then make it routine (~$0.70)

- [ ] **2.1** Short-run benchmark grid (2 epochs each): `amp` {off, bf16, fp16} ×
      `channels_last` {off, on} × `compile` {off, on}, then `batch_size`
      {48, 128, 256} and `num_workers` {8, 16, 24}. The local `amp: false` is a
      *Turing* result — Ampere has native bf16 and should invert it, but measure.
- [ ] **2.2** **Find the loader ceiling explicitly.** Locally the JPEG loader is
      flat across workers 6/8/12, but that only proves it keeps up at 170 img/s.
      At 3000 img/s it is the prime suspect. Measure img/s against workers with
      the model replaced by a no-op before believing any headline speedup.
- [ ] **2.3** Land the winners as `configs/_cloud_base.yaml`; document the
      measured numbers in `tools/train/README.md`. Keep
      `multiprocessing_context='fork'` regardless of what the grid says.
- [ ] **2.4** Re-baseline `run_e` and `run_f` on the pod (~$0.05) so later
      comparisons have same-hardware references.

**Gate:** a documented, reproducible cloud config whose throughput is measured,
not assumed.

## Phase 3 — the experiments the 4 GB card forbade (~$2.20)

Ordered by value per unit of effort.

- [ ] **3.1 Full-data training — do this first.** `run_g` shows the model
      `300 × 48 = 14,400` images/epoch against a pool of **663,680** — **2.2 % per
      epoch**, and under half the dataset seen even once across the whole run.
      That is a wall-clock artefact (a true epoch is 46× larger = ~22 h locally),
      not a modelling decision. Set `steps_per_epoch: null`, change nothing else.
      **~100 min, ~$0.55, zero new code.** Highest-value single experiment
      available.
- [ ] **3.2 Unfreeze the backbone.** Our own finding is that *no frozen encoder
      beats ~80 % cross-style*; full fine-tuning is the untested lever and is
      exactly what 4 GB forbids. `unfreeze_blocks` and `freeze_backbone` already
      exist in `Config`. 48 GB makes the whole range reachable.
- [ ] **3.3 Architecture bakeoff.** Needs `models.py` work — `build_model`
      registers only `resnet18 | resnet50 | dinov2`. Add timm-backed ConvNeXt-T,
      EfficientNetV2-S, ViT-S/16, Swin-T; `timm` is already in the venv recipe.
      Land the registry change with tests before renting anything.
- [ ] **3.4 Resolution.** 224 → 320 → 384 for the tall-crop retrain (Next #1).
      Needs new caches at those sizes — build them locally on CPU (Phase 0.1
      pattern), upload, then train.
- [ ] **3.5** Feed results back into the tall-crop retrain and the residual
      type-confusion work (STATUS Next #1 and #6).

**Gate:** every candidate goes through `evaluate.py` and then the **local**
replay + curated + settled gates. The cloud produces candidates; nothing ships
on a cloud number alone.

## Budget

| Item | Cost |
|---|---|
| Network volume, 20 GB | **$1.40 / month** (standing) |
| Phase 1 — parity | ~$0.60 |
| Phase 2 — tuning + re-baseline | ~$0.70 |
| Phase 3 — experiments | ~$2.20 |
| **First month total** | **~$5.30** of $7.91 |

Every figure after Phase 1.6 is recomputed from the measured speedup.

## Risks and how each is handled

| Risk | Handling |
|---|---|
| **Forgotten pod drains the balance** | Trap-guarded runner; `reap` after every session; `pod delete` never `pod stop`; balance is the hard cap — do not enable auto-recharge |
| `--cloud-type` defaults to SECURE (~2× price) | Always explicit in `rp`; never create pods by hand |
| A6000 stock is Low in EU-SE-1 | A40 fallback in the *same* DC; volume stays valid |
| ~9 GB resident `StyleData` × workers → OOM kill | Filter pods for ≥32 GB RAM; keep `multiprocessing_context='fork'` |
| Loader becomes the bottleneck at 10–16× | Phase 2.2 measures it before any speedup is believed |
| Cloud numbers not comparable to historical local ones | Phase 2.4 re-baselines `run_e`/`run_f` on the pod |
| `_env.sh` bypassed → training completes, export dies | `setup.sh` ends with `_env.sh`; runner refuses to proceed if it fails |
| Community-cloud hosts are third parties | Data is public datasets + our own board photos; low sensitivity, named explicitly. A40 SECURE at $0.44 if that changes |
| Dataset provenance undocumented | Ship the cache, not the sources. Closing the provenance gap is separate work worth doing |

## Out of scope

Cache building, the replay harness, trial recordings, `truth.json`, and every
ship/no-ship decision stay local. This plan moves **training only**.
