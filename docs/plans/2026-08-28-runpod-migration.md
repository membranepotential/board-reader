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
- **Datacenter: `EU-RO-1`** *(revised 08-28 evening: `EU-SE-1` turned out not to
  support network volumes at all, and the A40 left it anyway — the A6000 is
  unreachable with a volume. EU-RO-1 carries two community GPUs beside a
  volume-capable DC; volume `board-reader` / `n102g25d6l`, 20 GB, created)*:
  - RTX 4090, 24 GB, **$0.34/hr** COMMUNITY, stock Low — primary
  - RTX PRO 6000, 96 GB, $1.69/hr COMMUNITY, stock Low — explicit escalation
    for VRAM-hungry runs only (never a silent fallback)
- Local baseline to beat: `run_g` = 2405 s, 170 img/s, `best_acc 0.8983`.

The 4090 (Ada) has native bf16 and ~2× the A6000's raw throughput at the same
price; 24 GB covers everything short of a large unfrozen backbone at 384 px,
which escalates to the PRO 6000 explicitly. Expected speedup **10–16×**, so a
`run_g`-equivalent lands at ~3 min and **~$0.02** — Phase 1.6 replaces this
estimate with a measurement.

## Phase 0 — local prep (no spend)

- [x] **0.1** *(done 08-28: all 11 styles, 8.4 GB — chessrender360 was a freeze casualty, rebuilt)* Complete the `pose_h1.9_r0.5_f1_maskgray` cache. It is missing 9 of
      11 styles (2.7 GB vs the 8.8 GB unmasked variant), so `run_h`/E7 member 2
      cannot run anywhere until it is built. Pure CPU — the local GPU is free
      meanwhile.
- [x] **0.2** *(done: `684507d`, 40-assertion suite)* Write `tools/train/cloud/rp` — the runner. Non-negotiable
      properties, in order: preflight (balance, no orphan pods, volume present) →
      create with `--cloud-type COMMUNITY` → **`trap … EXIT INT TERM` that deletes
      the pod** → `_env.sh` → run the config queue → pull artifacts → delete.
      Subcommands: `sweep`, `stage`, `reap`, `cost`.
- [x] **0.3** *(done: `684507d`)* Write `tools/train/cloud/setup.sh` — reproduces the venv recipe from
      `tools/train/README.md` on a fresh pod and ends by running `_env.sh`.
      Pin the CUDA wheel index to the pod image's CUDA, not the local cu13x.
- [x] **0.4** ~~Lower the account `spendLimit`~~ — void: `spendLimit` is a
      RunPod-set spend *rate* limit, not user-changeable (user, 08-28). The
      balance (no auto-recharge) is the safety net.
- [x] **0.5** *(measured during staging: ~5 MB/s sustained; 16.3 GB ≈ 55 min)* Measure home upstream and decide whether 7.9 GB goes up in one push
      or overnight. This sets the Phase 1 schedule, nothing else.

**Gate:** `rp` deletes its pod on `Ctrl-C`, on error, and on success. Test this
against a CPU pod (cents) before it ever guards a GPU.

## Phase 1 — beachhead: prove parity (~$0.60)

- [x] **1.1** *(done: `n102g25d6l`, grown to 30 GB — caches 16.4 G + on-volume venv ~5 G did not fit 20)* Create the network volume: 20 GB in `EU-SE-1` (**$1.40/month**,
      billed whether or not a pod is attached). Sized for
      `hr1.6_sp0.15_maskgray` (7.9 GB) + eval data (~0.1 GB) + run artifacts,
      with room for one more cache. Expand later rather than over-provision now.
- [x] **1.2** *(done via the S3 volume API instead — EU-RO-1 had zero CPU instances; see SKILL.md §3. Also staged data/external/chesscog_transfer, which evaluate.py reads as raw images, not cache)* Stage data on a **CPU pod** with the volume attached — no reason to
      pay for an idle GPU while 8 GB uploads. Ship
      `_cropcache/hr1.6_sp0.15_maskgray/`, `data/eval`, `data/uploads`,
      `data/images`, and the repo.
- [x] **1.3** *(done 08-28 evening — on a SECURE L4, not an A6000: EU-RO-1
      community stock was dry all session; venv persists on the volume)* Bring
      up a GPU pod, run `setup.sh`, confirm `_env.sh` passes.
- [x] **1.4** *(done: smoke_cloud.yaml)* Run the smoke config — proves the loop
      end to end for pennies.
- [x] **1.5** *(PASS: 0.8995 vs 0.8983, chesscog 0.860 — via run_g_cloud.yaml,
      styles pinned explicitly because "all" resolves in a different order on
      the pod and order changes batch composition)* **The parity gate.**
- [x] **1.6** *(measured: 628 img/s = 3.7×, 10.3 min per run_g vs 40.1 local —
      far under the 10–16× estimate; the Phase-2 grid explains why)* Record
      wall time.

**Gate:** `best_acc` within ~±0.005 of 0.8983. cuDNN kernel selection differs
across GPUs, so byte-identical is not the bar and never will be — but a larger
gap means something is wrong with the data or the geometry, not with cuDNN, and
must be chased before spending more.

## Phase 2 — tune the box, then make it routine (~$0.70)

- [x] **2.1** *(done 08-28 night, `rp bench` / `cloud/bench_grid.sh`, $0.29:
      every GPU knob is throughput-neutral — bf16/cl/fp16/compile/b128/b256
      all 624–644 img/s vs fp32's 640. The Turing result did NOT invert
      because the GPU was never the constraint; see 2.2)* Short-run benchmark
      grid.
- [x] **2.2** *(measured, `bench_loader.py`: no-op ceiling 637 (w4) / **737
      (w8)** / 606 (w16) / 581 (w24) on 6 vCPUs — the loader IS the wall;
      peaks ~1.3 workers/vCPU and degrades past it. ~110 img/s per vCPU is
      the scaling law: rent vCPUs, not GPU flops)* **Find the loader ceiling
      explicitly.**
- [x] **2.3** *(done — in `tools/train/README.md` "Cloud perf". DEVIATION: no
      `_cloud_base.yaml`: Config has no include mechanism, so a base file
      would be dead documentation; the single winner (`num_workers: 8`) is
      applied directly in run_i_cloud.yaml, amp/batch deliberately kept at the
      parity values since both are free)* Land the winners; document.
- [ ] **2.4** ~~Re-baseline `run_e`/`run_f` on the pod~~ — deferred: their
      *unmasked* caches are not on the volume (~17 GB / ~1 h more staging),
      and the comparisons that matter now (run_i vs run_g_cloud) are already
      same-hardware. Stage + run only when a cross-family cloud comparison is
      actually needed.

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
