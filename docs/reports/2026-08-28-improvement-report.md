# Improvement report — the 08-28 retrain program, closed

_2026-08-28 late · closes docs/reports/2026-08-28-retrain-program.md (E1–E7 +
the cloud continuation). Every replay number is replay-vs-replay against the
frozen controls with the pinned 08-28 bank (`bank-baseline-20260828.npz`)._

## What shipped

**Serve-time occlusion masking is now the deployed default** (`BR_CROP_MASK:
gray` in compose.yaml): pixels inside a *closer* neighbor's bounding prism are
gray-filled before the crop reaches the classifier **and the bank embedder**.
Models unchanged (run_e ⊕ run_f). This is E6's configuration, promoted after it
beat every alternative on the 4-take gate.

### The 4-take settled gate (perfect boards / per-square)

| take | control | **e6f-gray (shipped)** | e7 masked-retrained pair |
|---|---|---|---|
| 08-13 (q↔k-locked) | 6/31 · 97.5 % | **22/31 · 99.0 %** | 0/31 · 96.4 % |
| 08-28 proof trial | 5/23 · 93.3 % | 5/23 · 92.7 % | 2/23 · 91.2 % |
| 08-27 181802 | 3/43 · 91.4 % | 3/43 · 91.2 %¹ | 3/43 · 90.9 % |
| 08-27 175539 | 5/12 · 96.5 % | 5/12 · 96.1 %¹ | 4/12 · 96.4 % |
| **total perfect** | **19–20** | **35** | **9** |

¹ e6-gray rows (same config, pre-review-fix run; e6f rerun confirmed the fixed
masking is byte-identical on the takes it re-scored).

The headline: **perfect boards nearly double** (≈20 → 35 of 109), driven by the
08-13 take where masking unlatches the d8 q↔k bank lock (6 → 22 of 31), at a
sub-point per-square tax on the other takes. Curated set under the shipped
config: **305/320** (baseline 304/320 unmasked, pinned bank) — no tax there
either; the masked-retrained cascade scored 303/320.

## What the cloud built, and why it is benched

The E7 thesis — retrain the members on masked crops so training matches masked
serving — was completed on RunPod and **failed the deployed gate**:

| candidate | holdout (vlab) | chesscog_transfer (unseen real) |
|---|---|---|
| run_e (baseline) | 0.898 | 0.819 |
| run_g = masked run_e (local) | 0.8983 | 0.851 |
| run_g_cloud (parity rerun) | 0.8995 | 0.860 |
| run_h_cloud = masked run_f | 0.8944 (run_f: 0.894) | 0.860 |

Member-level, masking training is a clean win: **+3–4 pp on unseen real
photos, no holdout cost, for both geometry families.** Deployed in the cascade,
the pair loses every take it was meant to win (table above), masked or unmasked
at serve. Diagnosis from the replays:

- **E6's d8 win is a bank-side effect, not a classifier effect.** Masked crops
  shift the DINOv2 embedding enough that the bank's nearest-exemplar lookup
  unlatches from the wrong exemplar. The retrained classifiers re-assert the
  q→k read (raw stream: 28/31 events), so better members do not compose with a
  bank-dominant (30/70) blend.
- ~~**New defect found: settled coherence doubles the error.** Raw d8 q→k plus
  a correct e8 king reads as "two kings"; the coherence layer demotes the
  *real* king, turning one wrong square into a symmetric q↔k swap on every
  event (settled 31+31). Filed as follow-up: prefer demoting the
  lower-confidence of duplicate kings.~~ **Refuted same night**
  ([[2026-08-28-two-kings-decode]]): the decoder already demotes the
  lower-confidence king and is Pareto-best over 4 rules × 4 takes; the 31/31
  figure belongs to this table's *rejected* e7 row (the shipped config splits
  17/17), and the confidence evidence genuinely favors the wrong square — the
  lever is the bank lock on d8, not the tie-break.

run_g/run_h are kept as **OOD assets** (`scratch/runs/run_g*`, `run_h_cloud`):
the +4 pp generalization is real and waits on bank-domain alignment (re-embed
the bank from masked crops) and the full-data retrain (run_i) before a rematch.

## The cloud pipeline itself (the enabling improvement)

- `tools/train/cloud/rp` — trap-guarded RunPod runner (sweep / stage / adopt /
  reap / cost), live-debugged: ports published at create, exit codes propagate
  (`setsid --wait`), chown-free rsync, endpoint retries, artifact pull incl.
  weight sidecars. 40-assertion stub suite.
- **Staging is pod-free** over RunPod's S3 volume API (17.6 GB up at ~5 MB/s);
  EU-RO-1 volume `board-reader` (30 GB) holds both masked caches + eval data +
  the on-volume venv, so a fresh pod provisions in ~2 min.
- **Training resolves styles from the caches alone** (no raw datasets on the
  pod) — meta-backed style registry + symlink-following discovery, 9 tests.
- **Parity proven**: cloud run_g 0.8995 vs local 0.8983 (gate ±0.005), same
  chesscog within noise. **Measured** L4 throughput 628 img/s vs 170 local
  (3.7×; 10.3 min per run_g vs 40.1) — before the Phase-2 amp/batch tuning that
  Ada-class cards actually reward.
- Total spend, including every debugging pod: **~$0.66** of $7.91.

## Where this leaves the residual

08-13's lock is broken; the proof-trial residual (P→empty / N→P clusters — the
drift-and-bleed family) and the curated 16/320 stand. The levers, in order:
Phase-2 cloud tuning then full-data run_i (46× more data per epoch, ~$3),
bank re-embedding under masking for the run_g/run_h rematch, and the
two-kings coherence fix.
