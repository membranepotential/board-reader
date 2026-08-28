# Bank rematch — masked banks and full-data training both NO-GO; the shipped stack stands

_2026-08-29 · closes the rematch queued by the
[[2026-08-28-improvement-report|improvement report]] ("re-embed the bank from
masked crops, then rematch run_g/run_h"). Replay rows `e6f-maskedbank` /
`e7c-maskedbank` under `recordings/*/replays/`; log
`scratch/gates-rematch-20260829.log`._

## The verdict table (settled perfect / per-square, 4-take gate)

| take | **shipped** (run_e⊕run_f + live bank) | + masked bank | run_g_c⊕run_h_c + masked bank |
|---|---|---|---|
| 08-28 proof trial | **5/23 · 92.7** | 2/23 · 91.7 | 2/23 · 91.4 |
| 08-13 (q↔k lock) | **22/31 · 99.0** | 12/31 · 98.6 | 8/31 · 97.2 |
| 08-27 181802 | 3/43 · 91.2 | 3/43 · 91.1 | 3/43 · 90.6 |
| 08-27 175539 | 5/12 · 96.1 | 5/12 · 97.0 | 4/12 · 97.5 |
| **total perfect** | **35** | 22 | 17 |
| curated | 305/320 | **308/320** | ~305/320 |

**Ship nothing.** The deployed configuration — run_e ⊕ run_f, the live-grown
baseline bank, `BR_CROP_MASK=gray` — beats every challenger built this cycle.

## Why the masked bank lost takes it "should" have won

The rebuilt banks ([[calibration-and-the-bank#Rebuilding the bank offline
(`rebuild-bank`, 2026-08-28)|rebuild-bank]]) are guarded, corpus-clean, and
masked-domain-aligned — and they still lose 13 perfect boards to the baseline.
The reason is **membership, not masking**: the baseline bank's strength is the
exemplars auto-calibration grew *from the gate sessions' own frames* (768→1024
mid-trial on 08-28, and earlier live growth). Those frames' pixels exist only
in `recordings/`; `/calibrate` uploads never saw them, so no upload-corpus
rebuild can reproduce them. The comparison silently became "session-adapted
bank" vs "July-uploads bank" — and per-session adaptation is the bank's entire
job ([[calibration-and-the-bank]]: near-perfect *where it has been*).

The curated set inverts because it has no session dimension: there the cleaner,
larger corpus wins (305 → 308). This is the standing learning at bank scale:
**a curated gain is not a deployment verdict** — and it now has a sharper
corollary: *bank experiments must control for session membership before any
other variable is interpretable.*

## Full-data training (run_i) — also benched

`run_i_cloud` = run_g with true 663k-image epochs (Phase 3.1, $2.36):
synthetic holdout **0.8995 → 0.9283**, still climbing at epoch 20 — and
chesscog_transfer (unseen real photos) **0.860 → 0.828**. The wall-clock cap
was not what kept run_g honest: at 46× the data per epoch the model fits the
synthetic style mix at the expense of real-photo transfer. The
"synthetic-holdout gain transferred as 0.0pp" lesson from the geometry sweep
returns at full scale, now with a negative sign. `run_g_cloud` remains the
best OOD bbox member (0.860); all three retrained members stay benched.

## What this cycle actually bought

- The **cloud pipeline** (parity-proven, loader-bound-diagnosed, ~$3.30 total
  spend) — cheap, fast candidate production whose candidates now die in hours,
  not weeks.
- **`rebuild-bank`** with the shared live guard — which found 10 genuinely bad
  uploads and a live-path hole (`verify_labels` passes all-empty placements).
- Three sharpened negative results (masked bank, masked members, full data),
  each with a mechanism, not just a number.

## The actionable residue

1. **Bank provenance** (new): persist the source crop (or a masked+unmasked
   embedding pair) for every exemplar written — calibrate *and* auto-calibrate.
   Until then, no bank re-embed can ever be domain-migrated fairly, and the
   masked-bank question stays open rather than answered.
2. A session-fair rematch is possible *today* by rebuilding with the
   auto-calibrate frames mined from `recordings/` (frames + settled placements
   are both recorded) — the missing exemplars' sources exist, just not under
   `data/uploads/`.
3. The 6 repairable guard-rejected uploads (~+20% corpus) and the
   `verify_labels` all-empty hole (STATUS Open questions).
