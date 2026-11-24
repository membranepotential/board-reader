---
paths:
  - 'packages/inference/src/board_reader_inference/measurement/**'
  - 'packages/inference/tests/measurement/**'
  - 'tools/gate_table.py'
  - 'tools/migrate_labels_to_chess_frame.py'
  - 'tests/test_gate_table.py'
  - 'tests/test_migrate_labels.py'
  - 'docs/measuring-accuracy.md'
---

# Measuring accuracy: replay, the gate, trials, labels

## What to report

- Lead with the perfect-board rate and put the per-square rate next to it, never alone, because at 94.5 % per square a board averages ~3.5 wrong squares, which is an unusable position.
- Score the raw, prior and settled streams apart, never summed, because two streams can tie per square while their perfect-board rates differ completely (0/81 vs 20/81 on one take).
- When the per-square rate is near 98 % but the perfect-board rate is near 20 %, read the per-square residual breakdown before blaming the take, because that profile is the signature of one systematically wrong square.
- Quote a trial or gate number only from its `report.json` (`trial.run` always writes it), never from a chat message or a STATUS line, because `truth.json` is hand-authored, git-ignored and editable afterwards, and two headline numbers have already become unreproducible that way.
- Report a rank statistic (board-ranking AUC) next to any claim about the confidence gap between perfect and imperfect boards, and fit temperature leave-one-board-out (`calibration.leave_one_out_scaled`), because temperature scaling squeezes every confidence toward 1.0, which shrinks the gap while the ranking does not change, and a `T` fitted on the scored boards grades itself.

## What counts as evidence

- Gate every recognition, decode or settle change on the seven-take replay gate (`make gate GATE_NAME=<new>`), never on stills, the curated set or a single model's own eval, because rules that won on stills have scored zero in the gate, and gains of a single model did not carry over to the deployed stack. A stills win is the reason to run the gate, not to ship.
- Accept a post-hoc rule only if it wins in every gate take on its own with zero broken squares, because with ~100 errors a pooled net gain is easy to produce by fitting; requiring the win in every take was the only check against overfitting that held.
- Never quote a gate row for a candidate trained or fitted on frames from the gate takes (relevant to the optional LoRA, DESIGN M5), and keep every eval set disjoint from what a model was fitted on, because a candidate trained on the gate sessions once gained +20 settled-perfect boards while every uncontaminated signal stayed flat.
- Compare a replay's settled row only with runs at the same read rate, and judge settle changes by the errors that are not settle lag, because the debounce depends on reads per second, and on a short take with a move every 5 s the lag alone puts settled below prior per square no matter how good the debounce is.

## Replays and the gate

- For a refactor that must not change behaviour, diff each take's `replays/<name>/events.ndjson` against the base line by line on (`frame_seq`, `placement`, `prior.placement`, `settled.placement`) and the evidence digest, because that is stronger than `gate_table`'s totals. Delete intermediate replay dirs afterwards; keep `cq-v6` and `tkr-shipped` (the latter can only be rescored by `eval-session`, never replayed).
- Read `manifest.replay` (which recognizer was built) before diffing two replay dirs, because their env blocks can look identical while the built stack differed.
- Design settle and schedule changes offline: read every frame of a take once and cache (seq, `square_dist`), run `replay.replay()` with a memoized `FrameRead` per schedule (~1 s per take, byte-identical to a real replay), then re-run `BoardStabilizer` over that replay's prior placements per settle config. The sanity check: today's stabilizer must reproduce the recorded settled row of every `report.json` to the digit.
- Replay the held-out takes 163352, 170333 and 172736 with `--restamp-stability`, because their motion meter was broken and every schedule collapses to the keepalive otherwise.

## Truth files and labels

- Remember that `guard_seqs` counts frames, not seconds: at 8 fps a guard of 12 hides ~1.4 s after each truth boundary, at 2 fps ~6 s, and it removes a much larger share of a sparse policy's reads. When an old recording's settled stream has to be derived, feed `BoardStabilizer` every event, including the guard-excluded ones, because the live gateway saw those too.
- Author `truth.json` placements by pushing SAN moves with python-chess from the last trusted segment, and find boundaries from ChessQ's placement runs over every frame (long runs are positions, 1-frame runs are hands) checked by eye on a contact sheet, because the result is then legal by construction. Keep `from_seq` strictly increasing; plies with no event between them collapse into one segment. A segment that changes two pieces of different colours is a merged-move bug.
- Check every boundary of an old take against ChessQ's per-frame runs, not just its tail: when the new position is read before a segment's `from_seq`, the boundary is late (five such boundaries and two truncated files were found in the three held-out takes on 2026-11-24), and a truncated file silently scores a stalled settled board as correct.
- Split an old recording on its last `seq` restart before measuring anything per frame, because seq is not globally monotonic there (20260827-175539 holds a 48-frame false start, then restarts at 0), so keying by seq drops one stream.
- Treat labels saved with `corrections: 0` as an upper bound, not independent truth, and read the position off the board or the source video before you look at the overlay, because any square where the operator and the model are both wrong then scores as correct.
- Score a recognizer only on chess-frame labels (`evaluate.scorable`; a label file without `frame` is corner-frame, it predates D10), and keep `/estimate` writing the read's frame into the upload sidecar, so `/labels` labels in the frame the read was made in, not whatever serves at label time. `labels.best_rotation` is the one frame diagnostic.
- Never resolve board orientation by square-colour parity or printed border coordinates, because players set boards up ignoring the printed coordinates and non-FIDE boards exist (both seen in one session); use chess-content plausibility or the recognizer.
- Check the board count an eval reports against what its label files claim before you compare with a past number, because `labels.load_labels` skips labels whose image is gone and skips hidden directories, without an error. Keep label backups in a hidden directory (`.pre-chess-frame/`) for exactly that reason: `rglob` once found a dot-directory copy first and scored the curated set against the old labels.
- Check every item when a label step has a per-item choice made by eye (orientation, which side is White, a truth boundary), and report a number per item, because a spot check once missed three of fourteen broken boards and the set was trusted for weeks.
- When a test turns a placement by quarter turns, remember that one `np.rot90` is counter-clockwise (the new a8 is the old h8) and corners follow by `np.roll(corners, -k)`.
