---
paths:
  - 'packages/inference/src/board_reader_inference/stream/stabilize.py'
  - 'packages/inference/src/board_reader_inference/stream/schedule.py'
  - 'packages/inference/src/board_reader_inference/stream/ingest.py'
  - 'packages/inference/src/board_reader_inference/stream/rules.py'
  - 'packages/inference/tests/stream/test_stabilize.py'
  - 'packages/inference/tests/stream/test_schedule.py'
  - 'packages/inference/tests/stream/test_ingest_schedule.py'
  - 'packages/inference/tests/stream/test_rules.py'
---

# Read scheduling, the settle debounce and its guard

- Express every debounce window in wall-clock seconds plus a minimum read count, never as a number of reads, because the read rate varies more than 10x with load and schedule (0.9/s loaded vs 7.5/s replayed on one stream), so a K tuned at one rate is wrong at the other.
- Keep the settle coherence rule: a commit may not raise the settled board's excess (`rules.violation(...).total`); keep the outer guard `_excess(read) <= held_excess` before the `STALE_WINDOWS` escape, and keep that escape surgical (stop dragging as soon as the excess clears), because without the rule two squares settle on halves of two exclusive readings (two black kings on most boards of one take), and without the deadline a blocked square starves forever.
- Do not replace the unbroken-run quorum with a windowed majority or long-horizon vote counting, because both were scored on 24 recorded runs and lost on every pooled axis (fewer perfect boards, more flicker): the quorum is a quality filter, and a rule that settles more also settles noise.
- Keep a `FramePolicy` observe-at-submit, decide-at-drain: `FrameIngest.submit()` observes every arriving frame (a burst end is visible only from the whole run, and the latest-wins inbox hides most frames), and the worker asks `due()` exactly once, on the frame it picked, because `due()` answering True consumes the read.
- Give policies two floats (`capture_ts`, `stability`), never a header object, and coerce only in `schedule.stability_of` (null, NaN or missing → 1.0, "at rest"), because a recorded `FrameEntry.header` is a dict while a wire `Frame.header` is a method, and a NaN stability loses every comparison and can declare a move complete mid-hand.
- Validate policy numbers with `math.isfinite` and `> 0`, phrase timing checks as `elapsed >= interval`, re-anchor a backwards clock to `ts` instead of clearing `last_read_ts`, cancel owed burst-end reads when a new burst starts, and refuse `spacing_s > keepalive_s`, because each of these once switched the read floor off or turned it into every-frame reading (a NaN `keepalive_s` passes a bare `> 0` and inverts the limit).
- Handle a frozen edge clock at the ingest layer (the monotonic backstop at `BACKSTOP_SLACK` = 2x `keepalive_s`), never inside a policy, because a policy must stay capture-time for replay determinism, and 2x keeps `frames_backstopped` a real diagnostic instead of tripping on ingest latency.
- Format a policy's canonical name with ints as ints and floats via `repr` (trim a bare `.0`), never `%g`, because `%g` renders 1000000 as `1e+06`, which `parse_policy` refuses, and rounds two different policies into one name and one replay directory.
- Justify a scheduling change by GPU budget and display stability, never by accuracy, because `trial.py`'s guard window already drops the mid-move frames a burst-end policy skips, so skipping them cannot show up as a score. Note that `burst_end_reads=1` can never settle a move on its own (one read is no quorum).
- Keep the settle window strictly between the burst-end spacing and the run's span (`spacing_s < SETTLE_SECONDS <= (burst_end_reads - 1) * spacing_s`, shipped 0.3 < 0.5 <= 0.6), never at or below the spacing, because in the 2026-11-24 round-2 sweep (simulated tail) a window that two reads can span let a hand settle on the gate takes in 19 of 30 configs, a three-read window in 2 of 30, both at 0.2 s spacing (a 0.4 s span). Size those reads against the edge forwarder's tail (`tail_s`, 1 s): the run must fit inside it, or each read after the first waits for a 1 fps keepalive frame (~1 s each on the 2026-11-24 live take).
- Design a schedule change that depends on frames the recorded edge never sent (a forwarder tail, a new rate) with `scratch/experiments/latency2/sim.py`: it replays the recorded stream through the real `AdaptiveForwarder` and fills the old keepalive gaps with synthetic rest frames that borrow the nearest recorded frame's read; report numbers that rest on those synthetic reads as projected, not measured, because two synthetic frames can copy one hand frame's read and fake a quorum.
