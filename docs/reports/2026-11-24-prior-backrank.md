# What the chess prior is worth under ChessQ: one settled board in 361 (2026-11-24)

## Outcome

`stream/prior.py` serves three uses. Under ChessQ V6 they are worth this much on the
seven-take replay gate, against the base `sp03-settle05-7`:

| use | what it does | effect when removed |
|---|---|---|
| 1. the per-read decoder (`prior.decode`, `PositionEvent.prior`) | repairs an impossible read | repaired 2 of 446 reads. Prior perfect equals raw perfect on every take, with or without it. |
| 2. the stabilizer reads the repaired board (`prior.placement`), not the raw one | feeds the debounce | settled perfect 315 → 314 of 361: one castle settles one read (about 2 s) later. |
| 3. the stabilizer's guard (`stabilize._excess`, `prior.violation`) | refuses a settle that makes the board more impossible | 3 impossible settled boards published (two kings twice, two queens once). Settled perfect does not change. |

**Recommendation:** 1 and 2 can go, at the cost of the one board above. Keep 3: it is
the only thing that keeps impossible boards off the settled stream, and it costs
0.02 ms per read. If `prior.py` goes, its count constraints (`tally`, `violation`)
move into `stabilize.py` for the guard.

**The back-rank-pawn constraint** stays off. The lead's gate showed it adds nothing:
the prior column equals raw on all seven takes. The decoder never even saw a pawn on
rank 1/8 here. On a legal board the constraint cannot fire, because a legal board has
no pawn on rank 1/8, and a promoted piece there is not a pawn.

## Method

- Base: `sp03-settle05-7` (raw 341/361, prior 341/361, settled 315/361).
- Two replays of the seven gate takes, run like `make gate` (fp32 V6 on the GPU, the
  `serve` schedule, the same `--restamp-stability` takes). A scratch runner
  monkeypatched the read path; no product code changed:
  - **A, `prior-off`**: `ingest.decode` returns the raw argmax with no flips, so the
    stabilizer consumes the raw board. The guard stays.
  - **B, `prior-off-noexcess`**: A, and `stabilize._excess` returns 0 (no prior at all).
- Each replay's events were compared to the base by `(seq, frame_seq, capture_ts)`.
  Both variants read the same 446 events with byte-identical raw placements, so every
  difference below comes from the prior.
- Leaks use the settle-tuning report's rule: a settled square whose class is in
  neither the current, the next nor any earlier truth segment.
- The decoder's cost was timed in the A replay: the real `decode` ran on every read
  and its answer was thrown away.

## Results

Settled perfect per take (raw and prior perfect are identical in all three runs):

| take | raw perfect | base settled | A: no decoder | B: no prior at all |
|---|---|---|---|---|
| 20260813-193308 | 42/47 | 46/47 | 46/47 | 46/47 |
| 20260827-175539 | 15/15 | 12/15 | **11/15** | **11/15** |
| 20260828-111421 | 31/34 | 31/34 | 31/34 | 31/34 |
| 20260827-181802 | 57/60 | 51/60 | 51/60 | 51/60 |
| 20260827-163352 | 21/22 | 14/22 | 14/22 | 14/22 |
| 20260827-170333 | 62/63 | 56/63 | 56/63 | 56/63 |
| 20260827-172736 | 113/120 | 105/120 | 105/120 | 105/120 |
| **total** | 341/361 | 315/361 | 314/361 | 314/361 |
| leaks | | 0 | 0 | 0 |
| impossible settled boards (of 446 events) | | 0 | 0 | 3 |

### What the decoder repaired

Two reads in 446, both during black's or white's castling, both `piece_count` (17
pieces of one colour):

- **175539, frame 221** (in a truth guard, a hand on the board): raw read e8 as a
  rook beside the castled rook on f8. The decoder emptied e8.
- **172736, frame 565**: raw read e1 as a queen next to the king on g1. The decoder
  emptied e1.

### Every settled board that changes

| take, frame | base | A | B | why |
|---|---|---|---|---|
| 175539, 238 | castled, correct | `rnbqk3/…`, e8 still the king, f8 and g8 pending | `rnbqkrk1/…`, **two black kings** | Without the repair at 221, e8 has only one "empty" read by 238, so it settles one read later (correct again at 255). In B, f8/g8 commit without e8. |
| 172736, 581 | `…/RNBQ2KR`, wrong (king g1, rook still h1) | `…/RNBQK2R`, wrong (the board before castling) | `…/RNBQK1KR`, **two white kings** | Both base and A are wrong here and both settle correctly at 593; A shows the previous true board instead of a board that never existed. |
| 172736, 1397 | `rnb1r1k1/ppppqppp/…`, wrong | same as base | `rnbqr1k1/ppppqppp/…`, **two black queens** | The queen returns from e7 to d8. The guard holds d8 back until e7 clears; without it d8 commits first. |

Only 175539 frame 238 is a scored board that flips between right and wrong. That is
the whole 315 → 314.

### Cost

| step | per read |
|---|---|
| `prior.decode`, all 446 reads | median 0.105 ms, p90 0.15 ms, p99 0.25 ms, max 7.5 ms |
| `prior.decode`, the 2 repaired reads | 6.2 ms and 7.5 ms (the full 64×12 greedy search) |
| `stabilize._excess` (the guard) | 0.02 ms |

A ChessQ read takes about 200 ms on the GPU, so neither cost matters.

## Not measured

- `/estimate` and the snapshot page show the repaired board
  (`response.prior?.placement`). Without the decoder a snapshot could show two kings.
  `eval` on the labeled stills (`--prior` versus `--no-prior`) would measure how often;
  it was not run here.

## Follow-up: the decoder removed, the guard kept (same day)

The user took the recommendation (DESIGN D14): `prior.py` and the wire's `prior`
block are gone, the stabilizer settles the raw read, and the guard's count
conditions live in `stream/rules.py`, counted on a python-chess board. A new
per-board setting, `rules: standard | free` on `/control`, turns the guard off for a
board with extra material or a variant.

**Final gate** (`make gate`, fp32 V6, the `serve` schedule), against the base
`sp03-settle05-7`:

| take | raw perfect | base settled | `guard-only` (standard) | `guard-free` (free) |
|---|---|---|---|---|
| 20260813-193308 | 42/47 | 46/47 | 46/47 | 46/47 |
| 20260827-175539 | 15/15 | 12/15 | 11/15 | 11/15 |
| 20260828-111421 | 31/34 | 31/34 | 31/34 | 31/34 |
| 20260827-181802 | 57/60 | 51/60 | 51/60 | 51/60 |
| 20260827-163352 | 21/22 | 14/22 | 14/22 | 14/22 |
| 20260827-170333 | 62/63 | 56/63 | 56/63 | 56/63 |
| 20260827-172736 | 113/120 | 105/120 | 105/120 | 105/120 |
| **total** | 341/361 | 315/361 | 314/361 | 314/361 |
| leaks | | 0 | 0 | 0 |
| impossible settled boards (of 446 events) | | 0 | 0 | 3 |

Raw per square is .9982 in all three. All 446 raw placements equal the base's.
`guard-only` reproduces variant A above on every one of the 446 settled boards, and
`guard-free` reproduces variant B on every one: the product code does what the
scratch runner predicted. `guard-only` is the new gate base (`GATE_BASE`).

**Extra impossibility checks, measured.** Three more conditions hold in any position
of a standard game whoever is to move, so the guard could count them: (a) both kings
in check at once, (b) no side to move is possible because of the checks (a king
attacked by three or more pieces, or by two that no single move can have uncovered:
python-chess's `TOO_MANY_CHECKERS` and `IMPOSSIBLE_CHECK`, with `OPPOSITE_CHECK`, set
for both sides to move), and (c) a pawn on rank 1 or 8 (safe now, the frame is known).
A double check is legal and is not one of them. Each was counted on the raw and the
settled boards of every replay of the seven gate takes, and the stabilizer was re-run
offline over the base's 446 raw reads with each one added to the guard:

| | (a) both kings in check | (b) impossible checks | (c) pawn on rank 1/8 |
|---|---|---|---|
| truth segments it fires on | 0 | 0 | 0 |
| ChessQ raw reads it fires on (3252, all ChessQ replays) | 0 | 0 | 0 |
| ChessQ settled boards it fires on (3252) | 0 | 0 | 0 |
| settled perfect with it added to the guard | 314/361 | 314/361 | 314/361 |
| settled boards it changes | 0 | 0 | 0 |
| older-stack raw reads it fires on (14937) | 6 (2 not caught by the counts) | 0 | 3035 (107) |
| older-stack settled boards it fires on (14937) | 0 | 0 | 1070 (922) |
| of those, true boards | 0 | 0 | 0 |

Under ChessQ none of the three ever fires, so none would block anything, and all
three stay out (the keep rule: block something wrong, never a true board). The
older stacks' numbers show the checks are not vacuous, and that (c) would have
mattered for them; those replays partly predate the chess frame, so they are no
evidence for the deployed reader.

