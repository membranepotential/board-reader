# Plan — a PGN-building client in the viewer (2026-11-24)

**Goal (user, 2026-11-24):** the live page turns each board's position stream into a
valid chess game, shows the PGN as it grows, copies it to the clipboard and opens it in
Lichess analysis.

This is DESIGN §8 WP-H3, the optional reference move reconstructor of the application
layer. It does not change D1: the CV layer still streams positions only. The decoder is
a client of `/ws/viewer` in `packages/viewer/src/lib/game/`, and the positions store
(`lib/stores/positions.svelte.ts`) keeps no history of its own.

## What ChessCam actually does

ChessCam's decoder lives in its engine package
[Pbatch/ChessCamRecorder](https://github.com/Pbatch/ChessCamRecorder)
(`src/vision/bayesianTracker.ts`, `src/vision/beamTracker.ts`, read on 2026-11-24). The
"Bayesian" label is accurate, but the algorithm has two layers:

1. **`BayesianTracker`: a forward filter (a sum over hypotheses).** It starts from the
   last position that is final. The hypotheses are "no move", every legal move, and every
   legal move plus a reply. On each frame, a little of the "no move" share moves to the
   moves (`moveChance` 0.01 per 250 ms). Then each hypothesis is weighed by the frame's
   likelihood on the squares it changes. That likelihood is ChessQ's per-square
   probability, softened by a temperature (3.0), mixed with a uniform floor (0.05, so a
   hand over one square cannot rule out a move) and raised to an evidence weight (2.5 per
   250 ms, scaled by the time between frames). A move is shown once its share passes
   0.95. It is played for good once the move *with a reply* passes 0.95, so the reply
   confirms it. Promotions are grouped and the queen wins a tie. A small `flipChance`
   covers a board read the wrong way round.
2. **`BeamTracker`: a beam over histories (Viterbi-like).** When a tracker plays a
   move, the history splits: one copy plays it and the other does not. The best 4
   histories survive (margin 37.5 per unit of evidence weight, 10 s horizon), histories
   that reach the same position merge and keep the better one, and moves that every
   history shares become final. This lets a wrong move be taken back.

The README reports 39 of 43 held-out blitz games perfect. That was measured on per-frame
reads at 4 fps from the app's own model.

I found no other published camera-to-PGN decoder with a temporal model; a short web
search turned up only per-image recognizers (for example
[End-to-End Chess Recognition](https://arxiv.org/abs/2310.04086)). CameraChess (see
[2026-11-24-chesscam-camerachess.md](2026-11-24-chesscam-camerachess.md)) scores
one step against a fixed previous position, which cannot recover from a wrong move.
Unverified beyond that.

## The method chosen: beam Viterbi over legal positions, per event

The hidden state is a legal chess position. The decoder uses
[chessops](https://github.com/niklasf/chessops), the Lichess team's library, for move
generation, SAN and PGN. It keeps a **beam** of hypotheses. Each one is a position, the
move path that reached it from the start, and a log score.

- **Steps are position events, not frames.** `/ws/viewer` publishes about one event
  every 2 s (the keepalive) plus bursts around motion. In the gate takes (fast play, about
  3 s per move), two plies often fall between two consecutive events, and some moves are
  never visible in any event. A per-frame time model like ChessCam's does not fit that
  input, so each event is one HMM step with fixed priors.
- **Transitions** from a position: stay, one legal ply, two legal plies (a move the
  reads missed), or undo the last ply (a takeback). The priors are fixed per event, and a
  move's prior is shared evenly among the legal moves. Underpromotions get a small extra
  penalty, so a queen wins when the reads cannot tell the pieces apart.
- **Emission:** the sum over the 64 squares of `log p(label)`, where `p` is the event's
  `square_dist` softened by a temperature and mixed with a uniform floor. The floor keeps
  one occluded or misread square from vetoing the right position. An illegal-looking read
  has no special case: no legal position explains it well, so the best one wins by a
  small margin and the next clean read settles it.
- **Viterbi, not a forward sum.** Candidates that reach the same position merge and keep
  the best path (max-product). The output is one move sequence, and the max keeps the
  path that explains the reads, which is the question asked. ChessCam's beam layer makes
  the same choice.
- **Commit (fixed lag, driven by the evidence):** a move is committed once every
  hypothesis within the margin of the best shares it, which is the common prefix of their
  paths. A decisive read commits at once; an ambiguous one waits. The page shows the best
  path, with the moves not yet committed marked as provisional.
- **Start:** the standard initial position. A stream that starts mid-game is out of
  scope; the page then shows how many squares the read disagrees with the game, and the
  user starts a new game once the board is set up. A FEN start is a follow-up.
- **Hands and status:** only `status: ok` events are decoded. Hands need no special case
  beyond the floor: the node already settles, and a read with a hand on the board
  explains no position well.
- **Move times:** a move is timed by the read that first took it. If a later event's
  settled `placement` shows the same board with an earlier `changed_utc`, the move takes
  that earlier time, because `changed_utc` counts reads that were never published.

## PGN and Lichess

- The PGN is written by chessops `makePgn`, with `[Date]`, `[UTCDate]` and `[UTCTime]`
  from the first decoded read. Each move gets a `{ [%emt h:mm:ss] }` comment with the time
  since the previous move. Lichess reads `%emt` (and `%clk`) when it imports a PGN
  (`lila/modules/study/src/main/CommentParser.scala`). An OTB game without a clock has no
  remaining time, so `%clk` would be a lie.
- The link is `https://lichess.org/analysis/pgn/<movetext>`. Lichess URL-decodes the
  path, turns `_` and `+` into spaces and parses it with chessops `parsePgn`
  (`lila/app/controllers/UserAnalysis.scala`, `ui/analyse/src/pgnImport.ts`). That route
  keeps only the moves, so the link carries the movetext alone, with spaces as `_`.
  Lichess reads the first 5000 characters of the path, which is about 700 plies.
  Clicking the link is the only thing that sends a game to Lichess; the code uploads
  nothing.

## Measuring it

`tools/pgn_eval.ts` runs the same decoder modules over the seven gate takes' replayed
events (`recordings/<take>/replays/<name>/events.ndjson`) and compares the result with
the move list derived from `truth.json`. Recorded events keep only the top two classes
per square (`evidence`), not the full `square_dist`. The harness rebuilds a distribution
from them: the two named probabilities, with the rest spread evenly over the other 11
classes.

Results on 2026-11-24, replay `guard-only` (the served schedule), default parameters.
Latency runs from the truth segment's first frame (just before the piece is lifted) to
the read that committed the move:

| take | reads | plies right / true | latency s (median / max) | what the take holds |
|---|---|---|---|---|
| 20260813-193308 | 65 | 5/5 | 0.8 / 1.6 | slow opening |
| 20260827-175539 | 26 | 14/14 | 3.5 / 5.0 | fast: two plies often between reads |
| 20260828-111421 | 52 | 14/14 | 2.1 / 3.4 | one truth segment holds two plies |
| 20260827-181802 | 74 | 19/19 | 3.1 / 4.7 | Taimanov Sicilian, captures, O-O-O |
| 20260827-163352 | 26 | 7/7 | 2.8 / 4.3 | slow opening |
| 20260827-170333 | 70 | 7/7 | 3.2 / 4.4 | `1.e4` taken back, then replayed |
| 20260827-172736 | 133 | 14/14 | 3.5 / 5.0 | hard take |

In total, 80 of 80 plies match and 7 of 7 games are exact. All seven takes start from
the initial position, and none holds a promotion, an en passant capture or a hand-held
pause long enough to test the mismatch note. Every take still decodes exactly with
margins from 8 to 25, temperatures from 1 to 4, width 3, floor 0.01 or 0.2, and with
replays `sp03-settle05-7` and `cq-v6`. A margin of 6 fails: in 20260827-181802, where
`3.d4 cxd4` fell between two reads, the beam drops the right path and decodes `3.d3`.
Seven takes and 80 plies are a small sample, so these parameters are a start, not a
tuned result.

Cost: a median 1.5 ms per read on a desktop, 45 ms at the 95th percentile, and 340 ms at
worst. The spikes come when new positions enter the beam and their two-ply successors
are generated. A branch-and-bound on the two-ply successors (see `#expand`) cut the
95th percentile from 170 ms without changing any result.
