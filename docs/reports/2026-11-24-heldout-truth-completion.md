# Truth completion for the three held-out takes (2026-11-24)

20260827-170333 and 20260827-172736 had truth files that stopped early: the live event
stream stalled at seq 478 and 884, so the operator never labelled the moves after that.
A ChessQ V6 read of every frame (the cached reads in `scratch/experiments/settle-sweep/`)
showed long stable runs past those points; each new move was checked by eye on JPEG
contact sheets (`scratch/experiments/truthfix/sheets/`, gitignored).

- **Added:** 170333 3...Bc5 (497), 4.O-O (666); 172736 6.d3 (897), 6...Qe7 (1048),
  7.Nc3 (1241), 7...Qd8 (1369).
- **Late boundaries moved** to the first frame with the hand on the moving piece (the new
  position was read before the old `from_seq`): 163352 a6 228→222, Ba4 240→235; 172736
  4...O-O 692→674, Re1 760→751, Re8 819→805. 163352 was complete; its game ends at 4.Ba4.
- Every placement was pushed with python-chess; a separate check confirms each segment is
  one legal move from the previous one (except 170333's documented reset to the start at 134).

**Rescore** (raw perfect, `cq-v6` replays): 170333 18/28 → 26/26, 172736 29/50 → 50/52,
163352 unchanged. One drop is real: on 170333 `s0-everyframe` the settled board stalls on
3.Bc4, which the old, shorter truth had scored as correct.
