---
aliases: [viterbi, legality decoding, auto-labeling algorithm]
tags: [board-reader/ideas, board-reader/data, board-reader/recognition]
---

# Viterbi decoding — how legal chess turns a 92 % reader into a labeling machine

_Companion to [[ideas#A. Legality-constrained auto-labeling of game video (the data engine)|ideas §A]].
The engine this describes is **planned, not built** (plan in progress →
`docs/plans/2026-08-31-autolabel-engine.md` when it lands). This note explains the
algorithm itself and why it fits this problem unusually well._

> [!note] Written for the crop stack (2026-08-31)
> The "92 %" is the removed crop classifier. ChessQ reads ~99.8 % per square on the labeled
> stills and settles 104/119 gate boards perfectly ([[measuring-accuracy]]), so the decode
> has less to fix; the algorithm is unchanged. Its stage 0 was measured NO-GO on the
> corpus of the time ([[2026-08-31-autolabel-stage0]]).

## The problem it solves

The reader estimates each frame independently at ~92–96 % per-square accuracy
([[measuring-accuracy]]). On a 64-square board that means a typical frame carries
2–5 wrong squares — good enough to watch, not good enough to *train on*: labels
at 95 % teach the next model the current model's mistakes.

But a recorded **game** is not a bag of independent frames. Between two
consecutive settled positions, exactly one legal move happened (or none). That
constraint is brutally informative: from any position there are ~30–40 legal
moves, while the reader's per-frame hypothesis space is effectively $13^{64}$
placements. Almost every misread the estimator produces is *unreachable* — no
legal move takes the previous position to it. The sequence knows things no
single frame can.

The Viterbi algorithm is the standard way to cash in that knowledge: it finds
the **single most probable sequence of hidden states** given noisy per-step
observations and a transition structure. Here: the most probable *game* given
the reader's per-frame confidences and the rules of chess.

## The algorithm in general

Viterbi decodes a hidden Markov model. You have:

- hidden states $s$ — what was actually true at step $t$ (unobservable),
- observations $o_t$ — what you measured at step $t$,
- an **emission model** $B_s(o_t) = p(o_t \mid s)$ — how likely the measurement
  is if $s$ were true,
- a **transition model** $A(s' \to s)$ — how likely state $s$ follows $s'$.

The most probable full path $\arg\max_{s_1..s_T} p(s_1..s_T \mid o_1..o_T)$
naively requires scoring every possible sequence — exponential in $T$. Viterbi
observes that the best path to state $s$ at step $t$ must extend the best path
to *some* state at step $t{-}1$ (optimal substructure), so dynamic programming
suffices:

$$\delta_t(s) = \max_{s'} \big[\, \delta_{t-1}(s') + \log A(s' \to s) \,\big] + \log B_s(o_t)$$

Keep a backpointer to the argmax $s'$ at every cell; at the end, take the best
final state and walk the backpointers backwards. Cost: $O(T \cdot S \cdot d)$
where $d$ is the number of predecessors per state — for chess, $d \approx$ the
legal-move count, not $S$. Crucially, a locally terrible observation cannot
force a locally attractive but globally unreachable state: the path *through*
it would score worse than the path around it.

## The mapping onto this project

| HMM piece | board-reader instantiation |
|---|---|
| hidden state $s$ | the true position (a full 64-square placement) |
| step $t$ | one settled read — burst-end scheduling already reads once per move, so steps ≈ plies |
| observation $o_t$ | the reader's per-square distributions (`square_dist`, 13 classes × 64 squares) |
| emission $\log B_s(o_t)$ | $\sum_{sq} \log \texttt{square\_dist}[sq][\,s[sq]\,]$ — squares are conditionally independent given the position |
| transition $A(s' \to s)$ | uniform over the legal moves of $s'$, plus a self-loop ($s = s'$, no move happened between reads) |
| decoded path | the reconstructed game — every frame gets its position as a **label** |

The emission side is the part we already have: the estimator's whole output
([[recognition-pipeline]]) *is* $\log B$. The transition side costs nothing to
get right — `python-chess` generates legal moves exactly.

> [!example] Why this kills the Q→K defect specifically
> The 08-31 batch's dominant error ([[2026-08-31-new-style-batch]]) is the
> queen's home square reading as a second king. Per-frame, the decoder must
> guess which "king" to demote, and the evidence genuinely crosses — the
> estimator simply believes the wrong square more
> ([[2026-08-28-two-kings-decode]] showed no per-frame tie-break can fix this).
> In a *sequence*, the question dissolves: the position where the queen still
> stands on d1 was observed twenty plies ago, and no legal continuation ever
> put a second king there. The path that keeps d1 = queen dominates every path
> that doesn't. Information that is not in the frame **is** in the game.

## What makes chess-Viterbi non-textbook

Three deviations from the classroom version, and they are where the planned
engine's actual design work lives:

1. **The state space cannot be enumerated.** $S$ is astronomically large, so
   there is no $T \times S$ table. Instead: **beam search** — carry the top-$k$
   scoring positions at each step, expand each by its legal moves + self-loop,
   score, prune back to $k$. The start position anchors the beam (games start
   from a known state, or from a hand-confirmed first read). This makes it
   approximate; the bet is that the true position never falls out of a
   reasonably wide beam, because the emission score of the truth is almost
   always near the top even when its argmax is wrong.
2. **Step alignment is fuzzy.** Reads do not perfectly coincide with plies:
   a move can slip between reads (hence the self-loop), a hand can occlude a
   read mid-move, framing drift can inject garbage stretches. The recorder
   keeps every read with timestamps, and the existing truth-authoring machinery
   already solved a cousin of this problem (monotone alignment of a placement
   sequence against `events.ndjson` with a small look-ahead window) — the
   engine reuses that shape rather than assuming read = ply.
3. **Label lock-in is the failure mode that matters.** Viterbi always returns
   *some* path; a confidently wrong path would mass-produce confidently wrong
   labels and train the next model on them. So decoded games are only trusted
   through gates: agreement with the hand-authored `truth.json` takes we
   already score ([[measuring-accuracy]]), PGN cross-checks where the game was
   also recorded conventionally, and per-step posterior margins (how much the
   winning path beats the runner-up) as a per-label confidence to threshold on.

## The layering irony

DESIGN D3 forbids the **live** system from being a temporal consumer: the CV
layer streams stateless per-frame positions, and move logic belongs downstream.
This engine deliberately builds exactly that forbidden consumer — *offline, at
labeling time*, in `tools/`, never imported by `packages/`. The live product
stays a stateless estimator; Viterbi is how we mine its recordings for training
data. Same math a downstream move-tracker would use, pointed backwards at our
own footage instead of forward at a user's game.

## Why this is the scalable lever

Every real game played becomes a labeled dataset for free — frames + settled
placements are already recorded per session ([[ideas#C. Own-recordings flywheel|ideas §C]]).
Hand-labeling a board costs minutes per frame; Viterbi labels a whole game at
once and its quality *improves as the reader improves*, because better
emissions tighten the decode. That closed loop — reader labels data, data
trains reader — is the standard self-training trap, except the teacher here is
not the model's own beliefs but the **structure of chess**, which never
overfits.

## Related

- [[ideas]] — §A is the engine this math powers; §B (PGN-aligned broadcasts) is
  the special case where the transition sequence is *known*, not decoded
- [[2026-08-28-two-kings-decode]] — the proof that per-frame decoding hit its
  information ceiling, which is the argument for sequence decoding
- [[measuring-accuracy]] — the truth/eval machinery the decoded labels must be
  gated against
- [[recognition-pipeline]] — where `square_dist`, the emission model, comes from
