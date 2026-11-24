# CLAUDE.md — board-reader

Camera-based chess **board reader**: a CV system that reads live OTB positions and
streams **timestamped piece placements + per-square confidences** (not moves, not
legality — those are the downstream application layer's job).

**Read first:** [STATUS.md](STATUS.md) (where the project is) ·
**[docs/DESIGN.md](docs/DESIGN.md)** (decisions — authoritative) ·
[docs/index.md](docs/index.md) (how it currently behaves — architecture, recognition
pipeline, measuring accuracy, operating envelope, runbook).

## Load-bearing principles (do not violate without updating DESIGN.md)

1. **User-testable from the start.** Every milestone must be runnable end-to-end by
   a person and *observable* — `docker compose up` → open the viewer → see it work.
   We never land a milestone that can't be demoed live. New behaviour ships with
   both automated tests **and** a path in [docs/TESTING.md](docs/TESTING.md). The
   walking skeleton (M0) exists precisely to make this true from day one.
2. **Everything is tested.** No feature merges without tests. `make test` (~16 s) is
   green before any commit **that changes something the suite covers**. Don't run
   it when nothing testable changed — a docs, comment, `Makefile` or `STATUS.md`
   edit is not a reason to spend a suite run. Logic-heavy code gets unit tests;
   the pipe gets an end-to-end test.
3. **Milestones are the build order; work packages are not.** Implement in
   **M0→M7** order (DESIGN §7). The **WP-A…WP-I** packages (DESIGN §8) are a
   decomposition by component, *not* a sequence — pull from them per milestone.
4. **The CV layer is a stateless per-frame estimator.** It streams positions, never
   moves/legality/PGN/game-state (DESIGN D1, §1). An illegal-looking board is
   reported as `ok`.
5. **Edge and inference meet only on the wire.** See architecture below — this is
   enforced by tests and by packaging.

## Architecture (packages = deployment boundaries)

uv **workspace**; each `packages/*` member is independently deployable.

| Package (`import`) | Role (DESIGN layer) | Depends on |
|---|---|---|
| `board-reader-core` (`board_reader_core`) | shared contract: L3 protocol §5, L1→L2 wire §4.5, sim helpers | numpy, opencv |
| `board-reader-edge` (`board_reader_edge`) | the **edge** node ("Pi", L1): capture, static crop, forward (the phone's `/camera` page is the other edge) | core only |
| `board-reader-inference` (`board_reader_inference`) | the **inference** node (L2): stateless recognizer (ChessQ) + WS gateway | core only |
| `board-reader-viewer` (`board_reader_viewer`) | serves the SvelteKit pages (live, `/snapshot`, `/camera`, `/control`) and relays `/api` + `/ws` to the inference node | none of ours |

**Naming:** DESIGN.md calls L2 the "central / Jetson" node; in code it is
**`inference`** (and the edge is the "Pi"). Same concepts, friendlier names.

**The separation rule (critical):** `edge` and `inference` **never import each
other** — they communicate solely via the wire (`board_reader_core.wire` frames in,
`board_reader_core.protocol` events out). `core` never imports either node. This is
guarded by `tests/test_layer_separation.py` (AST import scan + manifest check) and
by packaging (neither node depends on the other). The CLI/compose are the only
glue that wires both, and only across the network.

## Dev workflow

**Commit at logical checkpoints, unprompted.** When a unit of work is done and
verified — a fix landed with its tests, a feature demoed green, a subagent's work
validated — commit it right then (conventional message, one concern per commit).
Never let several finished workstreams pile up uncommitted in the tree.

**Never pre-stage in a checkout another session shares.** A `git add` or `git rm`
sits in the shared index, and the other session's bare `git commit` takes it (a docs
commit once swept in another session's staged deletions). Delete with plain `rm`,
then commit with `git commit -F msg -- <paths>`: a pathspec commit records worktree
deletions of tracked files and ignores the rest of the index. A brand-new file still
needs `git add`, so run it in the same command as the commit.

**Testing costs ~16 s. If it is costing you more, read
[docs/TESTING.md §0](docs/TESTING.md#0-read-this-first-the-60-second-version)** —
it is the 60-second agent briefing: which command to run when, the four habits that
waste the most time, what a hang means, and the one `/ingest` gotcha
(`packages/inference/tests/conftest.py`). Don't read the rest of TESTING.md unless
you are adding a test.

```bash
uv sync                       # install the whole workspace + dev tools
make models                   # the ChessQ weights (pinned, SHA-256 checked) + the fp32 export; without them the stub serves
make test                     # the whole suite in parallel (-n auto), minus GPU, ~16s
make test-gpu                 # the real-ONNX-session test (ChessQ V6 on a still); needs an idle GPU
make test-all                 # everything, GPU included
make test-front               # the frontend: Vitest + Playwright (not covered by `make test`)
make lint                     # ruff, pinned in the dev group
uv run pytest <path>[::<test>] -q   # one file/test while iterating — no `-n`, it costs more
uv run pytest packages/core   # one package's tests
uv run pytest                 # the whole suite SERIALLY (~23s) — prefer `make test`
make gate GATE_NAME=<new>     # replay the four gate takes; table against cq-v6-tuned (needs the GPU)

# Run locally (one terminal): inference + viewer on http :8080 (live, /snapshot)
# and a second viewer bind on https :8443, whose secure origin /camera needs.
# `make dev` depends on `ui`, so this alone is only needed after touching
# packages/viewer/ui/ without wanting the rest of the stack.
make ui                       # build the SvelteKit frontend -> board_reader_viewer/web/
make dev                      # ChessQ V6 fp32 on the GPU; `make dev-cpu` serves int8 on the CPU

# Or by hand, without Docker (two terminals):
BR_CHESSQ=models/chessq/cq-vit-s-644-v6-seed1-onnx-fp32.onnx \
  uv run board-reader-inference serve          # gateway on :8000 (int8 file on a CPU)
uv run board-reader-edge feed                  # synthetic feed -> :8000
uv run board-reader-viewer --port 8080         # UI on :8080  -> open it

# Run the whole thing on Docker (the canonical demo):
docker compose up --build                      # then open http://localhost:8080
```

See **[docs/TESTING.md](docs/TESTING.md)** for the full manual + automated test guide.

## Supply chain

Dependencies are **7-day seasoned**: `pyproject.toml` sets
`[tool.uv] exclude-newer` to *today − 7 days*, so newly-published (possibly
compromised) releases are never resolved. To take newer deps, bump that date
deliberately and re-run `uv sync`, then `uv run pytest`. Python deps go through
`uv` only.

The UI (`packages/viewer/ui`) mirrors this on the npm side: `pnpm-workspace.yaml`
sets `minimumReleaseAge: 10080` (minutes = 7 days; pnpm 11's own default is 1440,
one day), so a freshly-published package is never resolved there either. Bump one
package deliberately via `minimumReleaseAgeExclude`, then re-run `pnpm install`
and `pnpm test`.

## Learnings and rules

- **Recall learnings before you design; conventions go into the rules.** The
  learnings store `.learnings.ndjson` is **frozen as of 2026-11-24: add nothing to
  it.** Before designing or reviewing a change, run
  `learnings recall --paths <files in scope>` (or `/recall`) for the research
  verdicts and how-tos that still apply there.
- A finding that is a standing convention for some paths (something every agent
  working there must do or avoid, every time) goes into the matching
  `.claude/rules/<area>.md` as a plain instruction with its reason, in the same
  change; a new rules file gets `paths:` frontmatter. Claude Code loads a rules file
  when an agent touches a matching file. Anything narrower belongs in the commit
  message; an experiment's outcome belongs in `docs/reports/`.
- When a learning turns out wrong or outdated, remove it with
  `learnings forget --id <id>` (or `/forget`); it soft-deletes, so the line stays for
  provenance. When a file move breaks an entry's paths, repair it with
  `learnings rescope --id <id> --paths <new glob>` in the same change, never by
  forgetting it. `learnings audit` must report 0 drifted entries.
- **Never open `.learnings.ndjson` directly**: recall reads only what applies to your
  paths, at a fraction of the tokens.

DESIGN.md stays authoritative for *decisions*.
