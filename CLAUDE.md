# CLAUDE.md — board-reader

Camera-based chess **board reader**: a CV system that reads live OTB positions and
streams **timestamped piece placements + per-square confidences** (not moves, not
legality — those are the downstream application layer's job).

**Read first:** [STATUS.md](STATUS.md) (where the project is) ·
**[docs/DESIGN.md](docs/DESIGN.md)** (decisions — authoritative) ·
[docs/index.md](docs/index.md) (how it currently behaves — architecture, recognition
pipeline, calibration, operating envelope, runbook). `aspects.yaml` maps which parts
must stay in sync with which.

## Load-bearing principles (do not violate without updating DESIGN.md)

1. **User-testable from the start.** Every milestone must be runnable end-to-end by
   a person and *observable* — `docker compose up` → open the viewer → see it work.
   We never land a milestone that can't be demoed live. New behaviour ships with
   both automated tests **and** a path in [docs/TESTING.md](docs/TESTING.md). The
   walking skeleton (M0) exists precisely to make this true from day one.
2. **Everything is tested.** No feature merges without tests. `uv run pytest` is
   green before any commit. Logic-heavy code gets unit tests; the pipe gets an
   end-to-end test.
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
| `board-reader-edge` (`board_reader_edge`) | the **edge** node ("Pi", L1): capture/forward; later detect+rectify | core only |
| `board-reader-inference` (`board_reader_inference`) | the **inference** node (L2): stateless estimator + WS gateway | core only |
| `board-reader-viewer` (`board_reader_viewer`) | serves the live-position web UI | none (static) |

**Naming:** DESIGN.md calls L2 the "central / Jetson" node; in code it is
**`inference`** (and the edge is the "Pi"). Same concepts, friendlier names.

**The separation rule (critical):** `edge` and `inference` **never import each
other** — they communicate solely via the wire (`board_reader_core.wire` frames in,
`board_reader_core.protocol` events out). `core` never imports either node. This is
guarded by `tests/test_layer_separation.py` (AST import scan + manifest check) and
by packaging (neither node depends on the other). The CLI/compose are the only
glue that wires both, and only across the network.

## Dev workflow

```bash
uv sync                       # install the whole workspace + dev tools
uv run pytest                 # the whole test suite (must be green)
uv run pytest packages/core   # one package's tests

# Run locally without Docker (two terminals):
uv run board-reader-inference serve            # gateway on :8000
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

## Compound engineering (learnings)

This project uses a shared, path-scoped learnings store (`.learnings.ndjson`).

- **Before** working in an area: `learnings recall` (or the `/recall` skill) to load
  what we already know about it.
- **After** a non-trivial change, before committing: capture durable, non-obvious
  learnings with `learnings learn --text "..." --paths "<glob>"` (or `/learn`).
  Learnings ship with the code that produced them.
- Relevant skills: `/recall`, `/learn`, `/ce-plan`, `/ce-review`, `/ce-compound`.

Keep DESIGN.md authoritative for *decisions*; keep learnings for *gotchas and
how-tos* discovered while building.
