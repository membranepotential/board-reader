# Third-party licenses and redistribution notes

board-reader's own source is **Apache-2.0** ([LICENSE](LICENSE)). Two things do not
follow from that, and both matter before you redistribute anything:

1. The **built viewer bundle** incorporates GPL-licensed code, so the bundle as
   distributed is **GPL-3.0-or-later**.
2. The **model artifacts** are **not redistributable at all**: the core recognizer
   (ChessQ Lite V6) is PolyForm Noncommercial, and our older trained models derive
   from non-commercially-licensed training data. They are git-ignored on purpose.

---

## 1. The viewer UI bundle is GPL-3.0-or-later

`packages/viewer/ui` depends on **[`@lichess-org/chessground`](https://github.com/lichess-org/chessground)**
(`GPL-3.0-or-later`) for board rendering, and pulls in its stylesheets — which embed
the **cburnett** piece sprites (`GPLv2+`, © Colin M.L. Burnett, per
[lila's COPYING.md](https://github.com/lichess-org/lila/blob/master/COPYING.md)) as
data URIs.

`make ui` links these into `packages/viewer/src/board_reader_viewer/web/`. That output
is a combined work:

- **Our source stays Apache-2.0.** Apache-2.0 is one-way compatible with GPLv3, so our
  code may be combined into the GPL work. (cburnett is GPLv2-*or-later*, so it can be
  taken as v3 and combined too.)
- **The bundle is GPL-3.0-or-later.** Anyone distributing the built `web/` directory —
  a wheel with package data, a Docker image, a tarball — is distributing under
  GPL-3.0-or-later and owes recipients the corresponding source and a copy of the GPL.
- **The repository itself carries no GPL code.** The bundle is git-ignored; the sources
  here only reference the package by name.

Nothing outside `packages/viewer/ui` touches chessground. `board-reader-core`,
`board-reader-edge` and `board-reader-inference` have **no GPL dependencies** and are
plain Apache-2.0 — the wire protocol and the estimator are reusable without any
copyleft obligation.

## 2. Model artifacts are NOT redistributable

**Do not attach `models/*.onnx`, exemplar banks, or fine-tuned checkpoints to a release,
and do not bake them into a published image.**

The shipped recognition model is trained on:

| Source | License | Consequence |
|---|---|---|
| ChessReD (Masouris & van Gemert, 4TU.ResearchData) | **CC BY-NC-SA 4.0** | non-commercial; share-alike |
| ChessR Blender renders (CGTrader-derived geometry) | **murky / unclear** | segregated and droppable by design |
| ChessRender360 | CC BY 4.0 | clean |
| chesscog transfer set | CC BY | clean |
| Zenodo piece crops | CC BY | clean, auxiliary only |
| 8 scouted 3D sets | CC0 / CC BY | clean |

The resulting weights are a derivative of the first two rows: **fine for personal and
research use, not for commercial use, and not ours to redistribute.** See
[docs/training-and-data.md](docs/training-and-data.md) for the full ledger.

`.gitignore` enforces this — `models/` (except the ChessQ release descriptors, which
hold no weights), `data/`, `scratch/`, `recordings/` and `*.onnx` never enter the
repository. Keep it that way.

**ChessQ Lite V6** — **the core recognizer** since 2026-11-24 (`BR_CHESSQ`, set by
`compose.yaml` and `make dev`; DESIGN D10) — is Lichess's board recognizer,
distributed at [huggingface.co/joelseytre/chessq-lite](https://huggingface.co/joelseytre/chessq-lite)
under the **PolyForm Noncommercial License 1.0.0**: personal and research use only,
no commercial use. The user accepted that licence for this project's core on
2026-11-24, so **the default configuration of board-reader is noncommercial** as a
whole, whatever the Apache-2.0 source allows on its own.

- **Weights:** `make models` downloads them from the pinned revision into the
  git-ignored `models/chessq/` and checks size and SHA-256 against the release
  descriptor. They are never committed, attached to a release or baked into an image.
- **Descriptor:** the release's descriptor JSON
  (`models/chessq/cq-vit-s-644-v6-seed1-onnx-int8-r1.json`) **is** committed — it is
  metadata only (download URL at a pinned revision, size, SHA-256, input/output
  contract, the licence name), and `make models` needs it on a fresh clone.
- **Code:** the module that loads the model (`board_reader_inference.chessq`) is our
  own Apache-2.0 code and contains none of its weights.

## 3. Recorded footage

`tools/video/` downloads chess broadcasts to derive crops. **The footage is copyrighted
and is never committed** — `data/`, `recordings/` and `*.mp4|mkv|webm` are git-ignored.
`tools/video/crops.csv` holds only video IDs and crop rectangles (metadata), not media.

## 4. Runtime dependencies

All Python runtime dependencies are permissive:

| Package | License |
|---|---|
| numpy | BSD-3-Clause (with 0BSD / MIT / Zlib / CC0-1.0 components) |
| opencv-python-headless | Apache-2.0 (wheels bundle **FFmpeg under LGPLv2.1**) |
| onnxruntime-gpu | MIT |
| fastapi, pydantic | MIT |
| starlette, uvicorn, websockets, httpx | BSD-3-Clause |
| cryptography | Apache-2.0 OR BSD-3-Clause |

The **LGPLv2.1 FFmpeg** inside the opencv wheels is the one obligation here, and it only
bites when distributing binaries (e.g. the Docker image): LGPL permits this as shipped —
dynamically linked, unmodified — provided the notice is carried and recipients can
replace the library. Source distribution is unaffected.

The frontend build toolchain (Svelte, Vite, SvelteKit, Vitest, Playwright, TypeScript)
is MIT and build-time only — it does not enter the bundle.
