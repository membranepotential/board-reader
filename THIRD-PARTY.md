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

`packages/viewer` depends on **[`@lichess-org/chessground`](https://github.com/lichess-org/chessground)**
(`GPL-3.0-or-later`) for board rendering, and pulls in its stylesheets — which embed
the **cburnett** piece sprites (`GPLv2+`, © Colin M.L. Burnett, per
[lila's COPYING.md](https://github.com/lichess-org/lila/blob/master/COPYING.md)) as
data URIs.

`make ui` links these into `packages/viewer/build/`. That output
is a combined work:

- **Our source stays Apache-2.0.** Apache-2.0 is one-way compatible with GPLv3, so our
  code may be combined into the GPL work. (cburnett is GPLv2-*or-later*, so it can be
  taken as v3 and combined too.)
- **The bundle is GPL-3.0-or-later.** Anyone distributing the built `build/` directory (or the viewer image) —
  a wheel with package data, a Docker image, a tarball — is distributing under
  GPL-3.0-or-later and owes recipients the corresponding source and a copy of the GPL.
- **The repository itself carries no GPL code.** The bundle is git-ignored; the sources
  here only reference the package by name.

Nothing outside `packages/viewer` touches chessground. `board-reader-core`,
`board-reader-edge` and `board-reader-inference` have **no GPL dependencies** and are
plain Apache-2.0 — the wire protocol and the estimator are reusable without any
copyleft obligation.

## 2. Model artifacts are NOT redistributable

**Do not attach `models/*.onnx` (or any older in-house model, exemplar bank or
checkpoint still on disk) to a release, and do not bake them into a published image.**

The in-house crop classifier that served until 2026-11-24 was trained on ChessReD
(**CC BY-NC-SA 4.0**) and render geometry of unclear licence, so those weights were never
redistributable either; they are no longer served (see
[docs/training-and-data.md](docs/training-and-data.md)).

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

- **Weights:** `make models` downloads two release files from the pinned revision
  into the git-ignored `models/chessq/` — the int8 ONNX (served on a CPU) and the FP16
  safetensors — and checks size and SHA-256 against their release descriptors. It
  then exports the safetensors to an fp32 ONNX (served on a GPU) beside them. The
  export is a derivative of the weights and carries the **same PolyForm
  Noncommercial licence**. None of the three is ever committed, attached to a
  release or baked into an image.
- **Export tool:** `tools/chessq_export` loads the safetensors with the author's
  [`chessqueries`](https://github.com/JSeytre/chessqueries) package (PolyForm
  Noncommercial 1.0.0 too), pinned to one commit in its own uv project. It is a
  build-time dependency only: nothing in the workspace or the Docker images
  imports it or PyTorch.
- **Descriptors:** the two release descriptors
  (`models/chessq/cq-vit-s-644-v6-seed1-onnx-int8-r1.json`,
  `…-safetensors-fp16-r1.json`) and our export's descriptor
  (`…-onnx-fp32.json`) **are** committed — metadata only (download URL at a pinned
  revision, size, SHA-256, input/output contract, the licence name; for the export
  also its source, the package commit and the export's SHA-256), and `make models`
  needs them on a fresh clone.
- **Code:** the module that loads the model (`board_reader_inference.recognition.chessq`) is our
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
