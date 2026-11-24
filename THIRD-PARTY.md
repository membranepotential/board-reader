# Third-party licenses and redistribution notes

board-reader's own source is **Apache-2.0** ([LICENSE](LICENSE)). Two things do not
follow from that, and both matter before you redistribute anything:

1. The **built viewer bundle** incorporates GPL-licensed code, so the bundle as
   distributed is **GPL-3.0-or-later**.
2. The **trained model artifacts** derive from non-commercially-licensed training
   data and are **not redistributable at all**. They are git-ignored on purpose.

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

`.gitignore` enforces this — `models/`, `data/`, `scratch/`, `recordings/` and `*.onnx`
never enter the repository. Keep it that way.

The **backbone** is the exception: DINOv2 ViT-S/14 is **Apache-2.0** (code and weights,
[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)) and
`tools/export_dinov2_onnx.py` converts the official Meta weights locally rather than
pulling a third-party re-host.

**ChessQ Lite** (the opt-in `BR_CHESSQ` estimator) is Lichess's board recognizer,
distributed at [huggingface.co/joelseytre/chessq-lite](https://huggingface.co/joelseytre/chessq-lite)
under the **PolyForm Noncommercial License 1.0.0**: personal and research use only,
no commercial use. We download the release into the git-ignored `models/chessq/` and
check its SHA-256 against the release descriptor; it is never committed or
redistributed, and the code that loads it (`board_reader_inference.chessq`) contains
none of its weights.

The optional `init: sin` training path
(`tools/train/configs/loso_resnet_sin.yaml`) expects a Stylized-ImageNet ResNet-50
checkpoint from [rgeirhos/texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape).
It is not redistributed here; fetch it yourself into the git-ignored `scratch/weights/`.

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
