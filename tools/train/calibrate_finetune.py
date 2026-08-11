"""Fine-tune the deployed piece classifier on ONE physical board's calibrate frames.

Product form of the 2026-08-11 ``calib_finetune`` experiment
(``scratch/experiments/calib_finetune_confirm/report.md``). The operator photographs
their own board in the start position a few times, across a few sittings; this
harvests free labels from those photos (the layout is known), LP-FT's the base model
on them, and exports a **tuned model that is only ever served averaged 50/50 with its
base** (``BR_MODEL_TUNED``).

Two rules are wired in here rather than written in a doc, because both were measured
and both are silent failures when broken:

* **Three capture sessions minimum.** Frames from a single sitting bought nothing —
  3 frames from 1 session scored 0.932 on user_photos against the base's 0.938, i.e.
  slightly *harmful*, while 3 frames from 3 sessions scored 0.968. Diversity of
  lighting/framing is the ingredient, not the frame count. ``--force`` overrides.
* **Never serve the tuned model alone.** On its own it invents pieces on empty
  squares (hallucinations 1 → 3.2). In the 50/50 average with its base they go back
  to 1 on every measured run, and the type win survives. That is why the export
  carries the base's geometry and why the server refuses ``BR_MODEL_TUNED`` without
  ``BR_MODEL``.

Usage (under the scratch torch venv — see ``calibrate_finetune.sh``)::

    calibrate_finetune.py --dir data/uploads --out scratch/tuned/mine
    calibrate_finetune.py --frames a.jpg b.jpg c.jpg --out scratch/tuned/mine

Everything below the frame-selection layer imports torch, and is therefore imported
lazily: the session-clustering rule is the part with a unit test, and that test runs
in the torch-free workspace suite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

# tools/train/calibrate_finetune.py -> repo root is two parents up. Spelled out
# rather than imported from _bootstrap: importing _bootstrap installs a stub
# ``board_reader_inference`` package into sys.modules, which must not happen in
# the workspace test process that imports this module for its clustering tests.
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE = ROOT / "scratch/runs/run_e/best.pt"
DEFAULT_GLOB = "*-calibrate.json"
SIZE = 224

#: A gap this long between captures starts a new **session**. The measured sessions
#: were hours to days apart and the within-sitting frames minutes apart, so anything
#: from ~10 min to ~6 h separates them identically; an hour is the round number in
#: the middle. What matters is that consecutive shots of one sitting never split.
SESSION_GAP_S = 60 * 60.0

#: The product rule the experiment established (see the module docstring).
MIN_SESSIONS = 3

#: Verbatim from ``scratch/experiments/calib_finetune/tables_sweep2.md``, quoted at
#: the operator whenever they reach for ``--force``.
SINGLE_SESSION_MEASUREMENT = (
    "3 frames from 1 session scored 0.9323 [0.9281-0.9375] on user_photos — BELOW "
    "the un-tuned base's 0.9375. 3 frames from 3 sessions scored 0.9681. Tuning on "
    "one sitting is measured worthless-to-harmful; it is condition diversity that "
    "buys the gain, not frame count."
)

# The LP-FT recipe, exactly as validated (arms C2'/C3', seeds 10-14). These are not
# tuning knobs — changing one invalidates the measurement the ship decision rests on.
HEAD_ITERS, HEAD_LR = 100, 1e-3
FULL_ITERS, FULL_LR = 150, 1e-4
WEIGHT_DECAY = 0.05
BATCH_SIZE = 32
WARMUP_FRAC = 0.05


class Refused(SystemExit):
    """The inputs do not satisfy a measured product rule; nothing was trained."""


@dataclass(frozen=True, slots=True)
class CalibFrame:
    """One start-position photo of the board, with the corners the operator marked."""

    name: str
    image_path: Path
    sidecar_path: Path
    ts: float
    corners: tuple[float, ...]  # x0,y0,...,x3,y3 — a8,h8,h1,a1

    @property
    def captured(self) -> str:
        return datetime.fromtimestamp(self.ts, UTC).strftime("%Y-%m-%d %H:%M")


def _load_frame(sidecar: Path) -> CalibFrame:
    """One sidecar -> a usable calibrate frame, or a refusal explaining what is off."""
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"{sidecar}: unreadable sidecar ({e})") from e
    kind = meta.get("kind")
    if kind != "calibrate":
        raise Refused(
            f"{sidecar}: kind={kind!r}, not 'calibrate'. Only calibrate frames carry "
            "the guarantee this tool's labels rest on — that the board is in the "
            "start position."
        )
    # A calibrate sidecar may record the placement it was taken for. If it says
    # something other than the start position, the free labels would be wrong on
    # every moved piece, and the fine-tune would learn the mistake.
    placement = str(meta.get("placement", "")).split()[0] if meta.get("placement") else ""
    if placement and placement != _start_placement():
        raise Refused(
            f"{sidecar}: placement {placement!r} is not the start position. This tool "
            "labels crops from the known start layout; a different position needs "
            "labels it does not have."
        )
    raw = str(meta.get("corners", ""))
    nums = [float(v) for v in raw.split(",")] if raw else []
    if len(nums) != 8:
        raise Refused(f"{sidecar}: corners={raw!r} is not 8 comma-separated numbers")
    ts = meta.get("ts")
    if not isinstance(ts, int | float):
        raise Refused(
            f"{sidecar}: no numeric 'ts'. Capture time is what groups frames into "
            "sessions, and the session count is the gate on this whole tool."
        )
    image = sidecar.with_suffix(".jpg")
    if not image.exists():
        raise Refused(f"{sidecar}: no image beside it at {image.name}")
    return CalibFrame(sidecar.stem, image, sidecar, float(ts), tuple(nums))


@cache
def _start_placement() -> str:
    """START_PLACEMENT, imported late so this module stays stdlib-only at import."""
    src = str(ROOT / "packages/core/src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from board_reader_core.protocol import START_PLACEMENT

    return START_PLACEMENT


def resolve_sidecars(frames: list[Path], directory: Path | None, pattern: str) -> list[Path]:
    """The sidecar paths named by the CLI: explicit files, or a directory + glob."""
    if frames and directory is not None:
        raise Refused("pass --frames or --dir, not both")
    if directory is not None:
        found = sorted(directory.glob(pattern))
        if not found:
            raise Refused(f"no {pattern} under {directory}")
        return found
    if not frames:
        raise Refused("pass --frames <paths> or --dir <dir>")
    return [p if p.suffix == ".json" else p.with_suffix(".json") for p in frames]


def load_frames(sidecars: list[Path]) -> list[CalibFrame]:
    """Load every named frame, oldest first. Any unusable frame refuses the run."""
    return sorted((_load_frame(p) for p in sidecars), key=lambda f: f.ts)


def group_sessions(
    frames: list[CalibFrame], gap_s: float = SESSION_GAP_S
) -> list[list[CalibFrame]]:
    """Split time-ordered frames into capture sessions at every gap over ``gap_s``."""
    sessions: list[list[CalibFrame]] = []
    for frame in sorted(frames, key=lambda f: f.ts):
        if sessions and frame.ts - sessions[-1][-1].ts <= gap_s:
            sessions[-1].append(frame)
        else:
            sessions.append([frame])
    return sessions


def describe_sessions(sessions: list[list[CalibFrame]]) -> str:
    lines = []
    for i, session in enumerate(sessions):
        lines.append(
            f"  session {i + 1}: {len(session)} frame(s), "
            f"{session[0].captured} .. {session[-1].captured}"
        )
        lines.extend(f"      {f.name}" for f in session)
    return "\n".join(lines)


def check_session_gate(sessions: list[list[CalibFrame]], *, force: bool) -> None:
    """Enforce the ≥3-session rule; ``force`` downgrades it to a loud warning."""
    if len(sessions) >= MIN_SESSIONS:
        return
    detail = (
        f"{len(sessions)} capture session(s) in the given frames, need "
        f"{MIN_SESSIONS}:\n{describe_sessions(sessions)}"
    )
    if not force:
        raise Refused(
            f"REFUSED: {detail}\n\n{SINGLE_SESSION_MEASUREMENT}\n\n"
            "Photograph the board again in a different sitting (different time of "
            "day / lighting / camera placement) and re-run. --force overrides."
        )
    print(
        f"\n{'!' * 78}\n!! --force: training on {detail}\n!!\n"
        f"!! {SINGLE_SESSION_MEASUREMENT}\n"
        f"!! The resulting model is OUTSIDE every measured configuration.\n{'!' * 78}\n",
        file=sys.stderr,
    )


# --- everything below needs torch --------------------------------------------------


def _harvest(frames: list[CalibFrame], geometry):
    """Crops + start-position labels for every frame, cut with the base's geometry."""
    import cv2
    import numpy as np

    from board_reader_core.protocol import CLASS_INDEX, parse_placement

    from board_reader_inference.squares import crop_squares_photo

    labels_one = np.array(
        [CLASS_INDEX[c] for row in parse_placement(_start_placement()) for c in row],
        dtype=np.int64,
    )
    crops, labels = [], []
    for frame in frames:
        img = cv2.imread(str(frame.image_path))
        if img is None:
            raise Refused(f"unreadable image {frame.image_path}")
        corners = np.asarray(frame.corners, dtype=np.float32).reshape(4, 2)
        crops.extend(crop_squares_photo(img, corners, geometry=geometry, size=SIZE))
        labels.append(labels_one)
    return crops, np.concatenate(labels)


def _phase(model, loader, params, *, iters, lr, device):
    """One LP-FT phase: AdamW + 5% linear warmup into cosine, ``iters`` steps."""
    import torch
    import torch.nn as nn

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    warmup = max(1, int(WARMUP_FRAC * iters))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iters - warmup)),
        ],
        milestones=[warmup],
    )
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    losses, it = [], iter(loader)
    for _ in range(iters):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.detach().item())
    return losses


def finetune(base_ckpt: Path, frames: list[CalibFrame], seed: int):
    """LP-FT the base checkpoint on these frames' crops.

    Returns ``(tuned, cfg, base, crops, labels, stats)`` — the base model and the
    harvested crops come back too, because the scorecard needs exactly them and
    re-cutting 64 crops per multi-megapixel photo to get them is pure waste.
    """
    from copy import deepcopy

    import numpy as np
    import torch
    from augment import to_rgb_tensor, train_transform
    from config import Config

    from models import build_model

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["config"])
    base = build_model(
        cfg.model,
        pretrained=False,
        unfreeze_blocks=cfg.unfreeze_blocks,
        freeze_backbone=cfg.freeze_backbone,
    )
    base.load_state_dict(ckpt["state_dict"])
    model = deepcopy(base).to(device)

    crops, labels = _harvest(frames, cfg.geometry)
    tf = train_transform(SIZE, shape_bias=False)

    class CropSet(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(crops)

        def __getitem__(self, i: int):
            return tf(to_rgb_tensor(crops[i])), int(labels[i])

    # Two forked workers: the augmentation pipeline, not the GPU, is what this waits
    # on. `fork` because py3.14's forkserver default pickles a copy of the dataset
    # per worker — harmless at this size, but the training harness was OOM-killed by
    # exactly that, so the whole project keeps one story.
    loader = torch.utils.data.DataLoader(
        CropSet(),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=False,
        persistent_workers=True,
        multiprocessing_context="fork",
        prefetch_factor=4,
    )

    t0 = time.time()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    head = _phase(model, loader, list(model.fc.parameters()),
                  iters=HEAD_ITERS, lr=HEAD_LR, device=device)
    for p in model.parameters():
        p.requires_grad = True
    full = _phase(model, loader, list(model.parameters()),
                  iters=FULL_ITERS, lr=FULL_LR, device=device)
    model.eval()
    stats = {
        "device": str(device),
        "crops": len(crops),
        "head_loss_first_last": [round(head[0], 4), round(head[-1], 4)],
        "full_loss_first_last": [round(full[0], 4), round(full[-1], 4)],
        "seconds": round(time.time() - t0, 1),
    }
    return model, cfg, base.to(device).eval(), crops, labels, stats


def _probs(model, crops, device):
    """(N,13) softmax probabilities for pre-cut crops, through the eval transform."""
    import torch
    from augment import eval_transform, to_rgb_tensor

    tf = eval_transform(SIZE)
    with torch.no_grad():
        batch = torch.stack([tf(to_rgb_tensor(c)) for c in crops]).to(device)
        return torch.softmax(model(batch), dim=1).cpu().numpy()


def scorecard(base, tuned, crops, labels, frames, device) -> dict:
    """Fit on the frames that were trained on — a smoke test, not a generalisation number."""

    p_base, p_tuned = _probs(base, crops, device), _probs(tuned, crops, device)
    rows = []
    for i, frame in enumerate(frames):
        sl = slice(i * 64, (i + 1) * 64)
        y = labels[sl]
        rows.append(
            {
                "frame": frame.name,
                "base": int((p_base[sl].argmax(1) == y).sum()),
                "tuned": int((p_tuned[sl].argmax(1) == y).sum()),
                "averaged": int(
                    ((0.5 * p_base[sl] + 0.5 * p_tuned[sl]).argmax(1) == y).sum()
                ),
            }
        )
    total = {
        k: int(sum(r[k] for r in rows)) for k in ("base", "tuned", "averaged")
    }
    return {"per_frame": rows, "total": total, "squares": len(labels)}


def base_onnx_for(base_ckpt: Path, base_model) -> str:
    """What to point ``BR_MODEL`` at: the base's ONNX export, not its checkpoint.

    ``export.py`` writes ``model.onnx`` beside the checkpoint it exported, so that
    sibling is the candidate — but only *after* checking it really is this
    checkpoint's export. A sibling left over from an earlier export of a run that
    kept training is the one mistake that would quietly serve an average of a
    fine-tune and a model it was never fine-tuned from, and nothing downstream
    could tell. A random batch through both settles it in about a second.
    """
    sibling = base_ckpt.parent / "model.onnx"
    if not sibling.exists():
        return "<the ONNX export of your base model>"
    import onnxruntime as ort
    import torch

    x = torch.randn(4, 3, SIZE, SIZE)
    with torch.no_grad():
        ref = base_model.cpu().eval()(x).numpy()
    sess = ort.InferenceSession(str(sibling), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    if abs(ref - got).max() > 1e-3:
        print(
            f"NOTE: {sibling} exists but is NOT an export of {base_ckpt} — it must be "
            "stale. Re-export the base (tools/train/export.py --ckpt) before serving.",
            file=sys.stderr,
        )
        return "<the ONNX export of your base model — the sibling one is stale>"
    return str(sibling)


def print_scorecard(card: dict, out: Path, base_onnx: str) -> None:
    n = card["squares"]
    print("\n" + "=" * 78)
    print("SCORECARD — fit on the TRAINING frames (these frames were trained on;")
    print("            this is a smoke test that the fine-tune took, NOT accuracy)")
    print("=" * 78)
    print(f"{'frame':<38}{'base':>10}{'tuned':>10}{'averaged':>12}")
    for r in card["per_frame"]:
        print(f"{r['frame']:<38}{r['base']:>8}/64{r['tuned']:>8}/64{r['averaged']:>10}/64")
    t = card["total"]
    print(f"{'TOTAL':<38}{t['base']:>7}/{n}{t['tuned']:>7}/{n}{t['averaged']:>9}/{n}")
    print("=" * 78)
    print("SERVE IT AS THE AVERAGE, NEVER ALONE:")
    print(f"    BR_MODEL={base_onnx}")
    print(f"    BR_MODEL_TUNED={(out / 'tuned.onnx').resolve()}")
    print("The tuned model alone was measured to hallucinate pieces onto empty")
    print("squares (1 -> 3.2 per 5 boards); it is the base's vote in the 50/50")
    print("average that suppresses them. The server refuses BR_MODEL_TUNED without")
    print("BR_MODEL for that reason.")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_argument_group("input frames (calibrate photos of ONE board)")
    src.add_argument("--frames", type=Path, nargs="+", default=[],
                     help="explicit .jpg or .json paths")
    src.add_argument("--dir", type=Path, default=None, help="directory to glob instead")
    src.add_argument("--glob", default=DEFAULT_GLOB, help=f"glob for --dir (default {DEFAULT_GLOB})")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE,
                    help=f"base checkpoint to fine-tune (default {DEFAULT_BASE})")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help=f"train anyway below {MIN_SESSIONS} capture sessions (unmeasured)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the sessions found and stop, before any GPU time")
    args = ap.parse_args()

    frames = load_frames(resolve_sidecars(args.frames, args.dir, args.glob))
    sessions = group_sessions(frames)
    print(f"{len(frames)} calibrate frame(s) in {len(sessions)} capture session(s):")
    print(describe_sessions(sessions))
    check_session_gate(sessions, force=args.force)
    if args.dry_run:
        return
    if not args.base.exists():
        raise Refused(f"no base checkpoint at {args.base}")

    # Imported here, and only here, so a refusal above costs no torch import.
    import _bootstrap  # noqa: F401
    import export

    print(f"\nfine-tuning {args.base} on {len(frames) * 64} crops (seed {args.seed})")
    model, cfg, base, crops, labels, stats = finetune(args.base, frames, args.seed)
    print(f"  {stats}")

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "tuned.pt"
    onnx_path = args.out / "tuned.onnx"
    # Save with the BASE's config: the sidecar export writes from it, so the tuned
    # model ships the geometry it was actually cropped with — the base's — which is
    # also the precondition the served average asserts.
    import torch

    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, ckpt_path)
    print(f"wrote {ckpt_path}")
    # The onnxscript traceback printed during export is expected noise; the success
    # signal is the "round-trip OK" line below it.
    export.export(ckpt_path, onnx_path)

    device = next(model.parameters()).device
    card = scorecard(base, model, crops, labels, frames, device)
    base_onnx = base_onnx_for(args.base, base)
    (args.out / "calibration.json").write_text(
        json.dumps(
            {
                "written": datetime.now(UTC).isoformat(timespec="seconds"),
                "base": str(args.base),
                "seed": args.seed,
                "recipe": {
                    "head_iters": HEAD_ITERS, "head_lr": HEAD_LR,
                    "full_iters": FULL_ITERS, "full_lr": FULL_LR,
                    "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                    "shape_bias": False, "size": SIZE,
                },
                "geometry": cfg.geometry.to_dict(),
                "sessions": [[f.name for f in s] for s in sessions],
                "forced_below_session_gate": bool(args.force and len(sessions) < MIN_SESSIONS),
                "train": stats,
                "scorecard": card,
                "serve": {
                    "BR_MODEL": base_onnx,
                    "BR_MODEL_TUNED": str(onnx_path.resolve()),
                },
            },
            indent=1,
        )
    )
    print_scorecard(card, args.out, base_onnx)


if __name__ == "__main__":
    main()
