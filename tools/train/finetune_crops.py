"""Fine-tune the deployed piece classifier on ALREADY-CUT, labelled crop exports —
the leave-session-out (E3) counterpart to ``calibrate_finetune.py``.

``calibrate_finetune.py`` harvests its own labels from start-position photos (the
layout is known). This tool instead consumes what ``board-reader-inference
eval-session --export-crops`` already cut and labelled from a scored trial's
``truth.json`` — mid-game crops, not just the start position — so it can fine-tune
on positions calibrate_finetune has no way to label.

It reuses calibrate_finetune's LP-FT machinery directly (imported, not copied):
the two-phase head-then-full training loop (:func:`calibrate_finetune._phase`),
the exact recipe hyperparameters, its ONNX-sibling resolver
(:func:`calibrate_finetune.base_onnx_for`), and the ``Refused`` exception. What it
does NOT reuse verbatim is ``calibrate_finetune.scorecard``/``print_scorecard``/
``_probs``: those assume every unit scored is exactly 64 contiguous squares (one
photo) run through ONE forward pass, and this tool's pooled sets are both bigger
(low thousands of crops OOM'd a 4GB card in one batch) and not 64-aligned once
empty-capping removes individual crops rather than whole boards. This module's
:func:`_probs` (chunked), :func:`scorecard` and :func:`print_scorecard` are the
session-grouped, batched equivalents — same base/tuned/averaged fit-check, same
"smoke test, not accuracy" caveat, but batched inference and a variable
denominator per group instead of a hardcoded ``/64``.

Two rules mirror calibrate_finetune's, adapted to this input shape:

* **Three export directories (sessions) minimum**, same spirit as
  calibrate_finetune's three-capture-session gate: one recording's crops teach the
  model that one sitting's lighting/board/camera framing, not the cross-session
  type distinction. ``--force`` overrides.
* **Empty crops are capped**, not left to dominate. Start-position photos are
  naturally ~50% empty by construction (32/64 squares); mid-game crops keep that
  same ~50% baseline in the trials measured here too, but nothing about
  ``--export-crops`` guarantees it forever (a near-empty endgame position would
  push it far higher), so the pool is capped at ``--empty-cap-multiple`` (default
  2x) the largest non-empty class's count rather than trusting the source data.

Usage (under the scratch torch venv — see ``calibrate_finetune.sh`` for the venv
invocation pattern)::

    finetune_crops.py --export-dirs data/external/trial_a data/external/trial_b \\
        data/external/trial_c --out scratch/tuned/mine --dry-run
    finetune_crops.py --export-dirs data/external/trial_a data/external/trial_b \\
        data/external/trial_c --out scratch/tuned/mine

Everything below the directory/label-loading layer imports torch and is therefore
imported lazily: the dir-discovery, label-loading, session-gate and empty-cap
layer is stdlib-only (+ the equally stdlib-only bits of calibrate_finetune it
imports) and unit-tested in the torch-free workspace suite.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

from calibrate_finetune import (
    BATCH_SIZE,
    DEFAULT_BASE,
    FULL_ITERS,
    FULL_LR,
    HEAD_ITERS,
    HEAD_LR,
    SIZE,
    WEIGHT_DECAY,
    Refused,
    _phase,
    base_onnx_for,
)

# tools/train/finetune_crops.py -> repo root is two parents up (same depth as
# calibrate_finetune.py).
ROOT = Path(__file__).resolve().parents[2]

#: What `eval-session --export-crops` writes (board_reader_inference.trial).
MANIFEST_NAME = "labels_normalized.json"
EXPORT_META_NAME = "export.json"

#: Same product rule as calibrate_finetune.MIN_SESSIONS, one export dir = one
#: recording = one session here.
MIN_SESSIONS = 3

#: cap pooled 'empty' crops at this multiple of the largest non-empty class count.
EMPTY_CAP_MULTIPLE = 2.0

SESSION_RULE = (
    "calibrate_finetune.py measured this exact failure mode for start-position "
    "photos: 3 frames from ONE sitting scored 0.9323 on user_photos — BELOW the "
    "untuned base's 0.9375 — while 3 frames from THREE sittings scored 0.9681. "
    "Session diversity, not crop count, is what buys the win; the same risk "
    "applies to crop exports, since a single recording is one sitting's lighting/ "
    "board/camera framing, not the cross-session type distinction this tool is "
    "for."
)


@dataclass(frozen=True, slots=True)
class CropRecord:
    """One already-cut, already-labelled square crop from an export manifest."""

    image_path: Path
    label: str  # a CLASSES symbol, "empty" included
    session: str  # the export dir's name, for scorecard grouping post-cap


@dataclass(frozen=True, slots=True)
class ExportedSession:
    """One `--export-crops` output directory == one recording == the leave-out unit."""

    name: str
    directory: Path
    geometry: dict
    records: tuple[CropRecord, ...]

    @property
    def class_counts(self) -> Counter:
        return Counter(r.label for r in self.records)


@cache
def _class_index() -> dict[str, int]:
    """CLASS_INDEX, imported late so this module stays stdlib-only at import
    (same idiom as calibrate_finetune._start_placement)."""
    src = str(ROOT / "packages/core/src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from board_reader_core.protocol import CLASS_INDEX

    return CLASS_INDEX


def resolve_export_dirs(dirs: list[Path]) -> list[Path]:
    """Validate the given export directories; de-duplicate, preserving order."""
    if not dirs:
        raise Refused("pass --export-dirs <dir> [<dir> ...]")
    seen: dict[Path, None] = {}
    for d in dirs:
        resolved = d.resolve()
        if not resolved.is_dir():
            raise Refused(f"{d}: not a directory")
        if not (resolved / MANIFEST_NAME).exists():
            raise Refused(
                f"{d}: no {MANIFEST_NAME} — not an `eval-session --export-crops` output"
            )
        seen.setdefault(resolved, None)
    return list(seen)


def _load_geometry(directory: Path) -> dict:
    meta = directory / EXPORT_META_NAME
    if not meta.exists():
        raise Refused(
            f"{directory}: no {EXPORT_META_NAME} — cannot verify the crop geometry "
            "matches the base checkpoint's"
        )
    try:
        data = json.loads(meta.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"{meta}: unreadable ({e})") from e
    geometry = data.get("geometry")
    if not isinstance(geometry, dict):
        raise Refused(f"{meta}: no 'geometry' block")
    return geometry


def load_session(directory: Path) -> ExportedSession:
    """One export dir -> its crop records. Any unusable record refuses the run."""
    manifest = directory / MANIFEST_NAME
    try:
        raw = json.loads(manifest.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"{manifest}: unreadable manifest ({e})") from e
    if not raw:
        raise Refused(f"{manifest}: no crop records")
    index = _class_index()
    records = []
    for rec in raw:
        label = rec.get("piece")
        if label not in index:
            raise Refused(f"{manifest}: unknown piece code {label!r}")
        image_path = directory / rec["image"]
        if not image_path.exists():
            raise Refused(f"{manifest}: crop image missing at {image_path}")
        records.append(CropRecord(image_path, label, directory.name))
    return ExportedSession(directory.name, directory, _load_geometry(directory), tuple(records))


def describe_sessions(sessions: list[ExportedSession]) -> str:
    return "\n".join(
        f"  {s.name}: {len(s.records)} crops from {s.directory}" for s in sessions
    )


def check_session_gate(sessions: list[ExportedSession], *, force: bool) -> None:
    """Enforce the >=3-export-dir rule; ``force`` downgrades it to a loud warning."""
    if len(sessions) >= MIN_SESSIONS:
        return
    detail = (
        f"{len(sessions)} export dir(s) (session(s)), need {MIN_SESSIONS}:\n"
        f"{describe_sessions(sessions)}"
    )
    if not force:
        raise Refused(
            f"REFUSED: {detail}\n\n{SESSION_RULE}\n\n"
            "Export crops from more recording sessions (`eval-session "
            "--export-crops`) and re-run. --force overrides."
        )
    print(
        f"\n{'!' * 78}\n!! --force: training on {detail}\n!!\n!! {SESSION_RULE}\n"
        f"!! The resulting model is OUTSIDE calibrate_finetune's measured session "
        f"configuration.\n{'!' * 78}\n",
        file=sys.stderr,
    )


def cap_empties(
    records: list[CropRecord], *, multiple: float = EMPTY_CAP_MULTIPLE, seed: int = 0
) -> list[CropRecord]:
    """Cap pooled ``empty`` crops at ``multiple``x the largest non-empty class count.

    A deterministic (seeded) subsample, not a hard drop of whatever sorts last —
    every empty crop across every session is equally likely to survive. ``multiple
    <= 0`` disables capping entirely (the raw pool is used as-is).
    """
    if multiple <= 0:
        return list(records)
    non_empty = [r for r in records if r.label != "empty"]
    empties = [r for r in records if r.label == "empty"]
    if not non_empty or not empties:
        return list(records)
    cap = round(multiple * max(Counter(r.label for r in non_empty).values()))
    if len(empties) <= cap:
        return list(records)
    kept = random.Random(seed).sample(empties, cap)
    return non_empty + kept


def _geometry_matches(a: dict, b: dict) -> bool:
    """Same crop geometry, tolerant of float noise. Pure and torch-free."""
    if a.get("kind", "bbox") != b.get("kind", "bbox"):
        return False
    return math.isclose(
        a.get("headroom", 0.0), b.get("headroom", 0.0), abs_tol=1e-6
    ) and math.isclose(a.get("side_pad", 0.0), b.get("side_pad", 0.0), abs_tol=1e-6)


# --- everything below needs torch --------------------------------------------------


def _check_geometry(cfg, sessions: list[ExportedSession]) -> None:
    """Refuse if any export's crop geometry doesn't match the base checkpoint's —
    training on crops cut with a different box than the model is served with at
    inference would just teach it a different, wrong, distribution."""
    base_geometry = cfg.geometry.to_dict()
    for s in sessions:
        if not _geometry_matches(s.geometry, base_geometry):
            raise Refused(
                f"{s.name}: export geometry {s.geometry} does not match the base "
                f"checkpoint's crop geometry {base_geometry} — these crops were cut "
                "with a different box than the model is served with. Re-export with "
                "the matching geometry, or fine-tune from a base that used this one."
            )


def _load_pixels(records: list[CropRecord]):
    """Read each crop off disk, letterboxed/resized exactly like cropcache's
    crop-mode samples — these are already single-square images (``--export-crops``
    cut them with ``crop_squares_photo`` already), not full photos to slice."""
    import cv2
    import numpy as np
    from cropcache import _pad_resize

    index = _class_index()
    crops, labels = [], []
    for r in records:
        img = cv2.imread(str(r.image_path))
        if img is None:
            raise Refused(f"unreadable crop image {r.image_path}")
        crops.append(_pad_resize(img, SIZE))
        labels.append(index[r.label])
    return crops, np.asarray(labels, dtype=np.int64)


def finetune(base_ckpt: Path, sessions: list[ExportedSession], records: list[CropRecord], seed: int):
    """LP-FT the base checkpoint on already-cut, already-capped crops.

    Same recipe as :func:`calibrate_finetune.finetune` (imports its ``_phase`` for
    both LP-FT phases, and its exact hyperparameters) but skips ``_harvest``: these
    crops were already cut with the deployed geometry by ``--export-crops``, not
    sliced from full calibrate photos here. ``sessions`` is only used to check that
    every export's recorded geometry matches the base checkpoint's.

    Returns ``(tuned, cfg, base, crops, labels, session_names, stats)`` — crops,
    labels and session_names are what was actually trained on (post empty-cap),
    for :func:`scorecard`.
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
    _check_geometry(cfg, sessions)
    base = build_model(
        cfg.model,
        pretrained=False,
        unfreeze_blocks=cfg.unfreeze_blocks,
        freeze_backbone=cfg.freeze_backbone,
    )
    base.load_state_dict(ckpt["state_dict"])
    model = deepcopy(base).to(device)

    crops, labels = _load_pixels(records)
    session_names = [r.session for r in records]
    tf = train_transform(SIZE, shape_bias=False)

    class CropSet(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(crops)

        def __getitem__(self, i: int):
            return tf(to_rgb_tensor(crops[i])), int(labels[i])

    # Same DataLoader shape as calibrate_finetune.finetune — see its comment on
    # why `fork` workers.
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
        "sessions": len(sessions),
        "head_loss_first_last": [round(head[0], 4), round(head[-1], 4)],
        "full_loss_first_last": [round(full[0], 4), round(full[-1], 4)],
        "seconds": round(time.time() - t0, 1),
    }
    return model, cfg, base.to(device).eval(), crops, labels, session_names, stats


def _probs(model, crops, device, batch: int = BATCH_SIZE):
    """(N,13) softmax probabilities for pre-cut crops, through the eval transform.

    calibrate_finetune._probs stacks every crop into ONE forward pass — fine at
    its ~448-crop scale, but this tool's pooled sets run into the low thousands
    and OOM'd a 4GB card doing exactly that. Chunked, otherwise identical.
    """
    import torch
    from augment import eval_transform, to_rgb_tensor

    tf = eval_transform(SIZE)
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            chunk = torch.stack([tf(to_rgb_tensor(c)) for c in crops[i : i + batch]]).to(device)
            out.append(torch.softmax(model(chunk), dim=1).cpu().numpy())
    import numpy as np

    return np.concatenate(out, axis=0)


def scorecard(base, tuned, crops, labels, session_names: list[str], device) -> dict:
    """Fit on the crops actually trained on (post empty-cap) — a smoke test, not
    accuracy, grouped by *session* rather than by *frame*: capping removes
    individual empty crops, not whole 64-square boards, so a fixed-size-per-unit
    grouping (calibrate_finetune.scorecard's) no longer applies."""
    import numpy as np

    p_base, p_tuned = _probs(base, crops, device), _probs(tuned, crops, device)
    names = sorted(set(session_names), key=session_names.index)
    rows = []
    for name in names:
        idx = np.array([i for i, s in enumerate(session_names) if s == name])
        y = labels[idx]
        rows.append(
            {
                "session": name,
                "n": len(idx),
                "base": int((p_base[idx].argmax(1) == y).sum()),
                "tuned": int((p_tuned[idx].argmax(1) == y).sum()),
                "averaged": int(
                    ((0.5 * p_base[idx] + 0.5 * p_tuned[idx]).argmax(1) == y).sum()
                ),
            }
        )
    total = {k: int(sum(r[k] for r in rows)) for k in ("base", "tuned", "averaged")}
    return {"per_session": rows, "total": total, "squares": len(labels)}


def print_scorecard(card: dict, out: Path, base_onnx: str) -> None:
    n = card["squares"]
    print("\n" + "=" * 78)
    print("SCORECARD — fit on the crops actually TRAINED ON (post empty-cap); this")
    print("            is a smoke test that the fine-tune took, NOT held-out accuracy")
    print("=" * 78)
    print(f"{'session':<38}{'base':>12}{'tuned':>12}{'averaged':>14}")
    for r in card["per_session"]:
        rn = r["n"]
        print(
            f"{r['session']:<38}{r['base']:>8}/{rn:<4}"
            f"{r['tuned']:>8}/{rn:<4}{r['averaged']:>10}/{rn:<4}"
        )
    t = card["total"]
    print(f"{'TOTAL':<38}{t['base']:>7}/{n}{t['tuned']:>7}/{n}{t['averaged']:>9}/{n}")
    print("=" * 78)
    print("SERVE IT AS THE AVERAGE, NEVER ALONE (see calibrate_finetune.py: the")
    print("tuned model alone was measured to hallucinate pieces onto empty squares):")
    print(f"    BR_MODEL={base_onnx}")
    print(f"    BR_MODEL_TUNED={(out / 'tuned.onnx').resolve()}")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--export-dirs", type=Path, nargs="+", required=True,
        help="one or more `eval-session --export-crops` output directories, one per session",
    )
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE,
                    help=f"base checkpoint to fine-tune (default {DEFAULT_BASE})")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help=f"train anyway below {MIN_SESSIONS} export dirs (unmeasured)")
    ap.add_argument(
        "--empty-cap-multiple", type=float, default=EMPTY_CAP_MULTIPLE,
        help="cap pooled 'empty' crops at this multiple of the largest non-empty "
             "class's count (default %(default)s; 0 disables)",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="report sessions/classes/capping and stop, before any GPU time")
    args = ap.parse_args()

    sessions = [load_session(d) for d in resolve_export_dirs(args.export_dirs)]
    print(f"{len(sessions)} export dir(s) (session(s)):")
    for s in sessions:
        print(f"  {s.name}: {len(s.records)} crops  {dict(s.class_counts.most_common())}")
    check_session_gate(sessions, force=args.force)

    all_records = [r for s in sessions for r in s.records]
    print(
        f"\ncombined: {len(all_records)} crops across {len(sessions)} session(s)  "
        f"{dict(Counter(r.label for r in all_records).most_common())}"
    )
    capped = cap_empties(all_records, multiple=args.empty_cap_multiple, seed=args.seed)
    dropped = len(all_records) - len(capped)
    if dropped:
        print(
            f"empty-cap ({args.empty_cap_multiple}x): dropped {dropped} empty crop(s) "
            f"(seed {args.seed}) -> {dict(Counter(r.label for r in capped).most_common())}"
        )
    else:
        print(f"empty-cap ({args.empty_cap_multiple}x): no cap needed")

    if args.dry_run:
        return
    if not args.base.exists():
        raise Refused(f"no base checkpoint at {args.base}")

    # Imported here, and only here, so a refusal above costs no torch import.
    import _bootstrap  # noqa: F401
    import export as export_tool

    print(f"\nfine-tuning {args.base} on {len(capped)} crops (seed {args.seed})")
    model, cfg, base, crops, labels, session_names, stats = finetune(
        args.base, sessions, capped, args.seed
    )
    print(f"  {stats}")

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "tuned.pt"
    onnx_path = args.out / "tuned.onnx"
    import torch

    # Save with the BASE's config, same reasoning as calibrate_finetune: the
    # sidecar export writes from it, so the tuned model ships the geometry it was
    # actually cropped with.
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, ckpt_path)
    print(f"wrote {ckpt_path}")
    export_tool.export(ckpt_path, onnx_path)

    device = next(model.parameters()).device
    card = scorecard(base, model, crops, labels, session_names, device)
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
                "sessions": [
                    {"name": s.name, "dir": str(s.directory), "crops": len(s.records)}
                    for s in sessions
                ],
                "min_sessions": MIN_SESSIONS,
                "forced_below_session_gate": bool(args.force and len(sessions) < MIN_SESSIONS),
                "empty_cap_multiple": args.empty_cap_multiple,
                "crops_before_cap": len(all_records),
                "crops_after_cap": len(capped),
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
