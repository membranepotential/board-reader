"""Learning-curve + diagnostic plots for a training run directory.

Re-rendered at the end of every epoch by :mod:`train` (and runnable standalone on
any finished or in-flight run, including pre-instrumentation runs — panels a run
has no data for are simply skipped):

    python plots.py scratch/runs/run_f [more run dirs...]

Outputs, per run dir:

* ``curves.png``      — train loss (per-step + per-epoch) and val accuracy /
                        occupancy accuracy with the best epoch marked.
* ``diagnostics.png`` — error-category rates (hallucinated / missed / wrong type),
                        LR schedule, and throughput per epoch.
* ``confusion.png``   — latest epoch's val confusion matrix, row-normalised
                        colour (recall), annotated with counts.

The data these are drawn from lives in ``history.json`` / ``events.jsonl`` —
those are the agent-readable record; the PNGs are the human view of the same run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
from board_reader_core.protocol import CLASSES

from runlog import read_events

# Categorical slots (validated fixed order, light mode) + text/surface tokens.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def _load_history(run_dir: Path) -> list[dict]:
    path = run_dir / "history.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _step_losses(run_dir: Path) -> tuple[list[int], list[float]]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return [], []
    gsteps, losses = [], []
    for ev in read_events(path, kind="step"):
        gsteps.append(ev["gstep"])
        losses.append(ev["loss"])
    return gsteps, losses


def _best_epoch(hist: list[dict]) -> int:
    return int(np.argmax([h["val"]["accuracy"] for h in hist]))


def _curves(run_dir: Path, hist: list[dict]) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [h["epoch"] for h in hist]

    gsteps, losses = _step_losses(run_dir)
    steps_per_epoch = max(gsteps) / max(1e-9, max(epochs) + 1) if gsteps and epochs else None
    if gsteps:
        x = [g / steps_per_epoch for g in gsteps] if steps_per_epoch else gsteps
        ax_loss.plot(x, losses, color=BLUE, alpha=0.3, linewidth=1)
    ax_loss.plot([e + 1 for e in epochs], [h["train_loss"] for h in hist],
                 color=BLUE, linewidth=2, marker="o", markersize=4, label="train loss")
    ax_loss.axhline(np.log(len(CLASSES)), color=INK2, linewidth=0.8, linestyle=":")
    ax_loss.annotate("chance (ln 13)", xy=(0.99, np.log(len(CLASSES))),
                     xycoords=("axes fraction", "data"), ha="right", va="bottom",
                     fontsize=8, color=INK2)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_title("Train loss")
    ax_loss.set_ylim(bottom=0)

    acc = [h["val"]["accuracy"] for h in hist]
    occ = [h["val"]["occupancy_accuracy"] for h in hist]
    x = [e + 1 for e in epochs]  # same convention as the loss panel: epochs completed
    ax_acc.plot(x, acc, color=BLUE, linewidth=2, marker="o", markersize=4,
                label="per-square accuracy")
    ax_acc.plot(x, occ, color=ORANGE, linewidth=2, marker="o", markersize=4,
                label="occupancy accuracy")
    b = _best_epoch(hist)
    ax_acc.axvline(x[b], color=INK2, linewidth=0.8, linestyle=":")
    ax_acc.annotate(f"best {acc[b]:.3f} @ e{epochs[b]}", xy=(x[b], acc[b]),
                    xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK2)
    ax_acc.set_xlabel("epoch")
    ax_acc.set_title("Validation (holdout style)")
    ax_acc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=120)
    plt.close(fig)


def _diagnostics(run_dir: Path, hist: list[dict]) -> None:
    fig, (ax_err, ax_lr, ax_tp) = plt.subplots(1, 3, figsize=(13, 3.6))
    epochs = [h["epoch"] + 1 for h in hist]  # epochs completed, matching curves.png

    halluc = [h["val"]["empty_to_piece"] / max(1, h["val"]["empty_total"]) for h in hist]
    missed = [h["val"]["piece_to_empty"] / max(1, h["val"]["piece_total"]) for h in hist]
    wrong = [h["val"]["piece_to_piece"] / max(1, h["val"]["piece_total"]) for h in hist]
    ax_err.plot(epochs, halluc, color=BLUE, linewidth=2, marker="o", markersize=3,
                label="hallucinated (empty→piece)")
    ax_err.plot(epochs, missed, color=ORANGE, linewidth=2, marker="o", markersize=3,
                label="missed (piece→empty)")
    ax_err.plot(epochs, wrong, color=AQUA, linewidth=2, marker="o", markersize=3,
                label="wrong type (piece→piece)")
    ax_err.set_xlabel("epoch")
    ax_err.set_title("Val error rates by category")
    ax_err.set_ylim(bottom=0)
    ax_err.legend(fontsize=7.5)

    if all("lr_end" in h for h in hist):
        ax_lr.plot(epochs, [h["lr_end"] for h in hist], color=BLUE, linewidth=2)
        ax_lr.set_title("LR at epoch end")
        ax_lr.set_xlabel("epoch")
        ax_lr.set_ylim(bottom=0)
    else:
        ax_lr.set_axis_off()

    if all("imgs_per_s" in h for h in hist):
        ax_tp.plot(epochs, [h["imgs_per_s"] for h in hist], color=BLUE, linewidth=2,
                   marker="o", markersize=3, label="throughput")
        ax_tp.set_title("Train throughput (img/s)")
        ax_tp.set_xlabel("epoch")
        ax_tp.set_ylim(bottom=0)
    else:
        ax_tp.set_axis_off()

    fig.tight_layout()
    fig.savefig(run_dir / "diagnostics.png", dpi=120)
    plt.close(fig)


def _confusion(run_dir: Path, hist: list[dict]) -> None:
    if "confusion" not in hist[-1]["val"]:
        return
    b = _best_epoch(hist)
    conf = np.asarray(hist[b]["val"]["confusion"], dtype=np.int64)
    recall = conf / np.maximum(1, conf.sum(axis=1, keepdims=True))

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.grid(False)
    im = ax.imshow(recall, cmap="Blues", vmin=0.0, vmax=1.0)
    n = len(CLASSES)
    ax.set_xticks(range(n), CLASSES)
    ax.set_yticks(range(n), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Val confusion @ best epoch {hist[b]['epoch']} (colour = row recall)")
    for i in range(n):
        for j in range(n):
            if conf[i, j] == 0:
                continue
            ax.text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if recall[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(run_dir / "confusion.png", dpi=120)
    plt.close(fig)


def render_run_plots(run_dir: Path) -> None:
    """Render/refresh all plots for one run directory (no-op if no history yet)."""
    run_dir = Path(run_dir)
    hist = _load_history(run_dir)
    if not hist:
        return
    _curves(run_dir, hist)
    _diagnostics(run_dir, hist)
    _confusion(run_dir, hist)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        render_run_plots(Path(arg))
        print(f"plots -> {arg}")


if __name__ == "__main__":
    main()
