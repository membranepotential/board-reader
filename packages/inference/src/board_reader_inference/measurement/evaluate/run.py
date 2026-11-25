"""Evaluate the **deployed** recognizer over every labeled board.

This runs the recognizer :func:`~board_reader_inference.recognition.recognizer.build_from_env`
builds from the environment, so the number belongs to the thing the snapshot page
talks to. The recognizer is printed above every report, because an accuracy figure
without its model is a rumour.

Truth comes from :mod:`~board_reader_inference.measurement.labels` (operator-confirmed
labels and the curated ``ground_truth.json``). A recognizer (DESIGN D10) reads the
whole photo, resolves the orientation itself and is scored only on chess-frame
labels; a corner-frame label cannot say which way its board really faces, so it
is skipped and counted.

    uv run board-reader-inference eval
    uv run board-reader-inference eval data/uploads
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from board_reader_core.protocol import CLASSES, GRID, parse_placement

from board_reader_inference.measurement.evaluate.board import (
    BoardResult,
    read_board,
    score_board,
)
from board_reader_inference.measurement.evaluate.confidence import (
    reliability,
    score_temperature,
)
from board_reader_inference.measurement.evaluate.report import Report
from board_reader_inference.measurement.labels import LabeledBoard
from board_reader_inference.recognition.recognizer import Recognizer


def describe(recognizer: Recognizer) -> str:
    """A one-line identity for the recognizer about to be measured."""
    name = type(recognizer).__name__
    return f"{name}({recognizer.model_path})" if recognizer.model_path else name


def evaluate(
    recognizer: Recognizer,
    boards: list[LabeledBoard],
    *,
    temperature: bool = False,
) -> Report:
    """Read every labeled board and assemble the report."""
    results: list[BoardResult] = []
    confidences: list[np.ndarray] = []
    hits: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    truth_indices: list[np.ndarray] = []
    skipped = [b.board_id for b in boards if b.frame != "chess"]
    for board in (b for b in boards if b.frame == "chess"):
        dist = read_board(recognizer, board)
        result = score_board(board, dist)
        results.append(result)
        truth = parse_placement(board.placement)
        truth_idx = np.array(
            [[CLASSES.index(truth[r][c]) for c in range(GRID)] for r in range(GRID)]
        )
        dists.append(dist)
        truth_indices.append(truth_idx)
        confidences.append(dist.max(axis=-1).ravel())
        hits.append((dist.argmax(axis=-1) == truth_idx).ravel())
    bins, ece = reliability(
        np.concatenate(confidences) if confidences else np.empty(0),
        np.concatenate(hits) if hits else np.empty(0, dtype=bool),
    )
    warnings: list[str] = []
    if skipped:
        warnings.append(
            f"{len(skipped)} corner-frame label(s) skipped: their truth does not say "
            "which way the board faces (tools/migrate_labels_to_chess_frame.py)"
        )
    return Report(
        boards=results,
        estimator=describe(recognizer),
        bins=bins,
        ece=ece,
        warnings=warnings,
        skipped=skipped,
        temperature=(
            score_temperature(results, dists, truth_indices) if temperature else None
        ),
    )


#: Where labeled boards live by default: the curated set plus everything the
#: snapshot page has had confirmed. Both are git-ignored, so an empty result on
#: a fresh clone is expected, not an error.
DEFAULT_PATHS = ("data/eval", "data/uploads")


def run(
    paths: list[str] | None = None,
    *,
    recognizer: Recognizer | None = None,
    temperature: bool = False,
    json_out: str | None = None,
) -> Report:
    """Load labels, build the deployed recognizer if none was injected, report."""
    from board_reader_inference.measurement.labels import load_labels

    candidates = [p for p in (paths or DEFAULT_PATHS) if Path(p).exists()]
    boards = load_labels(candidates)
    if recognizer is None:
        from board_reader_inference.recognition.recognizer import build_from_env

        recognizer = build_from_env()
    report = evaluate(recognizer, boards, temperature=temperature)
    if json_out:
        Path(json_out).write_text(json.dumps(report.to_dict(), indent=2))
    return report
