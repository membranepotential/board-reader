"""The estimator tiers that served before the recognizer seam (DESIGN D10).

Selected by :func:`board_reader_inference.recognizer.build_from_env` when
``BR_CHESSQ`` is unset: an explicit calibration (``BR_CENTROIDS``), the trained
crop classifier (``BR_MODEL``, with ``BR_MODEL_B`` / ``BR_MODEL_TUNED``) boosted by
a persisted session bank, a bank alone, or the M0 :class:`StubEstimator`. Moved
here unchanged from the gateway; Phase B of the simplification plan deletes it.
"""

from __future__ import annotations

import os
from pathlib import Path

from board_reader_inference.estimator import CentroidEstimator, Estimator, StubEstimator
from board_reader_inference.session import booster_weight, session_bank_path
from board_reader_inference.squares import BboxGeometry
from board_reader_inference.trained import (
    DEFAULT_CASCADE_TAU,
    AveragedEstimator,
    CascadeEstimator,
    TrainedEstimator,
    TrainedReader,
)


def estimator_from_bank(
    path: str, *, share_from: Estimator | None = None
) -> CentroidEstimator:
    """Build a :class:`CentroidEstimator` from a saved calibration file.

    Rebuilds the same backbone and box geometry the bank was calibrated with
    (dims must match; photo boxes must be cut identically). If ``share_from`` is
    an estimator on the same backbone, its embedder is reused — one ONNX session
    instead of two, which matters on a 4GB GPU.
    """
    from board_reader_inference.centroid import load_classifier
    from board_reader_inference.embed import make_embedder

    classifier = load_classifier(path)
    shared = getattr(share_from, "embedder", None)
    shared_backbone = getattr(getattr(share_from, "classifier", None), "backbone", None)
    embedder = shared if shared is not None and shared_backbone == classifier.backbone else None
    return CentroidEstimator(
        embedder if embedder is not None else make_embedder(classifier.backbone),
        classifier,
        geometry=BboxGeometry(
            headroom=classifier.box_headroom, side_pad=classifier.box_side_pad
        ),
    )


def _cascade_tau() -> float:
    """``BR_CASCADE_TAU`` -> the cascade's confidence gate (default 0.95)."""
    raw = os.environ.get("BR_CASCADE_TAU")
    if not raw:
        return DEFAULT_CASCADE_TAU
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"BR_CASCADE_TAU={raw!r} is not a number") from e


def _trained_component(model: str) -> TrainedReader | None:
    """The zero-shot trained member named by the environment, or ``None``.

    ``BR_MODEL`` alone is one :class:`TrainedEstimator`, unchanged. Exactly one
    companion may join it, and the choice is between *measured* configurations:

    * ``BR_MODEL_B`` — a second, differently-cropped classifier; the pair becomes
      a :class:`CascadeEstimator` (``BR_CASCADE_TAU`` moves the gate). Measured
      0.956 on user_photos, zero-shot on any board.
    * ``BR_MODEL_TUNED`` — a calibration fine-tune of ``BR_MODEL`` produced by
      ``tools/train/calibrate_finetune.py``; the pair becomes an
      :class:`AveragedEstimator`. Measured 0.969 — but only on the board it was
      calibrated for.

    Either way the result is *the trained component*: a session bank still wraps
    it in an ``EnsembleEstimator`` on top, with the same arithmetic.

    A broken ``BR_MODEL`` degrades to the next tier so ``docker compose up`` works
    on a fresh clone without ``models/`` — and with the cascade in compose.yaml
    that courtesy has one more shape: **both** model files absent is the fresh
    clone again, so it degrades too. Any *partial* pair (one file present, or a
    file that exists but will not load) is a misconfiguration and fails hard:
    quietly serving one model instead of the requested two would show up only as
    "the ensemble didn't help" months later. ``BR_MODEL_TUNED`` gets no
    fresh-clone courtesy at all — a tuned model only exists because somebody ran
    the tool, so setting it is always deliberate, and the tuned model *alone*
    hallucinates pieces onto empty squares (it is the base's vote in the average
    that suppresses them).
    """
    model_b = os.environ.get("BR_MODEL_B")
    tuned = os.environ.get("BR_MODEL_TUNED")
    if model_b and tuned:
        # Neither the cascade-of-three nor "which pair wins" has been measured,
        # and the two configurations answer different questions (zero-shot
        # breadth vs one calibrated board). Guessing here would silently ship an
        # unmeasured model.
        raise RuntimeError(
            "BR_MODEL_B and BR_MODEL_TUNED are both set; stacking them is an "
            "UNMEASURED configuration. Pick one: BR_MODEL_B is the zero-shot "
            "cascade (measured 0.956 on user_photos, any board), BR_MODEL_TUNED "
            "is the calibration average (measured 0.969, but only on the board "
            "its frames came from)."
        )
    try:
        trained = TrainedEstimator(model)
    except Exception as e:
        if tuned:
            raise RuntimeError(
                f"BR_MODEL_TUNED={tuned!r} requests the calibration average, but "
                f"its base BR_MODEL={model!r} is not usable ({e}). The average "
                "needs both members — the fine-tuned model alone was measured to "
                "hallucinate pieces onto empty squares."
            ) from e
        if model_b:
            if not Path(model).exists() and not Path(model_b).exists():
                print(
                    f"BR_MODEL={model!r} and BR_MODEL_B={model_b!r} both absent "
                    "(fresh clone?); serving without the trained component"
                )
                return None
            raise RuntimeError(
                f"BR_MODEL_B={model_b!r} requests the cascade, but its first member "
                f"BR_MODEL={model!r} is not usable ({e})"
            ) from e
        print(f"BR_MODEL={model!r} not usable ({e}); serving without it")
        return None
    if tuned:
        return AveragedEstimator(trained, TrainedEstimator(tuned))
    if not model_b:
        return trained
    return CascadeEstimator(trained, TrainedEstimator(model_b), tau=_cascade_tau())


def legacy_estimator(occupancy_est: CentroidEstimator | None = None) -> Estimator:
    """Pick the style (piece-type) estimator from the environment.

    ``BR_CENTROIDS=<path>`` -> that calibration. Unset but auto-calibration
    enabled (an occupancy estimator exists) -> restore the persisted session bank
    if a previous run calibrated one. Otherwise the M0 :class:`StubEstimator`.
    """
    path = os.environ.get("BR_CENTROIDS")
    if path:
        return estimator_from_bank(path, share_from=occupancy_est)
    # The multi-style-trained crop classifier: the zero-shot default
    # (2026-07-18 campaign). Explicit BR_CENTROIDS still outranks it. A persisted
    # session bank does NOT replace it — it boosts it: the 50/50 distribution
    # blend beats either member alone (439 vs 415/430 of 448 on the labeled
    # eval), because the bank is near-perfect in conditions it has seen and the
    # trained model degrades gracefully in ones it hasn't.
    model = os.environ.get("BR_MODEL")
    if not model and os.environ.get("BR_MODEL_TUNED"):
        # The average needs its base. Falling through to the stub here would
        # serve a fixed placement while the operator believes a calibrated model
        # is running (see _trained_component on why the tuned model has no solo
        # mode at all).
        raise RuntimeError(
            f"BR_MODEL_TUNED={os.environ['BR_MODEL_TUNED']!r} is set but BR_MODEL "
            "is not. The calibration average is the tuned model 50/50 with the "
            "base it was fine-tuned from; set BR_MODEL to that base."
        )
    if model:
        from board_reader_inference.estimator import EnsembleEstimator

        trained = _trained_component(model)
        if trained is not None:
            bank = session_bank_path()
            if bank and Path(bank).exists():
                try:
                    bank_est = estimator_from_bank(bank, share_from=occupancy_est)
                    return EnsembleEstimator(
                        trained,
                        bank_est,
                        weight=booster_weight(bank_est.classifier.exemplars.shape[0]),
                    )
                except (OSError, ValueError, KeyError):
                    pass  # unreadable/stale bank: zero-shot still works without it
            return trained
    bank = session_bank_path()
    if occupancy_est is not None and bank and Path(bank).exists():
        return estimator_from_bank(bank, share_from=occupancy_est)
    return StubEstimator()
