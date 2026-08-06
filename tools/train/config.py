"""Training configuration (YAML-backed dataclass)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

import _bootstrap
from board_reader_inference.squares import (
    BboxGeometry,
    CropGeometry,
    geometry_from_dict,
)

ROOT = _bootstrap.ROOT


@dataclass
class Config:
    # data
    styles: list[str] = field(default_factory=lambda: ["all"])  # ["all"] or explicit names
    holdout: str | None = None  # leave-style-out validation style; None -> group split
    max_boards: int | None = None  # cap ChessReD boards (smoke)
    val_frac: float = 0.25  # group-split fraction when holdout is None
    require_mode: str | None = None  # "board" | "crop": filter discovered styles by geometry

    # crop geometry (crop_squares_photo). Two ways to say it, and the cache is
    # keyed by whichever applies so variants never collide:
    #
    #   headroom / side_pad          bbox geometry, x square-width. headroom
    #                                extends UPWARD in image space; 0.15 (the
    #                                kNN-era default) decapitates tall pieces at
    #                                steep angles, 1.6 is the deployed value.
    #   crop_geometry: {kind: ...}   the full CropGeometry record — the only way
    #                                to ask for pose geometry, e.g.
    #                                {kind: pose, piece_height: 1.9,
    #                                 piece_radius: 0.5}. Wins when present.
    #
    # Both exist because every run before 2026-08-07 recorded the scalars, and
    # their checkpoints must keep loading.
    headroom: float = 0.15
    side_pad: float = 0.15
    crop_geometry: dict | None = None

    # model
    model: str = "resnet18"  # resnet18 | resnet50 | dinov2
    pretrained: bool = True  # resnet: back-compat (False -> scratch)
    init: str = "imagenet"  # resnet init: imagenet | scratch | sin
    sin_weights: str | None = None  # path to Stylized-ImageNet resnet50 checkpoint (init=sin)
    unfreeze_blocks: int = 2  # dinov2: last-N transformer blocks
    freeze_backbone: bool = True  # dinov2: freeze the rest

    # augmentation
    shape_bias: bool = True  # texture-destroying + AdaIN-approx stylization

    # optim
    epochs: int = 12
    patience: int = 0  # early-stop after N epochs with no val-accuracy gain (0 = off)
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.05

    # runtime
    device: str = "auto"  # auto | cuda | cpu (auto -> cuda only if enough free VRAM)
    amp: bool = True
    amp_dtype: str = "bfloat16"  # bfloat16 | float16 — fp16 autocast OVERFLOWS to NaN on
    # GTX 1650 (Turing); bf16 has fp32 exponent range, no NaN, no loss-scaler, half the memory.
    num_workers: int = 6
    seed: int = 0
    size: int = 224
    steps_per_epoch: int | None = None  # cap for smoke runs
    out_dir: str = "scratch/runs/run"

    @property
    def geometry(self) -> CropGeometry:
        """The crop geometry this run trains (and therefore must serve) with."""
        if self.crop_geometry is not None:
            return geometry_from_dict(self.crop_geometry)
        return BboxGeometry(headroom=self.headroom, side_pad=self.side_pad)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**known)

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
