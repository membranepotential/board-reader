"""Augmentation pipelines (torchvision transforms v2).

The harness must generalise to unseen chess sets under unseen lighting, so the
train pipeline is deliberately heavy on the nuisance factors that separate styles
and rooms — colour/white-balance, brightness down to *very dim* (the user's own
eval shots are dim), blur, and geometric jitter (small rotation + perspective, to
stand in for corner-detection slop and camera-angle spread) — while leaving piece
identity intact. Eval is a plain resize + ImageNet normalise.

**Shape-bias track (DESIGN motivation + Geirhos et al. ICLR'19).** Our "styles"
are texture/material variations of near-identical *shapes*, the textbook
texture-vs-shape-bias case, and CNNs default to texture. So the ``shape_bias``
pipeline layers texture-destroying ops (channel shuffle, random grayscale,
posterize/solarize, gaussian noise) plus a hand-rolled AdaIN approximation
(:class:`RandomColorStats`: instance-normalise per channel, then re-scale to random
colour statistics) on top of the base geometry/lighting augmentation. Stylization
alone is near-SOTA for domain generalization (arXiv:2006.11207); this is the
cheap, VGG-free version. The grayscale diagnostic in :mod:`evaluate` measures the
resulting shape bias directly.

Inputs are decoded BGR crops (OpenCV order); :func:`to_rgb_tensor` handles the
BGR->RGB flip so the rest of the pipeline is standard torchvision.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_rgb_tensor(bgr: np.ndarray) -> torch.Tensor:
    """Decoded BGR uint8 HxWx3 -> CxHxW uint8 RGB tensor (transforms v2 input)."""
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return torch.from_numpy(rgb).permute(2, 0, 1)


class RandomGamma(torch.nn.Module):
    """Random gamma on a normalised-to-[0,1] float tensor; pushes toward dim."""

    def __init__(self, gamma_range: tuple[float, float] = (0.6, 1.8), p: float = 0.5) -> None:
        super().__init__()
        self.lo, self.hi = gamma_range
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return x
        g = torch.empty(()).uniform_(self.lo, self.hi).item()
        return x.clamp_min(0.0).pow(g)


class RandomChannelShuffle(torch.nn.Module):
    """Randomly permute RGB channels — destroys the set's specific colour identity
    while preserving shape/contrast (a cheap, strong texture/style perturbation)."""

    def __init__(self, p: float = 0.3) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return x
        return x[torch.randperm(3)]


class RandomColorStats(torch.nn.Module):
    """AdaIN-style colour restyling (VGG-free): per-channel instance-normalise on a
    [0,1] float tensor, then re-scale to random target mean/std. Transfers a random
    colour "style" onto the crop's shape content — the pixel-space approximation of
    the stylization that drives shape bias (Geirhos et al.; arXiv:2006.11207)."""

    def __init__(self, p: float = 0.4, mean_range=(0.25, 0.75), std_range=(0.12, 0.35)) -> None:
        super().__init__()
        self.p = p
        self.mean_range = mean_range
        self.std_range = std_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return x
        mu = x.mean(dim=(1, 2), keepdim=True)
        sd = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
        tgt_mu = torch.empty(3, 1, 1).uniform_(*self.mean_range)
        tgt_sd = torch.empty(3, 1, 1).uniform_(*self.std_range)
        return (((x - mu) / sd) * tgt_sd + tgt_mu).clamp(0.0, 1.0)


def train_transform(size: int = 224, *, shape_bias: bool = True) -> v2.Transform:
    """Heavy augmentation for training crops.

    ``shape_bias`` adds the texture-destroying + AdaIN-approx stage (recommended for
    cross-style generalisation); set False for a plain lighting/geometry baseline.
    """
    geo_light = [
        v2.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.RandomApply([v2.RandomRotation(degrees=10)], p=0.5),
        v2.RandomPerspective(distortion_scale=0.25, p=0.5),
        v2.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.4, hue=0.05),
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5))], p=0.4),
    ]
    texture: list = []
    if shape_bias:
        texture = [
            RandomChannelShuffle(p=0.3),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.RandomPosterize(bits=4)], p=0.2),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
        ]
    tail = [
        v2.ToDtype(torch.float32, scale=True),
        RandomGamma((0.5, 2.0), p=0.5),  # dim/bright beyond linear brightness
    ]
    if shape_bias:
        tail += [
            RandomColorStats(p=0.4),  # AdaIN-approx stylization
            v2.RandomApply([v2.GaussianNoise(sigma=0.05)], p=0.3),
        ]
    tail += [
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ]
    return v2.Compose([*geo_light, *texture, *tail])


def eval_transform(size: int = 224, *, grayscale: bool = False) -> v2.Transform:
    """Deterministic resize + normalise for validation/test/inference parity.

    ``grayscale=True`` is the shape-bias diagnostic: a shape-biased model should
    barely drop when colour/texture cues are removed (channels kept = 3)."""
    ops: list = [v2.Resize((size, size), antialias=True)]
    if grayscale:
        ops.append(v2.Grayscale(num_output_channels=3))
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return v2.Compose(ops)
