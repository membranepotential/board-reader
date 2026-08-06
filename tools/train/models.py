"""Two crop -> 13-logit model recipes, both ONNX-exportable.

Both take a 224x224 ImageNet-normalised RGB crop and emit 13 logits over
:data:`board_reader_core.protocol.CLASSES`. That is the classifier seam we propose
to slot into the inference node: a single ONNX model replacing the
embed->centroid split (softmax of its logits IS the per-square distribution).

* ``resnet18`` / ``resnet50`` — ResNet (ImageNet-init, from scratch, or
  Stylized-ImageNet init), fc -> 13. The strong, cheap baseline; learns
  style-invariant features when trained across styles. SIN init (Geirhos et al.,
  rgeirhos/texture-vs-shape ResNet-50) starts already shape-biased.
* ``dinov2`` — official Meta ``dinov2_vits14`` backbone (torch.hub, Apache-2.0)
  with a linear head. Fine-tunes the last N transformer blocks + head (the
  dependency-free stand-in for LoRA; flip ``freeze_backbone`` for a pure linear
  probe). The backbone stays the deployed one, so features transfer to the ONNX
  embedder path if we ever want to.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision

import _bootstrap  # noqa: F401
from board_reader_core.protocol import NUM_CLASSES

MODELS = ("resnet18", "resnet50", "dinov2")
DINOV2_HUB = ("facebookresearch/dinov2", "dinov2_vits14")

# init modes for the ResNet recipe.
RESNET_INITS = ("imagenet", "scratch", "sin")


def _load_sin_weights(model: nn.Module, weights_path: str | Path) -> None:
    """Load Stylized-ImageNet ResNet-50 weights (rgeirhos/texture-vs-shape).

    Those checkpoints are DataParallel-wrapped (``module.`` prefix) and include the
    original 1000-class fc; we strip the prefix and drop fc (replaced with our head).
    """
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    missing, _ = model.load_state_dict(sd, strict=False)
    kept = [m for m in missing if not m.startswith("fc.")]
    if kept:
        raise RuntimeError(f"SIN weights missing non-fc keys: {kept[:5]}...")


def build_resnet(
    arch: str = "resnet18",
    *,
    init: str = "imagenet",
    sin_weights: str | Path | None = None,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Build a ResNet with the fc head replaced by ``num_classes`` logits.

    ``init``: ``imagenet`` (torchvision weights), ``scratch`` (random), or ``sin``
    (load ``sin_weights`` — a Stylized-ImageNet ResNet-50 checkpoint).
    """
    if init not in RESNET_INITS:
        raise ValueError(f"init must be one of {RESNET_INITS}")
    ctor = {"resnet18": torchvision.models.resnet18, "resnet50": torchvision.models.resnet50}[arch]
    weights_enum = {
        "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
    }[arch]
    model = ctor(weights=weights_enum if init == "imagenet" else None)
    if init == "sin":
        if arch != "resnet50":
            raise ValueError("SIN weights are ResNet-50 only")
        if not sin_weights:
            raise ValueError("init='sin' requires sin_weights path")
        _load_sin_weights(model, sin_weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class DinoV2Classifier(nn.Module):
    """Official DINOv2 ViT-S/14 backbone + linear head -> logits.

    ``forward`` returns raw logits (B, num_classes). Fine-tuning granularity is set
    at construction: ``freeze_backbone=True`` + ``unfreeze_blocks=0`` is a linear
    probe; ``unfreeze_blocks=n`` thaws the last ``n`` transformer blocks (+ final
    norm) — our LoRA-free last-blocks recipe.
    """

    def __init__(
        self,
        *,
        num_classes: int = NUM_CLASSES,
        unfreeze_blocks: int = 2,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = torch.hub.load(*DINOV2_HUB, verbose=False)
        self.embed_dim = int(self.backbone.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_blocks > 0:
                for blk in self.backbone.blocks[-unfreeze_blocks:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # (B, embed_dim) CLS embedding
        return self.head(feats)


def build_dinov2(
    *, unfreeze_blocks: int = 2, freeze_backbone: bool = True, num_classes: int = NUM_CLASSES
) -> nn.Module:
    return DinoV2Classifier(
        num_classes=num_classes,
        unfreeze_blocks=unfreeze_blocks,
        freeze_backbone=freeze_backbone,
    )


def build_model(name: str, **kw) -> nn.Module:
    """Construct a recipe by name. Unknown kwargs are ignored per-recipe."""
    if name in ("resnet18", "resnet50"):
        # `pretrained` kept for back-compat: False -> scratch, True -> `init` (default imagenet).
        init = kw.get("init", "imagenet")
        if kw.get("pretrained", True) is False and "init" not in kw:
            init = "scratch"
        return build_resnet(
            name,
            init=init,
            sin_weights=kw.get("sin_weights"),
            num_classes=kw.get("num_classes", NUM_CLASSES),
        )
    if name == "dinov2":
        return build_dinov2(
            unfreeze_blocks=kw.get("unfreeze_blocks", 2),
            freeze_backbone=kw.get("freeze_backbone", True),
            num_classes=kw.get("num_classes", NUM_CLASSES),
        )
    raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def param_summary(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in trainable_parameters(model))
    return f"{train/1e6:.2f}M trainable / {total/1e6:.2f}M total"
