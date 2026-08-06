"""Leave-style-out training loop (config-driven).

Trains a crop -> 13-logit classifier on N-1 styles and validates on the held-out
style; with a single style available it falls back to a leave-groups-out split so
the loop, metrics, and checkpointing are still exercised end to end (the smoke
path). Mixed precision + a class-balanced sampler; batch size is the caller's knob
for the 4 GB GPU.

    python train.py --config configs/smoke.yaml
    python train.py --config configs/loso_resnet.yaml --holdout chesscog
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import _bootstrap
from board_reader_core.protocol import CLASSES, NUM_CLASSES
from board_reader_inference.squares import BboxGeometry, geometry_tag

from augment import eval_transform, train_transform
from config import Config
from cropcache import CACHE_ROOT, build_cache
from dataset import (
    CropDataset,
    class_counts_str,
    class_sample_weights,
    load_styles,
    split_by_group,
    split_by_style,
)
from metrics import SquareStats
from models import build_model, param_summary, trainable_parameters
from styles import chessred_style, discover_styles

ROOT = _bootstrap.ROOT


def pick_device(pref: str, *, min_free_mib: int = 1500) -> torch.device:
    """Resolve ``auto|cuda|cpu``. ``auto`` uses CUDA only if enough VRAM is free
    (an inference server may be holding most of a 4 GB card)."""
    if pref == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    free, _ = torch.cuda.mem_get_info()
    if free / 2**20 >= min_free_mib:
        return torch.device("cuda")
    print(f"[device] only {free/2**20:.0f} MiB free on GPU; using CPU")
    return torch.device("cpu")


def resolve_styles(cfg: Config) -> list[str]:
    available = discover_styles()
    if cfg.require_mode is not None:
        available = {n: s for n, s in available.items() if s.mode == cfg.require_mode}
    if cfg.styles == ["all"]:
        return list(available)
    missing = [s for s in cfg.styles if s not in available]
    if missing:
        raise SystemExit(f"styles not found: {missing}; available: {list(available)}")
    return list(cfg.styles)


def geo_cache_root(cfg: Config):
    """Geometry-keyed cache dir so different geometries don't collide/reuse.
    The original ``BboxGeometry(0.15, 0.15)`` maps to the default root (run (a)'s
    cache), which predates the tagging scheme."""
    geometry = cfg.geometry
    if geometry == BboxGeometry(headroom=0.15, side_pad=0.15):
        return CACHE_ROOT
    return CACHE_ROOT / geometry_tag(geometry)


def ensure_caches(names: list[str], cfg: Config) -> None:
    available = discover_styles()
    root = geo_cache_root(cfg)
    for name in names:
        style = available[name]
        if name == "chessred" and cfg.max_boards is not None:
            style = chessred_style(max_boards=cfg.max_boards)
        build_cache(style, out_root=root, size=cfg.size, geometry=cfg.geometry)


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> SquareStats:
    model.eval()
    stats = SquareStats()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        stats.update(y.numpy(), pred)
    return stats


def train(cfg: Config) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir = (ROOT / cfg.out_dir) if not Path(cfg.out_dir).is_absolute() else Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(out_dir / "config.yaml")

    names = resolve_styles(cfg)
    print(f"styles: {names} | geometry {cfg.geometry}")
    ensure_caches(names, cfg)
    data = load_styles(names, cache_root=geo_cache_root(cfg))
    print(f"loaded {len(data)} crops; classes: {class_counts_str(data.labels)}")

    if cfg.holdout is not None:
        train_idx, val_idx = split_by_style(data, cfg.holdout)
        print(f"leave-style-out: holdout={cfg.holdout}")
    elif len(set(data.style_keys.tolist())) > 1:
        # multiple styles but no holdout named: hold out the first style
        holdout = names[0]
        train_idx, val_idx = split_by_style(data, holdout)
        print(f"no holdout set; defaulting to leave-style-out holdout={holdout}")
    else:
        train_idx, val_idx = split_by_group(data, val_frac=cfg.val_frac, seed=cfg.seed)
        print(f"single style: leave-groups-out val_frac={cfg.val_frac}")
    print(f"train {len(train_idx)} / val {len(val_idx)} crops")

    device = pick_device(cfg.device)
    print(f"device: {device}")

    train_ds = CropDataset(data, train_idx, train_transform(cfg.size, shape_bias=cfg.shape_bias))
    val_ds = CropDataset(data, val_idx, eval_transform(cfg.size))

    weights = class_sample_weights(train_ds.label_slice())
    n_draw = len(train_idx) if cfg.steps_per_epoch is None else cfg.steps_per_epoch * cfg.batch_size
    sampler = WeightedRandomSampler(weights, num_samples=int(n_draw), replacement=True)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=pin, drop_last=True, persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=cfg.num_workers > 0,
    )

    model = build_model(
        cfg.model, pretrained=cfg.pretrained, init=cfg.init, sin_weights=cfg.sin_weights,
        unfreeze_blocks=cfg.unfreeze_blocks, freeze_backbone=cfg.freeze_backbone,
        num_classes=NUM_CLASSES,
    ).to(device)
    print(f"model {cfg.model}: {param_summary(model)}")

    opt = torch.optim.AdamW(trainable_parameters(model), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = len(train_loader) * cfg.epochs
    warmup = max(1, int(steps * cfg.warmup_frac))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps - warmup)),
        ],
        milestones=[warmup],
    )
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_amp = cfg.amp and device.type == "cuda"
    # GradScaler is a fp16-only device; bf16 (fp32 exponent range) neither needs nor
    # wants it. fp16 autocast overflows to NaN on this Turing GPU — hence bf16 default.
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    print(f"amp: {use_amp} dtype={cfg.amp_dtype if use_amp else 'fp32'}")
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1.0
    epochs_since_best = 0
    ckpt_path = out_dir / "best.pt"
    history: list[dict] = []
    # Tailable per-epoch val log (line-buffered, flushed each epoch) so the run can
    # be followed live with `tail -f <out_dir>/val.log`.
    val_log = open(out_dir / "val.log", "w", buffering=1)
    val_log.write(f"# {cfg.model} holdout={cfg.holdout or '(group-split)'} styles={names}\n")
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 20 == 0:
                print(f"  e{epoch} s{step}/{len(train_loader)} loss {loss.item():.4f}", flush=True)
        val = evaluate_loader(model, val_loader, device)
        dt = time.time() - t0
        train_loss = running / max(1, len(train_loader))
        line = f"epoch {epoch}: train_loss {train_loss:.4f} | val {val.summary()} | {dt:.1f}s"
        print(line, flush=True)
        val_log.write(line + "\n")
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val.to_dict()})
        if val.accuracy > best_acc:
            best_acc = val.accuracy
            epochs_since_best = 0
            torch.save(
                {"state_dict": model.state_dict(), "config": cfg.__dict__,
                 "classes": list(CLASSES), "val_accuracy": val.accuracy},
                ckpt_path,
            )
        else:
            epochs_since_best += 1
            if cfg.patience > 0 and epochs_since_best >= cfg.patience:
                stop = f"early stop: no val gain in {cfg.patience} epochs (best {best_acc:.4f})"
                print(stop, flush=True)
                val_log.write(f"# {stop}\n")
                break
    val_log.write(f"# best val accuracy {best_acc:.4f}\n")
    val_log.close()
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"best val accuracy {best_acc:.4f} -> {ckpt_path}")
    return ckpt_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--holdout", default=None, help="override config holdout style")
    ap.add_argument("--styles", nargs="+", default=None,
                    help="override config styles (explicit warm set; avoids pulling in "
                         "partial/uncached styles that ['all'] would). Include the holdout.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    if args.holdout is not None:
        cfg.holdout = args.holdout
    if args.styles is not None:
        cfg.styles = args.styles
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    train(cfg)


if __name__ == "__main__":
    main()
