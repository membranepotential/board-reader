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

import _bootstrap
import numpy as np
import torch
import torch.nn as nn
from augment import eval_transform, train_transform
from config import Config
from cropcache import CACHE_ROOT, ensure_caches, geo_cache_root, resolve_styles
from dataset import (
    CropDataset,
    class_counts_str,
    class_sample_weights,
    load_styles,
    loader_worker_init,
    split_by_group,
    split_by_style,
    subsample_val,
)
from metrics import SquareStats
from runlog import RunLog
from torch.utils.data import DataLoader, WeightedRandomSampler

from board_reader_core.protocol import CLASSES, NUM_CLASSES
from models import build_model, param_summary, trainable_parameters

from board_reader_inference.squares import BboxGeometry, geometry_tag

try:
    from plots import render_run_plots
except ImportError:  # matplotlib not installed — train without pictures
    render_run_plots = None

ROOT = _bootstrap.ROOT


def pick_device(pref: str, *, min_free_mib: int = 1500) -> torch.device:
    """Resolve ``auto|cuda|cpu``. ``auto`` refuses to run when the GPU exists but
    is too full (an inference server may be holding most of the 4 GB card): a
    silent CPU fallback once turned a ~45-minute run into a days-long one, so
    that degradation is now an abort — force ``device: cpu`` to accept it."""
    if pref == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    free, _ = torch.cuda.mem_get_info()
    if free / 2**20 >= min_free_mib:
        return torch.device("cuda")
    raise SystemExit(
        f"[device] only {free/2**20:.0f} MiB free on the GPU (< {min_free_mib}); "
        "refusing the silent CPU fallback — free the GPU or set device: cpu"
    )


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module, loader: DataLoader, device: torch.device, *, channels_last: bool = False
) -> SquareStats:
    model.eval()
    stats = SquareStats()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
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
    if cfg.val_max is not None:
        full_val = len(val_idx)
        val_idx = subsample_val(val_idx, data.labels, cfg.val_max, seed=cfg.seed)
        if len(val_idx) < full_val:
            print(f"val capped {full_val} -> {len(val_idx)} crops (val_max) — SCREENING, not a score")
    print(f"train {len(train_idx)} / val {len(val_idx)} crops")

    device = pick_device(cfg.device)
    print(f"device: {device}")
    if device.type == "cuda":
        # Input shape is fixed (B,3,size,size); let cuDNN autotune its kernels once.
        torch.backends.cudnn.benchmark = True

    train_ds = CropDataset(data, train_idx, train_transform(cfg.size, shape_bias=cfg.shape_bias))
    val_ds = CropDataset(data, val_idx, eval_transform(cfg.size))

    weights = class_sample_weights(train_ds.label_slice())
    n_draw = len(train_idx) if cfg.steps_per_epoch is None else cfg.steps_per_epoch * cfg.batch_size
    sampler = WeightedRandomSampler(weights, num_samples=int(n_draw), replacement=True)
    pin = device.type == "cuda"
    # Workers MUST fork: the dataset is ~9 GB of RAM-resident JPEG bytes, and fork
    # shares it copy-on-write. Python 3.14 changed the default start method to
    # forkserver, which instead PICKLES a full copy into every worker — 8 workers
    # x 9 GB OOM-killed the first run_f launch. Workers never touch CUDA, so
    # forking after CUDA init is safe here.
    mp_ctx = "fork" if "fork" in torch.multiprocessing.get_all_start_methods() else None
    workers = {
        "num_workers": cfg.num_workers, "pin_memory": pin, "persistent_workers": cfg.num_workers > 0,
        "worker_init_fn": loader_worker_init,
        "prefetch_factor": cfg.prefetch_factor if cfg.num_workers > 0 else None,
        "multiprocessing_context": mp_ctx if cfg.num_workers > 0 else None,
    }
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=True, **workers,
    )
    # No backward pass in eval -> activations only; 2x the train batch is safely
    # inside even the 4 GB card's budget and halves the per-epoch val wall time.
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, **workers)

    model = build_model(
        cfg.model, pretrained=cfg.pretrained, init=cfg.init, sin_weights=cfg.sin_weights,
        unfreeze_blocks=cfg.unfreeze_blocks, freeze_backbone=cfg.freeze_backbone,
        num_classes=NUM_CLASSES,
    ).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    # Checkpoints save eager_model's state_dict — torch.compile wraps the module and
    # would otherwise prefix every key with `_orig_mod.`; the parameters are shared.
    eager_model = model
    if cfg.compile:
        model = torch.compile(model)
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
    best_epoch = -1
    epochs_since_best = 0
    stopped_early = False
    ckpt_path = out_dir / "best.pt"
    history: list[dict] = []
    n_steps = len(train_loader)
    # Two live views of the same run: val.log is the tailable human one
    # (`tail -f <out_dir>/val.log`), events.jsonl the machine-readable one.
    val_log = open(out_dir / "val.log", "w", buffering=1)  # noqa: SIM115 — tailable
    # for the whole run; a `with` would scope it to less than the training loop.
    val_log.write(f"# {cfg.model} holdout={cfg.holdout or '(group-split)'} styles={names}\n")
    run_log = RunLog(out_dir / "events.jsonl")
    run_log.log(
        "meta", config=cfg.__dict__, device=str(device), styles=names,
        n_train=len(train_idx), n_val=len(val_idx), steps_per_epoch=n_steps,
        model=cfg.model, params=param_summary(model),
    )

    def flush_history() -> None:
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    def gpu_mem_mib() -> int:
        return int(torch.cuda.memory_allocated() / 2**20) if device.type == "cuda" else 0

    for epoch in range(cfg.epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        t_log, s_log = t0, -1  # last step-record time / step index
        running = torch.zeros((), device=device)  # GPU-side sum: no per-step sync
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            if cfg.channels_last:
                x = x.to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.detach()
            if step % cfg.log_every == 0 or step == n_steps - 1:
                cur = loss.item()  # the one deliberate GPU sync per interval
                if not np.isfinite(cur):
                    msg = f"non-finite loss {cur} at epoch {epoch} step {step} — aborting"
                    run_log.log("abort", epoch=epoch, step=step, loss=cur, reason="non-finite loss")
                    val_log.write(f"# {msg}\n")
                    flush_history()
                    raise SystemExit(msg)
                now = time.time()
                imgs_s = (step - s_log) * cfg.batch_size / max(1e-6, now - t_log)
                lr_now = opt.param_groups[0]["lr"]
                print(
                    f"  e{epoch} s{step}/{n_steps} loss {cur:.4f} lr {lr_now:.2e} "
                    f"{imgs_s:.0f} img/s", flush=True,
                )
                run_log.log(
                    "step", epoch=epoch, step=step, gstep=epoch * n_steps + step,
                    loss=round(cur, 5), lr=lr_now, imgs_per_s=round(imgs_s, 1),
                    gpu_mem_mib=gpu_mem_mib(),
                )
                t_log, s_log = now, step
        t_train = time.time()
        val = evaluate_loader(model, val_loader, device, channels_last=cfg.channels_last)
        t_val = time.time()
        train_loss = (running / max(1, n_steps)).item()
        train_secs = t_train - t0
        imgs_s = n_steps * cfg.batch_size / max(1e-6, train_secs)
        peak_mib = (
            int(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0
        )
        line = (
            f"epoch {epoch}: train_loss {train_loss:.4f} | val {val.summary()} | "
            f"{train_secs:.1f}s train + {t_val - t_train:.1f}s val | {imgs_s:.0f} img/s"
        )
        print(line, flush=True)
        val_log.write(line + "\n")
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "lr_end": opt.param_groups[0]["lr"], "train_secs": round(train_secs, 2),
            "val_secs": round(t_val - t_train, 2), "imgs_per_s": round(imgs_s, 1),
            "gpu_mem_peak_mib": peak_mib, "val": val.to_dict(),
        })
        flush_history()  # crash-safe: every finished epoch is on disk
        run_log.log(
            "epoch", epoch=epoch, train_loss=round(train_loss, 5),
            train_secs=round(train_secs, 2), val_secs=round(t_val - t_train, 2),
            imgs_per_s=round(imgs_s, 1), gpu_mem_peak_mib=peak_mib, val=val.to_dict(),
        )
        if render_run_plots is not None:
            try:
                render_run_plots(out_dir)
            except Exception as e:  # plots must never kill a training run
                print(f"[plots] render failed: {e}", flush=True)
        if val.accuracy > best_acc:
            best_acc = val.accuracy
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(
                {"state_dict": eager_model.state_dict(), "config": cfg.__dict__,
                 "classes": list(CLASSES), "val_accuracy": val.accuracy},
                ckpt_path,
            )
        else:
            epochs_since_best += 1
            if cfg.patience > 0 and epochs_since_best >= cfg.patience:
                stop = f"early stop: no val gain in {cfg.patience} epochs (best {best_acc:.4f})"
                print(stop, flush=True)
                val_log.write(f"# {stop}\n")
                stopped_early = True
                break
    val_log.write(f"# best val accuracy {best_acc:.4f} (epoch {best_epoch})\n")
    val_log.close()
    run_log.log("end", best_acc=round(best_acc, 4), best_epoch=best_epoch,
                epochs_run=len(history), stopped_early=stopped_early)
    run_log.close()
    print(f"best val accuracy {best_acc:.4f} (epoch {best_epoch}) -> {ckpt_path}")
    return ckpt_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--holdout", default=None, help="override config holdout style")
    ap.add_argument("--styles", nargs="+", default=None,
                    help="override config styles (explicit warm set; avoids pulling in "
                         "partial/uncached styles that ['all'] would). Include the holdout.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stop after N epochs with no val gain (0 disables)")
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--val-max", type=int, default=None,
                    help="cap val to N crops (stratified). Screening only — a capped "
                         "val accuracy ranks candidates, it does not score them.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mask", choices=["gray", "blur"], default=None,
                    help="override config occlusion mask (E7)")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    if args.holdout is not None:
        cfg.holdout = args.holdout
    if args.styles is not None:
        cfg.styles = args.styles
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.patience is not None:
        cfg.patience = args.patience
    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.val_max is not None:
        cfg.val_max = args.val_max
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.mask is not None:
        cfg.mask = args.mask
    train(cfg)


if __name__ == "__main__":
    main()
