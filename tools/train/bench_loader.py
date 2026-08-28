"""Loader-ceiling benchmark (Phase 2.2 of the RunPod migration plan).

Times the *training* DataLoader with the model removed: decode, augment,
collate, pin, H2D copy — everything except forward/backward. The number this
prints is the hard ceiling any train-loop img/s can reach. Locally the loader
only ever had to keep up with ~170 img/s; at cloud speeds it is the prime
suspect, so measure it before believing any headline speedup.

    python bench_loader.py --config configs/run_g_cloud.yaml --workers 4 8 16 24
"""

from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401  — sys.path for the package imports
import torch
from augment import train_transform
from config import Config
from cropcache import geo_cache_root, resolve_styles
from dataset import (
    CropDataset,
    class_sample_weights,
    load_styles,
    loader_worker_init,
)
from torch.utils.data import DataLoader, WeightedRandomSampler


def bench_workers(
    ds: CropDataset, cfg: Config, n_workers: int, *, warmup: int, timed: int
) -> float:
    """img/s over ``timed`` batches, after ``warmup`` batches of spin-up."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    weights = class_sample_weights(ds.label_slice())
    sampler = WeightedRandomSampler(
        weights, num_samples=(warmup + timed + 2) * cfg.batch_size, replacement=True
    )
    # Mirror train.py's loader exactly — fork is load-bearing (COW-shares the
    # ~9 GB of resident JPEG bytes; forkserver pickles a copy per worker).
    mp_ctx = "fork" if "fork" in torch.multiprocessing.get_all_start_methods() else None
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=True,
        num_workers=n_workers, pin_memory=device.type == "cuda",
        persistent_workers=n_workers > 0, worker_init_fn=loader_worker_init,
        prefetch_factor=cfg.prefetch_factor if n_workers > 0 else None,
        multiprocessing_context=mp_ctx if n_workers > 0 else None,
    )
    it = iter(loader)
    for _ in range(warmup):
        x, _y = next(it)
        x.to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(timed):
        x, _y = next(it)
        x.to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    del it, loader  # join the persistent workers before the next point spawns its own
    return timed * cfg.batch_size / max(1e-6, time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16, 24])
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--timed", type=int, default=300)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    names = resolve_styles(cfg)
    data = load_styles(names, cache_root=geo_cache_root(cfg))
    ds = CropDataset(
        data, list(range(len(data))), train_transform(cfg.size, shape_bias=cfg.shape_bias)
    )
    print(f"loader ceiling: {len(ds)} crops, batch {cfg.batch_size}, "
          f"prefetch {cfg.prefetch_factor}, {args.timed} timed batches")
    for n in args.workers:
        rate = bench_workers(ds, cfg, n, warmup=args.warmup, timed=args.timed)
        print(f"  workers {n:>2}: {rate:7.0f} img/s", flush=True)


if __name__ == "__main__":
    main()
