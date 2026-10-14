"""Train the shared CoordConv GazeLNN on OSIE."""

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from project_paths import OSIE_CHECKPOINT, OSIE_DATA_ROOT
from gaze_model import CoordConv, GazeLLNArch, load_model_weights
from gaze_losses import KLDTWLoss
from OSIE_Modeldata import build_dataloaders, train, train_epoch, eval_epoch, evaluate_test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=OSIE_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=OSIE_CHECKPOINT)
    parser.add_argument("--resume", type=Path, help="Initialize from a shared-model checkpoint")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-samples", type=int, help="Explicit per-split sample cap for a smoke run")
    parser.add_argument("--no-pretrained-backbone", action="store_true", help="Random backbone initialization; for offline smoke runs")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate the best checkpoint after training")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.epochs, args.patience) <= 0 or args.num_workers < 0 or \
            not math.isfinite(args.lr) or not math.isfinite(args.gamma) or args.lr <= 0 or args.gamma <= 0:
        parser.error("Batch size, epochs, patience, learning rate and gamma must be positive; workers nonnegative")
    if args.limit_samples is not None and args.limit_samples <= 0:
        parser.error("Sample limit must be positive")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        loaders = build_dataloaders(str(args.data_root.resolve()), batch_size=args.batch_size,
                                    num_workers=args.num_workers, seed=args.seed)
        if args.limit_samples is not None:
            loaders = tuple(DataLoader(Subset(loader.dataset, range(min(args.limit_samples, len(loader.dataset)))),
                                       batch_size=args.batch_size, shuffle=i == 0, num_workers=args.num_workers)
                            for i, loader in enumerate(loaders))
        if not all(len(loader) for loader in loaders):
            raise ValueError("OSIE train/validation/test splits must all contain samples")
        model = GazeLLNArch(pretrained_backbone=not args.no_pretrained_backbone and args.resume is None).to(device)
        if args.resume:
            load_model_weights(model, args.resume, device)
        history = train(model, loaders[0], loaders[1], device=device, epochs=args.epochs, lr=args.lr,
                        patience=args.patience, gamma=args.gamma, checkpoint=args.checkpoint)
        checkpoint = args.checkpoint.expanduser().resolve()
        history_path = checkpoint.with_suffix(".history.json")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if args.evaluate_test:
            evaluate_test(model, loaders[2], device=device, checkpoint=checkpoint)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"OSIE training failed: {exc}\n")
    print(f"Checkpoint: {checkpoint}\nHistory: {history_path}")


if __name__ == "__main__":
    main()
