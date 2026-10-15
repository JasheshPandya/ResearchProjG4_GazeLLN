"""Train the shared CoordConv GazeLNN on OSIE."""

import argparse
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from project_paths import OSIE_CHECKPOINT, OSIE_DATA_ROOT, OUTPUT_FILE_DIR
from checkpoint_provenance import architecture_record, osie_loader_record
from run_exports import RunExport, export_training_results
from gaze_model import CoordConv, GazeLLNArch, load_model_weights
from gaze_losses import KLDTWLoss
from OSIE_Modeldata import build_dataloaders, train, train_epoch, eval_epoch, evaluate_test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=OSIE_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=OSIE_CHECKPOINT)
    parser.add_argument("--output-dir", "--out-dir", type=Path, help="Default: shared output/osie_training/<checkpoint stem>")
    parser.add_argument("--resume", type=Path, help="Initialize from a shared-model checkpoint")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
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
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be in [0, 2**32)")
    for key in ("data_root", "checkpoint", "resume"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    args.output_dir = (args.output_dir or OUTPUT_FILE_DIR / "osie_training" / args.checkpoint.stem).expanduser().resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        run = RunExport(args.output_dir, "osie_training",
                        dict(arguments=vars(args), resolved_device=device,
                             pretrained_backbone=not args.no_pretrained_backbone and args.resume is None),
                        sources=[Path(__file__), Path(__file__).with_name("OSIE_Modeldata.py"),
                                 Path(__file__).with_name("gaze_model.py"), Path(__file__).with_name("gaze_losses.py"),
                                 Path(__file__).with_name("gaze_preprocessing.py"), Path(__file__).with_name("run_exports.py")])
        with run.run():
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
            run.update(architecture=architecture_record(model),
                       initialization=getattr(model, "_initialization_provenance", None),
                       train=osie_loader_record(loaders[0]), val=osie_loader_record(loaders[1]),
                       test=osie_loader_record(loaders[2]),
                       input_preprocessing=getattr(getattr(loaders[0].dataset, "dataset", loaders[0].dataset), "input_preprocessing", None))
            history = train(model, loaders[0], loaders[1], device=device, epochs=args.epochs, lr=args.lr,
                            patience=args.patience, gamma=args.gamma, checkpoint=args.checkpoint,
                            run_settings=dict(**vars(args), run_config=str(run.path), run_id=run.record["run_id"]), test_loader=loaders[2])
            test_loss = evaluate_test(model, loaders[2], device=device, checkpoint=args.checkpoint) if args.evaluate_test else None
            export_training_results(run, args.checkpoint, history, test_loss=test_loss)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"OSIE training failed: {exc}\n")
    print(f"Checkpoint: {args.checkpoint}\nHistory: {args.checkpoint.with_suffix('.history.json')}\nTraining provenance: {args.checkpoint}.training.json\nRun configuration: {run.path}\nResults: {run.output_dir / 'results.json'}")
    return history


if __name__ == "__main__":
    main()
