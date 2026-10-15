"""Evaluate OSIE trained model on MIT scanpaths."""

import argparse
from pathlib import Path
import math
import random
import torch
import numpy as np
from tqdm import tqdm

from gaze_model import GazeLLNArch, load_gaze_model
from gaze_losses import KLDTWLoss
from project_paths import OSIE_CHECKPOINT, MIT_DATA_ROOT, OUTPUT_FILE_DIR
from checkpoint_provenance import architecture_record, osie_loader_record, fingerprint
from run_exports import RunExport, write_csv
from OSIE_Modeldata import _center_gaussian, eval_epoch
from mit_data import MITDataset, MITMatDataset, build_mit_dataloader
from metrics import evaluate_metrics

METRIC_NAMES = ["Levenshtein", "ScanMatch", "Hausdorff", "Frechet", "FastDTW", "TDE"]


def eval_epoch_with_metrics(model, loader, criterion, device, *, return_details=False):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        per_scanpath = []
        batches = 0

        for imgs, hmap_seq, dt_seq, mask, fix_coords in tqdm(loader, desc="Validating", leave=False):
            imgs       = imgs.to(device)
            hmap_seq   = hmap_seq.to(device)
            mask       = mask.to(device)
            dt_seq     = dt_seq.to(device)
            fix_coords = fix_coords.to(device)

            B, T, hmap_H, hmap_W = hmap_seq.shape

            vis_features = model.extract_features(imgs)

            prev_hmap = _center_gaussian(B, hmap_H, hmap_W, device=device, sigma=1.5)
            hx = None

            predictions = []
            pred_scanpaths = [[] for _ in range(B)]

            for t in range(T):
                ts = dt_seq[:, t].view(-1, 1)

                out_hmap, hx = model(vis_features, prev_hmap, hx, ts)

                predictions.append(out_hmap.squeeze(1))
                prev_hmap = out_hmap

                # Extract predicted coordinate
                for b in range(B):
                    if mask[b, t]:
                        heatmap = out_hmap[b, 0].cpu().numpy()
                        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                        orig_x = float(x * (imgs.shape[-1] / hmap_W))
                        orig_y = float(y * (imgs.shape[-2] / hmap_H))

                        # Retain one point per valid fixation, including duplicates.
                        pred_scanpaths[b].append([orig_x, orig_y])

            predictions = torch.stack(predictions, dim=1)
            loss = criterion(predictions, hmap_seq, mask)
            total_loss += loss.item()
            batches += 1

            # Ground truth scanpath extraction using exact coordinates
            for b in range(B):
                gt_scanpath = []
                for t in range(T):
                    if mask[b, t]:
                        x_val = float(fix_coords[b, t, 0].cpu())
                        y_val = float(fix_coords[b, t, 1].cpu())
                        gt_scanpath.append([x_val, y_val])

                row = dict(evaluation_index=len(per_scanpath), valid_fixations=len(gt_scanpath),
                           **{name: None for name in METRIC_NAMES})
                if len(pred_scanpaths[b]) > 0 and len(gt_scanpath) > 0:
                    values = evaluate_metrics(np.array(pred_scanpaths[b]), np.array(gt_scanpath),
                                              height=imgs.shape[-2], width=imgs.shape[-1])
                    row.update({key: float(value) if np.isfinite(value) else None for key, value in values.items()})
                per_scanpath.append(row)

        if not batches:
            raise ValueError("MIT evaluation requires a nonempty loader")
        summary = []
        print("\nEvaluation Metrics:")
        for name in METRIC_NAMES:
            values = [row[name] for row in per_scanpath if row[name] is not None]
            mean = float(np.mean(values)) if values else None
            summary.append(dict(metric=name, mean=mean, std=float(np.std(values)) if values else None,
                                n_finite=len(values), n_missing=len(per_scanpath) - len(values),
                                pooling="mean over finite scanpaths; population standard deviation"))
            print(f"{name}: {mean:.4f}" if mean is not None else f"{name}: N/A")

        loss = total_loss / batches
        return dict(loss=loss, summary=summary, per_scanpath=per_scanpath) if return_details else loss


def main(argv=None, loss_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=MIT_DATA_ROOT)
    parser.add_argument("--output-dir", "--out-dir", type=Path, help="Default: shared output/mit_evaluation/metrics or loss_only")
    parser.add_argument("--checkpoint", type=Path, default=OSIE_CHECKPOINT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, help="Explicit scanpath cap for a smoke run")
    parser.add_argument("--dataset-format", choices=["manifest", "mat"], default="manifest",
                        help="manifest.csv + fixation arrays, or original notebook images/ + eye/fixations.mat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=1.5, help="Target heatmap blur in grid pixels")
    parser.add_argument("--gamma", type=float, default=0.1, help="KL-DTW smoothing temperature")
    parser.add_argument("--loss-only", action="store_true", default=loss_only)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or (args.limit_samples is not None and args.limit_samples <= 0):
        parser.error("Batch size/sample cap must be positive; workers nonnegative")
    if args.dataset_format == "mat" and not args.loss_only:
        parser.error("The original MAT layout supports KL-DTW evaluation; use --loss-only")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu")
    if not 0 <= args.seed < 2**32 or not 0 < args.min_len <= args.max_len:
        parser.error("Seed must be in [0, 2**32); fixation lengths must satisfy 0 < min <= max")
    if any(not math.isfinite(value) or value <= 0 for value in (args.sigma, args.gamma)):
        parser.error("Sigma and gamma must be finite and positive")
    args.data_root = args.data_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = (args.output_dir or OUTPUT_FILE_DIR / "mit_evaluation" / ("loss_only" if args.loss_only else "metrics")).expanduser().resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        run = RunExport(args.output_dir, "mit_evaluation", dict(arguments=vars(args), resolved_device=device),
                        sources=[Path(__file__), Path(__file__).with_name("mit_data.py"),
                                 Path(__file__).with_name("OSIE_Modeldata.py"), Path(__file__).with_name("metrics.py"),
                                 Path(__file__).with_name("gaze_losses.py"), Path(__file__).with_name("gaze_model.py"),
                                 Path(__file__).with_name("gaze_preprocessing.py"), Path(__file__).with_name("run_exports.py")])
        with run.run():
            model = load_gaze_model(args.checkpoint, device)
            loader = build_mit_dataloader(args.data_root, args.batch_size, args.num_workers,
                args.dataset_format, args.limit_samples, img_size=(model.hparams.image_h, model.hparams.image_w),
                downsample=model.hparams.S, min_len=args.min_len, max_len=args.max_len, sigma=args.sigma, seed=args.seed)
            if not len(loader):
                raise ValueError("MIT evaluation requires at least one scanpath")
            criterion = KLDTWLoss(gamma=args.gamma)
            data_record = osie_loader_record(loader)
            base_dataset = getattr(loader.dataset, "dataset", loader.dataset)
            checkpoint_info = fingerprint(args.checkpoint)
            run.update(checkpoint=checkpoint_info, architecture=architecture_record(model),
                       data=data_record, input_preprocessing=base_dataset.input_preprocessing,
                       loss=dict(name="gaze_losses.KLDTWLoss", gamma=criterion.gamma, eps=criterion.eps,
                                 mask_policy=criterion.mask_policy,
                                 pooling="arithmetic mean of batch losses; equal batch weighting"),
                       metric_settings=dict(coordinate_space="model input pixels", image_size=list(base_dataset.img_size),
                           prediction="argmax heatmap; consecutive duplicate points retained", bins=[12, 8],
                           scanmatch_threshold=3.5, scanmatch_gap_penalty=0., tde_k=3,
                           tde_direction="predicted windows to nearest ground-truth window; mean distance divided by k",
                           feedback="autoregressive; center Gaussian prior sigma 1.5 grid pixels",
                           time_policy=(base_dataset.provenance or {}).get("time_policy")))
            if args.loss_only:
                loss = eval_epoch(model, loader, criterion, device)
                details = dict(summary=[], per_scanpath=[])
            else:
                details = eval_epoch_with_metrics(model, loader, criterion, device, return_details=True)
                loss = details["loss"]
            if not math.isfinite(loss):
                raise ValueError("MIT evaluation produced a non-finite KL-DTW loss")
            loss_row = dict(metric="KL-DTW", mean=float(loss), std=None, n_finite=len(loader), n_missing=0,
                            pooling="arithmetic mean of batch losses; equal batch weighting")
            summary = [loss_row, *details["summary"]]
            for row in details["per_scanpath"]:
                identity = data_record["selected_samples"][row["evaluation_index"]]
                row.update(sample_index=identity["index"], image=identity["image"], subject_index=identity["subject_index"])
            write_csv(args.output_dir / "summary.csv", summary,
                      ["metric", "mean", "std", "n_finite", "n_missing", "pooling"])
            if not args.loss_only:
                write_csv(args.output_dir / "per_scanpath.csv", details["per_scanpath"],
                          ["evaluation_index", "sample_index", "image", "subject_index", "valid_fixations", *METRIC_NAMES])
            run.write_results(dict(checkpoint=checkpoint_info, dataset_format=args.dataset_format,
                                   mode="loss_only" if args.loss_only else "metrics", loss=float(loss),
                                   scanpaths=len(loader.dataset), batches=len(loader), summary=summary,
                                   per_scanpath=details["per_scanpath"]))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"MIT evaluation failed: {exc}\n")
    print(f"Checkpoint: {args.checkpoint}\nKL-DTW test loss: {loss:.6f}\nRun configuration: {run.path}\nResults: {args.output_dir / 'results.json'}")
    return loss


if __name__ == "__main__":
    main()
