"""Evaluate OSIE trained model on MIT scanpaths."""

import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

from gaze_model import GazeLLNArch, load_gaze_model
from gaze_losses import KLDTWLoss
from project_paths import OSIE_CHECKPOINT, MIT_DATA_ROOT
from OSIE_Modeldata import _center_gaussian, eval_epoch
from mit_data import MITDataset, MITMatDataset, build_mit_dataloader
from metrics import evaluate_metrics

def eval_epoch_with_metrics(model, loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        all_metrics = {
            "Levenshtein": [], "ScanMatch": [], "Hausdorff": [],
            "Frechet": [], "FastDTW": [], "TDE": []
        }

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

                        # Avoid redundant consecutive predicted points
                        # Stop Removing deduplication
                        # if len(pred_scanpaths[b]) == 0 or pred_scanpaths[b][-1] != [orig_x, orig_y]:
                        #     pred_scanpaths[b].append([orig_x, orig_y])
                        pred_scanpaths[b].append([orig_x, orig_y])

            predictions = torch.stack(predictions, dim=1)
            loss = criterion(predictions, hmap_seq, mask)
            total_loss += loss.item()

            # Ground truth scanpath extraction using exact coordinates
            for b in range(B):
                gt_scanpath = []
                for t in range(T):
                    if mask[b, t]:
                        x_val = float(fix_coords[b, t, 0].cpu())
                        y_val = float(fix_coords[b, t, 1].cpu())
                        gt_scanpath.append([x_val, y_val])

                if len(pred_scanpaths[b]) > 0 and len(gt_scanpath) > 0:
                    metrics_res = evaluate_metrics(np.array(pred_scanpaths[b]), np.array(gt_scanpath))
                    for k, v in metrics_res.items():
                        all_metrics[k].append(v)

        print("\nEvaluation Metrics:")
        for k, v in all_metrics.items():
            if k == "TDE":
                # Paths shorter than the embedding dimension have no TDE score.
                v = [value for value in v if np.isfinite(value)]
                avg_val = sum(v) / len(v) if v else float('nan')
            else:
                avg_val = sum(v) / len(v) if len(v) > 0 else 0
            print(f"{k}: {avg_val:.4f}")

        return total_loss / len(loader)


def main(argv=None, loss_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=MIT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=OSIE_CHECKPOINT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, help="Explicit scanpath cap for a smoke run")
    parser.add_argument("--dataset-format", choices=["manifest", "mat"], default="manifest",
                        help="manifest.csv + fixation arrays, or original notebook images/ + eye/fixations.mat")
    parser.add_argument("--loss-only", action="store_true", default=loss_only)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or (args.limit_samples is not None and args.limit_samples <= 0):
        parser.error("Batch size/sample cap must be positive; workers nonnegative")
    if args.dataset_format == "mat" and not args.loss_only:
        parser.error("The original MAT layout supports KL-DTW evaluation; use --loss-only")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu")
    try:
        model = load_gaze_model(args.checkpoint, device)
        loader = build_mit_dataloader(args.data_root.resolve(), args.batch_size, args.num_workers,
                                     args.dataset_format, args.limit_samples)
        criterion = KLDTWLoss()
        loss = eval_epoch(model, loader, criterion, device) if args.loss_only else \
            eval_epoch_with_metrics(model, loader, criterion, device)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"MIT evaluation failed: {exc}\n")
    print(f"Checkpoint: {args.checkpoint.resolve()}\nKL-DTW test loss: {loss:.6f}")
    return loss


if __name__ == "__main__":
    main()
