"""Score simple gaze controls against aligned DIEM heatmaps (NPZ heatmaps/valid).

Run from the repository root: python -m Baselines.evaluate_gaze --help
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .maps import FrameControl, probability_map


def load_targets(path):
    with np.load(path, allow_pickle=False) as data:
        maps = np.asarray(data["heatmaps"], dtype=np.float32)
        valid = np.asarray(data["valid"])
    if maps.ndim != 3 or not all(maps.shape) or valid.shape != (len(maps),) or valid.dtype != np.bool_:
        raise ValueError("Targets need heatmaps (T,H,W) and boolean valid (T,)")
    if not np.isfinite(maps).all() or (maps < 0).any():
        raise ValueError("Targets must be finite and nonnegative")
    if not valid.any() or (maps[valid].sum(axis=(1, 2)) <= 0).any():
        raise ValueError("Every valid frame needs a nonempty target; at least one valid frame is required")
    return maps, valid


def map_metrics(prediction, target):
    p, q = probability_map(prediction).astype(np.float64), probability_map(target).astype(np.float64)
    # Epsilon is followed by re-normalization so KL uses proper distributions.
    pe, qe = p + 1e-12, q + 1e-12
    pe, qe = pe / pe.sum(), qe / qe.sum()
    pc, qc = p - p.mean(), q - q.mean()
    denom = np.sqrt(np.sum(pc ** 2) * np.sum(qc ** 2))
    y, x = np.mgrid[:p.shape[0], :p.shape[1]]
    error = np.hypot(np.sum(x * (p - q)), np.sum(y * (p - q)))
    return dict(kl_gt_pred=float(np.sum(qe * np.log(qe / pe))),
                similarity=float(np.minimum(p, q).sum()),
                cc=float(np.sum(pc * qc) / denom) if denom > 1e-15 else None,
                centroid_error_grid_px=float(error))


def temporal_controls(predictions, seed=42):
    """Prediction-only controls; shuffled maps are an offline diagnostic."""
    predictions = np.asarray(predictions)
    lag = np.concatenate([np.full_like(predictions[:1], 1 / np.prod(predictions.shape[1:])), predictions[:-1]])
    static = np.repeat(predictions[:1], len(predictions), axis=0)
    output = dict(gaze=predictions, gaze_lag=lag, gaze_static=static)
    if len(predictions) > 1:
        # A seeded nonzero circular shift has no frame mapped to itself.
        shift = int(np.random.default_rng(seed).integers(1, len(predictions)))
        output["gaze_shifted"] = np.roll(predictions, shift, axis=0)
    return output


def score_sequences(sequences, targets, valid):
    rows, summary = [], []
    for method, predictions in sequences.items():
        if predictions.shape != targets.shape:
            raise ValueError(f"{method}: prediction and target shapes differ: {predictions.shape} != {targets.shape}")
        if not np.isfinite(predictions).all() or (predictions < 0).any() or (predictions.sum(axis=(1, 2)) <= 0).any():
            raise ValueError(f"{method}: predictions must be finite nonnegative maps with positive mass")
        scores = []
        for i in np.flatnonzero(valid):
            score = map_metrics(predictions[i], targets[i])
            scores.append(score)
            rows.append(dict(method=method, frame=int(i), **score))
        pooled = dict(method=method, frames_evaluated=len(scores))
        for metric in scores[0]:
            values = [s[metric] for s in scores if s[metric] is not None]
            pooled[metric] = float(np.mean(values)) if values else None
            pooled[f"n_{metric}"] = len(values)
        summary.append(pooled)
    return rows, summary


def frame_sequences(video, shape, methods, center_sigma, spatial_sigma, temporal_alpha, checkpoint=None, frames=None):
    import cv2
    controls = {m: FrameControl(m, shape, center_sigma, spatial_sigma, temporal_alpha) for m in methods}
    cap = cv2.VideoCapture(str(video))
    output = {m: [] for m in methods}
    if checkpoint is not None:
        from Encoder.qp_map_generator import call_model, apply_spatial_smoothing, _MODEL_CACHE
        # A new video must not inherit recurrent state or a different checkpoint.
        for key in ("model", "device", "prev_hmap", "hx"):
            _MODEL_CACHE[key] = None
        output["gaze"] = []
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video needs a valid frame rate")
        previous_prediction = None
        count = 0
        while frames is None or count < frames:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            for method, control in controls.items():
                output[method].append(control(frame))
            if checkpoint is not None:
                hm = apply_spatial_smoothing(call_model(frame, checkpoint_path=checkpoint, dt=1 / fps), spatial_sigma)
                raw = hm.squeeze(0).numpy()
                if raw.shape != shape:
                    raise ValueError(f"Model grid {raw.shape} does not match targets {shape}")
                current = probability_map(raw)
                if previous_prediction is not None:
                    current = probability_map(temporal_alpha * current + (1 - temporal_alpha) * previous_prediction)
                previous_prediction = current
                output["gaze"].append(current)
    finally:
        cap.release()
    return {m: np.asarray(v) for m, v in output.items()}


def fingerprint(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True, help="Aligned DIEM NPZ with heatmaps and valid")
    parser.add_argument("--target-start-index", type=int, default=0,
                        help="First target corresponding to video frame zero; use 1 for DIEM build_heatmaps.py output")
    parser.add_argument("--frames", type=int, help="Explicit prefix length for a smoke run; default: entire aligned sequence")
    parser.add_argument("--video", type=Path, help="Source video, exactly one frame per target heatmap")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--predictions", type=Path, help="NPZ with heatmaps (T,H,W), in target order")
    inputs.add_argument("--checkpoint", type=Path, help="Trained DIEM GazeLNN checkpoint; also requires --video")
    parser.add_argument("--methods", nargs="+", choices=["uniform", "center", "motion"], default=["uniform", "center"])
    parser.add_argument("--center-sigma", type=float, default=0.2)
    parser.add_argument("--spatial-sigma", type=float, default=1.5)
    parser.add_argument("--temporal-alpha", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("Baselines/gaze_results"))
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods):
        parser.error("Methods must be unique")
    if args.target_start_index < 0 or (args.frames is not None and args.frames <= 0):
        parser.error("Target start index must be nonnegative and frames must be positive")
    if ("motion" in args.methods or args.checkpoint) and not args.video:
        parser.error("Motion and model inference require --video")
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    try:
        targets, valid = load_targets(args.targets)
        stop = args.target_start_index + args.frames if args.frames is not None else len(targets)
        if args.target_start_index >= len(targets) or stop > len(targets):
            raise ValueError("Requested target interval exceeds the annotation sequence")
        targets, valid = targets[args.target_start_index:stop], valid[args.target_start_index:stop]
        if not valid.any():
            raise ValueError("Requested target interval has no valid frames")
        start = time.perf_counter()
        if args.video:
            sequences = frame_sequences(args.video, targets.shape[1:], args.methods, args.center_sigma,
                                         args.spatial_sigma, args.temporal_alpha, args.checkpoint, args.frames)
        else:
            sequences = {m: np.repeat(FrameControl(m, targets.shape[1:], args.center_sigma,
                args.spatial_sigma, args.temporal_alpha)()[None], len(targets), axis=0) for m in args.methods}
        if args.predictions:
            with np.load(args.predictions, allow_pickle=False) as data:
                predictions = np.asarray(data["heatmaps"], dtype=np.float32)
            if args.frames is not None:
                predictions = predictions[:args.frames]
            if predictions.shape != targets.shape:
                raise ValueError("Prediction grid/frame count differs from targets; explicit alignment is required")
            sequences.update(temporal_controls(predictions, args.seed))
        elif args.checkpoint:
            sequences.update(temporal_controls(sequences["gaze"], args.seed))
        generation_time = time.perf_counter() - start
        rows, summary = score_sequences(sequences, targets, valid)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, values in (("per_frame", rows), ("summary", summary)):
            with (args.out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
        provenance = {name: fingerprint(path) for name, path in
                      (("targets", args.targets), ("predictions", args.predictions), ("checkpoint", args.checkpoint)) if path}
        if args.video:
            provenance["video"] = fingerprint(args.video)
        results = dict(summary=summary, provenance=provenance, generation_time_s=generation_time,
            protocol=dict(center_sigma=args.center_sigma, spatial_sigma=args.spatial_sigma,
                temporal_alpha=args.temporal_alpha, seed=args.seed, valid_frames=int(valid.sum()),
                total_frames=len(valid), grid=list(targets.shape[1:]), pooling="mean over valid frames of this clip",
                target_start_index=args.target_start_index, requested_frames=args.frames,
                kl="KL(ground truth || prediction), epsilon 1e-12 then renormalize",
                cc="undefined for constant maps; null with explicit count",
                temporal="lag: previous prediction, uniform initial map; static: first prediction; shifted: seeded nonzero circular shift, offline diagnostic",
                model_dt="1 / source video FPS when inferring a DIEM checkpoint",
                alignment="caller supplies exact temporal alignment; frame-count/grid mismatch is rejected"))
        (args.out_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
        lines = ["# Gaze heatmap controls", "", "Scores use measured gaze heatmaps and the same valid frames for every method.", "",
                 "| Method | Frames | KL(GT‖prediction) ↓ | Similarity ↑ | CC ↑ | Centroid error (grid pixels) ↓ |",
                 "|---|---:|---:|---:|---:|---:|"]
        for row in summary:
            values = ["N/A" if row[k] is None else f"{row[k]:.5f}" for k in
                      ("kl_gt_pred", "similarity", "cc", "centroid_error_grid_px")]
            lines.append(f"| {row['method']} | {row['frames_evaluated']} | " + " | ".join(values) + " |")
        lines += ["", "Uniform-map CC is undefined. Centroid distance is a distribution statistic, not fixation or scanpath error.",
                  "The shifted control may use future predictions; it is only an offline timing diagnostic.",
                  "This evaluates one clip, without a held-out split claim. Use video-level held-out clips for research comparisons.",
                  "Alignment, hashes and metric definitions are recorded in results.json."]
        (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if args.checkpoint:
            np.savez_compressed(args.out_dir / "predictions.npz", heatmaps=sequences["gaze"])
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Gaze evaluation failed: {exc}\n")
    print(f"Report: {args.out_dir.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
