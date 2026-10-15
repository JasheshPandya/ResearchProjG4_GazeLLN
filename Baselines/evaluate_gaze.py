"""Score gaze and static-saliency controls against aligned DIEM heatmaps.

Run from the repository root: python -m Baselines.evaluate_gaze --help
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from .maps import FrameControl, probability_map
from project_paths import (STATIC_SALIENCY_CHECKPOINT, INPUT_VIDEO_DIR, OUTPUT_FILE_DIR,
                           discover_videos, video_id)
from dataset_evaluation import (add_statistics_arguments, validate_statistics, aggregate_records,
                                paired_comparisons, statistics_protocol, write_dataset_report)
from video_timing import DT_POLICY, frame_interval
from DIEM import config as diem_config
from DIEM.splits import membership, verify_checkpoint, read_manifest, file_sha256


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


def frame_sequences(video, shape, methods, center_sigma, spatial_sigma, temporal_alpha, checkpoint=None, frames=None,
                    static_checkpoint=STATIC_SALIENCY_CHECKPOINT):
    import cv2
    controls = {m: FrameControl(m, shape, center_sigma, spatial_sigma, temporal_alpha) for m in methods if m != "static_saliency"}
    if "static_saliency" in methods:
        if temporal_alpha != 1:
            raise ValueError("Static saliency requires --temporal-alpha 1")
        from .static_saliency import StaticSaliency
        controls["static_saliency"] = StaticSaliency(static_checkpoint, shape, spatial_sigma)
    cap = cv2.VideoCapture(str(video))
    output = {m: [] for m in methods}
    if checkpoint is not None:
        from Encoder.qp_map_generator import call_model, apply_spatial_smoothing, reset_model_state
        reset_model_state()
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
                hm = apply_spatial_smoothing(call_model(frame, checkpoint_path=checkpoint, dt=frame_interval(fps)), spatial_sigma)
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
    return dict(path=str(path.resolve()), sha256=file_sha256(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, help="Aligned DIEM NPZ with heatmaps and valid")
    parser.add_argument("--target-start-index", type=int, default=0,
                        help="First target corresponding to video frame zero; use 1 for DIEM build_heatmaps.py output")
    parser.add_argument("--frames", type=int, help="Explicit prefix length for a smoke run; default: entire aligned sequence")
    parser.add_argument("--video", type=Path, help="Source video, exactly one frame per target heatmap")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--predictions", type=Path, help="NPZ with heatmaps (T,H,W), in target order")
    inputs.add_argument("--checkpoint", type=Path, help="Trained DIEM GazeLNN checkpoint; also requires --video")
    parser.add_argument("--methods", nargs="+", choices=["uniform", "center", "motion", "static_saliency"], default=["uniform", "center"])
    parser.add_argument("--static-checkpoint", type=Path, default=STATIC_SALIENCY_CHECKPOINT)
    parser.add_argument("--split-manifest", type=Path, default=diem_config.SPLIT_MANIFEST)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--video-name", help="Manifest video identifier; defaults to target NPZ stem")
    parser.add_argument("--no-split-check", action="store_true", help="Explicit diagnostic mode; makes no held-out claim")
    parser.add_argument("--allow-unverified-checkpoint", action="store_true",
                        help="Allow historical model/predictions with unknown training splits; label results unverified")
    parser.add_argument("--target-fps", type=float, help="Annotation rate override; otherwise NPZ fps or DIEM config rate")
    parser.add_argument("--center-sigma", type=float, default=0.2)
    parser.add_argument("--spatial-sigma", type=float, default=1.5)
    parser.add_argument("--temporal-alpha", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", "--output-dir", type=Path, default=OUTPUT_FILE_DIR / "gaze_evaluation")
    parser.add_argument("--input-dir", type=Path, default=INPUT_VIDEO_DIR)
    parser.add_argument("--targets-dir", type=Path, default=diem_config.OUTPUT_ROOT / "heatmaps",
                        help="Dataset targets named <video stem>.npz")
    parser.add_argument("--dataset-manifest", type=Path, help="JSON clips with name, video, targets, optional predictions/video_name/target_start_index/frames; paths relative to JSON")
    add_statistics_arguments(parser)
    args = parser.parse_args(argv)
    try:
        validate_statistics(args.bootstrap_samples, args.confidence)
        if args.dataset_manifest and args.targets:
            raise ValueError("Choose either --targets or --dataset-manifest")
        if args.targets is None:
            return evaluate_dataset(args)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Gaze dataset evaluation failed: {exc}\n")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Methods must be unique")
    if args.target_start_index < 0 or (args.frames is not None and args.frames <= 0):
        parser.error("Target start index must be nonnegative and frames must be positive")
    if (any(m in args.methods for m in ("motion", "static_saliency")) or args.checkpoint) and not args.video:
        parser.error("Motion and trained model inference require --video")
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    try:
        video_name = args.video_name or args.targets.stem
        split_info = dict(membership_verified=False, partition=None, video_name=video_name)
        if not args.no_split_check:
            split_info = membership(args.split_manifest, video_name, args.split)
            if args.video and args.video.stem != video_name:
                raise ValueError("Source video basename differs from its declared split identifier")
        training_verified = False
        checkpoint_note = None
        if args.checkpoint and not args.no_split_check and args.split == "test":
            try:
                split_info.update(verify_checkpoint(args.checkpoint, args.split_manifest, video_name))
                training_verified = True
            except ValueError as exc:
                if not args.allow_unverified_checkpoint:
                    raise
                checkpoint_note = str(exc)
        targets, valid = load_targets(args.targets)
        with np.load(args.targets, allow_pickle=False) as target_file:
            target_fps = args.target_fps if args.target_fps is not None else float(target_file["fps"]) if "fps" in target_file else diem_config.GAZE_FPS
        frame_interval(target_fps)
        video_fps = None
        if args.video:
            from Encoder.workflow import video_info
            video_fps = video_info(args.video)[3]
            if not np.isclose(video_fps, target_fps, rtol=1e-4):
                raise ValueError("Source FPS and annotation FPS differ; align them before gaze evaluation")
        stop = args.target_start_index + args.frames if args.frames is not None else len(targets)
        if args.target_start_index >= len(targets) or stop > len(targets):
            raise ValueError("Requested target interval exceeds the annotation sequence")
        targets, valid = targets[args.target_start_index:stop], valid[args.target_start_index:stop]
        if not valid.any():
            raise ValueError("Requested target interval has no valid frames")
        start = time.perf_counter()
        if args.video:
            sequences = frame_sequences(args.video, targets.shape[1:], args.methods, args.center_sigma,
                                         args.spatial_sigma, args.temporal_alpha, args.checkpoint, args.frames, args.static_checkpoint)
        else:
            sequences = {m: np.repeat(FrameControl(m, targets.shape[1:], args.center_sigma,
                args.spatial_sigma, args.temporal_alpha)()[None], len(targets), axis=0) for m in args.methods}
        if args.predictions:
            with np.load(args.predictions, allow_pickle=False) as data:
                predictions = np.asarray(data["heatmaps"], dtype=np.float32)
                prediction_info = json.loads(str(data["provenance_json"])) if "provenance_json" in data else None
            try:
                if not prediction_info or prediction_info.get("dt_policy") != DT_POLICY or \
                        prediction_info.get("video_name") != video_name or \
                        not np.isclose(prediction_info.get("fps", -1), target_fps, rtol=1e-4):
                    raise ValueError("Saved predictions lack matching video/time provenance; regenerate them")
                if not args.no_split_check and args.split == "test":
                    record = prediction_info.get("training_record") or {}
                    expected = read_manifest(args.split_manifest)
                    if record.get("schema_version") != 1 or not record.get("checkpoint_sha256") or \
                            record.get("split_manifest_sha256") != file_sha256(args.split_manifest) or \
                            record.get("checkpoint_sha256") != prediction_info.get("checkpoint_sha256") or \
                            record.get("dt_policy") != DT_POLICY or record.get("splits") != expected["splits"]:
                        raise ValueError("Saved predictions have no verifiable held-out checkpoint provenance")
                    training_verified = True
            except ValueError as exc:
                if not args.allow_unverified_checkpoint:
                    raise
                checkpoint_note = str(exc)
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
        if "static_saliency" in args.methods:
            provenance["static_checkpoint"] = fingerprint(args.static_checkpoint)
        learned_gaze = bool(args.checkpoint or args.predictions)
        held_out_verified = bool(split_info["membership_verified"] and args.split == "test" and
                                 (not learned_gaze or training_verified))
        results = dict(summary=summary, provenance=provenance, generation_time_s=generation_time,
            protocol=dict(center_sigma=args.center_sigma, spatial_sigma=args.spatial_sigma,
                temporal_alpha=args.temporal_alpha, seed=args.seed, valid_frames=int(valid.sum()),
                total_frames=len(valid), grid=list(targets.shape[1:]), pooling="mean over valid frames of this clip",
                target_start_index=args.target_start_index, requested_frames=args.frames,
                kl="KL(ground truth || prediction), epsilon 1e-12 then renormalize",
                cc="undefined for constant maps; null with explicit count",
                temporal="lag: previous prediction, uniform initial map; static: first prediction; shifted: seeded nonzero circular shift, offline diagnostic",
                dt_policy=DT_POLICY, fps=video_fps or target_fps, dt_seconds=frame_interval(video_fps or target_fps),
                split=split_info, held_out_verified=held_out_verified, checkpoint_provenance_note=checkpoint_note,
                static_saliency="DeepGazeI, trained MIT1003 weights, uniform center prior, independent frames, no EMA" if "static_saliency" in args.methods else None,
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
                  f"Held-out provenance verified: {held_out_verified}. Partition: {split_info['partition']}. Per-video scores; dataset runs pool these in dataset_report.md.",
                  f"Checkpoint/prediction provenance note: {checkpoint_note or 'none'}",
                  "Historical weights without a matching training record are diagnostics, not proven held-out results.",
                  "Alignment, hashes and metric definitions are recorded in results.json."]
        (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if args.checkpoint:
            training_path = Path(str(args.checkpoint) + ".training.json")
            prediction_info = dict(dt_policy=DT_POLICY, fps=video_fps, video_name=video_name,
                checkpoint_sha256=provenance["checkpoint"]["sha256"],
                training_record=json.loads(training_path.read_text()) if training_path.is_file() else None)
            np.savez_compressed(args.out_dir / "predictions.npz", heatmaps=sequences["gaze"],
                                provenance_json=json.dumps(prediction_info))
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Gaze evaluation failed: {exc}\n")
    print(f"Report: {args.out_dir.resolve() / 'report.md'}")
    return results


GAZE_METRICS = ["kl_gt_pred", "similarity", "cc", "centroid_error_grid_px"]


def evaluate_dataset(args):
    if args.predictions or args.video_name:
        raise ValueError("--predictions and --video-name require --targets; use per-clip entries in a dataset manifest")
    if args.dataset_manifest:
        if args.video:
            raise ValueError("--video and --dataset-manifest cannot be combined")
        base = args.dataset_manifest.resolve().parent
        with args.dataset_manifest.open(encoding="utf-8") as stream:
            clips = json.load(stream)["clips"]
        clips = [dict(c) for c in clips]
        for clip in clips:
            for key in ("video", "targets", "predictions"):
                if clip.get(key):
                    path = Path(str(clip[key]).replace("\\", "/"))
                    clip[key] = path.resolve() if path.is_absolute() else (base / path).resolve()
    else:
        videos = discover_videos(args.input_dir, args.video)
        clips = [dict(name=v.name, video=v, targets=args.targets_dir.resolve() / f"{v.stem}.npz") for v in videos]
        if len({c["targets"] for c in clips}) != len(clips):
            raise ValueError("Videos with equal stems need an explicit dataset manifest with distinct target files")
    names = [c["name"] for c in clips]
    if any(not isinstance(n, str) or not n or n in (".", "..") or "/" in n or "\\" in n for n in names):
        raise ValueError("Clip names must be nonempty identifiers without directory separators")
    if not names or len(names) != len(set(names)):
        raise ValueError("Dataset requires nonempty clips with unique names")
    sources = [c.get("video") or c.get("targets") for c in clips]
    if len(sources) != len(set(sources)):
        raise ValueError("A source sequence may appear only once in the dataset manifest")
    if args.checkpoint and any(c.get("predictions") for c in clips):
        raise ValueError("Dataset predictions and --checkpoint cannot be combined")
    # All annotations are mandatory: do not silently change the evaluation cohort.
    for clip in clips:
        for key in ("targets", "video", "predictions"):
            if clip.get(key) and not Path(clip[key]).is_file():
                raise ValueError(f"{clip['name']}: missing {key}: {clip[key]}")
        if not clip.get("targets"):
            raise ValueError(f"{clip['name']}: missing targets")
    records, provenance = [], []
    for clip in clips:
        folder = args.out_dir / "per_video" / video_id(clip["name"])
        command = ["--targets", str(clip["targets"]), "--out-dir", str(folder),
                   "--target-start-index", str(clip.get("target_start_index", args.target_start_index)),
                   "--center-sigma", str(args.center_sigma), "--spatial-sigma", str(args.spatial_sigma),
                   "--temporal-alpha", str(args.temporal_alpha), "--seed", str(args.seed),
                   "--split-manifest", str(args.split_manifest), "--split", args.split,
                   "--static-checkpoint", str(args.static_checkpoint), "--methods", *args.methods]
        for key in ("video", "predictions"):
            if clip.get(key):
                command += ["--" + key, str(clip[key])]
        if args.checkpoint:
            command += ["--checkpoint", str(args.checkpoint)]
        for key in ("no_split_check", "allow_unverified_checkpoint"):
            if getattr(args, key):
                command += ["--" + key.replace("_", "-")]
        if clip.get("video_name"):
            command += ["--video-name", str(clip["video_name"])]
        frames = clip.get("frames", args.frames)
        if frames is not None:
            command += ["--frames", str(frames)]
        if args.target_fps is not None:
            command += ["--target-fps", str(args.target_fps)]
        result = main(command)
        records += [dict(video=clip["name"], **row) for row in result["summary"]]
        provenance.append(dict(video=clip["name"], report=str((folder / "results.json").resolve()),
                               provenance=result["provenance"], protocol=result["protocol"]))
    settings = dict(samples=args.bootstrap_samples, confidence=args.confidence, seed=args.statistics_seed,
                    expected_videos=names)
    payload = dict(videos=names, per_video=records, provenance=provenance,
        aggregate=aggregate_records(records, ["method"], GAZE_METRICS, **settings),
        paired=paired_comparisons(records, GAZE_METRICS, **settings),
        held_out_verified=all(p["protocol"]["held_out_verified"] for p in provenance),
        protocol=statistics_protocol(args.bootstrap_samples, args.confidence, args.statistics_seed))
    payload["protocol"]["within_video"] = "Mean over valid aligned frames, separately for every metric; undefined CC omitted with counts."
    write_dataset_report(args.out_dir, payload)
    print(f"Dataset report: {args.out_dir.resolve() / 'dataset_report.md'}")
    return payload


if __name__ == "__main__":
    main()
