"""Persist video membership and training provenance without silently re-splitting."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from video_timing import DT_POLICY
if __package__:
    from . import config as cfg
else:
    import config as cfg

PARTITIONS = ("train", "val", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or \
            not isinstance(manifest.get("splits"), dict) or set(manifest["splits"]) != set(PARTITIONS):
        raise ValueError("Invalid split manifest schema")
    if manifest.get("dt_policy") != DT_POLICY:
        raise ValueError("Split manifest uses an incompatible video time convention")
    seen = set()
    for partition in PARTITIONS:
        names = manifest["splits"][partition]
        if not isinstance(names, list) or not names:
            raise ValueError(f"Split {partition} must contain video names")
        for name in names:
            if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\\" in name or name in seen:
                raise ValueError("Split names must be unique nonempty video identifiers, with no overlap")
            seen.add(name)
    return manifest


def write_manifest(path, manifest):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    read_manifest(pending)
    pending.replace(path)


def create_manifest(names, seed=0, val_frac=0.15, test_frac=0.15):
    names = sorted(names)
    if len(names) != len(set(names)) or len(names) < 3:
        raise ValueError("At least three distinct videos are required for nonempty train/val/test splits")
    if not np.isfinite([val_frac, test_frac]).all() or min(val_frac, test_frac) <= 0 or val_frac + test_frac >= 1:
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    order = np.random.RandomState(seed).permutation(len(names))
    n_val, n_test = max(1, int(len(names) * val_frac)), max(1, int(len(names) * test_frac))
    if n_val + n_test >= len(names):
        raise ValueError("Split fractions leave no training videos")
    val = {names[i] for i in order[:n_val]}
    test = {names[i] for i in order[n_val:n_val + n_test]}
    return dict(schema_version=1, dataset="DIEM", seed=seed, val_fraction=val_frac, test_fraction=test_frac,
        dt_policy=DT_POLICY, assignment_provenance="new assignment; not a reconstruction of historical checkpoint training",
        splits=dict(train=[n for n in names if n not in val | test], val=sorted(val), test=sorted(test)))


def persisted_split(names, path, seed=0, val_frac=0.15, test_frac=0.15):
    path = Path(path).expanduser().resolve()
    names = list(names)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate video names in discovered data")
    if not path.exists():
        write_manifest(path, create_manifest(names, seed, val_frac, test_frac))
    manifest = read_manifest(path)
    frozen = set().union(*(set(manifest["splits"][part]) for part in PARTITIONS))
    added, missing = set(names) - frozen, frozen - set(names)
    if added or missing:
        raise ValueError(f"Discovered videos differ from frozen splits: new={sorted(added)}, missing={sorted(missing)}. "
                         "Prepare missing data, explicitly extend the manifest, or create a separately named manifest.")
    if (manifest.get("seed"), manifest.get("val_fraction"), manifest.get("test_fraction")) != (seed, val_frac, test_frac):
        raise ValueError("Requested split parameters differ from the persisted manifest; use its recorded parameters")
    return manifest


def membership(path, video_name, partition="test"):
    manifest = read_manifest(path)
    if partition not in PARTITIONS or video_name not in manifest["splits"][partition]:
        actual = next((p for p in PARTITIONS if video_name in manifest["splits"][p]), "unassigned")
        raise ValueError(f"Video {video_name!r} belongs to {actual}, not requested {partition}")
    return dict(manifest=str(Path(path).resolve()), manifest_sha256=file_sha256(path),
                video_name=video_name, partition=partition, membership_verified=True)


def training_split_binding(manifest_path, train_names, val_names):
    manifest = read_manifest(manifest_path)
    if len(train_names) != len(set(train_names)) or len(val_names) != len(set(val_names)) or \
            set(train_names) != set(manifest["splits"]["train"]) or set(val_names) != set(manifest["splits"]["val"]):
        raise ValueError("Training/validation loaders do not match the recorded video splits")
    return dict(split_manifest_sha256=file_sha256(manifest_path), splits=manifest["splits"], dt_policy=DT_POLICY)


def record_training(checkpoint, manifest_path, train_names, val_names, **settings):
    """Legacy split-only record helper; trainers use TrainingRecorder for full provenance."""
    record = dict(schema_version=1, checkpoint_sha256=file_sha256(checkpoint),
                  **training_split_binding(manifest_path, train_names, val_names), settings=settings)
    path = Path(str(checkpoint) + ".training.json")
    pending = Path(str(path) + ".tmp")
    pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)
    return record


def verify_checkpoint(checkpoint, manifest_path, video_name):
    path = Path(str(checkpoint) + ".training.json")
    if not path.is_file():
        raise ValueError("Checkpoint has no recorded training split; historical held-out membership cannot be proven. "
                         "Retrain with persisted splits, or use --allow-unverified-checkpoint for a labelled diagnostic.")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("checkpoint_sha256") != file_sha256(checkpoint) or \
            record.get("split_manifest_sha256") != file_sha256(manifest_path) or record.get("dt_policy") != DT_POLICY:
        raise ValueError("Checkpoint provenance does not match these weights, split manifest or elapsed-time policy")
    if record.get("splits") != read_manifest(manifest_path)["splits"]:
        raise ValueError("Checkpoint training record and manifest memberships differ")
    if video_name in record["splits"]["train"] or video_name in record["splits"]["val"]:
        raise ValueError("Requested held-out video was used for training or validation")
    return dict(training_split_verified=True, training_record=str(path.resolve()),
                training_record_sha256=file_sha256(path), checkpoint_sha256=record["checkpoint_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=cfg.DIEM_ROOT)
    parser.add_argument("--output", type=Path, default=cfg.SPLIT_MANIFEST)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--videos", nargs="+", help="Explicit cohort subset of raw video identifiers")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--extend", action="store_true", help="Append new videos to train without reassigning any old video")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace all assignments; invalidates old checkpoint provenance")
    args = parser.parse_args(argv)
    if args.extend and args.overwrite:
        parser.error("Choose extend or overwrite")
    try:
        names = sorted(d.name for d in args.raw_root.iterdir() if d.is_dir() and
                       bool(cfg.video_files(d)) and any(d.glob(cfg.GAZE_FILE_GLOB)))
        if args.videos is not None:
            if set(args.videos) - set(names):
                raise ValueError("Requested cohort contains unavailable raw videos")
            names = sorted(args.videos)
        if args.extend and args.output.is_file():
            manifest = read_manifest(args.output)
            frozen = set().union(*(set(v) for v in manifest["splits"].values()))
            if frozen - set(names):
                raise ValueError("Existing assigned videos are missing from raw inputs")
            manifest["splits"]["train"] += sorted(set(names) - frozen)
            write_manifest(args.output, manifest)
        elif args.overwrite:
            manifest = create_manifest(names, args.seed, args.val_fraction, args.test_fraction)
            write_manifest(args.output, manifest)
        else:
            manifest = persisted_split(names, args.output, args.seed, args.val_fraction, args.test_fraction)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Split manifest failed: {exc}\n")
    print(json.dumps(dict(path=str(args.output.resolve()), sha256=file_sha256(args.output),
                          counts={p: len(v) for p, v in manifest["splits"].items()}), indent=2))


if __name__ == "__main__":
    main()
