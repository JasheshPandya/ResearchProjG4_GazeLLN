"""
Step 3: PyTorch Dataset serving contiguous chunks of (frame, heatmap, valid,
delta_t) tuples for continuous (Option B) fine-tuning of GazeLNN on DIEM.

Design notes (see chat discussion this implements):
- Splitting is done at the VIDEO level, never at the frame level, to avoid
  leaking correlated content/gaze between train/val/test.
- Each item covers the next non-overlapping chunk of one video, including
  a terminal partial chunk padded for batching. sequence_length distinguishes
  real timesteps from padding; valid masks gaze labels independently.
- Hidden state and previous prediction carry only to the exact next frame,
  with gradients detached at chunk boundaries. Overlap, gaps, reordered chunks
  and replay are rejected by the training loop's per-video frame cursor.
- delta_t per step is 1/GAZE_FPS (real elapsed time between frames), NOT the
  fixation-duration delta_t the original GazeLNN paper uses for its
  within-image scanpath steps. This is a deliberate change in what delta_t
  means, discussed in the fine-tuning strategy.
- Frames flagged invalid (< MIN_SUBJECTS_PER_FRAME valid subjects) should be
  masked out of the loss -- the `valid` mask is returned per-frame for this.
- `heatmaps` here are (T, cfg.HMAP_HEIGHT, cfg.HMAP_WIDTH) = (T, 32, 48) by
  default, matching GazeLLNArch's own downsampled heatmap grid (image_size
  // S) exactly -- built in build_heatmaps.py using the same coordinate
  rescaling + sigma convention as the existing OSIE preprocessing, so this
  plugs directly into forward() without any resolution mismatch.
- Default feedback is autoregressive. Optional teacher forcing uses valid
  ground-truth heatmaps; unannotated real frames use predicted feedback.
  Both modes advance state for real elapsed frames and exclude padding.

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from video_timing import frame_interval
from gaze_preprocessing import GazePreprocessor, normalize_rgb_tensor
from project_paths import file_sha256
from DIEM.frame_integrity import validate_frame_cache, validate_frame_file, read_target_alignment

if __package__:
    from . import config as cfg
    from .splits import persisted_split
    from .splits import read_manifest
else:
    import config as cfg
    from splits import persisted_split
    from splits import read_manifest

try:
    import torch
    from torch.utils.data import Dataset, Sampler
except ImportError:  # allow inspecting/testing this module without torch installed
    torch = None
    Dataset = object
    Sampler = object


@dataclass
class VideoEntry:
    name: str
    frames_dir: Path          # directory of extracted+resized video frames (see note below)
    heatmap_npz: Path
    num_frames: int           # real decoded frames; excludes unused annotation index zero


def _discover_videos(processed_root: Path, frames_root: Path) -> List[VideoEntry]:
    """
    Assumes you've already extracted and resized video frames to
    frames_root/<video_name>/frame_{i:06d}.png (via extract_frames.py), at
    cfg.MODEL_INPUT_WIDTH x cfg.MODEL_INPUT_HEIGHT, matching the heatmap
    resolution built in build_heatmaps.py. The extraction entry point uses
    the same OpenCV decoder and Pillow resize as model inference and stores
    resize provenance alongside the unnormalized RGB PNGs. Frame indices
    start at one to match DIEM annotation frame numbers.
    """
    heatmaps_dir = processed_root / "heatmaps"
    entries = []
    for npz_path in sorted(heatmaps_dir.glob("*.npz")):
        video_name = npz_path.stem
        frames_dir = frames_root / video_name
        if not frames_dir.exists():
            raise ValueError(f"Missing extracted frames for {video_name}: {frames_dir}")
        with np.load(npz_path) as d:
            alignment = read_target_alignment(d)
            num_frames = alignment["source_frame_count"]
            annotation_fps = float(d["fps"]) if "fps" in d else cfg.GAZE_FPS
        if not np.isclose(annotation_fps, cfg.GAZE_FPS, rtol=1e-4):
            raise ValueError(f"{video_name}: target FPS differs from configured DIEM annotation rate")
        source_videos = cfg.video_files(cfg.DIEM_ROOT / video_name)
        if not source_videos or file_sha256(source_videos[0]) != alignment["source_sha256"]:
            raise ValueError(f"{video_name}: DIEM source video is missing or differs from target source")
        if source_videos:
            import cv2
            cap = cv2.VideoCapture(str(source_videos[0]))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_interval(fps)
                if not np.isclose(fps, annotation_fps, rtol=1e-4):
                    raise ValueError(f"{video_name}: source FPS and gaze time base differ; align before training")
            finally:
                cap.release()
        entries.append(VideoEntry(video_name, frames_dir, npz_path, num_frames))
    return entries


def split_videos(entries: List[VideoEntry], val_frac=0.15, test_frac=0.15, seed=0, manifest_path=None):
    """Create once, then strictly reuse exact persisted video memberships."""
    manifest = persisted_split([e.name for e in entries], manifest_path or cfg.SPLIT_MANIFEST,
                               seed=seed, val_frac=val_frac, test_frac=test_frac)
    by_name = {e.name: e for e in entries}
    return tuple([by_name[name] for name in manifest["splits"][part]] for part in ("train", "val", "test"))


class DIEMWindowDataset(Dataset):
    """One item = one contiguous chunk from one video: frames, heatmaps, valid
    mask, and delta_t (constant = 1/fps here, kept as a tensor for API
    parity with the fixation-duration delta_t used in the original paper).

    Default frame tensors use the shared ImageNet-normalized RGB contract.
    preprocessing='legacy_rgb01' explicitly restores unnormalized [0,1] data;
    transform, when supplied, replaces normalization on resized [0,1] tensors.
    """

    def __init__(self, entries: List[VideoEntry],
                 window_len: int = cfg.WINDOW_LEN_FRAMES,
                 stride: int = None,
                 transform=None, preprocessing="imagenet", image_size=None):
        if torch is None:
            raise ImportError("PyTorch is required to use DIEMWindowDataset")
        stride = window_len if stride is None else stride
        if not isinstance(window_len, int) or window_len <= 0 or stride != window_len:
            raise ValueError("Stateful DIEM training requires positive window_len and stride == window_len (no overlap or gaps)")
        if len({entry.name for entry in entries}) != len(entries) or any(entry.num_frames <= 0 for entry in entries):
            raise ValueError("DIEM entries require unique video names and positive frame counts")
        self.entries = entries
        self.window_len = window_len
        self.stride = stride
        self.temporal_policy = cfg.RECURRENT_STATE_POLICY
        # A custom transform preserves the old full-tensor override API: it
        # receives resized [0,1] frames and replaces normalization. Its contract
        # is explicitly custom/unknown rather than falsely labeled ImageNet.
        self.transform = transform
        self.preprocessor = GazePreprocessor(image_size or (cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH), preprocessing)
        self.input_preprocessing = self.preprocessor.spec if transform is None else dict(
            version=1, profile="custom_tensor_transform", transform=repr(transform))
        self.frames_preparation = {}
        self.frame_integrity = {}
        for entry in entries:
            record = validate_frame_cache(entry.frames_dir, self.preprocessor.image_size)
            with np.load(entry.heatmap_npz, allow_pickle=False) as data:
                alignment = read_target_alignment(data)
                heatmaps, valid = data["heatmaps"], data["valid"]
                if heatmaps.ndim != 3 or not all(heatmaps.shape) or valid.dtype != np.bool_ or valid.shape != (len(heatmaps),) or \
                        not np.isfinite(heatmaps).all() or (heatmaps < 0).any() or \
                        (heatmaps[valid].sum(axis=(1, 2)) <= 0).any():
                    raise ValueError(f"Invalid heatmap/valid arrays: {entry.heatmap_npz}")
                if record["num_frames"] != entry.num_frames or alignment["source_frame_count"] != entry.num_frames or \
                        record["source_sha256"] != alignment["source_sha256"] or \
                        not np.isclose(record["fps"], alignment["fps"], rtol=1e-4) or \
                        "fps" not in data or not np.isclose(float(data["fps"]), alignment["fps"], rtol=1e-4) or \
                        not np.isclose(alignment["fps"], cfg.GAZE_FPS, rtol=1e-4):
                    raise ValueError(f"Frame/annotation source, count or FPS mismatch: {entry.name}")
            self.frames_preparation[entry.name] = record
            self.frame_integrity[entry.name] = dict(
                alignment=alignment, all_png_hashes_and_pixels_verified=True,
                first_png_number=1, first_target_index=1, recurrent_cursor_origin=0,
                targets_sha256=file_sha256(entry.heatmap_npz))
        self.index: List[Tuple[int, int]] = []  # (entry_idx, start_frame)
        for ei, e in enumerate(entries):
            # Include a final partial chunk. Each frame index occurs once.
            for s in range(0, e.num_frames, window_len):
                self.index.append((ei, s))

    def __len__(self):
        return len(self.index)

    def _load_frame(self, frames_dir: Path, frame_idx: int) -> np.ndarray:
        # Check again at read time: files may have changed after preflight.
        path = frames_dir / f"frame_{frame_idx:06d}.png"
        record = next(self.frames_preparation[e.name] for e in self.entries if e.frames_dir == frames_dir)
        return validate_frame_file(path, self.preprocessor.image_size, record["frame_sha256"][path.name])

    def __getitem__(self, idx: int):
        ei, start = self.index[idx]
        entry = self.entries[ei]
        end = min(start + self.window_len, entry.num_frames)

        if file_sha256(entry.heatmap_npz) != self.frame_integrity[entry.name]["targets_sha256"]:
            raise ValueError(f"Targets changed after integrity preflight: {entry.heatmap_npz}")
        with np.load(entry.heatmap_npz, allow_pickle=False) as d:
            heatmaps = d["heatmaps"][start + 1:end + 1]  # (T, H, W)
            valid = d["valid"][start + 1:end + 1]        # (T,)

        frames_t = torch.stack([self.preprocessor.unit_tensor(self._load_frame(entry.frames_dir, f))
                                for f in range(start + 1, end + 1)])  # RGB float32 [0,1]

        T = frames_t.shape[0]
        if T < self.window_len:
            pad_t = self.window_len - T
            frames_t = torch.cat([frames_t, frames_t.new_zeros((pad_t, *frames_t.shape[1:]))])
            heatmaps = np.concatenate(
                [heatmaps, np.zeros((pad_t, *heatmaps.shape[1:]), dtype=heatmaps.dtype)], axis=0
            )
            valid = np.concatenate([valid, np.zeros(pad_t, dtype=bool)], axis=0)

        heatmaps_t = torch.from_numpy(heatmaps).float()                          # (T,H,W)
        valid_t = torch.from_numpy(valid.copy())                                 # (T,)
        delta_t = torch.zeros(self.window_len, dtype=torch.float32)
        delta_t[:T] = frame_interval(cfg.GAZE_FPS)  # padding is not elapsed video time

        if self.transform is not None:
            frames_t = self.transform(frames_t)
        else:
            frames_t = normalize_rgb_tensor(frames_t, self.preprocessor.profile)

        # is_video_start flags the first chunk of a video, so the training
        # loop knows when to reset (rather than carry forward) the CfC hidden
        # state -- the frame cursor also verifies every continuation is contiguous.
        is_video_start = torch.tensor(start == 0)

        return {
            "video_name": entry.name,
            "start_frame": start,
            "sequence_length": T,  # actual timesteps; tail padding must not advance state
            "frames": frames_t,
            "heatmaps": heatmaps_t,
            "valid": valid_t,
            "delta_t": delta_t,
            "is_video_start": is_video_start,
        }


def get_prev_hmaps(heatmaps: "torch.Tensor", is_video_start: "torch.Tensor") -> "torch.Tensor":
    """Derive the `prev_hmap` input GazeLLNArch.forward() expects, from a
    batch's target heatmap sequence, via teacher forcing (shift by one step).

    Args:
        heatmaps: (B, T, Hg, Wg) ground-truth heatmap sequence for the batch
                  (as returned by DIEMWindowDataset, stacked over the batch
                  dim by your DataLoader).
        is_video_start: (B,) bool -- True where this window is the first
                  window of its video (i.e. where the CfC hidden state gets
                  reset rather than carried over from the previous window).

    Returns:
        prev_hmap: (B, T, Hg, Wg) -- prev_hmap[:, t] = heatmaps[:, t-1] for
                  t>0. For t=0: uses a uniform distribution (maximum-
                  uncertainty prior) if is_video_start is True for that
                  batch element, since there's no real previous fixation to
                  condition on; otherwise (t=0 of a later window in the same
                  video) it's on you to carry the true previous heatmap over
                  from the last step of the prior window in your training
                  loop -- this function only fills in the video-start case,
                  since it doesn't have access to that cross-window state.
    """
    if torch is None:
        raise ImportError("PyTorch is required to use get_prev_hmaps")
    B, T, Hg, Wg = heatmaps.shape
    prev = torch.zeros_like(heatmaps)
    prev[:, 1:] = heatmaps[:, :-1]
    uniform = torch.full((Hg, Wg), 1.0 / (Hg * Wg), dtype=heatmaps.dtype, device=heatmaps.device)
    for b in range(B):
        if bool(is_video_start[b]):
            prev[b, 0] = uniform
    return prev


class VideoOrderedBatchSampler(Sampler):
    """Group contiguous chunks by video, retaining terminal partial chunks.

    Whole videos may be shuffled; chunks within each video stay in order.
    Per-video state cursors remain valid when short videos leave a batch.
    """

    def __init__(self, dataset: DIEMWindowDataset, videos_per_batch: int = 4, shuffle_videos: bool = True, seed: int = 0):
        validate_chunk_dataset(dataset)
        if not isinstance(videos_per_batch, int) or videos_per_batch <= 0:
            raise ValueError("videos_per_batch must be a positive integer")
        self.dataset = dataset
        self.videos_per_batch = videos_per_batch
        self.shuffle_videos = shuffle_videos
        self.seed = seed

        by_video: dict[int, List[int]] = {}
        for i, (ei, start) in enumerate(dataset.index):
            by_video.setdefault(ei, []).append(i)
        for ei in by_video:
            by_video[ei].sort(key=lambda i: dataset.index[i][1])  # order by start_frame
        self.by_video = by_video

    def __iter__(self):
        video_ids = list(self.by_video.keys())
        if self.shuffle_videos:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(video_ids)

        for batch_start in range(0, len(video_ids), self.videos_per_batch):
            group = video_ids[batch_start:batch_start + self.videos_per_batch]
            max_len = max(len(self.by_video[v]) for v in group)
            for step in range(max_len):
                batch = []
                for v in group:
                    seq = self.by_video[v]
                    if step < len(seq):
                        batch.append(seq[step])
                if batch:
                    yield batch

    def __len__(self):
        video_ids = list(self.by_video.keys())
        total = 0
        for batch_start in range(0, len(video_ids), self.videos_per_batch):
            group = video_ids[batch_start:batch_start + self.videos_per_batch]
            total += max(len(self.by_video[v]) for v in group)
        return total


def validate_chunk_dataset(dataset):
    """Reject obsolete/custom replay schedules before any optimization starts."""
    if not isinstance(dataset.window_len, int) or dataset.stride != dataset.window_len or dataset.window_len <= 0:
        raise ValueError("Stateful DIEM training requires contiguous non-overlapping chunks (stride == window_len)")
    if len({entry.name for entry in dataset.entries}) != len(dataset.entries) or any(entry.num_frames <= 0 for entry in dataset.entries):
        raise ValueError("DIEM entries require unique video names and positive frame counts")
    expected = [(ei, start) for ei, entry in enumerate(dataset.entries)
                for start in range(0, entry.num_frames, dataset.window_len)]
    if list(dataset.index) != expected:
        raise ValueError("DIEM chunk index must cover every video frame exactly once, including its final partial chunk")


def build_datasets(processed_root: Path = None, frames_root: Path = None, split_manifest: Path = None, preprocessing="imagenet", window_len=cfg.WINDOW_LEN_FRAMES):
    processed_root = processed_root or cfg.OUTPUT_ROOT
    frames_root = frames_root or (cfg.OUTPUT_ROOT / "frames")
    entries = _discover_videos(processed_root, frames_root)
    if not entries:
        raise RuntimeError(
            "No videos discovered -- check that build_heatmaps.py has been run "
            "and that video frames have been extracted to OUTPUT_ROOT/frames/<video_name>/"
        )
    manifest_path = Path(split_manifest or cfg.SPLIT_MANIFEST).resolve()
    parameters = read_manifest(manifest_path) if manifest_path.is_file() else {}
    train_e, val_e, test_e = split_videos(entries, manifest_path=manifest_path,
        seed=parameters.get("seed", 0), val_frac=parameters.get("val_fraction", 0.15),
        test_frac=parameters.get("test_fraction", 0.15))
    print(f"videos: train={len(train_e)} val={len(val_e)} test={len(test_e)}")
    datasets = tuple(DIEMWindowDataset(entries, preprocessing=preprocessing, window_len=window_len) for entries in (train_e, val_e, test_e))
    for dataset in datasets:
        dataset.split_manifest = manifest_path
    return datasets
