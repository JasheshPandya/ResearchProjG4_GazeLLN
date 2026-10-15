#!/usr/bin/env python3
"""
evaluate_compression.py

Ablation evaluation harness for gaze, uniform and other allocation baselines.

Given a reference (source) video and a set of already-encoded variants
(produced separately, e.g. via Encoder/run_pipeline.py with and without
per-macroblock QP offsets), this script computes, per variant:

    - Bitrate (kbps) / file size
    - Global Y-PSNR and Y-SSIM (whole frame, luma channel)
    - ROI Y-PSNR / Y-SSIM        (macroblocks the saliency model marked
                                   high-priority -- derived from the QP
                                   offset maps written by qp_map_generator.py)
    - Periphery Y-PSNR / Y-SSIM  (all remaining macroblocks)
    - VMAF (global), if the local ffmpeg build has --enable-libvmaf
    - Global five-scale Y-MS-SSIM, global/ROI/periphery Y-MSE
    - Fifth-percentile per-frame PSNR/SSIM/MS-SSIM/VMAF and minimum VMAF
    - Encoder and sequential pipeline throughput, when timings are supplied

...then aggregates results into:
    - results_detailed.csv   one row per encoded variant
    - summary_bdrate.csv     BD-Rate / BD-quality, named test vs anchor per clip
    - report.md              human-readable write-up of both tables + notes
    - results.json           full machine-readable dump
    - run.log                complete execution log (for reproducibility)
    - per_frame/*.csv        compact, aligned frame quality scores
    - plots/                 PNG/PDF research figures and provenance index

WHY A SHARED ROI MASK
----------------------
The whole point of the ablation is "did gaze-guided QP allocation spend bits
where it mattered". To test that fairly, the SAME saliency-derived ROI mask
(taken from one canonical qp_dir per clip, i.e. the mask actually used to
drive the gaze-guided encode) is used to score every variant of a clip --
uniform encodes included. That way "ROI quality" measures the same regions
for both methods; only the encoder's bit allocation differs.

MANIFEST SCHEMA
---------------
{
  "roi_percentile": 25,          // optional, default 25 -> top 25% of MB
                                  // area by saliency is "ROI"
  "vmaf_model_path": null,       // optional path to a VMAF .json model
  "clips": [
    {
      "name": "your_clip.mp4",
      "reference": "path/to/original_source.mp4",
      "roi_source": "path/to/qp_dir",   // output dir of qp_map_generator.py
                                         // for this clip (used for ALL
                                         // variants of this clip)
      "variants": [
        {
          "method": "uniform",          // "uniform" or "gaze" (free text
                                         // otherwise, but BD-rate compares
                                         // all methods against uniform, and gaze against other baselines)
          "label": "crf23",             // any string, used for the RD sweep
          "path": "path/to/uniform_crf23.mp4",
          "encode_time_s": 4.2,         // optional, encoder wall time
          "qp_generation_time_s": 0.0  // zero when no gaze stage is run
        },
        {
          "method": "gaze",
          "label": "crf23",
          "path": "path/to/gaze_crf23.mp4",
          "encode_time_s": 6.8,
          "qp_generation_time_s": 3.2,  // optional, includes gaze inference
          "pipeline_time_s": 10.0      // optional measured sequential total
        }
        // ... more CRF points. >=4 points per method recommended for a
        // trustworthy BD-Rate; 2 points minimum will still run (linear
        // interpolation, flagged as low-confidence in the report).
      ]
    }
  ]
}

USAGE
-----
    python evaluate_compression.py --manifest manifest.json --out_dir results/
    python evaluate_compression.py --print_example > manifest.json
"""

import argparse
import csv
import json
import hashlib
import logging
import os
import subprocess
import sys
from fractions import Fraction
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from project_paths import OUTPUT_FILE_DIR
from dataset_evaluation import (add_statistics_arguments, validate_statistics, aggregate_records,
                                paired_comparisons, statistics_protocol, write_dataset_report)

if __package__:
    from .runtime import find_tool
else:
    from runtime import find_tool

# --------------------------------------------------------------------------- #
# Logging setup (console + run.log, attached once out_dir is known)
# --------------------------------------------------------------------------- #

log = logging.getLogger("eval_compression")
log.setLevel(logging.INFO)
FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"


def attach_logfile(out_dir: Path):
    fh = logging.FileHandler(out_dir / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)
    return fh


@contextmanager
def evaluation_logging(out_dir):
    """Close run-owned handlers even on failure (important for Windows files)."""
    file_handler = attach_logfile(out_dir)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(console)
    try:
        yield
    finally:
        for handler in (file_handler, console):
            log.removeHandler(handler)
            handler.close()


# --------------------------------------------------------------------------- #
# Example manifest (for --print_example)
# --------------------------------------------------------------------------- #

EXAMPLE_MANIFEST = {
    "roi_percentile": 25,
    "vmaf_model_path": None,
    "clips": [
        {
            "name": "your_clip.mp4",
            "reference": "../input video/your_clip.mp4",
            "roi_source": "_work/<video-id>/qp_gaze",
            "variants": [
                {"method": "uniform", "label": "crf20", "path": "out/uniform_crf20.mp4"},
                {"method": "uniform", "label": "crf23", "path": "out/uniform_crf23.mp4"},
                {"method": "uniform", "label": "crf26", "path": "out/uniform_crf26.mp4"},
                {"method": "uniform", "label": "crf29", "path": "out/uniform_crf29.mp4"},
                {"method": "gaze", "label": "crf20", "path": "out/gaze_crf20.mp4", "qp_dir": "out/qp_dir_crf20"},
                {"method": "gaze", "label": "crf23", "path": "out/gaze_crf23.mp4", "qp_dir": "out/qp_dir_crf23"},
                {"method": "gaze", "label": "crf26", "path": "out/gaze_crf26.mp4", "qp_dir": "out/qp_dir_crf26"},
                {"method": "gaze", "label": "crf29", "path": "out/gaze_crf29.mp4", "qp_dir": "out/qp_dir_crf29"},
            ],
        }
    ],
}

EXAMPLE_MANIFEST["clips"][0]["variants"][0].update(
    encode_time_s=4.2, qp_generation_time_s=0.0)
EXAMPLE_MANIFEST["clips"][0]["variants"][4].update(
    encode_time_s=6.8, qp_generation_time_s=3.2)


# --------------------------------------------------------------------------- #
# Media probing (bitrate, resolution, duration) via ffprobe
# --------------------------------------------------------------------------- #

def ffprobe_json(path: Path) -> dict:
    cmd = [
        FFPROBE_BIN, "-v", "error", "-print_format", "json",
        "-show_entries", "format=duration,size,bit_rate:stream=width,height,avg_frame_rate,nb_frames,bit_rate,duration,pix_fmt",
        "-select_streams", "v:0", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    return json.loads(out.stdout)


def get_media_info(path: Path) -> dict:
    info = ffprobe_json(path)
    fmt = info.get("format", {})
    stream = (info.get("streams") or [{}])[0]

    duration = float(fmt.get("duration", 0.0) or 0.0)
    size_bytes = int(fmt.get("size", 0) or 0)

    # Prefer video-only bitrate; container bitrate may include audio/overhead.
    bit_rate = stream.get("bit_rate")
    bitrate_source = "video_stream" if bit_rate not in (None, "N/A", "0") else "container"
    if bitrate_source == "container":
        bit_rate = fmt.get("bit_rate")
    if bit_rate not in (None, "N/A", "0"):
        bitrate_kbps = float(bit_rate) / 1000.0
    elif duration > 0:
        bitrate_kbps = (size_bytes * 8.0) / duration / 1000.0
    else:
        bitrate_kbps = float("nan")

    try:
        fps = float(Fraction(stream.get("avg_frame_rate", "0/1") or "0/1"))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    frame_count = stream.get("nb_frames")
    frame_count = int(frame_count) if frame_count not in (None, "N/A") else None

    return {
        "width": int(stream.get("width", 0) or 0),
        "height": int(stream.get("height", 0) or 0),
        "duration_s": duration,
        "size_mb": size_bytes / (1024 * 1024),
        "bitrate_kbps": bitrate_kbps,
        "bitrate_source": bitrate_source,
        "fps": fps,
        "frame_count": frame_count,
        "pix_fmt": stream.get("pix_fmt", "unknown"),
    }


# VMAF availability check + computation

_VMAF_AVAILABLE: Optional[bool] = None


def vmaf_available() -> bool:
    global _VMAF_AVAILABLE
    if _VMAF_AVAILABLE is None:
        try:
            out = subprocess.run([FFMPEG_BIN, "-filters"], capture_output=True, text=True)
        except FileNotFoundError:
            _VMAF_AVAILABLE = False
            log.warning("ffmpeg is not on PATH; VMAF is unavailable.")
            return False
        _VMAF_AVAILABLE = "libvmaf" in out.stdout
        if not _VMAF_AVAILABLE:
            log.warning(
                "This ffmpeg build does not expose the 'libvmaf' filter. "
                "VMAF scores will be reported as N/A. Rebuild ffmpeg with "
                "--enable-libvmaf, or install a build that has it, to get VMAF."
            )
    return _VMAF_AVAILABLE


def compute_vmaf(ref_path: Path, dist_path: Path, model_path: Optional[str], work_dir: Path,
                 frame_limit: Optional[int] = None) -> tuple:
    """Compute VMAF for a distorted video against a reference.

    Returns
    -------
    (global_vmaf, per_frame_scores) : tuple[Optional[float], list[float]]
        global_vmaf  -- mean VMAF over all frames (None if unavailable).
        per_frame_scores -- list of per-frame VMAF values (empty if unavailable).
    """
    if not vmaf_available():
        return None, []

    import re, tempfile, shutil

    # ffmpeg's filter option parser can't handle Windows paths (backslashes
    # are escape chars, colons are option separators). Write the JSON log to
    # a temp file first, then copy it to the vmaf_logs subfolder.
    # We run ffmpeg with cwd=tempdir and use just the filename to avoid
    # any path separator or colon issues.
    tmp_fd, tmp_log = tempfile.mkstemp(suffix=".json", prefix="vmaf_")
    os.close(tmp_fd)
    tmp_log_path = Path(tmp_log)
    tmp_dir = str(tmp_log_path.parent)
    log_filename = tmp_log_path.name

    # Copy custom models to a plain filename to avoid filter escaping on Windows.
    model_opt = ":model=version=vmaf_v0.6.1"
    model_copy = None
    if model_path:
        model_copy = tmp_log_path.with_suffix(".model.json")
        shutil.copy2(model_path, model_copy)
        model_opt = f":model=path={model_copy.name}"
    filt = (
        f"[0:v]settb=AVTB,setpts=PTS-STARTPTS[dist];[1:v]settb=AVTB,setpts=PTS-STARTPTS[ref];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={log_filename}:pool=mean:shortest=1:repeatlast=0{model_opt}"
    )
    cmd = [FFMPEG_BIN, "-y", "-i", str(dist_path.resolve()), "-i", str(ref_path.resolve()),
           "-lavfi", filt]
    if frame_limit is not None:
        cmd += ["-frames:v", str(frame_limit)]
    cmd += ["-f", "null", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_dir)
    finally:
        if model_copy is not None:
            model_copy.unlink(missing_ok=True)
    if out.returncode != 0:
        log.warning(f"VMAF computation failed for {dist_path.name}: {out.stderr[-500:]}")
        tmp_log_path.unlink(missing_ok=True)
        return None, []

    # Save the JSON log to vmaf_logs/ subfolder (best-effort).
    vmaf_logs_dir = work_dir / "vmaf_logs"
    vmaf_logs_dir.mkdir(parents=True, exist_ok=True)
    final_log_path = vmaf_logs_dir / f"vmaf_{dist_path.stem}.json"
    log_written = tmp_log_path.exists() and tmp_log_path.stat().st_size > 0
    if log_written:
        shutil.copy2(str(tmp_log_path), str(final_log_path))
        log.info(f"  VMAF log saved to {final_log_path}")
    tmp_log_path.unlink(missing_ok=True)

    # Do not parse an older run's saved log if the current run produced none.
    per_frame_scores = []
    if log_written:
        try:
            with open(final_log_path) as jf:
                vmaf_json = json.load(jf)
            for frame in vmaf_json.get("frames", []):
                metrics = frame.get("metrics", {})
                score = metrics.get("vmaf")
                if score is not None:
                    per_frame_scores.append(float(score))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log.warning(f"Could not parse per-frame VMAF from {final_log_path}: {exc}")

    # Parse the global VMAF score from ffmpeg's stderr output.
    # ffmpeg always prints: [Parsed_libvmaf_N @ 0xADDR] VMAF score: XX.XXXXXX
    global_vmaf = None
    match = re.search(r"VMAF score:\s*([\d.]+)", out.stderr)
    if match:
        global_vmaf = float(match.group(1))
    elif per_frame_scores:
        # Fallback: compute from per-frame scores if stderr parsing failed.
        global_vmaf = float(np.mean(per_frame_scores))
    else:
        log.warning(f"Could not parse VMAF score from ffmpeg output for {dist_path.name}")

    return global_vmaf, per_frame_scores


# ROI / periphery masks, derived from the saliency-driven QP offsets already
# written by Encoder/qp_map_generator.py (metadata.json + qpoffset_NNNNNN.bin)

def write_frame_metrics(result, work_dir, series):
    """Compact frame CSV for fresh temporal plots, separate from results.json."""
    key = json.dumps([result.clip, result.method, result.label, result.path])
    stem = hashlib.sha256(key.encode()).hexdigest()[:16]
    relative = Path("per_frame") / f"scores_{stem}.csv"
    target = work_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if any(values and len(values) != result.frames_evaluated for values in series.values()):
        raise ValueError("Per-frame export must align with the scored video")
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", *series])
        for index in range(result.frames_evaluated):
            values = []
            for scores in series.values():
                value = scores[index] if scores else float("nan")
                values.append("" if np.isnan(value) else format(value, ".10g"))
            time_s = format(index / result.fps, ".10g") if np.isfinite(result.fps) and result.fps > 0 else ""
            writer.writerow([index, time_s, *values])
    result.frame_metrics_path = relative.as_posix()


def vmaf_windowed_averages(per_frame_scores: list, window: int = 20) -> list:
    """Compute the average VMAF over non-overlapping windows of `window` frames.

    Returns a list of (start_frame, end_frame, avg_vmaf) tuples.  The last
    window may be shorter than `window` frames if the total frame count is
    not evenly divisible.
    """
    if not per_frame_scores:
        return []
    avgs = []
    scores = np.asarray(per_frame_scores, dtype=np.float64)
    for start in range(0, len(scores), window):
        end = min(start + window, len(scores))  # exclusive
        avgs.append((start, end - 1, float(np.mean(scores[start:end]))))
    return avgs


# ROI / periphery masks, derived from the saliency-driven QP offsets already
# written by Encoder/qp_map_generator.py (metadata.json + qpoffset_NNNNNN.bin)

@dataclass
class RoiSource:
    mb_width: int
    mb_height: int
    qp_min: float
    qp_max: float
    num_frames: int
    qp_dir: Path

    def saliency_grid(self, frame_idx: int) -> np.ndarray:
        """Recover normalized [0,1] saliency (1=high) for one frame,
        inverting qp_map_generator.map_to_qp_offsets()."""
        idx = min(frame_idx, self.num_frames - 1)  # hold last frame if short
        bin_path = self.qp_dir / f"qpoffset_{idx:06d}.bin"
        offsets = np.fromfile(bin_path, dtype=np.float32).reshape(self.mb_height, self.mb_width)
        rng = self.qp_max - self.qp_min
        if rng <= 1e-8:
            return np.zeros_like(offsets)
        # offset = qp_max - (qp_max - qp_min) * normalized  =>  invert:
        normalized = (self.qp_max - offsets) / rng
        return np.clip(normalized, 0.0, 1.0)

    def roi_mask(self, frame_idx: int, out_h: int, out_w: int, roi_percentile: float) -> np.ndarray:
        """Boolean mask at native (out_h, out_w) resolution. True = ROI
        (top `roi_percentile`% of macroblocks by saliency, this frame)."""
        sal = self.saliency_grid(frame_idx)
        if not 0 < roi_percentile < 100:
            raise ValueError("roi_percentile must be between 0 and 100 (exclusive)")
        if (self.mb_height, self.mb_width) != ((out_h + 15) // 16, (out_w + 15) // 16):
            raise ValueError("ROI macroblock grid does not match the reference resolution")
        # Exact count, deterministic ties. A threshold can select the whole frame
        # when the saliency map is flat, leaving no meaningful periphery score.
        count = int(np.ceil(sal.size * roi_percentile / 100))
        mb_mask = np.zeros(sal.size, dtype=bool)
        mb_mask[np.argsort(-sal.ravel(), kind="stable")[:count]] = True
        mb_mask = mb_mask.reshape(sal.shape)
        return np.repeat(np.repeat(mb_mask, 16, axis=0), 16, axis=1)[:out_h, :out_w]


def load_roi_source(qp_dir: Path) -> RoiSource:
    meta_path = qp_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata.json in {qp_dir}. Expected the output directory of "
            f"Encoder/qp_map_generator.py (per-frame qpoffset_*.bin + metadata.json)."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    if any(meta[key] <= 0 for key in ("mb_width", "mb_height", "num_frames")):
        raise ValueError("ROI metadata must specify positive dimensions and frame count")
    return RoiSource(
        mb_width=meta["mb_width"],
        mb_height=meta["mb_height"],
        qp_min=meta["qp_min"],
        qp_max=meta["qp_max"],
        num_frames=meta["num_frames"],
        qp_dir=qp_dir,
    )


# Pixel-domain metrics: Y-PSNR, Y-SSIM (Wang et al. 2004, 11x11 Gaussian window)

_SSIM_KSIZE = (11, 11)
_SSIM_SIGMA = 1.5


def _gblur(x: np.ndarray) -> np.ndarray:
    """ Gaussian window is separable -- cv2.GaussianBlur uses a separable,
    SIMD-optimized path and is dramatically faster than an equivalent cv2.filter2D with a dense 2D kernel
    (seconds -> milliseconds per 1080p frame in testing). Keep everything in float32 for speed. """
    return cv2.GaussianBlur(x, _SSIM_KSIZE, _SSIM_SIGMA)


def squared_error_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Compute pixel-wise squared error between two arrays. """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return (a - b) ** 2


def psnr_from_se_map(se_map: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    vals = se_map[mask] if mask is not None else se_map
    if vals.size == 0:
        return float("nan")
    return psnr_from_mse(float(np.mean(vals, dtype=np.float64)))


def psnr_from_mse(mse: float) -> float:
    if np.isnan(mse):
        return float("nan")
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def _ssim_maps(a: np.ndarray, b: np.ndarray) -> tuple:
    """Full per-pixel SSIM map (Wang et al. 2004, Gaussian-windowed).
    Compute once per frame pair and reuse for global/ROI/periphery
    averages -- this is the expensive step, so avoid recomputing it
    per region."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    C1 = np.float32((0.01 * 255) ** 2)
    C2 = np.float32((0.03 * 255) ** 2)

    mu1 = _gblur(a)
    mu2 = _gblur(b)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = np.maximum(_gblur(a * a) - mu1_sq, 0)
    sigma2_sq = np.maximum(_gblur(b * b) - mu2_sq, 0)
    sigma12 = _gblur(a * b) - mu1_mu2

    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    return luminance * cs, cs


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _ssim_maps(a, b)[0]


_MS_SSIM_WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])


def ms_ssim(a: np.ndarray, b: np.ndarray, first_maps: Optional[tuple] = None) -> float:
    """Five-scale Y-MS-SSIM, Wang et al. (2003), 11x11 Gaussian, sigma 1.5.

    Use valid windows, 2x2 mean downsampling (reflect-pad odd dimensions),
    and nonnegative factors before fractional powers. Return NaN if any scale
    is smaller than 11 pixels; do not silently change the number of scales.
    """
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("MS-SSIM requires equally sized 2-D luma frames")
    a, b = a.astype(np.float32), b.astype(np.float32)
    factors = []
    for level in range(5):
        if min(a.shape) < 11:
            return float("nan")
        smap, cs = first_maps if level == 0 and first_maps is not None else _ssim_maps(a, b)
        component = smap if level == 4 else cs
        factors.append(np.clip(np.mean(component[5:-5, 5:-5], dtype=np.float64), 0, 1))
        if level < 4:
            padding = ((0, a.shape[0] % 2), (0, a.shape[1] % 2))
            a = np.pad(a, padding, mode="reflect")
            b = np.pad(b, padding, mode="reflect")
            a = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3))
            b = b.reshape(b.shape[0] // 2, 2, b.shape[1] // 2, 2).mean(axis=(1, 3))
    return float(np.prod(np.asarray(factors) ** _MS_SSIM_WEIGHTS))


def score_percentile(scores, percentile=5) -> Optional[float]:
    """Ignore unavailable scores but retain +inf for perfect PSNR frames."""
    values = np.asarray(scores, dtype=float)
    values = values[~np.isnan(values)]
    if not len(values):
        return None
    # Explicit interpolation avoids inf-inf producing NaN for lossless frames.
    values.sort()
    position = (len(values) - 1) * percentile / 100
    lo, hi = int(np.floor(position)), int(np.ceil(position))
    if lo == hi or values[lo] == values[hi]:
        return float(values[lo])
    if np.isinf(values[hi]):
        return float("inf")
    return float(values[lo] + (position - lo) * (values[hi] - values[lo]))


def available_mean(scores) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[~np.isnan(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def ssim_from_map(smap: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    vals = smap[mask] if mask is not None else smap
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def to_luma(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]


# --------------------------------------------------------------------------- #
# Per-variant evaluation: decode reference + variant together, score frames
# --------------------------------------------------------------------------- #

@dataclass
class VariantResult:
    clip: str
    method: str
    label: str
    path: str
    width: int = 0
    height: int = 0
    frames_evaluated: int = 0
    duration_s: float = 0.0
    size_mb: float = 0.0
    bitrate_kbps: float = float("nan")
    bitrate_source: str = "unknown"
    fps: float = 0.0
    bits_per_pixel_frame: Optional[float] = None
    roi_fraction: float = float("nan")
    mse_global: float = float("nan")
    mse_roi: float = float("nan")
    mse_periphery: float = float("nan")
    psnr_global: float = float("nan")
    psnr_roi: float = float("nan")
    psnr_periphery: float = float("nan")
    ssim_global: float = float("nan")
    ssim_roi: float = float("nan")
    ssim_periphery: float = float("nan")
    ms_ssim_global: float = float("nan")
    psnr_p5: Optional[float] = None
    ssim_p5: Optional[float] = None
    ms_ssim_p5: Optional[float] = None
    vmaf_global: Optional[float] = None
    frame_metrics_path: Optional[str] = None
    vmaf_per_20f: list = field(default_factory=list)  # [(start, end, avg), ...]
    vmaf_p5: Optional[float] = None
    vmaf_min: Optional[float] = None
    vmaf_frames: int = 0
    vmaf_model: str = "version=vmaf_v0.6.1"
    encode_time_s: Optional[float] = None
    encode_fps: Optional[float] = None
    qp_generation_time_s: Optional[float] = None
    pipeline_time_s: Optional[float] = None
    pipeline_fps: Optional[float] = None
    pipeline_processing_ms_per_frame: Optional[float] = None
    notes: str = ""


def apply_timings(res: VariantResult, variant: dict, frame_count: Optional[int]):
    for name in ("encode_time_s", "qp_generation_time_s", "pipeline_time_s"):
        value = variant.get(name)
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            setattr(res, name, value)
    if res.pipeline_time_s is None and res.encode_time_s is not None:
        if res.qp_generation_time_s is not None:
            res.pipeline_time_s = res.encode_time_s + res.qp_generation_time_s
    if frame_count and frame_count > 0:
        if res.encode_time_s is not None and res.encode_time_s > 0:
            res.encode_fps = frame_count / res.encode_time_s
        if res.pipeline_time_s is not None and res.pipeline_time_s > 0:
            res.pipeline_fps = frame_count / res.pipeline_time_s
            res.pipeline_processing_ms_per_frame = 1000 * res.pipeline_time_s / frame_count


def evaluate_variant(clip_name: str, ref_path: Path, variant: dict,
                      roi_source: RoiSource, roi_percentile: float,
                      vmaf_model_path: Optional[str], work_dir: Path,
                      manifest_dir: Optional[Path] = None) -> VariantResult:
    dist_path = resolve_manifest_path(variant["path"], manifest_dir or Path.cwd())
    method = variant["method"]
    label = variant.get("label", dist_path.stem)

    res = VariantResult(clip=clip_name, method=method, label=label, path=str(dist_path))
    notes = []

    if not dist_path.exists():
        res.notes = f"MISSING FILE: {dist_path}"
        log.error(res.notes)
        return res

    info = get_media_info(dist_path)
    res.width, res.height = info["width"], info["height"]
    res.duration_s = info["duration_s"]
    res.size_mb = info["size_mb"]
    res.bitrate_kbps = info["bitrate_kbps"]
    res.bitrate_source = info["bitrate_source"]
    res.fps = info["fps"]
    res.vmaf_model = str(vmaf_model_path) if vmaf_model_path else "version=vmaf_v0.6.1"
    if res.width and res.height and res.fps > 0:
        res.bits_per_pixel_frame = res.bitrate_kbps * 1000 / (res.width * res.height * res.fps)

    cap_ref = cv2.VideoCapture(str(ref_path))
    cap_dist = cv2.VideoCapture(str(dist_path))
    if not cap_ref.isOpened() or not cap_dist.isOpened():
        cap_ref.release()
        cap_dist.release()
        res.notes = "Could not open reference or variant video with OpenCV."
        log.error(res.notes)
        return res

    n_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT))
    n_dist = int(cap_dist.get(cv2.CAP_PROP_FRAME_COUNT))
    apply_timings(res, variant, info["frame_count"] or n_dist)
    n_frames = min(n_ref, n_dist)
    ref_fps = cap_ref.get(cv2.CAP_PROP_FPS)
    dist_fps = cap_dist.get(cv2.CAP_PROP_FPS)
    ref_shape = (int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH)))
    dist_shape = (int(cap_dist.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap_dist.get(cv2.CAP_PROP_FRAME_WIDTH)))
    if n_ref != n_dist or not np.isclose(ref_fps, dist_fps, rtol=1e-4) or ref_shape != dist_shape:
        res.notes = (f"Incompatible videos: frames {n_ref}/{n_dist}, fps {ref_fps:g}/{dist_fps:g}, "
                     f"size {ref_shape}/{dist_shape}. Align frames and resolution before scoring.")
        log.error(res.notes)
        cap_ref.release()
        cap_dist.release()
        return res
    if roi_source.num_frames < n_frames:
        res.notes = f"ROI source has {roi_source.num_frames} frames; video has {n_frames}."
        log.error(res.notes)
        cap_ref.release()
        cap_dist.release()
        return res

    psnr_g, psnr_r, psnr_p = [], [], []
    ssim_g, ssim_r, ssim_p = [], [], []
    ms_scores = []
    se_totals = np.zeros(3, dtype=np.float64)
    pixel_counts = np.zeros(3, dtype=np.int64)

    for idx in range(n_frames):
        ok_r, fr_ref = cap_ref.read()
        ok_d, fr_dist = cap_dist.read()
        if not (ok_r and ok_d):
            break

        y_ref = to_luma(fr_ref)
        y_dist = to_luma(fr_dist)

        roi_mask = roi_source.roi_mask(idx, y_ref.shape[0], y_ref.shape[1], roi_percentile)
        periph_mask = ~roi_mask

        se_map = squared_error_map(y_ref, y_dist)
        psnr_g.append(psnr_from_se_map(se_map))
        psnr_r.append(psnr_from_se_map(se_map, roi_mask))
        psnr_p.append(psnr_from_se_map(se_map, periph_mask))
        for region, vals in enumerate((se_map, se_map[roi_mask], se_map[periph_mask])):
            se_totals[region] += np.sum(vals, dtype=np.float64)
            pixel_counts[region] += vals.size

        maps = _ssim_maps(y_ref, y_dist)
        smap = maps[0]
        ssim_g.append(ssim_from_map(smap))
        ssim_r.append(ssim_from_map(smap, roi_mask))
        ssim_p.append(ssim_from_map(smap, periph_mask))
        ms_scores.append(ms_ssim(y_ref, y_dist, first_maps=maps))

    cap_ref.release()
    cap_dist.release()

    res.frames_evaluated = len(psnr_g)
    if res.frames_evaluated == 0:
        notes.append("No frames could be scored.")
        res.notes = "; ".join(notes)
        log.error(f"[{clip_name}/{method}/{label}] {res.notes}")
        return res

    if res.frames_evaluated != n_frames:
        res.notes = f"Decode stopped after {res.frames_evaluated}/{n_frames} frames; quality scores discarded."
        return res
    # Pool squared error over pixels before converting to dB. This also weights
    # varying ROI areas correctly, rather than averaging per-frame dB values.
    for region, name in enumerate(("global", "roi", "periphery")):
        mse = se_totals[region] / pixel_counts[region] if pixel_counts[region] else float("nan")
        setattr(res, f"mse_{name}", float(mse))
        setattr(res, f"psnr_{name}", psnr_from_mse(mse))
    res.roi_fraction = float(pixel_counts[1] / pixel_counts[0])
    res.ssim_global = available_mean(ssim_g)
    res.ssim_roi = available_mean(ssim_r)
    res.ssim_periphery = available_mean(ssim_p)
    res.ms_ssim_global = available_mean(ms_scores)
    res.psnr_p5 = score_percentile(psnr_g)
    res.ssim_p5 = score_percentile(ssim_g)
    res.ms_ssim_p5 = score_percentile(ms_scores)
    if np.isnan(res.ms_ssim_global):
        notes.append("Five-scale MS-SSIM unavailable: a downsampled scale is smaller than 11 pixels.")

    vmaf_score, vmaf_per_frame = compute_vmaf(ref_path, dist_path, vmaf_model_path, work_dir, frame_limit=n_frames)
    if vmaf_per_frame and (len(vmaf_per_frame) != n_frames or not np.isfinite(vmaf_per_frame).all()):
        notes.append("VMAF frame scores are incomplete or invalid; VMAF scores discarded.")
        vmaf_score, vmaf_per_frame = None, []
    res.vmaf_global = vmaf_score
    res.vmaf_per_20f = vmaf_windowed_averages(vmaf_per_frame, window=20)
    res.vmaf_frames = len(vmaf_per_frame)
    res.vmaf_p5 = score_percentile(vmaf_per_frame)
    res.vmaf_min = min(vmaf_per_frame) if vmaf_per_frame else None
    write_frame_metrics(res, work_dir, {
        "psnr_global": psnr_g, "psnr_roi": psnr_r, "psnr_periphery": psnr_p,
        "ssim_global": ssim_g, "ssim_roi": ssim_r, "ssim_periphery": ssim_p,
        "ms_ssim_global": ms_scores, "vmaf_global": vmaf_per_frame,
    })

    res.notes = "; ".join(notes)
    if res.vmaf_per_20f:
        n_windows = len(res.vmaf_per_20f)
        log.info(f"  VMAF per-20-frame windows ({n_windows} windows): "
                 + ", ".join(f"f{s}-{e}: {a:.2f}" for s, e, a in res.vmaf_per_20f))
    log.info(
        f"[{clip_name}/{method}/{label}] {res.frames_evaluated} frames | "
        f"{res.bitrate_kbps:.1f} kbps | PSNR(g/roi/per)="
        f"{res.psnr_global:.2f}/{res.psnr_roi:.2f}/{res.psnr_periphery:.2f} dB | "
        f"SSIM(g/roi/per)={res.ssim_global:.4f}/{res.ssim_roi:.4f}/{res.ssim_periphery:.4f}"
        + f" | MS-SSIM={res.ms_ssim_global:.4f}"
        + (f" | VMAF={res.vmaf_global:.2f}" if res.vmaf_global is not None else "")
    )
    return res


# --------------------------------------------------------------------------- #
# BD-Rate / BD-quality (Bjontegaard delta), classic piecewise-polynomial method
# --------------------------------------------------------------------------- #

def _bd_fit_degree(n_points: int) -> int:
    return max(1, min(3, n_points - 1))


def validate_rd(rate_anchor, qual_anchor, rate_test, qual_test) -> str:
    """Reject unavailable, degenerate, or nonmonotonic RD curves before fitting."""
    for rates, qualities in ((rate_anchor, qual_anchor), (rate_test, qual_test)):
        rates, qualities = np.asarray(rates, float), np.asarray(qualities, float)
        if rates.ndim != 1 or qualities.ndim != 1 or len(rates) != len(qualities):
            return "rate and quality arrays must be equally sized 1-D arrays"
        if len(rates) < 2:
            return "insufficient RD points (need >=2 per method, >=4 recommended)"
        if not np.isfinite(rates).all() or not np.isfinite(qualities).all() or np.any(rates <= 0):
            return "RD points contain unavailable/non-finite quality or invalid bitrate"
        order = np.argsort(rates)
        if np.any(np.diff(rates[order]) <= 0) or np.any(np.diff(qualities[order]) <= 0):
            return "RD curves must have distinct bitrates and strictly increasing quality"
    return ""


def bd_rate(rate_anchor, qual_anchor, rate_test, qual_test):
    """% bitrate change of `test` vs `anchor` at equal quality.
    Negative = test needs fewer bits for the same quality (an improvement
    when `test` is the gaze-guided method)."""
    invalid = validate_rd(rate_anchor, qual_anchor, rate_test, qual_test)
    if invalid:
        return None, invalid
    rate_anchor, qual_anchor = np.asarray(rate_anchor, float), np.asarray(qual_anchor, float)
    rate_test, qual_test = np.asarray(rate_test, float), np.asarray(qual_test, float)

    if len(rate_anchor) < 2 or len(rate_test) < 2:
        return None, "insufficient RD points (need >=2 per method, >=4 recommended)"

    log_r_a, log_r_t = np.log(rate_anchor), np.log(rate_test)
    deg_a, deg_t = _bd_fit_degree(len(rate_anchor)), _bd_fit_degree(len(rate_test))

    p_a = np.polyfit(qual_anchor, log_r_a, deg_a)
    p_t = np.polyfit(qual_test, log_r_t, deg_t)

    lo = max(qual_anchor.min(), qual_test.min())
    hi = min(qual_anchor.max(), qual_test.max())
    if hi <= lo:
        return None, "no overlapping quality range between methods"

    int_a = np.polyval(np.polyint(p_a), hi) - np.polyval(np.polyint(p_a), lo)
    int_t = np.polyval(np.polyint(p_t), hi) - np.polyval(np.polyint(p_t), lo)
    avg_log_diff = (int_t - int_a) / (hi - lo)

    pct = (np.exp(avg_log_diff) - 1.0) * 100.0
    note = "" if (len(rate_anchor) >= 4 and len(rate_test) >= 4) else \
        f"low-confidence (only {len(rate_anchor)}/{len(rate_test)} RD points; recommend >=4)"
    return float(pct), note


def bd_quality(rate_anchor, qual_anchor, rate_test, qual_test):
    """Average quality delta of `test` vs `anchor` at equal bitrate.
    Positive = test has higher quality at the same bitrate."""
    invalid = validate_rd(rate_anchor, qual_anchor, rate_test, qual_test)
    if invalid:
        return None, invalid
    rate_anchor, qual_anchor = np.asarray(rate_anchor, float), np.asarray(qual_anchor, float)
    rate_test, qual_test = np.asarray(rate_test, float), np.asarray(qual_test, float)

    if len(rate_anchor) < 2 or len(rate_test) < 2:
        return None, "insufficient RD points (need >=2 per method, >=4 recommended)"

    log_r_a, log_r_t = np.log(rate_anchor), np.log(rate_test)
    deg_a, deg_t = _bd_fit_degree(len(rate_anchor)), _bd_fit_degree(len(rate_test))

    p_a = np.polyfit(log_r_a, qual_anchor, deg_a)
    p_t = np.polyfit(log_r_t, qual_test, deg_t)

    lo = max(log_r_a.min(), log_r_t.min())
    hi = min(log_r_a.max(), log_r_t.max())
    if hi <= lo:
        return None, "no overlapping bitrate range between methods"

    int_a = np.polyval(np.polyint(p_a), hi) - np.polyval(np.polyint(p_a), lo)
    int_t = np.polyval(np.polyint(p_t), hi) - np.polyval(np.polyint(p_t), lo)
    avg_diff = (int_t - int_a) / (hi - lo)

    note = "" if (len(rate_anchor) >= 4 and len(rate_test) >= 4) else \
        f"low-confidence (only {len(rate_anchor)}/{len(rate_test)} RD points; recommend >=4)"
    return float(avg_diff), note


# --------------------------------------------------------------------------- #
# Report writers
# --------------------------------------------------------------------------- #

DETAILED_FIELDS = [
    "clip", "method", "label", "path", "width", "height", "frames_evaluated",
    "duration_s", "size_mb", "bitrate_kbps",
    "psnr_global", "psnr_roi", "psnr_periphery",
    "ssim_global", "ssim_roi", "ssim_periphery",
    "vmaf_global", "encode_time_s", "notes",
    "bitrate_source", "fps", "bits_per_pixel_frame", "roi_fraction",
    "mse_global", "mse_roi", "mse_periphery", "ms_ssim_global",
    "psnr_p5", "ssim_p5", "ms_ssim_p5", "vmaf_p5", "vmaf_min", "vmaf_frames", "vmaf_model",
    "frame_metrics_path",
    "encode_fps", "qp_generation_time_s", "pipeline_time_s", "pipeline_fps", "pipeline_processing_ms_per_frame",
]

SUMMARY_METRICS = [
    ("psnr", "psnr_global"), ("ssim", "ssim_global"), ("vmaf", "vmaf_global"),
    ("ms_ssim", "ms_ssim_global"), ("psnr_roi", "psnr_roi"), ("ssim_roi", "ssim_roi"),
    ("psnr_periphery", "psnr_periphery"), ("ssim_periphery", "ssim_periphery"),
    ("vmaf_p5", "vmaf_p5"),
]


def _vmaf_window_columns(results: list) -> list:
    """Determine the dynamic vmaf_avg_fN_M column names needed to cover
    the longest vmaf_per_20f list across all results."""
    max_windows = max((len(r.vmaf_per_20f) for r in results), default=0)
    cols = []
    for i in range(max_windows):
        start = i * 20
        end = start + 19
        cols.append(f"vmaf_avg_f{start}_{end}")
    return cols


def write_detailed_csv(results: list, out_path: Path):
    vmaf_cols = _vmaf_window_columns(results)
    all_fields = DETAILED_FIELDS + vmaf_cols
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in DETAILED_FIELDS}
            # Fill in the per-20-frame VMAF average columns.
            for idx, col in enumerate(vmaf_cols):
                if idx < len(r.vmaf_per_20f):
                    _, _, avg = r.vmaf_per_20f[idx]
                    row[col] = f"{avg:.4f}"
                else:
                    row[col] = ""
            w.writerow(row)


def compute_summary(results: list) -> list:
    """Compare every method to uniform, plus gaze to each other baseline."""
    by_clip = {}
    for r in results:
        by_clip.setdefault(r.clip, {}).setdefault(r.method, []).append(r)
    rows = []
    for clip, methods in by_clip.items():
        pairs = [("uniform", method) for method in methods if method != "uniform"] if "uniform" in methods else []
        if "gaze" in methods:
            pairs += [(method, "gaze") for method in methods if method not in ("uniform", "gaze")]
        if not pairs:
            rows.append({"clip": clip, "note": "need uniform and another method, or gaze and another baseline"})
        for anchor, test in pairs:
            u = sorted(methods[anchor], key=lambda r: r.bitrate_kbps)
            g = sorted(methods[test], key=lambda r: r.bitrate_kbps)
            row = dict(clip=clip, anchor_method=anchor, test_method=test,
                       n_anchor_points=len(u), n_test_points=len(g),
                       n_uniform_points=len(u) if anchor == "uniform" else None,
                       n_gaze_points=len(g) if test == "gaze" else None)
            for metric_name, attr in SUMMARY_METRICS:
                ru, qu = [x.bitrate_kbps for x in u], [getattr(x, attr) for x in u]
                rg, qg = [x.bitrate_kbps for x in g], [getattr(x, attr) for x in g]
                if any(v is None or not np.isfinite(v) for v in qu + qg):
                    row[f"bdrate_{metric_name}_pct"] = None
                    row[f"bd_{metric_name}"] = None
                    row[f"{metric_name}_note"] = f"{metric_name} unavailable/non-finite for one or more points"
                    continue
                pct, note1 = bd_rate(ru, qu, rg, qg)
                delta, note2 = bd_quality(ru, qu, rg, qg)
                row[f"bdrate_{metric_name}_pct"] = pct
                row[f"bd_{metric_name}"] = delta
                row[f"{metric_name}_note"] = note1 or note2
            rows.append(row)
    return rows


def write_summary_csv(summary_rows: list, out_path: Path):
    fields = ["clip", "anchor_method", "test_method", "n_anchor_points", "n_test_points", "n_uniform_points", "n_gaze_points"]
    for name, _ in SUMMARY_METRICS:
        fields.extend([f"bdrate_{name}_pct", f"bd_{name}", f"{name}_note"])
    fields.append("note")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in fields})


def fmt(v, nd=3):
    if v is None:
        return "N/A"
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def json_safe(value):
    """Strict JSON: unavailable values are null; perfect PSNR is 'Infinity'."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def resolve_manifest_path(value, manifest_dir: Path) -> Path:
    # Existing manifests were produced on Windows; allow their relative paths
    # to be read on POSIX as well.
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (manifest_dir / path).resolve()


def write_markdown_report(results: list, summary_rows: list, out_path: Path,
                           manifest_path: Path, roi_percentile: float):
    import datetime
    lines = []
    lines.append("# Compression Baselines -- Evaluation Report\n")
    lines.append(f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- ROI definition: top {roi_percentile:.0f}% of macroblocks by predicted "
                  f"saliency (per frame), derived from the QP-offset maps of the canonical "
                  f"`roi_source` for each clip. Same mask used to score every variant of a clip.")
    lines.append("- Pixel metrics use 8-bit OpenCV luma (Y): MSE, PSNR, SSIM and MS-SSIM. "
                 "PSNR uses pooled pixel MSE (not mean frame dB); SSIM uses 11x11 Gaussian windows, sigma=1.5. "
                 "MS-SSIM uses five scales and Wang weights; unavailable for undersized frames. "
                 "These pixel metrics do not measure chroma fidelity.")
    lines.append("- ROI selects an exact ranked macroblock count (rounded up, ties deterministic); "
                 "reported ROI pixel fraction accounts for partial edge macroblocks. "
                 "ROI masks are predicted gaze, not independent human eye-tracking measurements.")
    lines.append(f"- VMAF: {'available' if vmaf_available() else 'NOT available in this ffmpeg build (N/A below)'}.\n")

    lines.append("## 1. Per-variant results\n")
    header = ["Clip", "Method", "Label", "Bitrate (kbps)", "Size (MiB)",
              "PSNR global/ROI/periph (dB)", "SSIM global/ROI/periph",
              "VMAF", "Encode time (s)", "Notes"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in sorted(results, key=lambda r: (r.clip, r.method, r.bitrate_kbps)):
        psnr_str = f"{fmt(r.psnr_global,2)} / {fmt(r.psnr_roi,2)} / {fmt(r.psnr_periphery,2)}"
        ssim_str = f"{fmt(r.ssim_global,4)} / {fmt(r.ssim_roi,4)} / {fmt(r.ssim_periphery,4)}"
        lines.append("| " + " | ".join([
            r.clip, r.method, r.label, fmt(r.bitrate_kbps, 1), fmt(r.size_mb, 2),
            psnr_str, ssim_str, fmt(r.vmaf_global, 2), fmt(r.encode_time_s, 2),
            r.notes or "",
        ]) + " |")

    lines.append("\n## Additional quality measurements\n")
    header = ["Clip", "Method", "Label", "MSE global/ROI/periph", "ROI pixels (%)",
              "MS-SSIM", "PSNR p5 (dB)", "SSIM p5", "MS-SSIM p5", "VMAF p5/min", "VMAF model"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in results:
        mse_str = " / ".join(fmt(getattr(r, f"mse_{region}"), 3) for region in ("global", "roi", "periphery"))
        lines.append("| " + " | ".join([
            r.clip, r.method, r.label, mse_str, fmt(r.roi_fraction * 100, 2),
            fmt(r.ms_ssim_global, 4), fmt(r.psnr_p5, 2), fmt(r.ssim_p5, 4), fmt(r.ms_ssim_p5, 4),
            f"{fmt(r.vmaf_p5, 2)} / {fmt(r.vmaf_min, 2)}", r.vmaf_model,
        ]) + " |")
    lines.append("\np5 is the fifth percentile of per-frame scores, not the percentile of 20-frame averages. "
                 "Higher is better. MSE is lower-is-better. Infinite PSNR means identical luma samples.")

    lines.append("\n## Processing cost\n")
    header = ["Clip", "Method", "Label", "Encoder fps", "QP generation (s)",
              "Sequential pipeline (s)", "Pipeline fps", "Processing ms/frame", "Bitrate source"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in results:
        lines.append("| " + " | ".join([
            r.clip, r.method, r.label, fmt(r.encode_fps, 2), fmt(r.qp_generation_time_s, 3),
            fmt(r.pipeline_time_s, 3), fmt(r.pipeline_fps, 2), fmt(r.pipeline_processing_ms_per_frame, 3),
            r.bitrate_source,
        ]) + " |")
    lines.append("\nPipeline time is QP generation plus encoding when both are measured, or an explicit "
                 "pipeline_time_s. Each standalone gaze encode is charged the full QP-generation cost, "
                 "even when the sweep caches those maps. Missing stage timings remain N/A. "
                 "Throughput and average processing cost are not capture-to-display latency.")

    lines.append("\n## 2. BD-Rate / BD-quality summary (test vs. anchor, per clip)\n")
    lines.append(
        "Negative BD-Rate % = the test method needs fewer bits than its anchor for equal quality "
        "(bitrate savings). Positive BD-quality = the test method has higher quality at equal "
        "bitrate. Both are computed over the bitrate/quality range common to both methods.\n"
    )
    header2 = ["Clip", "Anchor", "Test", "Metric", "#Anchor pts", "#Test pts", "BD-Rate (%)", "BD-quality", "Notes"]
    lines.append("| " + " | ".join(header2) + " |")
    lines.append("|" + "---|" * len(header2))
    for row in summary_rows:
        for name, _ in SUMMARY_METRICS:
            lines.append("| " + " | ".join([
                row["clip"], row.get("anchor_method", "-"), row.get("test_method", "-"), name,
                str(row.get("n_anchor_points", "-")), str(row.get("n_test_points", "-")),
                fmt(row.get(f"bdrate_{name}_pct"), 2), fmt(row.get(f"bd_{name}"), 4),
                row.get(f"{name}_note") or row.get("note", ""),
            ]) + " |")

    lines.append("\n## 3. How to read this\n")
    lines.append(
        "- **Global** columns show whole-frame quality -- the classic RD comparison.\n"
        "- **ROI** columns isolate the macroblocks the saliency model flagged as important; "
        "this is where the gaze-guided method should show a clear advantage.\n"
        "- **Periphery** columns show the rest of the frame; some quality loss here relative "
        "to uniform encoding at the same overall bitrate is expected and is the mechanism "
        "by which bits were saved/reallocated.\n"
        "- BD-Rate/BD-quality require >=2 RD points per method to compute at all, and the "
        "cubic fit uses up to degree 3; >=4 points are recommended but do not guarantee "
        "a reliable fit. Invalid/nonmonotonic curves are reported as N/A. Sweep more CRF/QP-scale "
        "values if you see that flag.\n"
        "- Compare methods at equal bitrate or equal quality, not equal CRF alone.\n"
        "- PSNR/SSIM implementation, pooling, decoding path and VMAF model must stay fixed "
        "across comparisons. Old mean-frame-dB results are not directly comparable to pooled-MSE PSNR.\n"
    )

    out_path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _default_manifest() -> Path:
    """Use the shared output root."""
    return OUTPUT_FILE_DIR / "manifest.json"


def main(argv=None):
    global FFMPEG_BIN, FFPROBE_BIN, _VMAF_AVAILABLE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=_default_manifest(),
                        help="Dataset manifest generated by run_ablation_sweep.py; default: shared output/manifest.json.")
    parser.add_argument("--out_dir", "--out-dir", "--output-dir", type=Path, default=None,
                        help="Default: compression_evaluation beside the manifest.")
    parser.add_argument("--print_example", action="store_true", help="Print an example manifest.json and exit.")
    parser.add_argument("--ffmpeg", help="FFmpeg executable; otherwise discover on PATH or in this virtual environment.")
    parser.add_argument("--ffprobe", help="FFprobe executable; otherwise discover on PATH or in this virtual environment.")
    parser.add_argument("--no-plots", action="store_true", help="Export metrics without generating figures.")
    parser.add_argument("--plot-frames", type=int, nargs="+",
                        help="Zero-based spatial example frames; otherwise each clip's plot_frames or midpoint.")
    add_statistics_arguments(parser)
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir or args.manifest.resolve().parent / "compression_evaluation"
    try:
        validate_statistics(args.bootstrap_samples, args.confidence)
    except ValueError as exc:
        parser.error(str(exc))

    if args.print_example:
        print(json.dumps(EXAMPLE_MANIFEST, indent=2))
        return

    if not args.manifest.exists():
        parser.error(
            f"No --manifest provided and default not found at {args.manifest}.\n"
            f"Either pass --manifest path/to/manifest.json or run run_ablation_sweep.py first."
        )

    try:
        FFPROBE_BIN = find_tool("ffprobe", explicit=args.ffprobe)
        FFMPEG_BIN = find_tool("ffmpeg", explicit=args.ffmpeg, required=bool(args.ffmpeg)) or "ffmpeg"
    except RuntimeError as exc:
        parser.error(str(exc))
    _VMAF_AVAILABLE = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with evaluation_logging(args.out_dir):
        log.info(f"Loading manifest: {args.manifest}")
        manifest_dir = args.manifest.resolve().parent
        with open(args.manifest) as f:
            manifest = json.load(f)

        roi_percentile = float(manifest.get("roi_percentile", 25))
        if not 0 < roi_percentile < 100:
            parser.error("roi_percentile must be between 0 and 100 (exclusive)")
        vmaf_model_path = manifest.get("vmaf_model_path")
        if vmaf_model_path:
            vmaf_model_path = str(resolve_manifest_path(vmaf_model_path, manifest_dir))
            if not Path(vmaf_model_path).exists():
                parser.error(f"VMAF model not found: {vmaf_model_path}")
        vmaf_available()  # logs a warning once, up front, if unavailable

        clips = manifest.get("clips", [])
        names = [c["name"] for c in clips]
        if not names or len(set(names)) != len(names):
            parser.error("Manifest needs nonempty clips with unique names")
        references = [resolve_manifest_path(c["reference"], manifest_dir) for c in clips]
        if len(references) != len(set(references)):
            parser.error("A source video may appear only once in the dataset manifest")
        for clip in clips:
            keys = [(v["method"], v.get("label", Path(v["path"]).stem)) for v in clip["variants"]]
            if not keys or len(keys) != len(set(keys)):
                parser.error(f"Clip {clip['name']} needs unique method/label variants")
        all_results = []
        coverage = []
        for clip in manifest["clips"]:
            clip_name = clip["name"]
            ref_path = resolve_manifest_path(clip["reference"], manifest_dir)
            log.info(f"\n=== Clip: {clip_name} ===")

            if not ref_path.exists():
                log.error(f"Reference video not found: {ref_path}. Skipping clip.")
                coverage.append(dict(video=clip_name, status="missing_reference"))
                continue

            try:
                roi_src_path = resolve_manifest_path(clip["roi_source"], manifest_dir)
                roi_source = load_roi_source(roi_src_path)
            except FileNotFoundError as e:
                log.error(str(e) + f" Skipping clip '{clip_name}'.")
                coverage.append(dict(video=clip_name, status="missing_roi"))
                continue

            for variant in clip["variants"]:
                # qp_dir identifies how a variant was encoded, never how it is scored.
                # All methods must be evaluated on the canonical clip-level ROI.
                res = evaluate_variant(clip_name, ref_path, variant, roi_source,
                                        roi_percentile, vmaf_model_path, args.out_dir,
                                        manifest_dir=manifest_dir)
                all_results.append(res)
                coverage.append(dict(video=clip_name, method=res.method, label=res.label,
                                     status="scored" if res.frames_evaluated > 0 else "failed", notes=res.notes))

        if not all_results:
            log.error("No results were produced. Check the manifest paths and try again.")
            sys.exit(1)

        write_detailed_csv(all_results, args.out_dir / "results_detailed.csv")
        summary_rows = compute_summary(all_results)
        write_summary_csv(summary_rows, args.out_dir / "summary_bdrate.csv")
        write_markdown_report(all_results, summary_rows, args.out_dir / "report.md", args.manifest, roi_percentile)

        metrics = ["bitrate_kbps", "size_mb", "mse_global", "mse_roi", "mse_periphery",
                   "psnr_global", "psnr_roi", "psnr_periphery", "ssim_global", "ssim_roi", "ssim_periphery",
                   "ms_ssim_global", "vmaf_global", "psnr_p5", "ssim_p5", "ms_ssim_p5", "vmaf_p5", "vmaf_min",
                   "encode_fps", "pipeline_fps", "pipeline_processing_ms_per_frame"]
        records = [asdict(r) for r in all_results]
        settings = dict(samples=args.bootstrap_samples, confidence=args.confidence, seed=args.statistics_seed,
                        expected_videos=names)
        bd_metrics = [key for name, _ in SUMMARY_METRICS for key in (f"bdrate_{name}_pct", f"bd_{name}")]
        bd_rows = [row for row in summary_rows if "anchor_method" in row]
        dataset = dict(videos=names, coverage=coverage,
            aggregate=aggregate_records(records, ["method", "label"], metrics, video_field="clip", **settings),
            paired=paired_comparisons(records, metrics, group_fields=["label"], video_field="clip", **settings),
            bdrate_aggregate=aggregate_records(bd_rows, ["anchor_method", "test_method"], bd_metrics,
                                               video_field="clip", **settings),
            protocol=statistics_protocol(args.bootstrap_samples, args.confidence, args.statistics_seed),
            manifest=str(args.manifest.resolve()))
        dataset["per_video"] = [dict(clip=row["clip"], method=row["method"], label=row["label"],
                                      **{key: row[key] if row[key] is not None and np.isfinite(row[key]) else None
                                         for key in metrics}) for row in records]
        dataset["protocol"]["within_video"] = "PSNR from pooled pixel MSE; SSIM/MS-SSIM/VMAF mean frames. Across videos: arithmetic mean of clip scores; BD curves fitted separately per clip before averaging."
        dataset["protocol"]["coverage"] = "All requested videos form the denominator. Missing references/ROI/variants recorded; inspect coverage before interpreting partial results."
        write_dataset_report(args.out_dir, dataset)
        with (args.out_dir / "report.md").open("a", encoding="utf-8") as stream:
            stream.write("\n\n## Dataset statistics\n\nSee [dataset report](dataset_report.md) for video-level means, confidence intervals, paired tests and coverage.\n")

        with open(args.out_dir / "results.json", "w") as f:
            json.dump(json_safe({
                "detailed": [asdict(r) for r in all_results],
                "summary": summary_rows,
                "dataset": dataset,
                "roi_percentile": roi_percentile,
                "protocol": {
                    "pixel_metrics": "8-bit luma decoded through OpenCV BGR->YCrCb",
                    "psnr_pooling": "pooled pixel MSE, then convert to dB",
                    "ssim_pooling": "arithmetic mean of per-frame spatial means",
                    "ms_ssim": "five scales, Wang weights, valid 11x11 Gaussian windows, 2x2 average downsampling",
                    "quality_percentile": "5th percentile of per-frame scores, linear interpolation",
                    "roi_source": "shared predicted-gaze mask, exact ranked macroblock count",
                    "vmaf_model": vmaf_model_path or "version=vmaf_v0.6.1",
                    "ffmpeg": FFMPEG_BIN,
                    "ffprobe": FFPROBE_BIN,
                    "timing": "offline sequential processing throughput; not measured capture-to-display latency",
                    "unavailable": "null; infinite PSNR is represented as the string Infinity",
                },
            }), f, indent=2, allow_nan=False)

        if not args.no_plots:
            try:
                if __package__:
                    from .plot_bdrate import generate_plots
                else:
                    from plot_bdrate import generate_plots
                plot_index = generate_plots(args.out_dir / "results_detailed.csv",
                                            manifest_path=args.manifest, frames=args.plot_frames)
                with (args.out_dir / "report.md").open("a", encoding="utf-8") as f:
                    f.write("\n\n## Generated figures\n\nSee [figure index](plots/README.md) for PNG/PDF figures, "
                            "bitrate-pair details and unavailable inputs.\n")
                    if plot_index["skipped"]:
                        f.write("\nSome figures were skipped; their reasons are recorded in the index.\n")
            except Exception:
                # A rendering failure must not discard an already completed evaluation.
                log.exception("Figure generation failed; metric exports remain available. "
                              "Retry with plot_bdrate.py --all --csv results_detailed.csv.")

        log.info(f"\nDone. Wrote:")
        log.info(f"  {args.out_dir / 'results_detailed.csv'}")
        log.info(f"  {args.out_dir / 'summary_bdrate.csv'}")
        log.info(f"  {args.out_dir / 'report.md'}")
        log.info(f"  {args.out_dir / 'results.json'}")
        log.info(f"  {args.out_dir / 'run.log'}")


if __name__ == "__main__":
    main()
