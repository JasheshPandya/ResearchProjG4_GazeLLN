#!/usr/bin/env python3
"""Export compression research figures as PNG and PDF, without re-encoding.

Single RD plot: python Encoder/plot_bdrate.py --metric psnr_roi
Full suite:     python Encoder/plot_bdrate.py --all --manifest manifest.json
Evaluation automatically calls generate_plots() after exporting its metrics.
"""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Windows/macOS/Linux, including machines without a display.
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter
import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from project_paths import OUTPUT_FILE_DIR

if __package__:
    from .evaluate_compression import (bd_rate, validate_rd, SUMMARY_METRICS,
                                       load_roi_source, resolve_manifest_path, to_luma)
else:
    from evaluate_compression import (bd_rate, validate_rd, SUMMARY_METRICS,
                                      load_roi_source, resolve_manifest_path, to_luma)

log = logging.getLogger("eval_compression")
METRIC_LABELS = {
    "vmaf_global": "Full-frame VMAF", "psnr_global": "Global Y-PSNR (dB)",
    "ssim_global": "Global Y-SSIM", "ms_ssim_global": "Global Y-MS-SSIM",
    "psnr_roi": "Predicted-ROI Y-PSNR (dB)", "ssim_roi": "Predicted-ROI Y-SSIM",
    "psnr_periphery": "Periphery Y-PSNR (dB)", "ssim_periphery": "Periphery Y-SSIM",
    "vmaf_p5": "VMAF fifth percentile",
    "vmaf_window_mean": "20-frame mean VMAF",
}
COLORS = {"uniform": "#3366CC", "gaze": "#DC3912"}
METHODS = {"uniform": "Baseline (no saliency offsets)", "gaze": "Gaze-guided"}
RD_PANELS = ("vmaf_global", "psnr_roi", "psnr_periphery")


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_name(value):
    """Avoid path traversal, reserved filenames, and case-insensitive collisions."""
    import re
    slug = re.sub(r"[^a-z0-9_-]+", "_", str(value).lower()).strip("_")[:70] or "clip"
    return slug + "_" + hashlib.sha256(str(value).encode()).hexdigest()[:8]


def load_csv(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_clip(rows, clip_name=None):
    clips = sorted({r["clip"] for r in rows})
    if clip_name is None:
        if len(clips) != 1:
            raise ValueError(f"Specify --clip for a single plot; CSV clips: {clips}")
        clip_name = clips[0]
    filtered = [r for r in rows if r["clip"] == clip_name]
    if not filtered:
        raise ValueError(f"No rows for clip {clip_name!r}")
    return clip_name, filtered


def split_methods(rows):
    return tuple(sorted((r for r in rows if r["method"] == method),
                        key=lambda r: number(r.get("bitrate_kbps")))
                 for method in ("uniform", "gaze"))


def extract_rd(rows, metric="vmaf_global"):
    return (np.array([number(r.get("bitrate_kbps")) for r in rows]),
            np.array([number(r.get(metric)) for r in rows]))


def fit_degree(n_points):
    return max(1, min(3, n_points - 1))


def fit_lograte_vs_quality(rates, quals):
    return np.poly1d(np.polyfit(quals, np.log(rates), fit_degree(len(rates))))


def compute_bdrate(rate_anchor, qual_anchor, rate_test, qual_test):
    return bd_rate(rate_anchor, qual_anchor, rate_test, qual_test)


def draw_rd(ax, rate_u, qual_u, rate_g, qual_g, metric):
    """Always show available measured points; only fit valid complete curves."""
    for method, rates, quals, marker in (("uniform", rate_u, qual_u, "o"),
                                       ("gaze", rate_g, qual_g, "^")):
        valid = np.isfinite(rates) & (rates > 0) & np.isfinite(quals)
        ax.scatter(rates[valid], quals[valid], marker=marker, s=36,
                   facecolors="white", edgecolors=COLORS[method], zorder=4,
                   label=METHODS[method])
    value, note = compute_bdrate(rate_u, qual_u, rate_g, qual_g)
    invalid = validate_rd(rate_u, qual_u, rate_g, qual_g)
    if not invalid:
        fits = [fit_lograte_vs_quality(r, q) for r, q in ((rate_u, qual_u), (rate_g, qual_g))]
        for method, q, fit in zip(("uniform", "gaze"), (qual_u, qual_g), fits):
            dense = np.linspace(q.min(), q.max(), 250)
            ax.plot(np.exp(fit(dense)), dense, color=COLORS[method], lw=1.6)
        lo, hi = max(qual_u.min(), qual_g.min()), min(qual_u.max(), qual_g.max())
        if value is not None:
            q = np.linspace(lo, hi, 250)
            ax.fill_betweenx(q, np.exp(fits[0](q)), np.exp(fits[1](q)),
                             color="#FFC857", alpha=.22)
            ax.axhline(lo, color="gray", ls="--", lw=.7)
            ax.axhline(hi, color="gray", ls="--", lw=.7)
    annotation = f"BD-rate: {value:+.2f}%" if value is not None else "BD-rate: unavailable"
    if note:
        annotation += "\n" + note
    ax.text(.03, .97, annotation, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="lightgray", alpha=.9))
    ax.set(xscale="log", xlabel="Video bitrate (kbps)", ylabel=METRIC_LABELS[metric])
    positive_rates = np.concatenate((rate_u, rate_g))
    positive_rates = positive_rates[np.isfinite(positive_rates) & (positive_rates > 0)]
    if not len(positive_rates):
        ax.set_xlim(1, 10)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", ls=":", alpha=.35)
    return {"metric": metric, "bdrate_pct": value, "note": note}


def save_figure(fig, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for extension in ("png", "pdf"):
            path = out_dir / f"{stem}.{extension}"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            paths.append(path.name)
    finally:
        plt.close(fig)
    return paths


def plot_bdrate(clip_name, rate_u, qual_u, rate_g, qual_g, out_dir, metric="vmaf_global"):
    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    detail = draw_rd(ax, rate_u, qual_u, rate_g, qual_g, metric)
    ax.set_title(clip_name)
    ax.legend(loc="lower right", fontsize=8)
    stem = f"bdrate_{safe_name(clip_name)}"
    if metric != "vmaf_global":
        stem += "_" + metric
    return save_figure(fig, out_dir, stem), detail


def plot_rd_panels(clip, rows, out_dir):
    uniform, gaze = split_methods(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    details = [draw_rd(ax, *extract_rd(uniform, metric), *extract_rd(gaze, metric), metric)
               for ax, metric in zip(axes, RD_PANELS)]
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle(f"{clip}: whole-frame, predicted-ROI and peripheral fidelity")
    fig.supxlabel("Measured points; cubic fits within measured ranges. Each panel has its own quality overlap.",
                  fontsize=9)
    return save_figure(fig, out_dir, f"rd_panels_{safe_name(clip)}"), details


def plot_all_methods(clip, rows, out_dir):
    """Measured RD points for every method; no inferred statistical significance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for method in sorted({r["method"] for r in rows}):
        group = sorted((r for r in rows if r["method"] == method), key=lambda r: number(r.get("bitrate_kbps")))
        for ax, metric in zip(axes, RD_PANELS):
            rates, qualities = extract_rd(group, metric)
            valid = np.isfinite(rates) & (rates > 0) & np.isfinite(qualities)
            ax.plot(rates[valid], qualities[valid], "o-", lw=1, ms=4, label=method)
            ax.set(xscale="log", xlabel="Video bitrate (kbps)", ylabel=METRIC_LABELS[metric])
            if not valid.any():
                ax.set_xlim(1, 10)
            ax.grid(True, which="both", ls=":", alpha=.35)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{clip}: all allocation methods, shared scoring mask")
    fig.supxlabel("Measured points connected in bitrate order. Pairwise BD-rate values are in summary_bdrate.csv.", fontsize=9)
    return save_figure(fig, out_dir, f"rd_all_methods_{safe_name(clip)}")


def select_bitrate_pair(rows):
    """Closest measured rates inside their common rate interval; never interpolate images."""
    uniform, gaze = split_methods(rows)
    def valid(r):
        return (number(r.get("bitrate_kbps")) > 0 and np.isfinite(number(r.get("bitrate_kbps")))
                and number(r.get("frames_evaluated")) > 0 and np.isfinite(number(r.get("psnr_roi"))))
    uniform, gaze = [r for r in uniform if valid(r)], [r for r in gaze if valid(r)]
    if not uniform or not gaze:
        raise ValueError("No evaluated baseline/gaze pair")
    lo = max(number(uniform[0]["bitrate_kbps"]), number(gaze[0]["bitrate_kbps"]))
    hi = min(number(uniform[-1]["bitrate_kbps"]), number(gaze[-1]["bitrate_kbps"]))
    if hi < lo:
        raise ValueError("No common measured bitrate interval for visual/temporal comparison")
    pairs = [(u, g) for u in uniform for g in gaze
             if lo <= number(u["bitrate_kbps"]) <= hi and lo <= number(g["bitrate_kbps"]) <= hi]
    if not pairs:
        raise ValueError("No measured operating points within the common bitrate interval")
    u, g = min(pairs, key=lambda pair: abs(np.log(number(pair[1]["bitrate_kbps"]) /
                                                     number(pair[0]["bitrate_kbps"]))))
    gap = (number(g["bitrate_kbps"]) / number(u["bitrate_kbps"]) - 1) * 100
    return u, g, {"baseline_label": u["label"], "gaze_label": g["label"],
                  "baseline_bitrate_kbps": number(u["bitrate_kbps"]),
                  "gaze_bitrate_kbps": number(g["bitrate_kbps"]),
                  "gaze_rate_delta_pct": gap, "within_five_percent": abs(gap) <= 5 + 1e-9}


def pair_caption(pair):
    return (f"Nearest available bitrates: baseline {pair['baseline_label']} "
            f"{pair['baseline_bitrate_kbps']:.1f} kbps; gaze {pair['gaze_label']} "
            f"{pair['gaze_bitrate_kbps']:.1f} kbps (rate delta {pair['gaze_rate_delta_pct']:+.1f}%).")


def read_frame_series(row, results_dir):
    """Use fresh compact CSVs, or verified legacy VMAF logs for older runs."""
    n = int(number(row.get("frames_evaluated")))
    filename = row.get("frame_metrics_path")
    if filename:
        path = resolve_manifest_path(filename, Path(results_dir).resolve())
        records = load_csv(path)
        if len(records) != n or [int(r["frame"]) for r in records] != list(range(n)):
            raise ValueError("Incomplete or misaligned per-frame CSV")
        fields = ("vmaf_global", "psnr_roi", "ssim_roi")
        series = {key: np.array([number(r.get(key)) for r in records]) for key in fields}
        series = {key: values for key, values in series.items() if np.isfinite(values).all()}
    else:
        # Do not reuse an older log when this evaluation had no valid VMAF.
        if not np.isfinite(number(row.get("vmaf_global"))) or number(row.get("vmaf_frames")) != n:
            raise ValueError("No current per-frame scores available")
        stem = Path(str(row["path"]).replace("\\", "/")).stem
        path = Path(results_dir) / "vmaf_logs" / f"vmaf_{stem}.json"
        if not path.exists():
            return read_windowed_vmaf(row, n)
        records = json.loads(path.read_text())["frames"]
        if len(records) != n or [r["frameNum"] for r in records] != list(range(n)):
            raise ValueError("Incomplete or misaligned legacy VMAF log")
        series = {"vmaf_global": np.array([number(r["metrics"].get("vmaf")) for r in records])}
    if "vmaf_global" in series:
        values = series["vmaf_global"]
        if not np.isfinite(values).all() or not np.isclose(values.mean(), number(row.get("vmaf_global")), atol=.001, rtol=0):
            raise ValueError("Per-frame VMAF does not match the current aggregate result")
    if not series:
        raise ValueError("Per-frame quality scores unavailable")
    return series


def read_windowed_vmaf(row, n):
    """Old CSVs retain 20-frame means; never pretend these are frame scores."""
    import re
    windows = []
    for key, value in row.items():
        match = re.fullmatch(r"vmaf_avg_f(\d+)_(\d+)", key)
        if match and np.isfinite(number(value)):
            start, end = map(int, match.groups())
            windows.append((start, min(end, n - 1), number(value)))
    windows.sort()
    position = 0
    for start, end, _ in windows:
        if start != position or end < start:
            raise ValueError("Incomplete or misaligned VMAF windows")
        position = end + 1
    if not windows or position != n:
        raise ValueError("No complete current frame scores or VMAF windows")
    weights = [end - start + 1 for start, end, _ in windows]
    values = np.array([value for _, _, value in windows])
    if not np.isclose(np.average(values, weights=weights), number(row.get("vmaf_global")), atol=.001, rtol=0):
        raise ValueError("Windowed VMAF does not match the current aggregate result")
    return {"vmaf_window_mean": values,
            "window_centers": np.array([(start + end) / 2 for start, end, _ in windows])}


def plot_temporal(clip, u, g, pair, results_dir, out_dir, clip_config):
    if (number(u.get("frames_evaluated")) != number(g.get("frames_evaluated")) or
            not np.isclose(number(u.get("fps")), number(g.get("fps")), rtol=1e-4)):
        raise ValueError("Temporal comparison requires matching frame counts and fps")
    series_u, series_g = read_frame_series(u, results_dir), read_frame_series(g, results_dir)
    metrics = [key for key in ("vmaf_global", "psnr_roi", "ssim_roi", "vmaf_window_mean")
               if key in series_u and key in series_g]
    if not metrics:
        raise ValueError("No shared per-frame quality metric")
    fps = number(u.get("fps"))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Missing frame rate for temporal plot")
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 2.5 * len(metrics)),
                             squeeze=False, sharex=True, layout="constrained")
    cuts = clip_config.get("scene_cut_frames", [])
    for ax, metric in zip(axes[:, 0], metrics):
        for method, series in (("uniform", series_u), ("gaze", series_g)):
            values = series[metric]
            frame_positions = series["window_centers"] if metric == "vmaf_window_mean" else np.arange(len(values))
            ax.plot(frame_positions / fps, values, color=COLORS[method], lw=1,
                    label=METHODS[method])
            percentile = number((u if method == "uniform" else g).get("vmaf_p5")) if metric == "vmaf_window_mean" else np.percentile(values, 5)
            if np.isfinite(percentile):
                ax.axhline(percentile, color=COLORS[method], ls=":", lw=.8)
        for cut in cuts:
            if isinstance(cut, int) and 0 < cut < number(u["frames_evaluated"]):
                ax.axvline(cut / fps, color="gray", ls="--", lw=.8)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("Time (s)")
    pooling = "20-frame means (legacy CSV)" if "vmaf_window_mean" in metrics else "per-frame quality"
    fig.suptitle(f"{clip}: {pooling}\n{pair_caption(pair)}", fontsize=11)
    fig.supxlabel("Dotted lines: per-frame fifth percentiles (saved scores for legacy CSVs). Dashed vertical lines: supplied scene boundaries.", fontsize=8)
    return save_figure(fig, out_dir, f"temporal_quality_{safe_name(clip)}"), metrics


def plot_runtime(clip, rows, out_dir, hardware=None):
    rows = [r for group in split_methods(rows) for r in group
            if number(r.get("frames_evaluated")) > 0 and np.isfinite(number(r.get("encode_time_s")))]
    if not rows:
        raise ValueError("No measured encoding times")
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * .65), 4.8), layout="constrained")
    unknown = False
    for i, r in enumerate(rows):
        n = number(r["frames_evaluated"])
        encoding = number(r["encode_time_s"]) * 1000 / n
        generation = number(r.get("qp_generation_time_s")) * 1000 / n
        ax.bar(i, encoding, color="#3366CC", label="Encoding" if i == 0 else None)
        if np.isfinite(generation):
            ax.bar(i, generation, bottom=encoding, color="#E69F00")
        else:
            unknown = True
            ax.text(i, encoding, "?", ha="center", va="bottom")
        total = number(r.get("pipeline_time_s")) * 1000 / n
        if np.isfinite(total):
            ax.scatter(i, total, marker="_", s=100, color="black", zorder=3)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color="#3366CC", label="Encoding"),
                       Patch(color="#E69F00", label="QP generation"),
                       Line2D([], [], color="black", marker="_", ls="", label="Reported pipeline total")], fontsize=8)
    ax.set_xticks(range(len(rows)), [f"{'Baseline' if r['method'] == 'uniform' else 'Gaze'}\n{r['label']}" for r in rows], rotation=40, ha="right")
    ax.set_ylabel("Offline processing cost (ms/frame)")
    ax.set_title(f"{clip}: runtime breakdown")
    hardware_text = json.dumps(hardware, ensure_ascii=True) if hardware else "Hardware not recorded"
    fig.supxlabel(hardware_text + "; not capture-to-display latency." +
                  (" ? = QP generation time unknown, not zero." if unknown else ""), fontsize=8)
    ax.grid(axis="y", alpha=.25)
    return save_figure(fig, out_dir, f"runtime_{safe_name(clip)}")


def read_video_frame(path, index, expected):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        h, w, n, fps = expected
        actual = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS))
        if actual[:3] != (h, w, n) or not np.isclose(actual[3], fps, rtol=1e-4):
            raise ValueError(f"Video metadata no longer matches evaluation: {path}")
        if not 0 <= index < n:
            raise ValueError(f"Frame {index} is outside [0, {n - 1}]")
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not decode frame {index}: {path}")
        return frame
    finally:
        cap.release()


def variant_path(row, clip_config, manifest_dir):
    # Prefer portable manifest paths when an experiment has moved machines.
    for variant in clip_config.get("variants", []):
        candidate = resolve_manifest_path(variant["path"], manifest_dir)
        if variant["method"] == row["method"] and variant.get("label", candidate.stem) == row["label"]:
            return candidate
    return resolve_manifest_path(row["path"], manifest_dir)


def choose_crop(mask, roi, crop_h, crop_w):
    """Highest ROI/periphery occupancy; deterministic center tie-break."""
    target = mask if roi else ~mask
    integral = cv2.integral(target.astype(np.uint8))
    counts = (integral[crop_h:, crop_w:] - integral[:-crop_h, crop_w:]
              - integral[crop_h:, :-crop_w] + integral[:-crop_h, :-crop_w])
    ys, xs = np.where(counts == counts.max())
    center = np.argmin((ys - (mask.shape[0] - crop_h) / 2) ** 2 + (xs - (mask.shape[1] - crop_w) / 2) ** 2)
    y, x = int(ys[center]), int(xs[center])
    return y, x, crop_h, crop_w


def plot_spatial(clip, u, g, pair, clip_config, manifest_dir, roi_percentile, frame_index, out_dir):
    expected = tuple(int(number(u[k])) for k in ("height", "width", "frames_evaluated")) + (number(u["fps"]),)
    if any(number(u[k]) != number(g[k]) for k in ("height", "width", "frames_evaluated")) or not np.isclose(number(u["fps"]), number(g["fps"]), rtol=1e-4):
        raise ValueError("Spatial comparison requires matching evaluated videos")
    images = [read_video_frame(path, frame_index, expected) for path in
              (resolve_manifest_path(clip_config["reference"], manifest_dir),
               variant_path(u, clip_config, manifest_dir), variant_path(g, clip_config, manifest_dir))]
    h, w = expected[:2]
    roi = load_roi_source(resolve_manifest_path(clip_config["roi_source"], manifest_dir))
    if roi.num_frames < expected[2]:
        raise ValueError("QP maps do not cover the evaluated video")
    mask = roi.roi_mask(frame_index, h, w, roi_percentile)
    saliency = np.repeat(np.repeat(roi.saliency_grid(frame_index), 16, 0), 16, 1)[:h, :w]
    offsets = roi.qp_max - (roi.qp_max - roi.qp_min) * saliency
    errors = [np.abs(to_luma(images[0]).astype(float) - to_luma(im).astype(float)) for im in images[1:]]
    error_max = max(1., max(float(np.percentile(e, 99)) for e in errors))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), layout="constrained")
    axes[0, 0].imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Source + predicted-ROI boundary")
    if mask.any() and (~mask).any():
        axes[0, 0].contour(mask.astype(float), levels=[.5], colors=["#00FFFF"], linewidths=.7)
    artist = axes[0, 1].imshow(saliency, cmap="magma", vmin=0, vmax=1)
    axes[0, 1].set_title("Saliency reconstructed from QP map")
    fig.colorbar(artist, ax=axes[0, 1], shrink=.7)
    artist = axes[0, 2].imshow(offsets, cmap="coolwarm", vmin=roi.qp_min, vmax=roi.qp_max)
    axes[0, 2].set_title("Injected QP offsets")
    fig.colorbar(artist, ax=axes[0, 2], shrink=.7)
    axes[1, 0].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Shared predicted ROI ({mask.mean() * 100:.2f}% pixels)")
    for ax, error, title in zip(axes[1, 1:], errors, ("Baseline absolute luma error", "Gaze absolute luma error")):
        artist = ax.imshow(error, cmap="inferno", vmin=0, vmax=error_max)
        ax.set_title(title)
    fig.colorbar(artist, ax=list(axes[1, 1:]), shrink=.7, label=f"8-bit luma error (clipped at {error_max:.1f})")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{clip}, frame {frame_index}\n{pair_caption(pair)}", fontsize=11)
    allocation_files = save_figure(fig, out_dir, f"allocation_{safe_name(clip)}_f{frame_index:06d}")

    crop_h, crop_w = max(1, min(h, max(32, h // 4))), max(1, min(w, max(32, w // 4)))
    crops = [choose_crop(mask, region, crop_h, crop_w) for region in (True, False)]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout="constrained")
    titles = ["Source", f"Baseline {u['label']}", f"Gaze {g['label']}"]
    for col, im in enumerate(images):
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(titles[col])
        for row_index, (y, x, ch, cw) in enumerate(crops, 1):
            axes[row_index, col].imshow(rgb[y:y + ch, x:x + cw], interpolation="nearest")
    from matplotlib.patches import Rectangle
    for ax in axes[0]:
        for color, (y, x, ch, cw) in zip(("#00FFFF", "#FFD700"), crops):
            ax.add_patch(Rectangle((x, y), cw, ch, fill=False, edgecolor=color, linewidth=1))
    for row_index, name, crop in zip((1, 2), ("ROI crop", "Periphery crop"), crops):
        y, x, ch, cw = crop
        occupancy = mask[y:y + ch, x:x + cw].mean() * 100
        axes[row_index, 0].set_ylabel(f"{name}\nROI coverage {occupancy:.1f}%")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{clip}, frame {frame_index}: identical spatial crops\n{pair_caption(pair)}", fontsize=11)
    fig.supxlabel("These are measured encodes at the displayed rates, not an interpolated equal-bitrate reconstruction.", fontsize=8)
    comparison_files = save_figure(fig, out_dir, f"comparison_{safe_name(clip)}_f{frame_index:06d}")
    return allocation_files, comparison_files, crops


def plot_summary(rows, out_dir):
    clips = sorted({r["clip"] for r in rows})
    metrics = ("vmaf_global", "psnr_roi", "psnr_periphery")
    fig, ax = plt.subplots(figsize=(10, max(3.5, len(clips) * 1.5)), layout="constrained")
    values = []
    for j, metric in enumerate(metrics):
        for i, clip in enumerate(clips):
            _, clip_rows = filter_clip(rows, clip)
            u, g = split_methods(clip_rows)
            value, note = compute_bdrate(*extract_rd(u, metric), *extract_rd(g, metric))
            y = i + (j - 1) * .24
            if value is None:
                ax.text(0, y, "N/A", fontsize=8, va="center")
            else:
                ax.barh(y, value, height=.22, color=("#3366CC", "#009E73", "#DC3912")[j])
                ax.annotate(f"{value:+.2f}%", (value, y), xytext=(4 if value >= 0 else -4, 0),
                            textcoords="offset points", ha="left" if value >= 0 else "right", va="center", fontsize=8)
            values.append({"clip": clip, "metric": metric, "bdrate_pct": value, "note": note})
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=METRIC_LABELS[m]) for c, m in
                       zip(("#3366CC", "#009E73", "#DC3912"), metrics)], fontsize=8)
    ax.set_yticks(range(len(clips)), clips)
    ax.axvline(0, color="black", lw=.8)
    ax.set_xlabel("Gaze versus baseline BD-rate (%)")
    ax.set_title("Per-clip allocation trade-off")
    ax.margins(x=.2)
    ax.grid(axis="x", alpha=.2)
    fig.supxlabel("Negative = bitrate savings at equal measured quality. No across-clip significance is inferred.", fontsize=8)
    return save_figure(fig, out_dir, "bdrate_summary"), values


def generate_plots(csv_path, out_dir=None, manifest_path=None, clip_name=None, frames=None):
    """Generate supported plots; record skipped figures without losing metric exports."""
    csv_path = Path(csv_path).resolve()
    results_dir = csv_path.parent
    out_dir = Path(out_dir).resolve() if out_dir else results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_csv(csv_path)
    if clip_name:
        _, rows = filter_clip(rows, clip_name)
    if not rows:
        raise ValueError("No evaluation rows to plot")
    candidate = results_dir.parent / "manifest.json"
    manifest_path = Path(manifest_path).resolve() if manifest_path else candidate if candidate.exists() else None
    manifest = json.loads(manifest_path.read_text()) if manifest_path else {}
    configs = {c["name"]: c for c in manifest.get("clips", [])}
    manifest_dir = manifest_path.parent if manifest_path else results_dir
    index = {"results_csv": str(csv_path), "results_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
             "manifest": str(manifest_path) if manifest_path else None,
             "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path else None,
             "artifacts": [], "skipped": []}

    def record(kind, clip, output, **metadata):
        index["artifacts"].append({"kind": kind, "clip": clip, "files": output, **metadata})

    def attempt(kind, clip, function):
        existing_figures = set(plt.get_fignums())
        try:
            function()
        except (OSError, ValueError, KeyError, TypeError, OverflowError, cv2.error) as exc:
            note = f"{kind} for {clip}: {exc}"
            index["skipped"].append(note)
            log.warning("Skipped plot: %s", note)
        finally:
            for figure in set(plt.get_fignums()) - existing_figures:
                plt.close(figure)

    for clip in sorted({r["clip"] for r in rows}):
        _, clip_rows = filter_clip(rows, clip)
        if len({r["method"] for r in clip_rows}) > 2:
            attempt("rd_all_methods", clip, lambda: record("rd_all_methods", clip, plot_all_methods(clip, clip_rows, out_dir)))
        u, g = split_methods(clip_rows)
        config = configs.get(clip, {})
        def rd_panels():
            files, details = plot_rd_panels(clip, clip_rows, out_dir)
            record("rd_panels", clip, files, metrics=details)
        attempt("rd_panels", clip, rd_panels)
        for metric in ("vmaf_global", "psnr_roi", "ssim_roi", "psnr_periphery", "ms_ssim_global"):
            def single(metric=metric):
                files, detail = plot_bdrate(clip, *extract_rd(u, metric), *extract_rd(g, metric), out_dir, metric)
                record("rd_single", clip, files, **detail)
            attempt("rd_single", clip, single)
        def runtime():
            record("runtime", clip, plot_runtime(clip, clip_rows, out_dir, config.get("hardware", manifest.get("hardware"))))
        attempt("runtime", clip, runtime)
        try:
            pair_u, pair_g, pair = select_bitrate_pair(clip_rows)
        except ValueError as exc:
            index["skipped"].append(f"Visual and temporal comparison for {clip}: {exc}")
            continue
        def temporal():
            files, metrics = plot_temporal(clip, pair_u, pair_g, pair, results_dir, out_dir, config)
            record("temporal_quality", clip, files, pair=pair, metrics=metrics)
        attempt("temporal_quality", clip, temporal)
        selected_frames = frames if frames is not None else config.get("plot_frames", [int(number(pair_u["frames_evaluated"])) // 2])
        for frame_index in selected_frames:
            def spatial(frame_index=frame_index):
                if not isinstance(frame_index, int) or isinstance(frame_index, bool):
                    raise ValueError("Plot frames must be integer zero-based indices")
                a, c, crops = plot_spatial(clip, pair_u, pair_g, pair, config, manifest_dir,
                                         number(manifest.get("roi_percentile", 25)), frame_index, out_dir)
                record("spatial_allocation", clip, a, pair=pair, frame=frame_index,
                       saliency_source="Reconstructed from canonical QP offsets, not raw model heatmaps")
                record("visual_comparison", clip, c, pair=pair, frame=frame_index, crops_yxhw=crops)
            attempt("spatial_comparison", clip, spatial)
    def summary():
        files, details = plot_summary(rows, out_dir)
        record("bdrate_summary", None, files, metrics=details)
    attempt("bdrate_summary", "all clips", summary)
    (out_dir / "plots.json").write_text(json.dumps(index, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Compression figures", "", f"Results CSV SHA-256: `{index['results_sha256']}`", "",
             "Only the artifacts listed below belong to this result snapshot. Older unlisted files may be stale.", "",
             "ROI means model-predicted ROI, not measured human gaze. Pair captions show actual rates; images are not interpolated.", ""]
    for artifact in index["artifacts"]:
        lines.append(f"- {artifact['kind']} ({artifact['clip'] or 'all clips'}): " +
                     ", ".join(f"[{name}]({name})" for name in artifact["files"]))
    if index["skipped"]:
        lines += ["", "## Skipped figures", ""] + ["- " + note for note in index["skipped"]]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved %d figure sets to %s (%d skipped)", len(index["artifacts"]), out_dir, len(index["skipped"]))
    return index


def default_csv():
    return OUTPUT_FILE_DIR / "compression_evaluation" / "results_detailed.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=default_csv())
    parser.add_argument("--clip", help="One clip; --all otherwise includes every clip")
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--metric", choices=[attr for _, attr in SUMMARY_METRICS], default="vmaf_global")
    parser.add_argument("--all", action="store_true", help="Generate the full research figure suite")
    parser.add_argument("--manifest", type=Path, help="Required for spatial figures; auto-discover beside eval_results")
    parser.add_argument("--frames", type=int, nargs="+", help="Zero-based example frames; default: clip midpoint")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.all:
            generate_plots(args.csv, args.out_dir, args.manifest, args.clip, args.frames)
        else:
            clip, rows = filter_clip(load_csv(args.csv), args.clip)
            u, g = split_methods(rows)
            files, detail = plot_bdrate(clip, *extract_rd(u, args.metric), *extract_rd(g, args.metric),
                                      args.out_dir or args.csv.parent, args.metric)
            print(detail)
            print("Saved: " + ", ".join(files))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
