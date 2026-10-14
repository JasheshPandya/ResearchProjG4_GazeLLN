# Compression evaluation

See [platform setup](README.md) for Windows, Linux and macOS. Run from the project directory, using Python with the project dependencies installed:

```bash
python Encoder/evaluate_compression.py --manifest Encoder/ablation_out/manifest.json --out_dir Encoder/ablation_out/eval_results
python Encoder/plot_bdrate.py --all --csv Encoder/ablation_out/eval_results/results_detailed.csv --manifest Encoder/ablation_out/manifest.json
python Encoder/plot_bdrate.py --metric psnr_roi
python Encoder/plot_bdrate.py --metric ssim_roi
python Encoder/plot_bdrate.py --metric ms_ssim_global
python Encoder/plot_bdrate.py --metric vmaf_global
```

Evaluation now generates research figures automatically in `out_dir/plots/`, as both PNG and PDF. The suite includes:

- A three-panel RD figure for full-frame VMAF, predicted-ROI PSNR, and peripheral PSNR, plus individual metric plots.
- A per-clip BD-rate summary, preserving individual clips rather than treating CRF points as independent experiments.
- A temporal quality plot using aligned per-frame VMAF, ROI PSNR and ROI SSIM when available.
- An offline runtime breakdown of measured encoding and QP generation cost; missing stage timings are marked unknown.
- A spatial allocation example with source, reconstructed saliency, QP offsets, the shared ROI mask, and baseline/gaze error maps using the same color scale.
- A source/baseline/gaze comparison with identical ROI and peripheral crops.

The closest available baseline/gaze bitrate pair inside their common rate interval is selected for temporal and spatial examples. Captions record both CRF labels, actual rates, and the relative rate gap. They do **not** claim equal bitrate or interpolate reconstructed images. Example frames default to the clip midpoint, independently of distortion scores. Override with `--plot-frames 120 240` on evaluation, `--frames 120 240` on standalone plotting, or a clip-level `plot_frames` list in the manifest. Optional clip-level `scene_cut_frames` annotate supplied zero-based scene boundaries; they are not automatically detected. Optional clip-level or manifest-level `hardware` text/dictionary is displayed on the runtime figure.

`plots/README.md` lists the current artifacts and skips; `plots/plots.json` records input CSV/manifest checksums, BD-rate values, selected bitrate pairs and crop coordinates. Only listed artifacts belong to that snapshot; older unlisted files can be stale. Spatial figures require accessible source/variant videos and canonical QP maps. Unavailable inputs skip their figures without losing the completed metrics. RD panels mark unavailable BD-rate explicitly and do not fit invalid curves.

Compact per-frame scores are exported to `per_frame/*.csv`, referenced by the new `frame_metrics_path` column and JSON field, rather than adding large frame arrays to `results.json`. Older runs can use saved VMAF logs only when frame indices, counts and mean agree with the current evaluation. If those logs are missing, complete saved 20-frame CSV averages can produce a clearly labelled **20-frame mean** temporal plot; they are never presented as per-frame scores.

Use `--no-plots` to evaluate without generating figures. `plot_bdrate.py --all` regenerates figures from saved results without repeating metric computation or encoding. The existing `--metric` mode remains available for one RD figure. Plotting uses a headless backend on Windows, Linux and macOS.

FFmpeg and FFprobe must be on PATH, or supplied with `--ffmpeg` and `--ffprobe` to the evaluator. VMAF needs an FFmpeg build with `libvmaf`; other metrics remain available without that filter. The default model is explicitly `vmaf_v0.6.1`. Set `vmaf_model_path` in the manifest for a custom JSON model. Relative paths are resolved from the manifest directory.

The evaluator writes `results_detailed.csv`, `summary_bdrate.csv`, `results.json`, `report.md`, `run.log`, and VMAF JSON logs. Existing result column names are retained; new columns include:

| Columns | Meaning |
|---|---|
| `mse_global`, `mse_roi`, `mse_periphery` | Squared error pooled over all scored pixels; lower is better. |
| `ms_ssim_global` | Five-scale luma MS-SSIM; higher is better. |
| `psnr_p5`, `ssim_p5`, `ms_ssim_p5`, `vmaf_p5` | Fifth percentile of individual frame quality scores; higher is better. |
| `vmaf_min`, `vmaf_frames`, `vmaf_model` | Worst VMAF frame, frame count, and model identity. |
| `roi_fraction` | Actual fraction of pixels covered by the shared predicted-gaze mask. |
| `bitrate_source`, `bits_per_pixel_frame` | Video-stream bitrate when available, otherwise container bitrate; normalized data cost per pixel per frame. |
| `encode_fps`, `pipeline_fps`, `pipeline_processing_ms_per_frame` | Processing throughput and average processing cost when the required timings exist. |

Pixel metrics use 8-bit luma decoded through OpenCV. They do not measure chroma fidelity. SSIM uses full-frame 11x11 Gaussian maps with reflected borders. MS-SSIM uses valid 11x11 windows, sigma 1.5, Wang's five weights, 2x2 average downsampling, and reflected padding for odd dimensions; it returns unavailable when a scale is smaller than 11 pixels. It is global only; single-scale SSIM provides the ROI/periphery comparisons.

**PSNR now pools squared error before converting to dB.** Previous results averaged per-frame dB, so rerun both methods before comparing against the new results. Identical luma produces infinite PSNR. In strict JSON, unavailable scores are `null` and infinite scores are the string `"Infinity"`.

All variants use the clip's canonical `roi_source`; variant `qp_dir` entries describe encoding inputs and cannot override the evaluation mask. ROI selects the highest-ranked macroblocks, rounding its count up and resolving ties deterministically. Partial edge macroblocks cover their actual 16x16 pixel positions. This is a predicted-gaze evaluation, not independent human-gaze validation.

Videos must have matching frame counts, frame rates, and dimensions. The evaluator rejects mismatches instead of silently resizing or truncating. Inputs should also have matching content and aligned timestamps; matching metadata alone cannot establish content alignment.

BD-rate and BD-quality are reported for global PSNR, SSIM, MS-SSIM, VMAF, ROI/periphery PSNR and SSIM, and fifth-percentile VMAF. Negative BD-rate means bitrate savings at equal quality over the overlapping quality range. At least two distinct points per method are needed; four or more are recommended. Missing, non-finite, duplicate or nonmonotonic RD points are reported as unavailable. The polynomial fit uses up to degree three. Compare equal bitrate or equal quality, rather than equal CRF.

`run_ablation_sweep.py` records `encode_time_s`, `qp_generation_time_s`, and `pipeline_time_s` for future runs. QP generation includes inference, preprocessing, smoothing and file writing. A standalone gaze encode is charged the full generation cost even when the sweep reuses cached maps. Existing manifests without that timing keep pipeline throughput unavailable. These measurements describe an offline sequential pipeline; they do not measure capture-to-display latency.

Definitions: [SSIM and MS-SSIM authors' resources](https://ece.uwaterloo.ca/~z70wang/research/ssim/), [FFmpeg PSNR and VMAF documentation](https://ffmpeg.org/ffmpeg-filters.html), [BD-rate tutorial](https://arxiv.org/abs/2401.04039).
