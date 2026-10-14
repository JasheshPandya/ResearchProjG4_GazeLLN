# Simple baseline comparisons

The first baseline set covers video gaze heatmap prediction and compression allocation. These controls require no additional training or downloads. The existing trained DIEM checkpoint can be included for comparison.

| Method | Definition | Prediction | Compression |
|---|---|---|---|
| `uniform` | Constant probability map; zero external QP offsets | Yes | Yes |
| `center` | Center Gaussian; sigma 0.2 times each grid dimension | Yes | Yes |
| `motion` | Absolute difference between current and previous grayscale frames resized to the heatmap grid | Yes | Yes |
| `gaze` | Existing GazeLNN checkpoint | Yes | Yes |
| `gaze_lag` | Previous model prediction; uniform initialization | Yes | — |
| `gaze_static` | Repeat the model's first prediction | Yes | — |
| `gaze_shifted` | Seeded nonzero circular shift of predictions | Offline diagnostic | — |

Uniform and center use no image content. Motion uses only the current and previous frame, with a uniform map on the first frame or when no change is detected. Maps receive spatial smoothing (default sigma 1.5 grid pixels), then optional temporal EMA (default alpha 1, disabled). Center width and all smoothing settings are fixed parameters; select them on training/validation clips, not test clips.

Compression controls use the same 32×48 heatmap grid, bilinear upsampling to macroblocks, min/max normalization, and QP range as the gaze generator. A flat control map gets zero offsets instead of a blanket QP increase. The encoder's internal x264 allocation remains enabled; `uniform` means **no external saliency offsets**.

## Compression

From the repository root, using the Python environment described in `Encoder/README.md`:

```bash
python Encoder/run_ablation_sweep.py \
  --video Encoder/sample_video.mp4 \
  --out-dir Baselines/compression_results \
  --qp-dir Baselines/qp_gaze \
  --checkpoint DIEM/DIEM_model.pt \
  --methods uniform gaze center motion \
  --crfs 18 21 24 27 30 33

python Encoder/evaluate_compression.py \
  --manifest Baselines/compression_results/manifest.json \
  --out_dir Baselines/compression_results/eval_results
```

Use `--reuse-qp --qp-dir Encoder/qp_dir` only when those gaze maps belong to this exact source clip. Cached maps have unknown generation timing. `--roi-source PATH` overrides the common scoring QP maps; it is required if gaze is omitted. All variants use one shared scoring mask. Without an independently supplied ROI, this is **predicted-ROI quality**, not measured human-gaze quality.

The summary CSV identifies `anchor_method` and `test_method`. It compares center, motion and gaze against uniform, then gaze against center and motion. Negative BD-rate means the test needs fewer bits at equal quality. The existing two-method columns remain available; `n_anchor_points` and `n_test_points` apply to every comparison. The plot suite adds a measured RD overview of all methods; spatial/temporal examples and the existing summary figure continue to show uniform versus gaze.

Each standalone nonuniform encode is charged the full map generation cost, even though a sweep reuses the maps across CRFs. Smoothing, decode, inference and file writing are included in that generation cost.

## Gaze prediction

This evaluates **video heatmap distributions**, using KL(GT‖prediction), histogram intersection (similarity), spatial correlation (CC), and centroid distance in heatmap-grid pixels. It complements the existing MIT scanpath evaluator; it does not evaluate individual fixation sequences. A uniform map has undefined CC, exported as null with an explicit metric count. Invalid target frames are excluded identically for all methods.

Prepare gaze targets using the existing DIEM scripts. Their output includes an unused frame zero, so `--target-start-index 1` is necessary when matching video frame zero to DIEM annotation frame one. The source video and annotations must share their time base. Grid and selected frame counts are checked; exact timestamp alignment remains the caller's responsibility.

```bash
python -m Baselines.evaluate_gaze \
  --targets DIEM/DIEM_Processed/heatmaps/VIDEO_NAME.npz \
  --target-start-index 1 \
  --video DIEM/DIEM_ROOT/VIDEO_NAME/video/VIDEO_NAME.mp4 \
  --checkpoint DIEM/DIEM_model.pt \
  --methods uniform center motion \
  --out-dir Baselines/gaze_results
```

Use `--frames N` for an explicitly limited prefix, also when annotations end before the video. Missing selected target/video frames cause an error. Model predictions are saved as `predictions.npz`, and can be supplied later through `--predictions PATH`. Prediction arrays start at video frame zero and need to match the selected target interval; `--frames` applies the same prefix limit to them. Running without a checkpoint/prediction file evaluates only the simple controls.

The video predictor uses DIEM's elapsed-time convention, `dt = 1 / video_fps`, and resets hidden state at the start of each video. The existing compression gaze generator uses its deployment default `dt = 1`; these model outputs should not be treated as identical. Center/motion controls share their definitions across both tasks.

The shifted control can use future predictions and is only an offline diagnostic. Lag/static/shifted controls use predictions, never current or future ground-truth gaze. All outputs include per-frame scores, a summary CSV, a Markdown report, input hashes, parameters and the selected interval. For research comparisons, use video-level held-out clips and aggregate per-video results separately; this single-clip runner does not infer a train/test split or statistical significance.

## Validation performed

```bash
python -m unittest -v Baselines.test_baselines test_metrics \
  Encoder.test_portability Encoder.test_encoder \
  Encoder.test_compression_metrics Encoder.test_compression_plots
```

An end-to-end compression smoke run encoded four methods at CRFs 18/24/30/36 on a synthetic 30-frame, 256×256 clip, then produced metrics, five pairwise comparisons and all-method RD plots. Its source/maps/encodes are under `/tmp/gazelnn-baselines-smoke`; generated reports are in `Baselines/compression_smoke_results`. Temporary source files may disappear after a system restart.

`Baselines/gaze_smoke_results` contains an initial comparison on the first 300 frames of `news_video_republic_960x720`, using the local DIEM checkpoint and measured gaze targets generated under `Baselines/diem_processed`. The model improves over the simple image controls on this segment, while static/lag/shifted prediction scores are similar. This short run is a functionality check and does not establish temporal benefit or generalization.
