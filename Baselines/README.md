# Baseline comparisons

The project compares gaze prediction and compression allocation using shared map definitions and one compression/evaluation pipeline.

| Method | Definition | Prediction | Compression |
|---|---|---|---|
| `uniform` | Constant probability map; zero external QP offsets | Yes | Yes |
| `center` | Center Gaussian; sigma 0.2 times each grid dimension | Yes | Yes |
| `motion` | Absolute difference between current and previous resized grayscale frames | Yes | Yes |
| `static_saliency` | Trained DeepGaze I prediction computed independently for every frame | Yes | Yes |
| `gaze` | Recurrent GazeLNN with a trained DIEM checkpoint | Yes | Yes |
| `gaze_lag` | Previous model prediction, uniform initial map | Yes | — |
| `gaze_static` | Repeat the first model prediction | Yes | — |
| `gaze_shifted` | Seeded nonzero circular shift of model predictions | Offline diagnostic | — |

`static_saliency` implements the paper's frame-independent saliency baseline. `gaze_static` is a different diagnostic: it freezes one prediction for the whole video. The shifted diagnostic can use future predictions and is not an online predictor.

## Trained static saliency

The adapter uses the authors' [DeepGaze implementation](https://github.com/matthias-k/DeepGaze), pinned to release v1.1.0, commit `c33b89f08016e41e68cec4e4d9f1a73a14211386`. The complete trained [DeepGaze I weights](https://github.com/matthias-k/DeepGaze/releases/download/v1.01/deepgaze1.pth) are downloaded once to `checkpoints/deepgaze1.pth`, with a SHA-256/provenance sidecar. These are the authors' PyTorch-trained variant, not a claim of byte-for-byte reproduction of the original paper's training implementation. Upstream describes MIT1003 as its training data; no DIEM adaptation is performed here.

```bash
python -m pip install -r requirements-static-saliency.txt
python -m Baselines.download_static_weights
```

The weights have been downloaded in this checkout. Both downloader and runtime verify the pinned checksum; inference performs no network fetch and fails if trained weights or the pinned package are absent. The large weight file is excluded from normal Git tracking; its retrieval script and provenance are retained.

Every frame is resized to 384×256, passed as RGB in 0..255 through the model's own normalizer, and converted from log density to probability. The center prior is explicitly uniform. The probability map is reduced to the same 32×48 grid used by GazeLNN, then receives the selected spatial smoothing. There is no recurrent state or temporal EMA in this baseline; selecting it requires `--temporal-alpha 1`.

## Compression

```bash
python Encoder/run_ablation_sweep.py \
  --input-dir "input video" \
  --output-dir "output file" \
  --checkpoint DIEM/DIEM_model.pt \
  --methods uniform gaze center motion static_saliency \
  --crfs 18 21 24 27 30 33

python Encoder/evaluate_compression.py \
  --manifest "output file/manifest.json" \
  --out_dir "output file/compression_evaluation"
```

For the paper's three baselines plus the proposed method, omit `motion`. `--static-checkpoint PATH` selects the same pinned weights at another location. The original default sweep methods remain `uniform gaze`; additional methods must be selected explicitly.

All methods use the same native encoder, CRF points, preset and scoring ROI. Nonuniform controls use the same heatmap grid, bilinear macroblock upsampling, min/max normalization and QP limits. Flat controls use zero offsets. `uniform` means no external saliency offsets; the codec retains its shared internal behavior. All variants receive one shared scoring mask, which is a predicted ROI unless independently provided through `--roi-source`. That option is required when omitting gaze.

The summary reports all nonuniform methods against uniform, plus gaze against each additional baseline. Negative BD-rate means the test method needs fewer bits at equal quality within overlap. Equal CRF alone does not establish equal bitrate. The plot suite has an all-method RD overview; richer spatial/temporal examples remain focused on uniform versus gaze.

Map generation time includes model loading, inference, decoding, smoothing and writing. A standalone nonuniform encode is charged the full measured generation stage even when a sweep shares maps across CRFs. Cached timing remains unavailable.

## One video time convention

`video_timing.py` defines `elapsed_seconds_per_frame_v1`: positive `dt = 1 / FPS` seconds for each step of an aligned constant-frame-rate sequence, including its first processed frame. DIEM training, compression inference and gaze evaluation now use it. The low-level recurrent inference function requires an explicit time step; there is no default `dt=1`.

DIEM uses its annotated 30 Hz rate. Available source-video FPS is checked against annotation FPS before training; gaze evaluation checks its video against NPZ `fps`, an explicit `--target-fps`, or the DIEM default for older targets. Align videos/annotations before evaluating mismatched rates. Variable-frame-rate timestamp handling is not implemented by this convention.

QP metadata and gaze results record policy/FPS/seconds. Gaze map reuse rejects legacy metadata and old time conventions. Regenerate maps without `--reuse-qp` before comparing new results; earlier `dt=1` encodes are not interchangeable with elapsed-time encodes. Saved prediction NPZ files also carry time provenance.

## Persisted video splits

```bash
python -m DIEM.splits
```

`DIEM/splits.json` currently records **58 train / 12 validation / 12 test** video names from the 82 available raw clips, seed 0. It defines a new experiment cohort; it does not reconstruct the training history of the existing DIEM checkpoint.

Training uses the recorded lists instead of recomputing membership. Reordering discovery does not change assignments. Added/missing prepared videos cause an error, rather than a new split. Prepare the full recorded cohort, or choose a separately named manifest and matching processed-data cohort. `DIEM_SPLIT_MANIFEST` overrides the default manifest path, and `DIEM_OUTPUT_ROOT` selects the processed root.

```bash
# Define a smaller, explicit cohort of at least three available videos.
python -m DIEM.splits --output DIEM/my_splits.json --videos VIDEO_A VIDEO_B VIDEO_C

# Append new raw videos to train, leaving every existing membership unchanged.
python -m DIEM.splits --extend
```

`--overwrite` explicitly regenerates assignments and invalidates previous checkpoint provenance. Use it only for a deliberate new experiment. `--seed`, `--val-fraction` and `--test-fraction` set initial creation parameters. Persisted parameters are read back by training.

Each best DIEM checkpoint now gets `<checkpoint>.training.json`, binding its exact weight hash to the manifest hash, full video lists, elapsed-time policy and run settings. The loader memberships are checked before optimization and checkpoint recording. Existing historical weights receive no invented training record.

## Gaze prediction evaluation

The evaluator compares heatmap distributions using KL(GT‖prediction), histogram intersection, spatial CC and centroid distance in grid pixels. It masks invalid targets identically for every method. Uniform-map CC is undefined and exported as null with a count. These are not individual fixation/scanpath scores.

```bash
python -m Baselines.evaluate_gaze \
  --targets "output file/diem/heatmaps/HELD_OUT_VIDEO.npz" \
  --target-start-index 1 \
  --video "input video/HELD_OUT_VIDEO.mp4" \
  --checkpoint DIEM/DIEM_model.pt \
  --methods uniform center motion static_saliency \
  --split-manifest DIEM/splits.json \
  --out-dir "output file/gaze_evaluation"
```

The default partition is `test`. The target stem (or explicit `--video-name`) must belong to that partition; a supplied source video's basename must agree. Held-out GazeLNN evaluation requires matching checkpoint training provenance. Saved model predictions require corresponding embedded provenance. Unknown training history, edited weights, changed manifests and incompatible timing are rejected by default.

Use `--split train` or `--split val` for explicitly named diagnostic partitions. `--allow-unverified-checkpoint` allows historical weights/predictions with missing provenance and labels results unverified. `--no-split-check` is an explicit diagnostic bypass, with no held-out claim. Membership/provenance status and manifest hash are recorded in the report/JSON. Membership checks establish consistency with the recorded DIEM experiment, not independent auditing of pretraining data.

The existing `DIEM_model.pt` predates split recording, so a strict held-out claim for it needs documented training history or retraining with the frozen cohort. The earlier news-video example belongs to train under the new manifest and must not be presented as a test result.

DIEM target arrays leave entry zero unused; `--target-start-index 1` aligns video frame zero with annotation frame one. Use `--frames N` for an explicit prefix. Counts, grids and rates are checked; exact content/timestamp alignment is still the caller's responsibility. `--predictions PATH` replaces checkpoint inference with saved maps; model inference writes `predictions.npz` for reuse.

Reports contain per-frame scores, summaries, input hashes, selected interval, time policy and split verification. Dataset runs pool video scores with equal weights, video-bootstrap confidence intervals and paired sign-flip tests. A single clip still has no dataset-level uncertainty estimate.

## Verification performed

Temporary checks verified frozen splits, explicit extension, membership rejection, checkpoint/prediction provenance, elapsed time, rejection of old caches and training-record output. A five-method, four-CRF synthetic check produced 20 encodes and seven pairwise comparisons through the same evaluator, with all-method plots and no skipped figures. A→B→A inference produced exactly equal predictions for both occurrences of A. Verification scripts and their synthetic outputs were temporary and removed after use; no permanent test files were added.

## Multi-video gaze evaluation

Omit `--targets` to discover all source videos in `input video/`; each needs `<video stem>.npz` in `--targets-dir` (default `output file/diem/heatmaps`). `--out-dir` defaults to `output file/gaze_evaluation`. All clips use the same selected controls, alignment defaults and split-verification policy. Each recurrent model and motion control resets at clip boundaries. Missing targets and duplicate identifiers fail rather than changing the cohort silently.

```bash
python -m Baselines.evaluate_gaze --input-dir "input video" --targets-dir "output file/diem/heatmaps" --target-start-index 1 --checkpoint DIEM/DIEM_model.pt --methods uniform center motion
```

For heterogeneous filenames, saved predictions or per-clip alignment, pass `--dataset-manifest PATH`. Paths are relative to that JSON file:

```json
{
  "clips": [
    {"name": "clip_a", "video": "../input video/clip_a.mov", "targets": "targets/a.npz", "target_start_index": 1},
    {"name": "clip_b", "video": "../input video/clip_b.mp4", "targets": "targets/b.npz", "target_start_index": 0}
  ]
}
```

Optional per-clip `predictions`, `video_name` (split identifier) and `frames` are supported. Saved predictions and a global `--checkpoint` are mutually exclusive. Equal-stem inputs need distinct target paths in a manifest. `--targets PATH` retains the single-video interface.

Dataset exports contain per-video scores, pooled estimates, finite/missing video counts, confidence intervals, paired deltas and raw/Holm-adjusted p-values. Within each video, scores average the identical valid frame set for all methods; undefined CC remains null. See `dataset_results.json` for hashes, alignment, verification status and the full statistical protocol. No dataset is labeled held out unless every clip passes membership and applicable checkpoint verification.
