# DIEM preparation and autoregressive fine-tuning

This guide describes the current entry points and defaults in `config.py`,
`dataset.py` and `fine_tune.py`. Run commands from the repository root using
[the locked Python environment](../ENVIRONMENT.md).

## Inventory and validation scope

At this documentation refresh (2026-10-15), the default raw root contains
**82 video folders**, each with `event_data/*.txt`. Their identifiers match
`DIEM/splits.json`: **58 train, 12 validation and 12 test**, split seed 0.
These checks establish folder inventory and split membership only. They do
not establish that every clip has been decoded, calibrated, aligned, trained
on or evaluated successfully.

Earlier README notes described checks on six clips and reported drop rates
and a possible calibration pattern in two 576-pixel-high videos. Those are
historical observations, not a validation record for the present 82-video
cohort. Assess current videos using their own preparation records,
calibration reports, overlays and run exports.

The frozen split is an experiment assignment, not reconstructed training
history for existing checkpoint weights. A raw folder, a split entry or an
existing checkpoint alone is not evidence of held-out model performance.

## Paths and configuration

Paths resolve from the project location and environment settings, independently
of the working directory. The defaults require no path edits in `config.py`.
The table uses paths relative to the project root:

| Purpose | Default | Override |
|---|---|---|
| Raw subject annotations | `DIEM/DIEM_ROOT/<video_name>/event_data/*.txt` | `DIEM_ROOT` |
| Stimulus videos | `DIEM/DIEM_ROOT/<video_name>/video/<filename>.<extension>` | `DIEM_ROOT` |
| Prepared annotations, frames and calibration report | `output file/diem/` | `DIEM_OUTPUT_ROOT`, or `GAZELNN_OUTPUT_DIR` with `/diem` appended |
| Persisted split | `DIEM/splits.json` | `DIEM_SPLIT_MANIFEST`; training also accepts `--split-manifest` |
| Initial trained weights | `checkpoints/OSIE_model.pt` | `--pretrained-checkpoint`, or `--from-scratch` |
| Fine-tuned weights | `DIEM/DIEM_model.pt` | `--checkpoint` |
| Training reports | `output file/diem_training/<checkpoint stem>/` | `GAZELNN_OUTPUT_DIR`, or `--output-dir` |

Each raw annotation folder must contain exactly one supported stimulus in
its `video/` subfolder. The stimulus may have any filename; the parent
annotation folder identifies the clip. Supported extensions include MP4,
MKV, AVI, MOV, WebM, M4V, MPG, MPEG, WMV and Y4M, case insensitive.
DIEM preparation, calibration, split checks and training source validation
use only these dataset-local videos. `input video/` and `GAZELNN_INPUT_DIR`
do not select DIEM sources. `VIDEO_FILE_GLOB` is retained as a legacy MP4
pattern; `video_files()` uses the supported-extension list.

Training `--processed-root` selects prepared data, and `--frames-root`
defaults to its `frames/` subfolder. These flags do not redirect preparation
scripts: set the environment paths before running those scripts when using
another prepared root. An existing `DIEM/DIEM_Processed/` folder is a legacy
location, not the current default. Its files must meet the current integrity
contracts before they can be used for training.

## Prepare, inspect, then train

Extract the raw annotations and stimulus videos into the configured raw root.
The original DIEM archive layout uses the existing defaults:

```text
DIEM/DIEM_ROOT/<video_name>/event_data/<subject>.txt
DIEM/DIEM_ROOT/<video_name>/video/<video_name>.mp4
```

Run the preparation stages in order:

```bash
uv run --locked python DIEM/parse_gaze.py
uv run --locked python DIEM/build_heatmaps.py
uv run --locked python DIEM/extract_frames.py
```

Parsing and frame extraction enumerate raw video folders. Heatmap generation
enumerates folders containing parsed fixations. The stages write under the
configured prepared root:

```text
output file/diem/
  fixations/<video_name>/<subject>.json
  heatmaps/<video_name>.npz
  frames/<video_name>/frame_000001.png
  frames/<video_name>/frames_metadata.json
  calibration_report.csv
```

Review `calibration_report.csv` and inspect overlays after frames have been
extracted. For example, this identifier exists in the current cohort:

```bash
uv run --locked python DIEM/inspect_calibration.py 50_people_brooklyn_no_voices_1280x720 --frames 5
```

The inspector also accepts `--videos` with comma-separated identifiers. It
writes overlays under the system temporary directory's
`calibration_debug/<video_name>/` (normally `/tmp/calibration_debug/` on Linux),
not under the persistent prepared root. Green/red points indicate whether
calibrated coordinates fall inside the source image; they do not establish
that the coordinates or timestamps are correct. The inspector can display
an empty background when a sampled PNG is missing, so overlays do not
replace the training loader's integrity checks.

Check the current cohort against the persisted split, then start the actual
training entry point:

```bash
uv run --locked python -m DIEM.splits
uv run --locked python -m DIEM.fine_tune
```

Split verification requires the declared videos' raw gaze files and their
DIEM-local stimulus videos. It does not decode all frames or validate calibration.
Preparation and training errors must be resolved for the selected cohort;
training rejects missing frame folders and mismatched prepared memberships
rather than silently reducing the cohort.

### Re-running preparation

The three preparation scripts accept `--videos` with comma-separated names
and `--skip-existing`. Their reuse checks differ:

- Parsing skips a video when its output folder already contains JSON files;
  that check does not prove completeness or unchanged raw annotations.
- Heatmap reuse checks recorded source/frame mapping and source identity;
  it does not regenerate targets after calibration or raw annotation changes.
- Frame reuse checks source/preprocessing provenance, the exact PNG inventory,
  image hashes, RGB dimensions and decoding.

Omit `--skip-existing` when regenerating affected data. After a calibration
change, rebuild the affected targets and review their overlays again:

```bash
uv run --locked python DIEM/build_heatmaps.py --videos 50_people_brooklyn_no_voices_1280x720
```

The report merges the rebuilt video's row with existing rows. An old row's
presence is not evidence that it was regenerated for the current settings.
After raw annotation changes, reparse first and then rebuild targets.

### Convenience wrapper

`uv run --locked python DIEM/run_diem_pipeline.py` invokes parsing, heatmap
building and frame extraction, then inspects videos whose report rows have
`needs_review == "no"`, and finally invokes fine-tuning with its defaults.
It does not pause for human review, inspect the flagged set automatically,
or forward custom training flags. If no videos are selected for inspection,
the inspector receives no video argument and exits; subprocess failures
stop the wrapper. Use the explicit stages above when reviewing calibration
or choosing experiment settings.

## Calibration policy and its limits

The global default is `identity`. Current per-video overrides select
`auto_recenter` for `ami_ib4010_closeup_720x576` and
`news_sherry_drinking_mice_768x576`. These configured choices came from
historical observations; they do not certify those clips or other videos
of the same resolution.

`auto_recenter` translates the pooled fixation median to the source image
center. `auto_percentile` stretches pooled coordinate percentiles to the
image bounds. Both derive a transform from the observed gaze distribution;
a lower out-of-frame rate alone cannot establish a correct calibration.
`manual` uses explicitly configured scale/offset values. Native source
resolution is read with OpenCV; a resolution suffix in a folder name is
not the current measurement method.

The builder records mode, native dimensions, dropped-fixation fraction and a
`needs_review` flag in `calibration_report.csv`. A drop fraction above 15%
sets that flag. This is a triage threshold, not an acceptance test: inspect
flagged clips and review the remaining cohort as appropriate. Ground-truth
calibration or acquisition/display metadata is stronger evidence than visual
plausibility or a drop-rate improvement.

Per-video overrides take precedence over the global mode, including the
inspector's `--mode` setting. Read the effective mode it prints when comparing
calibration choices. Changing calibration requires rebuilding the targets;
it does not alter existing checkpoint weights.

## The default fine-tuning loop

The CLI in `fine_tune.py` builds the datasets and ordered loaders, constructs
the shared `gaze_model.GazeLLNArch`, and calls `fine_tune()`. Each training or
validation epoch creates a fresh `VideoStateStore`; `run_window()` extracts
visual features and processes each contiguous chunk with per-video hidden
state and previous heatmap feedback.

**The default is autoregressive** (`teacher_forcing=False`): the previous
model prediction, detached from gradients, feeds the next step. Each video
starts with a centered Gaussian heatmap prior (sigma 1.5 grid pixels) and
zero hidden state. Ground-truth heatmaps supply the masked continuous KL
loss. `--teacher-forcing` is an explicit alternative: valid ground truth
feeds the next step, with predicted feedback for invalid labels. The separate
`get_prev_hmaps()` helper is not part of the default fine-tuning path.

| Setting | CLI default |
|---|---|
| Chunk length / stride | 24 / 24 real frames |
| Videos per batch / workers | 8 / 4 |
| Epoch limit / patience | 100 / 20 |
| Adam backbone / head learning rate | `1e-6` / `1e-4` |
| Backbone parameters | Frozen; `--train-backbone` opts in |
| Gradient clip norm | 1.0 |
| Teacher forcing | Disabled |
| AMP | Requested; enabled on CUDA, disabled on CPU |
| Device / seed | Automatic CUDA-or-CPU selection / 42 |
| Input preprocessing | `imagenet` |

The frozen-backbone path extracts features without backbone gradients;
training mode can still update BatchNorm running statistics. An optional
`--unfreeze-backbone-after-epoch N` delays the configured backbone training
choice. The CLI seeds Python, NumPy and Torch before data/model construction.

```bash
uv run --locked python -m DIEM.fine_tune --help
uv run --locked python -m DIEM.fine_tune --epochs 100 --videos-per-batch 8 --window-len 24
```

`--processed-root`, `--frames-root`, `--split-manifest`, `--checkpoint`,
`--pretrained-checkpoint` and `--output-dir` select run paths. `--from-scratch`
uses fresh model initialization; adding `--no-pretrained-backbone` also avoids
ImageNet initialization/download. `--no-amp` disables AMP explicitly. Loading
trained weights restores weights rather than optimizer, epoch or RNG state.
The direct Python `fine_tune()` API remains available for custom callers;
use `run_window()` to handle features, state and feedback in a custom loop.

## Dataset batches and causal recurrent chunks

`build_datasets()` discovers prepared targets, checks their source mapping,
and reuses exact video memberships from the selected split manifest. A
`VideoOrderedBatchSampler` retains chunk order within each video while the
training loader shuffles video order; validation uses fixed video order.

Each item contains `frames (T,3,H,W)`, `heatmaps (T,Hg,Wg)`, gaze-label
`valid (T,)`, `delta_t (T,)`, `video_name`, zero-based `start_frame`, actual
`sequence_length` and `is_video_start`. The model's `forward()` consumes
visual features plus feedback/state; `run_window()` performs that conversion
and temporal handling for these batches.

For 50 real frames, the default cursor ranges are `[0,24)`, `[24,48)` and
`[48,50)`. The final chunk is padded for batching. Padding has no features,
recurrent updates, elapsed time or loss. An unannotated real frame has
`valid=False` and still advances video time and state.

Both hidden state and previous heatmap detach at chunk boundaries for
truncated BPTT. The next chunk must start at the stored `next_frame` cursor;
overlap, gaps, reordered continuation, replay and duplicate video names in
a batch are rejected. Training and validation reset state each epoch.
Continuous KL averages valid frames, and epoch losses are weighted by
valid-frame count. The recorded temporal policy is
`contiguous_chunks_with_detached_state_v1`; older overlapping-window losses
are not directly comparable to this policy.

## Frame integrity and annotation mapping

Training excludes annotation index zero. Recurrent cursor zero reads
`frame_000001.png` and target index 1. `VideoEntry.num_frames` counts real
source frames; annotation slot zero is empty and invalid.

Heatmap generation uses the source frame count and preserves unannotated
leading/trailing frames with invalid gaze masks. Zero, negative, reversed
or beyond-source annotation intervals are rejected. Target NPZs store
`frame_numbers=[-1,1,...,N]`, `alignment_json`, source hash/count and nominal
FPS alongside the heatmaps and validity mask.

Extraction records source SHA-256, an exact 1-based PNG inventory, per-image
hashes, FPS, dimensions and resize provenance. Training verifies every PNG's
hash and decoding before use and rechecks image/target hashes when reading.
Missing, extra, corrupt or changed images, unavailable/changed sources and
source/count/FPS disagreements fail explicitly. Real missing images are not
replaced by black pixels. Only batch tails use black RGB padding before
normalization. These checks add startup and read cost.

Older targets and frame caches require regeneration without `--skip-existing`:

```bash
uv run --locked python DIEM/build_heatmaps.py
uv run --locked python DIEM/extract_frames.py
```

Extraction stages the new files before replacing the cache and removes
leftover frame PNGs from longer previous extractions. These preparation
commands replace prepared artifacts and preserve existing model weights.

The configured annotation rate is 30 Hz, and elapsed time is `1/FPS` seconds
for every real frame. The elapsed-time convention assumes constant frame rate; variable-rate
timestamp alignment is not implemented. FPS, source identity, frame counts
and nominal frame numbers are structural checks. External gaze-clock synchronization remains
unverified; independent timestamp evidence or synchronization review is
needed to establish it. The gaze evaluator reports source/frame mapping
verification separately from timestamp alignment.

## Explicit training and inference preprocessing

Defaults are RGB uint8, shared Pillow bilinear resize to 256x384, float32
CHW scaling to `[0,1]`, then ImageNet channel means
`[0.485,0.456,0.406]` and standard deviations `[0.229,0.224,0.225]` exactly
once. Extracted PNGs are unnormalized RGB bytes. The backbone consumes the
normalized dataset tensors. Default heatmaps are 32x48 (`S=8`), with Gaussian
sigma 1.5 in heatmap-grid pixels and validity threshold
`MIN_SUBJECTS_PER_FRAME=3`.

`--preprocessing legacy_rgb01` selects the previous unnormalized convention
for an explicit reproduction experiment. Custom dataset `transform` callables
receive resized `[0,1]` TCHW tensors and replace normalization; provenance
labels their contract custom. Training/validation policies and model input
sizes must agree. Shape compatibility alone does not establish historical
checkpoint preprocessing.

Checkpoint declarations select preprocessing for encoder/gaze inference;
historical weights without declarations retain unknown training history.
Changing inference preprocessing does not retrain weights. A new consistent
experiment requires matching preparation, training, inference and regenerated
QP maps. DeepGaze's separate model-specific preprocessing is documented in
[Baselines/README.md](../Baselines/README.md).

## Frozen splits, checkpoints and run exports

`DIEM/splits.json` freezes the present 82 identifiers at 58/12/12, seed 0.
For a smaller prepared cohort, create a separately named manifest with
`DIEM.splits --output` and `--videos`, listing at least three available
identifiers. Select that file with `--split-manifest` or
`DIEM_SPLIT_MANIFEST`.
Prepared data must match that complete cohort. `--extend` appends new raw
videos to train; `--overwrite` deliberately replaces assignments and
invalidates prior checkpoint provenance. Folder counts are not a substitute
for these membership checks.

The shared architecture is `gaze_model.GazeLLNArch` with input and output
CoordConv; `DIEM_Modeldata.py` provides compatibility imports. A best-model
save writes weights plus `<checkpoint>.training.json` and
`<checkpoint stem>.history.json`. The training record binds weights to the
split, architecture/version, initialization, preprocessing, hyperparameters,
source/environment fingerprints, completed epoch history and best epoch.
Historical weights receive no invented record. See
[checkpoint provenance](../checkpoints/README.md#training-provenance).

`run_config.json` is saved before loading training data/models, with status
updates for completion, failure or interruption. The report directory also
receives `results.json` and `history.csv` after a successful run. These bind
the run ID to the checkpoint/hash, full loss history and selection. Use
separate checkpoint/report paths to retain experiments. See
[run exports](../README.md#run-settings-and-result-exports).

Gaze evaluation's split and checkpoint provenance checks support held-out
claims; `--allow-unverified-checkpoint`, `--split train`/`val` and
`--no-split-check` are diagnostic alternatives. Membership verification,
source/frame integrity, calibration quality and timestamp synchronization
are distinct pieces of evidence.

## Checking code behavior

```bash
uv run --locked python -m unittest discover -s tests -v
```

The regression suite covers synthetic/temporary data, including chunk
continuation, padding, integrity failures and result exports. Passing it is
not a substitute for checking the current 82 real videos and recording their
calibration, alignment and evaluation outcomes. This documentation refresh
checked inventory, persisted membership and CLI definitions; it did not run
full-cohort preparation, training or scientific validation.
