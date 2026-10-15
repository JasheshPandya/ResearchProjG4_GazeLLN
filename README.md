# GazeLNN video compression

The project runs entirely through Python entry points. All OSIE, DIEM, MIT and encoder model imports use **one architecture**, `gaze_model.GazeLLNArch`, with input and output CoordConv enabled. `gaze_losses.py` provides the shared scanpath KL-DTW loss; `project_paths.py` defines defaults independent of the current working directory.

The previous notebook training/evaluation code is now available as `trainGazeLNN.py` and `testMITlowres.py`. The notebook files have been removed. `OSIE_Modeldata.py` and `DIEM/DIEM_Modeldata.py` retain their public model names as imports of the shared class, rather than defining separate architectures.

## Setup

Use the pinned CPython 3.14.4 environment. `pyproject.toml` pins direct dependencies, and `uv.lock` pins transitive dependencies and artifact hashes. See [reproducible environment setup](ENVIRONMENT.md) for platform scope, optional DeepGaze, compatibility exports and update instructions. Native encoding additionally needs the dependencies described in [Encoder/README.md](Encoder/README.md).

```bash
python -m pip install uv==0.13.1
uv sync --locked
uv pip check
uv run --locked python Encoder/build.py
```

Use `uv run --locked python ...` for subsequent commands, or activate the new `.venv/`. An existing `venv/` is preserved.


## Input videos, compressed outputs and dataset evaluation

Put one or more videos, with any filenames, in **`input video/`**. Run from the repository root:

```bash
python run_dataset.py
```

All final videos go directly into the single **`output file/`** directory. For example, `input video/My holiday.mov` becomes `output file/compressed_My holiday.mp4`. Inputs with equal stems (including case variants or different extensions) receive a short filename suffix to prevent collisions. No per-video final-output folders are created. An output filename registry also protects same-stem videos introduced in later runs and existing files with unknown ownership. A successful rerun replaces that registered source's previous output atomically; failed encodes preserve the older output. Sweep variants append method and CRF to the compressed filename.

Supported input formats: MP4, MKV, AVI, MOV, WebM, M4V, MPG, MPEG, WMV and Y4M (YUV4MPEG2), with case-insensitive extensions. Discovery is deterministic and scans the input directory itself. Empty inputs fail with an actionable message. Formats still need decoder support in the installed OpenCV/FFmpeg build.

Y4M sources are decoded directly for gaze prediction, encoding and compression evaluation; no intermediate MP4 conversion is needed. Final compressed outputs remain H.264 MP4 files. Use the original Y4M as the evaluation reference and compare its file size with the compressed MP4 for storage reduction.

One runner handles dataset stages, always in pipeline order:

```bash
# Encode all input videos using the trained DIEM model.
python run_dataset.py --tasks encode

# Generate one multi-clip manifest and evaluate the entire sweep.
python run_dataset.py --tasks sweep compression --methods uniform gaze center motion --crfs 18 21 24 27 30 33

# Evaluate aligned human-gaze targets for the same input videos.
python run_dataset.py --tasks gaze --methods uniform gaze center --targets-dir "output file/diem/heatmaps" --target-start-index 1
```

Gaze evaluation requires real annotations and preserves the existing split/checkpoint verification rules. It never invents targets for arbitrary input videos. For historical weights, `--allow-unverified-checkpoint` explicitly marks provenance unverified; `--no-split-check` explicitly selects diagnostic evaluation. Put only the intended evaluation cohort in the input folder, or use the gaze evaluator's dataset manifest. Arbitrary video compression does not require gaze annotations or training.

`--input-dir PATH` and `--output-dir PATH` override the shared directories. Environment defaults are `GAZELNN_INPUT_DIR` and `GAZELNN_OUTPUT_DIR`; both resolve independently of the working directory through `project_paths.py`. Single-video options remain supported in the individual encoder tools.

All generated artifacts live under the shared output root: `_work/<video-id>/qp_gaze` for isolated caches, `manifest.json` for a multi-video sweep, `pipeline_results.json` for encode timings, `video_names.json` for persistent filename ownership, `compression_evaluation/` and `gaze_evaluation/` for reports, and `diem/` for prepared annotations/frames. Legacy generated files are preserved in their existing locations; use explicit paths or regenerate into the new shared root. DIEM preparation and training use only the stimulus in `DIEM_ROOT/<video_name>/video/`, alongside that folder's raw eye-tracking annotations. The shared `input video/` directory selects compression/evaluation inputs separately.

Both evaluation tasks export `dataset_results.json`, `dataset_aggregate.csv`, `dataset_paired.csv` and `dataset_report.md`, alongside per-video/per-frame results. Compression also exports `dataset_bdrate_aggregate.csv`. Both tasks write `dataset_per_video.csv`; compression's `results_detailed.csv` also retains each clip/method/CRF score. Each video contributes one pooled value per metric/condition, with equal weight across videos. Compression PSNR is computed from pooled pixel MSE within each clip; dataset PSNR is the mean of clip PSNR values. BD curves are fitted per video before averaging their deltas, rather than concatenating unrelated RD points.

Reports include finite-video counts, missing counts, sample standard deviations, seeded percentile-bootstrap confidence intervals, paired test-minus-anchor differences, two-sided sign-flip p-values and Holm-adjusted p-values. Frames are never treated as independent statistical samples. Uncertainty is unavailable with fewer than two finite videos. Equal-CRF differences are descriptive; BD comparisons establish equal-quality/equal-rate relationships within each clip's overlap. Tune `--bootstrap-samples`, `--confidence` and `--statistics-seed` on the runner or evaluators.

Run regression checks with `python -m unittest discover -s tests -v`.

## Checkpoints

The existing OSIE weights are retained once at **`checkpoints/OSIE_model.pt`**. The former root and `DIEM/` copies were verified identical before consolidation. DIEM fine-tuning and MIT evaluation use this canonical path by default. The trained DIEM checkpoint remains at `DIEM/DIEM_model.pt`.

Existing input/output CoordConv checkpoint names and tensors are preserved. Loading is strict: single-CoordConv legacy weights are rejected rather than partially initialized. Loading a complete checkpoint does not download MobileNet initialization weights; training from scratch uses pretrained MobileNet unless explicitly disabled.

## Train on OSIE

```bash
python trainGazeLNN.py --device auto --batch-size 8 --epochs 100
```

The default output is `checkpoints/OSIE_model.pt`, which replaces the existing checkpoint when a better validation model is saved. Use `--checkpoint PATH` to retain the existing trained weights. A companion `.history.json` records train/validation losses. Both OSIE and DIEM training also write `<checkpoint>.training.json`: hash-bound architecture/version, exact split/sample selection, hyperparameters, initialization/parent-checkpoint lineage, full epoch history, best-epoch selection and environment/source fingerprints. History is updated through the complete run, including early stopping or interruptions. See [checkpoint provenance](checkpoints/README.md#training-provenance) for the schema and compatibility rules. `--evaluate-test` evaluates the best saved model afterward. `--resume PATH` initializes weights; it does not restore optimizer or epoch state.

The converted training retains the former notebook's centered prior, per-fixation time deltas, feature caching without backbone gradients, autoregressive feedback, KL-DTW loss and early stopping. CPU is supported; the batch-size default is 8 rather than the notebook's hard-coded 256.

OSIE feature extraction freezes MobileNet parameters and keeps the backbone in evaluation mode during every training epoch, including its BatchNorm running statistics. Adam optimizes only trainable head parameters. Shared OSIE/MIT KL-DTW removes masked steps before normalization and alignment, normalizes each sequence by its valid length, and averages nonempty sequences. An entirely masked batch returns differentiable zero. New training/evaluation records identify this loss policy as `valid_steps_only_v1`. Padded-sequence losses from the previous implementation are not directly comparable; establish a new baseline. Existing checkpoint tensors and architecture remain compatible.

## Evaluate MIT

```bash
# Scanpath metrics plus KL-DTW loss, using the canonical OSIE checkpoint.
python Evaluate_OSIE_model.py --device auto

# Notebook-equivalent KL-DTW evaluation.
python testMITlowres.py --device auto

# Evaluate the DIEM-fine-tuned weights instead.
python Evaluate_OSIE_model.py --checkpoint DIEM/DIEM_model.pt
```

Both entry points accept `--data-root`, `--checkpoint`, `--batch-size`, `--num-workers` and `--limit-samples`. The default MIT input uses `manifest.csv`, `images/` and fixation arrays. The original notebook's `images/` plus `eye/fixations.mat` layout is supported by `testMITlowres.py --dataset-format mat --data-root PATH`. Its loader/helper API remains available for imports.

## Fine-tune and encode

Prepare DIEM data as described in [DIEM/README.md](DIEM/README.md), then run:

```bash
python DIEM/fine_tune.py
python run_dataset.py
```

Fine-tuning defaults to autoregressive predicted feedback with a frozen backbone and contiguous 24-frame chunks. `--teacher-forcing` explicitly selects an alternative. Initialization uses `checkpoints/OSIE_model.pt` and the default save path is `DIEM/DIEM_model.pt`. Encoder inference defaults to the latter. Configure fine-tuning with CLI flags; `python -m DIEM.fine_tune --help` lists them. For baseline sweeps and measured-gaze evaluation, see [Baselines/README.md](Baselines/README.md).


## GazeLNN image preprocessing

`gaze_preprocessing.py` defines the shared input contract for OSIE/MIT images, DIEM windows, encoder gaze inference and baseline GazeLNN evaluation: **RGB uint8 → Pillow bilinear resize to the model's `(height, width)` → float32 CHW in [0,1] → ImageNet channel normalization**. Mean is `[0.485, 0.456, 0.406]`; standard deviation is `[0.229, 0.224, 0.225]`. Default input size is 256×384. Color conversion from OpenCV BGR happens before this shared transform; normalization happens exactly once, before `extract_features`.

DIEM frame extraction now uses the same OpenCV decoder and Pillow resize as inference. It stores unnormalized RGB PNGs plus `frames_metadata.json`. The dataset applies scaling and normalization on load, including black padding. Training requires the current frame-integrity metadata and source-bound targets; regenerate legacy targets and frames with `DIEM/build_heatmaps.py` and `DIEM/extract_frames.py` (without `--skip-existing`). This changes the previous FFmpeg-resize preparation path.

New checkpoint sidecars declare `input_preprocessing`. Encoder and gaze evaluation automatically honor that declaration and reject a conflicting explicit profile. For older weights with no declaration, inference defaults to ImageNet normalization with an unknown-history warning. If you know a historical model was trained on unnormalized [0,1] inputs, select `--preprocessing legacy_rgb01` on `run_dataset.py`, encoder entry points or gaze evaluation. The override is recorded as a diagnostic choice; it does not establish missing training history. DIEM's `--preprocessing` defaults to `imagenet`; `legacy_rgb01` is an explicit reproduction option.

Existing trained weights are preserved. Changing inference preprocessing alone cannot adapt weights trained under a different convention; train/fine-tune under the shared policy for a new consistent experiment. Regenerate QP maps: reuse now checks the preprocessing specification and implementation hash in addition to source/checkpoint/settings. Gaze reports and saved prediction NPZs record the active profile and whether it matches declared training preprocessing; historical predictions with no declaration remain labeled unknown.

DeepGaze I keeps its upstream model-specific contract: RGB float32 in [0,255], OpenCV area resize and its own internal normalizer. GazeLNN's external ImageNet normalization is not applied to DeepGaze or the uniform/center/motion controls.

DIEM custom `transform` callables retain the full-tensor override API: they receive resized [0,1] TCHW frames and replace default normalization. They are recorded as custom/unverified rather than labeled ImageNet. Training and validation policies must agree. For direct custom-size models, supply matching `DIEMWindowDataset(image_size=(H, W))`.

Regression checks: `python -m unittest discover -s tests -v`. These code checks do not establish calibration or timestamp alignment for every real input video.

## Static saliency, video time and saved splits

The trained frame-independent baseline is DeepGaze I, with pinned upstream code and checksum-verified weights in `checkpoints/deepgaze1.pth`. Install its optional dependencies and fetch weights with:

```bash
uv sync --locked --extra static-saliency
uv run --locked --extra static-saliency python -m Baselines.download_static_weights
```

Select it with `--methods uniform gaze center static_saliency` in the compression sweep; all methods use the same encoder and scoring protocol. The weights are already available locally in this checkout.

Video training and inference now use elapsed seconds, `dt = 1 / FPS`, through `video_timing.py`. Old QP caches without this policy must be regenerated.

`DIEM/splits.json` freezes the current 82-video cohort at 58 train / 12 validation / 12 test. The raw folder inventory matches those identifiers; this is a membership check, not evidence of full-cohort calibration, alignment, training or evaluation. Earlier six-clip observations are historical; see [DIEM validation scope](DIEM/README.md#inventory-and-validation-scope). `python -m DIEM.splits` verifies it; `--extend` appends new videos to train, while `--overwrite` explicitly creates new assignments. Training refuses missing/unassigned prepared videos instead of silently changing membership. Each saved best DIEM checkpoint receives a `.training.json` record of its weight hash and exact split manifest.

Gaze evaluation defaults to test membership and matching training provenance. The existing historical DIEM weights predate those records; use an explicit diagnostic override or retrain with the saved splits before claiming held-out performance. Full commands and override meanings are in [Baselines/README.md](Baselines/README.md).

## Recurrent video chunk semantics

DIEM training and validation now use contiguous 24-frame chunks with stride 24, including terminal partial chunks. Each frame index is processed once. State carries only to the exact next frame, with detached hidden state and previous heatmap at chunk boundaries. Replay/overlap, skipped chunks and out-of-order state reuse are rejected. Padding advances no state or elapsed time and never enters the backbone. Real frames with missing gaze labels still advance state; their loss is masked.

Train/validation losses pool valid frames, with epoch aggregation weighted by valid-frame count. The checkpoint record identifies the corrected temporal policy and schedule. Existing checkpoints are preserved; historical overlapping-window validation losses need reevaluation under a consistent policy. See [DIEM temporal semantics](DIEM/README.md#dataset-batches-and-causal-recurrent-chunks). Regression checks: `python -m unittest discover -s tests -p 'test_recurrent_chunks.py' -v`.


DIEM training now requires complete, source-bound prepared data: real frame 1 maps to target index 1, while annotation slot zero is excluded. Missing/corrupt images fail explicitly; unlabeled real frames remain valid elapsed timesteps, and only batch tails are padded. Regenerate older targets/frames with `python -m DIEM.build_heatmaps` and `python -m DIEM.extract_frames` without `--skip-existing`. See [DIEM frame integrity](DIEM/README.md#frame-integrity-and-annotation-mapping). Gaze evaluation automatically reads the new mapping and reports source/frame verification separately from unverified external timestamp synchronization.


## Run settings and result exports

OSIE training, DIEM training and both MIT evaluation entry points write `run_config.json` before loading data. It records every CLI argument including defaults and resolved paths, the requested/effective device, seed, environment and source hashes. Runtime sections add the actual model architecture, preprocessing, selected samples or video chunks, split/annotation fingerprints, loss/optimizer settings and initialization as they become available. Each record has a run ID, completion/failure/interruption status and elapsed time. The checkpoint training sidecars remain available and reference the CLI run ID/config path.

| Entry point | Default export directory under `output file/` | Result files |
|---|---|---|
| `trainGazeLNN.py` | `osie_training/<checkpoint stem>/` | `results.json`, `history.csv` |
| `DIEM/fine_tune.py` | `diem_training/<checkpoint stem>/` | `results.json`, `history.csv` |
| `Evaluate_OSIE_model.py` | `mit_evaluation/metrics/` | `results.json`, `summary.csv`, `per_scanpath.csv` |
| `testMITlowres.py` or MIT `--loss-only` | `mit_evaluation/loss_only/` | `results.json`, `summary.csv` |

All accept `--output-dir PATH` (alias `--out-dir`); `GAZELNN_OUTPUT_DIR` changes the common default root. Use a distinct export directory/checkpoint for experiments you want to retain; repeating a run in the same directory replaces its exports. Existing checkpoint defaults are retained; `--checkpoint` controls weight placement. Training results include loss history, checkpoint hash and best-epoch selection. OSIE `--evaluate-test` also saves its test loss.

```bash
python -m DIEM.fine_tune --device auto --epochs 100 --videos-per-batch 8 --window-len 24
python -m DIEM.fine_tune --processed-root "output file/diem" --head-lr 0.0001 --train-backbone --unfreeze-backbone-after-epoch 5
python Evaluate_OSIE_model.py --device auto --output-dir "output file/mit_experiment"
python testMITlowres.py --dataset-format mat --data-root PATH --output-dir "output file/mit_mat_loss"
```

DIEM also exposes both learning rates, patience, gradient clipping, initialization checkpoint or `--from-scratch`, `--no-pretrained-backbone`, teacher forcing, AMP, workers, seed, preprocessing and split manifest. `--frames-root` defaults to `<processed-root>/frames`; chunks always have stride equal to `--window-len`. `--no-amp` explicitly disables AMP. The direct Python `fine_tune()` API remains supported.

MIT exports the six scanpath scores and KL-DTW loss. Summary tables include finite/missing counts and population standard deviations for scanpath metrics; per-scanpath tables retain the evaluation index, base sample index, image and subject. Undefined scores (including TDE for fewer than three fixations) are JSON `null` and blank CSV cells, excluded from metric means and counted as missing. The KL-DTW value retains the existing arithmetic mean of batch losses, with equal batch weighting recorded explicitly. Loss-only mode exports that loss without claiming scanpath metrics. MIT accepts `--seed`, `--min-len`, `--max-len`, `--sigma` and `--gamma`; image/grid sizes follow the loaded model. Manifest and MAT annotation time conventions are recorded separately.

Regression checks: `python -m unittest discover -s tests -v`.
