# GazeLNN video compression

The project runs entirely through Python entry points. All OSIE, DIEM, MIT and encoder model imports use **one architecture**, `gaze_model.GazeLLNArch`, with input and output CoordConv enabled. `gaze_losses.py` provides the shared scanpath KL-DTW loss; `project_paths.py` defines defaults independent of the current working directory.

The previous notebook training/evaluation code is now available as `trainGazeLNN.py` and `testMITlowres.py`. The notebook files have been removed. `OSIE_Modeldata.py` and `DIEM/DIEM_Modeldata.py` retain their public model names as imports of the shared class, rather than defining separate architectures.

## Setup

Activate the intended Python environment and install `requirements.txt`. Native encoding additionally needs the dependencies described in [Encoder/README.md](Encoder/README.md).

```bash
python -m pip install -r requirements.txt
python Encoder/build.py
```


## Input videos, compressed outputs and dataset evaluation

Put one or more videos, with any filenames, in **`input video/`**. Run from the repository root:

```bash
python run_dataset.py
```

All final videos go directly into the single **`output file/`** directory. For example, `input video/My holiday.mov` becomes `output file/compressed_My holiday.mp4`. Inputs with equal stems (including case variants or different extensions) receive a short filename suffix to prevent collisions. No per-video final-output folders are created. An output filename registry also protects same-stem videos introduced in later runs and existing files with unknown ownership. A successful rerun replaces that registered source's previous output atomically; failed encodes preserve the older output. Sweep variants append method and CRF to the compressed filename.

Supported containers: MP4, MKV, AVI, MOV, WebM, M4V, MPG, MPEG and WMV, with case-insensitive extensions. Discovery is deterministic and scans the input directory itself. Empty inputs fail with an actionable message. Formats still need decoder support in the installed OpenCV/FFmpeg build.

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

All generated artifacts live under the shared output root: `_work/<video-id>/qp_gaze` for isolated caches, `manifest.json` for a multi-video sweep, `pipeline_results.json` for encode timings, `video_names.json` for persistent filename ownership, `compression_evaluation/` and `gaze_evaluation/` for reports, and `diem/` for prepared annotations/frames. Legacy generated files are preserved in their existing locations; use explicit paths or regenerate into the new shared root. DIEM raw eye-tracking annotations remain in `DIEM_ROOT`; stimulus videos are matched by stem in `input video/`.

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

The default output is `checkpoints/OSIE_model.pt`, which replaces the existing checkpoint when a better validation model is saved. Use `--checkpoint PATH` to retain the existing trained weights. A companion `.history.json` records train/validation losses. `--evaluate-test` evaluates the best saved model afterward. `--resume PATH` initializes weights; it does not restore optimizer or epoch state.

The converted training retains the former notebook's centered prior, per-fixation time deltas, feature caching without backbone gradients, autoregressive feedback, KL-DTW loss and early stopping. CPU is supported; the batch-size default is 8 rather than the notebook's hard-coded 256.

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

Fine-tuning initializes from `checkpoints/OSIE_model.pt` and saves `DIEM/DIEM_model.pt`. Encoder inference defaults to the latter. Review `RUN_CONFIG` before fine-tuning. For baseline sweeps and measured-gaze evaluation, see [Baselines/README.md](Baselines/README.md).

## Static saliency, video time and saved splits

The trained frame-independent baseline is DeepGaze I, with pinned upstream code and checksum-verified weights in `checkpoints/deepgaze1.pth`. Install its optional dependencies and fetch weights with:

```bash
python -m pip install -r requirements-static-saliency.txt
python -m Baselines.download_static_weights
```

Select it with `--methods uniform gaze center static_saliency` in the compression sweep; all methods use the same encoder and scoring protocol. The weights are already available locally in this checkout.

Video training and inference now use elapsed seconds, `dt = 1 / FPS`, through `video_timing.py`. Old QP caches without this policy must be regenerated.

`DIEM/splits.json` freezes the current 82-video cohort at 58 train / 12 validation / 12 test. `python -m DIEM.splits` verifies it; `--extend` appends new videos to train, while `--overwrite` explicitly creates new assignments. Training refuses missing/unassigned prepared videos instead of silently changing membership. Each saved best DIEM checkpoint receives a `.training.json` record of its weight hash and exact split manifest.

Gaze evaluation defaults to test membership and matching training provenance. The existing historical DIEM weights predate those records; use an explicit diagnostic override or retrain with the saved splits before claiming held-out performance. Full commands and override meanings are in [Baselines/README.md](Baselines/README.md).
