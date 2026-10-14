# GazeLNN video compression

The project runs entirely through Python entry points. All OSIE, DIEM, MIT and encoder model imports use **one architecture**, `gaze_model.GazeLLNArch`, with input and output CoordConv enabled. `gaze_losses.py` provides the shared scanpath KL-DTW loss; `project_paths.py` defines defaults independent of the current working directory.

The previous notebook training/evaluation code is now available as `trainGazeLNN.py` and `testMITlowres.py`. The notebook files have been removed. `OSIE_Modeldata.py` and `DIEM/DIEM_Modeldata.py` retain their public model names as imports of the shared class, rather than defining separate architectures.

## Setup

Activate the intended Python environment and install `requirements.txt`. Native encoding additionally needs the dependencies described in [Encoder/README.md](Encoder/README.md).

```bash
python -m pip install -r requirements.txt
python Encoder/build.py
```

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
python Encoder/run_pipeline.py --video Encoder/sample_video.mp4
```

Fine-tuning initializes from `checkpoints/OSIE_model.pt` and saves `DIEM/DIEM_model.pt`. Encoder inference defaults to the latter. Review `RUN_CONFIG` before fine-tuning. For baseline sweeps and measured-gaze evaluation, see [Baselines/README.md](Baselines/README.md).

Verification-only test files and their CI workflow were removed after use. `testMITlowres.py` remains because it evaluates trained models on MIT data; it is part of the scientific workflow.

[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) documents all files, outputs, workflows and remaining scientific limitations, including the still-missing static-saliency baseline and time-step/preprocessing consistency work.
