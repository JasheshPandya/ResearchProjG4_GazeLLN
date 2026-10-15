# OSIE checkpoint

`OSIE_model.pt` is the single retained OSIE checkpoint. Its canonical default path is defined in `project_paths.py` and used by OSIE training, MIT evaluation and DIEM fine-tuning.

During consolidation on 15 October 2026, both former copies (`OSIE_model.pt` at the repository root and `DIEM/OSIE_model.pt`) had SHA-256:

`d1d3789873ca1b4d200dc15c84e9cdf274f8db1006bce95fffdc9462ca9bb6d0`

The retained file had the same checksum after migration. This fingerprint records the migration, not a promise that later training will leave the file unchanged. The state dict contains both `coordconv_in` and `coordconv_out`; it loads strictly into `gaze_model.GazeLLNArch`.

## Static saliency weights

`deepgaze1.pth` contains the [authors' trained DeepGaze I weights](https://github.com/matthias-k/DeepGaze/releases/download/v1.01/deepgaze1.pth), downloaded for the per-frame independent baseline. SHA-256 is `1dffa45f1fb5a8fb97c91a5fbaa2722cb061dded31e9d38468d295c703241d94`, size 244,412,378 bytes. `deepgaze1.pth.json` records its source and pinned implementation revision.

The large binary is local and excluded from normal Git tracking. Restore it with `python -m Baselines.download_static_weights`; the downloader verifies the pinned fingerprint and never substitutes random weights. Install the matching implementation with `uv sync --locked --extra static-saliency`; [environment setup](../ENVIRONMENT.md) also describes the compatibility requirements export.

## Training provenance

Both OSIE training (`trainGazeLNN.py` or `OSIE_Modeldata.train`) and DIEM fine-tuning save `<checkpoint>.training.json` alongside the plain PyTorch state dict. `checkpoint_provenance.py` implements the common record; the explicit architecture version lives in `gaze_model.GazeLLNArch.ARCHITECTURE_VERSION`.

| Record | Contents |
|---|---|
| Architecture | Qualified class, architecture version, constructor dimensions/hyperparameters, model-source SHA-256 and state-dict shape/dtype signature |
| Split/data | OSIE image assignments, annotation MAT fingerprint, exact selected base-sample indices and image/subject identities (including nested sample-capped subsets); DIEM manifest/hash, full video assignments, used target hashes and window indices |
| Hyperparameters | Optimizer defaults and rates, loss/epsilon, epoch limit, early stopping, clipping, loaders/samplers, preprocessing, feedback, backbone-freezing schedule, device, AMP, seeds and entry-point configuration |
| Initialization | Random versus named ImageNet backbone initialization; or source-checkpoint path/hash, verified parent-record snapshot/hash and explicit unknown/invalid parent-history status |
| History/selection | Every completed epoch's train/validation loss, wall time, learning rates and DIEM backbone state; separate best epoch/validation loss, run ID, UTC timestamps and completion/early-stop/interruption/failure status |
| Environment | Python/library versions, CUDA/backend flags, Git revision/dirty state and hashes of relevant training/preprocessing source files |

The sidecar's `checkpoint_sha256` binds it to the selected weights. History is updated after every completed epoch, including epochs after the best checkpoint. `<checkpoint stem>.history.json` retains the simple train/validation-loss arrays for existing consumers. The checkpoint record keeps `schema_version: 1` for existing DIEM split/prediction verifiers and adds `provenance_version: 1` for this richer format.

The shared loader can reconstruct recorded GazeLNN constructor parameters without downloading backbone initialization. An absent record retains the legacy default/explicit architecture behavior. A record with a mismatched weight hash is rejected by `read_training_record`; diagnostic weight loading treats its training history as invalid and uses default/explicit architecture rather than trusting it. DIEM held-out verification remains strict unless the existing explicit diagnostic override is selected. Weight and record files are prepared before replacement; readers detect a hash mismatch if a process dies between replacing the two files.

These are training provenance records, not optimizer/scaler/RNG restoration snapshots. `--resume` initializes weights and starts a fresh optimizer/epoch history. Entry points seed before constructing models/loaders; direct callers must manage their RNG setup. Package versions and source hashes identify the implementation but do not constitute a dataset snapshot or guarantee bitwise reproduction across devices.

Existing historical OSIE/DIEM files are not retroactively assigned architecture, splits, hyperparameters or training history. A new run initialized from historical weights records their actual file hash and labels the unavailable parent history unknown. No existing weights are retrained or overwritten just to create metadata.

Verification: `python -m unittest test_checkpoint_provenance -v`.

## Pixel preprocessing declaration

New trainer records include top-level `input_preprocessing` from the actual train/validation dataset policy, not just constructor defaults. The default is the shared ImageNet-normalized RGB profile in `gaze_preprocessing.py`; DIEM's explicit `legacy_rgb01` profile omits normalization. Custom tensor transforms are labeled custom, and direct caller datasets without a declaration remain unknown. DIEM frame-preparation metadata is included separately so older offline resize artifacts are not assigned invented provenance.

GazeLNN inference selects the checkpoint-declared profile; a conflicting explicit override fails. Missing declarations retain the prior ImageNet inference default with an unknown-history warning, or accept an explicit diagnostic `--preprocessing legacy_rgb01`. Saved predictions and QP maps carry the active specification. Cache reuse also checks the shared preprocessing implementation hash; old caches need regeneration. See the root README for the full contract and migration steps.

## Recurrent chunk provenance

New DIEM records include the `contiguous_chunks_with_detached_state_v1` temporal policy, actual chunk length/stride/index, strict next-frame state cursors, tail-padding exclusion and valid-frame loss pooling. State is reset for each epoch/validation pass and detached at chunk boundaries. Old checkpoints retain their historical provenance; no corrected chronology is retroactively assigned to earlier overlapping-window runs.


Scripted OSIE/DIEM training also writes independent `run_config.json`, `results.json` and `history.csv` reports under the shared output root (override with `--output-dir`). Configurations are saved before loading datasets/models, so an early failure still has a settings/status record. Training sidecars include `hyperparameters.run_settings.run_id` and `run_config` to connect those reports to weights. These exports supplement the hash-bound checkpoint record; they do not provide optimizer/RNG resume state. See [run exports](../README.md#run-settings-and-result-exports) for MIT result tables and default paths.
