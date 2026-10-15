# OSIE checkpoint

`OSIE_model.pt` is the single retained OSIE checkpoint. Its canonical default path is defined in `project_paths.py` and used by OSIE training, MIT evaluation and DIEM fine-tuning.

During consolidation on 15 October 2026, both former copies (`OSIE_model.pt` at the repository root and `DIEM/OSIE_model.pt`) had SHA-256:

`d1d3789873ca1b4d200dc15c84e9cdf274f8db1006bce95fffdc9462ca9bb6d0`

The retained file had the same checksum after migration. This fingerprint records the migration, not a promise that later training will leave the file unchanged. The state dict contains both `coordconv_in` and `coordconv_out`; it loads strictly into `gaze_model.GazeLLNArch`.

## Static saliency weights

`deepgaze1.pth` contains the [authors' trained DeepGaze I weights](https://github.com/matthias-k/DeepGaze/releases/download/v1.01/deepgaze1.pth), downloaded for the per-frame independent baseline. SHA-256 is `1dffa45f1fb5a8fb97c91a5fbaa2722cb061dded31e9d38468d295c703241d94`, size 244,412,378 bytes. `deepgaze1.pth.json` records its source and pinned implementation revision.

The large binary is local and excluded from normal Git tracking. Restore it with `python -m Baselines.download_static_weights`; the downloader verifies the pinned fingerprint and never substitutes random weights. Install the matching implementation with `requirements-static-saliency.txt`.

New DIEM fine-tuning saves an adjacent `.training.json` sidecar at the configured output checkpoint location. It records the weight hash, exact `DIEM/splits.json` memberships/hash and time policy for held-out evaluation. Existing historical OSIE/DIEM files are not retroactively assigned training provenance.
