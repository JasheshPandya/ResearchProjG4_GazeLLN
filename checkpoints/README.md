# OSIE checkpoint

`OSIE_model.pt` is the single retained OSIE checkpoint. Its canonical default path is defined in `project_paths.py` and used by OSIE training, MIT evaluation and DIEM fine-tuning.

During consolidation on 15 October 2026, both former copies (`OSIE_model.pt` at the repository root and `DIEM/OSIE_model.pt`) had SHA-256:

`d1d3789873ca1b4d200dc15c84e9cdf274f8db1006bce95fffdc9462ca9bb6d0`

The retained file had the same checksum after migration. This fingerprint records the migration, not a promise that later training will leave the file unchanged. The state dict contains both `coordconv_in` and `coordconv_out`; it loads strictly into `gaze_model.GazeLLNArch`.
