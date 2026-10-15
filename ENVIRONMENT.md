# Reproducing the Python environment

The reference runtime is **CPython 3.14.4**, standard GIL build, on Linux
x86-64. Its working package versions were retained rather than replaced with
an untested older stack. The dependency metadata resolves without conflicts;
CPU regression checks pass. Windows/macOS package markers are included in the
lock, but those platforms and GPU execution have not been validated here.

| Component | Pin |
|---|---|
| Python | 3.14.4 |
| uv | 0.13.1 |
| PyTorch / torchvision | 2.14.1 / 0.29.1 |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| PyTorch Lightning / ncps | 2.6.6 / 1.0.1 |
| Pillow / imageio | 12.3.0 / 2.38.1 |
| OpenCV | opencv-python 5.0.0.93 |

The complete direct dependency list is in `pyproject.toml`. `uv.lock` also pins
transitive packages, platform-specific dependencies and registry artifact
hashes. `.python-version` selects the reference interpreter. No notebook
tooling is required. The optional DeepGaze implementation is fixed to commit
`c33b89f08016e41e68cec4e4d9f1a73a14211386`, with boltons 26.2.0.

## Install and run

From the repository root, bootstrap the exact lockfile tool using an existing
Python with pip:

```bash
python -m pip install uv==0.13.1
uv sync --locked
uv pip check
uv run --locked python -m unittest discover -s tests -v
uv run --locked python run_dataset.py
```

uv creates `.venv/` and can download Python 3.14.4 when that interpreter is
missing. This leaves an existing `venv/` alone. `uv run` chooses the project
environment without shell activation. For interactive use, activate
`.venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` on Windows.
The project is a collection of scripts, so uv installs dependencies without
building/installing the repository as a Python package.

`--locked` rejects an out-of-date lock instead of silently resolving a new
environment. This follows [uv's locking and synchronization
contract](https://docs.astral.sh/uv/concepts/projects/sync/). Avoid an
unqualified `uv lock --upgrade` when reproducing an existing experiment.

For the optional trained static-saliency baseline:

```bash
uv sync --locked --extra static-saliency
uv run --locked --extra static-saliency python -m Baselines.download_static_weights
uv run --locked --extra static-saliency python Encoder/run_ablation_sweep.py --methods uniform gaze center static_saliency
```

Git is required to obtain the pinned DeepGaze source. Checkpoints and datasets
are separate inputs; dependency synchronization does not download them.

## Wheels, source builds and GPU variants

This lock uses the public PyPI distributions. In particular, the Linux
PyTorch distribution brings its CUDA 13 runtime dependencies, which are
pinned transitively even when running with `--device cpu`. The compatible
PyTorch/torchvision pairing is recorded in the [upstream release
metadata](https://pypi.org/project/torchvision/0.29.1/). A working NVIDIA driver
and suitable hardware are external requirements for GPU execution.

FastDTW and optional DeepGaze are built from source. uv uses an isolated,
hash-constrained setuptools 84.0.0 backend, matching the reference environment.
Its [build constraints and hash
checks](https://docs.astral.sh/uv/reference/settings/#build-constraint-dependencies)
are part of the project settings. The reference FastDTW wheel uses its pure
Python implementation; installing Cython or disabling build isolation can
change which implementation is built.

Do not replace torch/torchvision with arbitrary CPU, ROCm or alternate CUDA
wheels and continue calling it the same environment. Such a variant needs
its own source/index configuration, lock and validation. This repository
does not claim a tested GPU or alternative-wheel lock.

The Python lock does not pin the operating system, compiler, NVIDIA driver,
native FFmpeg/x264/libvmaf libraries, data or checkpoint weights, and does not
guarantee bitwise identical training results. Native encoder setup remains
in [Encoder/README.md](Encoder/README.md). Record those system inputs with
published experiments.

## Compatibility requirements exports

`requirements.txt` is a generated snapshot of the base runtime with exact
versions and hashes. `requirements-static-saliency.txt` is a complete snapshot
including the optional baseline and its immutable Git revision. The latter
omits pip hash mode because pip cannot hash-check a VCS dependency. Prefer
`uv sync --locked` for reproduction: it also enforces the interpreter and
isolated source-build constraints, which ordinary pip requirements do not.

After an intentional dependency update, regenerate both files from the lock:

```bash
uv lock
uv export --locked --no-emit-project --output-file requirements.txt
uv export --locked --extra static-saliency --no-hashes --no-emit-project --output-file requirements-static-saliency.txt
uv lock --check
uv pip check
uv run --locked python -m unittest discover -s tests -v
```

Edit direct pins in `pyproject.toml` first. Commit that file, `uv.lock`, both
exports and any intentional interpreter change together. Updating a pin is a
new environment that requires validation; the existing lock retains prior
transitive versions unless deliberately upgraded.

Validation performed for this specification: base and optional locked sync
plans, equality of locked versions with the installed reference packages,
`pip check`, fresh isolated FastDTW/DeepGaze builds with the pinned backend,
imageio installation/import, and all 49 regression tests using those freshly
built packages. The complete GPU-sized environment was not reinstalled into
a second virtual environment during this check.
