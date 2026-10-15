# Running on Windows, Linux and macOS

The encoder uses native GCC/Clang, FFmpeg development libraries, x264 and `pkg-config`. Python scripts discover tools on PATH and use the same build helper. Pipelines reuse a local encoder only when its build record matches the current C source, OS and CPU architecture; changed source triggers a rebuild. Binaries must be built separately for each OS and CPU architecture; a Windows `.exe` cannot run on Linux or macOS. Python 3.11 or 3.12 is a practical choice for the scientific dependencies.

## Native dependencies

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install build-essential pkg-config ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libx264-dev
```

macOS, with Xcode Command Line Tools and Homebrew:

```bash
xcode-select --install
brew install pkgconf ffmpeg x264
```

Windows: install [MSYS2](https://www.msys2.org/), open its **UCRT64** terminal, and install matching packages:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-pkgconf mingw-w64-ucrt-x86_64-ffmpeg mingw-w64-ucrt-x86_64-libx264
```

For PowerShell or a Windows Python environment, add your MSYS2 UCRT64 `bin` directory to PATH, or set `MSYS2_MINGW_PREFIX` to the actual UCRT64 directory. Keep compiler, `pkg-config`, libraries and DLLs from the same MSYS2 environment; do not mix UCRT64 and MINGW64. Visual Studio's MSVC compiler is not the build helper's supported Windows toolchain.

The `CC`/`--compiler` setting can select another GCC/Clang executable. `PKG_CONFIG` and `PKG_CONFIG_PATH` can select a tool or library installation. An alternative CMake build uses the same `pkg-config` dependencies:

```bash
cmake -S Encoder -B Encoder/build-native
cmake --build Encoder/build-native --config Release
```

On Windows, select a GCC/Clang-compatible CMake generator (for example Ninja with `-G Ninja -DCMAKE_C_COMPILER=gcc`) and install that build tool. Do not use the Visual Studio generator with MSYS2 libraries.

VMAF also requires `libvmaf` in the installed FFmpeg build. Check with `ffmpeg -filters`; look for `libvmaf`. The evaluator keeps other quality metrics available when that filter is absent. See [FFmpeg filter documentation](https://ffmpeg.org/ffmpeg-filters.html#libvmaf), [Homebrew FFmpeg](https://formulae.brew.sh/formula/ffmpeg), and [MSYS2 FFmpeg](https://packages.msys2.org/packages/mingw-w64-ucrt-x86_64-ffmpeg).

## Python environment

From the repository root:

```bash
python -m venv venv
```

Activate with `source venv/bin/activate` on Linux/macOS, or `.\venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install -r requirements.txt
python Encoder/build.py
```

This creates `Encoder/encoder` on Linux/macOS and `Encoder/encoder.exe` on Windows. Model inference uses the single input/output CoordConv architecture in `gaze_model.py` and a compatible trained checkpoint. The default is `DIEM/DIEM_model.pt`, resolved from the repository root, or select `--checkpoint PATH`. Missing/incompatible checkpoints fail explicitly. Loading complete weights does not download pretrained backbone initialization. CUDA is optional; CPU inference also runs on all three OSes. Dataset and checkpoint files still need to be supplied.

## Encode and evaluate

Defaults discover every supported video in project-root `input video/` and write all final videos directly into `output file/` as `compressed_<input stem>.mp4` (with a short suffix for colliding stems). Script and default data paths resolve from their files, so invoking a script by its absolute path works from other directories.

```bash
python Encoder/run_pipeline.py
python Encoder/run_ablation_sweep.py
python Encoder/evaluate_compression.py
python Encoder/plot_bdrate.py --metric psnr_roi
```

Use `--input-dir` and `--output-dir` for the shared directories. Choose a single source with `--video`, and optionally `--output` or `--qp-dir` for that one video. The sweep accepts `--out-dir` as an alias for `--output-dir`. Each sweep writes one manifest containing all clips and uniquely named method/CRF outputs directly in the shared output folder. Add `--evaluate` to score the completed manifest immediately. Use `--crfs 18 21 24 27 30 33` to set sweep points. Manifest paths are relative to the manifest, written with forward slashes, and work with Windows and Unix path separators.

Add `--methods uniform gaze center motion static_saliency` to include center-Gaussian, frame-difference and trained frame-independent DeepGaze I baselines. Static saliency uses the pinned dependencies in `requirements-static-saliency.txt` and downloaded `checkpoints/deepgaze1.pth`; it requires temporal alpha 1. The evaluator reports each method against uniform and gaze against the other baselines, with a shared clip-level scoring mask. See [baseline definitions and gaze-prediction evaluation](../Baselines/README.md).

To repair existing encodes while keeping the already generated QP maps for the **same source video**:

```bash
python Encoder/run_ablation_sweep.py --reuse-qp
```

Cached maps are checked for frame count, grid, file sizes, source SHA-256, checkpoint SHA-256 and generation settings. Old caches lacking this provenance must be regenerated. Each video has an isolated cache under the shared output's `_work/<video-id>/qp_gaze`. Generation/pipeline timings remain unavailable for cached gaze maps rather than inventing inference time. Add `--skip-build` to reuse a previously rebuilt native encoder, or `--encoder PATH` for another prebuilt native executable. Encoded files are replaced only after the frame count, dimensions and frame rate pass validation.

Gaze inference now uses elapsed seconds, `dt = 1 / source FPS`, matching video training/evaluation. Cached maps must also carry the matching `dt_policy` and `dt_seconds`; legacy caches are rejected. Regenerate without `--reuse-qp` before comparing results from the new convention.

For custom FFmpeg installations:

```bash
python Encoder/evaluate_compression.py --ffmpeg PATH_TO_FFMPEG --ffprobe PATH_TO_FFPROBE
```

The evaluator can also discover executables inside the Python virtual environment when invoking its Python directly without activation. See [compression metrics](COMPRESSION_METRICS.md) for report definitions.

DIEM raw annotations default to `DIEM/DIEM_ROOT`; stimulus videos are matched by stem in the shared `input video/` folder. Prepared DIEM data defaults to `output file/diem`. Set `DIEM_ROOT` and `DIEM_OUTPUT_ROOT` environment variables to override those locations. Calibration overlays use the OS temporary directory.

Dataset statistics are exported alongside the existing per-video metrics in `output file/compression_evaluation/`. They use equal video weights, video-bootstrap intervals and paired video-level tests; per-video BD deltas are pooled separately. See the root README for the common `run_dataset.py` interface. Run regression checks with `python -m unittest discover -s tests -v`.
