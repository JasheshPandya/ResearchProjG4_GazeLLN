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

Defaults use `Encoder/sample_video.mp4`, with case-sensitive spelling on Linux/macOS. Script and default data paths resolve from their files, so invoking a script by its absolute path works from other directories.

```bash
python Encoder/run_pipeline.py
python Encoder/run_ablation_sweep.py
python Encoder/evaluate_compression.py
python Encoder/plot_bdrate.py --metric psnr_roi
```

Choose another video/output with `--video`, `--output` (single encode), `--out-dir` (sweep), and `--qp-dir`. Use `--crfs 18 21 24 27 30 33` to set sweep points. Manifest paths are relative to the manifest, written with forward slashes, and work with Windows and Unix path separators.

Add `--methods uniform gaze center motion` to include center-Gaussian and causal frame-difference baselines. The evaluator reports each method against uniform and gaze against the other baselines, with a shared clip-level scoring mask. See [baseline definitions and gaze-prediction evaluation](../Baselines/README.md).

To repair existing encodes while keeping the already generated QP maps for the **same source video**:

```bash
python Encoder/run_ablation_sweep.py --reuse-qp
```

The cached maps' count, grid and file sizes are checked before use. Content provenance cannot be inferred from those checks; do not reuse maps from another clip. Generation/pipeline timings remain unavailable for cached gaze maps rather than inventing inference time. Add `--skip-build` to reuse a previously rebuilt native encoder, or `--encoder PATH` for another prebuilt native executable. Encoded files are replaced only after the frame count, dimensions and frame rate pass validation.

For custom FFmpeg installations:

```bash
python Encoder/evaluate_compression.py --ffmpeg PATH_TO_FFMPEG --ffprobe PATH_TO_FFPROBE
```

The evaluator can also discover executables inside the Python virtual environment when invoking its Python directly without activation. See [compression metrics](COMPRESSION_METRICS.md) for report definitions.

DIEM defaults now resolve from its script directory (`DIEM/DIEM_ROOT`, `DIEM/DIEM_Processed`). Set `DIEM_ROOT` and `DIEM_OUTPUT_ROOT` environment variables to override those locations. Calibration overlays use the OS temporary directory.

Verification-only regression files and their test-only CI workflow were removed after use. The build, encoding and scientific evaluation tools remain available.
