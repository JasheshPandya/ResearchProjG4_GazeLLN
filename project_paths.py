"""Working-directory independent project data/checkpoint defaults."""

from pathlib import Path
import os
import hashlib
import re
import json

PROJECT_ROOT = Path(__file__).resolve().parent
OSIE_DATA_ROOT = PROJECT_ROOT / "OSIE"
MIT_DATA_ROOT = PROJECT_ROOT / "mit_lowres_highres_only"
OSIE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "OSIE_model.pt"
DIEM_CHECKPOINT = PROJECT_ROOT / "DIEM" / "DIEM_model.pt"
STATIC_SALIENCY_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "deepgaze1.pth"

# Shared by every video entry point. Overrides are useful for mounted datasets.
INPUT_VIDEO_DIR = Path(os.environ.get("GAZELNN_INPUT_DIR", PROJECT_ROOT / "input video")).expanduser().resolve()
OUTPUT_FILE_DIR = Path(os.environ.get("GAZELNN_OUTPUT_DIR", PROJECT_ROOT / "output file")).expanduser().resolve()
VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".y4m"})


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_videos(input_dir=INPUT_VIDEO_DIR, video=None):
    """Deterministic flat-folder discovery; never silently ignore an empty input."""
    if video is not None:
        paths = [Path(video).expanduser().resolve()]
    else:
        directory = Path(input_dir).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Input video directory does not exist: {directory}")
        paths = sorted((p.resolve() for p in directory.iterdir()
                        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS), key=lambda p: p.name)
    if not paths:
        raise ValueError(f"No supported videos in {input_dir}. Add one or more videos first.")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Video does not exist: {path}")
    return paths


def video_id(video):
    """Stable portable directory key, independent of discovery order/cohort.

    Hash the complete filename (including extension/case) to disambiguate equal
    stems and filenames that sanitize to the same slug on Windows/macOS.
    """
    name = Path(video).name
    slug = re.sub(r"[^a-z0-9_-]+", "_", Path(video).stem.lower()).strip("_")[:60] or "video"
    return f"{slug}_{hashlib.sha256(name.encode('utf-8')).hexdigest()[:12]}"


def compressed_names(videos, output_dir=None):
    """Flat output filenames; disambiguate equal stems, case and unsafe names."""
    stems = [p.stem.casefold() for p in videos]
    registry = _output_registry(output_dir)
    owners = {name.casefold(): source for name, source in registry.items()}
    previous = {source: name for name, source in registry.items()}
    names = {}
    for path, stem in zip(videos, stems):
        unsafe = bool(re.search(r'[<>:"/\\|?*\x00-\x1f]', path.stem)) or path.stem.endswith((' ', '.'))
        unsafe = unsafe or len(path.stem.encode('utf-8')) > 180
        suffix = video_id(path) if stems.count(stem) > 1 or unsafe else path.stem
        source = str(path.resolve())
        name = previous.get(source, f"compressed_{suffix}.mp4")
        attempt = 0
        while (name.casefold() in owners and owners[name.casefold()] != source) or \
                (output_dir is not None and (Path(output_dir) / name).exists() and name.casefold() not in owners):
            # Keep another invocation's same-stem output, and preserve outputs
            # with unknown ownership. Repeated runs of this source reuse its name.
            token = hashlib.sha256(f"{source}:{attempt}".encode()).hexdigest()[:12]
            name = f"compressed_{video_id(path)}_{token}.mp4"
            attempt += 1
        names[path] = name
        owners[name.casefold()] = source
    if len({name.casefold() for name in names.values()}) != len(videos):
        raise ValueError("Input filenames map to conflicting compressed output names")
    return names


def _output_registry(output_dir):
    path = Path(output_dir) / "video_names.json" if output_dir is not None else None
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                         or not k.startswith("compressed_") or not k.endswith(".mp4")
                                         or Path(k).name != k or "/" in k or "\\" in k for k, v in value.items()):
        raise ValueError(f"Invalid output filename registry: {path}")
    return value


def reserve_compressed_names(names, output_dir):
    output_dir = Path(output_dir)
    registry = _output_registry(output_dir)
    registry.update({name: str(path.resolve()) for path, name in names.items()})
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = output_dir / "video_names.json.tmp"
    pending.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    pending.replace(output_dir / "video_names.json")


def reserved_output_names(output_dir):
    return {name.casefold() for name in _output_registry(output_dir)}
