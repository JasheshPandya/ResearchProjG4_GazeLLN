"""Fetch the authors' trained DeepGaze I weights into the project checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

from project_paths import STATIC_SALIENCY_CHECKPOINT

WEIGHTS_URL = "https://github.com/matthias-k/DeepGaze/releases/download/v1.01/deepgaze1.pth"
SOURCE_REVISION = "c33b89f08016e41e68cec4e4d9f1a73a14211386"
WEIGHTS_SHA256 = "1dffa45f1fb5a8fb97c91a5fbaa2722cb061dded31e9d38468d295c703241d94"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(output=STATIC_SALIENCY_CHECKPOINT):
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_file():
        with tempfile.TemporaryDirectory(prefix="saliency_download_", dir=output.parent) as folder:
            pending = Path(folder) / "deepgaze1.pth"
            with urlopen(WEIGHTS_URL, timeout=120) as response, pending.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            digest = sha256_file(pending)
            if WEIGHTS_SHA256 is not None and digest != WEIGHTS_SHA256:
                raise ValueError("Downloaded saliency checkpoint does not match its pinned SHA-256")
            pending.replace(output)
    digest = sha256_file(output)
    if WEIGHTS_SHA256 is not None and digest != WEIGHTS_SHA256:
        raise ValueError("Existing saliency checkpoint does not match its pinned SHA-256")
    metadata = dict(model="DeepGazeI (authors' PyTorch retraining)", weights_url=WEIGHTS_URL,
                    source_repository="https://github.com/matthias-k/DeepGaze", source_revision=SOURCE_REVISION,
                    sha256=digest, bytes=output.stat().st_size, training_data="MIT1003, per upstream model documentation")
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STATIC_SALIENCY_CHECKPOINT)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(download(args.output), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Weight download failed: {exc}\n")


if __name__ == "__main__":
    main()
