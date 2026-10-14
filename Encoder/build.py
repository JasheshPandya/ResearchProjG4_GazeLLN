"""Build the encoder with GCC/Clang and pkg-config on Windows, Linux, or macOS."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess

if __package__:
    from .runtime import find_tool, subprocess_environment
else:
    from runtime import find_tool, subprocess_environment

WORK_DIR = Path(__file__).resolve().parent
PACKAGES = ("libavcodec", "libavformat", "libavutil", "libswscale", "x264")


def encoder_path(work_dir=WORK_DIR):
    return Path(work_dir) / ("encoder.exe" if os.name == "nt" else "encoder")


def build_identity():
    return {"source_sha256": hashlib.sha256((WORK_DIR / "encoder.c").read_bytes()).hexdigest(),
            "os": platform.system(), "architecture": platform.machine()}


def build_stamp(path):
    return Path(str(path) + ".build.json")


def is_current_encoder(path):
    if not Path(path).is_file():
        return False
    try:
        return json.loads(build_stamp(path).read_text()) == build_identity()
    except (OSError, ValueError):
        return False


def build_encoder(output=None, compiler=None, env=None):
    env = env if env is not None else subprocess_environment()
    requested = compiler or env.get("CC")
    if requested:
        cc = find_tool(requested, explicit=requested, env=env)
    else:
        cc = next((tool for name in ("cc", "gcc", "clang")
                   if (tool := find_tool(name, env=env, required=False))), None)
        if cc is None:
            raise RuntimeError("Install a C compiler (GCC/Clang), or set CC to its executable. "
                               "See Encoder/README.md for your operating system.")
    pkg_config = find_tool("pkg-config", explicit=env.get("PKG_CONFIG"), env=env)
    flags = subprocess.run([pkg_config, "--cflags", "--libs", *PACKAGES],
                           capture_output=True, text=True, env=env)
    if flags.returncode:
        raise RuntimeError("FFmpeg/x264 development libraries are missing or not visible to pkg-config.\n"
                           + flags.stderr.strip() + "\nSee Encoder/README.md.")
    output = Path(output or encoder_path()).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [cc, "-std=c99", "-O3", "-Wall", "-Wextra", str(WORK_DIR / "encoder.c"),
               "-o", str(output), *shlex.split(flags.stdout)]
    print("Building:", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)
    build_stamp(output).write_text(json.dumps(build_identity(), indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", help="GCC/Clang executable; otherwise use CC or PATH.")
    parser.add_argument("--output", type=Path, help="Output executable path.")
    args = parser.parse_args()
    try:
        output = build_encoder(args.output, args.compiler)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Build failed: {exc}\n")
    print(f"Built: {output}")


if __name__ == "__main__":
    main()
