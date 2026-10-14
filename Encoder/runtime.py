"""Portable tool discovery and subprocess environment for encoder scripts."""

import os
from pathlib import Path
import shutil
import sys


def subprocess_environment():
    env = os.environ.copy()
    # Optional MSYS2 installation; never prepend Windows paths on Unix.
    prefix = env.get("MSYS2_MINGW_PREFIX")
    if os.name == "nt" and prefix:
        env["PATH"] = str(Path(prefix) / "bin") + os.pathsep + env.get("PATH", "")
    return env


def find_tool(name, explicit=None, env=None, required=True):
    env = env if env is not None else subprocess_environment()
    candidate = str(explicit or name)
    found = shutil.which(candidate, path=env.get("PATH"))
    if found:
        return str(Path(found).resolve())
    # Also support invoking a virtual environment's Python without activation.
    if not explicit:
        suffix = ".exe" if os.name == "nt" else ""
        for folder in ("Scripts", "bin"):
            path = Path(sys.prefix) / folder / (name + suffix)
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
    if required:
        raise RuntimeError(f"Cannot find {candidate!r}. Install it and add it to PATH, "
                           "or provide its executable path explicitly. See Encoder/README.md.")
    return None
