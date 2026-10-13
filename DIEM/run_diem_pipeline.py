"""Run DIEM preparation and fine-tuning from any working directory."""

import csv
from pathlib import Path
import subprocess
import sys

import config as cfg

WORK_DIR = Path(__file__).resolve().parent


def run_script(script, *args):
    print(f"Starting {script}...", flush=True)
    subprocess.run([sys.executable, str(WORK_DIR / script), *args], cwd=WORK_DIR, check=True)


def main():
    try:
        for script in ("parse_gaze.py", "build_heatmaps.py", "extract_frames.py"):
            run_script(script)
        videos = []
        report = cfg.OUTPUT_ROOT / "calibration_report.csv"
        if report.exists():
            with report.open(newline="", encoding="utf-8") as stream:
                videos = [row["video"] for row in csv.DictReader(stream)
                          if row.get("needs_review") == "no"]
        args = ["--frames", "5"]
        if videos:
            args += ["--videos", ",".join(videos)]
        run_script("inspect_calibration.py", *args)
        run_script("fine_tune.py")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"DIEM pipeline stopped: {exc}")


if __name__ == "__main__":
    main()
