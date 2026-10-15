"""Extract 1-indexed RGB PNG frames with the shared inference resize policy.

OpenCV decoding matches encoder/gaze evaluation; Pillow bilinear resizing
matches gaze_preprocessing.py. Stored PNGs contain unnormalized RGB bytes.
"""

from __future__ import annotations

import json
import tempfile

import cv2
from pathlib import Path

if __package__:
    from . import config as cfg
else:
    import config as cfg
from gaze_preprocessing import resize_rgb_image, preprocessing_spec, preprocessing_implementation_sha256
from project_paths import file_sha256
from DIEM.frame_integrity import validate_frame_cache


def extract_frames_for_video(video_dir: Path, frames_root: Path) -> None:
    video_name = video_dir.name
    matches = cfg.video_files(video_dir)
    if not matches:
        raise ValueError(f"No source video for {video_dir}; put the DIEM stimulus in {video_dir / 'video'}")
    video_path = matches[0]

    out_dir = frames_root / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    size = (cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH)
    source_sha = file_sha256(video_path)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        from video_timing import frame_interval
        frame_interval(fps)
        # Complete the new extraction before replacing any old frame files.
        with tempfile.TemporaryDirectory(prefix="extract_", dir=out_dir) as temporary:
            pending = Path(temporary)
            count = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                count += 1
                image = resize_rgb_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size)
                image.save(pending / f"frame_{count:06d}.png")
            if count == 0 or count != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                raise ValueError(f"Incomplete frame extraction for {video_path}: decoded {count}")
            if file_sha256(video_path) != source_sha:
                raise ValueError("Source changed during extraction")
            metadata = dict(version=2, source_video=str(video_path.resolve()), source_sha256=source_sha,
                frame_sha256={p.name: file_sha256(p) for p in sorted(pending.glob("frame_*.png"))},
                num_frames=count, start_number=1, fps=fps, image_size=list(size), color="RGB uint8; no normalization",
                resize=preprocessing_spec(size)["resize"],
                preprocessing_implementation_sha256=preprocessing_implementation_sha256())
            (pending / "frames_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            for index in range(1, count + 1):
                path = pending / f"frame_{index:06d}.png"
                path.replace(out_dir / path.name)
            expected = set(metadata["frame_sha256"])
            for obsolete in out_dir.glob("frame_*.png"):
                if obsolete.name not in expected:
                    obsolete.unlink()
            (pending / "frames_metadata.json").replace(out_dir / "frames_metadata.json")
        print(f"  extracted {count} RGB frames -> {out_dir}")
    finally:
        cap.release()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None,
                     help="comma-separated video names to restrict to "
                          "(default: all videos under DIEM_ROOT)")
    ap.add_argument("--skip-existing", action="store_true",
                     help="skip a video if its frames dir already has PNGs -- "
                          "recommended for full-dataset runs, since this is "
                          "the slowest and most disk-heavy step (one PNG per "
                          "frame per video; a ~90s clip at 30fps is ~2700 "
                          "files). Re-running the whole dataset from scratch "
                          "every time you add a new video is wasteful.")
    args = ap.parse_args()

    frames_root = cfg.OUTPUT_ROOT / "frames"
    video_dirs = sorted(d for d in cfg.DIEM_ROOT.iterdir() if d.is_dir())
    if args.videos:
        wanted = set(args.videos.split(","))
        video_dirs = [d for d in video_dirs if d.name in wanted]

    for vd in video_dirs:
        out_dir = frames_root / vd.name
        if args.skip_existing and out_dir.exists() and any(out_dir.glob("frame_*.png")):
            sources = cfg.video_files(vd)
            if not sources:
                ap.error(f"No source video for {vd}")
            try:
                validate_frame_cache(out_dir, (cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH), sources[0])
            except (ValueError, OSError, KeyError) as exc:
                ap.error(str(exc))
            print(f"Skipping {vd.name} (verified extracted frames, --skip-existing)")
            continue
        print(f"Extracting frames for {vd.name} ...")
        extract_frames_for_video(vd, frames_root)


if __name__ == "__main__":
    main()
