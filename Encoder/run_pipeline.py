"""Compress every video in the shared input folder into the shared output folder."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from project_paths import (INPUT_VIDEO_DIR, OUTPUT_FILE_DIR, discover_videos, video_id,
                           compressed_names, reserve_compressed_names)

if __package__:
    from .runtime import subprocess_environment
    from .workflow import encode_video, prepare_qp_maps, select_encoder, video_info
else:
    from runtime import subprocess_environment
    from workflow import encode_video, prepare_qp_maps, select_encoder, video_info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_VIDEO_DIR)
    parser.add_argument("--output-dir", "--out-dir", type=Path, default=OUTPUT_FILE_DIR)
    parser.add_argument("--video", type=Path, help="Process only this video instead of discovering the input folder.")
    parser.add_argument("--output", type=Path, help="Explicit output filename, allowed only for one video.")
    parser.add_argument("--qp-dir", type=Path, help="Single-video QP cache override.")
    parser.add_argument("--crf", type=float, default=26)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--reuse-qp", action="store_true", help="Reuse maps with matching video/content and generation settings.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--compiler")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.crf) or not 0 <= args.crf <= 51:
        parser.error("CRF must be finite and between 0 and 51")
    try:
        videos = discover_videos(args.input_dir, args.video)
        if len(videos) != 1 and (args.output or args.qp_dir):
            raise ValueError("--output and --qp-dir require exactly one input video")
        output_dir = args.output_dir.expanduser().resolve()
        names = compressed_names(videos, output_dir)
        for video in videos:
            video_info(video)
            output = args.output.resolve() if args.output else output_dir / names[video]
            if output.resolve() in videos:
                raise ValueError(f"Output would overwrite an input video: {output}")
        env = subprocess_environment()
        encoder = select_encoder(args.encoder, args.skip_build, args.compiler, env)
        if not args.output:
            reserve_compressed_names(names, output_dir)
        rows = []
        for video in videos:
            output = args.output.resolve() if args.output else output_dir / names[video]
            maps = args.qp_dir.resolve() if args.qp_dir else output_dir / "_work" / video_id(video) / "qp_gaze"
            qp_time = prepare_qp_maps(video, maps, args.reuse_qp, env, checkpoint=args.checkpoint)
            encode_time = encode_video(encoder, video, output, maps, args.crf, args.preset, env)
            rows.append(dict(video=video.name, source=str(video), output=str(output), qp_dir=str(maps),
                             encode_time_s=encode_time, qp_generation_time_s=qp_time,
                             pipeline_time_s=qp_time + encode_time if qp_time is not None else None))
            print(f"Encoded: {output} ({output.stat().st_size / 1024**2:.2f} MB)")
        output_dir.mkdir(parents=True, exist_ok=True)
        pending = output_dir / "pipeline_results.json.tmp"
        pending.write_text(json.dumps(dict(videos=rows, crf=args.crf, preset=args.preset), indent=2), encoding="utf-8")
        pending.replace(output_dir / "pipeline_results.json")
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Pipeline failed: {exc}\n")
    return rows


if __name__ == "__main__":
    main()
