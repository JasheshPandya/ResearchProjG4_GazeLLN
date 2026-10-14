"""Build the native encoder, generate QP maps and encode one gaze-guided video."""

import argparse
from pathlib import Path
import subprocess

if __package__:
    from .build import WORK_DIR
    from .runtime import subprocess_environment
    from .workflow import encode_video, prepare_qp_maps, select_encoder, video_info
else:
    from build import WORK_DIR
    from runtime import subprocess_environment
    from workflow import encode_video, prepare_qp_maps, select_encoder, video_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=WORK_DIR / "sample_video.mp4")
    parser.add_argument("--output", type=Path, default=WORK_DIR / "output_video.mp4")
    parser.add_argument("--qp-dir", type=Path, default=WORK_DIR / "qp_dir")
    parser.add_argument("--crf", type=float, default=26)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--reuse-qp", action="store_true", help="Reuse maps belonging to this exact source video.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--encoder", type=Path, help="Use a prebuilt native encoder.")
    parser.add_argument("--compiler", help="GCC/Clang executable for rebuilding.")
    parser.add_argument("--checkpoint", type=Path, help="Trained DIEM checkpoint for QP generation.")
    args = parser.parse_args()
    if not 0 <= args.crf <= 51:
        parser.error("CRF must be between 0 and 51")
    video, output, maps = args.video.resolve(), args.output.resolve(), args.qp_dir.resolve()
    env = subprocess_environment()
    try:
        video_info(video)
        encoder = select_encoder(args.encoder, args.skip_build, args.compiler, env)
        qp_time = prepare_qp_maps(video, maps, args.reuse_qp, env, checkpoint=args.checkpoint)
        encode_time = encode_video(encoder, video, output, maps, args.crf, args.preset, env)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Pipeline failed: {exc}\n")
    print(f"Encoded: {output} ({output.stat().st_size / 1024**2:.2f} MB)")
    print(f"Encoder time: {encode_time:.3f} s")
    if qp_time is not None:
        print(f"Sequential pipeline time: {qp_time + encode_time:.3f} s")


if __name__ == "__main__":
    main()
