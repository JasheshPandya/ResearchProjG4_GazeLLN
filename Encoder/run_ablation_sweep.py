"""Encode reproducible CRF sweeps with uniform, gaze, center and motion allocation."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

if __package__:
    from .build import WORK_DIR
    from .runtime import subprocess_environment
    from .workflow import encode_video, prepare_qp_maps, select_encoder, validate_qp_maps, video_info
    from .baseline_qp_generator import generate_maps
else:
    from build import WORK_DIR
    from runtime import subprocess_environment
    from workflow import encode_video, prepare_qp_maps, select_encoder, validate_qp_maps, video_info
    from baseline_qp_generator import generate_maps


def relative_path(path, directory):
    path = Path(path).resolve()
    try:
        return Path(os.path.relpath(path, directory)).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=WORK_DIR / 'sample_video.mp4')
    parser.add_argument('--out-dir', type=Path, default=WORK_DIR / 'ablation_out')
    parser.add_argument('--qp-dir', type=Path, default=WORK_DIR / 'qp_dir')
    parser.add_argument('--methods', nargs='+', choices=['uniform', 'gaze', 'center', 'motion'], default=['uniform', 'gaze'])
    parser.add_argument('--crfs', nargs='+', type=float, default=[18, 21, 24, 27, 30, 33])
    parser.add_argument('--preset', default='medium')
    parser.add_argument('--qp-min', type=float, default=-3)
    parser.add_argument('--qp-max', type=float, default=10)
    parser.add_argument('--spatial-sigma', type=float, default=1.5)
    parser.add_argument('--temporal-alpha', type=float, default=1)
    parser.add_argument('--center-sigma', type=float, default=0.2)
    parser.add_argument('--roi-source', type=Path, help='Shared scoring QP maps; required when gaze is omitted.')
    parser.add_argument('--reuse-qp', action='store_true', help='Reuse gaze maps for this exact source clip.')
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--encoder', type=Path)
    parser.add_argument('--compiler')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args(argv)
    if any(not math.isfinite(v) or not 0 <= v <= 51 for v in args.crfs):
        parser.error('CRFs must be finite and between 0 and 51')
    if len(set(args.methods)) != len(args.methods) or len(set(args.crfs)) != len(args.crfs):
        parser.error('Methods and CRFs must be unique')
    if not all(map(math.isfinite, [args.qp_min, args.qp_max, args.spatial_sigma, args.temporal_alpha, args.center_sigma])) or args.qp_min >= args.qp_max or args.spatial_sigma < 0 or not 0 <= args.temporal_alpha <= 1 or args.center_sigma <= 0:
        parser.error('Invalid QP range, smoothing or center sigma')
    if 'gaze' not in args.methods and args.roi_source is None:
        parser.error('Specify --roi-source for a shared scoring mask when omitting gaze')
    video, output, gaze_maps = args.video.resolve(), args.out_dir.resolve(), args.qp_dir.resolve()
    env = subprocess_environment()
    try:
        video_info(video)
        if args.roi_source:
            validate_qp_maps(video, args.roi_source.resolve())
        encoder = select_encoder(args.encoder, args.skip_build, args.compiler, env)
        output.mkdir(parents=True, exist_ok=True)
        maps, generation_times = {}, {'uniform': 0.0}
        if 'gaze' in args.methods:
            maps['gaze'] = gaze_maps
            generation_times['gaze'] = prepare_qp_maps(video, gaze_maps, args.reuse_qp, env,
                args.qp_min, args.qp_max, args.spatial_sigma, args.temporal_alpha, args.checkpoint)
        for method in args.methods:
            if method in ('center', 'motion'):
                maps[method] = output / f'qp_{method}'
                generation_times[method] = generate_maps(video, maps[method], method, args.qp_min,
                    args.qp_max, args.spatial_sigma, args.temporal_alpha, args.center_sigma)
                validate_qp_maps(video, maps[method])
        variants = []
        # Fresh empty directory guarantees zero offsets without deleting user files.
        with tempfile.TemporaryDirectory(prefix='uniform_qp_', dir=output) as empty:
            maps['uniform'] = Path(empty)
            for crf in args.crfs:
                label = f'crf{crf:g}'
                for method in args.methods:
                    encoded = output / f'{method}_{label}.mp4'
                    duration = encode_video(encoder, video, encoded, maps[method], crf, args.preset, env)
                    generation = generation_times[method]
                    variant = dict(method=method, label=label, path=relative_path(encoded, output), crf=crf,
                                   encode_time_s=duration, qp_generation_time_s=generation,
                                   pipeline_time_s=duration + generation if generation is not None else None)
                    if method != 'uniform':
                        variant['qp_dir'] = relative_path(maps[method], output)
                    variants.append(variant)
        roi_source = args.roi_source.resolve() if args.roi_source else gaze_maps
        manifest = dict(roi_percentile=25, protocol=dict(preset=args.preset, qp_min=args.qp_min,
            qp_max=args.qp_max, spatial_sigma=args.spatial_sigma, temporal_alpha=args.temporal_alpha,
            center_sigma=args.center_sigma, roi='one shared scoring mask for every method'), clips=[dict(
            name=video.stem, reference=relative_path(video, output), roi_source=relative_path(roi_source, output),
            variants=variants)])
        pending = output / 'manifest.json.tmp'
        pending.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
        pending.replace(output / 'manifest.json')
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Sweep failed: {exc}\n')
    print(f"Manifest: {output / 'manifest.json'}")


if __name__ == '__main__':
    main()
