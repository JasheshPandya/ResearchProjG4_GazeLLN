"""Run selected video-pipeline stages over one input/output directory pair."""

import argparse
from pathlib import Path

from project_paths import INPUT_VIDEO_DIR, OUTPUT_FILE_DIR, discover_videos
from dataset_evaluation import add_statistics_arguments, validate_statistics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', choices=['encode', 'sweep', 'compression', 'gaze'], default=['encode'],
                        help='Stages run in this order: encode, sweep, compression evaluation, gaze evaluation.')
    parser.add_argument('--input-dir', type=Path, default=INPUT_VIDEO_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_FILE_DIR)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--crf', type=float, default=26)
    parser.add_argument('--crfs', nargs='+', type=float, default=[18, 21, 24, 27, 30, 33])
    parser.add_argument('--preset', default='medium')
    parser.add_argument('--reuse-qp', action='store_true')
    parser.add_argument('--methods', nargs='+', choices=['uniform', 'gaze', 'center', 'motion', 'static_saliency'],
                        default=['uniform', 'gaze'])
    parser.add_argument('--targets-dir', type=Path, help='Aligned gaze NPZs named <video stem>.npz')
    parser.add_argument('--target-start-index', type=int, default=0)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--split-manifest', type=Path)
    parser.add_argument('--no-split-check', action='store_true')
    parser.add_argument('--allow-unverified-checkpoint', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    add_statistics_arguments(parser)
    args = parser.parse_args(argv)
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    try:
        if any(task in args.tasks for task in ('encode', 'sweep', 'gaze')):
            discover_videos(args.input_dir)
        validate_statistics(args.bootstrap_samples, args.confidence)
        if len(args.tasks) != len(set(args.tasks)):
            raise ValueError('Tasks must be unique')
        if len(args.methods) != len(set(args.methods)):
            raise ValueError('Methods must be unique')
        if 'gaze' in args.tasks and args.methods == ['gaze']:
            raise ValueError('Gaze evaluation requires at least one control method')
        if 'sweep' in args.tasks and 'gaze' not in args.methods:
            raise ValueError('The dataset sweep requires gaze for a shared scoring ROI; use Encoder/run_ablation_sweep.py --roi-source for external ROI maps')
        if 'gaze' in args.tasks and 'gaze' in args.methods and args.checkpoint is None:
            from project_paths import DIEM_CHECKPOINT
            args.checkpoint = DIEM_CHECKPOINT
    except ValueError as exc:
        parser.error(str(exc))
    shared = ['--input-dir', str(args.input_dir), '--output-dir', str(args.output_dir)]
    encoding = ['--preset', args.preset]
    if args.checkpoint:
        encoding += ['--checkpoint', str(args.checkpoint)]
    if args.reuse_qp:
        encoding += ['--reuse-qp']
    statistics = ['--bootstrap-samples', str(args.bootstrap_samples), '--confidence', str(args.confidence),
                  '--statistics-seed', str(args.statistics_seed)]
    if 'encode' in args.tasks:
        from Encoder.run_pipeline import main as encode
        encode(shared + encoding + ['--crf', str(args.crf)])
    if 'sweep' in args.tasks:
        from Encoder.run_ablation_sweep import main as sweep
        sweep(shared + encoding + ['--methods', *args.methods, '--crfs', *map(str, args.crfs)])
    if 'compression' in args.tasks:
        from Encoder.evaluate_compression import main as compression
        command = ['--manifest', str(args.output_dir / 'manifest.json'),
                   '--out-dir', str(args.output_dir / 'compression_evaluation'), *statistics]
        compression(command + (['--no-plots'] if args.no_plots else []))
    if 'gaze' in args.tasks:
        from Baselines.evaluate_gaze import main as gaze
        command = ['--input-dir', str(args.input_dir), '--out-dir', str(args.output_dir / 'gaze_evaluation'),
                   '--target-start-index', str(args.target_start_index), '--split', args.split, *statistics,
                   '--methods', *[m for m in args.methods if m != 'gaze']]
        # Gaze is selected through a checkpoint; controls are the other methods.
        if 'gaze' in args.methods:
            command += ['--checkpoint', str(args.checkpoint)]
        for option in ('targets_dir', 'split_manifest'):
            if getattr(args, option):
                command += ['--' + option.replace('_', '-'), str(getattr(args, option))]
        for flag in ('no_split_check', 'allow_unverified_checkpoint'):
            if getattr(args, flag):
                command += ['--' + flag.replace('_', '-')]
        gaze(command)


if __name__ == '__main__':
    main()
