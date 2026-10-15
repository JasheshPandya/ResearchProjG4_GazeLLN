"""Shared, working-directory independent encoding operations."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from video_timing import DT_POLICY, frame_interval
from project_paths import DIEM_CHECKPOINT, file_sha256

if __package__:
    from .build import WORK_DIR, build_encoder, encoder_path, is_current_encoder
else:
    from build import WORK_DIR, build_encoder, encoder_path, is_current_encoder


def run_cmd(command, env=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    if result.stdout:
        print(result.stdout, end="", flush=True)
    return result


def select_encoder(explicit=None, skip_build=False, compiler=None, env=None):
    path = Path(explicit or encoder_path()).resolve()
    if skip_build or explicit:
        if not path.is_file():
            raise RuntimeError(f"Encoder does not exist: {path}. Run python Encoder/build.py first.")
        return path
    if compiler is None and is_current_encoder(path):
        print(f"Using encoder built from the current source: {path}")
        return path
    return build_encoder(compiler=compiler, env=env)


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        info = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                cap.get(cv2.CAP_PROP_FPS))
        if min(info) <= 0:
            raise RuntimeError(f"Video has invalid frame count, dimensions or frame rate: {path}")
        return info
    finally:
        cap.release()


def validate_qp_maps(video, qp_dir):
    frames, width, height, _ = video_info(video)
    metadata_path = Path(qp_dir) / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Missing QP metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    mb_width, mb_height = (width + 15) // 16, (height + 15) // 16
    if (metadata.get("num_frames"), metadata.get("mb_width"), metadata.get("mb_height")) != \
            (frames, mb_width, mb_height):
        raise RuntimeError("QP map frame count/grid does not match the input video. Regenerate maps.")
    for index in range(frames):
        path = Path(qp_dir) / f"qpoffset_{index:06d}.bin"
        if not path.is_file() or path.stat().st_size != mb_width * mb_height * 4:
            raise RuntimeError(f"Missing or incomplete QP map: {path}")


def prepare_qp_maps(video, qp_dir, reuse=False, env=None, qp_min=-3, qp_max=10,
                    spatial_sigma=1.5, temporal_alpha=1, checkpoint=None):
    if reuse:
        validate_qp_maps(video, qp_dir)
        metadata = json.loads((Path(qp_dir) / "metadata.json").read_text())
        expected = dict(source_sha256=file_sha256(video), checkpoint_sha256=file_sha256(checkpoint or DIEM_CHECKPOINT),
                        qp_min=qp_min, qp_max=qp_max, spatial_sigma=spatial_sigma, temporal_alpha=temporal_alpha)
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Cached gaze maps do not match source content/checkpoint/settings. Regenerate without --reuse-qp.")
        fps = video_info(video)[3]
        if metadata.get("dt_policy") != DT_POLICY or abs(metadata.get("dt_seconds", -1) - frame_interval(fps)) > 1e-9:
            raise RuntimeError("Cached gaze maps use an unknown/old time convention. Regenerate without --reuse-qp.")
        print(f"Reusing validated QP maps: {qp_dir}")
        return None  # Generation time was not measured during this run.
    start = time.perf_counter()
    command = [sys.executable, WORK_DIR / "qp_map_generator.py", "--video_path", video,
             "--output_dir", qp_dir, "--qp_min", qp_min, "--qp_max", qp_max,
             "--spatial_sigma", spatial_sigma, "--temporal_alpha", temporal_alpha]
    if checkpoint is not None:
        command += ["--checkpoint", Path(checkpoint).resolve()]
    run_cmd(command, env)
    duration = time.perf_counter() - start
    validate_qp_maps(video, qp_dir)
    return duration


def encode_video(encoder, video, output, qp_dir, crf, preset, env=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Commit only a successfully encoded, verified video. Keep an older output
    # available if the encoder fails; closed temp files also work on Windows.
    with tempfile.TemporaryDirectory(prefix="encode_", dir=output.parent) as tmp:
        pending = Path(tmp) / "output.mp4"
        start = time.perf_counter()
        run_cmd([encoder, video, pending, qp_dir, "--crf", crf, "--preset", preset], env)
        duration = time.perf_counter() - start
        ref, encoded = video_info(video), video_info(pending)
        if ref[:3] != encoded[:3] or abs(ref[3] - encoded[3]) > ref[3] * 1e-4:
            raise RuntimeError(f"Encoder changed frame count/dimensions/rate: {ref} -> {encoded}")
        pending.replace(output)
    return duration
