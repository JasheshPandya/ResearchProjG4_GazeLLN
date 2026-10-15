"""Generate center, motion and trained static-saliency QP maps on the shared grid."""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Baselines.maps import FrameControl, qp_offsets
from project_paths import STATIC_SALIENCY_CHECKPOINT


def generate_maps(video, output_dir, method, qp_min=-3, qp_max=10,
                  spatial_sigma=1.5, temporal_alpha=1, center_sigma=0.2,
                  static_checkpoint=STATIC_SALIENCY_CHECKPOINT):
    import cv2
    start = time.perf_counter()
    if method == "static_saliency":
        if temporal_alpha != 1:
            raise ValueError("Static saliency requires temporal_alpha=1 for independent per-frame inference")
        from Baselines.static_saliency import StaticSaliency
        control = StaticSaliency(static_checkpoint, spatial_sigma=spatial_sigma)
    else:
        control = FrameControl(method, center_sigma=center_sigma, spatial_sigma=spatial_sigma,
                               temporal_alpha=temporal_alpha)
    cap = cv2.VideoCapture(str(video))
    output_dir = Path(output_dir)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mb_width, mb_height = (width + 15) // 16, (height + 15) // 16
        output_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            offsets = qp_offsets(control(frame), mb_height, mb_width, qp_min, qp_max)
            offsets.tofile(output_dir / f"qpoffset_{count:06d}.bin")
            count += 1
        if count == 0:
            raise RuntimeError(f"No decoded frames: {video}")
        metadata = dict(method=method, source_video=str(Path(video).resolve()), num_frames=count,
                        mb_width=mb_width, mb_height=mb_height, qp_min=qp_min, qp_max=qp_max,
                        spatial_sigma=spatial_sigma, temporal_alpha=temporal_alpha, center_sigma=center_sigma,
                        heatmap_shape=[32, 48], dtype="float32", order="raster_scan_row_major",
                        flat_map_policy="zero QP offsets")
        if method == "static_saliency":
            metadata["saliency_model"] = control.provenance
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        cap.release()
    return time.perf_counter() - start
