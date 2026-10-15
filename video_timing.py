"""Shared time convention for aligned constant-frame-rate video sequences."""

import math

DT_POLICY = "elapsed_seconds_per_frame_v1"


def frame_interval(fps):
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video/annotation FPS must be finite and positive")
    return 1.0 / fps
