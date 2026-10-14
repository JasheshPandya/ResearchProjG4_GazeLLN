"""Image-only controls. No model checkpoint or evaluation gaze is used."""

import numpy as np
from scipy.ndimage import gaussian_filter


def probability_map(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Heatmaps must be nonempty, finite, nonnegative 2-D arrays")
    total = values.sum()
    return (values / total if total > 0 else np.full(values.shape, 1 / values.size)).astype(np.float32)


def center_map(height, width, sigma_fraction=0.2):
    """Gaussian centered on the image, sigma as a fraction of each dimension."""
    if min(height, width) < 1 or not np.isfinite(sigma_fraction) or sigma_fraction <= 0:
        raise ValueError("Dimensions and center sigma must be positive and finite")
    y, x = np.mgrid[:height, :width]
    values = np.exp(-0.5 * (((x - (width - 1) / 2) / (width * sigma_fraction)) ** 2 +
                            ((y - (height - 1) / 2) / (height * sigma_fraction)) ** 2))
    return probability_map(values)


class FrameControl:
    """Causal center/uniform/frame-difference controls; create once per video."""

    def __init__(self, method, shape=(32, 48), center_sigma=0.2, spatial_sigma=1.5, temporal_alpha=1.0):
        if method not in ("uniform", "center", "motion"):
            raise ValueError(f"Unknown frame control: {method}")
        if min(shape) < 1 or spatial_sigma < 0 or not np.isfinite(spatial_sigma):
            raise ValueError("Invalid grid or spatial sigma")
        if not np.isfinite(temporal_alpha) or not 0 <= temporal_alpha <= 1:
            raise ValueError("Temporal alpha must be in [0, 1]")
        self.method, self.shape = method, shape
        self.center = center_map(*shape, center_sigma)
        self.sigma, self.alpha = spatial_sigma, temporal_alpha
        self.previous_gray = self.previous_map = None

    def __call__(self, frame=None):
        import cv2
        if self.method == "uniform":
            current = np.full(self.shape, 1 / np.prod(self.shape), dtype=np.float32)
        elif self.method == "center":
            current = self.center.copy()
        else:
            if frame is None:
                raise ValueError("Motion control requires a video frame")
            # Resize before subtraction; no future frame is consulted.
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.shape[::-1],
                              interpolation=cv2.INTER_AREA).astype(np.float32)
            current = probability_map(np.abs(gray - self.previous_gray)) if self.previous_gray is not None \
                else np.full(self.shape, 1 / np.prod(self.shape), dtype=np.float32)
            self.previous_gray = gray
        if self.sigma:
            current = gaussian_filter(current, self.sigma)
        if self.previous_map is not None:
            current = self.alpha * current + (1 - self.alpha) * self.previous_map
        self.previous_map = probability_map(current)
        return self.previous_map.copy()


def qp_offsets(heatmap, mb_height, mb_width, qp_min=-3, qp_max=10):
    """Match the gaze generator's bilinear resize, min/max scaling and QP range.

    Flat maps produce zero offsets: no preferential spatial allocation.
    """
    import cv2
    if not np.isfinite([qp_min, qp_max]).all() or qp_min >= qp_max:
        raise ValueError("QP limits must be finite and increasing")
    heatmap = probability_map(heatmap)
    grid = cv2.resize(heatmap, (mb_width, mb_height), interpolation=cv2.INTER_LINEAR)
    span = float(grid.max() - grid.min())
    if span <= 1e-8:
        return np.zeros((mb_height, mb_width), dtype=np.float32)
    normalized = (grid - grid.min()) / span
    return np.clip(qp_max - (qp_max - qp_min) * normalized, qp_min, qp_max).astype(np.float32)
