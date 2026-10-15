"""Torch pixel metrics, loaded only when GPU evaluation is requested.

Luma conversion and ROI ranking remain identical to the OpenCV CPU evaluator.
Convolutions use IEEE float32 (no TF32 or mixed precision). Small CPU/GPU SSIM
differences from convolution/reduction order are expected, not bitwise equality.
"""

import cv2
import numpy as np
import torch
from torch.nn import functional as F


class TorchFrameScorer:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.used_cpu_fallback = False
        kernel = torch.as_tensor(cv2.getGaussianKernel(11, 1.5, cv2.CV_32F).ravel(),
                                 device=self.device)
        self.horizontal = kernel.reshape(1, 1, 1, 11)
        self.vertical = kernel.reshape(1, 1, 11, 1)
        self.weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                                    dtype=torch.float64, device=self.device)

    def _pad(self, values, axis):
        size = values.shape[axis]
        if size > 5:
            return F.pad(values, (5, 5, 0, 0) if axis == -1 else (0, 0, 5, 5), mode="reflect")
        # OpenCV BORDER_REFLECT_101 also supports images smaller than the window.
        indices = torch.arange(-5, size + 5, device=self.device)
        if size == 1:
            indices = torch.zeros_like(indices)
        else:
            period = 2 * (size - 1)
            indices = indices.remainder(period)
            indices = torch.minimum(indices, period - indices)
        return values.index_select(axis, indices)

    def _maps(self, a, b):
        # Batch five fields through separable Gaussian convolutions.
        fields = torch.stack((a, b, a * a, b * b, a * b))[:, None]
        fields = F.conv2d(self._pad(fields, -1), self.horizontal)
        fields = F.conv2d(self._pad(fields, -2), self.vertical)[:, 0]
        mu1, mu2, aa, bb, ab = fields.unbind(0)
        mu1_sq, mu2_sq, mu12 = mu1.square(), mu2.square(), mu1 * mu2
        sigma1 = (aa - mu1_sq).clamp_min(0)
        sigma2 = (bb - mu2_sq).clamp_min(0)
        sigma12 = ab - mu12
        luminance = (2 * mu12 + 6.5025) / (mu1_sq + mu2_sq + 6.5025)
        cs = (2 * sigma12 + 58.5225) / (sigma1 + sigma2 + 58.5225)
        return luminance * cs, cs

    @torch.inference_mode()
    def __call__(self, y_ref, y_dist, roi_source, index, roi_percentile):
        mask = torch.as_tensor(roi_source.roi_mask(index, *y_ref.shape, roi_percentile),
                               device=self.device)
        a = torch.as_tensor(np.ascontiguousarray(y_ref), device=self.device).float()
        b = torch.as_tensor(np.ascontiguousarray(y_dist), device=self.device).float()
        se = (a - b).square()
        roi_count = mask.sum()
        counts = torch.stack((roi_count.new_tensor(a.numel()), roi_count, a.numel() - roi_count))
        totals = torch.stack([v.sum(dtype=torch.float64) for v in (se, se[mask], se[~mask])])
        del se
        # cuDNN otherwise permits TF32, which changes low-variance SSIM scores.
        with torch.backends.cudnn.flags(enabled=True, allow_tf32=False, fp32_precision="ieee"):
            maps = self._maps(a, b)
            ssim = torch.stack([v.mean(dtype=torch.float64)
                                for v in (maps[0], maps[0][mask], maps[0][~mask])])
            factors = []
            for level in range(5):
                if min(a.shape) < 11:
                    break
                smap, cs = maps if level == 0 else self._maps(a, b)
                component = smap if level == 4 else cs
                factors.append(component[5:-5, 5:-5].mean(dtype=torch.float64).clamp(0, 1))
                if level < 4:
                    padding = (0, a.shape[1] % 2, 0, a.shape[0] % 2)
                    a = F.avg_pool2d(F.pad(a[None, None], padding, mode="reflect"), 2)[0, 0]
                    b = F.avg_pool2d(F.pad(b[None, None], padding, mode="reflect"), 2)[0, 0]
            ms_score = ((torch.stack(factors) ** self.weights).prod() if len(factors) == 5
                        else totals.new_tensor(float("nan")))
        # One synchronization/transfer of scalars rather than one per metric.
        values = torch.cat((totals, counts, ssim, ms_score.reshape(1))).cpu().numpy()
        totals, counts, ssim = values[:3], values[3:6].astype(np.int64), values[6:9]
        with np.errstate(divide="ignore", invalid="ignore"):
            psnr = 10 * np.log10(255. ** 2 / (totals / counts))
        return totals, counts, psnr.tolist(), ssim.tolist(), float(values[9])
