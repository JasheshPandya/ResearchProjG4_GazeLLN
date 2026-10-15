"""Shared scanpath KL-DTW loss."""

import torch
import torch.nn as nn

def _cost_matrix(
    pred:   torch.Tensor,   # (B, T, H, W)
    target: torch.Tensor,   # (B, T, H, W)
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute pairwise KL divergence between every predicted step i and every
    ground truth step j, giving a cost matrix of shape (B, T_pred, T_gt).

    cost[b, i, j] = KL(target[b,j] || pred[b,i])
    """
    B, T, H, W = pred.shape
    # flatten spatial dims for broadcasting: (B, T, H*W)
    p = pred.view(B, T, -1).clamp(min=eps)
    t = target.view(B, T, -1).clamp(min=eps)

    # normalise
    p = p / p.sum(dim=-1, keepdim=True)
    t = t / t.sum(dim=-1, keepdim=True)

    # (B, T_pred, 1, H*W) vs (B, 1, T_gt, H*W)
    p_exp = p.unsqueeze(2)           # (B, T, 1, H*W)
    t_exp = t.unsqueeze(1)           # (B, 1, T, H*W)

    # KL(t || p) = sum(t * (log t - log p))
    cost = (t_exp * (t_exp.log() - p_exp.log())).sum(dim=-1)  # (B, T, T)

    return cost

def soft_dtw(cost_matrix: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """
    Soft-DTW: a fully differentiable approximation to DTW.

    Uses the soft-min (log-sum-exp) recurrence instead of hard min, so
    gradients flow back through the alignment path.

    Args:
        cost_matrix : (B, T, T) pairwise cost between pred and target steps
        gamma       : smoothing temperature.
                      gamma → 0  recovers hard DTW
                      gamma → ∞  makes all paths equally likely

    Returns:
        dtw_loss : (B,) soft-DTW distance for each item in the batch
    """
    B, T_pred, T_gt = cost_matrix.shape
    device = cost_matrix.device

    # initialise DP table with +inf
    # R[b, i, j] = soft-DTW distance up to (i, j)
    R = torch.full((B, T_pred + 1, T_gt + 1), fill_value=1e9, device=device)
    R[:, 0, 0] = 0.0

    for i in range(1, T_pred + 1):
        for j in range(1, T_gt + 1):
            c = cost_matrix[:, i - 1, j - 1]   # (B,)

            # three possible predecessor cells
            r0 = R[:, i - 1, j    ]   # came from above
            r1 = R[:, i    , j - 1]   # came from left
            r2 = R[:, i - 1, j - 1]   # came from diagonal

            # soft-min via log-sum-exp trick
            # softmin(a, b, c) = -gamma * log(exp(-a/γ) + exp(-b/γ) + exp(-c/γ))
            stacked = torch.stack([r0, r1, r2], dim=-1) / gamma        # (B, 3)
            soft_min = -gamma * torch.logsumexp(-stacked, dim=-1)      # (B,)

            R[:, i, j] = c + soft_min

    return R[:, T_pred, T_gt]

class KLDTWLoss(nn.Module):
    """
    KL-DTW loss combining spatial KL divergence with temporal DTW alignment.

    Args:
        gamma : soft-DTW smoothing temperature (default 0.1)
        eps   : numerical stability constant for KL divergence
    """

    mask_policy = "valid_steps_only_v1"

    def __init__(self, gamma: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.eps   = eps

    def forward(
        self,
        predictions: torch.Tensor,   # (B, T, H, W)
        targets:     torch.Tensor,   # (B, T, H, W)
        mask:        torch.Tensor,   # (B, T) bool — True = real fixation
    ) -> torch.Tensor:
        """
        Args:
            predictions : (B, T, H, W) — model predicted heatmaps
            targets     : (B, T, H, W) — ground truth heatmaps
            mask        : (B, T)       — False on padded steps

        Returns:
            Mean length-normalized loss over nonempty sequences. An entirely
            masked batch returns differentiable zero.
        """
        if predictions.ndim != 4 or targets.shape != predictions.shape:
            raise ValueError("Predictions and targets must have matching (B, T, H, W) shapes")
        if mask.shape != predictions.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Mask must be boolean with shape (B, T)")

        _, _, height, width = predictions.shape
        lengths = mask.sum(dim=1)
        losses = []
        # Remove invalid steps before normalization and alignment. Zeroing
        # them instead would create uniform distributions in _cost_matrix
        # and extra DTW paths. Group by length to retain batched computation.
        for length in lengths.unique().tolist():
            if length == 0:
                continue
            rows = lengths == length
            valid = mask[rows]
            pred = predictions[rows][valid].reshape(-1, length, height, width)
            target = targets[rows][valid].reshape(-1, length, height, width)
            cost = _cost_matrix(pred, target, eps=self.eps)
            losses.append(soft_dtw(cost, gamma=self.gamma) / length)

        if not losses:
            # An empty slice keeps a gradient connection without reading any
            # masked values, which may contain NaNs or arbitrary padding.
            return predictions.reshape(-1)[:0].sum()
        return torch.cat(losses).mean()
