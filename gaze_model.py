"""Single GazeLNN architecture for OSIE, DIEM, MIT and compression.

Both input and output CoordConv are always enabled. The state-dict names
match the existing trained OSIE and DIEM checkpoints.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from ncps.torch import CfCCell


def load_model_weights(model, checkpoint, device="cpu"):
    """Strictly load the shared architecture; reject incompatible legacy weights."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not all(
        name in state for name in ("coordconv_in.conv.weight", "coordconv_out.conv.weight")
    ):
        raise ValueError("Checkpoint must contain both input and output CoordConv weights for gaze_model.GazeLLNArch")
    model.load_state_dict(state, strict=True)
    return model


def load_gaze_model(checkpoint, device="cpu", **architecture):
    """Load complete trained weights without downloading backbone initialization."""
    model = GazeLLNArch(pretrained_backbone=False, **architecture).to(device)
    return load_model_weights(model, checkpoint, device)

class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        batch_size, _, h, w = x.size()


        y_grid = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(batch_size, 1, h, w)
        x_grid = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(batch_size, 1, h, w)


        x_coord = torch.cat([x, y_grid, x_grid], dim=1)
        return self.conv(x_coord)

class GazeLLNArch(pl.LightningModule):

    def __init__(self, image_h: int = 256, image_w: int = 384, S: int = 8,
                 hidden_size: int = 512, backbone_units: int = 1024, pretrained_backbone: bool = True):
        super().__init__()

        self.save_hyperparameters()


        self.hmap_h = self.hparams.image_h // self.hparams.S
        self.hmap_w = self.hparams.image_w // self.hparams.S
        self.hmap_feature_size = self.hmap_h * self.hmap_w


        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained_backbone else None)
        self.feature_extractor = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.image_feature_size = 960


        # Input CoordConv: enriches prev_hmap with spatial coords before CfC
        self.coordconv_in = CoordConv(in_channels=1, out_channels=1, kernel_size=3, padding=1)

        # Output CoordConv: applied to projected heatmap before feedback
        self.coordconv_out = CoordConv(in_channels=1, out_channels=1, kernel_size=3, padding=1)


        cfc_input_size = self.image_feature_size + self.hmap_feature_size
        self.cfc_cell = CfCCell(
            input_size=cfc_input_size,
            hidden_size=self.hparams.hidden_size,
            backbone_activation="lecun_tanh",
            backbone_units=self.hparams.backbone_units,
            backbone_layers=1
        )


        self.project = nn.Linear(self.hparams.hidden_size, self.hmap_feature_size)

    def extract_features(self, x):
        """Extracts and caches visual features using the MobileNet backbone."""
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flattens to (B, 960)
        return x

    def forward(self, vis_features, prev_hmap, hx, ts):
        """
        Args:
            vis_features: (B, 960) — precomputed by extract_features
            prev_hmap: Previous fixation heatmap of shape (B, hmap_h, hmap_w) or (B, 1, hmap_h, hmap_w)
            hx: Hidden state tensor for the CfCCell
            ts: Elapsed time timespan tensor of shape (B,)
        """

        # Ensure prev_hmap has a channel dimension (B, 1, hmap_h, hmap_w)
        if prev_hmap.dim() == 3:
            prev_hmap = prev_hmap.unsqueeze(1)
        hmap_coords = self.coordconv_in(prev_hmap).flatten(1)          # Shape: (B, 1536)

        # Ensure ts has shape (B, 1) instead of just (B,)
        if ts.dim() == 1:
            ts = ts.view(-1, 1)

        # hx cannot be set as None
        if hx is None:
            hx = torch.zeros(vis_features.size(0), self.hparams.hidden_size, device=vis_features.device, dtype=vis_features.dtype)

        x = torch.cat([vis_features, hmap_coords], dim=1)
        x, hx = self.cfc_cell(x, hx, ts)                # Shape: (B, 1536)


        out_hmap = self.project(x).view(-1, 1, self.hmap_h, self.hmap_w)  # Shape: (B, 1, 32, 48)
        out_hmap = self.coordconv_out(out_hmap)

        # normalize to a proper spatial distribution before feedback
        B_ = out_hmap.shape[0]
        out_hmap_flat = out_hmap.view(B_, -1)                              # (B, 32*48)
        out_hmap_flat = F.softmax(out_hmap_flat, dim=-1)                   # sums to 1 per sample
        out_hmap = out_hmap_flat.view(B_, 1, self.hmap_h, self.hmap_w)    # (B, 1, 32, 48)

        return out_hmap, hx
