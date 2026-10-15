"""Trained, frame-independent DeepGaze I saliency at the shared heatmap grid."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import json
from importlib.metadata import distribution

from project_paths import STATIC_SALIENCY_CHECKPOINT
from .download_static_weights import sha256_file, WEIGHTS_SHA256, SOURCE_REVISION
from .maps import probability_map


class StaticSaliency:
    """No gaze history, recurrent state or temporal smoothing enters inference."""

    def __init__(self, checkpoint=STATIC_SALIENCY_CHECKPOINT, shape=(32, 48), spatial_sigma=1.5,
                 device=None, input_shape=(256, 384)):
        import torch
        from torch import nn
        from torchvision.models import alexnet
        try:
            from deepgaze_pytorch.features.alexnet import Normalizer
            from deepgaze_pytorch.modules import DeepGazeII, FeatureExtractor
            installation = json.loads(distribution("deepgaze_pytorch").read_text("direct_url.json") or "{}")
            if installation.get("vcs_info", {}).get("commit_id") != SOURCE_REVISION:
                raise RuntimeError("Static saliency requires the pinned official DeepGaze source in requirements-static-saliency.txt")
        except ImportError as exc:
            raise RuntimeError("Install the static baseline with python -m pip install -r requirements-static-saliency.txt") from exc

        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError("Missing trained static-saliency weights. Run python -m Baselines.download_static_weights")
        digest = sha256_file(self.checkpoint)
        if WEIGHTS_SHA256 is not None and digest != WEIGHTS_SHA256:
            raise ValueError("Static-saliency weights differ from the pinned trained release")
        if spatial_sigma < 0 or not np.isfinite(spatial_sigma):
            raise ValueError("Spatial sigma must be finite and nonnegative")
        self.shape, self.input_shape, self.sigma = tuple(shape), tuple(input_shape), spatial_sigma
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Assemble the authors' exact DeepGazeI configuration with their modules.
        # Local torchvision AlexNet avoids the upstream factory's legacy torch.hub
        # download. Complete trained weights below replace every initialization.
        features = nn.Sequential(Normalizer(), alexnet(weights=None))
        extractor = FeatureExtractor(features, ["1.features.10"])
        readout = nn.Sequential(OrderedDict([("conv0", nn.Conv2d(256, 1, 1, bias=False))]))
        self.model = DeepGazeII(features=extractor, readout_network=readout, downsample=2,
                                readout_factor=4, saliency_map_factor=4)
        weights = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.provenance = dict(model="DeepGazeI", source_revision=SOURCE_REVISION,
            checkpoint=str(self.checkpoint), checkpoint_sha256=digest, center_prior="uniform log density",
            independent_frames=True, input_shape=list(self.input_shape), temporal_alpha=1.0,
            input_preprocessing=dict(input_color="RGB", input_range=[0, 255], output_dtype="float32",
                resize="OpenCV INTER_AREA", normalization="upstream DeepGaze AlexNet Normalizer inside model",
                external_imagenet_normalization=False))

    def __call__(self, frame):
        import cv2
        import torch
        from scipy.ndimage import gaussian_filter

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.input_shape[::-1], interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()[None].to(self.device)
        # The upstream normalizer expects RGB values in 0..255, not ImageNet-normalized inputs.
        center = torch.full((1, *self.input_shape), -np.log(np.prod(self.input_shape)), device=self.device)
        with torch.inference_mode():
            log_density = self.model(tensor, center)
            density = log_density.exp().cpu().numpy().squeeze()
        resized = cv2.resize(density, self.shape[::-1], interpolation=cv2.INTER_AREA)
        if self.sigma:
            resized = gaussian_filter(resized, self.sigma)
        return probability_map(resized)
