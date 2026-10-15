"""Shared GazeLNN pixel contract for image/video training and inference.

Decoded RGB uint8 -> Pillow bilinear resize (stretch) -> float32 CHW / 255
-> ImageNet channel normalization. The explicit legacy_rgb01 profile omits
normalization for reproduction of older, documented [0,1] training runs.
DeepGaze has its own upstream pixel contract and does not use this module.
"""

from pathlib import Path
import warnings

import numpy as np
from PIL import Image

DEFAULT_IMAGE_SIZE = (256, 384)
PREPROCESSING_PROFILES = ("imagenet", "legacy_rgb01")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocessing_spec(image_size=DEFAULT_IMAGE_SIZE, profile="imagenet"):
    if profile not in PREPROCESSING_PROFILES:
        raise ValueError(f"Unknown GazeLNN preprocessing profile: {profile}")
    if len(image_size) != 2 or any(int(value) != value or value <= 0 for value in image_size):
        raise ValueError("Preprocessing size must contain positive integer height/width")
    return dict(version=1, profile=profile, image_size=list(map(int, image_size)),
                input_color="RGB", input_dtype="uint8", input_range=[0, 255],
                resize="Pillow bilinear, antialiased downsampling, stretch to (height,width)",
                scale=1 / 255, output_dtype="float32", output_layout="CHW",
                mean=list(IMAGENET_MEAN) if profile == "imagenet" else None,
                std=list(IMAGENET_STD) if profile == "imagenet" else None)


def resize_rgb_image(image, image_size=DEFAULT_IMAGE_SIZE):
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("GazeLNN expects uint8 RGB HWC pixels; do not scale/normalize them beforehand")
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a Pillow image or RGB uint8 NumPy array")
    image = image.convert("RGB")
    height, width = image_size
    if image.size != (width, height):
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return image


def normalize_rgb_tensor(tensor, profile="imagenet"):
    """Normalize CHW or (...,C,H,W) float32 [0,1] tensors once, without mutation."""
    if profile not in PREPROCESSING_PROFILES:
        raise ValueError(f"Unknown GazeLNN preprocessing profile: {profile}")
    if tensor.ndim < 3 or tensor.shape[-3] != 3 or not tensor.is_floating_point():
        raise ValueError("Expected a floating RGB tensor with channels at axis -3")
    if profile == "legacy_rgb01":
        return tensor.clone()
    shape = (1,) * (tensor.ndim - 3) + (3, 1, 1)
    mean = tensor.new_tensor(IMAGENET_MEAN).reshape(shape)
    std = tensor.new_tensor(IMAGENET_STD).reshape(shape)
    return (tensor - mean) / std


class GazePreprocessor:
    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, profile="imagenet"):
        self.spec = preprocessing_spec(image_size, profile)
        self.image_size = tuple(self.spec["image_size"])
        self.profile = profile

    def unit_tensor(self, image):
        import torch
        image = resize_rgb_image(image, self.image_size)
        array = np.array(image, dtype=np.uint8, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div(255)

    def __call__(self, image):
        return normalize_rgb_tensor(self.unit_tensor(image), self.profile)

    def __repr__(self):
        return f"GazePreprocessor(image_size={self.image_size}, profile={self.profile!r})"


def add_preprocessing_argument(parser):
    parser.add_argument("--preprocessing", choices=PREPROCESSING_PROFILES,
                        help="Default: checkpoint-declared profile, or ImageNet with an unknown-history warning. Use legacy_rgb01 only for known historical [0,1] training.")


def resolve_checkpoint_preprocessing(checkpoint, profile=None):
    """Use hash-verified declarations; never infer historical normalization from weights."""
    from checkpoint_provenance import read_training_record
    try:
        record = read_training_record(checkpoint)
    except ValueError:
        record = None
    constructor = ((record or {}).get("architecture") or {}).get("constructor", {})
    size = (constructor.get("image_h", 256), constructor.get("image_w", 384))
    recorded = (record or {}).get("input_preprocessing")
    known = isinstance(recorded, dict) and recorded.get("profile") in PREPROCESSING_PROFILES
    if known:
        expected = preprocessing_spec(size, recorded["profile"])
        if recorded != expected:
            raise ValueError("Checkpoint preprocessing declaration is incompatible with its architecture/current implementation")
        if profile is not None and profile != recorded["profile"]:
            raise ValueError("Requested preprocessing conflicts with checkpoint training; use its recorded profile")
        return dict(spec=expected, training_policy_verified=True, selection="checkpoint", note=None)
    if recorded and profile is None:
        raise ValueError("Checkpoint uses custom/unsupported preprocessing; choose an explicit --preprocessing for a diagnostic")
    return dict(spec=preprocessing_spec(size, profile or "imagenet"), training_policy_verified=False,
                selection="explicit_override" if profile else "default_unknown",
                note="Checkpoint training preprocessing is unknown; the selected inference profile does not reconstruct its history.")


def warn_unknown_preprocessing(resolved):
    if not resolved["training_policy_verified"]:
        warnings.warn(resolved["note"] + f" Active profile: {resolved['spec']['profile']}.", RuntimeWarning, stacklevel=2)


def preprocessing_implementation_sha256():
    from project_paths import file_sha256
    return file_sha256(Path(__file__))


def dataset_preprocessing(dataset):
    from torch.utils.data import Subset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return getattr(dataset, "input_preprocessing", None)


def training_preprocessing(train_loader, val_loader, model=None):
    training = dataset_preprocessing(train_loader.dataset)
    validation = dataset_preprocessing(val_loader.dataset)
    if training != validation:
        raise ValueError("Training and validation preprocessing must match")
    parameters = getattr(model, "hparams", {})
    if training and training.get("image_size") and "image_h" in parameters and "image_w" in parameters and \
            training["image_size"] != [parameters["image_h"], parameters["image_w"]]:
        raise ValueError("Dataset preprocessing size differs from model architecture")
    return training
