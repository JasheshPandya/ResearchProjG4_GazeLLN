"""Compatibility imports for the shared model; no separate DIEM architecture."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaze_model import CoordConv, GazeLLNArch
from gaze_losses import KLDTWLoss, _cost_matrix, soft_dtw

__all__ = ["CoordConv", "GazeLLNArch", "KLDTWLoss", "soft_dtw"]
