"""Working-directory independent project data/checkpoint defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OSIE_DATA_ROOT = PROJECT_ROOT / "OSIE"
MIT_DATA_ROOT = PROJECT_ROOT / "mit_lowres_highres_only"
OSIE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "OSIE_model.pt"
DIEM_CHECKPOINT = PROJECT_ROOT / "DIEM" / "DIEM_model.pt"
