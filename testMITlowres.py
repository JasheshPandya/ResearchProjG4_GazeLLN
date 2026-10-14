"""MIT KL-DTW evaluation; supports manifest and original MAT data."""

from Evaluate_OSIE_model import main as evaluate_main
from OSIE_Modeldata import eval_epoch, evaluate_test
from mit_data import MITMatDataset as MITDataset, build_mit_dataloader
from gaze_model import GazeLLNArch, CoordConv
from gaze_losses import KLDTWLoss


def build_dataloaders(data_root, img_size=(256, 384), downsample=8, min_len=4,
                      max_len=8, sigma=1.5, batch_size=8, num_workers=4, seed=42,
                      dataset_format="mat"):
    """Retain the notebook helper with a working dataset-format selection."""
    return build_mit_dataloader(data_root, batch_size, num_workers, dataset_format,
        img_size=img_size, downsample=downsample, min_len=min_len, max_len=max_len, sigma=sigma, seed=seed)


def main(argv=None):
    return evaluate_main(argv, loss_only=True)


if __name__ == "__main__":
    main()
