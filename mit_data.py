"""MIT datasets shared by scripted scanpath and KL-DTW evaluation."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from OSIE_Modeldata import _gaussian_heatmap, _pad_or_truncate, extract_scanpaths, OSIEDataset

class MITDataset(Dataset):
    """
    Dataset for loading MIT Low Resolution dataset from manifest.csv and .npy files.
    """
    def __init__(
        self,
        data_root: str,
        img_size=(256, 384),
        downsample: int = 8,
        min_len: int = 4,
        max_len: int = 8,
        sigma: float = 1.5,
        seed: int = 42,
    ):
        self.stimuli_dir = os.path.join(data_root, 'images')
        self.img_size = img_size
        self.hmap_size = (img_size[0] // downsample, img_size[1] // downsample)
        self.max_len = max_len
        self.sigma = sigma

        manifest_path = os.path.join(data_root, 'manifest.csv')
        df = pd.read_csv(manifest_path)

        self.samples = []

        for _, row in df.iterrows():
            img_name = os.path.basename(row['image_path'])
            npy_path = os.path.join(data_root, row['fixation_npy_path'])
            num_subjects = int(row['num_subjects'])
            try:
                fix_data = np.load(npy_path, allow_pickle=True)

                # fix_data is (N, 2) with ALL subjects' gaze data concatenated.
                # Split into per-subject blocks for proper scanpath evaluation.
                points_per_subj = len(fix_data) // num_subjects
                if points_per_subj == 0:
                    continue

                for s in range(num_subjects):
                    start_idx = s * points_per_subj
                    end_idx = start_idx + points_per_subj
                    subj_data = fix_data[start_idx:end_idx]

                    # Extract fixation events (handles both raw gaze and fixation-level data)
                    fixations = MITDataset._process_subject_block(subj_data, max_len)

                    if len(fixations) < min_len:
                        continue

                    xs = fixations[:, 0].tolist()
                    ys = fixations[:, 1].tolist()

                    # Δt: 0 for first fixation, 1.0 for rest (paper Section III-B:
                    # "during deployment, Δt is fixed to 1")
                    dts = [0.0] + [1.0] * (len(xs) - 1)

                    xs, mask = _pad_or_truncate(xs, max_len, pad_value=xs[-1])
                    ys, _    = _pad_or_truncate(ys, max_len, pad_value=ys[-1])
                    dts, _   = _pad_or_truncate(dts, max_len, pad_value=1.0)

                    self.samples.append({
                        'img_name': img_name,
                        'fix_x':    xs,
                        'fix_y':    ys,
                        'fix_dt':   dts,
                        'mask':     mask,
                    })
            except Exception as e:
                print(f"Skipping {npy_path}: {e}")

        if len(self.samples) == 0:
            raise ValueError("No scanpaths were loaded! The structure of fix_data does not match expectations.")

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"[MITDataset] Loaded {len(self.samples)} scanpath pairs.")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _process_subject_block(points, max_len):
        """Process a subject's gaze data block into a fixation scanpath.

        Handles both raw gaze samples (consecutive near-duplicates that need
        fixation extraction) and fixation-level data (already sparse).

        Args:
            points: numpy array of shape (N, 2) with gaze/fixation coordinates
            max_len: maximum number of fixations to return

        Returns:
            numpy array of shape (M, 2) where M <= max_len
        """
        if len(points) <= max_len:
            return points

        # Check median inter-point distance to determine data type:
        # Raw gaze samples → very small distances (< 5px)
        # Fixation-level data → larger distances (saccade amplitudes)
        sample_size = min(50, len(points) - 1)
        diffs = np.linalg.norm(np.diff(points[:sample_size + 1], axis=0), axis=1)
        median_diff = np.median(diffs)

        if median_diff < 5.0:
            # Raw gaze samples — extract fixation events using velocity threshold
            fixations = [points[0].copy()]
            current_center = points[0].copy()
            n_in_fix = 1

            for pt in points[1:]:
                dist = np.linalg.norm(pt - current_center)
                if dist > 20.0:  # Saccade threshold in pixels
                    fixations.append(pt.copy())
                    current_center = pt.copy()
                    n_in_fix = 1
                    if len(fixations) >= max_len:
                        break
                else:
                    # Update running average of current fixation center
                    n_in_fix += 1
                    current_center += (pt - current_center) / n_in_fix

            return np.array(fixations[:max_len])
        else:
            # Already fixation-level data — take first max_len
            return points[:max_len]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_H, img_W = self.img_size
        hmap_H, hmap_W = self.hmap_size

        img_path = os.path.join(self.stimuli_dir, sample['img_name'])
        img = Image.open(img_path).convert('RGB')
        orig_W, orig_H = img.size

        img_tensor = self.transform(img)

        heatmap_seq = []
        for i in range(self.max_len):
            if sample['mask'][i]:
                hmap = _gaussian_heatmap(
                    x=sample['fix_x'][i], y=sample['fix_y'][i],
                    img_w=orig_W, img_h=orig_H,
                    hmap_w=hmap_W, hmap_h=hmap_H,
                    sigma=self.sigma,
                )
            else:
                hmap = np.zeros((hmap_H, hmap_W), dtype=np.float32)
            heatmap_seq.append(hmap)

        heatmap_seq = torch.from_numpy(np.stack(heatmap_seq, axis=0))
        dt_seq = torch.tensor(sample['fix_dt'], dtype=torch.float32)
        dt_seq[0] = 0.0
        mask = torch.tensor(sample['mask'], dtype=torch.bool)
        fix_coords = torch.tensor(list(zip(sample['fix_x'], sample['fix_y'])), dtype=torch.float32)

        # Scale GT coordinates from original image space to model space (384×256)
        # so they are in the same coordinate system as model predictions
        scale_x = float(img_W) / float(orig_W)
        scale_y = float(img_H) / float(orig_H)
        fix_coords[:, 0] *= scale_x
        fix_coords[:, 1] *= scale_y

        return img_tensor, heatmap_seq, dt_seq, mask, fix_coords

class MITMatDataset(OSIEDataset):
    """Original MIT-notebook input: images/ plus eye/fixations.mat, no split."""

    def __init__(self, data_root, img_size=(256, 384), downsample=8, min_len=4, max_len=8, sigma=1.5, seed=42):
        self.stimuli_dir = os.path.join(data_root, "images")
        self.img_size = img_size
        self.hmap_size = (img_size[0] // downsample, img_size[1] // downsample)
        self.max_len, self.sigma = max_len, sigma
        entries = np.atleast_1d(sio.loadmat(Path(data_root) / "eye/fixations.mat", simplify_cells=True)["fixations"])
        self.samples = [sample for entry in entries for sample in extract_scanpaths(entry, min_len, max_len)]
        if not self.samples:
            raise ValueError("No MIT MAT scanpaths loaded")
        self.transform = transforms.Compose([
            transforms.Resize(img_size), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def build_mit_dataloader(data_root, batch_size=8, num_workers=0, dataset_format="manifest", limit_samples=None, **dataset_options):
    dataset_class = MITDataset if dataset_format == "manifest" else MITMatDataset
    if dataset_format not in ("manifest", "mat"):
        raise ValueError("MIT dataset format must be manifest or mat")
    dataset = dataset_class(data_root=data_root, **dataset_options)
    if limit_samples is not None:
        if limit_samples <= 0:
            raise ValueError("Sample limit must be positive")
        dataset = Subset(dataset, range(min(limit_samples, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                      pin_memory=torch.cuda.is_available())
