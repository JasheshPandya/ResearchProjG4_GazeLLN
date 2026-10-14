"""OSIE data and autoregressive training, converted from trainGazeLNN.

The architecture and loss are shared with all other project entry points.
Legacy imports of GazeLLNArch, CoordConv and KLDTWLoss remain supported.
"""

import os
import random
import time
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from project_paths import OSIE_CHECKPOINT
from gaze_model import GazeLLNArch, CoordConv, load_model_weights
from gaze_losses import KLDTWLoss, _cost_matrix, soft_dtw

def _gaussian_heatmap(
    x: float,
    y: float,
    img_w: int,
    img_h: int,
    hmap_w: int,
    hmap_h: int,
    sigma: float = 1.5,
) -> np.ndarray:
    """
    Place a Gaussian blob at pixel (x, y) on a downsampled heatmap grid.

    Args:
        x, y      : fixation location in *original* image pixel coordinates
        img_w/h   : original image size (before downsampling)
        hmap_w/h  : heatmap size (= img / downsample)
        sigma     : Gaussian std-dev in heatmap pixels

    Returns:
        hmap : (hmap_h, hmap_w) float32 array summing to 1
    """
    hmap = np.zeros((hmap_h, hmap_w), dtype=np.float32)

    # scale fixation coords to heatmap space
    xi = int(round(x * hmap_w / img_w))
    yi = int(round(y * hmap_h / img_h))

    xi = np.clip(xi, 0, hmap_w - 1)
    yi = np.clip(yi, 0, hmap_h - 1)

    hmap[yi, xi] = 1.0
    hmap = gaussian_filter(hmap, sigma=sigma)

    total = hmap.sum()
    if total > 0:
        hmap /= total

    return hmap

def _pad_or_truncate(
    seq: List,
    max_len: int,
    pad_value,
) -> Tuple[List, List[bool]]:
    """Truncate to max_len and return (padded_seq, mask).
    mask[i] = True  →  real fixation
    mask[i] = False →  padding (loss should ignore these)
    """
    real_len = min(len(seq), max_len)
    mask = [True] * real_len + [False] * (max_len - real_len)
    seq = seq[:real_len] + [pad_value] * (max_len - real_len)
    return seq, mask

def _center_gaussian(
    batch_size: int,
    hmap_h: int,
    hmap_w: int,
    device,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Gaussian blob at image center using the existing _gaussian_heatmap utility.
    Shape: (B, 1, hmap_h, hmap_w)
    """
    hmap = _gaussian_heatmap(
        x=hmap_w / 2,                   # center x in heatmap coords — pass as if img_w == hmap_w
        y=hmap_h / 2,                   # center y
        img_w=hmap_w,
        img_h=hmap_h,
        hmap_w=hmap_w,
        hmap_h=hmap_h,
        sigma=sigma,
    )
    hmap_t = torch.from_numpy(hmap).to(device)          # (hmap_h, hmap_w)

    return hmap_t.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).clone()

def extract_scanpaths(
    entry: Dict,
    min_len: int = 4,
    max_len: int = 8,
) -> List[Dict]:
    """
    Extract per-subject scanpaths from one OSIE entry.

    Returns a list of dicts, each with keys:
        img_name  : str
        fix_x     : list[float]  (length <= max_len)
        fix_y     : list[float]
        fix_dt    : list[float]  (durations in ms)
        mask      : list[bool]   (True = real, False = padding)
    """
    scanpaths = []
    img_name = entry['img']

    for subj in entry['subjects']:
        xs  = np.atleast_1d(np.array(subj['fix_x'],        dtype=np.float32)).tolist()
        ys  = np.atleast_1d(np.array(subj['fix_y'],        dtype=np.float32)).tolist()
        dts = np.atleast_1d(np.array(subj['fix_duration'], dtype=np.float32)).tolist()

        if len(xs) < min_len:
            continue  # discard short scanpaths per GazeLNN / tSPM-Net

        xs, mask = _pad_or_truncate(xs, max_len, pad_value=xs[-1])
        ys, _    = _pad_or_truncate(ys, max_len, pad_value=ys[-1])
        dts, _   = _pad_or_truncate(dts, max_len, pad_value=0.0)

        scanpaths.append({
            'img_name': img_name,
            'fix_x':    xs,
            'fix_y':    ys,
            'fix_dt':   dts,
            'mask':     mask,
        })

    return scanpaths

class OSIEDataset(Dataset):
    """
    PyTorch Dataset for OSIE scanpath prediction.

    Each item is one (image, scanpath) pair — one subject's viewing of one
    image.  With 700 images × ~15 subjects the dataset has ~10 500 items
    before filtering.

    Args:
        data_root   : path to the osie/ folder containing stimuli/ and eye/
        split       : 'train' | 'val' | 'test'
        img_size    : (H, W) to resize images to. GazeLNN uses (256, 384).
        downsample  : spatial downsampling factor for heatmaps.
                      heatmap size = (img_size[0]//ds, img_size[1]//ds)
        min_len     : discard scanpaths shorter than this
        max_len     : truncate/pad scanpaths to this length
        sigma       : Gaussian std-dev (in heatmap pixels) for fixation blobs
        seed        : random seed for the 80/10/10 image split
    """

    # 80 / 10 / 10  image-level split (consistent with the paper)
    SPLIT_RATIOS = {'train': 0.80, 'val': 0.10, 'test': 0.10}

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (256, 384),
        downsample: int = 8,
        min_len: int = 4,
        max_len: int = 8,
        sigma: float = 1.5,
        seed: int = 42,
    ):
        assert split in ('train', 'val', 'test')
        self.stimuli_dir = os.path.join(data_root, 'stimuli')
        self.img_size    = img_size          # (H, W)
        self.hmap_size   = (img_size[0] // downsample, img_size[1] // downsample)
        self.max_len     = max_len
        self.sigma       = sigma

        # ---- load fixations ------------------------------------------------
        mat_path = os.path.join(data_root, 'eye', 'fixations.mat')
        raw = sio.loadmat(mat_path, simplify_cells=True)
        all_entries = raw['fixations']          # length 700

        # ---- deterministic image-level split --------------------------------
        n_total = len(all_entries)
        indices = list(range(n_total))
        rng = random.Random(seed)
        rng.shuffle(indices)

        n_train = int(n_total * 0.80)
        n_val   = int(n_total * 0.10)

        if split == 'train':
            chosen = indices[:n_train]
        elif split == 'val':
            chosen = indices[n_train : n_train + n_val]
        else:
            chosen = indices[n_train + n_val:]

        # ---- flatten to (image, subject) scanpath pairs --------------------
        self.samples: List[Dict] = []
        for idx in chosen:
            entry = all_entries[idx]
            scanpaths = extract_scanpaths(entry, min_len=min_len, max_len=max_len)
            self.samples.extend(scanpaths)

        # ---- image transforms ----------------------------------------------
        self.transform = transforms.Compose([
            transforms.Resize(img_size),          # PIL Resize takes (H, W)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],       # ImageNet stats
                std =[0.229, 0.224, 0.225],
            ),
        ])

        print(
            f"[OSIEDataset] split={split:5s} | "
            f"images={len(chosen)} | scanpath pairs={len(self.samples)}"
        )


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int):
        sample   = self.samples[idx]
        img_H, img_W = self.img_size
        hmap_H, hmap_W = self.hmap_size

        #  load & transform image
        img_path = os.path.join(self.stimuli_dir, sample['img_name'])
        img = Image.open(img_path).convert('RGB')

        # original size needed to scale fixation coords
        orig_W, orig_H = img.size   # PIL gives (W, H)

        img_tensor = self.transform(img)   # (3, H, W)

        # ---- build heatmap sequence ----------------------------------------
        heatmap_seq = []
        for i in range(self.max_len):
            if sample['mask'][i]:
                hmap = _gaussian_heatmap(
                    x=sample['fix_x'][i],
                    y=sample['fix_y'][i],
                    img_w=orig_W,
                    img_h=orig_H,
                    hmap_w=hmap_W,
                    hmap_h=hmap_H,
                    sigma=self.sigma,
                )
            else:
                # padding step — zero heatmap
                hmap = np.zeros((hmap_H, hmap_W), dtype=np.float32)

            heatmap_seq.append(hmap)

        heatmap_seq = torch.from_numpy(
            np.stack(heatmap_seq, axis=0)
        )  # (T, hmap_H, hmap_W)

        # ---- fixation durations (∆t for CfC) --------------------------------
        dt_seq = torch.tensor(sample['fix_dt'], dtype=torch.float32)  # (T,)

        # first fixation ∆t = 0 per the paper
        dt_seq[0] = 0.0

        # ---- padding mask ---------------------------------------------------
        mask = torch.tensor(sample['mask'], dtype=torch.bool)  # (T,)

        return img_tensor, heatmap_seq, dt_seq, mask

def build_dataloaders(
    data_root: str,
    img_size: Tuple[int, int] = (256, 384),
    downsample: int = 8,
    min_len: int = 4,
    max_len: int = 8,
    sigma: float = 1.5,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Each batch yields:
        imgs         : (B, 3, H, W)       — normalised image tensor
        heatmap_seq  : (B, T, Hd, Wd)     — per-fixation Gaussian heatmaps
        dt_seq       : (B, T)             — fixation durations in ms
        padding_mask : (B, T)             — True where fixation is real
    """
    kwargs = dict(
        data_root  = data_root,
        img_size   = img_size,
        downsample = downsample,
        min_len    = min_len,
        max_len    = max_len,
        sigma      = sigma,
        seed       = seed,
    )

    train_ds = OSIEDataset(split='train', **kwargs)
    val_ds   = OSIEDataset(split='val',   **kwargs)
    test_ds  = OSIEDataset(split='test',  **kwargs)

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = True,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

def train_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one full pass over the training dataloader.

    Args:
        model     : GazeLLNArch instance
        loader    : training DataLoader
        optimizer : Adam optimizer
        criterion : KLDTWLoss instance
        device    : "cuda" or "cpu"

    Returns:
        avg_loss : float — mean loss over all batches
    """
    model.train()
    total_loss = 0.0

    # Adds a progress bar to see the batches processing
    for imgs, hmap_seq, dt_seq, mask in tqdm(loader, desc="Training", leave=False):

        # move everything to device
        imgs    = imgs.to(device)       # (B, 3, H, W)
        hmap_seq = hmap_seq.to(device)   # (B, T, hmap_H, hmap_W)  ← targets
        mask    = mask.to(device)       # (B, T)
        dt_seq = dt_seq.to(device)


        B  = imgs.shape[0]
        T  = hmap_seq.shape[1]
        hmap_H = hmap_seq.shape[2]
        hmap_W = hmap_seq.shape[3]

        with torch.no_grad():
            vis_features = model.extract_features(imgs)             # (B, 960)


        # initialise recurrent state
        # prev_hmap = torch.zeros(B, 1, hmap_H, hmap_W, device=device)
        prev_hmap = _center_gaussian(B, hmap_H, hmap_W, device=device, sigma = 1.5)    # (B, 1, hmap_H, hmap_W)
        hx        = None                                                # CfC hidden state
        # ts      = torch.ones(B, device=device)                        # fixed ts=1 per paper

        # autoregressive forward pass
        predictions = []

        for t in range(T):

            ts = dt_seq[:,t].view(-1, 1)
            out_hmap, hx = model(vis_features, prev_hmap, hx, ts)  # (B, 1, hmap_H, hmap_W)
            predictions.append(out_hmap.squeeze(1))
            prev_hmap = out_hmap.detach()

            # next step uses current prediction as history
            # detach to avoid backprop through the full sequence history
            prev_hmap = out_hmap.detach()     # (B, 1, hmap_H, hmap_W)

        # stack predictions: (B, T, hmap_H, hmap_W)
        predictions = torch.stack(predictions, dim=1)

        # compute loss
        loss = criterion(predictions, hmap_seq, mask)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device):
    """
    Runs one full pass over a validation or test dataloader.
    No gradient computation — inference only.

    Args:
        model     : GazeLLNArch instance
        loader    : val or test DataLoader
        criterion : KLDTWLoss instance
        device    : "cuda" or "cpu"

    Returns:
        avg_loss : float — mean loss over all batches
    """

    model.eval()
    with torch.no_grad():
        total_loss = 0.0

        for batch in tqdm(loader, desc="Validating", leave=False):
            imgs, hmap_seq, dt_seq, mask = batch[:4]

            imgs     = imgs.to(device)
            hmap_seq = hmap_seq.to(device)
            mask     = mask.to(device)
            dt_seq = dt_seq.to(device)

            B      = imgs.shape[0]
            T      = hmap_seq.shape[1]
            hmap_H = hmap_seq.shape[2]
            hmap_W = hmap_seq.shape[3]

            with torch.no_grad():
                vis_features = model.extract_features(imgs)  # (B, 960)

            # prev_hmap = torch.zeros(B, 1, hmap_H, hmap_W, device=device)
            prev_hmap = _center_gaussian(B, hmap_H, hmap_W, device=device, sigma = 1.5)
            hx        = None
            # ts      = torch.ones(B, device=device)

            predictions = []

            for t in range(T):
                ts = dt_seq[:, t].view(-1, 1)
                out_hmap, hx = model(vis_features, prev_hmap, hx, ts)
                predictions.append(out_hmap.squeeze(1))
                prev_hmap = out_hmap

            predictions = torch.stack(predictions, dim=1)
            loss = criterion(predictions, hmap_seq, mask)
            total_loss += loss.item()

        return total_loss / len(loader)

def train(
    model,
    train_loader,
    val_loader,
    device:      str   = "cuda",
    epochs:      int   = 100,
    lr:          float = 0.0001,
    patience:    int   = 20,
    gamma:       float = 0.1,
    checkpoint:  str   = OSIE_CHECKPOINT,
):
    """
    Full training loop with Adam optimiser, KL-DTW loss, and early stopping.

    Args:
        model        : GazeLLNArch instance
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        device       : "cuda" or "cpu"
        epochs       : maximum number of epochs (default 100)
        lr           : learning rate for Adam (default 0.0001)
        patience     : early stopping patience in epochs (default 20)
        gamma        : soft-DTW smoothing temperature (default 0.1)
        checkpoint   : path to save the best model weights

    Returns:
        history : dict with keys "train_loss" and "val_loss" (lists)
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model     = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = KLDTWLoss(gamma=gamma)

    best_val_loss    = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"Training on {device} | epochs={epochs} | lr={lr} | patience={patience}\n")

    for epoch in range(1, epochs + 1):

        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = eval_epoch(model, val_loader,   criterion, device)

        end_time = time.time()
        elapsed_time = end_time - start_time
        mins = int(elapsed_time // 60)
        secs = int(elapsed_time % 60)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch:03d}/{epochs} | time={mins}m {secs:02d}s | train={train_loss:.4f} | val={val_loss:.4f}", end="")

        # ---- check for improvement ----------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint)
            print("  ← saved best model", end="")
        else:
            epochs_no_improve += 1
            print(f"  (no improvement for {epochs_no_improve}/{patience} epochs)", end="")

        print()

        # ---- early stopping -----------------------------------------------
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            print(f"Best val loss: {best_val_loss:.4f}")
            break

    print("\nTraining complete.")
    print(f"Best model saved to: {checkpoint}")

    return history

def evaluate_test(model, test_loader, device="cuda", checkpoint=OSIE_CHECKPOINT):
    """
    Loads the best saved weights and evaluates on the test set.

    Returns:
        test_loss : float
    """
    load_model_weights(model, checkpoint, device)
    model = model.to(device)
    criterion = KLDTWLoss()

    test_loss = eval_epoch(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}")
    return test_loss

if __name__ == "__main__":
    from trainGazeLNN import main
    main()
