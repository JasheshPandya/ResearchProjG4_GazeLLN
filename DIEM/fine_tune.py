"""
Fine-tuning loop for GazeLLNArch on continuous video (DIEM), building on
dataset.py's contiguous-chunk Dataset -- rewritten against the ACTUAL model in
modeldata.py and the real training conventions in trainGazeLNN.py (which
produced OSIE_model.pt), not an earlier draft. Concretely, this version
matches (and was verified against, see the chat writeup) the following:

  1. forward() takes precomputed `vis_features` (B, 960), not raw images --
     call model.extract_features(images) first.
  2. The model's output is ALREADY softmax-normalized internally (each
     sample sums to 1) and has a channel dim: (B, 1, Hg, Wg).
  3. hx=None and ts of shape (B,) are handled INSIDE forward() now.
  4. Training in trainGazeLNN.py is FREE-RUNNING / autoregressive, not
     teacher-forced: prev_hmap for step t+1 is the model's own prediction
     from step t (detached during training). Kept as the default here, with
     an opt-in teacher_forcing=True mode available.
  5. The autoregressive prior (prev_hmap before the first real prediction)
     is a small Gaussian centered on the heatmap grid (sigma=1.5) -- see
     _center_gaussian() below, ported directly from trainGazeLNN.py.
  6. train_epoch/eval_epoch in trainGazeLNN.py wrap `extract_features` in
     torch.no_grad() UNCONDITIONALLY -- the backbone is never updated
     through their entire OSIE run. Matched here via train_backbone=False
     by default.

This script extends their per-batch single-image feature-caching pattern to
the continuous-video equivalent: one batched extract_features() call over
all T distinct frames in a window.

Loss: ContinuousKLLoss replaces KLDTWLoss (continuous frames are already 1:1
time-aligned to ground truth, so DTW's alignment search is unnecessary),
updated to match the model's already-softmaxed output (no double softmax).

Cross-chunk continuation: VideoStateStore tracks the detached CfC hidden
state, previous heatmap and exact next-frame cursor by video name. Chunks
must be contiguous and non-overlapping; terminal padding is excluded from
feature extraction, recurrence and loss. Missing gaze labels mask loss only.
Batch and epoch losses weight valid frames, including short terminal chunks.
"""

from __future__ import annotations

import argparse
import math
import time
import random
from tqdm import tqdm
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __package__:
    from . import config as cfg
    from .dataset import build_datasets, VideoOrderedBatchSampler, validate_chunk_dataset
else:
    import config as cfg
    from dataset import build_datasets, VideoOrderedBatchSampler, validate_chunk_dataset
from gaze_model import GazeLLNArch, load_model_weights
from project_paths import OSIE_CHECKPOINT, DIEM_CHECKPOINT, OUTPUT_FILE_DIR
from DIEM.splits import training_split_binding, read_manifest
from checkpoint_provenance import TrainingRecorder, diem_loader_record, fingerprint, architecture_record
from run_exports import RunExport, export_training_results
from gaze_preprocessing import training_preprocessing


def _gaussian_heatmap(x, y, img_w, img_h, hmap_w, hmap_h, sigma=1.5):
    hmap = np.zeros((hmap_h, hmap_w), dtype=np.float32)
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


def _center_gaussian(batch_size, hmap_h, hmap_w, device, sigma=1.5):
    """Gaussian blob at the heatmap center. Shape: (B, 1, hmap_h, hmap_w).
    Ported from trainGazeLNN.py so the fine-tuning prior matches what
    OSIE.pt was actually trained with."""
    hmap = _gaussian_heatmap(
        x=hmap_w / 2, y=hmap_h / 2,
        img_w=hmap_w, img_h=hmap_h, hmap_w=hmap_w, hmap_h=hmap_h, sigma=sigma,
    )
    hmap_t = torch.from_numpy(hmap).to(device)
    return hmap_t.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).clone()


class ContinuousKLLoss(nn.Module):
    """Per-timestep masked KL divergence. predictions are assumed ALREADY
    softmax-normalized by the model -- do not apply softmax again here."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        B, T, H, W = predictions.shape
        # Float32 keeps epsilon nonzero under AMP, including zero-padded maps.
        pred = predictions.reshape(B, T, -1).float().clamp(min=self.eps)
        target = targets.reshape(B, T, -1).float().clamp(min=self.eps)
        pred = pred / pred.sum(dim=-1, keepdim=True)
        target = target / target.sum(dim=-1, keepdim=True)

        kl = (target * (target.log() - pred.log())).sum(dim=-1)  # (B, T)
        mask_f = mask.float()
        kl = kl * mask_f

        # Every valid frame has equal weight, including short terminal chunks.
        return kl.sum() / mask_f.sum().clamp(min=1)


class VideoStateStore:
    """Detached state after an exact video-frame prefix, with a next-frame cursor."""

    def __init__(self):
        self._store = {}

    def clear(self):
        self._store.clear()

    def gather(self, video_names, is_video_start, hidden_size, hmap_h, hmap_w, device, *, start_frames):
        if len(set(video_names)) != len(video_names):
            raise ValueError("A batch cannot contain multiple chunks from the same video")
        B = len(video_names)
        hx = torch.zeros(B, hidden_size, device=device)
        prev_hmap = _center_gaussian(B, hmap_h, hmap_w, device=device)
        for i, name in enumerate(video_names):
            start = int(start_frames[i])
            if bool(is_video_start[i]) != (start == 0):
                raise ValueError(f"{name}: is_video_start disagrees with start_frame={start}")
            state = self._store.get(name)
            if state is None:
                if start != 0:
                    raise ValueError(f"{name}: missing recurrent history before frame {start}; start the video at frame 0")
                continue
            if start != state["next_frame"]:
                raise ValueError(f"{name}: noncontiguous recurrent state: expected frame {state['next_frame']}, got {start}; overlap, gaps and replay are forbidden")
            hx[i] = state["hx"].to(device)
            prev_hmap[i] = state["prev_hmap"].to(device)
        return hx, prev_hmap

    def scatter(self, video_names, hx, last_prev_hmap, *, next_frames):
        for i, name in enumerate(video_names):
            self._store[name] = {
                "hx": hx[i].detach().clone(),
                "prev_hmap": last_prev_hmap[i].detach().clone(),
                "next_frame": int(next_frames[i]),
            }


def run_window(model, batch, state_store: VideoStateStore, device,
               ts_mode: str = "real", teacher_forcing: bool = False,
               train_backbone: bool = False):
    """Process contiguous real timesteps; tail padding never enters the backbone/state."""
    if ts_mode != "real":
        raise ValueError("Video models use elapsed seconds; ts_mode must be real")
    if "start_frame" not in batch or "sequence_length" not in batch:
        raise ValueError("Stateful chunks require start_frame and sequence_length metadata")
    frames = batch["frames"].to(device)
    heatmaps = batch["heatmaps"].to(device)
    valid = batch["valid"].to(device)
    delta_t = batch["delta_t"].to(device)
    is_start = batch["is_video_start"]
    video_names = batch["video_name"]
    starts = torch.as_tensor(batch["start_frame"], device=device)
    lengths = torch.as_tensor(batch["sequence_length"], device=device)

    B, T = frames.shape[:2]
    if B == 0 or T == 0:
        raise ValueError("Recurrent chunks must have nonempty batch and time dimensions")
    if starts.shape != (B,) or lengths.shape != (B,) or starts.is_floating_point() or lengths.is_floating_point() or \
            (starts < 0).any() or (lengths < 1).any() or (lengths > T).any():
        raise ValueError("Chunks need nonnegative integer starts and lengths in [1,T]")
    if len(video_names) != B or len(is_start) != B:
        raise ValueError("Chunk video/start metadata does not match the batch size")
    timestep_mask = torch.arange(T, device=device)[None] < lengths[:, None]
    if (valid & ~timestep_mask).any():
        raise ValueError("Padded timesteps cannot have valid gaze targets")
    if not torch.isfinite(delta_t[timestep_mask]).all() or (delta_t[timestep_mask] <= 0).any():
        raise ValueError("Real frame timesteps must be positive finite elapsed seconds")
    hidden_size = model.hparams.hidden_size
    Hg, Wg = model.hmap_h, model.hmap_w
    hx, prev_hmap = state_store.gather(video_names, is_start, hidden_size, Hg, Wg, device, start_frames=starts)

    # Exclude padding before feature extraction too: black padding must not
    # change BatchNorm statistics or influence real-frame features in training.
    real_frames = frames[timestep_mask]
    if train_backbone:
        real_features = model.extract_features(real_frames)
    else:
        with torch.no_grad():
            real_features = model.extract_features(real_frames)
    features = real_features.new_zeros((B, T, real_features.shape[-1]))
    features[timestep_mask] = real_features

    predictions = []
    for t in range(T):
        active = torch.nonzero(timestep_mask[:, t], as_tuple=False).flatten()
        if not len(active):
            predictions.append(torch.zeros_like(predictions[-1]))
            continue
        out_hmap, next_hx = model(features[active, t], prev_hmap[active], hx[active], delta_t[active, t])
        full_prediction = out_hmap.new_zeros((B, 1, Hg, Wg)).index_copy(0, active, out_hmap)
        predictions.append(full_prediction.squeeze(1))
        hx = hx.index_copy(0, active, next_hx.to(hx.dtype))
        next_prev = out_hmap.detach()
        if teacher_forcing:
            # Invalid annotations are missing labels, not missing timesteps.
            next_prev = torch.where(valid[active, t, None, None, None], heatmaps[active, t, None], next_prev)
        prev_hmap = prev_hmap.index_copy(0, active, next_prev.to(prev_hmap.dtype))

    state_store.scatter(video_names, hx, prev_hmap, next_frames=starts + lengths)
    return torch.stack(predictions, dim=1), heatmaps, valid


def train_epoch(model, loader, optimizer, criterion, device, scaler=None, ts_mode="real",
                 teacher_forcing=False, train_backbone=False, grad_clip=1.0, epoch_idx=0):
    model.train()
    state_store = VideoStateStore()
    total_loss, n_valid_frames = 0.0, 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch_idx}", leave=False)

    for batch in pbar:
        with torch.amp.autocast('cuda', enabled=scaler is not None and scaler.is_enabled()):
            predictions, targets, valid = run_window(
                model, batch, state_store, device, ts_mode, teacher_forcing, train_backbone
            )
            if not valid.any():
                continue

            loss = criterion(predictions, targets, valid)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        count = int(valid.sum())
        total_loss += loss.item() * count
        n_valid_frames += count

        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return total_loss / max(n_valid_frames, 1)


def eval_epoch(model, loader, criterion, device, scaler=None, ts_mode="real", teacher_forcing=False, epoch_idx=0):
    model.eval()
    state_store = VideoStateStore()
    total_loss, n_valid_frames = 0.0, 0

    with torch.no_grad():

        pbar = tqdm(loader, desc=f"Eval Epoch {epoch_idx}", leave=False)

        for batch in pbar:
            with torch.amp.autocast('cuda', enabled=scaler is not None and scaler.is_enabled()):
                predictions, targets, valid = run_window(
                    model, batch, state_store, device, ts_mode, teacher_forcing, train_backbone=False
                )
                if not valid.any():
                    continue
                loss = criterion(predictions, targets, valid)

            count = int(valid.sum())
            total_loss += loss.item() * count
            n_valid_frames += count

            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return total_loss / max(n_valid_frames, 1)


def fine_tune(
    model,
    train_loader,
    val_loader,
    device: str = "cuda",
    epochs: int = 100,
    backbone_lr: float = 1e-6,
    head_lr: float = 1e-4,
    patience: int = 20,
    grad_clip: float = 1.0,
    ts_mode: str = "real",
    teacher_forcing: bool = False,
    train_backbone: bool = False,
    unfreeze_backbone_after_epoch: Optional[int] = None,
    pretrained_checkpoint: Optional[str] = None,
    checkpoint: str = DIEM_CHECKPOINT,
    use_amp: bool = True,
    split_manifest: Optional[str] = None,
    seed: Optional[int] = None,
    run_settings=None,
):
    validate_chunk_dataset(train_loader.dataset)
    validate_chunk_dataset(val_loader.dataset)
    manifest_path = Path(split_manifest or getattr(train_loader.dataset, "split_manifest", cfg.SPLIT_MANIFEST)).resolve()
    train_names = [e.name for e in train_loader.dataset.entries]
    val_names = [e.name for e in val_loader.dataset.entries]
    # Validate provenance before any optimization; final record also binds the exact saved weights.
    manifest = read_manifest(manifest_path)
    split_binding = training_split_binding(manifest_path, train_names, val_names)
    if ts_mode != "real":
        raise ValueError("Video training requires elapsed-time ts_mode=real")
    if pretrained_checkpoint is not None:
        load_model_weights(model, pretrained_checkpoint, device)
        print(f"Loaded pretrained checkpoint {pretrained_checkpoint}")

    checkpoint = Path(checkpoint).expanduser().resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    criterion = ContinuousKLLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device == "cuda")

    backbone_params = list(model.feature_extractor.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("feature_extractor.")]
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": other_params, "lr": head_lr},
    ], fused=(device == "cuda"))
    recorder = TrainingRecorder(model, checkpoint, dataset="DIEM", split_binding=split_binding,
        input_preprocessing=training_preprocessing(train_loader, val_loader, model),
        splits=dict(unit="video", manifest=fingerprint(manifest_path), assignments=manifest,
                    train=diem_loader_record(train_loader), val=diem_loader_record(val_loader)),
        hyperparameters=dict(epochs=epochs, backbone_lr=backbone_lr, head_lr=head_lr, patience=patience,
            grad_clip=grad_clip, ts_mode=ts_mode, fps=cfg.GAZE_FPS, teacher_forcing=teacher_forcing,
            train_backbone=train_backbone, unfreeze_backbone_after_epoch=unfreeze_backbone_after_epoch,
            device=str(device), amp_requested=use_amp, amp_enabled=scaler.is_enabled(), seed=seed,
            seed_policy="Entry point seeds Python/NumPy/Torch before dataset/model construction; direct callers supply their declared seed and manage RNG setup",
            optimizer=dict(name="torch.optim.Adam", defaults=optimizer.defaults),
            loss=dict(name="DIEM.fine_tune.ContinuousKLLoss", eps=criterion.eps), run_settings=run_settings,
            preprocessing={name: value for name, value in vars(cfg).items()
                           if name.isupper() and isinstance(value, (str, int, float, dict, Path))},
            feedback="teacher forcing" if teacher_forcing else "autoregressive detached previous prediction",
            recurrent_state="reset at video start; detach and carry only to the exact next frame; exclude tail padding",
            temporal_policy=cfg.RECURRENT_STATE_POLICY,
            loss_pooling="mean over valid frames in each batch; epoch loss weighted by valid-frame count",
            initial_heatmap="center Gaussian, sigma=1.5 grid pixels"),
        sources=[Path(__file__), ROOT_DIR / "gaze_model.py", ROOT_DIR / "video_timing.py",
                 ROOT_DIR / "checkpoint_provenance.py", ROOT_DIR / "gaze_preprocessing.py", Path(cfg.__file__),
                 ROOT_DIR / "DIEM/dataset.py", ROOT_DIR / "DIEM/splits.py", ROOT_DIR / "DIEM/build_heatmaps.py"])

    effective_train_backbone = train_backbone if unfreeze_backbone_after_epoch is None else False
    for p in backbone_params:
        p.requires_grad = effective_train_backbone
    print(f"Backbone trainable: {effective_train_backbone}"
          + (f" (unfreezing after epoch {unfreeze_backbone_after_epoch})" if unfreeze_backbone_after_epoch else ""))

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"Fine-tuning on {device} | epochs={epochs} | backbone_lr={backbone_lr} "
          f"| head_lr={head_lr} | patience={patience} | ts_mode={ts_mode} "
          f"| teacher_forcing={teacher_forcing}\n")

    with recorder.run():
        for epoch in range(1, epochs + 1):
            if unfreeze_backbone_after_epoch is not None and epoch == unfreeze_backbone_after_epoch + 1:
                effective_train_backbone = train_backbone
                for p in backbone_params:
                    p.requires_grad = effective_train_backbone
                print(f"[epoch {epoch}] backbone trainable set to {effective_train_backbone}")

            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler,
                                      ts_mode, teacher_forcing, effective_train_backbone, grad_clip, epoch)
            val_loss = eval_epoch(model, val_loader, criterion, device, scaler, ts_mode, teacher_forcing, epoch)
            dt = time.time() - t0
            recorder.epoch(epoch, train_loss, val_loss, dt, learning_rates=[group["lr"] for group in optimizer.param_groups], backbone_trainable=effective_train_backbone)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(f"Epoch {epoch:03d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f} | {dt:.1f}s", end="")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                recorder.save_best(model)
                print("  <- saved best model", end="")
            else:
                epochs_no_improve += 1
                print(f"  (no improvement for {epochs_no_improve}/{patience} epochs)", end="")
            print()

            recorder.update()

            if epochs_no_improve >= patience:
                recorder.record["status"] = "early_stopped"
                print(f"\nEarly stopping triggered at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break

    print(f"\nFine-tuning complete. Best model saved to: {checkpoint}")
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune GazeLNN on contiguous DIEM video chunks")
    parser.add_argument("--processed-root", type=Path, default=cfg.OUTPUT_ROOT)
    parser.add_argument("--frames-root", type=Path, help="Default: <processed-root>/frames")
    parser.add_argument("--split-manifest", type=Path, default=cfg.SPLIT_MANIFEST)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--pretrained-checkpoint", type=Path, default=OSIE_CHECKPOINT)
    initialization.add_argument("--from-scratch", action="store_true", help="Do not initialize from OSIE weights")
    parser.add_argument("--no-pretrained-backbone", action="store_true", help="Random backbone for --from-scratch; avoids downloads")
    parser.add_argument("--checkpoint", type=Path, default=DIEM_CHECKPOINT)
    parser.add_argument("--output-dir", "--out-dir", type=Path, help="Default: shared output/diem_training/<checkpoint stem>")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--unfreeze-backbone-after-epoch", type=int)
    parser.add_argument("--teacher-forcing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", dest="use_amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--videos-per-batch", type=int, default=8)
    parser.add_argument("--window-len", type=int, default=cfg.WINDOW_LEN_FRAMES,
                        help="Contiguous chunk length; stride always equals this value")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocessing", choices=["imagenet", "legacy_rgb01"], default="imagenet")
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.videos_per_batch, args.window_len) <= 0 or args.num_workers < 0:
        parser.error("Epochs, patience, videos per batch and window length must be positive; workers nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in (args.backbone_lr, args.head_lr, args.grad_clip)):
        parser.error("Learning rates and gradient clip must be finite and positive")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be in [0, 2**32)")
    if args.unfreeze_backbone_after_epoch is not None and args.unfreeze_backbone_after_epoch < 0:
        parser.error("Unfreeze delay must be nonnegative")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu")
    for key in ("processed_root", "split_manifest", "checkpoint", "pretrained_checkpoint"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    args.frames_root = (args.frames_root or args.processed_root / "frames").expanduser().resolve()
    args.output_dir = (args.output_dir or OUTPUT_FILE_DIR / "diem_training" / args.checkpoint.stem).expanduser().resolve()
    pretrained_checkpoint = None if args.from_scratch else args.pretrained_checkpoint
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    configuration = dict(arguments=vars(args), resolved_device=device,
                         initialization_checkpoint=pretrained_checkpoint,
                         pretrained_backbone=pretrained_checkpoint is None and not args.no_pretrained_backbone,
                         window_stride=args.window_len, ts_mode="real",
                         diem_settings={name: value for name, value in vars(cfg).items() if name.isupper()})
    try:
        run = RunExport(args.output_dir, "diem_training", configuration,
                        sources=[Path(__file__), ROOT_DIR / "DIEM/config.py", ROOT_DIR / "DIEM/dataset.py",
                                 ROOT_DIR / "DIEM/frame_integrity.py", ROOT_DIR / "gaze_model.py",
                                 ROOT_DIR / "gaze_preprocessing.py", ROOT_DIR / "run_exports.py"])
        with run.run():
            train_ds, val_ds, test_ds = build_datasets(processed_root=args.processed_root,
                frames_root=args.frames_root, split_manifest=args.split_manifest,
                preprocessing=args.preprocessing, window_len=args.window_len)
            if not len(train_ds) or not len(val_ds):
                raise ValueError("DIEM train and validation partitions must both contain real frames")
            loaders = [DataLoader(dataset, batch_sampler=VideoOrderedBatchSampler(dataset,
                videos_per_batch=args.videos_per_batch, shuffle_videos=i == 0, seed=args.seed),
                num_workers=args.num_workers, pin_memory=device == "cuda")
                for i, dataset in enumerate((train_ds, val_ds))]
            model = GazeLLNArch(image_h=cfg.MODEL_INPUT_HEIGHT, image_w=cfg.MODEL_INPUT_WIDTH, S=cfg.S,
                               pretrained_backbone=configuration["pretrained_backbone"])
            run.update(architecture=architecture_record(model),
                       input_preprocessing=train_ds.input_preprocessing,
                       amp_enabled=args.use_amp and device == "cuda",
                       train=diem_loader_record(loaders[0]), val=diem_loader_record(loaders[1]),
                       test_videos=[entry.name for entry in test_ds.entries],
                       split_manifest=fingerprint(args.split_manifest))
            history = fine_tune(model, *loaders, device=device, epochs=args.epochs,
                backbone_lr=args.backbone_lr, head_lr=args.head_lr, patience=args.patience,
                grad_clip=args.grad_clip, teacher_forcing=args.teacher_forcing,
                train_backbone=args.train_backbone, unfreeze_backbone_after_epoch=args.unfreeze_backbone_after_epoch,
                pretrained_checkpoint=pretrained_checkpoint, checkpoint=args.checkpoint,
                use_amp=args.use_amp, split_manifest=args.split_manifest, seed=args.seed,
                run_settings=dict(**vars(args), run_config=str(run.path), run_id=run.record["run_id"]))
            export_training_results(run, args.checkpoint, history)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"DIEM training failed: {exc}\n")
    print(f"Checkpoint: {args.checkpoint}\nRun configuration: {run.path}\nResults: {run.output_dir / 'results.json'}")
    return history


if __name__ == "__main__":
    main()
