#!/usr/bin/env python3
"""
Unified SSL pretraining script for Lfp2vec multi-backbone benchmark.

Supports: wav2vec2, wavlm, patchtst, moment, ttm
Usage:
    # Standalone
    python train_ssl.py --backbone wavlm --config configs/wavlm.yaml

    # Via Ray Tune (called by run_benchmark.sbatch)
    # Launched internally by Ray — see ssl_trial_trainable()
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from sklearn.metrics import balanced_accuracy_score

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Also add Alphabrain_staging for CompactDataset import
ALPHABRAIN_ROOT = PROJECT_ROOT.parent / "Alphabrain_staging"
sys.path.insert(0, str(ALPHABRAIN_ROOT))

from backbones import get_backbone
from utils.preprocessing import preprocess_batch

try:
    import wandb
except ImportError:
    wandb = None

try:
    from ray import tune
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


def _in_ray_tune_session() -> bool:
    """Check if we're actually inside an active Ray Tune trial (not just ray installed)."""
    if not HAS_RAY:
        return False
    return "TUNE_ORIG_WORKING_DIR" in os.environ


# ===================== Region labels ===================== #

REGION_ID_TO_NAME = {
    0: 'VIS', 1: 'HPF', 2: 'OLF', 3: 'AMY', 4: 'CTX-STR', 5: 'TH-GEN', 6: 'VIS-TH',
    7: 'AUD-TH', 8: 'SM-TH', 9: 'RT-ZI', 10: 'MB-Tect', 11: 'MB-Teg', 12: 'Misc'
}


# ===================== Helpers ===================== #

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===================== W&B Rich Logging Helpers ===================== #

def _log_region_bar_chart(region_stats, epoch, region_id_to_name=None):
    """Log region loss bar chart to W&B."""
    if wandb is None or wandb.run is None:
        return
    if not region_stats:
        return

    if region_id_to_name is None:
        region_id_to_name = REGION_ID_TO_NAME

    region_data = []
    for rid, rname in region_id_to_name.items():
        key = f"loss_{rname}"
        if key in region_stats and np.isfinite(region_stats[key]):
            region_data.append([rname, region_stats[key]])
    if region_data:
        table = wandb.Table(data=region_data, columns=["Region", "Loss"])
        wandb.log({
            "region_losses/bar": wandb.plot.bar(
                table, "Region", "Loss",
                title=f"Region Losses (Epoch {epoch})",
            ),
            "epoch": epoch,
        })


def _quick_linear_probe(backbone, train_loader, val_loader, device, epoch_count=50,
                        num_classes=None, max_probe_samples=0):
    # num_classes is required — kept as kwarg for call-site clarity
    """Fast linear probe to measure SSL representation quality.

    Extracts frozen embeddings then trains a linear classifier for `epoch_count` epochs.
    Filters out samples with label_id == -1 (atlas samples without label at this depth).

    Args:
        num_classes: Classifier output size from the global effective label map
            (dataset.num_effective_classes). Labels are used directly without remapping.
            Required — raises RuntimeError if None.
        max_probe_samples: If >0, extract only this many samples from train_loader
            and split 80/20 into train/val. Skips val_loader entirely. Much faster
            for Ray Tune where we only need relative ranking between trials.
            0 = full extraction from both loaders (original behavior).

    Returns (best_val_acc, best_val_bal_acc, best_val_preds, val_labels, label_remap)
    where label_remap maps original_label_id -> classifier id.
    """
    backbone.eval()
    fast_mode = max_probe_samples > 0

    collected = 0

    with torch.no_grad():
        if fast_mode:
            # Fast mode: extract only from train_loader, split 80/20 later
            all_embs, all_labels = [], []
            for batch in train_loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                waveforms, batch_labels = batch[0], batch[1]
                if isinstance(batch_labels, str):
                    continue
                waveforms = waveforms.float().to(device, non_blocking=True)
                waveforms = preprocess_batch(waveforms, backbone.needs_resampling)
                pooled = backbone.encode(waveforms)
                all_embs.append(pooled)
                all_labels.append(batch_labels if torch.is_tensor(batch_labels)
                                  else torch.tensor(batch_labels))
                collected += waveforms.shape[0]
                if collected >= max_probe_samples:
                    break

            if not all_embs:
                log("[LinearProbe] No embeddings extracted, skipping")
                return None

            all_embs = torch.cat(all_embs)
            all_labels = torch.cat(all_labels).long().to(device)

            # Filter out label_id == -1
            valid = all_labels >= 0
            if not valid.all():
                all_embs = all_embs[valid]
                all_labels = all_labels[valid]

            if len(all_embs) < 10:
                log("[LinearProbe] Too few valid samples, skipping")
                return None

            # 80/20 split on GPU
            n = len(all_embs)
            perm = torch.randperm(n, device=device)
            split = int(n * 0.8)
            train_embs, val_embs = all_embs[perm[:split]], all_embs[perm[split:]]
            train_labels, val_labels = all_labels[perm[:split]], all_labels[perm[split:]]
        else:
            # Full mode: extract from both loaders separately
            train_embs, train_labels = [], []
            val_embs, val_labels = [], []
            for loader, embs, labs in [
                (train_loader, train_embs, train_labels),
                (val_loader, val_embs, val_labels),
            ]:
                for batch in loader:
                    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                        continue
                    waveforms, batch_labels = batch[0], batch[1]
                    if isinstance(batch_labels, str):
                        continue
                    waveforms = waveforms.float().to(device, non_blocking=True)
                    waveforms = preprocess_batch(waveforms, backbone.needs_resampling)
                    pooled = backbone.encode(waveforms)
                    embs.append(pooled)
                    labs.append(batch_labels if torch.is_tensor(batch_labels)
                                else torch.tensor(batch_labels))

            if not train_embs or not val_embs:
                log("[LinearProbe] No embeddings extracted, skipping")
                return None

            train_embs = torch.cat(train_embs)
            train_labels = torch.cat(train_labels).long().to(device)
            val_embs = torch.cat(val_embs)
            val_labels = torch.cat(val_labels).long().to(device)

            # Filter out label_id == -1
            train_valid = train_labels >= 0
            val_valid = val_labels >= 0
            if not train_valid.all() or not val_valid.all():
                train_embs = train_embs[train_valid]
                train_labels = train_labels[train_valid]
                val_embs = val_embs[val_valid]
                val_labels = val_labels[val_valid]

    if len(train_embs) == 0 or len(val_embs) == 0:
        log("[LinearProbe] Empty split after filtering, skipping")
        return None

    log(f"[LinearProbe] train={len(train_embs)}, val={len(val_embs)}, "
        f"hidden_dim={train_embs.shape[1]}{' (fast)' if fast_mode else ''}")

    # Labels are already contiguous 0..N-1 from global effective map.
    if num_classes is None:
        raise RuntimeError(
            "num_classes must be provided (from global effective label map). "
            "Pass num_classes=dataset.num_effective_classes."
        )
    label_remap = {i: i for i in range(num_classes)}  # identity

    # Train linear classifier — all on GPU
    hidden_dim = train_embs.shape[1]
    classifier = nn.Linear(hidden_dim, num_classes).to(device)
    opt = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_val_bal_acc = 0.0
    best_val_preds = None
    for ep in range(epoch_count):
        classifier.train()
        indices = torch.randperm(len(train_embs), device=device)
        for start in range(0, len(train_embs), 512):
            idx = indices[start:start + 512]
            logits = classifier(train_embs[idx])
            loss = criterion(logits, train_labels[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_embs)
            val_preds = val_logits.argmax(1)
            val_acc = (val_preds == val_labels).float().mean().item()
            # Balanced accuracy on GPU — no CPU transfer
            per_class_correct = torch.zeros(num_classes, device=device)
            per_class_total = torch.zeros(num_classes, device=device)
            per_class_total.scatter_add_(0, val_labels, torch.ones_like(val_labels, dtype=torch.float32))
            per_class_correct.scatter_add_(0, val_labels, (val_preds == val_labels).float())
            present = per_class_total > 0
            per_class_acc = torch.where(present, per_class_correct / per_class_total, torch.zeros_like(per_class_total))
            val_bal_acc = per_class_acc[present].mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_bal_acc = val_bal_acc
                best_val_preds = val_preds.clone()

    log(f"[LinearProbe] Best val accuracy: {best_val_acc*100:.2f}% "
        f"bal_acc: {best_val_bal_acc*100:.2f}% ({num_classes} classes)")
    return best_val_acc, best_val_bal_acc, best_val_preds, val_labels, label_remap


# ===================== Training loop ===================== #

def get_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """Compute gradient norm across all parameters (for monitoring)."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += float(p.grad.detach().data.norm(norm_type).item() ** norm_type)
    return total_norm ** (1.0 / norm_type)


def ssl_train_epoch(
    backbone: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip_norm: float = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> tuple:
    """One epoch of SSL pretraining — backbone-agnostic.

    Returns:
        (avg_loss, avg_grad_norm) tuple
    """
    backbone.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    n_batches = 0

    for step, batch in enumerate(dataloader):
        input_values = batch[0].float().to(device, non_blocking=True)
        coords = batch[2].float().to(device, non_blocking=True) if len(batch) > 2 else None
        input_values = preprocess_batch(input_values, backbone.needs_resampling)

        optimizer.zero_grad(set_to_none=True)

        # Pass coords for XYZ/dist auxiliary loss (only if backbone supports it)
        needs_coords = coords is not None and (hasattr(backbone, "xyz_lambda") or hasattr(backbone, "dist_lambda") or hasattr(backbone, "dist_pred_head") or getattr(backbone, "distpred_lambda", 0) > 0)
        ssl_kwargs = {"coords": coords} if needs_coords else {}
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = backbone.ssl_forward(input_values, **ssl_kwargs)
        else:
            loss = backbone.ssl_forward(input_values, **ssl_kwargs)

        if loss is None or not torch.isfinite(loss):
            log(f"[WARN] Non-finite loss at step {step}, skipping")
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip_norm)
            optimizer.step()

        # Track grad norm AFTER clipping but BEFORE scheduler step
        total_grad_norm += get_grad_norm(backbone)

        # Step the LR scheduler per-batch (warmup operates per step)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_grad_norm = total_grad_norm / max(n_batches, 1)
    return avg_loss, avg_grad_norm


@torch.no_grad()
def ssl_validate_epoch(
    backbone: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    """One epoch of SSL validation — backbone-agnostic."""
    backbone.eval()
    total_loss = 0.0
    n_batches = 0

    for input_values, _labels in dataloader:
        input_values = input_values.float().to(device, non_blocking=True)
        input_values = preprocess_batch(input_values, backbone.needs_resampling)

        if use_amp:
            with torch.cuda.amp.autocast():
                loss = backbone.ssl_forward(input_values)
        else:
            loss = backbone.ssl_forward(input_values)

        if loss is None or not torch.isfinite(loss):
            continue

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def ssl_validate_epoch_with_region_stats(
    backbone: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
    region_id_to_name: Optional[Dict[int, str]] = None,
) -> tuple:
    """One epoch of SSL validation with per-region loss tracking.

    Uses the same SSL loss but also groups by region ID to detect
    whether the model learns uniformly across brain regions.

    Args:
        region_id_to_name: Custom mapping from label int → name.
            Falls back to REGION_ID_TO_NAME (13-class) if None.

    Returns:
        (avg_loss, region_stats_dict) tuple
    """
    if region_id_to_name is None:
        region_id_to_name = REGION_ID_TO_NAME

    backbone.eval()
    total_loss = 0.0
    n_batches = 0

    # Accumulate losses by region/bin
    region_losses: dict = {}

    for batch in dataloader:
        input_values = batch[0].float().to(device, non_blocking=True)
        region_ids = batch[1] if len(batch) > 1 else None
        coords = batch[2].float().to(device, non_blocking=True) if len(batch) > 2 else None
        input_values = preprocess_batch(input_values, backbone.needs_resampling)

        needs_coords = coords is not None and (hasattr(backbone, "xyz_lambda") or hasattr(backbone, "dist_lambda") or hasattr(backbone, "dist_pred_head") or getattr(backbone, "distpred_lambda", 0) > 0)
        ssl_kwargs = {"coords": coords} if needs_coords else {}
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = backbone.ssl_forward(input_values, **ssl_kwargs)
        else:
            loss = backbone.ssl_forward(input_values, **ssl_kwargs)

        if loss is None or not torch.isfinite(loss):
            continue

        batch_loss = loss.item()
        total_loss += batch_loss
        n_batches += 1

        # Per-sample loss approximation: broadcast batch loss to all samples
        if torch.is_tensor(region_ids):
            for rid in region_ids:
                rid_int = int(rid.item())
                if rid_int < 0:
                    continue  # skip -1 (no label at this depth)
                if rid_int not in region_losses:
                    region_losses[rid_int] = []
                region_losses[rid_int].append(batch_loss)

    avg_loss = total_loss / max(n_batches, 1)

    # Compute region statistics
    region_stats: dict = {}
    region_means: list = []

    for rid in sorted(region_losses.keys()):
        region_name = region_id_to_name.get(rid, f"bin_{rid}")
        if region_losses[rid]:
            mean_loss = float(np.mean(region_losses[rid]))
            region_stats[f"loss_{region_name}"] = mean_loss
            region_stats[f"n_samples_{region_name}"] = len(region_losses[rid])
            region_means.append(mean_loss)
        else:
            region_stats[f"loss_{region_name}"] = float("nan")
            region_stats[f"n_samples_{region_name}"] = 0

    if region_means:
        mean_of_means = float(np.mean(region_means))
        std_of_means = float(np.std(region_means))
        min_loss = float(np.min(region_means))
        max_loss = float(np.max(region_means))

        region_stats["region_loss_cv"] = std_of_means / mean_of_means if mean_of_means > 0 else float("nan")
        region_stats["region_loss_max_min_ratio"] = max_loss / min_loss if min_loss > 0 else float("nan")

        hardest_idx = int(np.argmax(region_means))
        easiest_idx = int(np.argmin(region_means))
        region_stats["hardest_region_id"] = hardest_idx
        region_stats["easiest_region_id"] = easiest_idx
        region_stats["hardest_region_loss"] = max_loss
        region_stats["easiest_region_loss"] = min_loss
    else:
        region_stats["region_loss_cv"] = float("nan")
        region_stats["region_loss_max_min_ratio"] = float("nan")

    return avg_loss, region_stats


# ===================== Detailed logging variants ===================== #

def _log_reconstruction_wandb(predicted, target, mask_bool, prefix, epoch):
    """Log reconstruction visualization to W&B.

    For 1 sample: reduces 512-dim to L2 norm along feature dim → [1500] time series,
    then plots target vs predicted with mask overlay.

    Args:
        predicted: [B, 1500, D] decoder output (detached)
        target: [B, 1500, D] teacher target (detached)
        mask_bool: [B, 1500] bool
        prefix: "train" or "val"
        epoch: current epoch number
    """
    if wandb is None or wandb.run is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Take first sample
        pred_norm = predicted[0].float().norm(dim=-1).cpu().numpy()   # [1500]
        tgt_norm = target[0].float().norm(dim=-1).cpu().numpy()       # [1500]
        mask_1d = mask_bool[0].cpu().numpy().astype(float)            # [1500]

        fig, ax = plt.subplots(figsize=(14, 4))
        t = np.arange(len(tgt_norm))

        # Shade masked regions
        mask_alpha = mask_1d * 0.15
        for i in range(len(mask_1d)):
            if mask_1d[i] > 0:
                ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.12)

        ax.plot(t, tgt_norm, label="Target (teacher)", color="tab:blue", linewidth=0.8, alpha=0.8)
        ax.plot(t, pred_norm, label="Predicted (student)", color="tab:orange", linewidth=0.8, alpha=0.8)
        ax.set_xlabel("Frame")
        ax.set_ylabel("L2 norm")
        ax.set_title(f"{prefix} reconstruction — epoch {epoch} "
                     f"(masked {mask_1d.sum():.0f}/{len(mask_1d)} frames)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, len(tgt_norm))
        fig.tight_layout()

        wandb.log({f"reconstruction/{prefix}": wandb.Image(fig), "epoch": epoch})
        plt.close(fig)
    except Exception as e:
        log(f"[WARN] Reconstruction plot failed ({prefix}, ep{epoch}): {e}")


def ssl_train_epoch_detailed(
    backbone: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip_norm: float = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> tuple:
    """One epoch of SSL pretraining with detailed loss component tracking.

    Calls ssl_forward_detailed() on backbones that support it to get separate
    ssl_loss and xyz_loss. Falls back to ssl_forward() for other backbones.

    Returns:
        (avg_loss, avg_ssl_loss, avg_xyz_loss, avg_grad_norm, last_detailed)
        where last_detailed is dict with mask_bool/predicted/target from last batch
        (or None if backbone doesn't support detailed forward).
    """
    backbone.train()
    total_loss = 0.0
    total_ssl_loss = 0.0
    total_xyz_loss = 0.0
    total_dist_loss = 0.0
    total_distpred_loss = 0.0
    total_grad_norm = 0.0
    n_batches = 0
    last_detailed = None

    has_detailed = hasattr(backbone, "ssl_forward_detailed")

    for step, batch in enumerate(dataloader):
        input_values = batch[0].float().to(device, non_blocking=True)
        coords = batch[2].float().to(device, non_blocking=True) if len(batch) > 2 else None
        input_values = preprocess_batch(input_values, backbone.needs_resampling)

        optimizer.zero_grad(set_to_none=True)

        needs_coords = coords is not None and (hasattr(backbone, "xyz_lambda") or hasattr(backbone, "dist_lambda") or hasattr(backbone, "dist_pred_head") or getattr(backbone, "distpred_lambda", 0) > 0)
        ssl_kwargs = {"coords": coords} if needs_coords else {}

        if has_detailed:
            if use_amp:
                with torch.cuda.amp.autocast():
                    result = backbone.ssl_forward_detailed(input_values, **ssl_kwargs)
            else:
                result = backbone.ssl_forward_detailed(input_values, **ssl_kwargs)

            loss = result["loss"]
            batch_ssl_loss = result["ssl_loss"].item()
            batch_xyz_loss = result["xyz_loss"].item()
            batch_dist_loss = result.get("dist_loss", torch.tensor(0.0)).item()
            batch_distpred_loss = result.get("distpred_loss", torch.tensor(0.0)).item()
            last_detailed = {
                "mask_bool": result["mask_bool"],
                "predicted": result["predicted"],
                "target": result["target"],
            }
        else:
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = backbone.ssl_forward(input_values, **ssl_kwargs)
            else:
                loss = backbone.ssl_forward(input_values, **ssl_kwargs)
            batch_ssl_loss = loss.item()
            batch_xyz_loss = 0.0
            batch_dist_loss = 0.0
            batch_distpred_loss = 0.0

        if loss is None or not torch.isfinite(loss):
            log(f"[WARN] Non-finite loss at step {step}, skipping")
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip_norm)
            optimizer.step()

        total_grad_norm += get_grad_norm(backbone)

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_ssl_loss += batch_ssl_loss
        total_xyz_loss += batch_xyz_loss
        total_dist_loss += batch_dist_loss
        total_distpred_loss += batch_distpred_loss
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_ssl_loss = total_ssl_loss / max(n_batches, 1)
    avg_xyz_loss = total_xyz_loss / max(n_batches, 1)
    avg_dist_loss = total_dist_loss / max(n_batches, 1)
    avg_distpred_loss = total_distpred_loss / max(n_batches, 1)
    avg_grad_norm = total_grad_norm / max(n_batches, 1)
    return avg_loss, avg_ssl_loss, avg_xyz_loss, avg_dist_loss, avg_distpred_loss, avg_grad_norm, last_detailed


@torch.no_grad()
def ssl_validate_epoch_detailed(
    backbone: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
    region_id_to_name: Optional[Dict[int, str]] = None,
) -> tuple:
    """One epoch of SSL validation with detailed loss components + reconstruction capture.

    Returns:
        (avg_loss, avg_ssl_loss, avg_xyz_loss, region_stats, captured_batch)
        where captured_batch is dict with mask_bool/predicted/target from first batch.
    """
    if region_id_to_name is None:
        region_id_to_name = REGION_ID_TO_NAME

    backbone.eval()
    total_loss = 0.0
    total_ssl_loss = 0.0
    total_xyz_loss = 0.0
    total_dist_loss = 0.0
    total_distpred_loss = 0.0
    n_batches = 0
    captured_batch = None

    region_losses: dict = {}
    has_detailed = hasattr(backbone, "ssl_forward_detailed")

    for batch in dataloader:
        input_values = batch[0].float().to(device, non_blocking=True)
        region_ids = batch[1] if len(batch) > 1 else None
        coords = batch[2].float().to(device, non_blocking=True) if len(batch) > 2 else None
        input_values = preprocess_batch(input_values, backbone.needs_resampling)

        needs_coords = coords is not None and (hasattr(backbone, "xyz_lambda") or hasattr(backbone, "dist_lambda") or hasattr(backbone, "dist_pred_head") or getattr(backbone, "distpred_lambda", 0) > 0)
        ssl_kwargs = {"coords": coords} if needs_coords else {}

        if has_detailed:
            if use_amp:
                with torch.cuda.amp.autocast():
                    result = backbone.ssl_forward_detailed(input_values, **ssl_kwargs)
            else:
                result = backbone.ssl_forward_detailed(input_values, **ssl_kwargs)

            loss = result["loss"]
            batch_ssl_loss = result["ssl_loss"].item()
            batch_xyz_loss = result["xyz_loss"].item()
            batch_dist_loss = result.get("dist_loss", torch.tensor(0.0)).item()
            batch_distpred_loss = result.get("distpred_loss", torch.tensor(0.0)).item()

            # Capture first batch for reconstruction visualization
            if captured_batch is None:
                captured_batch = {
                    "mask_bool": result["mask_bool"],
                    "predicted": result["predicted"],
                    "target": result["target"],
                }
        else:
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = backbone.ssl_forward(input_values, **ssl_kwargs)
            else:
                loss = backbone.ssl_forward(input_values, **ssl_kwargs)
            batch_ssl_loss = loss.item()
            batch_xyz_loss = 0.0
            batch_dist_loss = 0.0
            batch_distpred_loss = 0.0

        if loss is None or not torch.isfinite(loss):
            continue

        batch_loss = loss.item()
        total_loss += batch_loss
        total_ssl_loss += batch_ssl_loss
        total_xyz_loss += batch_xyz_loss
        total_dist_loss += batch_dist_loss
        total_distpred_loss += batch_distpred_loss
        n_batches += 1

        # Per-sample loss approximation: broadcast batch loss to all samples
        if torch.is_tensor(region_ids):
            for rid in region_ids:
                rid_int = int(rid.item())
                if rid_int < 0:
                    continue
                if rid_int not in region_losses:
                    region_losses[rid_int] = []
                region_losses[rid_int].append(batch_loss)

    avg_loss = total_loss / max(n_batches, 1)
    avg_ssl_loss = total_ssl_loss / max(n_batches, 1)
    avg_xyz_loss = total_xyz_loss / max(n_batches, 1)
    avg_dist_loss = total_dist_loss / max(n_batches, 1)
    avg_distpred_loss = total_distpred_loss / max(n_batches, 1)

    # Compute region statistics (same as ssl_validate_epoch_with_region_stats)
    region_stats: dict = {}
    region_means: list = []

    for rid in sorted(region_losses.keys()):
        region_name = region_id_to_name.get(rid, f"bin_{rid}")
        if region_losses[rid]:
            mean_loss = float(np.mean(region_losses[rid]))
            region_stats[f"loss_{region_name}"] = mean_loss
            region_stats[f"n_samples_{region_name}"] = len(region_losses[rid])
            region_means.append(mean_loss)
        else:
            region_stats[f"loss_{region_name}"] = float("nan")
            region_stats[f"n_samples_{region_name}"] = 0

    if region_means:
        mean_of_means = float(np.mean(region_means))
        std_of_means = float(np.std(region_means))
        min_loss = float(np.min(region_means))
        max_loss = float(np.max(region_means))
        region_stats["region_loss_cv"] = std_of_means / mean_of_means if mean_of_means > 0 else float("nan")
        region_stats["region_loss_max_min_ratio"] = max_loss / min_loss if min_loss > 0 else float("nan")
        hardest_idx = int(np.argmax(region_means))
        easiest_idx = int(np.argmin(region_means))
        region_stats["hardest_region_id"] = hardest_idx
        region_stats["easiest_region_id"] = easiest_idx
        region_stats["hardest_region_loss"] = max_loss
        region_stats["easiest_region_loss"] = min_loss
    else:
        region_stats["region_loss_cv"] = float("nan")
        region_stats["region_loss_max_min_ratio"] = float("nan")

    return avg_loss, avg_ssl_loss, avg_xyz_loss, avg_dist_loss, avg_distpred_loss, region_stats, captured_batch


# ===================== Ray Tune Trainable ===================== #

def ssl_trial_trainable(config: Dict[str, Any]) -> None:
    """Ray Tune trainable function — works with any backbone."""

    # Setup HuggingFace cache
    _user = os.environ.get("USER", "ws3")
    os.environ.setdefault("HF_HOME", f"/scratch/{_user}/.cache/huggingface")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    backbone_name = config["backbone"]
    log(f"Starting trial: backbone={backbone_name}")

    # Seed
    seed = int(config.get("seed", 42))
    config_str = str(sorted([(k, v) for k, v in config.items() if isinstance(v, (int, float, str, bool))]))
    trial_seed = seed + (abs(hash(config_str)) % 10_000_000)
    seed_all(trial_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # channels_csv → always use 2D probe shank coords; override any explicit coord_dim
    if config.get("channels_csv"):
        if config.get("coord_dim", 2) != 2:
            log(f"[WARN] channels_csv provided but coord_dim={config['coord_dim']} — overriding to 2")
        config["coord_dim"] = 2
        log("coord_dim=2 (channels_csv provided — using 2D probe shank coords)")

    # Build backbone
    log("Building backbone...")
    backbone = get_backbone(backbone_name, config).to(device)
    log(f"Backbone loaded: {backbone.get_num_params():,} trainable params, "
        f"hidden_dim={backbone.hidden_dim}, needs_resampling={backbone.needs_resampling}")

    # Init wandb for per-trial rich logging (images, HTML, tables)
    # This coexists with Ray's WandbLoggerCallback which only logs scalars
    _trial_wandb = False
    if wandb is not None and _in_ray_tune_session():
        try:
            from ray.tune import get_context as _get_tune_context
            trial_name = _get_tune_context().get_trial_name()
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "lfp2vec_training"),
                entity=os.environ.get("WANDB_ENTITY", None),
                group=f"ssl_{backbone_name}",
                name=f"{backbone_name}_{trial_name}",
                config=config,
                reinit=True,
            )
            _trial_wandb = True
            log(f"W&B per-trial logging initialized: {backbone_name}_{trial_name}")
        except Exception as e:
            log(f"[WARN] wandb per-trial init failed: {e}")

    # Optimizer
    lr_ssl = float(config.get("lr_ssl", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(backbone.get_optimizer_params(lr_ssl), weight_decay=weight_decay)

    use_amp = bool(config.get("amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))

    # LR warmup — critical for pretrained backbones on out-of-domain data.
    # Without warmup, large initial gradients * full LR → catastrophic weight
    # updates → NaN loss → grad collapse to 0.
    warmup_steps = int(config.get("warmup_steps", 203))
    warmup_start_factor = float(config.get("warmup_start_factor", 0.01))

    # Datasets
    compact_data_path = config.get("compact_data_path", "") or os.environ.get("COMPACT_DATA_PATH", "")

    # Atlas config
    atlas_depth = config.get("atlas_depth", None)
    if atlas_depth is not None:
        atlas_depth = int(atlas_depth)

    if compact_data_path and os.path.isdir(compact_data_path):
        from blind_localization.data.lazyloader_dataset import CompactDataset

        tile_to_length = config.get("tile_to_length", None)
        if tile_to_length is not None:
            tile_to_length = int(tile_to_length)
            log(f"Tiling signals to {tile_to_length} samples ({tile_to_length/1250:.1f}s @ 1250 Hz)")

        compact_kwargs = dict(
            include_labels=False,
            dataset=config.get("data", "Allen"),
            verbose=True,
            already_resampled=False,
            gpu_resample=True,
            input_sr=1250,
            model_sr=16000 if backbone.needs_resampling else 1250,
            super_regions=True,
            tile_to_length=tile_to_length,
            atlas_depth=atlas_depth,
            channels_csv=config.get("channels_csv", None),
        )

        train_dataset = CompactDataset(
            compact_dir=os.path.join(compact_data_path, "train"), **compact_kwargs
        )
        val_dataset = CompactDataset(
            compact_dir=os.path.join(compact_data_path, "val"), **compact_kwargs
        )

        # Labeled datasets for linear probe + per-region validation (include_labels=True)
        compact_kwargs_labeled = {**compact_kwargs, "include_labels": True}
        train_dataset_labeled = CompactDataset(
            compact_dir=os.path.join(compact_data_path, "train"), **compact_kwargs_labeled
        )
        val_dataset_labeled = CompactDataset(
            compact_dir=os.path.join(compact_data_path, "val"), **compact_kwargs_labeled
        )

        log(f"CompactDataset: train={len(train_dataset)}, val={len(val_dataset)}, "
            f"train_labeled={len(train_dataset_labeled)}, val_labeled={len(val_dataset_labeled)}")

        # Build dynamic region_id_to_name from atlas metadata (or fall back to 13-class)
        atlas_region_id_to_name = None
        atlas_metadata = None
        if atlas_depth is not None and train_dataset_labeled._atlas_label_maps:
            atlas_metadata = train_dataset_labeled._atlas_metadata
            label_map = train_dataset_labeled.effective_label_map  # global intersection map, contiguous 0..N-1
            atlas_region_id_to_name = {v: k for k, v in label_map.items()}
            log(f"Atlas region map (depth {atlas_depth}, global intersection): {len(atlas_region_id_to_name)} classes "
                f"(cumulative had {len(train_dataset_labeled.cumulative_label_map)})")
            # Verify train and val use the same global label map source
            train_src = train_dataset_labeled.global_label_map_source
            val_src = val_dataset_labeled.global_label_map_source
            log(f"Label map source: train={train_src}, val={val_src}")
            if train_src != val_src:
                raise RuntimeError(
                    f"Train and val datasets use different global label maps! "
                    f"train={train_src}, val={val_src}. "
                    f"All datasets must use the same label map family."
                )
    else:
        raise RuntimeError(f"Compact data not found at: {compact_data_path}")

    batch_size = int(config.get("batch_size_ssl", 64))
    num_workers = int(config.get("num_workers", 4))
    prefetch_factor = int(config.get("prefetch_factor", 20))

    # Probe-batch mode: group samples by probe_id so each batch contains
    # channels from the same Neuropixels probe.
    # Uses probe_groups.pkl if present; otherwise builds groups on-the-fly
    # from meta_str (requires channels_csv / extracols datasets).
    probe_batch = bool(config.get("probe_batch", False))
    if probe_batch:
        from blind_localization.data.lazyloader_dataset import ProbeBatchSampler
        _dl_common = dict(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
        )
        train_pg_path = os.path.join(compact_data_path, "train", "probe_groups.pkl")
        val_pg_path = os.path.join(compact_data_path, "val", "probe_groups.pkl")

        if os.path.exists(train_pg_path) and os.path.exists(val_pg_path):
            train_pg = train_pg_path
            val_pg = val_pg_path
            log("ProbeBatchSampler: loading probe_groups.pkl from disk")
        else:
            log("ProbeBatchSampler: probe_groups.pkl not found — building from meta_str (channels_csv mode)")
            train_pg = train_dataset.build_probe_groups()
            val_pg = val_dataset.build_probe_groups()

        train_sampler = ProbeBatchSampler(train_pg, batch_size=batch_size, shuffle=True)
        val_sampler = ProbeBatchSampler(val_pg, batch_size=batch_size, shuffle=False)
        log(f"ProbeBatchSampler: train {len(train_sampler.probe_ids)} probes, "
            f"val {len(val_sampler.probe_ids)} probes, batch_size={batch_size}")
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                  persistent_workers=(num_workers > 0), **_dl_common)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **_dl_common)
        # Labeled loaders reuse the same probe groups (already built above)
        val_loader_labeled = DataLoader(val_dataset_labeled, batch_sampler=val_sampler, **_dl_common)
        train_loader_labeled = DataLoader(
            train_dataset_labeled,
            batch_sampler=ProbeBatchSampler(train_pg, batch_size=batch_size, shuffle=False),
            **_dl_common,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
        )
        val_loader_labeled = DataLoader(
            val_dataset_labeled,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
        )
        train_loader_labeled = DataLoader(
            train_dataset_labeled,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
        )

    # LR scheduler: linear warmup + cosine decay (wav2vec2 paper standard)
    ssl_epochs = int(config.get("ssl_epochs", 10))
    steps_per_epoch = len(train_loader)
    total_steps = ssl_epochs * steps_per_epoch

    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay_steps = max(total_steps - warmup_steps, 1)
        decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=0,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps],
        )
        log(f"LR schedule: warmup {lr_ssl*warmup_start_factor:.2e} → {lr_ssl:.2e} "
            f"over {warmup_steps} steps, then cosine decay → 0 over {decay_steps} steps "
            f"(total {total_steps} steps, {ssl_epochs} epochs × {steps_per_epoch} steps/epoch)")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=0,
        )
        log(f"LR schedule: cosine decay {lr_ssl:.2e} → 0 over {total_steps} steps (no warmup)")

    # ---- Resume from checkpoint ----
    start_epoch = 0
    resume_path = config.get("resume_from", None)
    if resume_path and os.path.isfile(resume_path):
        log(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        backbone.load_state_dict(ckpt["backbone_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            # Fast-forward scheduler to the correct step
            resume_epoch = ckpt["epoch"] + 1
            ff_steps = resume_epoch * steps_per_epoch
            log(f"No scheduler state in checkpoint — fast-forwarding {ff_steps} steps")
            for _ in range(ff_steps):
                scheduler.step()
        start_epoch = ckpt["epoch"] + 1
        log(f"Resumed at epoch {start_epoch}, lr={optimizer.param_groups[0]['lr']:.2e}")
    elif resume_path:
        log(f"WARNING: resume_from path not found: {resume_path}, starting from scratch")

    # Training
    results_dir = config.get("results_dir", "") or os.environ.get(
        "RAY_RESULTS_DIR",
        f"/scratch/{os.environ.get('USER', 'ws3')}/ray_results/benchmarks"
    )

    # Compute mask_factor for loss normalization (prevents mask_prob gaming)
    # Audio backbones (contrastive, sum-reduced): mask_factor = mask_time_prob * mask_time_length
    # data2vec_native (MSE, mean-reduced): mask_factor = 1.0 (already per-element normalized)
    # Time-series backbones: mask_factor = mask_ratio (or 1.0 if no masking param)
    mask_time_prob = float(config.get("mask_time_prob", 0.0))
    mask_time_length = int(config.get("mask_time_length", 0))
    mask_ratio = float(config.get("mask_ratio", 0.0))

    if backbone_name == "data2vec_native":
        mask_factor = 1.0  # MSE loss uses mean reduction — already per-element normalized
    elif mask_time_prob > 0 and mask_time_length > 0:
        mask_factor = mask_time_prob * mask_time_length  # contrastive audio backbones
    elif mask_ratio > 0:
        mask_factor = mask_ratio  # time-series backbones
    else:
        mask_factor = 1.0  # fallback

    log(f"Mask factor: {mask_factor:.4f} (mask_time_prob={mask_time_prob}, "
        f"mask_time_length={mask_time_length}, mask_ratio={mask_ratio})")

    # Detailed logging mode — uses ssl_forward_detailed() for separate loss components
    detailed_logging = bool(config.get("detailed_logging", False))
    probe_interval = int(config.get("probe_interval", 0))  # 0 = every epoch (legacy)
    log(f"Detailed logging: {detailed_logging}, probe_interval: {probe_interval}")

    # Rolling averages for smoothed metrics (last 3 epochs)
    train_losses = []
    val_losses = []
    grad_norms = []
    best_val_loss = float("inf")
    best_probe_acc = 0.0
    prev_probe_ckpt_path = None    # track to delete old epoch-stamped ckpts
    prev_valloss_ckpt_path = None

    # Restore best metrics from resume checkpoint
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        best_probe_acc = ckpt.get("best_probe_acc", ckpt.get("probe_acc", 0.0))
        log(f"Restored best_val_loss={best_val_loss:.4f}, best_probe_acc={best_probe_acc:.4f}")
        del ckpt

    t_start = time.time()

    def _mean(lst):
        return sum(lst) / len(lst) if lst else float("inf")

    def _report_bad_trial(reason: str, epoch: int):
        """Report inf to Ray Tune for a failed trial (NaN/OOM) so ASHA can kill it."""
        bad_report = {
            "val_loss": float("inf"),
            "train_loss": float("inf"),
            "grad_norm": float("inf"),
            "loss_per_mask": float("inf"),
            "mask_factor": mask_factor,
            "lr": 0.0,
            "epoch": epoch,
            "elapsed_sec": time.time() - t_start,
            "backbone": backbone_name,
            "num_params": backbone.get_num_params(),
            "spatial_corr": 0.0,
            "combined_score": -999.0,
            reason: 1,
        }
        if _in_ray_tune_session():
            tune.report(bad_report)
        # Cleanup per-trial wandb on early exit
        if _trial_wandb:
            try:
                wandb.finish()
            except Exception:
                pass

    for epoch in range(start_epoch, ssl_epochs):
        t0 = time.time()

        # Reset per-epoch collapse stat accumulators (wav2vec2 and any backbone that supports it)
        if hasattr(backbone, 'reset_collapse_stats'):
            backbone.reset_collapse_stats()

        train_ssl_loss = 0.0
        train_xyz_loss = 0.0
        train_dist_loss = 0.0
        train_distpred_loss = 0.0
        val_ssl_loss = 0.0
        val_xyz_loss = 0.0
        val_dist_loss = 0.0
        val_distpred_loss = 0.0
        train_detailed_batch = None
        val_captured_batch = None

        try:
            if detailed_logging:
                train_loss, train_ssl_loss, train_xyz_loss, train_dist_loss, train_distpred_loss, grad_norm, train_detailed_batch = \
                    ssl_train_epoch_detailed(
                        backbone, train_loader, optimizer, device, use_amp, scaler, grad_clip_norm,
                        scheduler=scheduler,
                    )
            else:
                train_loss, grad_norm = ssl_train_epoch(
                    backbone, train_loader, optimizer, device, use_amp, scaler, grad_clip_norm,
                    scheduler=scheduler,
                )

            # NaN early-stop: detect non-finite train loss or grad norm
            if not np.isfinite(train_loss) or not np.isfinite(grad_norm):
                log(f"Epoch {epoch}: NaN detected (train_loss={train_loss}, grad_norm={grad_norm}), "
                    f"early stopping trial")
                _report_bad_trial("nan_early_stop", epoch)
                return

            # Validation with per-region loss tracking
            if detailed_logging:
                val_loss, val_ssl_loss, val_xyz_loss, val_dist_loss, val_distpred_loss, region_stats, val_captured_batch = \
                    ssl_validate_epoch_detailed(
                        backbone, val_loader_labeled, device, use_amp,
                        region_id_to_name=atlas_region_id_to_name,
                    )
            else:
                val_loss, region_stats = ssl_validate_epoch_with_region_stats(
                    backbone, val_loader_labeled, device, use_amp,
                    region_id_to_name=atlas_region_id_to_name,
                )

            # NaN early-stop: detect non-finite val loss
            if not np.isfinite(val_loss):
                log(f"Epoch {epoch}: NaN detected in val_loss={val_loss}, early stopping trial")
                _report_bad_trial("nan_early_stop", epoch)
                return

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
                log(f"Epoch {epoch}: OOM detected, early stopping trial")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _report_bad_trial("oom", epoch)
                return
            raise

        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        grad_norms.append(float(grad_norm))

        # Smoothed metrics (rolling avg of last 3 epochs)
        avg_val_loss = _mean(val_losses[-3:])
        avg_train_loss = _mean(train_losses[-3:])
        loss_per_mask = avg_val_loss / mask_factor if mask_factor > 0 else float("inf")

        if detailed_logging:
            loss_parts = f"ssl={train_ssl_loss:.4f} xyz={train_xyz_loss:.4f}"
            val_parts = f"val_ssl={val_ssl_loss:.4f} val_xyz={val_xyz_loss:.4f}"
            if train_dist_loss > 0 or val_dist_loss > 0:
                loss_parts += f" dist={train_dist_loss:.4f}"
                val_parts += f" val_dist={val_dist_loss:.4f}"
            if train_distpred_loss > 0 or val_distpred_loss > 0:
                loss_parts += f" distpred={train_distpred_loss:.4f}"
                val_parts += f" val_distpred={val_distpred_loss:.4f}"
            log(f"Epoch {epoch}/{ssl_epochs}: loss={train_loss:.4f} "
                f"{loss_parts} | "
                f"val_loss={val_loss:.4f} {val_parts} | "
                f"grad={grad_norm:.4f} lr={current_lr:.2e} time={elapsed:.1f}s")
        else:
            log(f"Epoch {epoch}/{ssl_epochs}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} loss_per_mask={loss_per_mask:.4f} "
                f"grad_norm={grad_norm:.4f} lr={current_lr:.2e} time={elapsed:.1f}s")

        # Region stats summary
        if region_stats:
            cv = region_stats.get("region_loss_cv", float("nan"))
            ratio = region_stats.get("region_loss_max_min_ratio", float("nan"))
            log(f"Epoch {epoch}: region_loss_cv={cv:.4f}, max_min_ratio={ratio:.4f}")

        # Collapse diagnostics from backbone accumulators (wav2vec2-specific)
        collapse_stats: dict = {}
        if hasattr(backbone, 'get_collapse_stats'):
            collapse_stats = backbone.get_collapse_stats()
            if collapse_stats:
                parts = "  ".join(f"{k}={v:.4f}" for k, v in sorted(collapse_stats.items()))
                log(f"Epoch {epoch} [collapse] {parts}")

        report_dict = {
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss,
            "grad_norm": _mean(grad_norms[-3:]),
            "val_loss_raw": float(val_loss),
            "train_loss_raw": float(train_loss),
            "grad_norm_raw": float(grad_norm),
            "loss_per_mask": loss_per_mask,
            "mask_factor": mask_factor,
            "lr": float(current_lr),
            "epoch": epoch,
            "elapsed_sec": time.time() - t_start,
            "backbone": backbone_name,
            "num_params": backbone.get_num_params(),
            "oom": 0,
            "spatial_corr": 0.0,  # default; overwritten if embed_interval triggers
            "combined_score": -999.0,  # default; overwritten with spatial_corr computation
        }

        # Add collapse stats (collected after val epoch, before report_dict was created)
        for k, v in collapse_stats.items():
            report_dict[f"collapse/{k}"] = float(v)

        # Detailed loss components
        if detailed_logging:
            report_dict["train_ssl_loss"] = float(train_ssl_loss)
            report_dict["train_xyz_loss"] = float(train_xyz_loss)
            report_dict["train_dist_loss"] = float(train_dist_loss)
            report_dict["train_distpred_loss"] = float(train_distpred_loss)
            report_dict["val_ssl_loss"] = float(val_ssl_loss)
            report_dict["val_xyz_loss"] = float(val_xyz_loss)
            report_dict["val_dist_loss"] = float(val_dist_loss)
            report_dict["val_distpred_loss"] = float(val_distpred_loss)

        # Add per-region loss statistics
        if region_stats:
            report_dict.update(region_stats)

        # Log region loss bar chart + reconstruction viz to W&B
        if _trial_wandb or (wandb is not None and wandb.run is not None):
            _log_region_bar_chart(region_stats, epoch, region_id_to_name=atlas_region_id_to_name)

            # Reconstruction visualization (detailed mode only)
            if detailed_logging:
                if train_detailed_batch is not None:
                    _log_reconstruction_wandb(
                        train_detailed_batch["predicted"],
                        train_detailed_batch["target"],
                        train_detailed_batch["mask_bool"],
                        "train", epoch,
                    )
                if val_captured_batch is not None:
                    _log_reconstruction_wandb(
                        val_captured_batch["predicted"],
                        val_captured_batch["target"],
                        val_captured_batch["mask_bool"],
                        "val", epoch,
                    )

        # Periodic evaluation interval (confusion matrices, etc.)
        eval_interval = max(ssl_epochs // 5, 1)  # ~5 evaluations per run
        is_final_epoch = (epoch == ssl_epochs - 1)
        is_eval_epoch = (epoch % eval_interval == 0) or is_final_epoch

        # Save raw embeddings and/or compute per-epoch metrics
        embed_interval = int(config.get("embed_interval", 0))
        metrics_interval = int(config.get("metrics_interval", 0))
        _should_save_emb = embed_interval > 0 and (epoch % embed_interval == 0 or is_final_epoch)
        _should_compute_metrics = metrics_interval > 0 and (epoch % metrics_interval == 0 or is_final_epoch)
        if _should_save_emb or _should_compute_metrics:
            try:
                if _in_ray_tune_session():
                    from ray.tune import get_context as _get_tune_ctx
                    _trial_dir = _get_tune_ctx().get_trial_dir()
                    embed_dir = os.path.join(_trial_dir, "embeddings")
                else:
                    _trial_dir = results_dir
                    embed_dir = os.path.join(results_dir, "embeddings")
                if _should_save_emb:
                    os.makedirs(embed_dir, exist_ok=True)
                backbone.eval()
                all_emb, all_labels, all_coords = [], [], []
                max_embed_samples = int(config.get("max_embed_samples", 0))
                _n_collected = 0
                with torch.no_grad():
                    for batch in val_loader_labeled:
                        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            waveforms, batch_labels = batch[0], batch[1]
                        else:
                            continue
                        waveforms = waveforms.float().to(device, non_blocking=True)
                        waveforms = preprocess_batch(waveforms, backbone.needs_resampling)
                        pooled = backbone.encode(waveforms)
                        all_emb.append(pooled.cpu().numpy())
                        all_labels.append(batch_labels.cpu().numpy() if torch.is_tensor(batch_labels) else np.array(batch_labels))
                        if len(batch) > 2:
                            all_coords.append(batch[2].cpu().numpy())
                        _n_collected += waveforms.shape[0]
                        if max_embed_samples > 0 and _n_collected >= max_embed_samples:
                            break
                all_emb = np.concatenate(all_emb, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                if _should_save_emb:
                    save_path = os.path.join(embed_dir, f"embeddings_ep{epoch:04d}.npz")
                    np.savez_compressed(save_path, embeddings=all_emb, labels=all_labels)
                    log(f"Saved embeddings ({all_emb.shape}) to {save_path}")

                # Embedding std — mean per-dim std across batch (collapse detector)
                embed_std = float(all_emb.std(axis=0).mean())
                report_dict["embed_std"] = embed_std
                log(f"Embed std (mean per-dim): {embed_std:.4f}")

                # Avg pairwise cosine similarity (→ 1.0 = all embeddings identical = collapse)
                avg_cosine_sim = float("nan")
                try:
                    n_cos = min(300, all_emb.shape[0])
                    idx_cos = np.random.choice(all_emb.shape[0], n_cos, replace=False)
                    z_sub = all_emb[idx_cos].astype(np.float32)
                    norms = np.linalg.norm(z_sub, axis=1, keepdims=True)
                    z_sub = z_sub / np.maximum(norms, 1e-8)
                    sim_mat = z_sub @ z_sub.T  # [n_cos, n_cos]
                    np.fill_diagonal(sim_mat, 0.0)
                    avg_cosine_sim = float(np.abs(sim_mat).sum() / (n_cos * (n_cos - 1)))
                    report_dict["collapse/avg_cosine_sim"] = avg_cosine_sim
                    log(f"Avg cosine sim: {avg_cosine_sim:.4f}  (→1.0 = collapse)")
                except Exception as _e:
                    log(f"[WARN] cosine sim failed: {_e}")

                # Effective rank of embedding covariance (→ 1 = all variance on one axis = collapse)
                effective_rank = float("nan")
                try:
                    cov_emb = np.cov(all_emb.T)
                    eigvals = np.linalg.eigvalsh(cov_emb)
                    eigvals = np.maximum(eigvals, 0.0)
                    s = eigvals / (eigvals.sum() + 1e-10)
                    s_pos = s[s > 1e-10]
                    effective_rank = float(np.exp(-np.sum(s_pos * np.log(s_pos + 1e-10))))
                    report_dict["collapse/effective_rank"] = effective_rank
                    log(f"Effective rank: {effective_rank:.1f}  (max={all_emb.shape[1]})")
                except Exception as _e:
                    log(f"[WARN] effective rank failed: {_e}")

                if _trial_wandb or (wandb is not None and wandb.run is not None):
                    wb_log = {"embed_std": embed_std, "epoch": epoch}
                    if np.isfinite(avg_cosine_sim):
                        wb_log["collapse/avg_cosine_sim"] = avg_cosine_sim
                    if np.isfinite(effective_rank):
                        wb_log["collapse/effective_rank"] = effective_rank
                    wandb.log(wb_log)

                # Spatial correlation — Spearman rho between pairwise embedding distances
                # and pairwise CCF coordinate distances (no labels used)
                if all_coords:
                    try:
                        from scipy.spatial.distance import pdist
                        from scipy.stats import spearmanr
                        all_coords_np = np.concatenate(all_coords, axis=0)
                        # Subsample to 500 for speed (O(n^2) otherwise)
                        n_sub = min(500, all_emb.shape[0])
                        idx = np.random.choice(all_emb.shape[0], n_sub, replace=False)
                        # L2-normalize embeddings so norm drift doesn't dominate
                        emb_sub = all_emb[idx]
                        norms = np.linalg.norm(emb_sub, axis=1, keepdims=True)
                        emb_sub = emb_sub / np.maximum(norms, 1e-8)
                        emb_dists = pdist(emb_sub)
                        coord_dists = pdist(all_coords_np[idx])
                        rho, _ = spearmanr(emb_dists, coord_dists)
                        # Clamp NaN (collapsed embeddings / degenerate distances) to 0
                        spatial_corr = float(rho) if np.isfinite(rho) else 0.0
                        report_dict["spatial_corr"] = spatial_corr
                        log(f"Spatial corr (Spearman rho): {spatial_corr:.4f}")
                        # Combined score: sc - alpha * val_loss  (higher = better)
                        # sc in [0,1], val_loss typically [0.5, 3.0] — alpha=0.1 balances them
                        _alpha = config.get("combined_alpha", 0.1)
                        combined = spatial_corr - _alpha * avg_val_loss
                        report_dict["combined_score"] = combined
                        log(f"Combined score (sc - {_alpha}*val_loss): {combined:.4f}")
                        if _trial_wandb or (wandb is not None and wandb.run is not None):
                            wandb.log({"spatial_corr": spatial_corr, "combined_score": combined, "epoch": epoch})
                    except Exception as e:
                        log(f"Warning: Failed to compute spatial_corr: {e}")

                # Per-epoch metrics CSV — written to trial dir (no .npz saved)
                try:
                    import csv as _csv
                    _csv_path = os.path.join(_trial_dir, "metrics_per_epoch.csv")
                    _csv_row = {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "train_loss_raw": float(train_loss),
                        "val_loss_raw": float(val_loss),
                        "grad_norm": _mean(grad_norms[-3:]),
                        "lr": float(current_lr),
                        "spatial_corr": report_dict.get("spatial_corr", float("nan")),
                        "avg_cosine_sim": report_dict.get("collapse/avg_cosine_sim", float("nan")),
                        "embed_std": report_dict.get("embed_std", float("nan")),
                        "eff_rank": report_dict.get("collapse/effective_rank", float("nan")),
                        "embed_mean": float(all_emb.mean()),
                    }
                    _write_header = not os.path.exists(_csv_path)
                    with open(_csv_path, "a", newline="") as _f:
                        _w = _csv.DictWriter(_f, fieldnames=list(_csv_row.keys()))
                        if _write_header:
                            _w.writeheader()
                        _w.writerow(_csv_row)
                    log(f"[metrics_csv] ep{epoch}: sc={_csv_row['spatial_corr']:.4f}  cosine={_csv_row['avg_cosine_sim']:.4f}  eff_rank={_csv_row['eff_rank']:.1f}")
                except Exception as _csv_e:
                    log(f"[WARN] Failed to write metrics CSV: {_csv_e}")

                backbone.train()
            except Exception as e:
                log(f"Warning: Failed in metrics/embed block: {e}")

        # Quick linear probe (skip if probe_epochs=0 or not on probe interval)
        probe_acc = None
        probe_epoch_count = int(config.get("probe_epochs", 20))
        max_probe_samples = int(config.get("max_probe_samples", 0))

        is_probe_epoch = True
        if probe_interval > 0:
            is_probe_epoch = (epoch % probe_interval == 0) or is_final_epoch

        if probe_epoch_count <= 0:
            log(f"[LinearProbe] Skipped (probe_epochs=0)")
        elif not is_probe_epoch:
            log(f"[LinearProbe] Skipped (next at epoch {((epoch // probe_interval) + 1) * probe_interval})")
        else:
            try:
                # Use global effective class count if available
                probe_num_classes = None
                if hasattr(train_dataset_labeled, 'num_effective_classes'):
                    probe_num_classes = train_dataset_labeled.num_effective_classes

                result = _quick_linear_probe(
                    backbone, train_loader_labeled, val_loader_labeled, device,
                    epoch_count=probe_epoch_count,
                    num_classes=probe_num_classes,
                    max_probe_samples=max_probe_samples,
                )
                if result is not None:
                    depth_acc, depth_bal_acc, val_preds, val_true, label_remap = result
                    val_preds = val_preds.cpu()
                    val_true = val_true.cpu()
                    probe_acc = depth_acc
                    report_dict["probe_acc"] = depth_acc
                    report_dict["probe_bal_acc"] = depth_bal_acc
                    report_dict["linear_probe_acc"] = depth_acc
                    report_dict["linear_probe_bal_acc"] = depth_bal_acc

                    if _trial_wandb or (wandb is not None and wandb.run is not None):
                        wandb.log({
                            "probe_acc": depth_acc,
                            "probe_bal_acc": depth_bal_acc,
                            "epoch": epoch,
                        })

                        # Build class names for region labels
                        n_cls = int(val_true.max().item()) + 1
                        remap_inv = {v: k for k, v in label_remap.items()}
                        if atlas_depth is not None and hasattr(train_dataset_labeled, 'effective_label_map') and train_dataset_labeled.effective_label_map:
                            orig_id_to_name = {v: k for k, v in train_dataset_labeled.effective_label_map.items()}
                        else:
                            orig_id_to_name = atlas_region_id_to_name or REGION_ID_TO_NAME
                        class_names = [orig_id_to_name.get(remap_inv.get(i, -1), f"Class {i}") for i in range(n_cls)]

                        # Per-class accuracy breakdown (every epoch)
                        for cls_idx in range(n_cls):
                            cls_mask = val_true == cls_idx
                            if cls_mask.sum() > 0:
                                cls_acc = (val_preds[cls_mask] == cls_idx).float().mean().item()
                                cls_name = class_names[cls_idx]
                                wandb.log({
                                    f"class_acc/{cls_name}": cls_acc,
                                    "epoch": epoch,
                                })

                        # Confusion matrix at periodic intervals
                        if is_eval_epoch:
                            wandb.log({"linear_probe/confusion_matrix": wandb.plot.confusion_matrix(
                                y_true=val_true.numpy().tolist(),
                                preds=val_preds.numpy().tolist(),
                                class_names=class_names,
                                title=f"Probe CM (Epoch {epoch})",
                            ), "epoch": epoch})
            except Exception as e:
                log(f"[WARN] Linear probe failed at epoch {epoch}: {e}")

        # Save best-probe-accuracy checkpoint
        save_checkpoints = bool(config.get("save_checkpoints", True))
        if save_checkpoints and probe_acc is not None and probe_acc > best_probe_acc:
            best_probe_acc = probe_acc
            if _in_ray_tune_session():
                from ray.tune import get_context as _get_tune_ctx
                ckpt_dir = os.path.join(_get_tune_ctx().get_trial_dir(), "checkpoints")
            else:
                ckpt_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            ckpt_data = {
                "backbone_state_dict": backbone.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "loss_per_mask": loss_per_mask,
                "mask_factor": mask_factor,
                "probe_acc": probe_acc,
            }

            if detailed_logging:
                # Epoch-stamped filename + symlink
                probe_ckpt_name = f"ssl_{backbone_name}_ep{epoch}_probe{probe_acc:.4f}.pt"
                probe_ckpt_path = os.path.join(ckpt_dir, probe_ckpt_name)
                torch.save(ckpt_data, probe_ckpt_path)
                if prev_probe_ckpt_path and os.path.exists(prev_probe_ckpt_path):
                    os.remove(prev_probe_ckpt_path)
                prev_probe_ckpt_path = probe_ckpt_path
                # Update _best.pt symlink
                best_link = os.path.join(ckpt_dir, f"ssl_{backbone_name}_best.pt")
                if os.path.islink(best_link) or os.path.exists(best_link):
                    os.remove(best_link)
                os.symlink(probe_ckpt_name, best_link)
            else:
                probe_ckpt_path = os.path.join(ckpt_dir, f"ssl_{backbone_name}_best.pt")
                torch.save(ckpt_data, probe_ckpt_path)

            log(f"Best PROBE checkpoint saved to {probe_ckpt_path} "
                f"(epoch {epoch}, probe_acc={probe_acc:.4f}, val_loss={avg_val_loss:.4f})")

        # Report to Ray Tune
        if _in_ray_tune_session():
            tune.report(report_dict)

        # Explicit wandb.log for loss curves (works in both Ray Tune and standalone)
        if wandb is not None and wandb.run is not None:
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_loss_raw": float(train_loss),
                "val_loss_raw": float(val_loss),
                "loss_per_mask": loss_per_mask,
                "grad_norm": _mean(grad_norms[-3:]),
                "lr": float(current_lr),
                "epoch": epoch,
            })

        # Save best val_loss checkpoint (separate file, for reference)
        if save_checkpoints and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if _in_ray_tune_session():
                from ray.tune import get_context as _get_tune_ctx
                ckpt_dir = os.path.join(_get_tune_ctx().get_trial_dir(), "checkpoints")
            else:
                ckpt_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            valloss_data = {
                "backbone_state_dict": backbone.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "loss_per_mask": loss_per_mask,
                "mask_factor": mask_factor,
            }

            if detailed_logging:
                valloss_ckpt_name = f"ssl_{backbone_name}_ep{epoch}_valloss{avg_val_loss:.4f}.pt"
                ckpt_path = os.path.join(ckpt_dir, valloss_ckpt_name)
                torch.save(valloss_data, ckpt_path)
                if prev_valloss_ckpt_path and os.path.exists(prev_valloss_ckpt_path):
                    os.remove(prev_valloss_ckpt_path)
                prev_valloss_ckpt_path = ckpt_path
                best_link = os.path.join(ckpt_dir, f"ssl_{backbone_name}_best_valloss.pt")
                if os.path.islink(best_link) or os.path.exists(best_link):
                    os.remove(best_link)
                os.symlink(valloss_ckpt_name, best_link)
            else:
                ckpt_path = os.path.join(ckpt_dir, f"ssl_{backbone_name}_best_valloss.pt")
                torch.save(valloss_data, ckpt_path)

            log(f"Best val_loss checkpoint saved to {ckpt_path} (epoch {epoch}, val_loss={avg_val_loss:.4f})")

        # Save "latest" checkpoint for resume (every epoch, overwritten)
        if not _in_ray_tune_session():
            latest_ckpt_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(latest_ckpt_dir, exist_ok=True)
            latest_path = os.path.join(latest_ckpt_dir, f"ssl_{backbone_name}_latest.pt")
            torch.save({
                "backbone_state_dict": backbone.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
                "best_probe_acc": best_probe_acc,
                "loss_per_mask": loss_per_mask,
                "mask_factor": mask_factor,
            }, latest_path)

        # Log to wandb (standalone mode)
        if not _in_ray_tune_session() and wandb is not None and wandb.run is not None:
            wandb.log(report_dict)

    # Cleanup per-trial wandb run
    if _trial_wandb:
        try:
            wandb.finish()
        except Exception:
            pass


# ===================== Standalone entry ===================== #

_EVAL_SUPPORTED_BACKBONES = {"whisper", "whisper_distill", "whisper_distpred", "wav2vec2"}


def _run_post_training_eval(args, config: dict) -> None:
    """Invoke eval_whisper_region.py as a subprocess after training completes."""
    import subprocess

    backbone_name = config["backbone"]
    if backbone_name not in _EVAL_SUPPORTED_BACKBONES:
        log(f"[WARN] Post-training eval not supported for backbone '{backbone_name}'. "
            f"Supported: {sorted(_EVAL_SUPPORTED_BACKBONES)}. Skipping.")
        return

    # Mirror the same results_dir resolution used during training
    _user = os.environ.get("USER", "ws3")
    results_dir = config.get("results_dir", "") or os.environ.get(
        "RAY_RESULTS_DIR",
        f"/scratch/{_user}/ray_results/benchmarks"
    )

    # Mirror the same compact_data_path fallback used during training
    compact_data_path = (
        config.get("compact_data_path", "")
        or os.environ.get("COMPACT_DATA_PATH", "")
    )
    if not compact_data_path:
        log("[WARN] compact_data_path not set; cannot run eval. Skipping.")
        return

    # Pick best checkpoint: prefer probe-acc best, fall back to latest
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    best_ckpt = os.path.join(ckpt_dir, f"ssl_{backbone_name}_best.pt")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(ckpt_dir, f"ssl_{backbone_name}_latest.pt")
    if not os.path.exists(best_ckpt):
        log(f"[WARN] No checkpoint found in {ckpt_dir}. Skipping eval.")
        return

    eval_output_dir = args.eval_output_dir or os.path.join(results_dir, "eval")
    eval_script = Path(__file__).resolve().parent / "eval_whisper_region.py"

    cmd = [
        sys.executable, str(eval_script),
        "--compact_data_path", compact_data_path,
        "--output_dir", eval_output_dir,
        "--backbone", backbone_name,
        "--ssl_checkpoint", best_ckpt,
        "--mode", *args.eval_modes,
        "--probe_epochs", str(args.eval_probe_epochs),
        "--finetune_epochs", str(args.eval_finetune_epochs),
    ]

    # configs use pretrained_model; eval_whisper_region.py expects --model_id
    model_id = config.get("pretrained_model") or config.get("model_id")
    if model_id:
        cmd += ["--model_id", model_id]

    atlas_depth = config.get("atlas_depth")
    if atlas_depth is not None:
        cmd += ["--atlas_depth", str(int(atlas_depth))]

    log("\n" + "=" * 60)
    log("POST-TRAINING EVAL")
    log(f"  Checkpoint : {best_ckpt}")
    log(f"  Modes      : {args.eval_modes}")
    log(f"  Output dir : {eval_output_dir}")
    log("=" * 60 + "\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log(f"[WARN] Post-training eval exited with code {result.returncode}")
    else:
        log("Post-training eval completed successfully.")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="SSL pretraining for Lfp2vec benchmark")
    parser.add_argument("--backbone", required=True,
                        choices=["wav2vec2", "wavlm", "wavlm_native", "data2vec", "data2vec_native", "patchtst", "moment", "ttm", "whisper", "whisper_distill", "whisper_distpred"])
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--compact_data_path", type=str, default="")
    parser.add_argument("--lr_ssl", type=float, default=None)
    parser.add_argument("--ssl_epochs", type=int, default=None)
    parser.add_argument("--batch_size_ssl", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default="",
                        help="Directory for checkpoints, embeddings, etc.")
    parser.add_argument("--embed_interval", type=int, default=None,
                        help="Save raw embeddings (.npz) every N epochs (0 = disabled)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint .pt file to resume training from")
    # Post-training eval
    parser.add_argument("--run_eval_after_training", action="store_true",
                        help="Run linear/mlp/finetune/full_finetune eval after training completes")
    parser.add_argument("--eval_modes", type=str, nargs="+",
                        default=["linear", "mlp", "finetune", "full_finetune"],
                        help="Eval modes to run post-training")
    parser.add_argument("--eval_probe_epochs", type=int, default=80,
                        help="Probe epochs for post-training eval")
    parser.add_argument("--eval_finetune_epochs", type=int, default=70,
                        help="Finetune epochs for post-training eval")
    parser.add_argument("--eval_output_dir", type=str, default=None,
                        help="Output dir for eval results (default: {results_dir}/eval)")
    return parser.parse_args()


def main():
    args = parse_args()

    config = {}
    if args.config:
        config = load_config(args.config)

    # CLI overrides YAML (only when explicitly provided)
    config["backbone"] = args.backbone
    if args.compact_data_path:
        config["compact_data_path"] = args.compact_data_path
    else:
        config.setdefault("compact_data_path", "")
    if args.lr_ssl is not None:
        config["lr_ssl"] = args.lr_ssl
    if args.ssl_epochs is not None:
        config["ssl_epochs"] = args.ssl_epochs
    if args.batch_size_ssl is not None:
        config["batch_size_ssl"] = args.batch_size_ssl
    if args.amp:
        config["amp"] = True
    if args.seed is not None:
        config["seed"] = args.seed
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.results_dir:
        config["results_dir"] = args.results_dir
    if args.embed_interval is not None:
        config["embed_interval"] = args.embed_interval
    if args.resume_from is not None:
        config["resume_from"] = args.resume_from

    # Initialize wandb for standalone mode
    _wandb_ok = False
    if wandb is not None and not _in_ray_tune_session():
        try:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "lfp2vec_training"),
                entity=os.environ.get("WANDB_ENTITY", None),
                group=f"ssl_full_{args.backbone}",
                name=os.environ.get("WANDB_RUN_NAME", f"ssl_full_{args.backbone}"),
                config=config,
            )
            _wandb_ok = True
        except Exception as e:
            print(f"[WARN] wandb init failed (continuing without W&B): {e}")

    ssl_trial_trainable(config)

    if _wandb_ok and wandb is not None and wandb.run is not None:
        wandb.finish()

    if args.run_eval_after_training:
        _run_post_training_eval(args, config)


if __name__ == "__main__":
    main()
