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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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


# ===================== DDP helpers ===================== #

def _is_main_process() -> bool:
    """True on rank 0 (or in single-process mode). Read from env so the check
    works before/without `dist.init_process_group`."""
    return int(os.environ.get("RANK", "0")) == 0


def _init_distributed():
    """Initialize torch.distributed if launched under torchrun.

    Returns:
        (local_rank, world_size, global_rank, is_main)

    When LOCAL_RANK env var is not set, returns (0, 1, 0, True) and does NOT
    touch torch.distributed — single-GPU path stays backward compatible.
    """
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is None:
        return 0, 1, 0, True

    local_rank = int(local_rank_env)
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Rank-0-only diagnostics can leave the other ranks waiting at the
        # epoch-boundary barrier. Keep this configurable because large-scale
        # embedding/probe extraction can legitimately exceed the PyTorch
        # default without indicating a failed collective. NOTE: this is the
        # correct knob — TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC / NCCL_TIMEOUT
        # control the heartbeat monitor, not the collective-op timeout.
        from datetime import timedelta
        timeout_minutes = int(os.environ.get("LFP2VEC_DDP_TIMEOUT_MINUTES", "30"))
        if timeout_minutes <= 0:
            raise ValueError(
                "LFP2VEC_DDP_TIMEOUT_MINUTES must be a positive integer, "
                f"got {timeout_minutes}"
            )
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=timeout_minutes),
        )

    is_main = (global_rank == 0)
    return local_rank, world_size, global_rank, is_main


def _unwrap(module):
    """Return the underlying module from a DDP wrapper (or the module itself)."""
    return module.module if isinstance(module, DDP) else module


def log(msg: str):
    # Gate to rank 0 in DDP mode; in single-GPU mode this is always True.
    if not _is_main_process():
        return
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    backbone = _unwrap(backbone)  # DDP-safe; probe runs on rank 0 only
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
                waveforms = preprocess_batch(waveforms, backbone.needs_resampling, backbone.needs_normalization)
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
                    waveforms = preprocess_batch(waveforms, backbone.needs_resampling, backbone.needs_normalization)
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
    inner = _unwrap(backbone)  # for attribute access; backbone(...) routes through DDP
    total_loss = 0.0
    total_grad_norm = 0.0
    n_batches = 0

    for step, batch in enumerate(dataloader):
        input_values = batch[0].float().to(device, non_blocking=True)
        coords = batch[2].float().to(device, non_blocking=True) if len(batch) > 2 else None
        input_values = preprocess_batch(input_values, inner.needs_resampling, inner.needs_normalization)

        optimizer.zero_grad(set_to_none=True)

        # Pass coords for XYZ/dist auxiliary loss (only if backbone supports it)
        needs_coords = coords is not None and (hasattr(inner, "xyz_lambda") or hasattr(inner, "dist_lambda") or hasattr(inner, "dist_pred_head") or getattr(inner, "distpred_lambda", 0) > 0)
        ssl_kwargs = {"coords": coords} if needs_coords else {}
        # Call via backbone(...) (mode='ssl_forward') so DDP installs grad hooks.
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = backbone(input_values, mode="ssl_forward", **ssl_kwargs)
        else:
            loss = backbone(input_values, mode="ssl_forward", **ssl_kwargs)

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

        # Gumbel temperature annealing (wav2vec2 paper: decay per step)
        if hasattr(inner, "step_gumbel_temperature"):
            inner.step_gumbel_temperature()

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
    backbone = _unwrap(backbone)
    backbone.eval()
    total_loss = 0.0
    n_batches = 0

    for input_values, _labels in dataloader:
        input_values = input_values.float().to(device, non_blocking=True)
        input_values = preprocess_batch(input_values, backbone.needs_resampling, backbone.needs_normalization)

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

    backbone = _unwrap(backbone)
    backbone.eval()
    total_loss = 0.0
    n_batches = 0

    # Accumulate losses by region/bin
    region_losses: dict = {}

    for batch in dataloader:
        input_values = batch[0].float().to(device, non_blocking=True)
        region_ids = batch[1] if len(batch) > 1 else None
        coords = batch[2].float().to(device, non_blocking=True) if len(batch) > 2 else None
        input_values = preprocess_batch(input_values, backbone.needs_resampling, backbone.needs_normalization)

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
    inner = _unwrap(backbone)
    total_loss = 0.0
    total_ssl_loss = 0.0
    total_xyz_loss = 0.0
    total_dist_loss = 0.0
    total_distpred_loss = 0.0
    total_var_loss = 0.0
    total_contrastive_loss = 0.0
    total_diversity_loss = 0.0
    total_perplexity = 0.0
    last_gumbel_temp = None
    total_grad_norm = 0.0
    n_batches = 0
    last_detailed = None

    has_detailed = hasattr(inner, "ssl_forward_detailed")

    for step, batch in enumerate(dataloader):
        input_values = batch[0].float().to(device, non_blocking=True)
        coords = batch[2].float().to(device, non_blocking=True) if len(batch) > 2 else None
        input_values = preprocess_batch(input_values, inner.needs_resampling, inner.needs_normalization)

        optimizer.zero_grad(set_to_none=True)

        needs_coords = coords is not None and (hasattr(inner, "xyz_lambda") or hasattr(inner, "dist_lambda") or hasattr(inner, "dist_pred_head") or getattr(inner, "distpred_lambda", 0) > 0)
        ssl_kwargs = {"coords": coords} if needs_coords else {}

        if has_detailed:
            if use_amp:
                with torch.cuda.amp.autocast():
                    result = backbone(input_values, mode="ssl_forward_detailed", **ssl_kwargs)
            else:
                result = backbone(input_values, mode="ssl_forward_detailed", **ssl_kwargs)

            loss = result["loss"]
            batch_ssl_loss = result["ssl_loss"].item()
            batch_xyz_loss = result["xyz_loss"].item()
            batch_dist_loss = result.get("dist_loss", torch.tensor(0.0)).item()
            batch_distpred_loss = result.get("distpred_loss", torch.tensor(0.0)).item()
            batch_var_loss = result.get("var_loss", torch.tensor(0.0)).item()

            # Wav2Vec2-specific metrics (contrastive/diversity/perplexity)
            if "contrastive_loss" in result:
                _ctr = result["contrastive_loss"].item()
                _div = result["diversity_loss"].item()
                _ppl = result["codevector_perplexity"].item()
                if np.isfinite(_ctr):
                    total_contrastive_loss += _ctr
                if np.isfinite(_div):
                    total_diversity_loss += _div
                if np.isfinite(_ppl):
                    total_perplexity += _ppl
                last_gumbel_temp = result.get("gumbel_temperature")

            last_detailed = {k: result[k] for k in ("mask_bool", "predicted", "target") if k in result}
        else:
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = backbone(input_values, mode="ssl_forward", **ssl_kwargs)
            else:
                loss = backbone(input_values, mode="ssl_forward", **ssl_kwargs)
            batch_ssl_loss = loss.item()
            batch_xyz_loss = 0.0
            batch_dist_loss = 0.0
            batch_distpred_loss = 0.0
            batch_var_loss = 0.0

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

        # Gumbel temperature annealing (wav2vec2 paper: decay per step)
        if hasattr(inner, "step_gumbel_temperature"):
            inner.step_gumbel_temperature()

        total_loss += loss.item()
        total_ssl_loss += batch_ssl_loss
        total_xyz_loss += batch_xyz_loss
        total_dist_loss += batch_dist_loss
        total_distpred_loss += batch_distpred_loss
        total_var_loss += batch_var_loss
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_ssl_loss = total_ssl_loss / max(n_batches, 1)
    avg_xyz_loss = total_xyz_loss / max(n_batches, 1)
    avg_dist_loss = total_dist_loss / max(n_batches, 1)
    avg_distpred_loss = total_distpred_loss / max(n_batches, 1)
    avg_var_loss = total_var_loss / max(n_batches, 1)
    avg_grad_norm = total_grad_norm / max(n_batches, 1)

    # Wav2Vec2-specific metrics (contrastive/diversity/perplexity)
    w2v2_metrics = {}
    if total_contrastive_loss > 0 or total_perplexity > 0:
        w2v2_metrics = {
            "contrastive_loss": total_contrastive_loss / max(n_batches, 1),
            "diversity_loss": total_diversity_loss / max(n_batches, 1),
            "codevector_perplexity": total_perplexity / max(n_batches, 1),
        }
        if last_gumbel_temp is not None:
            w2v2_metrics["gumbel_temperature"] = last_gumbel_temp

    return avg_loss, avg_ssl_loss, avg_xyz_loss, avg_dist_loss, avg_distpred_loss, avg_var_loss, avg_grad_norm, last_detailed, w2v2_metrics


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

    backbone = _unwrap(backbone)
    backbone.eval()
    total_loss = 0.0
    total_ssl_loss = 0.0
    total_xyz_loss = 0.0
    total_dist_loss = 0.0
    total_distpred_loss = 0.0
    total_var_loss = 0.0
    n_batches = 0
    captured_batch = None

    region_losses: dict = {}
    has_detailed = hasattr(backbone, "ssl_forward_detailed")

    for batch in dataloader:
        input_values = batch[0].float().to(device, non_blocking=True)
        region_ids = batch[1] if len(batch) > 1 else None
        coords = batch[2].float().to(device, non_blocking=True) if len(batch) > 2 else None
        input_values = preprocess_batch(input_values, backbone.needs_resampling, backbone.needs_normalization)

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
            batch_var_loss = result.get("var_loss", torch.tensor(0.0)).item()

            # Capture first batch for reconstruction visualization
            # (only for reconstruction-based backbones like data2vec, not wav2vec2)
            if captured_batch is None and "mask_bool" in result:
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
            batch_var_loss = 0.0

        if loss is None or not torch.isfinite(loss):
            continue

        batch_loss = loss.item()
        total_loss += batch_loss
        total_ssl_loss += batch_ssl_loss
        total_xyz_loss += batch_xyz_loss
        total_dist_loss += batch_dist_loss
        total_distpred_loss += batch_distpred_loss
        total_var_loss += batch_var_loss
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
    avg_var_loss = total_var_loss / max(n_batches, 1)

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

    return avg_loss, avg_ssl_loss, avg_xyz_loss, avg_dist_loss, avg_distpred_loss, avg_var_loss, region_stats, captured_batch


# ===================== Ray Tune Trainable ===================== #

def ssl_trial_trainable(config: Dict[str, Any]) -> None:
    """Ray Tune trainable function — works with any backbone."""

    # Setup HuggingFace cache
    _user = os.environ.get("USER", "ws3")
    os.environ.setdefault("HF_HOME", f"/scratch/{_user}/.cache/huggingface")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # ---- DDP init (no-op if LOCAL_RANK not set) ----
    local_rank, world_size, global_rank, is_main = _init_distributed()
    is_ddp = world_size > 1
    if is_ddp:
        # Rank 0 prints; other ranks gated via log().
        print(f"[DDP] rank={global_rank}/{world_size} local_rank={local_rank} "
              f"backend={dist.get_backend()}", flush=True)

    backbone_name = config["backbone"]
    log(f"Starting trial: backbone={backbone_name} (world_size={world_size})")

    # Seed (offset by global_rank so each rank's stochastic ops differ;
    # DistributedSampler still controls cross-rank data partitioning).
    seed = int(config.get("seed", 42))
    config_str = str(sorted([(k, v) for k, v in config.items() if isinstance(v, (int, float, str, bool))]))
    trial_seed = seed + (abs(hash(config_str)) % 10_000_000) + global_rank
    seed_all(trial_seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Build backbone
    log("Building backbone...")
    backbone = get_backbone(backbone_name, config).to(device)
    if is_ddp:
        # Wav2Vec2 keeps optional auxiliary heads/codebook branches in the module
        # even when a given config does not activate them. DDP must detect those
        # unused parameters or it aborts on the next iteration while rebuilding
        # reduction buckets. Configs may explicitly disable this only when every
        # trainable parameter is guaranteed to participate in every forward pass.
        find_unused = bool(config.get("ddp_find_unused_parameters", True))
        backbone = DDP(
            backbone,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )
        log(f"DDP wrapped backbone (find_unused_parameters={find_unused})")
    backbone_module = _unwrap(backbone)
    log(f"Backbone loaded: {backbone_module.get_num_params():,} trainable params, "
        f"hidden_dim={backbone_module.hidden_dim}, needs_resampling={backbone_module.needs_resampling}")

    # Init wandb for per-trial rich logging (images, HTML, tables)
    # This coexists with Ray's WandbLoggerCallback which only logs scalars
    _trial_wandb = False
    if wandb is not None and _in_ray_tune_session() and is_main:
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

    # Optimizer (use backbone_module for get_optimizer_params; DDP forwards
    # gradients to the same underlying parameters)
    lr_ssl = float(config.get("lr_ssl", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(backbone_module.get_optimizer_params(lr_ssl), weight_decay=weight_decay)

    use_amp = bool(config.get("amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))

    # LR warmup — critical for pretrained backbones on out-of-domain data.
    # Without warmup, large initial gradients * full LR → catastrophic weight
    # updates → NaN loss → grad collapse to 0.
    warmup_steps = int(config.get("warmup_steps", 203))
    warmup_start_factor = float(config.get("warmup_start_factor", 0.01))

    # Datasets
    compact_data_path = os.environ.get("COMPACT_DATA_PATH", "") or config.get("compact_data_path", "")

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
            model_sr=16000 if backbone_module.needs_resampling else 1250,
            super_regions=True,
            tile_to_length=tile_to_length,
            atlas_depth=atlas_depth,
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

    # DDP: split the *global* config batch across ranks so the effective
    # batch (config.batch_size_ssl) stays constant. Round up to avoid 0
    # per-rank batches at tiny scales (config validation catches the silly cases).
    per_gpu_batch = max(batch_size // world_size, 1)
    if is_ddp:
        log(f"DDP DataLoader: per_gpu_batch={per_gpu_batch} "
            f"(global batch_size_ssl={batch_size} / world_size={world_size})")

    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=global_rank,
            shuffle=True, drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=global_rank,
            shuffle=False, drop_last=False,
        )
        val_labeled_sampler = DistributedSampler(
            val_dataset_labeled, num_replicas=world_size, rank=global_rank,
            shuffle=False, drop_last=False,
        )
        # Probe + train_labeled extraction runs on rank 0 only; samplers there
        # would shard data across ranks, so leave them as normal loaders.
        train_loader = DataLoader(
            train_dataset,
            batch_size=per_gpu_batch,
            sampler=train_sampler,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=per_gpu_batch,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
        )
        val_loader_labeled = DataLoader(
            val_dataset_labeled,
            batch_size=per_gpu_batch,
            sampler=val_labeled_sampler,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
        )
        train_loader_labeled = DataLoader(
            train_dataset_labeled,
            batch_size=per_gpu_batch,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=per_gpu_batch,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=per_gpu_batch,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
        )
        val_loader_labeled = DataLoader(
            val_dataset_labeled,
            batch_size=per_gpu_batch,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
        )
        train_loader_labeled = DataLoader(
            train_dataset_labeled,
            batch_size=per_gpu_batch,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
        )

    # LR scheduler
    ssl_epochs = int(config.get("ssl_epochs", 10))
    steps_per_epoch = len(train_loader)
    total_steps = ssl_epochs * steps_per_epoch
    eta_min = float(config.get("eta_min", 0.0))
    lr_schedule = config.get("lr_schedule", "cosine")  # "cosine" or "plateau"

    # epoch_scheduler: stepped per-epoch after val_loss (only for plateau mode)
    epoch_scheduler = None

    if lr_schedule == "plateau":
        # Warmup (per-step) then ReduceLROnPlateau (per-epoch)
        # LR reduces by factor when val_loss plateaus — adapts to actual convergence
        plateau_factor = float(config.get("plateau_factor", 0.5))
        plateau_patience = int(config.get("plateau_patience", 5))

        if warmup_steps > 0:
            # Use LinearLR for warmup, then switch to plateau after warmup completes
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            scheduler = None

        epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=eta_min,
        )
        log(f"LR schedule: warmup {lr_ssl*warmup_start_factor:.2e} → {lr_ssl:.2e} "
            f"over {warmup_steps} steps, then ReduceLROnPlateau "
            f"(factor={plateau_factor}, patience={plateau_patience}, min_lr={eta_min:.2e})")
    else:
        # Default: cosine decay (wav2vec2 paper standard)
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
                eta_min=eta_min,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps],
            )
            log(f"LR schedule: warmup {lr_ssl*warmup_start_factor:.2e} → {lr_ssl:.2e} "
                f"over {warmup_steps} steps, then cosine decay → {eta_min:.2e} over {decay_steps} steps "
                f"(total {total_steps} steps, {ssl_epochs} epochs × {steps_per_epoch} steps/epoch)")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=eta_min,
            )
            log(f"LR schedule: cosine decay {lr_ssl:.2e} → {eta_min:.2e} over {total_steps} steps (no warmup)")

    # ---- GCP full-recording run: step-based loop (W&B per step, mid-epoch resume) ----
    if bool(config.get("step_mode", False)):
        if config.get("channel_distance", False):
            from step_mode_channel import run_step_mode
        else:
            from step_mode import run_step_mode
        run_step_mode(dict(
            config=config, log=log, backbone=backbone, backbone_module=backbone_module,
            optimizer=optimizer, device=device, use_amp=use_amp, scaler=scaler,
            is_main=is_main, is_ddp=is_ddp, world_size=world_size, global_rank=global_rank,
            preprocess_batch=preprocess_batch, quick_linear_probe=_quick_linear_probe,
            train_dataset=train_dataset, per_gpu_batch=per_gpu_batch,
            compact_kwargs_labeled=compact_kwargs_labeled,
        ))
        return

    # ---- Resume from checkpoint ----
    start_epoch = 0
    resume_path = config.get("resume_from", None)
    if resume_path and os.path.isfile(resume_path):
        log(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        backbone_module.load_state_dict(ckpt["backbone_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        resume_step = start_epoch * steps_per_epoch

        # Rebuild scheduler for the NEW total_steps, then fast-forward to resume point.
        # We do NOT load the old scheduler state because T_max may have changed
        # (e.g., original was 50 epochs, now extended to 100).
        # Instead, create a fresh schedule for the full new duration and step to current position.
        log(f"Rebuilding LR schedule for {total_steps} total steps, fast-forwarding to step {resume_step}")
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
                eta_min=eta_min,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=eta_min,
            )

        # Fast-forward logic depends on the scheduler family.
        # - Cosine: step() `resume_step` times to land at the right cosine
        #   position. Optimizer's loaded LR is overwritten by design (cosine
        #   recomputes LR each step from base_lr).
        # - Plateau: the per-step scheduler is LinearLR warmup only. Once we
        #   are past `warmup_steps`, the warmup is done and plateau owns the
        #   LR. Stepping the warmup scheduler in that regime would stomp the
        #   plateau-reduced LR back to `base_lr * end_factor = lr_ssl`,
        #   throwing away every plateau reduction the checkpoint had baked in.
        #   So: if past warmup, null the scheduler and trust the loaded
        #   optimizer LR.
        if lr_schedule != "plateau":
            for _ in range(resume_step):
                scheduler.step()
        else:
            if warmup_steps > 0 and resume_step < warmup_steps:
                # Still in warmup — fast-forward to the current position.
                for _ in range(resume_step):
                    scheduler.step()
            else:
                # Past warmup (or no warmup). Hand LR control to plateau.
                scheduler = None

        log(f"Resumed at epoch {start_epoch}, lr={optimizer.param_groups[0]['lr']:.2e} "
            f"(step {resume_step}/{total_steps})")

        # LR override — bypass the saved LR state, give the model headroom to
        # keep learning after plateau had driven LR into the floor. Read from
        # config (main() pushes CLI --lr_override into config) because `args`
        # isn't in scope inside ssl_trial_trainable.
        _lr_override = config.get("lr_override", None)
        _lr_override_warmup = int(config.get("lr_override_warmup_steps", 200))
        if _lr_override is not None:
            target_lr = float(_lr_override)
            current_lr = optimizer.param_groups[0]["lr"]
            warmup_o = max(_lr_override_warmup, 0)
            if warmup_o > 0 and current_lr < target_lr:
                for g in optimizer.param_groups:
                    g["lr"] = target_lr
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=current_lr / target_lr,
                    end_factor=1.0,
                    total_iters=warmup_o,
                )
                lr_override_remaining_warmup = warmup_o
                log(f"  LR override: {current_lr:.2e} → {target_lr:.2e} "
                    f"over {warmup_o} warmup steps "
                    f"(start_factor={current_lr/target_lr:.4f}, end_factor=1.0)")
            else:
                for g in optimizer.param_groups:
                    g["lr"] = target_lr
                scheduler = None
                lr_override_remaining_warmup = 0
                log(f"  LR override: {current_lr:.2e} → {target_lr:.2e} (step jump)")
        else:
            lr_override_remaining_warmup = 0
    elif resume_path:
        log(f"WARNING: resume_from path not found: {resume_path}, starting from scratch")
        lr_override_remaining_warmup = 0
    else:
        lr_override_remaining_warmup = 0

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

    best_val_loss = float("inf")
    best_probe_acc = 0.0
    early_stop_patience = int(config.get("early_stop_patience", 0))  # 0 = disabled
    epochs_without_improvement = 0

    # Restore best metrics from resume checkpoint
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        best_probe_acc = ckpt.get("best_probe_acc", ckpt.get("probe_acc", 0.0))
        log(f"Restored best_val_loss={best_val_loss:.4f}, best_probe_acc={best_probe_acc:.4f}")
        del ckpt

    if early_stop_patience > 0:
        log(f"Early stopping: patience={early_stop_patience} epochs without val_loss improvement")

    t_start = time.time()

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
            "num_params": backbone_module.get_num_params(),
            reason: 1,
        }
        if _in_ray_tune_session() and is_main:
            tune.report(bad_report)
        # Cleanup per-trial wandb on early exit
        if _trial_wandb:
            try:
                wandb.finish()
            except Exception:
                pass

    for epoch in range(start_epoch, ssl_epochs):
        t0 = time.time()

        # DDP: re-seed the DistributedSampler per epoch for proper shuffling
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_ssl_loss = 0.0
        train_xyz_loss = 0.0
        train_dist_loss = 0.0
        train_distpred_loss = 0.0
        train_var_loss = 0.0
        val_ssl_loss = 0.0
        val_xyz_loss = 0.0
        val_dist_loss = 0.0
        val_distpred_loss = 0.0
        val_var_loss = 0.0
        train_detailed_batch = None
        train_w2v2_metrics = {}
        val_captured_batch = None

        try:
            if detailed_logging:
                train_loss, train_ssl_loss, train_xyz_loss, train_dist_loss, train_distpred_loss, train_var_loss, grad_norm, train_detailed_batch, train_w2v2_metrics = \
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
                val_loss, val_ssl_loss, val_xyz_loss, val_dist_loss, val_distpred_loss, val_var_loss, region_stats, val_captured_batch = \
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
        loss_per_mask = float(val_loss) / mask_factor if mask_factor > 0 else float("inf")

        if detailed_logging:
            loss_parts = f"ssl={train_ssl_loss:.4f} xyz={train_xyz_loss:.4f}"
            val_parts = f"val_ssl={val_ssl_loss:.4f} val_xyz={val_xyz_loss:.4f}"
            if train_dist_loss > 0 or val_dist_loss > 0:
                loss_parts += f" dist={train_dist_loss:.4f}"
                val_parts += f" val_dist={val_dist_loss:.4f}"
            if train_distpred_loss > 0 or val_distpred_loss > 0:
                loss_parts += f" distpred={train_distpred_loss:.4f}"
                val_parts += f" val_distpred={val_distpred_loss:.4f}"
            if train_var_loss > 0 or val_var_loss > 0:
                loss_parts += f" var={train_var_loss:.4f}"
                val_parts += f" val_var={val_var_loss:.4f}"
            # Wav2Vec2-specific metrics
            w2v2_parts = ""
            if train_w2v2_metrics:
                ppl = train_w2v2_metrics.get("codevector_perplexity", 0)
                gtemp = train_w2v2_metrics.get("gumbel_temperature", 0)
                closs = train_w2v2_metrics.get("contrastive_loss", 0)
                dloss = train_w2v2_metrics.get("diversity_loss", 0)
                w2v2_parts = f" | ppl={ppl:.1f} gtemp={gtemp:.4f} ctr={closs:.2f} div={dloss:.4f}"
            log(f"Epoch {epoch}/{ssl_epochs}: loss={train_loss:.4f} "
                f"{loss_parts} | "
                f"val_loss={val_loss:.4f} {val_parts} | "
                f"grad={grad_norm:.4f} lr={current_lr:.2e}{w2v2_parts} time={elapsed:.1f}s")
        else:
            log(f"Epoch {epoch}/{ssl_epochs}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} loss_per_mask={loss_per_mask:.4f} "
                f"grad_norm={grad_norm:.4f} lr={current_lr:.2e} time={elapsed:.1f}s")

        # Region stats summary
        if region_stats:
            cv = region_stats.get("region_loss_cv", float("nan"))
            ratio = region_stats.get("region_loss_max_min_ratio", float("nan"))
            log(f"Epoch {epoch}: region_loss_cv={cv:.4f}, max_min_ratio={ratio:.4f}")

        report_dict = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "grad_norm": float(grad_norm),
            "loss_per_mask": float(val_loss) / mask_factor if mask_factor > 0 else float("inf"),
            "lr": float(current_lr),
            "epoch": epoch,
            "elapsed_sec": time.time() - t_start,
        }

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

            # Wav2Vec2-specific: contrastive/diversity/perplexity/gumbel temp
            if train_w2v2_metrics:
                report_dict.update({f"train_{k}": v for k, v in train_w2v2_metrics.items()})

        # Periodic interval flags (used by probe, eval, embed)
        eval_interval = max(ssl_epochs // 5, 1)
        is_final_epoch = (epoch == ssl_epochs - 1)
        is_eval_epoch = (epoch % eval_interval == 0) or is_final_epoch

        probe_interval = int(config.get("probe_interval", 0))
        is_probe_epoch = True
        if probe_interval > 0:
            is_probe_epoch = (epoch % probe_interval == 0) or is_final_epoch

        # Save a recovery checkpoint before rank-0-only diagnostics. Without
        # this, a diagnostic/barrier failure after a successfully completed
        # train+validation epoch loses the entire epoch because the ordinary
        # "latest" checkpoint is written after the diagnostics.
        embed_interval = int(config.get("embed_interval", 0))
        probe_epoch_count = int(config.get("probe_epochs", 20))
        runs_embedding_diagnostic = (
            embed_interval > 0
            and (epoch % embed_interval == 0 or is_final_epoch)
        )
        runs_probe_diagnostic = probe_epoch_count > 0 and is_probe_epoch
        if (
            is_main
            and is_ddp
            and not _in_ray_tune_session()
            and (runs_embedding_diagnostic or runs_probe_diagnostic)
        ):
            recovery_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(recovery_dir, exist_ok=True)
            recovery_path = os.path.join(
                recovery_dir, f"ssl_{backbone_name}_recovery.pt"
            )
            recovery_payload = {
                "backbone_state_dict": backbone_module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": float(val_loss),
                "best_val_loss": min(best_val_loss, float(val_loss)),
                "best_probe_acc": best_probe_acc,
                "loss_per_mask": loss_per_mask,
                "mask_factor": mask_factor,
            }
            if scheduler is not None:
                recovery_payload["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(recovery_payload, recovery_path)
            log(
                f"Recovery checkpoint saved before rank-0 diagnostics: "
                f"{recovery_path}"
            )

        # Save raw embeddings periodically (for offline analysis) — rank 0 only
        if embed_interval > 0 and (epoch % embed_interval == 0 or is_final_epoch) and is_main:
            try:
                if _in_ray_tune_session():
                    from ray.tune import get_context as _get_tune_ctx
                    embed_dir = os.path.join(_get_tune_ctx().get_trial_dir(), "embeddings")
                else:
                    embed_dir = os.path.join(results_dir, "embeddings")
                os.makedirs(embed_dir, exist_ok=True)
                backbone_module.eval()
                all_emb, all_labels, all_coords = [], [], []
                with torch.no_grad():
                    for batch in val_loader_labeled:
                        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            waveforms, batch_labels = batch[0], batch[1]
                        else:
                            continue
                        waveforms = waveforms.float().to(device, non_blocking=True)
                        waveforms = preprocess_batch(waveforms, backbone_module.needs_resampling, backbone_module.needs_normalization)
                        pooled = backbone_module.encode(waveforms)
                        all_emb.append(pooled.cpu().numpy())
                        all_labels.append(batch_labels.cpu().numpy() if torch.is_tensor(batch_labels) else np.array(batch_labels))
                        if len(batch) > 2:
                            all_coords.append(batch[2].cpu().numpy())
                all_emb = np.concatenate(all_emb, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                save_path = os.path.join(embed_dir, f"embeddings_ep{epoch:04d}.npz")
                np.savez_compressed(save_path, embeddings=all_emb, labels=all_labels)
                log(f"Saved embeddings ({all_emb.shape}) to {save_path}")

                # Embedding std — mean per-dim std across batch (collapse detector)
                embed_std = float(all_emb.std(axis=0).mean())
                report_dict["embed_std"] = embed_std
                log(f"Embed std (mean per-dim): {embed_std:.4f}")
                if _trial_wandb or (wandb is not None and wandb.run is not None):
                    wandb.log({"embed_std": embed_std, "epoch": epoch})

                # Cosine similarity — mean pairwise cosine sim (soft collapse detector)
                # High cosine_sim (>0.98) means all embeddings point in same direction
                try:
                    n_cos = min(500, all_emb.shape[0])
                    idx_cos = np.random.choice(all_emb.shape[0], n_cos, replace=False)
                    emb_cos = all_emb[idx_cos]
                    norms_cos = np.linalg.norm(emb_cos, axis=1, keepdims=True)
                    emb_cos = emb_cos / np.maximum(norms_cos, 1e-8)
                    cos_matrix = emb_cos @ emb_cos.T
                    # Mean of upper triangle (excluding diagonal)
                    triu_idx = np.triu_indices(n_cos, k=1)
                    cosine_sim = float(cos_matrix[triu_idx].mean())
                    report_dict["cosine_sim"] = cosine_sim
                    log(f"Cosine sim (mean pairwise): {cosine_sim:.4f}")
                    if _trial_wandb or (wandb is not None and wandb.run is not None):
                        wandb.log({"cosine_sim": cosine_sim, "epoch": epoch})
                except Exception as e:
                    log(f"Warning: Failed to compute cosine_sim: {e}")

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
                        if _trial_wandb or (wandb is not None and wandb.run is not None):
                            wandb.log({"spatial_corr": spatial_corr, "epoch": epoch})
                    except Exception as e:
                        log(f"Warning: Failed to compute spatial_corr: {e}")

                backbone_module.train()
            except Exception as e:
                log(f"Warning: Failed to save embeddings: {e}")

        # Quick linear probe (skip if probe_epochs=0 or not on probe interval)
        # DDP: probe runs on rank 0 only; other ranks wait at a barrier below.
        probe_acc = None
        max_probe_samples = int(config.get("max_probe_samples", 0))

        if probe_epoch_count <= 0:
            log(f"[LinearProbe] Skipped (probe_epochs=0)")
        elif not is_probe_epoch:
            log(f"[LinearProbe] Skipped (next at epoch {((epoch // probe_interval) + 1) * probe_interval})")
        elif not is_main:
            pass  # non-main ranks skip probe; they sync at the barrier below
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

                        # Per-class accuracy + confusion matrix — only when
                        # num_classes is small enough to be readable (≤50).
                        # At depth 9 with 392 classes, these are pure noise.
                        n_cls = int(val_true.max().item()) + 1
                        if n_cls <= 50:
                            remap_inv = {v: k for k, v in label_remap.items()}
                            if atlas_depth is not None and hasattr(train_dataset_labeled, 'effective_label_map') and train_dataset_labeled.effective_label_map:
                                orig_id_to_name = {v: k for k, v in train_dataset_labeled.effective_label_map.items()}
                            else:
                                orig_id_to_name = atlas_region_id_to_name or REGION_ID_TO_NAME
                            class_names = [orig_id_to_name.get(remap_inv.get(i, -1), f"Class {i}") for i in range(n_cls)]

                            # Per-class accuracy breakdown
                            for cls_idx in range(n_cls):
                                cls_mask = val_true == cls_idx
                                if cls_mask.sum() > 0:
                                    cls_acc = (val_preds[cls_mask] == cls_idx).float().mean().item()
                                    cls_name = class_names[cls_idx]
                                    wandb.log({
                                        f"class_acc/{cls_name}": cls_acc,
                                        "epoch": epoch,
                                    })
            except Exception as e:
                log(f"[WARN] Linear probe failed at epoch {epoch}: {e}")

        # DDP: sync ranks after the rank-0-only probe block before continuing
        if is_ddp:
            dist.barrier()

        # Save best-probe-accuracy checkpoint — rank 0 only
        if is_main and probe_acc is not None and probe_acc > best_probe_acc:
            best_probe_acc = probe_acc
            if _in_ray_tune_session():
                from ray.tune import get_context as _get_tune_ctx
                ckpt_dir = os.path.join(_get_tune_ctx().get_trial_dir(), "checkpoints")
            else:
                ckpt_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            ckpt_data = {
                "backbone_state_dict": backbone_module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": float(val_loss),
                "loss_per_mask": loss_per_mask,
                "mask_factor": mask_factor,
                "probe_acc": probe_acc,
            }

            # Overwrite single best-probe checkpoint (no accumulation)
            probe_ckpt_path = os.path.join(ckpt_dir, f"ssl_{backbone_name}_best.pt")
            torch.save(ckpt_data, probe_ckpt_path)

            log(f"Best PROBE checkpoint saved to {probe_ckpt_path} "
                f"(epoch {epoch}, probe_acc={probe_acc:.4f}, val_loss={float(val_loss):.4f})")

        # Report to Ray Tune (rank 0 only)
        if _in_ray_tune_session() and is_main:
            tune.report(report_dict)

        # Explicit wandb.log for loss curves (works in both Ray Tune and standalone)
        if is_main and wandb is not None and wandb.run is not None:
            wandb_dict = {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "grad_norm": float(grad_norm),
                "lr": float(current_lr),
                "epoch": epoch,
            }
            # Wav2Vec2-specific metrics (contrastive/diversity/perplexity/gumbel)
            if train_w2v2_metrics:
                wandb_dict.update({f"train_{k}": v for k, v in train_w2v2_metrics.items()})
            wandb.log(wandb_dict)

        # Save best val_loss checkpoint (separate file, for reference) — rank 0 only
        val_loss_improved = float(val_loss) < best_val_loss
        if val_loss_improved:
            best_val_loss = float(val_loss)
            if is_main:
                if _in_ray_tune_session():
                    from ray.tune import get_context as _get_tune_ctx
                    ckpt_dir = os.path.join(_get_tune_ctx().get_trial_dir(), "checkpoints")
                else:
                    ckpt_dir = os.path.join(results_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)

                valloss_data = {
                    "backbone_state_dict": backbone_module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": float(val_loss),
                    "loss_per_mask": loss_per_mask,
                    "mask_factor": mask_factor,
                }

                # Overwrite single best-valloss checkpoint (no accumulation)
                ckpt_path = os.path.join(ckpt_dir, f"ssl_{backbone_name}_best_valloss.pt")
                torch.save(valloss_data, ckpt_path)

                log(f"Best val_loss checkpoint saved to {ckpt_path} (epoch {epoch}, val_loss={float(val_loss):.4f})")

        # Save "latest" checkpoint for resume (every epoch, overwritten) — rank 0 only.
        # scheduler may be None in plateau mode after warmup completes (and
        # also after lr_override warmup hands LR back to plateau), so gate
        # its state_dict.
        if is_main and not _in_ray_tune_session():
            latest_ckpt_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(latest_ckpt_dir, exist_ok=True)
            latest_path = os.path.join(latest_ckpt_dir, f"ssl_{backbone_name}_latest.pt")
            ckpt_payload = {
                "backbone_state_dict": backbone_module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": float(val_loss),
                "best_val_loss": best_val_loss,
                "best_probe_acc": best_probe_acc,
                "loss_per_mask": loss_per_mask,
                "mask_factor": mask_factor,
            }
            if scheduler is not None:
                ckpt_payload["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(ckpt_payload, latest_path)

        # Step plateau scheduler per-epoch (adapts LR to val_loss)
        if epoch_scheduler is not None:
            epoch_scheduler.step(float(val_loss))

        # LR-override warmup accounting — null the per-step scheduler once the
        # post-resume warmup is exhausted so ReduceLROnPlateau can freely adjust
        # the LR without being overwritten by the warmup scheduler's next step().
        if lr_override_remaining_warmup > 0:
            lr_override_remaining_warmup -= steps_per_epoch
            if lr_override_remaining_warmup <= 0:
                scheduler = None
                log(f"  [warmup] LR override warmup complete; scheduler handed "
                    f"over to plateau. LR = {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping counter
        if val_loss_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            log(f"Early stopping at epoch {epoch}: no val_loss improvement for "
                f"{early_stop_patience} epochs (best={best_val_loss:.4f}, current={float(val_loss):.4f})")
            break

        # Log to wandb (standalone mode) — rank 0 only
        if is_main and not _in_ray_tune_session() and wandb is not None and wandb.run is not None:
            wandb.log(report_dict)

    # Cleanup per-trial wandb run
    if _trial_wandb:
        try:
            wandb.finish()
        except Exception:
            pass

    # DDP cleanup
    if is_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


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
    parser.add_argument("--max_probe_samples", type=int, default=None,
                        help="Cap the periodic linear-probe extraction (0 = full dataset)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint .pt file to resume training from")
    # Resume-time LR override. When resuming from a checkpoint where plateau
    # had driven LR into the floor, this bumps LR back up so the model has
    # headroom to keep learning. Applied AFTER optimizer state is loaded.
    parser.add_argument("--lr_override", type=float, default=None,
                        help="Override optimizer LR after resume (e.g., 1e-4).")
    parser.add_argument("--lr_override_warmup_steps", type=int, default=200,
                        help="Linear warmup steps when lr_override is active.")
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
    if args.max_probe_samples is not None:
        config["max_probe_samples"] = args.max_probe_samples
    if args.resume_from is not None:
        config["resume_from"] = args.resume_from
    if args.lr_override is not None:
        config["lr_override"] = args.lr_override
    config["lr_override_warmup_steps"] = args.lr_override_warmup_steps

    # Initialize wandb for standalone mode — rank 0 only (DDP-aware).
    # Note: ssl_trial_trainable calls _init_distributed() internally, but at
    # this point we can already detect rank via env (set by torchrun).
    _wandb_ok = False
    if wandb is not None and not _in_ray_tune_session() and _is_main_process():
        try:
            _wb_id = os.environ.get("WANDB_RUN_ID") or None  # fixed id -> one run across restarts
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "lfp2vec_training"),
                entity=os.environ.get("WANDB_ENTITY", None),
                group=os.environ.get("WANDB_GROUP", f"ssl_full_{args.backbone}"),
                name=os.environ.get("WANDB_RUN_NAME", f"ssl_full_{args.backbone}"),
                id=_wb_id,
                resume="allow" if _wb_id else None,
                config=config,
                allow_val_change=True,
            )
            _wandb_ok = True
        except Exception as e:
            if os.environ.get("WANDB_REQUIRED", "0") == "1":
                raise RuntimeError(f"W&B init failed and WANDB_REQUIRED=1: {e}") from e
            print(f"[WARN] wandb init failed (continuing without W&B): {e}")

    ssl_trial_trainable(config)

    if _wandb_ok and wandb is not None and wandb.run is not None:
        wandb.finish()

    # Post-training eval runs on rank 0 only (single-process subprocess).
    if args.run_eval_after_training and _is_main_process():
        _run_post_training_eval(args, config)


if __name__ == "__main__":
    main()
