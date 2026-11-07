#!/usr/bin/env python3
"""Fine-tune backbone for XYZ coordinate regression with Gaussian NLL loss.

Stage 2 of the Voxelizer pipeline:
    SSL pre-training → **Gaussian XYZ fine-tuning** → Voxelizer

Unlike the MSE/SmoothL1 finetune script, this predicts (μ, log σ²) per sample
and trains with Gaussian NLL. The model learns per-sample uncertainty, which:
  - Down-weights ambiguous samples (e.g., region boundaries)
  - Produces calibrated uncertainty estimates for downstream use
  - Provides (μ, σ²) as input to the Voxelizer (Stage 3)

Usage:
    python3 finetune_xyz_gaussian.py \
        --backbone wav2vec2 \
        --config configs/wav2vec2_probe_dist_v2_embedonly_full.yaml \
        --checkpoint /path/to/ssl_wav2vec2_best_valloss.pt \
        --output-dir /path/to/output/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from scipy.stats import spearmanr
from sklearn.metrics import r2_score

import yaml

# Project imports
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA_ROOT = os.path.dirname(BENCH_ROOT)
sys.path.insert(0, BENCH_ROOT)
sys.path.insert(0, ALPHA_ROOT)

from blind_localization.data.lazyloader_dataset import CompactDataset
from backbones import get_backbone
from utils.preprocessing import preprocess_batch

# ── Constants ──
LP_EVAL_DIR = os.path.join(
    ALPHA_ROOT, "blind_localization/data/atlas_data/compact_atlas_lp_eval"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AXIS_NAMES = ["AP", "DV", "ML"]


def detect_splits(data_dir):
    """Auto-detect train/val/test-fold naming: Allen-style (train/val/test_fold{1..5})
    or IBL-style (probe_train/probe_val/test_fold_{0..4}).

    Returns (train_dir_name, val_dir_name, test_prefix, test_indices).
    """
    if os.path.isdir(os.path.join(data_dir, "probe_train")):
        return "probe_train", "probe_val", "test_fold_", list(range(5))
    return "train", "val", "test_fold", list(range(1, 6))

# Clamp log_var to prevent numerical issues
LOG_VAR_MIN = -10.0  # σ² ≥ ~4.5e-5 (in normalized space)
LOG_VAR_MAX = 10.0   # σ² ≤ ~22026


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Gaussian XYZ Regression Head
# ═══════════════════════════════════════════════════════════════════

class GaussianXYZHead(nn.Module):
    """MLP head that predicts (μ, log σ²) for 3D coordinates.

    Output: 6 values per sample — 3 means (unconstrained, z-scored target
    space) and 3 log-variances (unconstrained, clamped for stability).
    """

    def __init__(self, input_dim=768):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.mu_head = nn.Linear(128, 3)
        self.logvar_head = nn.Linear(128, 3)

    def forward(self, x):
        """Returns (mu, log_var), each shape [B, 3]."""
        h = self.shared(x)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h)
        log_var = log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        return mu, log_var


class GaussianXYZFineTuneModel(nn.Module):
    """Backbone + Gaussian XYZ head for end-to-end fine-tuning."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, waveforms):
        emb = self.backbone.encode(waveforms)
        return self.head(emb)


# ═══════════════════════════════════════════════════════════════════
# Loss Function
# ═══════════════════════════════════════════════════════════════════

class GaussianNLLLoss3D(nn.Module):
    """Gaussian negative log-likelihood for 3D coordinates.

    L = 0.5 * mean_over_batch[ sum_d( log_var_d + (y_d - mu_d)^2 / exp(log_var_d) ) ]

    The constant 0.5 * 3 * log(2π) is omitted (doesn't affect optimization).
    """

    def forward(self, mu, log_var, target):
        """
        Args:
            mu: [B, 3] predicted means (normalized coords)
            log_var: [B, 3] predicted log-variances
            target: [B, 3] ground-truth (normalized coords)
        Returns:
            Scalar loss
        """
        var = log_var.exp()
        nll = 0.5 * (log_var + (target - mu) ** 2 / var)
        return nll.sum(dim=1).mean()


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def make_dataloader(split_dir, batch_size, num_workers, shuffle):
    ds = CompactDataset(
        split_dir, include_labels=True, gpu_resample=True, atlas_depth=9
    )
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return dl, ds


# ═══════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion,
                    y_mean, y_std, needs_resample):
    model.train()
    total_loss = 0.0
    n_samples = 0

    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=DEVICE)
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=DEVICE)

    for batch in dataloader:
        waveforms, labels, coords = batch[0], batch[1], batch[2]
        waveforms = waveforms.float().to(DEVICE, non_blocking=True)
        waveforms = preprocess_batch(waveforms, needs_resample)
        coords = coords.float().to(DEVICE, non_blocking=True)

        # Normalize targets
        coords_n = (coords - y_mean_t) / y_std_t

        mu, log_var = model(waveforms)
        loss = criterion(mu, log_var, coords_n)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = waveforms.shape[0]
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, dataloader, y_mean, y_std, needs_resample):
    model.eval()
    all_mu = []
    all_logvar = []
    all_true = []

    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=DEVICE)
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=DEVICE)

    for batch in dataloader:
        waveforms, labels, coords = batch[0], batch[1], batch[2]
        waveforms = waveforms.float().to(DEVICE, non_blocking=True)
        waveforms = preprocess_batch(waveforms, needs_resample)
        coords = coords.float().to(DEVICE, non_blocking=True)

        mu_n, log_var_n = model(waveforms)
        # De-normalize mean predictions to µm
        mu = mu_n * y_std_t + y_mean_t
        # De-normalize variance: var_µm = var_normalized * std_µm²
        # log_var_µm = log_var_n + 2*log(y_std)
        log_var = log_var_n + 2 * torch.log(y_std_t)

        all_mu.append(mu.cpu().numpy())
        all_logvar.append(log_var.cpu().numpy())
        all_true.append(coords.cpu().numpy())

    preds = np.concatenate(all_mu, 0)
    logvars = np.concatenate(all_logvar, 0)
    true = np.concatenate(all_true, 0)
    stds = np.sqrt(np.exp(logvars))  # σ in µm

    # Spatial metrics
    r2 = r2_score(true, preds, multioutput="uniform_average")
    errors = np.sqrt(((preds - true) ** 2).sum(axis=1))
    mean_err = errors.mean()
    median_err = np.median(errors)
    p95_err = np.percentile(errors, 95)

    rho_vals = []
    for ax in range(3):
        rho, _ = spearmanr(true[:, ax], preds[:, ax])
        rho_vals.append(rho)
    mean_rho = np.mean(rho_vals)

    # Uncertainty metrics
    mean_std = stds.mean(axis=0)  # per-axis mean σ in µm

    # Calibration: fraction of samples within predicted 1σ/2σ (per-axis box)
    # Ideal for independent axes: 1σ → 0.6827^3 ≈ 31.8%, 2σ → 0.9545^3 ≈ 86.9%
    residuals = np.abs(preds - true)  # [N, 3]
    within_1sigma = (residuals < stds).all(axis=1).mean()
    within_2sigma = (residuals < 2 * stds).all(axis=1).mean()

    # Negative log-likelihood (proper scoring rule)
    nll = 0.5 * (logvars + (true - preds) ** 2 / np.exp(logvars)).sum(axis=1).mean()

    return {
        "r2": r2,
        "mean_rho": mean_rho,
        "mean_err": mean_err,
        "median_err": median_err,
        "p95_err": p95_err,
        "nll": float(nll),
        "mean_std_AP": float(mean_std[0]),
        "mean_std_DV": float(mean_std[1]),
        "mean_std_ML": float(mean_std[2]),
        "within_1sigma": float(within_1sigma),
        "within_2sigma": float(within_2sigma),
        "preds": preds,
        "true": true,
        "stds": stds,
    }


# ═══════════════════════════════════════════════════════════════════
# Backbone Parameter Grouping
# ═══════════════════════════════════════════════════════════════════

def build_param_groups(model, lr_head, lr_backbone, lr_cnn, backbone_type,
                       llrd_decay=1.0):
    """Build discriminative learning rate groups for backbone + head.

    Handles both wav2vec2 (model.wav2vec2.*) and whisper (student.*/
    feature_extractor.*) naming conventions. Freezes SSL-only auxiliary
    modules (teacher, predictor, decoder, SSL heads) that are not needed
    for downstream fine-tuning.

    llrd_decay: per-layer LR decay for encoder (1.0 = no decay).
        Layer i gets lr = lr_backbone * llrd_decay^(n_layers-1-i).
        Top layer (closest to head) keeps lr_backbone; lower layers get
        exponentially smaller LRs. CNN gets lr_cnn (independent of LLRD).
    """
    WHISPER_FREEZE_PREFIXES = (
        "teacher.", "predictor.", "decoder.", "mask_emb",
        "xyz_head.", "dist_pred_head.", "moco_queue", "queue_ptr",
    )

    import re as _re

    cnn_params = []
    # For LLRD: dict from layer_idx → list of params
    layer_params: dict = {}
    other_encoder_params = []  # positional embeddings, layer_norm, etc.
    head_params = list(model.head.parameters())

    for name, p in model.backbone.named_parameters():
        is_cnn = False
        layer_idx = None
        is_frozen_aux = False

        if backbone_type == "wav2vec2":
            if "feature_extractor" in name or "feature_projection" in name:
                is_cnn = True
            elif "encoder.layers." in name:
                m = _re.search(r"encoder\.layers\.(\d+)\.", name)
                layer_idx = int(m.group(1)) if m else None
            elif "encoder" in name:
                # pos_conv_embed, layer_norm, etc.
                pass  # goes to other_encoder_params
            else:
                is_frozen_aux = True
        else:
            if any(name.startswith(pfx) for pfx in WHISPER_FREEZE_PREFIXES):
                is_frozen_aux = True
            elif "feature_extractor" in name:
                is_cnn = True
            elif "student" in name or "encoder" in name:
                m = _re.search(r"\.layers\.(\d+)\.", name)
                layer_idx = int(m.group(1)) if m else None
            else:
                is_frozen_aux = True

        if is_frozen_aux:
            p.requires_grad = False
        elif is_cnn:
            if lr_cnn > 0:
                p.requires_grad = True
                cnn_params.append(p)
            else:
                p.requires_grad = False
        elif layer_idx is not None:
            if lr_backbone > 0:
                p.requires_grad = True
                layer_params.setdefault(layer_idx, []).append(p)
            else:
                p.requires_grad = False
        else:
            # positional embeddings, final layer_norm, etc.
            if lr_backbone > 0:
                p.requires_grad = True
                other_encoder_params.append(p)
            else:
                p.requires_grad = False

    groups = []

    if cnn_params:
        groups.append({"params": cnn_params, "lr": lr_cnn})
        log(f"  CNN params: {sum(p.numel() for p in cnn_params)/1e6:.1f}M, lr={lr_cnn:.2e}")
    else:
        log(f"  CNN: frozen (lr=0)")

    if layer_params:
        n_layers = max(layer_params.keys()) + 1
        total_enc = 0
        for idx in sorted(layer_params.keys()):
            layer_lr = lr_backbone * (llrd_decay ** (n_layers - 1 - idx))
            n = sum(p.numel() for p in layer_params[idx])
            groups.append({"params": layer_params[idx], "lr": layer_lr})
            total_enc += n
        if llrd_decay < 1.0:
            bot_lr = lr_backbone * (llrd_decay ** (n_layers - 1))
            log(f"  Encoder: {n_layers} layers, LLRD decay={llrd_decay}, "
                f"lr top={lr_backbone:.2e} → bottom={bot_lr:.2e}, "
                f"total={total_enc/1e6:.1f}M params")
        else:
            log(f"  Encoder params: {total_enc/1e6:.1f}M, lr={lr_backbone:.2e}")
        if other_encoder_params:
            groups.append({"params": other_encoder_params, "lr": lr_backbone})
    elif lr_backbone > 0 and other_encoder_params:
        groups.append({"params": other_encoder_params, "lr": lr_backbone})
        log(f"  Encoder (non-layer) params: {sum(p.numel() for p in other_encoder_params)/1e6:.1f}M, lr={lr_backbone:.2e}")
    else:
        log(f"  Encoder: frozen (lr=0)")

    groups.append({"params": head_params, "lr": lr_head})
    log(f"  Head params: {sum(p.numel() for p in head_params)/1e3:.1f}K, lr={lr_head:.2e}")

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Total trainable: {total_trainable/1e6:.1f}M")

    return groups


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune backbone for XYZ regression with Gaussian NLL loss"
    )
    parser.add_argument("--backbone", type=str, default="wav2vec2")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-3,
                        help="Learning rate for Gaussian XYZ head")
    parser.add_argument("--lr-backbone", type=float, default=5e-6,
                        help="Learning rate for backbone encoder")
    parser.add_argument("--lr-cnn", type=float, default=0.0,
                        help="Learning rate for CNN feature extractor (0=frozen)")
    parser.add_argument("--llrd-decay", type=float, default=1.0,
                        help="Per-layer LR decay for encoder (e.g. 0.8). "
                             "Top layer keeps lr-backbone; each lower layer "
                             "is multiplied by this factor. 1.0=no decay.")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <output-dir>/latest_ckpt.pt if present")
    parser.add_argument("--data-dir", type=str, default=LP_EVAL_DIR,
                        help="Dataset root (auto-detects Allen-style vs IBL-style "
                             "train/val/test-fold naming)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log(f"Device: {DEVICE}")
    log(f"Output: {args.output_dir}")

    # ── Load config and backbone ──
    with open(args.config) as f:
        config = yaml.safe_load(f)

    log(f"Loading backbone: {args.backbone}")
    log(f"Config: {args.config}")
    log(f"Checkpoint: {args.checkpoint}")

    backbone = get_backbone(args.backbone, config)
    if args.checkpoint.lower() == "none":
        log("No SSL checkpoint — using raw HF pretrained weights")
    else:
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "backbone_state_dict" in raw:
            state = raw["backbone_state_dict"]
            log(f"Loaded backbone_state_dict from wrapper (epoch={raw.get('epoch', '?')})")
        elif isinstance(raw, dict) and "model_state_dict" in raw:
            state = raw["model_state_dict"]
            log(f"Loaded model_state_dict from wrapper")
        else:
            state = raw
            log(f"Loaded raw state_dict")
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing:
            log(f"  WARNING: {len(missing)} missing keys (first 3: {missing[:3]})")
        if unexpected:
            log(f"  WARNING: {len(unexpected)} unexpected keys (first 3: {unexpected[:3]})")
    backbone = backbone.to(DEVICE)
    needs_resample = backbone.needs_resampling
    hidden_dim = backbone.hidden_dim

    log(f"Backbone loaded: hidden_dim={hidden_dim}, "
        f"params={backbone.get_num_params(trainable_only=False)/1e6:.1f}M")

    # ── Build model ──
    log("\n" + "=" * 70)
    log("  GAUSSIAN XYZ FINE-TUNING")
    log("=" * 70)

    head = GaussianXYZHead(input_dim=hidden_dim)
    model = GaussianXYZFineTuneModel(backbone, head).to(DEVICE)
    criterion = GaussianNLLLoss3D()
    log(f"  Loss: GaussianNLL (heteroscedastic)")

    # Optimizer with discriminative LR (+ optional LLRD)
    param_groups = build_param_groups(
        model, args.lr_head, args.lr_backbone, args.lr_cnn, args.backbone,
        llrd_decay=args.llrd_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Create dataloaders (auto-detect Allen-style vs IBL-style splits)
    train_name, val_name, test_prefix, test_indices = detect_splits(args.data_dir)
    log(f"  Data dir: {args.data_dir}")
    log(f"  Splits: train={train_name}/ val={val_name}/ test={test_prefix}{{{test_indices[0]}..{test_indices[-1]}}}/")
    train_dl, _ = make_dataloader(
        os.path.join(args.data_dir, train_name),
        args.batch_size, args.num_workers, shuffle=True,
    )
    val_dl, _ = make_dataloader(
        os.path.join(args.data_dir, val_name),
        args.batch_size, args.num_workers, shuffle=False,
    )

    # Compute normalization stats from first pass through train
    log("  Computing coord normalization stats...")
    all_coords = []
    for batch in train_dl:
        all_coords.append(batch[2].numpy())
    all_coords = np.concatenate(all_coords, 0)
    y_mean = all_coords.mean(axis=0).astype(np.float32)
    y_std = all_coords.std(axis=0).astype(np.float32)
    log(f"  y_mean={y_mean}, y_std={y_std}")
    del all_coords

    # Save normalization stats (needed by Stage 3 Voxelizer)
    norm_path = os.path.join(args.output_dir, "coord_normalization.json")
    with open(norm_path, "w") as f:
        json.dump({"y_mean": y_mean.tolist(), "y_std": y_std.tolist()}, f)
    log(f"  Saved normalization stats to {norm_path}")

    # LR scheduler: warmup + cosine
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    warmup_sched = LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )

    # ── Training loop ──
    best_val_err = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    start_epoch = 0

    latest_path = os.path.join(args.output_dir, "latest_ckpt.pt")
    if args.resume and os.path.isfile(latest_path):
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_err = ckpt["best_val_err"]
        best_epoch = ckpt["best_epoch"]
        no_improve = ckpt["no_improve"]
        best_path = os.path.join(args.output_dir, "best_gaussian_xyz_model.pt")
        if os.path.isfile(best_path):
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_state = {k: v for k, v in best_ckpt["model_state_dict"].items()}
        log(f"  Resumed from epoch {start_epoch}, best_val={best_val_err:.0f}µm "
            f"@ ep{best_epoch}, no_improve={no_improve}")

    epoch = start_epoch - 1  # ensure defined if resume already completed all epochs
    if no_improve >= args.patience:
        log(f"  Early-stop state restored (no_improve={no_improve} ≥ patience={args.patience}); skipping training loop")
    for epoch in range(start_epoch, args.epochs):
        if no_improve >= args.patience:
            break  # already early-stopped in a prior run
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_dl, optimizer, scheduler, criterion,
            y_mean, y_std, needs_resample,
        )

        val_result = evaluate(model, val_dl, y_mean, y_std, needs_resample)
        elapsed = time.time() - t0

        cur_lr = optimizer.param_groups[-1]["lr"]  # head LR
        log(f"  Epoch {epoch+1:3d}/{args.epochs}: "
            f"train_nll={train_loss:.4f}  "
            f"val_R²={val_result['r2']:.4f}  "
            f"val_err={val_result['mean_err']:.0f}µm  "
            f"val_med={val_result['median_err']:.0f}µm  "
            f"σ_AP={val_result['mean_std_AP']:.0f}  "
            f"σ_DV={val_result['mean_std_DV']:.0f}  "
            f"σ_ML={val_result['mean_std_ML']:.0f}  "
            f"1σ={val_result['within_1sigma']:.1%}  "
            f"2σ={val_result['within_2sigma']:.1%}  "
            f"lr={cur_lr:.2e}  "
            f"time={elapsed:.0f}s")

        if val_result["mean_err"] < best_val_err:
            best_val_err = val_result["mean_err"]
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            log(f"    *** New best: {best_val_err:.0f}µm (epoch {best_epoch})")
            # Save checkpoint incrementally (survives job kill / timeout)
            _bsd = {k.replace("backbone.", "", 1): v
                    for k, v in best_state.items() if k.startswith("backbone.")}
            _hsd = {k.replace("head.", "", 1): v
                    for k, v in best_state.items() if k.startswith("head.")}
            torch.save({
                "model_state_dict": best_state,
                "backbone_state_dict": _bsd,
                "head_state_dict": _hsd,
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
                "hidden_dim": hidden_dim,
                "backbone": args.backbone,
                "best_epoch": best_epoch,
                "best_val_err": best_val_err,
            }, os.path.join(args.output_dir, "best_gaussian_xyz_model.pt"))
        else:
            no_improve += 1

        # Latest checkpoint for resume (overwritten each epoch)
        torch.save({
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_err": best_val_err,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
            "y_mean": y_mean.tolist(),
            "y_std": y_std.tolist(),
        }, os.path.join(args.output_dir, "latest_ckpt.pt"))

        if no_improve >= args.patience:
            log(f"  Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    # Load best model (fallback to current weights if no improvement ever happened)
    if best_state is None:
        log("  WARNING: no best_state captured; using current model weights for test eval")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(DEVICE)

    # ── Test evaluation ──
    log(f"\n  Best model from epoch {best_epoch} (val_err={best_val_err:.0f}µm)")
    log("  Evaluating on test folds...")

    test_results = []
    all_test_preds = []
    all_test_true = []
    all_test_stds = []

    for fold in test_indices:
        fold_dir = os.path.join(args.data_dir, f"{test_prefix}{fold}")
        if not os.path.isdir(fold_dir):
            continue
        test_dl, _ = make_dataloader(
            fold_dir, args.batch_size, args.num_workers, shuffle=False
        )
        result = evaluate(model, test_dl, y_mean, y_std, needs_resample)
        test_results.append(result)
        all_test_preds.append(result["preds"])
        all_test_true.append(result["true"])
        all_test_stds.append(result["stds"])
        log(f"  test_fold{fold}: R²={result['r2']:.4f}  "
            f"ρ={result['mean_rho']:.4f}  "
            f"err={result['mean_err']:.0f}µm (med={result['median_err']:.0f})  "
            f"NLL={result['nll']:.2f}  "
            f"1σ={result['within_1sigma']:.1%}  "
            f"2σ={result['within_2sigma']:.1%}")

    if test_results:
        mean_r2 = np.mean([r["r2"] for r in test_results])
        mean_rho = np.mean([r["mean_rho"] for r in test_results])
        mean_err = np.mean([r["mean_err"] for r in test_results])
        mean_p95 = np.mean([r["p95_err"] for r in test_results])
        mean_nll = np.mean([r["nll"] for r in test_results])
        mean_1sig = np.mean([r["within_1sigma"] for r in test_results])
        mean_2sig = np.mean([r["within_2sigma"] for r in test_results])
        mean_std_per_ax = np.mean(
            [[r["mean_std_AP"], r["mean_std_DV"], r["mean_std_ML"]]
             for r in test_results], axis=0
        )
        log(f"\n  TEST MEAN: R²={mean_r2:.4f}  ρ={mean_rho:.4f}  "
            f"err={mean_err:.0f}µm  p95={mean_p95:.0f}µm  NLL={mean_nll:.2f}")
        log(f"  Uncertainty: σ_AP={mean_std_per_ax[0]:.0f}µm  "
            f"σ_DV={mean_std_per_ax[1]:.0f}µm  "
            f"σ_ML={mean_std_per_ax[2]:.0f}µm")
        log(f"  Calibration: {mean_1sig:.1%} within 1σ (ideal 31.8%)  "
            f"{mean_2sig:.1%} within 2σ (ideal 86.9%)")

    # ── Save results ──
    summary = {
        "backbone": args.backbone,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "loss": "gaussian_nll",
        "best_epoch": best_epoch,
        "best_val_err": float(best_val_err),
        "lr_head": args.lr_head,
        "lr_backbone": args.lr_backbone,
        "lr_cnn": args.lr_cnn,
        "batch_size": args.batch_size,
        "epochs_trained": epoch + 1,
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "test_mean_r2": float(mean_r2) if test_results else None,
        "test_mean_rho": float(mean_rho) if test_results else None,
        "test_mean_err": float(mean_err) if test_results else None,
        "test_mean_p95": float(mean_p95) if test_results else None,
        "test_mean_nll": float(mean_nll) if test_results else None,
        "test_mean_1sigma": float(mean_1sig) if test_results else None,
        "test_mean_2sigma": float(mean_2sig) if test_results else None,
        "test_mean_std_AP": float(mean_std_per_ax[0]) if test_results else None,
        "test_mean_std_DV": float(mean_std_per_ax[1]) if test_results else None,
        "test_mean_std_ML": float(mean_std_per_ax[2]) if test_results else None,
    }
    out_path = os.path.join(args.output_dir, "finetune_gaussian_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n  Results saved to {out_path}")

    # Save model weights
    model_path = os.path.join(args.output_dir, "best_gaussian_xyz_model.pt")
    # Strip prefixes so each sub-dict loads standalone
    backbone_sd = {k.replace("backbone.", "", 1): v
                   for k, v in best_state.items() if k.startswith("backbone.")}
    head_sd = {k.replace("head.", "", 1): v
               for k, v in best_state.items() if k.startswith("head.")}
    torch.save({
        "model_state_dict": best_state,
        "backbone_state_dict": backbone_sd,
        "head_state_dict": head_sd,
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "hidden_dim": hidden_dim,
        "backbone": args.backbone,
        "best_epoch": best_epoch,
    }, model_path)
    log(f"  Model saved to {model_path}")

    # Save predictions + uncertainties (for Stage 3 / analysis)
    if all_test_preds:
        pred_path = os.path.join(args.output_dir, "gaussian_predictions.npz")
        np.savez(
            pred_path,
            preds=np.concatenate(all_test_preds, 0),     # [N, 3] µm
            true=np.concatenate(all_test_true, 0),        # [N, 3] µm
            stds=np.concatenate(all_test_stds, 0),        # [N, 3] µm
        )
        log(f"  Predictions + uncertainties saved to {pred_path}")


if __name__ == "__main__":
    main()
