#!/usr/bin/env python3
"""SSL training with channel distance contrastive auxiliary loss.

Single forward pass — channel distance computed inside ssl_forward_detailed
using output_hidden_states=True. No second encoder call needed.

Key fix from v1: adaptive loss scaling. dist_loss ≈ 2.7 (KL divergence),
ssl_loss ≈ 2200. Old fixed λ=0.1 meant dist was 0.01% of gradient (invisible).
Now: dist is scaled to be ~target_fraction (10%) of SSL loss.

Usage:
    python train_ssl_channel_dist.py \
        --backbone wav2vec2 \
        --config configs/wav2vec2_ibl_v2_25k_ssl_crossneg.yaml \
        --compact_data_path data/ibl/subject_split_v2/compact_atlas_ibl_v2_25k \
        --results_dir /path/to/results \
        --dist_target_fraction 0.1
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ALPHABRAIN_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ALPHABRAIN_ROOT)

from utils.probe_distance_sampler import (
    SameProbeBatchSampler,
    DistributedSameProbeBatchSampler,
    ChannelAwareDataset,
    ChannelDistanceContrastiveLoss,
)
from utils.preprocessing import preprocess_batch
from backbones import get_backbone

from train_ssl import log, _quick_linear_probe, _init_distributed, _is_main_process, _unwrap


def _ddp_mean(value: float, device, is_ddp: bool) -> float:
    """All-reduce a scalar to its cross-rank MEAN (no-op in single-GPU mode).

    Makes val_loss identical on every rank before the early-stop / plateau
    decision so all ranks `break` in lockstep (a divergent break would
    deadlock the next collective). Each rank computes val_loss over the full
    val set with its own (rank-offset, stochastic-masking) RNG, so the values
    differ slightly; averaging synchronizes the decision and denoises it.
    """
    if not is_ddp:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / dist.get_world_size()


def train_one_epoch(
    backbone, train_loader, optimizer, scheduler, scaler,
    device, use_amp, grad_clip_norm,
    channel_dist_loss_fn, dist_target_fraction,
    ssl_loss_weight=1.0,
):
    """Single-pass training with adaptive channel distance scaling."""
    backbone.train()
    inner = _unwrap(backbone)  # for attribute access; backbone(...) routes through DDP
    total_ssl_loss = 0.0
    total_ch_loss = 0.0
    total_grad_norm = 0.0
    n_batches = 0

    for batch in train_loader:
        input_values = batch[0].float().to(device, non_blocking=True)
        channel_numbers = batch[-1].float().to(device, non_blocking=True)

        input_values = preprocess_batch(
            input_values,
            inner.needs_resampling,
            inner.needs_normalization,
        )

        optimizer.zero_grad(set_to_none=True)

        # Single forward pass with output_hidden_states. Route through
        # backbone(...) (mode=...) so DDP installs its grad-sync hooks — both
        # ssl_loss and the CD loss (computed below from chunk_embeds, a forward
        # output) backprop through this graph, so a single backward syncs both.
        result = backbone(input_values, mode="ssl_forward_detailed")
        ssl_loss = result["ssl_loss"]
        chunk_embeds = result.get("chunk_embeds")

        if ssl_loss is None or not torch.isfinite(ssl_loss):
            continue

        ssl_loss_val = ssl_loss.item()

        # Channel distance loss on chunk_embeds
        ch_loss_val = 0.0
        if chunk_embeds is not None and dist_target_fraction > 0:
            ch_loss = channel_dist_loss_fn(chunk_embeds, channel_numbers)
            ch_loss_val = ch_loss.item()

            # Adaptive scaling: dist contributes target_fraction of SSL magnitude
            if ch_loss_val > 1e-8:
                adaptive_lambda = dist_target_fraction * ssl_loss_val / ch_loss_val
            else:
                adaptive_lambda = 0.0

            total_loss = ssl_loss_weight * ssl_loss + adaptive_lambda * ch_loss
        else:
            total_loss = ssl_loss_weight * ssl_loss

        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(inner.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(inner.parameters(), grad_clip_norm)
            optimizer.step()

        grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in inner.parameters() if p.grad is not None
        ) ** 0.5
        total_grad_norm += grad_norm

        if scheduler is not None:
            scheduler.step()

        if hasattr(inner, "step_gumbel_temperature"):
            inner.step_gumbel_temperature()

        total_ssl_loss += ssl_loss_val
        total_ch_loss += ch_loss_val
        n_batches += 1

    n = max(n_batches, 1)
    return total_ssl_loss / n, total_ch_loss / n, total_grad_norm / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--compact_data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--ssl_epochs", type=int, default=0)
    parser.add_argument("--embed_interval", type=int, default=5)
    parser.add_argument("--resume_from", type=str, default=None)

    # Channel distance params
    parser.add_argument("--channel_dist_sigma", type=float, default=1000.0)
    parser.add_argument("--channel_dist_temperature", type=float, default=0.2)
    parser.add_argument("--dist_target_fraction", type=float, default=0.1,
                        help="Dist loss = this fraction of SSL loss (adaptive scaling)")
    parser.add_argument("--samples_per_probe", type=int, default=16)
    parser.add_argument("--num_probes", type=int, default=4)
    parser.add_argument("--ssl_loss_weight", type=float, default=1.0,
                        help="Weight on SSL contrastive loss. 0.0 disables SSL "
                        "(CD-only ablation: train backbone purely from CD signal). "
                        "Adaptive λ for CD still computed against ssl_loss_val so "
                        "CD gradient scale matches the SSL+CD recipe.")

    # Combined IBL + Allen training. If set, Allen batches are interleaved
    # with IBL batches (round-robin per step). Val/probe stay IBL-only so
    # val_loss and probe_bal_acc stay directly comparable to IBL baselines.
    parser.add_argument("--allen_compact_data_path", type=str, default=None,
                        help="Path to teammate's Allen compact dataset root.")
    parser.add_argument("--allen_channels_csv", type=str,
                        default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/data/channels.csv'),
                        help="Allen channels.csv: id → local_index (0-383) for CD geometry.")

    # Resume-time LR override. When resuming from a checkpoint where plateau
    # had driven LR into the floor, this bumps LR back up so the model has
    # headroom to keep learning. Applied AFTER optimizer state is loaded.
    parser.add_argument("--lr_override", type=float, default=None,
                        help="Override optimizer LR after resume (e.g., 1e-4).")
    parser.add_argument("--lr_override_warmup_steps", type=int, default=200,
                        help="Linear warmup steps when lr_override is active.")

    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # channel_distance=True triggers hidden state extraction in backbone
    # dist_lambda stays 0 — adaptive scaling is handled in this training script
    config["channel_distance"] = True

    if args.ssl_epochs > 0:
        config["ssl_epochs"] = args.ssl_epochs
    if args.resume_from:
        config["resume_from"] = args.resume_from

    compact_data_path = args.compact_data_path or config.get("compact_data_path", "")
    config["compact_data_path"] = compact_data_path

    # ---- DDP init (no-op if LOCAL_RANK not set → single-GPU path unchanged) ----
    local_rank, world_size, global_rank, is_main = _init_distributed()
    is_ddp = world_size > 1
    if is_ddp:
        print(f"[DDP] rank={global_rank}/{world_size} local_rank={local_rank} "
              f"backend={dist.get_backend()}", flush=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    log(f"Device: {device} (world_size={world_size})")
    log(f"Channel distance: sigma={args.channel_dist_sigma}, temp={args.channel_dist_temperature}")
    log(f"Adaptive scaling: dist = {args.dist_target_fraction*100:.0f}% of SSL loss")
    log(f"Batch: {args.num_probes} probes × {args.samples_per_probe} samples "
        f"= {args.num_probes * args.samples_per_probe} (global; split across {world_size} rank(s))")

    backbone = get_backbone(config["backbone"], config).to(device)
    if is_ddp:
        # find_unused_parameters=True: wav2vec2's codebook + whisper's frozen
        # EMA teacher / partial-freeze leave some params unused per step.
        find_unused = bool(config.get("ddp_find_unused_parameters", True))
        backbone = DDP(backbone, device_ids=[local_rank], output_device=local_rank,
                       find_unused_parameters=find_unused)
        log(f"DDP wrapped backbone (find_unused_parameters={find_unused})")
    backbone_module = _unwrap(backbone)
    log(f"Backbone: {config['backbone']}, params={sum(p.numel() for p in backbone_module.parameters()):,}")

    from blind_localization.data.lazyloader_dataset import CompactDataset

    compact_kwargs = dict(
        include_labels=True,
        dataset=config.get("data", "IBL"),
        verbose=True,
        atlas_depth=int(config.get("atlas_depth", 5)),
        already_resampled=False,
        gpu_resample=True,
        input_sr=1250,
        model_sr=16000 if backbone_module.needs_resampling else 1250,
    )

    base_train_ds = CompactDataset(
        compact_dir=os.path.join(compact_data_path, "train"), **compact_kwargs
    )
    val_ds = CompactDataset(
        compact_dir=os.path.join(compact_data_path, "val"), **compact_kwargs
    )

    train_ds = ChannelAwareDataset(base_train_ds)

    if is_ddp:
        sampler = DistributedSameProbeBatchSampler(
            base_train_ds,
            samples_per_probe=args.samples_per_probe,
            num_probes=args.num_probes,
            seed=42,
            num_replicas=world_size,
            rank=global_rank,
        )
    else:
        sampler = SameProbeBatchSampler(
            base_train_ds,
            samples_per_probe=args.samples_per_probe,
            num_probes=args.num_probes,
            seed=42,
        )

    train_loader = DataLoader(
        train_ds, batch_sampler=sampler,
        num_workers=int(config.get("num_workers", 4)), pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(config.get("batch_size_ssl", 64)),
        shuffle=False, num_workers=int(config.get("num_workers", 4)), pin_memory=True,
    )

    # ── Optional: combined training with teammate's Allen compact dataset ──
    if args.allen_compact_data_path:
        from Lfp2vec_benchmarks.utils.allen_ibl_combined import (
            AllenChannelAwareDataset, interleave_loaders,
        )
        log(f"[combined] Allen dataset: {args.allen_compact_data_path}")
        allen_base_ds = CompactDataset(
            compact_dir=os.path.join(args.allen_compact_data_path, "train"),
            **compact_kwargs,
        )
        allen_train_ds = AllenChannelAwareDataset(allen_base_ds, args.allen_channels_csv)
        if is_ddp:
            allen_sampler = DistributedSameProbeBatchSampler(
                allen_base_ds,
                samples_per_probe=args.samples_per_probe,
                num_probes=args.num_probes,
                seed=43,
                num_replicas=world_size,
                rank=global_rank,
            )
        else:
            allen_sampler = SameProbeBatchSampler(
                allen_base_ds,
                samples_per_probe=args.samples_per_probe,
                num_probes=args.num_probes,
                seed=43,
            )
        allen_loader = DataLoader(
            allen_train_ds, batch_sampler=allen_sampler,
            num_workers=int(config.get("num_workers", 4)), pin_memory=True,
        )

        class CombinedLoader:
            """Round-robin iterable: yields IBL batch, then Allen batch, etc.
            Len = sum of both samplers so steps_per_epoch and LR scheduler
            behave correctly. Val/probe loaders stay IBL-only."""
            def __init__(self, ibl, allen):
                self.ibl = ibl
                self.allen = allen
            def __iter__(self):
                for _tag, batch in interleave_loaders(self.ibl, self.allen):
                    yield batch
            def __len__(self):
                return len(self.ibl) + len(self.allen)

        train_loader = CombinedLoader(train_loader, allen_loader)
        log(f"[combined] IBL batches/epoch: {len(sampler)}  "
            f"Allen batches/epoch: {len(allen_sampler)}  "
            f"Combined: {len(train_loader)}")

    # Labeled loaders for inline linear probe (separate from CD train loader —
    # probe doesn't need same-probe batching, just standard SGD).
    probe_batch_size = int(config.get("batch_size_ssl", 64))
    train_loader_labeled = DataLoader(
        base_train_ds, batch_size=probe_batch_size, shuffle=True,
        num_workers=int(config.get("num_workers", 4)), pin_memory=True,
    )
    val_loader_labeled = DataLoader(
        val_ds, batch_size=probe_batch_size, shuffle=False,
        num_workers=int(config.get("num_workers", 4)), pin_memory=True,
    )

    log(f"Train: {len(base_train_ds)} samples, {len(sampler)} batches/epoch")
    log(f"Val: {len(val_ds)} samples")

    channel_dist_loss_fn = ChannelDistanceContrastiveLoss(
        sigma=args.channel_dist_sigma,
        temperature=args.channel_dist_temperature,
        samples_per_probe=args.samples_per_probe,
    )

    lr = float(config.get("lr_ssl", 1e-4))
    optimizer = torch.optim.AdamW(
        backbone_module.parameters(), lr=lr,
        weight_decay=float(config.get("weight_decay", 0.01)),
    )

    ssl_epochs = int(config.get("ssl_epochs", 50))
    warmup_steps = int(config.get("warmup_steps", 400))
    warmup_start_factor = float(config.get("warmup_start_factor", 0.01))
    eta_min = float(config.get("eta_min", 0.0))
    steps_per_epoch = len(train_loader)
    total_steps = ssl_epochs * steps_per_epoch
    lr_schedule = config.get("lr_schedule", "cosine")  # "cosine" or "plateau"

    # epoch_scheduler stepped per-epoch after val_loss (only for plateau mode)
    epoch_scheduler = None

    if lr_schedule == "plateau":
        plateau_factor = float(config.get("plateau_factor", 0.5))
        plateau_patience = int(config.get("plateau_patience", 5))
        if warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_start_factor, end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            scheduler = None
        epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=plateau_factor,
            patience=plateau_patience, min_lr=eta_min,
        )
        log(f"LR schedule: warmup {lr*warmup_start_factor:.2e} → {lr:.2e} "
            f"over {warmup_steps} steps, then ReduceLROnPlateau "
            f"(factor={plateau_factor}, patience={plateau_patience}, min_lr={eta_min:.2e})")
    else:
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_start_factor, end_factor=1.0,
                total_iters=warmup_steps,
            )
            decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=eta_min,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=eta_min,
            )
        log(f"LR: {lr:.2e}, warmup={warmup_steps}, total_steps={total_steps} (cosine)")

    use_amp = bool(config.get("amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))

    start_epoch = 0
    best_val_loss = float("inf")
    best_probe_acc = 0.0
    early_stop_patience = int(config.get("early_stop_patience", 0))
    epochs_without_improvement = 0

    lr_override_remaining_warmup = 0  # >0 means we're in fresh warmup post-override

    if args.resume_from and os.path.isfile(args.resume_from):
        log(f"Resuming from {args.resume_from}")
        # Every rank loads the same ckpt → identical backbone + optimizer state.
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        backbone_module.load_state_dict(ckpt["backbone_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_probe_acc = ckpt.get("best_probe_acc", 0.0)

        resume_step = start_epoch * steps_per_epoch
        if lr_schedule == "plateau":
            # Rebuild warmup scheduler; plateau epoch_scheduler already created fresh.
            # If resume is past warmup, fast-forward warmup (no-op after completion).
            if warmup_steps > 0:
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=warmup_start_factor, end_factor=1.0,
                    total_iters=warmup_steps,
                )
                for _ in range(min(resume_step, warmup_steps)):
                    scheduler.step()
            else:
                scheduler = None
        else:
            if warmup_steps > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=warmup_start_factor, end_factor=1.0,
                    total_iters=warmup_steps,
                )
                decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=eta_min,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup_scheduler, decay_scheduler],
                    milestones=[warmup_steps],
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_steps, eta_min=eta_min,
                )
            for _ in range(resume_step):
                scheduler.step()
        log(f"Resumed at epoch {start_epoch}, lr={optimizer.param_groups[0]['lr']:.2e}")
        del ckpt

        # LR override — bypass the saved LR state, give the model headroom to
        # keep learning after plateau had driven LR into the floor. Optionally
        # add a fresh linear warmup to avoid a step discontinuity.
        if args.lr_override is not None:
            target_lr = float(args.lr_override)
            current_lr = optimizer.param_groups[0]["lr"]
            warmup = max(int(args.lr_override_warmup_steps), 0)
            if warmup > 0 and current_lr < target_lr:
                # LinearLR requires start_factor, end_factor ∈ [0, 1]. Set the
                # optimizer's base LR = target_lr and warmup from
                # (current/target) up to 1.0. After `warmup` steps LR == target_lr.
                # The scheduler is nulled out at the start of the epoch AFTER
                # warmup completes (see `lr_override_remaining_warmup` handling
                # below) so plateau can own the LR from then on without clash.
                for g in optimizer.param_groups:
                    g["lr"] = target_lr
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=current_lr / target_lr,
                    end_factor=1.0,
                    total_iters=warmup,
                )
                lr_override_remaining_warmup = warmup
                log(f"  LR override: {current_lr:.2e} → {target_lr:.2e} "
                    f"over {warmup} warmup steps "
                    f"(start_factor={current_lr/target_lr:.4f}, end_factor=1.0)")
            else:
                for g in optimizer.param_groups:
                    g["lr"] = target_lr
                scheduler = None
                lr_override_remaining_warmup = 0
                log(f"  LR override: {current_lr:.2e} → {target_lr:.2e} (step jump)")

    if early_stop_patience > 0:
        log(f"Early stopping: patience={early_stop_patience}")

    os.makedirs(args.results_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.results_dir, "checkpoints")
    embed_dir = os.path.join(args.results_dir, "embeddings")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)

    wandb = None
    try:
        import wandb as _wandb
        if os.environ.get("WANDB_RUN_NAME") and is_main:
            _wandb.init(
                project=os.environ.get("WANDB_PROJECT", "lfp2vec_training"),
                entity=os.environ.get("WANDB_ENTITY"),
                name=os.environ.get("WANDB_RUN_NAME"),
                config={**config,
                        "channel_dist_sigma": args.channel_dist_sigma,
                        "dist_target_fraction": args.dist_target_fraction,
                        "samples_per_probe": args.samples_per_probe,
                        "num_probes": args.num_probes},
            )
            wandb = _wandb
    except Exception:
        pass

    log(f"\nStarting training: epochs {start_epoch}→{ssl_epochs}")
    for epoch in range(start_epoch, ssl_epochs):
        t0 = time.time()

        # DDP: re-seed the distributed same-probe sampler(s) each epoch.
        if is_ddp:
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            if args.allen_compact_data_path and hasattr(allen_sampler, "set_epoch"):
                allen_sampler.set_epoch(epoch)

        ssl_loss, ch_loss, grad_norm = train_one_epoch(
            backbone, train_loader, optimizer, scheduler, scaler,
            device, use_amp, grad_clip_norm,
            channel_dist_loss_fn, args.dist_target_fraction,
            ssl_loss_weight=args.ssl_loss_weight,
        )

        # LR-override warmup accounting — once warmup is exhausted, null the
        # warmup scheduler so the ReduceLROnPlateau scheduler can freely adjust
        # LR without being overwritten by the warmup scheduler's next step().
        if lr_override_remaining_warmup > 0:
            lr_override_remaining_warmup -= steps_per_epoch
            if lr_override_remaining_warmup <= 0:
                scheduler = None
                log(f"  [warmup] LR override warmup complete, "
                    f"scheduler handed over to plateau. "
                    f"LR = {optimizer.param_groups[0]['lr']:.2e}")

        # Validation (standard SSL loss — directly comparable to pure SSL).
        # Runs on every rank over the full (non-sharded) val set; the resulting
        # val_loss is all-reduced to a common mean so early-stop / plateau
        # decisions stay in lockstep across ranks.
        backbone_module.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].float().to(device)
                x = preprocess_batch(x, backbone_module.needs_resampling, backbone_module.needs_normalization)
                vl = backbone_module.ssl_forward(x)
                if vl is not None and torch.isfinite(vl):
                    val_losses.append(vl.item())
        val_loss = np.mean(val_losses) if val_losses else float("inf")
        val_loss = _ddp_mean(val_loss, device, is_ddp)

        # Collapse metrics (per-rank, logged on rank 0 only — not a decision input)
        with torch.no_grad():
            val_batch = next(iter(val_loader))
            vx = val_batch[0][:128].float().to(device)
            vx = preprocess_batch(vx, backbone_module.needs_resampling, backbone_module.needs_normalization)
            val_emb = backbone_module.encode(vx)
            from torch.nn import functional as _F
            z = _F.normalize(val_emb, dim=-1)
            cosine_sim = (z @ z.T).fill_diagonal_(0).sum() / (128 * 127)
            embed_std = val_emb.std(dim=0).mean()

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        adaptive_lambda = args.dist_target_fraction * ssl_loss / max(ch_loss, 1e-8) if ch_loss > 1e-8 else 0

        log(f"Epoch {epoch}/{ssl_epochs}: ssl={ssl_loss:.1f} dist={ch_loss:.4f} (λ_eff={adaptive_lambda:.4g}) "
            f"| val={val_loss:.1f} | grad={grad_norm:.4f} lr={lr_now:.2e} "
            f"| cos={cosine_sim:.4f} std={embed_std:.4f} | {elapsed:.0f}s")

        # Inline linear probe (matches train_ssl.py schema — probe_interval + probe_epochs).
        probe_acc = None
        probe_bal_acc = None
        probe_interval = int(config.get("probe_interval", 0))
        probe_epoch_count = int(config.get("probe_epochs", 20))
        is_probe_epoch = (probe_interval > 0 and epoch % probe_interval == 0)
        if probe_epoch_count > 0 and is_probe_epoch and is_main:
            try:
                probe_num_classes = getattr(base_train_ds, "num_effective_classes", None)
                result = _quick_linear_probe(
                    backbone_module, train_loader_labeled, val_loader_labeled, device,
                    epoch_count=probe_epoch_count,
                    num_classes=probe_num_classes,
                    max_probe_samples=int(config.get("max_probe_samples", 0)),
                )
                if result is not None:
                    probe_acc, probe_bal_acc, _, _, _ = result
                    log(f"  Probe: acc={probe_acc:.4f}, bal_acc={probe_bal_acc:.4f} "
                        f"(best so far: {best_probe_acc:.4f})")
            except Exception as e:
                log(f"[WARN] Linear probe failed at epoch {epoch}: {e}")

        # DDP: resync after the rank-0-only probe so no rank races ahead into
        # the next epoch's all-reduce while rank 0 is still probing.
        if is_ddp:
            dist.barrier()

        if wandb and wandb.run:
            log_dict = {
                "train_ssl_loss": ssl_loss,
                "train_dist_loss": ch_loss,
                "adaptive_lambda": adaptive_lambda,
                "val_loss": val_loss,
                "grad_norm": grad_norm,
                "lr": lr_now,
                "cosine_sim": cosine_sim.item(),
                "embed_std": embed_std.item(),
                "epoch": epoch,
            }
            if probe_acc is not None:
                log_dict["probe_acc"] = probe_acc
                log_dict["probe_bal_acc"] = probe_bal_acc
            wandb.log(log_dict)

        # Save best-probe-accuracy checkpoint (separate from best_valloss).
        # Hierarchical eval picks whichever checkpoint is best for downstream.
        # Disk writes rank-0 only; best_val_loss tracking runs on ALL ranks
        # (val_loss is identical across ranks → epochs_without_improvement and
        # the early-stop break below stay in lockstep). probe_acc is None on
        # non-main ranks (probe ran on rank 0 only), so the best-probe save is
        # naturally rank-0.
        if is_main and probe_acc is not None and probe_acc > best_probe_acc:
            best_probe_acc = probe_acc
            torch.save({
                "backbone_state_dict": backbone_module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "best_probe_acc": best_probe_acc,
                "probe_acc": probe_acc,
                "probe_bal_acc": probe_bal_acc,
            }, os.path.join(ckpt_dir, "ssl_wav2vec2_best.pt"))
            log(f"  Best PROBE checkpoint saved (epoch {epoch}, probe_acc={probe_acc:.4f})")

        val_loss_improved = val_loss < best_val_loss
        if val_loss_improved:
            best_val_loss = val_loss
            if is_main:
                torch.save({
                    "backbone_state_dict": backbone_module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "best_probe_acc": best_probe_acc,
                }, os.path.join(ckpt_dir, "ssl_wav2vec2_best_valloss.pt"))
                log(f"  Best val_loss saved (epoch {epoch}, val_loss={val_loss:.4f})")

        if is_main:
            torch.save({
                "backbone_state_dict": backbone_module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "best_probe_acc": best_probe_acc,
            }, os.path.join(ckpt_dir, "ssl_wav2vec2_latest.pt"))

        # Step plateau scheduler per-epoch (adapts LR to val_loss)
        if epoch_scheduler is not None:
            epoch_scheduler.step(float(val_loss))

        if val_loss_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            log(f"Early stopping at epoch {epoch}: no improvement for {early_stop_patience} epochs")
            break

        embed_interval = args.embed_interval
        if embed_interval > 0 and (epoch % embed_interval == 0 or epoch == ssl_epochs - 1) and is_main:
            backbone_module.eval()
            all_emb = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].float().to(device)
                    x = preprocess_batch(x, backbone_module.needs_resampling, backbone_module.needs_normalization)
                    emb = backbone_module.encode(x)
                    all_emb.append(emb.cpu().numpy())
            all_emb = np.concatenate(all_emb)
            np.savez_compressed(
                os.path.join(embed_dir, f"embeddings_ep{epoch:04d}.npz"),
                embeddings=all_emb,
            )
            log(f"  Saved embeddings ({all_emb.shape})")

    log(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    log(f"Results at: {args.results_dir}")

    if wandb and wandb.run:
        wandb.finish()

    # ---- DDP teardown ----
    if is_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
