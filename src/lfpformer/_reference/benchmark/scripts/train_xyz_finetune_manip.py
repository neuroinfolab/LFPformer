#!/usr/bin/env python3
"""Manipulator-conditioned XYZ readout training.

Variant of train_xyz_finetune.py. Identical backbone / SSL+CD / joint scheme,
with XYZ heads fed the per-channel micro-manipulator CCF position. Supports
both residual and direct coordinate parameterizations:

    feat   = [ mean_pool(encoder(LFP))  ⊕  manip_norm ]      (768 + 3)
    residual: pred = manip_norm + head(feat)
    direct:   pred = head(feat)

Everything is in the SAME folded-LR normalized space as the headline pipeline
(CentroidNormalizer). manip_norm = normalizer.encode(manip_xyz_um).

manip_xyz is precomputed offline (precompute_manip_xyz.py) and loaded as a
per-split (N,3) µm array aligned to CompactDataset idx order — NO iblatlas at
train time. Invalid-manip rows (missing/degenerate trajectories) are dropped
via a Subset.

Checkpoint adds `manip_conditioned: True` and `head_input_dim` so the matching
eval (eval_manip_ft.py) knows to concat manip + add the residual. Run in
w2v2_cuda_cu128_ibl.sif.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

# Import the canonical training stack from the ws2 bench tree (apples-to-apples
# with Path-B 75K). Do NOT prepend this script's own dir, to avoid shadowing.
PB_BENCH = os.environ.get("PB_BENCH",
                          __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/Lfp2vec_benchmarks'))
PB_ROOT = os.path.dirname(PB_BENCH)
for _p in (os.path.join(PB_BENCH, "scripts"), PB_BENCH, PB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backbones import get_backbone
from utils.preprocessing import preprocess_batch
from utils.probe_distance_sampler import (
    SameProbeBatchSampler,
    ChannelAwareDataset,
    ChannelDistanceContrastiveLoss,
    parse_channel_id,
)
from train_ssl import log
from eval_regression_probe import (
    CentroidNormalizer,
    GaussianLinearProbe,
    GaussianNLL3D,
    compute_region_centroids,
)


class ManipMLPHead(nn.Module):
    """Nonlinear coordinate head matched to the frozen-probe MLP control."""

    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GaussianMLPProbe(nn.Module):
    """MLP analogue of GaussianLinearProbe."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = ManipMLPHead(input_dim, output_dim=6)

    def forward(self, x):
        out = self.net(x)
        return out[:, :3], out[:, 3:].clamp(-10.0, 10.0)


def make_heads(head_type, input_dim, device):
    if head_type == "linear":
        return (nn.Linear(input_dim, 3).to(device),
                GaussianLinearProbe(input_dim=input_dim).to(device))
    if head_type == "mlp":
        return (ManipMLPHead(input_dim).to(device),
                GaussianMLPProbe(input_dim).to(device))
    raise ValueError(f"unknown head_type={head_type!r}")


def apply_coordinate_parameterization(head_output, manip_norm, parameterization):
    """Convert a head output into the final normalized atlas coordinate."""
    if parameterization == "residual":
        return manip_norm + head_output
    if parameterization == "direct":
        return head_output
    raise ValueError(f"unknown parameterization={parameterization!r}")


# ────────────────────────────────────────────────────────────────────────────
#                 Probe-train dataset wrapper (with manip_xyz)
# ────────────────────────────────────────────────────────────────────────────

class LabeledProbeTrainDatasetManip(torch.utils.data.Dataset):
    """Yield (signal, channel_number, coords_xyz_um, manip_xyz_um).

    manip_xyz_um is a precomputed (N,3) µm array aligned to CompactDataset idx
    order (precompute_manip_xyz.py). channel_number parsed from meta_str like
    the original LabeledProbeTrainDataset so joint-mode CD loss still works.
    """

    def __init__(self, base_dataset, manip_xyz_path):
        self.base = base_dataset
        self._channel_numbers = self._build_channel_index()
        self._manip_xyz = np.load(manip_xyz_path).astype(np.float32)
        if len(self._manip_xyz) != len(self.base):
            raise ValueError(
                f"manip_xyz rows ({len(self._manip_xyz)}) != dataset "
                f"({len(self.base)}) — alignment broken for {manip_xyz_path}")

    def _build_channel_index(self):
        import pickle as _pkl
        compact_dir = self.base.compact_dir
        idx_data = _pkl.load(open(os.path.join(compact_dir, 'compact_index.pkl'), 'rb'))
        channels = []
        for fid in range(idx_data['n_files']):
            fpath = os.path.join(compact_dir, f'samples_{fid:06d}.pkl')
            samples = _pkl.load(open(fpath, 'rb'))
            for s in samples:
                meta = s[1] if len(s) >= 2 and isinstance(s[1], str) else ""
                channels.append(parse_channel_id(meta))
        return channels

    def valid_indices(self):
        return np.where(np.isfinite(self._manip_xyz).all(1))[0]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sig, _meta, coords = self.base[idx]
        ch = self._channel_numbers[idx] if self._channel_numbers else -1
        return (sig,
                int(ch),
                np.asarray(coords, dtype=np.float32),
                self._manip_xyz[idx])


# ────────────────────────────────────────────────────────────────────────────
#                       Per-channel xyz target collection
# ────────────────────────────────────────────────────────────────────────────

def collect_probe_train_meta(ds):
    n = len(ds)
    coords = np.zeros((n, 3), dtype=np.float64)
    fids = np.zeros(n, dtype=np.int64)
    for i in range(n):
        _, _, c = ds[i]
        h = ds.get_hierarchy(i)
        coords[i, 0] = float(c[0])
        coords[i, 1] = float(c[1])
        coords[i, 2] = float(c[2])
        fids[i] = int(h["fine_id"]) if h and "fine_id" in h else 0
    return coords, fids


def eids_in_order(compact_dir):
    """Session eid per sample, in CompactDataset idx order (same iteration as
    LabeledProbeTrainDatasetManip._build_channel_index)."""
    import pickle as _pkl
    idx_data = _pkl.load(open(os.path.join(compact_dir, 'compact_index.pkl'), 'rb'))
    eids = []
    for fid in range(idx_data['n_files']):
        for s in _pkl.load(open(os.path.join(compact_dir, f'samples_{fid:06d}.pkl'), 'rb')):
            meta = s[1] if len(s) >= 2 and isinstance(s[1], str) else ""
            eids.append(meta.split("_")[0] if meta else "")
    return eids


def subject_held_out_split(train_ds, val_ds, train_base, val_base, tr_valid, va_valid,
                           args):
    """Pool probe_train + probe_val (same probes, different time chunks) and
    re-split by SUBJECT: a fixed fraction of subjects becomes validation."""
    import json as _json
    from torch.utils.data import ConcatDataset
    smap = _json.load(open(args.subject_map))
    tr_e, va_e = eids_in_order(train_base.compact_dir), eids_in_order(val_base.compact_dir)
    if len(tr_e) != len(train_ds) or len(va_e) != len(val_ds):
        raise ValueError("eid list not aligned with dataset rows")
    missing = sorted({e for e in tr_e + va_e if e not in smap})
    if missing:
        raise ValueError(f"{len(missing)} sessions missing from {args.subject_map}: {missing[:3]}")
    s_tr = np.array([smap[e] for e in tr_e])
    s_va = np.array([smap[e] for e in va_e])
    subjects = sorted(set(s_tr) | set(s_va))
    order = np.random.RandomState(args.split_seed).permutation(len(subjects))
    n_val = max(1, int(round(args.subject_val_frac * len(subjects))))
    val_subj = sorted(subjects[i] for i in order[:n_val])

    pooled = ConcatDataset([train_ds, val_ds])
    idx = np.concatenate([tr_valid, va_valid + len(train_ds)])
    subj = np.concatenate([s_tr[tr_valid], s_va[va_valid]])
    is_val = np.isin(subj, val_subj)
    new_train = Subset(pooled, idx[~is_val].tolist())
    new_val = Subset(pooled, idx[is_val].tolist())
    assert not (set(subj[~is_val]) & set(subj[is_val])), "subject leak between train/val"
    log(f"  SUBJECT-held-out val (frac={args.subject_val_frac}, split_seed={args.split_seed}): "
        f"train {len(subjects) - n_val} subj / {len(new_train)} rows, "
        f"val {n_val} subj / {len(new_val)} rows; val subjects = {val_subj}")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "subject_split.json"), "w") as f:
        _json.dump({"val_subjects": val_subj, "n_subjects": len(subjects),
                    "n_train_rows": len(new_train), "n_val_rows": len(new_val),
                    "split_seed": args.split_seed, "frac": args.subject_val_frac}, f, indent=1)
    return new_train, new_val


def fit_normaliser_per_channel(probe_train_ds, probe_val_ds):
    log("Collecting probe_train + probe_val coords/fine_ids for normaliser…")
    tr_coords, tr_fids = collect_probe_train_meta(probe_train_ds)
    va_coords, va_fids = collect_probe_train_meta(probe_val_ds)

    shared_fids = sorted(set(tr_fids.tolist()) & set(va_fids.tolist()) - {0})
    log(f"  train ∩ val fine_ids: {len(shared_fids)} (excl. 0)")
    keep_tr = np.isin(tr_fids, shared_fids)
    centroids = compute_region_centroids(
        tr_coords[keep_tr], tr_fids[keep_tr], shared_fids,
    )
    log(f"  computed {len(centroids)} centroids for normaliser bbox seed")

    normalizer = CentroidNormalizer(centroids)
    combined = np.concatenate([tr_coords, va_coords], axis=0)
    ap_lo, ap_hi = float(combined[:, 0].min()), float(combined[:, 0].max())
    dv_lo, dv_hi = float(combined[:, 1].min()), float(combined[:, 1].max())
    d_all = np.abs(combined[:, 2] - normalizer.midline)
    d_lo, d_hi = float(d_all.min()), float(d_all.max())
    normalizer.ap_min = min(normalizer.ap_min, ap_lo)
    normalizer.ap_max = max(normalizer.ap_max, ap_hi)
    normalizer.dv_min = min(normalizer.dv_min, dv_lo)
    normalizer.dv_max = max(normalizer.dv_max, dv_hi)
    normalizer.d_min = min(normalizer.d_min, d_lo)
    normalizer.d_max = max(normalizer.d_max, d_hi)
    log(f"  per-channel-widened normaliser: "
        f"ap[{normalizer.ap_min:.0f},{normalizer.ap_max:.0f}] "
        f"dv[{normalizer.dv_min:.0f},{normalizer.dv_max:.0f}] "
        f"d[{normalizer.d_min:.0f},{normalizer.d_max:.0f}]")
    return normalizer, centroids


# ────────────────────────────────────────────────────────────────────────────
#                          Manip-residual steps
# ────────────────────────────────────────────────────────────────────────────

def _encode_pair(normalizer, coords_um, manip_um, device):
    """Return (target_norm, manip_norm) tensors on device, both (B,3)."""
    coords_np = coords_um.numpy() if isinstance(coords_um, torch.Tensor) else coords_um
    manip_np = manip_um.numpy() if isinstance(manip_um, torch.Tensor) else manip_um
    target_norm = torch.from_numpy(normalizer.encode(np.asarray(coords_np, np.float64))).to(device)
    manip_norm = torch.from_numpy(normalizer.encode(np.asarray(manip_np, np.float64))).to(device)
    return target_norm.float(), manip_norm.float()


@torch.no_grad()
def validate(backbone, mse_head, gauss_head, val_loader, normalizer, device,
             parameterization):
    backbone.eval(); mse_head.eval(); gauss_head.eval()
    mse_losses, nll_losses = [], []
    nll_fn = GaussianNLL3D()
    for sig, _ch, coords_um, manip_um in val_loader:
        sig = sig.float().to(device, non_blocking=True)
        target_norm, manip_norm = _encode_pair(normalizer, coords_um, manip_um, device)
        x = preprocess_batch(sig, backbone.needs_resampling, backbone.needs_normalization)
        emb = backbone.encode(x)
        feat = torch.cat([emb, manip_norm], dim=1)
        pred = apply_coordinate_parameterization(
            mse_head(feat), manip_norm, parameterization)
        mse_losses.append(F.mse_loss(pred, target_norm).item())
        mu_out, log_var = gauss_head(feat)
        mu = apply_coordinate_parameterization(
            mu_out, manip_norm, parameterization)
        nll_losses.append(nll_fn(mu, log_var, target_norm).item())
    return float(np.mean(mse_losses)), float(np.mean(nll_losses))


def step_supervised(backbone, mse_head, gauss_head, sig, coords_um, manip_um,
                    normalizer, device, nll_fn, lambda_xyz, parameterization):
    """MSE + NLL on a probe_train batch under the selected parameterization."""
    sig = sig.float().to(device, non_blocking=True)
    target_norm, manip_norm = _encode_pair(normalizer, coords_um, manip_um, device)
    x = preprocess_batch(sig, backbone.needs_resampling, backbone.needs_normalization)
    emb = backbone.encode(x)
    feat = torch.cat([emb, manip_norm], dim=1)
    pred = apply_coordinate_parameterization(
        mse_head(feat), manip_norm, parameterization)
    mu_out, log_var = gauss_head(feat)
    mu = apply_coordinate_parameterization(
        mu_out, manip_norm, parameterization)
    mse_loss = F.mse_loss(pred, target_norm)
    nll_loss = nll_fn(mu, log_var, target_norm)
    xyz_loss = mse_loss + nll_loss
    return lambda_xyz * xyz_loss, mse_loss.item(), nll_loss.item()


def step_ssl_cd(backbone, sig, channel_numbers, channel_dist_loss_fn,
                dist_target_fraction, ssl_loss_weight, device):
    sig = sig.float().to(device, non_blocking=True)
    channel_numbers = channel_numbers.float().to(device, non_blocking=True)
    x = preprocess_batch(sig, backbone.needs_resampling, backbone.needs_normalization)
    result = backbone.ssl_forward_detailed(x)
    ssl_loss = result["ssl_loss"]
    chunk_embeds = result.get("chunk_embeds")
    if ssl_loss is None or not torch.isfinite(ssl_loss):
        return None, 0.0, 0.0, 0.0
    ssl_val = ssl_loss.item()
    ch_val = 0.0
    adaptive_lambda = 0.0
    loss = ssl_loss_weight * ssl_loss
    if chunk_embeds is not None and dist_target_fraction > 0:
        ch_loss = channel_dist_loss_fn(chunk_embeds, channel_numbers)
        ch_val = ch_loss.item()
        if ch_val > 1e-8:
            adaptive_lambda = dist_target_fraction * ssl_val / ch_val
            loss = loss + adaptive_lambda * ch_loss
    return loss, ssl_val, ch_val, adaptive_lambda


# ────────────────────────────────────────────────────────────────────────────
#                                  main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["supervised", "joint"])
    parser.add_argument("--init_ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe_data_path", required=True)
    parser.add_argument("--ssl_data_path", default=None)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--manip_xyz_dir",
                        default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/one_cache/manip_xyz/eval_v2_100k'),
                        help="dir with {probe_train,probe_val}_manip_xyz.npy "
                        "(precompute_manip_xyz.py output)")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-4,
                        help="Peak LR for coordinate heads (and for the backbone "
                             "unless --backbone_lr is set).")
    parser.add_argument("--backbone_lr", type=float, default=None,
                        help="Optional separate peak LR for the unfrozen backbone. "
                             "Default: same as --lr. Prefer a lower value (e.g. 1e-5) "
                             "for full-FT to limit OOD damage.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of optimizer steps for linear LR warmup, then "
                             "cosine decay. 0 disables the schedule (constant LR).")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01,
                        help="Final cosine LR as a fraction of each param-group peak.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--target_mode", default="per_channel_xyz",
                        choices=["per_channel_xyz"])
    parser.add_argument("--lambda_xyz", type=float, default=1.0)
    parser.add_argument("--ssl_loss_weight", type=float, default=1.0)
    parser.add_argument("--channel_dist_sigma", type=float, default=500.0)
    parser.add_argument("--channel_dist_temperature", type=float, default=0.2)
    parser.add_argument("--dist_target_fraction", type=float, default=0.1)
    parser.add_argument("--samples_per_probe", type=int, default=16)
    parser.add_argument("--num_probes", type=int, default=4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--head_type", choices=["linear", "mlp"], default="linear",
                        help="coordinate readout architecture")
    parser.add_argument("--parameterization", choices=["residual", "direct"],
                        default="residual",
                        help="residual: pred=manip+head; direct: pred=head")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="train only the coordinate heads")
    # Opt-in subject-held-out validation. probe_train/probe_val in eval_v2_100k are
    # time-chunk splits of the SAME 141 probes, so the default val cannot detect
    # probe-level overfitting. With frac > 0, both splits are pooled and re-split
    # by SUBJECT (val = held-out subjects). Default 0 = original behaviour.
    parser.add_argument("--subject_val_frac", type=float, default=0.0,
                        help="fraction of training subjects held out for validation")
    parser.add_argument("--subject_map", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".cache_ibl_session_to_subject_eval_v2_trainval.json"),
                        help="session eid -> subject json")
    parser.add_argument("--split_seed", type=int, default=0,
                        help="subject-split seed (fixed across decoder seeds)")
    args = parser.parse_args()

    if args.mode == "joint" and not args.ssl_data_path:
        raise SystemExit("joint mode requires --ssl_data_path")
    if args.freeze_backbone and args.mode == "joint":
        raise SystemExit("--freeze_backbone requires --mode supervised; "
                         "SSL/CD updates have no trainable backbone parameters")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}  Mode: {args.mode}  "
        f"(MANIP-conditioned {args.parameterization}, {args.head_type}, "
        f"backbone={'frozen' if args.freeze_backbone else 'full-ft'})")

    # ── Backbone init ───────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # A complete trained backbone state dict is loaded immediately below.
    # Avoid fetching/constructing HF pretrained weights only to overwrite them.
    config["rand_init"] = True
    if args.mode == "joint":
        config["channel_distance"] = True
    log(f"Backbone: {config['backbone']}")
    backbone = get_backbone(config["backbone"], config).to(device)
    log(f"Loading init ckpt: {args.init_ckpt}")
    init_ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
    backbone.load_state_dict(init_ckpt["backbone_state_dict"])
    log(f"  init epoch={init_ckpt.get('epoch', '?')} "
        f"val_loss={init_ckpt.get('val_loss', '?')}")
    is_resume = "mse_head_state_dict" in init_ckpt and init_ckpt.get("manip_conditioned")
    if is_resume:
        log("  RESUME detected (manip ckpt) — load heads + optimizer + epoch")

    # ── xyz heads (input = hidden + 3 manip coords) ─────────────────────────
    hidden_dim = backbone.hidden_dim if hasattr(backbone, "hidden_dim") else 768
    head_in = hidden_dim + 3
    mse_head, gauss_head = make_heads(args.head_type, head_in, device)
    log(f"Heads: {args.head_type} input={head_in}  "
        f"[{args.parameterization}: "
        f"pred={'manip_norm + output' if args.parameterization == 'residual' else 'output'}]")
    if is_resume:
        mse_head.load_state_dict(init_ckpt["mse_head_state_dict"])
        gauss_head.load_state_dict(init_ckpt["gauss_head_state_dict"])
        log("  resumed mse_head + gauss_head from init_ckpt")

    if args.freeze_backbone:
        backbone.requires_grad_(False)
        backbone.eval()
        log("Backbone frozen: coordinate heads are the only trainable modules")

    # ── Datasets ───────────────────────────────────────────────────────────
    from blind_localization.data.lazyloader_dataset import CompactDataset
    compact_kwargs = dict(
        include_labels=True, dataset="IBL", verbose=False, atlas_depth=9,
        already_resampled=False, gpu_resample=True,
        input_sr=1250, model_sr=16000 if backbone.needs_resampling else 1250,
    )
    log(f"Loading probe_train + probe_val from {args.probe_data_path}")
    probe_train_base = CompactDataset(
        compact_dir=os.path.join(args.probe_data_path, "probe_train"), **compact_kwargs)
    probe_val_base = CompactDataset(
        compact_dir=os.path.join(args.probe_data_path, "probe_val"), **compact_kwargs)
    log(f"  probe_train {len(probe_train_base)}, probe_val {len(probe_val_base)}")

    normalizer, centroids = fit_normaliser_per_channel(probe_train_base, probe_val_base)

    probe_train_ds = LabeledProbeTrainDatasetManip(
        probe_train_base, os.path.join(args.manip_xyz_dir, "probe_train_manip_xyz.npy"))
    probe_val_ds = LabeledProbeTrainDatasetManip(
        probe_val_base, os.path.join(args.manip_xyz_dir, "probe_val_manip_xyz.npy"))
    tr_valid = probe_train_ds.valid_indices()
    va_valid = probe_val_ds.valid_indices()
    log(f"  valid-manip rows: train {len(tr_valid)}/{len(probe_train_ds)}  "
        f"val {len(va_valid)}/{len(probe_val_ds)}")
    if args.subject_val_frac > 0:
        probe_train_ds, probe_val_ds = subject_held_out_split(
            probe_train_ds, probe_val_ds, probe_train_base, probe_val_base,
            tr_valid, va_valid, args)
    else:
        probe_train_ds = Subset(probe_train_ds, tr_valid)
        probe_val_ds = Subset(probe_val_ds, va_valid)

    probe_train_loader = DataLoader(
        probe_train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    probe_val_loader = DataLoader(
        probe_val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    ssl_train_loader = None
    channel_dist_loss_fn = None
    if args.mode == "joint":
        log(f"Loading SSL-train from {args.ssl_data_path}")
        ssl_train_base = CompactDataset(
            compact_dir=os.path.join(args.ssl_data_path, "train"), **compact_kwargs)
        ssl_train_ds = ChannelAwareDataset(ssl_train_base)
        log(f"  ssl_train {len(ssl_train_base)} samples")
        sampler = SameProbeBatchSampler(
            ssl_train_base, samples_per_probe=args.samples_per_probe,
            num_probes=args.num_probes, seed=args.seed)
        ssl_train_loader = DataLoader(
            ssl_train_ds, batch_sampler=sampler,
            num_workers=args.num_workers, pin_memory=True)
        channel_dist_loss_fn = ChannelDistanceContrastiveLoss(
            sigma=args.channel_dist_sigma, temperature=args.channel_dist_temperature,
            samples_per_probe=args.samples_per_probe)
        log(f"  SSL-train batches/epoch: {len(sampler)}, "
            f"probe-train batches/epoch: {len(probe_train_loader)}")

    # ── Optimizer + cosine LR schedule ─────────────────────────────────────
    # Cosine decays LR on a STEP schedule (warmup → cosine → min_lr_ratio).
    # It is NOT ReduceLROnPlateau: val convergence does not trigger the cut;
    # early_stop_patience still handles "val stalled → stop". For full-FT,
    # prefer --backbone_lr << --lr so the head can fit while the backbone
    # moves gently (same pattern as Path-B short-cosine).
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise SystemExit("--warmup_ratio must be in [0, 1)")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise SystemExit("--min_lr_ratio must be in [0, 1]")

    backbone_lr = args.lr if args.backbone_lr is None else args.backbone_lr
    param_groups = []
    if not args.freeze_backbone:
        bb_params = [p for p in backbone.parameters() if p.requires_grad]
        if bb_params:
            param_groups.append({"params": bb_params, "lr": backbone_lr})
    head_params = [p for p in (list(mse_head.parameters()) + list(gauss_head.parameters()))
                   if p.requires_grad]
    param_groups.append({"params": head_params, "lr": args.lr})
    params = [p for g in param_groups for p in g["params"]]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    nll_fn = GaussianNLL3D()
    log(f"Optimizer: AdamW head_lr={args.lr:.2e} "
        f"backbone_lr={'frozen' if args.freeze_backbone else f'{backbone_lr:.2e}'} "
        f"wd={args.weight_decay}")
    log(f"Trainable params: {sum(p.numel() for p in params):,}")

    if args.mode == "supervised":
        steps_per_epoch = len(probe_train_loader)
    else:
        steps_per_epoch = len(ssl_train_loader) + len(probe_train_loader)
    # Total steps counted from epoch 0 so a mid-run resume still lands on the
    # intended cosine phase (fast-forward below when resuming).
    total_optimizer_steps = max(steps_per_epoch * args.epochs, 1)
    warmup_steps = int(round(args.warmup_ratio * total_optimizer_steps))
    scheduler = None
    if warmup_steps > 0 or args.min_lr_ratio < 1.0:
        def lr_factor(step):
            if warmup_steps > 0 and step < warmup_steps:
                return max((step + 1) / warmup_steps, 0.01)
            decay_steps = max(total_optimizer_steps - warmup_steps, 1)
            progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
        log(f"LR schedule: linear warmup={warmup_steps}/{total_optimizer_steps} steps "
            f"({args.warmup_ratio:.1%}), then cosine to "
            f"{args.min_lr_ratio:.1%} of each group's peak "
            f"({steps_per_epoch} steps/epoch)")
    else:
        log("LR schedule: constant (warmup_ratio=0 and min_lr_ratio=1)")

    start_epoch = 0
    resume_best_val_mse = float("inf")
    resume_no_improve = 0
    if is_resume:
        if "optimizer_state_dict" in init_ckpt:
            optimizer.load_state_dict(init_ckpt["optimizer_state_dict"])
            log("  resumed optimizer state")
        if scheduler is not None and init_ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(init_ckpt["scheduler_state_dict"])
            log("  resumed LR scheduler state")
        start_epoch = int(init_ckpt.get("epoch", -1)) + 1
        resume_best_val_mse = float(init_ckpt.get(
            "best_val_mse", init_ckpt.get("val_mse", float("inf"))))
        resume_no_improve = int(init_ckpt.get("no_improve", 0))
        # Fast-forward a freshly built cosine schedule if the ckpt had none.
        if scheduler is not None and init_ckpt.get("scheduler_state_dict") is None and start_epoch > 0:
            completed_steps = start_epoch * steps_per_epoch
            for _ in range(completed_steps):
                scheduler.step()
            log(f"  fast-forwarded LR scheduler by {completed_steps} steps")
        log(f"  start_epoch={start_epoch}  best_val_mse={resume_best_val_mse:.4f}  "
            f"no_improve={resume_no_improve}")
    del init_ckpt

    os.makedirs(args.results_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb = None
    try:
        import wandb as _wandb
        if os.environ.get("WANDB_RUN_NAME"):
            _wandb.init(
                project=os.environ.get("WANDB_PROJECT", "lfp2vec_training"),
                entity=os.environ.get("WANDB_ENTITY"),
                name=os.environ.get("WANDB_RUN_NAME"),
                config={**config, "mode": args.mode, "init_ckpt": args.init_ckpt,
                        "lambda_xyz": args.lambda_xyz,
                        "channel_dist_sigma": args.channel_dist_sigma,
                        "dist_target_fraction": args.dist_target_fraction,
                        "lr": args.lr, "manip_conditioned": True,
                        "head_type": args.head_type,
                        "parameterization": args.parameterization,
                        "freeze_backbone": args.freeze_backbone})
            wandb = _wandb
    except Exception:
        pass

    best_val_mse = resume_best_val_mse
    no_improve = resume_no_improve
    if start_epoch >= args.epochs:
        log(f"start_epoch ({start_epoch}) >= --epochs ({args.epochs}); nothing to do.")
        return

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if args.freeze_backbone:
            backbone.eval()
        else:
            backbone.train()
        mse_head.train(); gauss_head.train()
        ep_mse, ep_nll, ep_ssl, ep_cd, n_xyz, n_ssl = 0.0, 0.0, 0.0, 0.0, 0, 0

        if args.mode == "supervised":
            for sig, _ch, coords_um, manip_um in probe_train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss, mse_v, nll_v = step_supervised(
                    backbone, mse_head, gauss_head, sig, coords_um, manip_um,
                    normalizer, device, nll_fn, args.lambda_xyz,
                    args.parameterization)
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                ep_mse += mse_v; ep_nll += nll_v; n_xyz += 1
        else:
            ssl_iter = iter(ssl_train_loader)
            xyz_iter = iter(probe_train_loader)
            n_ssl_batches = len(ssl_train_loader)
            n_xyz_batches = len(probe_train_loader)
            total_steps = n_ssl_batches + n_xyz_batches
            for step in range(total_steps):
                use_ssl = (step % 2 == 0 and n_ssl_batches > 0) or n_xyz_batches == 0
                optimizer.zero_grad(set_to_none=True)
                if use_ssl:
                    try:
                        batch = next(ssl_iter)
                    except StopIteration:
                        ssl_iter = iter(ssl_train_loader); batch = next(ssl_iter)
                    sig = batch[0]; channel_numbers = batch[-1]
                    loss, ssl_v, ch_v, _ = step_ssl_cd(
                        backbone, sig, channel_numbers, channel_dist_loss_fn,
                        args.dist_target_fraction, args.ssl_loss_weight, device)
                    if loss is None:
                        continue
                    loss.backward()
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    if hasattr(backbone, "step_gumbel_temperature"):
                        backbone.step_gumbel_temperature()
                    ep_ssl += ssl_v; ep_cd += ch_v; n_ssl += 1
                else:
                    try:
                        batch = next(xyz_iter)
                    except StopIteration:
                        xyz_iter = iter(probe_train_loader); batch = next(xyz_iter)
                    sig, ch, coords_um, manip_um = batch
                    loss_ssl_cd, ssl_v, ch_v, _ = step_ssl_cd(
                        backbone, sig, ch.float(), channel_dist_loss_fn,
                        args.dist_target_fraction, args.ssl_loss_weight, device)
                    loss_xyz, mse_v, nll_v = step_supervised(
                        backbone, mse_head, gauss_head, sig, coords_um, manip_um,
                        normalizer, device, nll_fn, args.lambda_xyz,
                        args.parameterization)
                    loss = loss_xyz if loss_ssl_cd is None else loss_ssl_cd + loss_xyz
                    loss.backward()
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    if hasattr(backbone, "step_gumbel_temperature"):
                        backbone.step_gumbel_temperature()
                    if loss_ssl_cd is not None:
                        ep_ssl += ssl_v; ep_cd += ch_v; n_ssl += 1
                    ep_mse += mse_v; ep_nll += nll_v; n_xyz += 1

        val_mse, val_nll = validate(
            backbone, mse_head, gauss_head, probe_val_loader, normalizer, device,
            args.parameterization)
        elapsed = time.time() - t0
        lr_head = optimizer.param_groups[-1]["lr"]
        lr_bb = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else float("nan")
        lr_msg = (f"lr_head={lr_head:.2e}" if args.freeze_backbone
                  else f"lr_bb={lr_bb:.2e} lr_head={lr_head:.2e}")
        if n_ssl == 0:
            log(f"Epoch {epoch+1}/{args.epochs}  "
                f"train xyz_mse={ep_mse/max(n_xyz,1):.4f} nll={ep_nll/max(n_xyz,1):.4f}  "
                f"| val mse={val_mse:.4f} nll={val_nll:.4f} | {lr_msg} | {elapsed:.0f}s")
        else:
            log(f"Epoch {epoch+1}/{args.epochs}  "
                f"train ssl={ep_ssl/max(n_ssl,1):.1f} cd={ep_cd/max(n_ssl,1):.4f} "
                f"xyz_mse={ep_mse/max(n_xyz,1):.4f} nll={ep_nll/max(n_xyz,1):.4f}  "
                f"| val mse={val_mse:.4f} nll={val_nll:.4f} | {lr_msg} | {elapsed:.0f}s")

        if wandb and wandb.run:
            ld = {"epoch": epoch, "train_xyz_mse": ep_mse/max(n_xyz, 1),
                  "train_xyz_nll": ep_nll/max(n_xyz, 1),
                  "val_xyz_mse": val_mse, "val_xyz_nll": val_nll,
                  "lr_head": lr_head}
            if not args.freeze_backbone:
                ld["lr_backbone"] = lr_bb
            if n_ssl > 0:
                ld["train_ssl_loss"] = ep_ssl/max(n_ssl, 1)
                ld["train_cd_loss"] = ep_cd/max(n_ssl, 1)
            wandb.log(ld)

        improved = val_mse < best_val_mse - 1e-6
        if improved:
            best_val_mse = val_mse; no_improve = 0
        else:
            no_improve += 1

        save_payload = {
            "backbone_state_dict": backbone.state_dict(),
            "mse_head_state_dict": mse_head.state_dict(),
            "gauss_head_state_dict": gauss_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
            "normalizer_stats": normalizer.stats(),
            "centroids": {int(k): v.tolist() for k, v in centroids.items()},
            "epoch": epoch, "val_mse": val_mse, "val_nll": val_nll,
            "best_val_mse": best_val_mse, "no_improve": no_improve,
            "args": vars(args),
            "manip_conditioned": True,
            "head_input_dim": head_in,
            "manip_xyz_dir": args.manip_xyz_dir,
            "head_type": args.head_type,
            "parameterization": args.parameterization,
            "backbone_frozen": args.freeze_backbone,
        }
        torch.save(save_payload, os.path.join(ckpt_dir, "finetune_latest.pt"))
        if improved:
            torch.save(save_payload, os.path.join(ckpt_dir, "finetune_best_valmse.pt"))
            log(f"  best val_mse={best_val_mse:.4f} — saved")
        elif (args.early_stop_patience > 0 and no_improve >= args.early_stop_patience):
            log(f"Early stopping at epoch {epoch+1}: "
                f"no improvement for {args.early_stop_patience} epochs")
            break

    log(f"\nDone. Best val_mse={best_val_mse:.4f}")
    log(f"Results: {args.results_dir}")
    if wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
