#!/usr/bin/env python3
"""Eval for the manipulator-conditioned (residual) FT model.

Mirrors eval_regression_probe.py --use_model_head, but the heads take
[emb ⊕ manip_norm] and predict the RESIDUAL (pred = manip_norm + Δ). Writes
test_fold_*_predictions.npz in the EXACT format recompute_refined_3d_metrics.py
expects (keys: pred_xyz, pred_xyz_gauss, pred_sigma_gauss, true_xyz, fine_id,
meta_str), so the existing Kabsch pipeline runs on it unchanged.

After this, run:
  recompute_refined_3d_metrics.py --models_root <parent> --probe_key pred_xyz_gauss

manip_xyz is the precomputed per-split (N,3) µm array (precompute_manip_xyz.py),
aligned to CompactDataset idx order. Invalid-manip rows are dropped. Run in
w2v2_cuda_cu128_ibl.sif.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

PB_BENCH = os.environ.get("PB_BENCH",
                          __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/Lfp2vec_benchmarks'))
PB_ROOT = os.path.dirname(PB_BENCH)
for _p in (os.path.join(PB_BENCH, "scripts"), PB_BENCH, PB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backbones import get_backbone
from utils.preprocessing import preprocess_batch
from train_ssl import log
from eval_regression_probe import CentroidNormalizer, GaussianLinearProbe
from blind_localization.data.lazyloader_dataset import CompactDataset
from torch.utils.data import DataLoader


class ManipMLPHead(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GaussianMLPProbe(nn.Module):
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
    return manip_norm + head_output if parameterization == "residual" else head_output


def normalizer_from_stats(stats):
    nz = CentroidNormalizer.__new__(CentroidNormalizer)
    nz.ap_min = stats["ap_min"]; nz.ap_max = stats["ap_max"]
    nz.dv_min = stats["dv_min"]; nz.dv_max = stats["dv_max"]
    nz.midline = stats["midline"]
    nz.d_min = stats["d_min"]; nz.d_max = stats["d_max"]
    return nz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval_data_dir", required=True)
    p.add_argument("--manip_xyz_dir",
                   default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/one_cache/manip_xyz/eval_v2_100k'))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not ckpt.get("manip_conditioned"):
        raise SystemExit("checkpoint is NOT manip_conditioned — use eval_regression_probe.py")
    config = ckpt.get("config", {})
    config["rand_init"] = True  # skip HF pretrained load (offline dual-snapshot trap); backbone_state_dict overwrites all weights
    head_in = int(ckpt["head_input_dim"])
    head_type = ckpt.get("head_type", ckpt.get("args", {}).get("head_type", "linear"))
    parameterization = ckpt.get(
        "parameterization", ckpt.get("args", {}).get("parameterization", "residual"))
    log(f"manip-conditioned ckpt: epoch={ckpt.get('epoch')} "
        f"best_val_mse={ckpt.get('best_val_mse')}  head_in={head_in} "
        f"head={head_type} parameterization={parameterization}")

    backbone = get_backbone(config.get("backbone", "wav2vec2"), config).to(device)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.eval()
    mse_head, gauss_head = make_heads(head_type, head_in, device)
    mse_head.load_state_dict(ckpt["mse_head_state_dict"]); mse_head.eval()
    gauss_head.load_state_dict(ckpt["gauss_head_state_dict"]); gauss_head.eval()

    normalizer = normalizer_from_stats(ckpt["normalizer_stats"])
    scale = np.array([normalizer.ap_max - normalizer.ap_min,
                      normalizer.dv_max - normalizer.dv_min,
                      normalizer.d_max - normalizer.d_min], dtype=np.float64)

    compact_kwargs = dict(
        include_labels=True, dataset="IBL", verbose=False, atlas_depth=9,
        already_resampled=False, gpu_resample=True, input_sr=1250, model_sr=16000)
    pred_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    def get_meta_str(ds, idx):
        fi, off = ds._find_file_and_offset(idx)
        return str(ds._load_file(fi)[off][1])

    _splits = ["probe_train", "probe_val"] + [f"test_fold_{f}" for f in range(5)]
    for split in _splits:
        sd = os.path.join(args.eval_data_dir, split)
        if not os.path.isdir(sd):
            log(f"  {split}: missing split dir, skip"); continue
        if not os.path.exists(os.path.join(args.manip_xyz_dir, f"{split}_manip_xyz.npy")):
            log(f"  {split}: missing manip cache, skip"); continue
        ds = CompactDataset(compact_dir=sd, **compact_kwargs)
        n = len(ds)
        # coords + fine_id + meta_str in idx order
        true = np.zeros((n, 3), dtype=np.float64)
        fids = np.zeros(n, dtype=np.int64)
        metas = []
        for i in range(n):
            _, _, c = ds[i]
            h = ds.get_hierarchy(i)
            true[i] = (float(c[0]), float(c[1]), float(c[2]))
            fids[i] = int(h["fine_id"]) if h and "fine_id" in h else 0
            metas.append(get_meta_str(ds, i))
        manip = np.load(os.path.join(args.manip_xyz_dir, f"{split}_manip_xyz.npy"))
        valid = np.load(os.path.join(args.manip_xyz_dir, f"{split}_valid.npy"))
        if len(manip) != n:
            raise ValueError(f"{split}: manip rows {len(manip)} != dataset {n}")
        if len(valid) != n or valid.dtype != bool:
            raise ValueError(f"{split}: valid mask len/dtype mismatch "
                             f"({len(valid)},{valid.dtype}) vs n={n}")

        manip_norm_all = normalizer.encode(manip)  # (n,3); NaN for invalid rows
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        pred_xyz = np.zeros((n, 3), dtype=np.float32)
        pred_gauss = np.zeros((n, 3), dtype=np.float32)
        pred_sigma = np.zeros((n, 3), dtype=np.float32)
        off = 0
        with torch.no_grad():
            for batch in loader:
                sig = batch[0].float().to(device, non_blocking=True)
                b = sig.shape[0]
                x = preprocess_batch(sig, backbone.needs_resampling,
                                     backbone.needs_normalization)
                emb = backbone.encode(x)  # (b, hidden)
                mn = torch.from_numpy(
                    manip_norm_all[off:off + b]).float().to(device)
                feat = torch.cat([emb, mn], dim=1)
                pred_n = apply_coordinate_parameterization(
                    mse_head(feat), mn, parameterization)
                mu_out, log_var = gauss_head(feat)
                mu_n = apply_coordinate_parameterization(
                    mu_out, mn, parameterization)
                pred_xyz[off:off + b] = normalizer.decode(pred_n.cpu().numpy())
                pred_gauss[off:off + b] = normalizer.decode(mu_n.cpu().numpy())
                pred_sigma[off:off + b] = (
                    np.sqrt(np.exp(log_var.cpu().numpy())) * scale).astype(np.float32)
                off += b
        assert off == n, f"{split}: extracted {off} != {n}"

        keep = valid
        out = os.path.join(pred_dir, f"{split}_predictions.npz")
        np.savez_compressed(
            out,
            pred_xyz=pred_xyz[keep].astype(np.float32),
            pred_xyz_gauss=pred_gauss[keep].astype(np.float32),
            pred_sigma_gauss=pred_sigma[keep].astype(np.float32),
            true_xyz=true[keep].astype(np.float32),
            fine_id=fids[keep].astype(np.int64),
            meta_str=np.array([metas[i] for i in np.where(keep)[0]], dtype="U200"),
        )
        # quick folded RMSE sanity (raw, no Kabsch)
        from apply_kabsch_posthoc_ibl import fold_lr
        a = fold_lr(pred_gauss[keep]); t = fold_lr(true[keep])
        rmse = float(np.sqrt(((a - t) ** 2).sum(1).mean()))
        log(f"  {split}: kept {keep.sum()}/{n}  raw folded gauss RMSE={rmse:.1f} µm  -> {out}")

    log(f"\nDone. predictions -> {pred_dir}")
    log("Next: recompute_refined_3d_metrics.py --models_root "
        f"{os.path.dirname(args.output_dir)} --probe_key pred_xyz_gauss")


if __name__ == "__main__":
    main()
