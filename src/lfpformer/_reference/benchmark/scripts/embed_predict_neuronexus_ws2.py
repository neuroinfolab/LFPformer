#!/usr/bin/env python3
"""Zero-shot embed + predict on NeuroNexus 8-shank LFP using ws2's
IBL-trained SSL backbone + Gaussian probe (μ, log_var) + hemi side head.

Schema-compatible with embed_predict_neuronexus.py:
    predictions.npz   pred_xyz, pred_sigma, region_label, channel_id, chunk_idx,
                      probe_x, probe_y, shank, session, y_mean, y_std (per-axis span)

Default variant: ibl_chandist_s500_240k.

Coordinate convention (matches Allen CCF µm):
    AP = ws2 ap_norm * (ap_max - ap_min) + ap_min
    DV = ws2 dv_norm * (dv_max - dv_min) + dv_min
    d  = clip(ws2 d_norm, 0, +∞) * d_max         (midline distance)
    ML = midline ± d   based on hemi side_head sigmoid
σ in µm (chain rule on min-max normalisation):
    σ_AP = exp(0.5 * log_var[0]) * (ap_max - ap_min)
    σ_DV = exp(0.5 * log_var[1]) * (dv_max - dv_min)
    σ_d  = exp(0.5 * log_var[2]) * d_max     (used for ML uncertainty)
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parent
ALPHABRAIN_ROOT = BENCH_ROOT.parent
sys.path.insert(0, str(BENCH_ROOT))
sys.path.insert(0, str(ALPHABRAIN_ROOT))

from backbones.wav2vec2_backbone import Wav2Vec2Backbone  # noqa: E402

# Reuse NN data loading + log helper from the Allen-variant script
from scripts.embed_predict_neuronexus import (  # noqa: E402
    log, load_channel_map, load_neuronexus_all,
)


PRESETS = {
    "ibl_chandist_s500_240k": {
        "ssl_ckpt": (__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/benchmarks/w2v2_v2_crossneg_240k_chandist_s500_clean_6896816/checkpoints/ssl_wav2vec2_best.pt')),
        "probe_ckpt": (__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/eval_v2/models/crossneg_240k_cd_s500_clean_perchannel/probes/regression_probes.pt')),
        "display": "IBL crossneg_240k chandist σ=500 (ws2 gaussian probe)",
    },
    "ibl_chandist_s500_75k": {
        "ssl_ckpt": (__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/benchmarks/w2v2_v2_crossneg_75k_chandist_s500_5653103/checkpoints/ssl_wav2vec2_best_valloss.pt')),
        "probe_ckpt": (__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/eval_v2/models/crossneg_75k_cd_s500_perchannel/probes/regression_probes.pt')),
        "display": "IBL crossneg_75k chandist σ=500 (ws2 gaussian probe)",
    },
}


# ── Model loading ─────────────────────────────────────────────────────────
def load_ssl_backbone(ckpt_path, device):
    log(f"SSL backbone: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "backbone_state_dict" not in ck or "config" not in ck:
        raise RuntimeError("ckpt missing backbone_state_dict / config")
    cfg = dict(ck["config"])
    cfg["rand_init"] = True       # weights from ckpt; skip HF download
    cfg["queue_size"] = 0          # no MoCo queue at inference
    log(f"  backbone={cfg.get('backbone')} hidden={cfg.get('hidden_size')}")
    backbone = Wav2Vec2Backbone(cfg)
    missing, unexpected = backbone.load_state_dict(
        ck["backbone_state_dict"], strict=False)
    critical = (
        "model.wav2vec2.encoder",
        "model.wav2vec2.feature_extractor",
        "model.wav2vec2.feature_projection",
        "model.wav2vec2.masked_spec_embed",
    )
    crit_missing = [k for k in missing if k.startswith(critical)]
    if crit_missing:
        raise RuntimeError(
            f"Backbone weights missing ({len(crit_missing)} keys): "
            f"{crit_missing[:5]}...")
    log(f"  loaded: missing={len(missing)} unexpected={len(unexpected)}")
    backbone.to(device).eval()
    return backbone


def load_gaussian_probe(probe_path, device):
    log(f"Gaussian probe: {probe_path}")
    ck = torch.load(probe_path, map_location="cpu", weights_only=False)
    g = ck["gaussian_state_dict"]
    h = ck["hemi_state_dict"]
    out = {
        "W_mu":   g["mu.weight"].to(device).float(),       # (3, 768)
        "b_mu":   g["mu.bias"].to(device).float(),         # (3,)
        "W_lv":   g["log_var.weight"].to(device).float(),  # (3, 768)
        "b_lv":   g["log_var.bias"].to(device).float(),    # (3,)
        "W_side": h["side_head.weight"].to(device).float(),# (1, 768)
        "b_side": h["side_head.bias"].to(device).float(),  # (1,)
        "norm":   ck["normalizer_stats"],
    }
    log(f"  W_mu={tuple(out['W_mu'].shape)}  W_lv={tuple(out['W_lv'].shape)}  "
        f"W_side={tuple(out['W_side'].shape)}")
    log(f"  norm={out['norm']}")
    log(f"  best_val_nll={ck.get('best_val_nll')}  best_val_mse={ck.get('best_val_mse')}")
    return out


@torch.no_grad()
def forward_gauss_probe(backbone, probe, signals, src_sr, device, batch_size):
    n = len(signals)
    pred_xyz = np.empty((n, 3), dtype=np.float32)
    pred_sigma = np.empty((n, 3), dtype=np.float32)
    p_right = np.empty(n, dtype=np.float32)

    norm = probe["norm"]
    ap_span = float(norm["ap_max"] - norm["ap_min"])
    dv_span = float(norm["dv_max"] - norm["dv_min"])
    d_span = float(norm["d_max"])    # min is 0
    midline = float(norm["midline"])

    total = (n + batch_size - 1) // batch_size
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = signals[i:i + batch_size]
        wav = torch.from_numpy(batch).to(device, non_blocking=True)
        if src_sr != 16000:
            wav = torchaudio.functional.resample(wav, src_sr, 16000)
        emb = backbone.encode(wav)                                  # (B, 768)
        mu = emb @ probe["W_mu"].T + probe["b_mu"]                  # (B, 3) norm
        lv = emb @ probe["W_lv"].T + probe["b_lv"]                  # (B, 3) log σ²-norm
        side_logit = (emb @ probe["W_side"].T + probe["b_side"]).squeeze(-1)

        mu_np = mu.cpu().numpy()
        lv_np = lv.cpu().numpy()
        sl_np = side_logit.cpu().numpy()

        sigma_norm = np.exp(0.5 * lv_np)         # σ in normalised space, shape (B, 3)
        # Denormalize μ → AP, DV, d (all in µm)
        ap = mu_np[:, 0] * ap_span + float(norm["ap_min"])
        dv = mu_np[:, 1] * dv_span + float(norm["dv_min"])
        d = np.clip(mu_np[:, 2], 0.0, None) * d_span
        # Hemi sign for ML
        pr = 1.0 / (1.0 + np.exp(-sl_np))
        sign = np.where(pr >= 0.5, 1.0, -1.0).astype(np.float32)
        ml = midline + sign * d

        pred_xyz[i:i + len(batch)] = np.stack([ap, dv, ml], axis=1)
        # σ in µm via chain rule on min-max normalisation
        s_ap = sigma_norm[:, 0] * ap_span
        s_dv = sigma_norm[:, 1] * dv_span
        s_d = sigma_norm[:, 2] * d_span      # ML σ is the same as midline-distance σ
        pred_sigma[i:i + len(batch)] = np.stack([s_ap, s_dv, s_d], axis=1)
        p_right[i:i + len(batch)] = pr

        bi = i // batch_size + 1
        if bi % 50 == 0 or bi == total:
            dt = time.time() - t0
            log(f"    batch {bi}/{total}  {dt:.1f}s  "
                f"({(i + len(batch))/dt:.0f} samples/s)")

    return pred_xyz, pred_sigma, p_right


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ibl_chandist_s500_240k",
                    choices=list(PRESETS.keys()))
    ap.add_argument("--lfp-dir",
                    default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws4/region_decoding/data/Neuronexus/lfp'))
    ap.add_argument("--channel-map",
                    default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws4/region_decoding/data/Neuronexus/Neuronexus_chanMap.mat'))
    ap.add_argument("--output-dir",
                    default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026/neuronexus_zero_shot'))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--sample-rate", type=int, default=1250)
    ap.add_argument("--skip-ssl", action="store_true",
                    help="Skip SSL embeddings pass (only run XYZ probe)")
    ap.add_argument("--skip-xyz", action="store_true",
                    help="Skip XYZ probe pass (only run SSL embeddings)")
    args = ap.parse_args()

    preset = PRESETS[args.variant]
    out_base = Path(args.output_dir)
    out_dir = out_base / f"variant_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}  CUDA: {torch.cuda.is_available()}")
    log(f"=== Variant: {args.variant} ({preset['display']}) ===")
    log(f"=== Output: {out_dir} ===")

    log("Loading channel map ...")
    ch_table = load_channel_map(args.channel_map)
    log("Loading NeuroNexus signals ...")
    data = load_neuronexus_all(args.lfp_dir, ch_table)
    sigs = data["signals"]
    log(f"  signals shape={sigs.shape} dtype={sigs.dtype}")

    meta = {
        "variant": args.variant,
        "display": preset["display"],
        "ssl_ckpt": preset["ssl_ckpt"],
        "probe_ckpt": preset["probe_ckpt"],
        "n_samples": int(len(sigs)),
        "sample_rate_in": args.sample_rate,
        "sessions": sorted(set(data["session"].tolist())),
    }

    # ── Pass 1: zero-shot SSL embeddings ──
    if not args.skip_ssl:
        log("--- Pass 1/2: zero-shot SSL embeddings ---")
        backbone = load_ssl_backbone(preset["ssl_ckpt"], device)

        # Inline embedding loop (avoid reloading another backbone for Pass 2)
        from scripts.embed_predict_neuronexus import forward_embeddings
        emb = forward_embeddings(backbone, sigs, args.sample_rate,
                                 device, args.batch_size)
        emb_npz = out_dir / "embeddings.npz"
        np.savez_compressed(
            emb_npz,
            embeddings=emb,
            region_label=data["region_label"],
            channel_id=data["channel_id"],
            chunk_idx=data["chunk_idx"],
            probe_x=data["probe_x"],
            probe_y=data["probe_y"],
            shank=data["shank"],
            session=data["session"],
        )
        log(f"Saved {emb_npz} ({emb.shape}, "
            f"{emb_npz.stat().st_size / 1e6:.1f} MB)")
        del emb
        torch.cuda.empty_cache()
    else:
        backbone = load_ssl_backbone(preset["ssl_ckpt"], device)

    # ── Pass 2: ws2 gauss probe XYZ ──
    if not args.skip_xyz:
        log("--- Pass 2/2: ws2 gauss probe (μ, log_var) XYZ predictions ---")
        probe = load_gaussian_probe(preset["probe_ckpt"], device)
        pred, sigma, p_right = forward_gauss_probe(
            backbone, probe, sigs, args.sample_rate, device, args.batch_size)
        norm = probe["norm"]
        # Pseudo y_mean/y_std for downstream compatibility (per-axis spans/center).
        y_mean = np.array([(norm["ap_min"] + norm["ap_max"]) / 2,
                           (norm["dv_min"] + norm["dv_max"]) / 2,
                           norm["midline"]], dtype=np.float32)
        y_std = np.array([norm["ap_max"] - norm["ap_min"],
                          norm["dv_max"] - norm["dv_min"],
                          2.0 * norm["d_max"]], dtype=np.float32)

        pred_npz = out_dir / "predictions.npz"
        np.savez_compressed(
            pred_npz,
            pred_xyz=pred,
            pred_sigma=sigma,
            p_right=p_right,
            region_label=data["region_label"],
            channel_id=data["channel_id"],
            chunk_idx=data["chunk_idx"],
            probe_x=data["probe_x"],
            probe_y=data["probe_y"],
            shank=data["shank"],
            session=data["session"],
            y_mean=y_mean,
            y_std=y_std,
        )
        log(f"Saved {pred_npz} ({pred.shape}, "
            f"{pred_npz.stat().st_size / 1e6:.1f} MB)")
        log(f"  pred_xyz stats (µm): "
            f"AP=[{pred[:,0].min():.0f},{pred[:,0].max():.0f}]  "
            f"DV=[{pred[:,1].min():.0f},{pred[:,1].max():.0f}]  "
            f"ML=[{pred[:,2].min():.0f},{pred[:,2].max():.0f}]")
        log(f"  pred_sigma stats (µm): "
            f"AP={np.median(sigma[:,0]):.0f}  DV={np.median(sigma[:,1]):.0f}  "
            f"d={np.median(sigma[:,2]):.0f}  (medians)")
        log(f"  p_right: median={np.median(p_right):.3f}  "
            f"frac>=0.5={(p_right >= 0.5).mean():.3f}")
        del backbone, probe, pred, sigma

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Wrote {meta_path}")
    log("DONE.")


if __name__ == "__main__":
    main()
