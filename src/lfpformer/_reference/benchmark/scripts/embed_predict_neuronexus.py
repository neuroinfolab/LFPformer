"""Zero-shot embed + ft_gauss XYZ predict on Neuronexus 8-shank LFP for one
240k w2v2 variant: distc | qv3 (ssl_masked) | pdm (probe_dist_v2_matched).

Inputs
------
- Per-session pickles at /workspace/ws4/region_decoding/data/Neuronexus/lfp/
  Each file is a list of tuples (signal[3750] @ 1250 Hz, region_label, chunk_idx,
  channel_id). Region label ints in {0..4} ≈ {UNK, CA1, CA2, CA3, DG}.
- Per-channel probe geometry from channel_map.mat (Kilosort format):
    xcoords (16 unique x positions, 2-col × 8 shanks)
    ycoords (depth, 128 unique values @ 20 µm spacing)
    kcoords (shank id 1..8)
    connected (good channel mask)

Outputs (per variant under output-dir/variant_<name>/)
------
- embeddings.npz : embeddings (N, 768), region_label, session, channel_id,
                   probe_x, probe_y, shank, chunk_idx
- predictions.npz: pred_xyz (N, 3) µm in Allen CCF, plus same metadata
- meta.json      : variant, ckpt paths, n_samples, sessions

Memory: all signals loaded once (~3.8 GB float32) and reused across SSL +
ft_gauss passes. No subsampling.
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
import yaml

BENCH_ROOT = Path(__file__).resolve().parent.parent
ALPHA_ROOT = BENCH_ROOT.parent
sys.path.insert(0, str(BENCH_ROOT))
sys.path.insert(0, str(ALPHA_ROOT))

from backbones.wav2vec2_backbone import Wav2Vec2Backbone  # noqa: E402
from scripts.finetune_xyz_gaussian import GaussianXYZHead  # noqa: E402


# ── Variant registry ─────────────────────────────────────────────────────
NEURIPS_ROOT = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026'))
VARIANTS = {
    "distc": {
        "ssl_dir": NEURIPS_ROOT / "w2v2_240k_distc_best",
        "ssl_ckpt_name": "ssl_wav2vec2_best.pt",
        "config": BENCH_ROOT / "configs/wav2vec2_5k_distc_fullft.yaml",
        "display": "w2v2_240k_distc",
    },
    "qv3": {
        "ssl_dir": NEURIPS_ROOT / "w2v2_240k_quantizer_best_v3",
        "ssl_ckpt_name": "ssl_wav2vec2_best.pt",
        "config": BENCH_ROOT / "configs/wav2vec2_quantizer_v3_fullft.yaml",
        "display": "w2v2_240k_quantizer_v3 (ssl_masked)",
    },
    "pdm": {
        "ssl_dir": NEURIPS_ROOT / "w2v2_240k_probe_dist_v2_matched",
        "ssl_ckpt_name": "ssl_wav2vec2_best.pt",
        "config": BENCH_ROOT / "configs/wav2vec2_probe_dist_v2_fullft.yaml",
        "display": "w2v2_240k_probe_dist_v2_matched (pdm)",
    },
    "distc_75k": {
        "ssl_dir": NEURIPS_ROOT / "w2v2_75k_distc_best",
        "ssl_ckpt_name": "ssl_wav2vec2_best.pt",
        "config": BENCH_ROOT / "configs/wav2vec2_5k_distc_fullft.yaml",
        "display": "w2v2_75k_distc (best_probe)",
    },
    "qv3_75k": {
        "ssl_dir": NEURIPS_ROOT / "w2v2_75k_quantizer_best_v3",
        "ssl_ckpt_name": "ssl_wav2vec2_best_valloss.pt",
        "config": BENCH_ROOT / "configs/wav2vec2_quantizer_v3_fullft.yaml",
        "display": "w2v2_75k_quantizer_v3 (valloss)",
    },
    "pdm_75k": {
        "ssl_dir": NEURIPS_ROOT / "w2v2_75k_probe_dist_v2_matched",
        "ssl_ckpt_name": "ssl_wav2vec2_best_valloss.pt",
        "config": BENCH_ROOT / "configs/wav2vec2_probe_dist_v2_fullft.yaml",
        "display": "w2v2_75k_probe_dist_v2_matched (valloss)",
    },
}

# Class label mapping for plot legends. `mapping.json` normalizes string aliases
# to {CA1, CA2, CA3, DG, UNK}; the small CA2 column in labels_*.npy + 5-class
# union confirms this ordering.
REGION_NAMES = {0: "UNK", 1: "CA1", 2: "CA2", 3: "CA3", 4: "DG"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Data loading ─────────────────────────────────────────────────────────
def load_channel_map(mat_path):
    """Return dict id→(probe_x, probe_y, shank, connected) for 1024 channels.

    Channel IDs in pickles are 0-indexed (the 'channel_id' tuple element).
    Kilosort `chanMap` is 1-indexed; chanMap0ind matches the pickle ids.
    """
    with h5py.File(mat_path, "r") as f:
        x = f["xcoords"][...].squeeze().astype(np.float32)
        y = f["ycoords"][...].squeeze().astype(np.float32)
        k = f["kcoords"][...].squeeze().astype(np.int32)
        conn = f["connected"][...].squeeze().astype(np.uint8)
        ids = f["chanMap0ind"][...].squeeze().astype(np.int64)
    table = {}
    for i, ch_id in enumerate(ids):
        table[int(ch_id)] = (
            float(x[i]), float(y[i]), int(k[i]), int(conn[i])
        )
    log(f"channel_map: {len(table)} channels, "
        f"x range [{x.min():.0f},{x.max():.0f}] µm, "
        f"y range [{y.min():.0f},{y.max():.0f}] µm, "
        f"shanks {sorted(set(int(s) for s in k))}, "
        f"connected={int(conn.sum())}/{len(conn)}")
    return table


def load_neuronexus_all(lfp_dir, channel_table):
    """Concatenate all 8 sessions into flat arrays."""
    files = sorted(glob.glob(os.path.join(lfp_dir, "*_lfp.pickle")))
    files = [f for f in files if "full" not in os.path.basename(f)]
    log(f"Found {len(files)} session pickles under {lfp_dir}")

    sigs_chunks = []
    region_chunks, chunk_chunks, chid_chunks = [], [], []
    px_chunks, py_chunks, shank_chunks, sess_chunks = [], [], [], []

    for fp in files:
        sess_name = os.path.basename(fp).replace("_lfp.pickle", "")
        with open(fp, "rb") as fh:
            data = pickle.load(fh)
        n = len(data)
        sigs = np.empty((n, 3750), dtype=np.float32)
        regs = np.empty(n, dtype=np.int8)
        chs = np.empty(n, dtype=np.int32)
        cks = np.empty(n, dtype=np.int32)
        pxs = np.full(n, np.nan, dtype=np.float32)
        pys = np.full(n, np.nan, dtype=np.float32)
        shs = np.full(n, -1, dtype=np.int8)
        for i, s in enumerate(data):
            sigs[i] = np.asarray(s[0], dtype=np.float32)
            regs[i] = int(s[1])
            cks[i] = int(s[2])
            chs[i] = int(s[3])
            geo = channel_table.get(int(s[3]))
            if geo is not None:
                pxs[i], pys[i], shs[i] = geo[0], geo[1], geo[2]
        sigs_chunks.append(sigs)
        region_chunks.append(regs)
        chid_chunks.append(chs)
        chunk_chunks.append(cks)
        px_chunks.append(pxs)
        py_chunks.append(pys)
        shank_chunks.append(shs)
        sess_chunks.append(np.full(n, sess_name, dtype=object))
        unmapped = int(np.isnan(pxs).sum())
        log(f"  {sess_name:20s} N={n:>6d}  classes={sorted(set(int(r) for r in regs))} "
            f"chans={len(set(int(c) for c in chs))} unmapped={unmapped}")
        del data

    out = {
        "signals":    np.concatenate(sigs_chunks, axis=0),
        "region_label": np.concatenate(region_chunks),
        "channel_id":   np.concatenate(chid_chunks),
        "chunk_idx":    np.concatenate(chunk_chunks),
        "probe_x":      np.concatenate(px_chunks),
        "probe_y":      np.concatenate(py_chunks),
        "shank":        np.concatenate(shank_chunks),
        "session":      np.concatenate(sess_chunks).astype(str),
    }
    log(f"TOTAL: {len(out['signals'])} samples, "
        f"signal mem={out['signals'].nbytes / 1e9:.2f} GB")
    return out


# ── Model loading ────────────────────────────────────────────────────────
def load_ssl_backbone(ckpt_path, config_path, device):
    log(f"SSL ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["rand_init"] = True
    cfg["queue_size"] = 0
    backbone = Wav2Vec2Backbone(cfg)
    sd = ckpt.get("backbone_state_dict") or ckpt.get("model_state_dict") or ckpt
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    enc_missing = [k for k in missing
                   if k.startswith("model.wav2vec2.encoder")]
    if enc_missing:
        raise RuntimeError(f"SSL encoder missing {len(enc_missing)} keys")
    log(f"  loaded: missing={len(missing)} unexpected={len(unexpected)} "
        f"epoch={ckpt.get('epoch')} probe_acc={ckpt.get('probe_acc')}")
    backbone.to(device).eval()
    return backbone


def load_ft_gauss(ckpt_path, config_path, device):
    log(f"ft_gauss ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("backbone_state_dict", "head_state_dict", "y_mean", "y_std"):
        if k not in ckpt:
            raise RuntimeError(f"ft_gauss ckpt missing '{k}'")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["rand_init"] = True
    cfg["queue_size"] = 0
    backbone = Wav2Vec2Backbone(cfg)
    missing, unexpected = backbone.load_state_dict(
        ckpt["backbone_state_dict"], strict=False)
    crit = ("model.wav2vec2.encoder",
            "model.wav2vec2.feature_extractor",
            "model.wav2vec2.feature_projection")
    cmissing = [k for k in missing if k.startswith(crit)]
    if cmissing:
        raise RuntimeError(
            f"ft_gauss backbone missing critical keys: {cmissing[:5]}")
    backbone.to(device).eval()

    hidden_dim = int(ckpt.get("hidden_dim", cfg.get("hidden_size", 768)))
    head = GaussianXYZHead(input_dim=hidden_dim).to(device)
    head.load_state_dict(ckpt["head_state_dict"], strict=True)
    head.eval()
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
    log(f"  backbone missing={len(missing)} unexpected={len(unexpected)}")
    log(f"  head: GaussianXYZHead(in={hidden_dim})  "
        f"y_mean={y_mean.tolist()}  y_std={y_std.tolist()}")
    return backbone, head, y_mean, y_std


# ── Inference ────────────────────────────────────────────────────────────
@torch.no_grad()
def forward_embeddings(backbone, signals, src_sr, device, batch_size):
    n = len(signals)
    out = np.empty((n, 768), dtype=np.float32)
    total = (n + batch_size - 1) // batch_size
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = signals[i:i + batch_size]
        wav = torch.from_numpy(batch).to(device, non_blocking=True)
        if src_sr != 16000:
            wav = torchaudio.functional.resample(wav, src_sr, 16000)
        emb = backbone.encode(wav)
        out[i:i + len(batch)] = emb.cpu().numpy()
        bi = i // batch_size + 1
        if bi % 50 == 0 or bi == total:
            dt = time.time() - t0
            log(f"    embed batch {bi}/{total}  {dt:.1f}s  "
                f"({(i + len(batch))/dt:.0f} samples/s)")
    return out


@torch.no_grad()
def forward_xyz(backbone, head, signals, src_sr, y_mean, y_std,
                device, batch_size):
    n = len(signals)
    pred = np.empty((n, 3), dtype=np.float32)
    sigma = np.empty((n, 3), dtype=np.float32)
    total = (n + batch_size - 1) // batch_size
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = signals[i:i + batch_size]
        wav = torch.from_numpy(batch).to(device, non_blocking=True)
        if src_sr != 16000:
            wav = torchaudio.functional.resample(wav, src_sr, 16000)
        emb = backbone.encode(wav)
        mu, log_var = head(emb)
        mu_np = mu.cpu().numpy()
        sigma_norm = torch.exp(0.5 * log_var).cpu().numpy()  # z-score space σ
        pred[i:i + len(batch)] = mu_np * y_std[None, :] + y_mean[None, :]
        sigma[i:i + len(batch)] = sigma_norm * y_std[None, :]   # µm
        bi = i // batch_size + 1
        if bi % 50 == 0 or bi == total:
            dt = time.time() - t0
            log(f"    xyz batch {bi}/{total}  {dt:.1f}s  "
                f"({(i + len(batch))/dt:.0f} samples/s)")
    return pred, sigma


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    ap.add_argument("--lfp-dir",
                    default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws4/region_decoding/data/Neuronexus/lfp'))
    ap.add_argument("--channel-map",
                    default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws4/region_decoding/data/Neuronexus/channel_map.mat'))
    ap.add_argument("--output-dir",
                    default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026/neuronexus_zero_shot'))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--sample-rate", type=int, default=1250)
    ap.add_argument("--skip-ssl", action="store_true",
                    help="Skip zero-shot SSL embedding pass")
    ap.add_argument("--skip-xyz", action="store_true",
                    help="Skip ft_gauss XYZ prediction pass")
    args = ap.parse_args()

    variant_cfg = VARIANTS[args.variant]
    out_dir = Path(args.output_dir) / f"variant_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"=== Variant: {args.variant} ({variant_cfg['display']}) ===")
    log(f"Device: {device}")
    log(f"Output: {out_dir}")

    chan_table = load_channel_map(args.channel_map)
    data = load_neuronexus_all(args.lfp_dir, chan_table)
    sigs = data["signals"]

    ssl_ckpt = variant_cfg["ssl_dir"] / "checkpoints" / variant_cfg["ssl_ckpt_name"]
    ftg_ckpt = (variant_cfg["ssl_dir"] / "finetune_gaussian_fullft" /
                "best_gaussian_xyz_model.pt")
    config_path = variant_cfg["config"]

    meta = {
        "variant": args.variant,
        "display": variant_cfg["display"],
        "ssl_ckpt": str(ssl_ckpt),
        "ftg_ckpt": str(ftg_ckpt),
        "config": str(config_path),
        "n_samples": int(len(sigs)),
        "sample_rate_in": args.sample_rate,
        "sessions": sorted(set(data["session"].tolist())),
    }

    # ── Pass 1: zero-shot SSL embeddings ──
    if not args.skip_ssl:
        log("--- Pass 1/2: zero-shot SSL embeddings ---")
        backbone = load_ssl_backbone(ssl_ckpt, config_path, device)
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
        del backbone, emb
        torch.cuda.empty_cache()

    # ── Pass 2: ft_gauss XYZ predictions ──
    if not args.skip_xyz:
        log("--- Pass 2/2: ft_gauss fullft XYZ predictions ---")
        backbone, head, y_mean, y_std = load_ft_gauss(
            ftg_ckpt, config_path, device)
        pred, sigma = forward_xyz(backbone, head, sigs, args.sample_rate,
                                  y_mean, y_std, device, args.batch_size)
        pred_npz = out_dir / "predictions.npz"
        np.savez_compressed(
            pred_npz,
            pred_xyz=pred,
            pred_sigma=sigma,
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
        log(f"  pred_xyz stats: AP=[{pred[:,0].min():.0f},{pred[:,0].max():.0f}] "
            f"DV=[{pred[:,1].min():.0f},{pred[:,1].max():.0f}] "
            f"ML=[{pred[:,2].min():.0f},{pred[:,2].max():.0f}] (µm)")
        log(f"  pred_sigma stats (µm): AP=[{sigma[:,0].min():.0f},{sigma[:,0].max():.0f}] "
            f"DV=[{sigma[:,1].min():.0f},{sigma[:,1].max():.0f}] "
            f"ML=[{sigma[:,2].min():.0f},{sigma[:,2].max():.0f}] "
            f"median={np.median(sigma, axis=0).tolist()}")
        del backbone, head, pred, sigma

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Saved meta.json")
    log(f"DONE [{args.variant}]")


if __name__ == "__main__":
    main()
