"""Apply ws2's `kabsch_per_probe_raw` posthoc to IBL ViT/AnyNet/baseline raw preds.

For each model, loads raw per-sample predictions, derives per-sample (pid, ch)
from SSL's meta_str (row-aligned across all models), then runs ws2's
anatomy-aware SVD-Kabsch using NP1.0 channel geometry as the rigid template.

Saves refined predictions to:
    /workspace/ws1/ray_results/Neurips_2026/kabsch_refined_ibl/<key>_kabsch.npz
        keys = preds_kabsch (500000,3), true (500000,3),
               pids_str (500000,), channels (500000,), n_probes_fit, n_flipped

Reports overall 3D RMSE before / after Kabsch, both raw (no fold) and
folded-LR (matches ws2's headline metric convention).

Source primitives lifted from
    /workspace/ws2/Alphabrain_staging/Lfp2vec_benchmarks/scripts/recompute_refined_3d_metrics.py
    /workspace/ws2/Alphabrain_staging/Lfp2vec_benchmarks/scripts/viz_lp_eval_spatial_cloud_5row.py

Container: any with numpy. CPU only, ~2 min total for 8 models.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Bring ws2's primitives onto sys.path ──────────────────────────────
PL_ROOT = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging')
sys.path.insert(0, os.path.join(PL_ROOT, "Lfp2vec_benchmarks", "scripts"))


def _numpy_shim():
    if not hasattr(np, "_core"):
        import numpy.core as _nc
        sys.modules["numpy._core"] = _nc
        sys.modules["numpy._core.multiarray"] = getattr(_nc, "multiarray", _nc)
    elif not hasattr(np, "core"):
        import numpy._core as _nc
        sys.modules["numpy.core"] = _nc
        sys.modules["numpy.core.multiarray"] = getattr(_nc, "multiarray", _nc)


_numpy_shim()

from recompute_refined_3d_metrics import (   # noqa: E402
    parse_pid_channel,
    channel_xy_onprobe,
    kabsch_rigid,
    ALLEN_MIDLINE_UM,
)


# ── Paths ─────────────────────────────────────────────────────────────────
SSL_DIR = (__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/eval_v2/models/crossneg_75k_cd_s500_perchannel/predictions'))
SSL_DIR_BY_KEY = {
    "ssl_cd_s500_75k":   __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/eval_v2/models/crossneg_75k_cd_s500_perchannel/predictions'),
    "ssl_cd_s300_75k":   __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/eval_v2/models/crossneg_75k_cd_s300_perchannel/predictions'),
    "ssl_maskpred_75k":  __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/eval_v2/models/crossneg_75k_b64_perchannel/predictions'),
    "ssl_chandistc_75k": __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/eval_v2/models/crossneg_75k_cd_only_perchannel/predictions'),
    "ssl_ccfdistc_75k":  __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/eval_v2/models/crossneg_75k_ccf_dist_perchannel/predictions'),
}
BASE_DIR = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026/baseline_ibl')
VIT_NPZ = ("/workspace/ws3/BrainSTORM/ray_results/atlas_experiments/"
           "2026-04-26/vit_rival_xyz_hp_ibl/refine_posthoc_5fold/"
           "refined_predictions.npz")
ANY_NPZ = ("/workspace/ws3/BrainSTORM/ray_results/atlas_experiments/"
           "2026-04-26/anynet_rival_xyz_hp_ibl/refine_posthoc_5fold/"
           "refined_predictions.npz")
OUT_DIR = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026/kabsch_refined_ibl')


# ── ws2's Kabsch lifted verbatim (so this script is self-contained) ───
def kabsch_per_probe_raw(pred_raw, pids, channels):
    """Per-probe SVD-Kabsch using NP1.0 channel geometry as fixed template,
    with anatomical sign correction. Lifted from ws2/viz_lp_eval_spatial_cloud_5row.

    For each probe, build P_template and Q_pred per sample:
        P_i = (x_col, 0, depth) from channel_xy_onprobe(ch_i)
        Q_i = pred_raw[i]
    Solve R, t. Sign-correct: if R rotates depth axis to +DV, flip via Rx180.
    All samples sharing (pid, ch) get the same refined xyz.
    """
    Rx180 = np.diag([1.0, -1.0, -1.0])

    key_to_idx = defaultdict(list)
    for i in range(len(pids)):
        if pids[i] is None or channels[i] is None or channels[i] < 0:
            continue
        key_to_idx[(pids[i], int(channels[i]))].append(i)

    probe_to_samples = defaultdict(list)
    for (pid, ch), idxs in key_to_idx.items():
        x_col, depth = channel_xy_onprobe(int(ch))
        probe_to_samples[pid].append(
            (ch, float(x_col), float(depth),
             np.asarray(idxs, dtype=np.int64)))

    refined = pred_raw.copy()
    n_ok = 0
    n_flipped = 0
    n_total = len(probe_to_samples)
    for pid, entries in probe_to_samples.items():
        if len(entries) < 4:
            continue
        P_list, Q_list = [], []
        for _, x_col, depth, idxs in entries:
            n_s = len(idxs)
            P_list.append(np.tile(np.array([[x_col, 0.0, depth]]), (n_s, 1)))
            Q_list.append(pred_raw[idxs])
        P = np.vstack(P_list)
        Q = np.vstack(Q_list)
        try:
            R, t = kabsch_rigid(P, Q)
        except Exception:
            continue
        if (R @ np.array([0.0, 0.0, 1.0]))[1] > 0:
            R = R @ Rx180
            t = Q.mean(axis=0) - R @ P.mean(axis=0)
            n_flipped += 1
        for ch, x_col, depth, idxs in entries:
            refined[idxs] = R @ np.array([x_col, 0.0, depth]) + t
        n_ok += 1
    return refined, n_ok, n_total, n_flipped


def fold_lr(xyz):
    out = np.asarray(xyz, dtype=np.float64).copy()
    out[:, 2] = ALLEN_MIDLINE_UM - np.abs(out[:, 2] - ALLEN_MIDLINE_UM)
    return out


def rmse_3d(pred, true, fold=False):
    if fold:
        pred = fold_lr(pred)
        true = fold_lr(true)
    return float(np.sqrt(((pred - true) ** 2).sum(axis=1).mean()))


# ── Loaders ──────────────────────────────────────────────────────────────
def load_ssl_meta_and_true():
    """Concat 5 SSL test folds. Returns (meta_str, true) — meta_str authoritative
    for per-sample (pid, ch) parsing across ALL aligned models."""
    metas, trues = [], []
    for f in range(5):
        d = np.load(os.path.join(SSL_DIR, f"test_fold_{f}_predictions.npz"),
                    allow_pickle=True)
        metas.append(np.asarray(d["meta_str"]))
        trues.append(np.asarray(d["true_xyz"], dtype=np.float64))
    return np.concatenate(metas), np.concatenate(trues)


def load_baseline_raw(name):
    p = os.path.join(BASE_DIR, f"preds_{name}_ibl.npz")
    d = np.load(p, allow_pickle=True)
    return (np.asarray(d["preds"], dtype=np.float64),
            np.asarray(d["true"],  dtype=np.float64))


def load_vsp_raw(npz_path, key="gauss_raw"):
    d = np.load(npz_path, allow_pickle=True)
    return (np.asarray(d[key], dtype=np.float64),
            np.asarray(d["true"], dtype=np.float64))


def load_ssl_pred_raw(pred_dir, pred_key="pred_xyz"):
    """Concat 5 SSL test folds for arbitrary SSL anchor. Returns (pred, true)."""
    preds, trues = [], []
    for f in range(5):
        d = np.load(os.path.join(pred_dir, f"test_fold_{f}_predictions.npz"),
                    allow_pickle=True)
        preds.append(np.asarray(d[pred_key], dtype=np.float64))
        trues.append(np.asarray(d["true_xyz"], dtype=np.float64))
    return np.concatenate(preds), np.concatenate(trues)


# ── Driver ───────────────────────────────────────────────────────────────
MODELS = [
    {"key": "ssl_cd_s500_75k",   "source": "ssl",
     "label": "w2v2 MaskPred + ChanDistc σ=500 (75k)"},
    {"key": "ssl_cd_s300_75k",   "source": "ssl",
     "label": "w2v2 MaskPred + ChanDistc σ=300 (75k)"},
    {"key": "ssl_maskpred_75k",  "source": "ssl",
     "label": "w2v2 MaskPred only (75k)"},
    {"key": "ssl_chandistc_75k", "source": "ssl",
     "label": "w2v2 ChanDistc only (75k)"},
    {"key": "ssl_ccfdistc_75k",  "source": "ssl",
     "label": "w2v2 CCFDistc only (75k)"},
    {"key": "vit",          "source": "vsp",  "path": VIT_NPZ,
     "label": "ViT rival (ws3 IBL)"},
    {"key": "anynet",       "source": "vsp",  "path": ANY_NPZ,
     "label": "AnyNet rival (ws3 IBL)"},
    {"key": "histgbm_band", "source": "base", "label": "HistGBM / band"},
    {"key": "knn_band",     "source": "base", "label": "kNN-5 / band"},
    {"key": "knn_pca_psd",  "source": "base", "label": "kNN-5 / PCA(50) PSD"},
    {"key": "pca_ridge_psd","source": "base", "label": "PCA+Ridge / PSD"},
    {"key": "rf_band",      "source": "base", "label": "RF / band"},
    {"key": "svm_band",     "source": "base", "label": "SVM / band"},
    {"key": "ridge_band",   "source": "base", "label": "Ridge / band"},
    {"key": "ridge_psd",    "source": "base", "label": "Ridge / PSD"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--models", default="all",
                    help="comma-separated subset of " + ",".join(m["key"] for m in MODELS))
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("[*] loading SSL meta_str (authoritative pid/ch source) + true_xyz…",
          flush=True)
    meta_arr, true_ssl = load_ssl_meta_and_true()
    print(f"    {len(meta_arr):,} samples", flush=True)

    pids = np.empty(len(meta_arr), dtype=object)
    channels = np.full(len(meta_arr), -1, dtype=np.int32)
    n_unparsed = 0
    for i, m in enumerate(meta_arr):
        pid, ch = parse_pid_channel(m)
        if pid is None:
            n_unparsed += 1
            continue
        pids[i] = pid
        channels[i] = int(ch)
    print(f"    parse_pid_channel: {n_unparsed} unparseable", flush=True)
    n_unique_probes = len({p for p in pids if p is not None})
    print(f"    {n_unique_probes} unique probes", flush=True)

    selected = [m for m in MODELS
                if args.models == "all" or m["key"] in args.models.split(",")]

    summary = {}
    for m in selected:
        t0 = time.time()
        print(f"\n[*] {m['key']} ({m['label']})", flush=True)
        if m["source"] == "vsp":
            pred, true_m = load_vsp_raw(m["path"], key="gauss_raw")
        elif m["source"] == "ssl":
            pred, true_m = load_ssl_pred_raw(SSL_DIR_BY_KEY[m["key"]])
        else:
            pred, true_m = load_baseline_raw(m["key"])

        # Sanity: true_xyz must match SSL (rules out row-order drift)
        if not np.allclose(true_m, true_ssl, atol=1e-3):
            max_diff = float(np.abs(true_m - true_ssl).max())
            raise SystemExit(f"{m['key']}: true_xyz drift vs SSL "
                              f"(max abs diff = {max_diff:.3f})")

        rmse_raw     = rmse_3d(pred, true_ssl, fold=False)
        rmse_raw_lr  = rmse_3d(pred, true_ssl, fold=True)

        refined, n_ok, n_total, n_flipped = kabsch_per_probe_raw(
            pred, pids, channels)
        rmse_kab     = rmse_3d(refined, true_ssl, fold=False)
        rmse_kab_lr  = rmse_3d(refined, true_ssl, fold=True)

        out_npz = os.path.join(args.out_dir, f"{m['key']}_kabsch.npz")
        np.savez(out_npz,
                 preds_kabsch=refined.astype(np.float32),
                 true=true_ssl.astype(np.float32),
                 pids_str=np.array([str(p) if p is not None else ""
                                     for p in pids]),
                 channels=channels.astype(np.int32))
        print(f"    raw:  RMSE={rmse_raw:7.0f} µm  fold-LR={rmse_raw_lr:7.0f} µm",
              flush=True)
        print(f"    kab:  RMSE={rmse_kab:7.0f} µm  fold-LR={rmse_kab_lr:7.0f} µm",
              flush=True)
        print(f"    {n_ok}/{n_total} probes refined "
              f"({n_flipped} sign-flipped, {n_total - n_ok} skipped <4ch)",
              flush=True)
        print(f"    wrote {out_npz}  ({time.time()-t0:.1f}s)", flush=True)

        summary[m["key"]] = {
            "label": m["label"],
            "rmse_raw_um": rmse_raw,
            "rmse_raw_foldlr_um": rmse_raw_lr,
            "rmse_kabsch_um": rmse_kab,
            "rmse_kabsch_foldlr_um": rmse_kab_lr,
            "n_probes_fit": int(n_ok),
            "n_probes_total": int(n_total),
            "n_flipped": int(n_flipped),
            "out_npz": out_npz,
        }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] wrote {summary_path}", flush=True)

    print("\n=== Overall RMSE (3D, µm) ===", flush=True)
    print(f"{'model':<25} {'raw':>7} {'raw_lr':>7} {'kabsch':>7} {'kab_lr':>7}",
          flush=True)
    for k, v in summary.items():
        print(f"{k:<25} {v['rmse_raw_um']:7.0f} {v['rmse_raw_foldlr_um']:7.0f} "
              f"{v['rmse_kabsch_um']:7.0f} {v['rmse_kabsch_foldlr_um']:7.0f}",
              flush=True)


if __name__ == "__main__":
    main()
