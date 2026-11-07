#!/usr/bin/env python3
"""Post-hoc refinement of per-sample xyz predictions, then re-compute 3D RMSE.

Stages (applied cumulatively):
    (a) raw                    — the saved per-sample predictions as-is
    (b) channel_consensus      — unweighted mean of predictions per
                                 (probe_id, channel_idx) across time windows
    (c) weighted_consensus     — inverse-variance (1 / σ²) weighted mean using
                                 pred_sigma_gauss from the Gaussian probe
    (d) depth_regression       — after (c), per-probe least-squares fit of
                                 xyz = slope · channel_depth + intercept,
                                 then place every channel at intercept +
                                 slope · its known depth. Channel depth comes
                                 from NP1.0 geometry: depth = (channel_idx //
                                 2) * 20 µm.

3D RMSE is folded (LR axis → |lr − midline|) for all stages, so numbers are
directly comparable to the `rmse_3d` already in each model's regression.json.

Usage:
    python3 recompute_refined_3d_metrics.py --models_root <path>
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict

ALLEN_MIDLINE_UM = 5739.0


def parse_pid_channel(meta_str):
    """IBL format: eid_idx_probeNN_pid-chNNNN_region.
    Returns (probe_key, channel_idx) or (None, None) if unparseable.
    """
    parts = str(meta_str).split("_")
    if len(parts) < 4 or "-ch" not in parts[3]:
        return None, None
    try:
        ch = int(parts[3].split("-ch")[1])
    except (ValueError, IndexError):
        return None, None
    return f"{parts[0]}_{parts[2]}", ch


def channel_depth(ch, row_pitch_um=20.0, channels_per_row=2):
    """NP1.0 channel index → depth (µm) along the probe shaft."""
    return (ch // channels_per_row) * row_pitch_um


def channel_xy_onprobe(ch):
    """NP1.0 on-probe (x, depth) µm. y=0 (all channels on one face of shank).
    4-column checkerboard: cols at x ∈ {11, 27, 43, 59}. Even rows use cols
    {0, 2} = x ∈ {11, 43}; odd rows use cols {1, 3} = x ∈ {27, 59}."""
    row = ch // 2
    depth = row * 20.0
    if row % 2 == 0:
        x = 11.0 if (ch % 2) == 0 else 43.0
    else:
        x = 27.0 if (ch % 2) == 0 else 59.0
    return x, depth


def kabsch_rigid(P, Q, weights=None):
    """Weighted Kabsch: find R (3x3 rotation, det=+1) and t (3,) minimizing
    sum_i w_i ||R @ P_i + t - Q_i||². P, Q: (N, 3). Returns (R, t)."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if weights is None:
        w = np.ones(len(P), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    p_mean = (w[:, None] * P).sum(0)
    q_mean = (w[:, None] * Q).sum(0)
    P0 = P - p_mean
    Q0 = Q - q_mean
    H = (P0 * w[:, None]).T @ Q0
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    t = q_mean - R @ p_mean
    return R, t


def channel_consensus(pred_xyz, pids, channels, sigmas=None):
    """Per (pid, channel) mean across time windows. If `sigmas` provided, uses
    inverse-variance weighted mean; else unweighted mean."""
    key_to_idx = defaultdict(list)
    for i, (p, c) in enumerate(zip(pids, channels)):
        if p is None or c is None:
            continue
        key_to_idx[(p, c)].append(i)

    refined = pred_xyz.copy()
    for _, idxs in key_to_idx.items():
        idxs_arr = np.array(idxs, dtype=np.int64)
        preds = pred_xyz[idxs_arr]
        if sigmas is None:
            mean = preds.mean(axis=0)
        else:
            s = sigmas[idxs_arr]
            w = 1.0 / (s ** 2 + 1e-6)
            w_norm = w / w.sum(axis=0, keepdims=True)
            mean = (preds * w_norm).sum(axis=0)
        for i in idxs_arr:
            refined[i] = mean
    return refined


def depth_regression(pred_xyz, pids, channels):
    """Per-probe least-squares: xyz = slope * depth + intercept. Place each
    channel at intercept + slope * known_depth. Assumes `pred_xyz` has already
    been through channel consensus (so all samples of a given channel share
    one xyz)."""
    key_to_idx = defaultdict(list)
    for i, (p, c) in enumerate(zip(pids, channels)):
        if p is None or c is None:
            continue
        key_to_idx[(p, c)].append(i)

    probe_to_entries = defaultdict(list)
    for (pid, ch), idxs in key_to_idx.items():
        cent = pred_xyz[idxs[0]]  # all same after consensus
        d = channel_depth(ch)
        probe_to_entries[pid].append((ch, cent, d, np.array(idxs, dtype=np.int64)))

    refined = pred_xyz.copy()
    n_ok, n_skipped = 0, 0
    for pid, entries in probe_to_entries.items():
        if len(entries) < 3:
            n_skipped += 1
            continue
        depths = np.array([e[2] for e in entries], dtype=np.float64)
        cents  = np.array([e[1] for e in entries], dtype=np.float64)
        if depths.max() - depths.min() < 1.0:
            n_skipped += 1
            continue
        A = np.column_stack([depths, np.ones(len(depths))])
        params, _, _, _ = np.linalg.lstsq(A, cents, rcond=None)
        slope, intercept = params[0], params[1]  # each (3,)
        for _, _, d, idxs_arr in entries:
            new_xyz = intercept + slope * d
            for i in idxs_arr:
                refined[i] = new_xyz
        n_ok += 1
    return refined, (n_ok, n_skipped)


def rigid_probe_fit(pred_xyz, pids, channels, sigmas=None, use_weights=False):
    """Per-probe Kabsch rigid fit: R·(x_col, 0, depth) + t ≈ predicted xyz.
    Preserves NP1.0 4-column planar shape (3840 × 48 µm) in CCF. Assumes
    channel consensus already applied (all samples of same channel share xyz).
    If sigmas + use_weights: weight each channel by 1/mean(||σ||)².

    Anatomical sign correction: IBL probes are inserted skull→deep-brain, so
    ch=0 (probe tip, depth=0) is always at HIGHER DV than ch=383 (base,
    depth=3820). Equivalently, R · (0,0,1) — the rotated depth axis —
    must have NEGATIVE DV component. If unconstrained SVD picks an
    anatomically inverted R (positive DV component, common when per-channel
    predictions are noisy), apply a 180° rotation around the on-probe AP
    axis (diag(1,-1,-1), det=+1) to restore anatomy, then re-fit t so the
    rigid body stays centered on the predicted cloud's mean. This trades
    per-probe residual-to-prediction for anatomy correctness; aggregate
    RMSE drops slightly for clean models (CD σ=500) and rises for noisy
    ones (Pure SSL) — a more honest metric.
    """
    Rx180 = np.diag([1.0, -1.0, -1.0])  # 180° about on-probe AP axis
    key_to_idx = defaultdict(list)
    for i, (p, c) in enumerate(zip(pids, channels)):
        if p is None or c is None:
            continue
        key_to_idx[(p, c)].append(i)

    probe_entries = defaultdict(list)
    for (pid, ch), idxs in key_to_idx.items():
        cent = pred_xyz[idxs[0]]  # all same after consensus
        x_op, d_op = channel_xy_onprobe(ch)
        w = None
        if sigmas is not None and use_weights:
            s = sigmas[np.array(idxs, dtype=np.int64)]  # (n_time, 3)
            s_mean = float(np.linalg.norm(s, axis=1).mean())
            w = 1.0 / (s_mean ** 2 + 1e-6)
        probe_entries[pid].append(
            (ch, cent, (x_op, 0.0, d_op), np.array(idxs, dtype=np.int64), w))

    refined = pred_xyz.copy()
    n_ok, n_skipped = 0, 0
    for pid, entries in probe_entries.items():
        if len(entries) < 3:
            n_skipped += 1
            continue
        on_probe = np.array([e[2] for e in entries], dtype=np.float64)
        cents    = np.array([e[1] for e in entries], dtype=np.float64)
        if on_probe[:, 2].max() - on_probe[:, 2].min() < 1.0:
            n_skipped += 1
            continue
        weights = None
        if sigmas is not None and use_weights:
            weights = np.array([e[4] for e in entries], dtype=np.float64)
        R, t = kabsch_rigid(on_probe, cents, weights=weights)
        # Anatomical sign correction (see docstring above).
        if (R @ np.array([0.0, 0.0, 1.0]))[1] > 0:
            R = R @ Rx180
            t = cents.mean(axis=0) - R @ on_probe.mean(axis=0)
        for _, _, p_op, idxs_arr, _ in entries:
            new_xyz = R @ np.asarray(p_op, dtype=np.float64) + t
            for i in idxs_arr:
                refined[i] = new_xyz
        n_ok += 1
    return refined, (n_ok, n_skipped)


def folded_3d_metrics(pred_xyz, true_xyz, midline=ALLEN_MIDLINE_UM):
    """3D RMSE + median Euclidean, with LR axis folded to |lr - midline|."""
    err_ap = (pred_xyz[:, 0] - true_xyz[:, 0]).astype(np.float64)
    err_dv = (pred_xyz[:, 1] - true_xyz[:, 1]).astype(np.float64)
    true_d = np.abs(true_xyz[:, 2] - midline)
    pred_d = np.abs(pred_xyz[:, 2] - midline)
    err_lr = (pred_d - true_d).astype(np.float64)
    sq = err_ap ** 2 + err_dv ** 2 + err_lr ** 2
    return {
        "rmse_3d":             float(np.sqrt(sq.mean())),
        "median_euclidean_3d": float(np.median(np.sqrt(sq))),
        "rmse_per_axis": [
            float(np.sqrt((err_ap ** 2).mean())),
            float(np.sqrt((err_dv ** 2).mean())),
            float(np.sqrt((err_lr ** 2).mean())),
        ],
    }


def process_model(model_dir, probe_key="pred_xyz"):
    """Load all test-fold predictions, apply 4 refinement stages, return dict
    of stage -> metrics. probe_key selects which probe's predictions to refine
    (pred_xyz = MSE probe, pred_xyz_gauss = Gaussian μ)."""
    pred_dir = os.path.join(model_dir, "predictions")
    if not os.path.isdir(pred_dir):
        return None

    folds = []
    for f in range(5):
        p = os.path.join(pred_dir, f"test_fold_{f}_predictions.npz")
        if os.path.exists(p):
            folds.append(np.load(p))
    if not folds:
        return None

    pred = np.concatenate([d[probe_key] for d in folds], axis=0)
    true = np.concatenate([d["true_xyz"] for d in folds], axis=0)
    meta = np.concatenate([d["meta_str"] for d in folds], axis=0)
    sigmas = None
    if "pred_sigma_gauss" in folds[0].files:
        sigmas = np.concatenate([d["pred_sigma_gauss"] for d in folds], axis=0)

    pids, channels = [], []
    n_missing = 0
    for m in meta:
        p, c = parse_pid_channel(m)
        pids.append(p)
        channels.append(c)
        if p is None or c is None:
            n_missing += 1

    results = {
        "n_samples":  int(len(pred)),
        "n_missing_meta": int(n_missing),
        "probe_source":   probe_key,
    }

    # (a) raw
    results["raw"] = folded_3d_metrics(pred, true)

    # (b) channel consensus (unweighted)
    pred_b = channel_consensus(pred, pids, channels, sigmas=None)
    results["channel_consensus"] = folded_3d_metrics(pred_b, true)

    # (c) weighted consensus (only if Gaussian probe saved σ)
    if sigmas is not None and probe_key == "pred_xyz_gauss":
        pred_c = channel_consensus(pred, pids, channels, sigmas=sigmas)
        results["weighted_consensus"] = folded_3d_metrics(pred_c, true)
        base_for_dr = pred_c
    else:
        results["weighted_consensus"] = None  # only meaningful for Gaussian probe
        base_for_dr = pred_b

    # (d) depth regression (on the tightest consensus we have)
    pred_d, (n_ok, n_skipped) = depth_regression(base_for_dr, pids, channels)
    results["depth_regression"] = folded_3d_metrics(pred_d, true)
    results["depth_regression"]["n_probes_ok"] = n_ok
    results["depth_regression"]["n_probes_skipped"] = n_skipped

    # (e) rigid probe fit (Kabsch) — preserves NP1.0 4-column planar shape
    pred_e, (ne_ok, ne_skipped) = rigid_probe_fit(base_for_dr, pids, channels)
    results["rigid_probe_fit"] = folded_3d_metrics(pred_e, true)
    results["rigid_probe_fit"]["n_probes_ok"] = ne_ok
    results["rigid_probe_fit"]["n_probes_skipped"] = ne_skipped

    # (f) weighted rigid probe fit (1/σ² per channel, Gaussian probe only)
    if sigmas is not None and probe_key == "pred_xyz_gauss":
        pred_f, (nf_ok, nf_skipped) = rigid_probe_fit(
            base_for_dr, pids, channels, sigmas=sigmas, use_weights=True)
        results["weighted_rigid_probe_fit"] = folded_3d_metrics(pred_f, true)
        results["weighted_rigid_probe_fit"]["n_probes_ok"] = nf_ok
        results["weighted_rigid_probe_fit"]["n_probes_skipped"] = nf_skipped
    else:
        results["weighted_rigid_probe_fit"] = None
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_root",
        default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/eval_v2/models'))
    parser.add_argument("--probe_key", default="pred_xyz",
        choices=["pred_xyz", "pred_xyz_gauss", "pred_xyz_hemi"])
    args = parser.parse_args()

    import glob
    dirs = sorted(glob.glob(os.path.join(args.models_root, "*")))
    for d in dirs:
        if not os.path.isdir(d):
            continue
        if not os.path.isdir(os.path.join(d, "predictions")):
            continue
        name = os.path.basename(d)
        print(f"\n=== {name} ===")
        # Always include MSE probe. If Gaussian is present, run both.
        summary = {}
        for pk in ["pred_xyz", "pred_xyz_gauss"]:
            res = process_model(d, probe_key=pk)
            if res is None:
                continue
            print(f"  [{pk}]")
            for stage in ["raw", "channel_consensus", "weighted_consensus",
                          "depth_regression", "rigid_probe_fit",
                          "weighted_rigid_probe_fit"]:
                s = res.get(stage)
                if s is None:
                    continue
                print(f"    {stage:<22s} RMSE={s['rmse_3d']:.0f}µm  "
                      f"medEucl={s['median_euclidean_3d']:.0f}µm  "
                      f"per-axis [ap={s['rmse_per_axis'][0]:.0f} "
                      f"dv={s['rmse_per_axis'][1]:.0f} "
                      f"d={s['rmse_per_axis'][2]:.0f}]")
            summary[pk] = res

        if summary:
            out = os.path.join(d, "regression_refined.json")
            with open(out, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  wrote {out}")


if __name__ == "__main__":
    main()
