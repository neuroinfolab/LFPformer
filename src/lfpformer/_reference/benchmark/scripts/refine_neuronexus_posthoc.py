#!/usr/bin/env python3
"""Post-hoc XYZ refinement for NeuroNexus zero-shot predictions.

Inputs (per variant): predictions.npz with
    pred_xyz (N,3), session (str), shank (int8), channel_id (int32),
    chunk_idx (int32), probe_x (f32), probe_y (f32), region_label (int8)

Refinements (no ground-truth CCF used):
  1. consensus  — per (session, channel_id), mean pred_xyz across time chunks
  2. depth_regr — per (session, shank), OLS each XYZ axis on probe_y using the
                  consensus centroids; place each channel at intercept + slope*probe_y

Output: predictions_refined.npz alongside predictions.npz, with keys
    pred_xyz_raw, pred_xyz_consensus, pred_xyz_depth_regr,
    plus all original metadata copied through.
"""

import argparse
import time
from pathlib import Path

import numpy as np


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def channel_consensus(pred, session, channel_id):
    """Uniform mean of pred_xyz per (session, channel_id), broadcast to rows."""
    keys = np.array([f"{s}|{c}" for s, c in zip(session, channel_id)])
    uniq, inv = np.unique(keys, return_inverse=True)
    sums = np.zeros((len(uniq), 3), dtype=np.float64)
    counts = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(sums, inv, pred)
    np.add.at(counts, inv, 1)
    centroids = (sums / counts[:, None]).astype(np.float32)
    return centroids[inv], centroids, uniq, inv


def channel_consensus_weighted(pred, sigma, session, channel_id, eps=1.0):
    """σ²-weighted mean of pred_xyz per (session, channel_id) — broadcast to rows.

        μ_chan_axis = Σ μᵢ * wᵢ / Σ wᵢ      with  wᵢ = 1 / (σᵢ² + eps²)

    Each axis is weighted independently (σ is (N,3)). eps stabilises wᵢ
    when σ → 0 (rare; ft_gauss clamps log_var so σ has a floor).
    """
    keys = np.array([f"{s}|{c}" for s, c in zip(session, channel_id)])
    uniq, inv = np.unique(keys, return_inverse=True)
    var = sigma.astype(np.float64) ** 2 + eps ** 2     # (N, 3)
    w = 1.0 / var
    wsum = np.zeros((len(uniq), 3), dtype=np.float64)
    wxsum = np.zeros((len(uniq), 3), dtype=np.float64)
    np.add.at(wsum, inv, w)
    np.add.at(wxsum, inv, pred * w)
    centroids = (wxsum / np.clip(wsum, 1e-12, None)).astype(np.float32)
    return centroids[inv], centroids, uniq, inv


def rigid_transform(centroids_per_row, session, probe_x, probe_y, channel_id,
                    weights=None):
    """Per-session rigid (R, t) alignment of probe-frame (probe_x, probe_y, 0)
    to CCF predicted XYZ.  Treats the 8 shanks of NN as a single rigid body —
    output is 8 parallel lines with correct inter-shank spacing.

    Closed-form via Kabsch / Procrustes:
        Center both point sets, compute H = P_probe^T · W · P_pred,
        SVD H = U Σ V^T, R = V · diag(1, 1, sign(det(VU^T))) · U^T.

    `weights` (optional, shape (N,)) gates per-sample influence on the fit;
    pass `None` for uniform weighting.
    """
    refined = centroids_per_row.copy()
    sess_arr = np.asarray(session)
    px_arr = np.asarray(probe_x, dtype=np.float64)
    py_arr = np.asarray(probe_y, dtype=np.float64)
    ch_arr = np.asarray(channel_id)
    if weights is None:
        w_arr = np.ones(len(sess_arr), dtype=np.float64)
    else:
        w_arr = np.asarray(weights, dtype=np.float64)

    n_success = 0
    n_fallback = 0
    for sess in np.unique(sess_arr):
        smask = sess_arr == sess
        idx_in_sess = np.where(smask)[0]

        # One representative row per channel for the fit
        ch_in_sess = ch_arr[idx_in_sess]
        _, first_pos = np.unique(ch_in_sess, return_index=True)
        rep_rows = idx_in_sess[first_pos]
        if len(rep_rows) < 3:
            n_fallback += 1
            continue

        P_probe = np.column_stack([
            px_arr[rep_rows], py_arr[rep_rows],
            np.zeros(len(rep_rows), dtype=np.float64),
        ])                                              # (n, 3) probe frame
        P_pred = centroids_per_row[rep_rows].astype(np.float64)
        # Per-channel weight = sum of sample weights for that channel across
        # all its time chunks (not just rep_rows[0]). Approximates the
        # information-content of that channel's consensus estimate.
        ch_to_w = {}
        for j in idx_in_sess:
            cid = int(ch_arr[j])
            ch_to_w[cid] = ch_to_w.get(cid, 0.0) + float(w_arr[j])
        w = np.array([ch_to_w.get(int(ch_arr[r]), 0.0) for r in rep_rows],
                     dtype=np.float64)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones_like(w)

        wsum = w.sum()
        c_probe = (P_probe * w[:, None]).sum(axis=0) / wsum
        c_pred = (P_pred * w[:, None]).sum(axis=0) / wsum
        Q_probe = P_probe - c_probe
        Q_pred = P_pred - c_pred
        H = (Q_probe * w[:, None]).T @ Q_pred           # (3, 3)
        try:
            U, _, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            n_fallback += 1
            continue
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
        R = Vt.T @ D @ U.T
        t = c_pred - R @ c_probe

        # Apply R, t to every row in the session (uses each row's own probe_x,y,0)
        all_probe = np.column_stack([
            px_arr[idx_in_sess], py_arr[idx_in_sess],
            np.zeros(len(idx_in_sess), dtype=np.float64),
        ])
        refined[idx_in_sess] = (all_probe @ R.T + t).astype(np.float32)
        n_success += 1

    log(f"    rigid: {n_success} sessions fit, {n_fallback} fallback")
    return refined


def depth_regression(centroids_per_row, session, shank, probe_y, channel_id):
    """For each (session, shank), OLS each XYZ axis on probe_y using the
    unique-channel centroids; place each row at intercept + slope*probe_y."""
    refined = centroids_per_row.copy()
    n_success = 0
    n_fallback = 0

    # We need one (centroid, probe_y) per channel within each (session, shank).
    # Use channel_id as the unique within-session key.
    sess_arr = np.asarray(session)
    shank_arr = np.asarray(shank)
    ch_arr = np.asarray(channel_id)
    py_arr = np.asarray(probe_y, dtype=np.float64)

    group_keys = np.array([f"{s}|{int(sh)}" for s, sh in zip(sess_arr, shank_arr)])
    for gk in np.unique(group_keys):
        gmask = group_keys == gk
        idx_in_group = np.where(gmask)[0]

        # Pick one row per channel for the regression fit (centroid is the same
        # for all rows of a given channel after consensus, so first occurrence is fine)
        ch_in_group = ch_arr[idx_in_group]
        _, first_pos = np.unique(ch_in_group, return_index=True)
        rep_rows = idx_in_group[first_pos]

        cents = centroids_per_row[rep_rows].astype(np.float64)
        depths = py_arr[rep_rows]

        depth_range = depths.max() - depths.min()
        if len(rep_rows) < 3 or depth_range < 1e-3:
            # Leave consensus values in place
            n_fallback += 1
            continue

        A = np.column_stack([depths, np.ones(len(depths))])
        params, *_ = np.linalg.lstsq(A, cents, rcond=None)
        slope = params[0]      # (3,)
        intercept = params[1]  # (3,)

        for r in idx_in_group:
            refined[r] = (intercept + slope * py_arr[r]).astype(np.float32)
        n_success += 1

    log(f"    depth_regr: {n_success} (session,shank) groups fit, "
        f"{n_fallback} fallback")
    return refined


def fold_to_left(xyz, midline_um=5700.0):
    """Reflect right-hemisphere predictions to left hemisphere.

    ML (axis 2) values > midline are mapped to 2·midline − ML so all
    predictions land in the left half of Allen CCF (ML ∈ [0, midline]).
    AP and DV are untouched.
    """
    out = xyz.astype(np.float32, copy=True)
    right = out[:, 2] > midline_um
    out[right, 2] = 2.0 * midline_um - out[right, 2]
    return out, int(right.sum())


def refine_one(npz_path, out_path, symmetrize_left=False):
    log(f"Loading {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    pred = z["pred_xyz"]
    sigma = z["pred_sigma"] if "pred_sigma" in z.files else None
    if symmetrize_left:
        pred_folded, n_right = fold_to_left(pred)
        log(f"  symmetrize-left: folded {n_right:,}/{len(pred):,} rows "
            f"(ML > 5700 µm → 2·midline − ML)")
        pred = pred_folded
    session = z["session"]
    shank = z["shank"]
    channel_id = z["channel_id"]
    probe_x = z["probe_x"]
    probe_y = z["probe_y"]
    log(f"  N={len(pred):,}  unique_sessions={len(set(session.tolist()))}  "
        f"unique_shanks={len(set(shank.tolist()))}  "
        f"sigma_present={sigma is not None}")
    if sigma is not None:
        log(f"  sigma stats (µm): median={np.median(sigma, axis=0).tolist()}  "
            f"min={sigma.min(axis=0).tolist()}  max={sigma.max(axis=0).tolist()}")

    log("  consensus (uniform) per (session, channel_id) ...")
    consensus_per_row, _, uniq, _ = channel_consensus(pred, session, channel_id)
    log(f"    {len(uniq):,} unique channels across all sessions")

    log("  depth regression (uniform) per (session, shank) ...")
    depth_per_row = depth_regression(
        consensus_per_row, session, shank, probe_y, channel_id)

    log("  rigid (8-shank Procrustes) per session ...")
    rigid_per_row = rigid_transform(
        consensus_per_row, session, probe_x, probe_y, channel_id)

    payload = {k: z[k] for k in z.files if k != "pred_xyz"}
    payload["pred_xyz_raw"] = pred.astype(np.float32)
    payload["pred_xyz_consensus"] = consensus_per_row.astype(np.float32)
    payload["pred_xyz_depth_regr"] = depth_per_row.astype(np.float32)
    payload["pred_xyz_rigid"] = rigid_per_row.astype(np.float32)

    if sigma is not None:
        log("  σ-weighted consensus per (session, channel_id) ...")
        weighted_per_row, _, _, _ = channel_consensus_weighted(
            pred, sigma, session, channel_id)
        log("  σ-weighted consensus → depth regression ...")
        weighted_depth_per_row = depth_regression(
            weighted_per_row, session, shank, probe_y, channel_id)
        log("  σ-weighted rigid (per-sample 1/σ² weights, mean over axes) ...")
        # Per-sample scalar weight = mean(1/σ²) across the 3 axes
        sigma_var = sigma.astype(np.float64) ** 2 + 1.0   # eps=1µm² floor
        sample_w = (1.0 / sigma_var).mean(axis=1)
        weighted_rigid_per_row = rigid_transform(
            weighted_per_row, session, probe_x, probe_y, channel_id,
            weights=sample_w)
        payload["pred_xyz_weighted"] = weighted_per_row.astype(np.float32)
        payload["pred_xyz_weighted_depth_regr"] = (
            weighted_depth_per_row.astype(np.float32))
        payload["pred_xyz_weighted_rigid"] = weighted_rigid_per_row.astype(np.float32)

    log(f"Saving {out_path}")
    np.savez_compressed(out_path, **payload)
    log(f"  size={out_path.stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-dir", required=True,
                    help="Dir containing predictions.npz")
    ap.add_argument("--output-name", default=None,
                    help="Output filename (default: predictions_refined.npz, "
                         "or predictions_refined_lh.npz when --symmetrize-left)")
    ap.add_argument("--symmetrize-left", action="store_true",
                    help="Fold ML > 5700 µm to left hemisphere before refining")
    args = ap.parse_args()

    vdir = Path(args.variant_dir)
    npz = vdir / "predictions.npz"
    if not npz.exists():
        raise SystemExit(f"Missing {npz}")

    if args.output_name is None:
        out_name = ("predictions_refined_lh.npz" if args.symmetrize_left
                    else "predictions_refined.npz")
    else:
        out_name = args.output_name
    out = vdir / out_name
    refine_one(npz, out, symmetrize_left=args.symmetrize_left)


if __name__ == "__main__":
    main()
