#!/usr/bin/env python3
"""Post-hoc: precision(σ)-weighted CHANNEL pooling of the multiple same-channel
chunk predictions BEFORE per-probe Kabsch — the "gaussian averaging then Kabsch"
idea. Operates on the EXISTING probe_seq gaussian predictions (pred_xyz_gauss +
pred_sigma_gauss), no retraining.

Current kabsch_eval_1M_pathb.py groups chunks by (pid,channel) but fits Kabsch
over ALL chunks equally and IGNORES σ. Here we instead:
  1. collapse the n chunks of each channel to ONE point via inverse-variance mean
       mu_ch[a] = Σ(mu_i[a]/σ_i[a]²) / Σ(1/σ_i[a]²)      (per axis a)
       w_ch[a]  = Σ(1/σ_i[a]²)                            (aggregate precision)
  2. per-probe Kabsch on the per-channel points (optionally precision-weighted),
  3. fold-LR RMSE per channel.

Variants printed:
  baseline_perchunk  : reproduce current unweighted per-chunk Kabsch (sanity)
  mean_pool          : plain (unweighted) chunk→channel mean, unweighted Kabsch
  sigma_pool         : σ-weighted chunk→channel mean, unweighted Kabsch
  sigma_pool_wkabsch : σ-weighted chunk→channel mean, precision-weighted Kabsch

Usage: PRED_DIR=<...>/predictions python3 kabsch_gauss_chanpool_eval.py
"""
import os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PL = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/Lfp2vec_benchmarks/scripts')
if PL not in sys.path:
    sys.path.insert(0, PL)
from recompute_refined_3d_metrics import (  # noqa
    parse_pid_channel, channel_xy_onprobe, ALLEN_MIDLINE_UM)

PRED_DIR = os.environ.get("PRED_DIR")


def fold_lr(xyz):
    """Fold the ML axis about the midline.

    ⛔ COLUMN ORDER IS (AP, DV, ML) — the ML axis is column 2, NOT column 0.
    Proof from the prediction npz files: true_xyz col0 reaches 11924 µm, which
    exceeds the CCF ML extent (11400 µm), so col0 cannot be ML; col1 tops out
    near the DV extent (8000 µm) and col2 at 8536 µm < 11400 µm.
    This folded column 0 (AP) until 2026-07-24, which left ML unfolded and
    mangled AP; it inflated the probe-sequence transformer's RAW RMSE by
    ~240 µm at 75k while barely moving the per-channel decoders. Every other
    fold implementation in this directory (recompute_refined_3d_metrics.
    folded_3d_metrics, scoreboard_gavg_kabsch_whit_a1p0.fold_lh,
    score_decoder_capacity_gavg_kabsch.fold) already folds column 2.
    """
    out = np.asarray(xyz, dtype=np.float64).copy()
    m = out[:, 2] > ALLEN_MIDLINE_UM
    out[m, 2] = 2 * ALLEN_MIDLINE_UM - out[m, 2]
    return out


def rmse_3d(pred, true, fold=False):
    if fold:
        pred, true = fold_lr(pred), fold_lr(true)
    return float(np.sqrt(((pred - true) ** 2).sum(axis=1).mean()))


def kabsch_weighted(P, Q, w=None):
    """Rigid R,t minimizing Σ w_i ||R p_i + t - q_i||²; w broadcast per point."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    if w is None:
        w = np.ones(len(P))
    w = np.asarray(w, float); W = w.sum()
    pbar = (w[:, None] * P).sum(0) / W
    qbar = (w[:, None] * Q).sum(0) / W
    Pc, Qc = P - pbar, Q - qbar
    H = (w[:, None] * Pc).T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = qbar - R @ pbar
    return R, t


def load_all():
    preds_mu, preds_sig, trues, pids, chans = [], [], [], [], []
    for f in range(5):
        fp = os.path.join(PRED_DIR, f"test_fold_{f}_predictions.npz")
        if not os.path.exists(fp):
            continue
        d = np.load(fp, allow_pickle=True)
        mu = d["pred_xyz_gauss"].astype(np.float64)
        sig = d["pred_sigma_gauss"].astype(np.float64)
        tr = d["true_xyz"].astype(np.float64)
        meta = d["meta_str"].astype(str)
        for i in range(len(mu)):
            pid, ch = parse_pid_channel(meta[i])
            pids.append(pid); chans.append(-1 if ch is None else int(ch))
        preds_mu.append(mu); preds_sig.append(sig); trues.append(tr)
    return (np.vstack(preds_mu), np.vstack(preds_sig), np.vstack(trues),
            np.array(pids, dtype=object), np.array(chans, dtype=np.int64))


def pool_channels(mu, sig, true, pids, chans, mode):
    """Return per-channel arrays: geom P, pooled Q, weight w, true, pid."""
    key = defaultdict(list)
    for i in range(len(mu)):
        if pids[i] is None or chans[i] < 0:
            continue
        key[(pids[i], chans[i])].append(i)
    P, Q, W, T, PID = [], [], [], [], []
    for (pid, ch), idxs in key.items():
        idxs = np.asarray(idxs)
        xcol, depth = channel_xy_onprobe(int(ch))
        m = mu[idxs]                      # (n,3)
        if mode == "mean":
            q = m.mean(0); w = float(len(idxs))
        else:  # sigma inverse-variance
            prec = 1.0 / np.clip(sig[idxs] ** 2, 1e-6, None)   # (n,3)
            q = (m * prec).sum(0) / prec.sum(0)
            w = float(prec.sum(0).mean())                       # scalar precision
        P.append([xcol, 0.0, depth]); Q.append(q); W.append(w)
        T.append(true[idxs].mean(0)); PID.append(pid)
    return (np.array(P), np.array(Q), np.array(W), np.array(T),
            np.array(PID, dtype=object))


def per_probe_kabsch(P, Q, W, PID, use_w):
    Rx180 = np.diag([1.0, -1.0, -1.0])
    refined = Q.copy()
    groups = defaultdict(list)
    for i, p in enumerate(PID):
        groups[p].append(i)
    n_ok = n_flip = 0
    for p, idxs in groups.items():
        idxs = np.asarray(idxs)
        if len(idxs) < 4:
            continue
        w = W[idxs] if use_w else None
        try:
            R, t = kabsch_weighted(P[idxs], Q[idxs], w)
        except Exception:
            continue
        if (R @ np.array([0.0, 0.0, 1.0]))[1] > 0:
            R = R @ Rx180
            ww = np.ones(len(idxs)) if w is None else w
            t = (ww[:, None] * Q[idxs]).sum(0) / ww.sum() - R @ (
                (ww[:, None] * P[idxs]).sum(0) / ww.sum())
            n_flip += 1
        refined[idxs] = (R @ P[idxs].T).T + t
        n_ok += 1
    return refined, n_ok, n_flip


def report(name, P, Q, W, T, PID, use_w):
    raw_fold = rmse_3d(Q, T, fold=True)
    refined, n_ok, n_flip = per_probe_kabsch(P, Q, W, PID, use_w)
    kab = rmse_3d(refined, T, fold=False)
    kab_fold = rmse_3d(refined, T, fold=True)
    print(f"  {name:22s} raw-fold={raw_fold:7.1f}  Kabsch={kab:7.1f}  "
          f"Kabsch-foldLR={kab_fold:7.1f}  [{n_ok} probes, {n_flip} flip]")


def main():
    assert PRED_DIR, "set PRED_DIR"
    mu, sig, true, pids, chans = load_all()
    nchan = len({(p, c) for p, c in zip(pids, chans) if p is not None and c >= 0})
    print(f"PRED_DIR={PRED_DIR}")
    print(f"  {len(mu)} chunk-rows, {nchan} unique (probe,channel), "
          f"σ range [{sig.min():.0f},{sig.max():.0f}]µm  "
          f"(RMSE below = PER-CHANNEL, equal weight)\n")

    # sanity: reproduce current per-chunk unweighted Kabsch (pred_xyz_gauss)
    Pc, Qc, Wc, Tc, PIDc = [], [], [], [], []
    key = defaultdict(list)
    for i in range(len(mu)):
        if pids[i] is not None and chans[i] >= 0:
            key[(pids[i], chans[i])].append(i)
    for (pid, ch), idxs in key.items():
        xcol, depth = channel_xy_onprobe(int(ch))
        for j in idxs:
            Pc.append([xcol, 0.0, depth]); Qc.append(mu[j]); Wc.append(1.0)
            Tc.append(true[j]); PIDc.append(pid)
    report("baseline_perchunk", np.array(Pc), np.array(Qc), np.array(Wc),
           np.array(Tc), np.array(PIDc, dtype=object), use_w=False)

    for mode, label in [("mean", "mean_pool"), ("sigma", "sigma_pool")]:
        P, Q, W, T, PID = pool_channels(mu, sig, true, pids, chans, mode)
        report(label, P, Q, W, T, PID, use_w=False)
    P, Q, W, T, PID = pool_channels(mu, sig, true, pids, chans, "sigma")
    report("sigma_pool_wkabsch", P, Q, W, T, PID, use_w=True)
    print("\n[ref] per-chunk unweighted (current eval) s42 = ~1340µm foldLR; "
          "gaussian record mean = 1396µm")


if __name__ == "__main__":
    main()
