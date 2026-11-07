"""MAIN evaluation protocol (NeurIPS 2026 camera-ready), shared by Fig 3 / Fig 4.

For every model:
  1. per (probe, channel): Gaussian inverse-variance (1/sigma^2, per axis) weighted
     mean of that channel's per-sample predictions (plain mean if the model has no
     sigma, i.e. equal weights);
  2. per probe: rigid Kabsch fit of the on-probe channel geometry (x, 0, depth) to
     the pooled channel positions (unweighted across channels, no GT in the fit),
     with the same anatomical sign rule as kabsch_gauss_chanpool_eval.per_probe_kabsch.

Same maths as kabsch_gauss_chanpool_eval.pool_channels + per_probe_kabsch (it calls
per_probe_kabsch directly), but vectorised and geometry-agnostic, so it also serves
Allen probes (geometry from data/channels.csv) and returns per-sample positions for
the region-lookup pipelines.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kabsch_gauss_chanpool_eval as K  # noqa: E402


def pool_kabsch(mu, sig, pk, ch, xy):
    """mu (N,3) per-sample predictions; sig (N,3) sigma or None (-> plain mean);
    pk (N,) probe key; ch (N,) channel key; xy (N,2) on-probe (x, depth) in µm.

    Returns dict(sample=(N,3) Kabsch position broadcast to every sample of its
    channel, pooled=(G,3), kabsch=(G,3), group=(N,) channel index of each sample,
    first=(G,) first sample of each channel, n_ok=probes fitted)."""
    mu = np.asarray(mu, np.float64)
    keys = np.char.add(np.char.add(np.asarray(pk).astype(str), "|"), np.asarray(ch).astype(str))
    _, first, group = np.unique(keys, return_index=True, return_inverse=True)
    G = len(first)
    if sig is None:
        w = np.ones_like(mu)
    else:
        w = 1.0 / np.clip(np.asarray(sig, np.float64) ** 2, 1e-6, None)
    num = np.stack([np.bincount(group, mu[:, a] * w[:, a], G) for a in range(3)], 1)
    den = np.stack([np.bincount(group, w[:, a], G) for a in range(3)], 1)
    Q = num / den
    W = den.mean(1)
    gxy = np.asarray(xy, np.float64)[first]
    P = np.stack([gxy[:, 0], np.zeros(G), gxy[:, 1]], 1)
    PID = np.asarray(pk)[first].astype(object)
    ref, n_ok, _ = K.per_probe_kabsch(P, Q, W, PID, use_w=False)
    return dict(sample=ref[group], pooled=Q, kabsch=ref, group=group, first=first, n_ok=n_ok)


def ibl_xy(ch):
    """NP1.0 on-probe geometry for IBL local channel indices (vectorised
    K.channel_xy_onprobe)."""
    ch = np.asarray(ch, np.int64)
    row = ch // 2
    x = np.where(row % 2 == 0, np.where(ch % 2 == 0, 11.0, 43.0),
                 np.where(ch % 2 == 0, 27.0, 59.0))
    return np.stack([x, row * 20.0], 1)
