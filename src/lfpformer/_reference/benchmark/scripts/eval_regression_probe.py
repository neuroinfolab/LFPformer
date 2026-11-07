#!/usr/bin/env python3
"""Linear regression probe evaluation (advisor's xyz idea) + Gaussian variant.

Trains TWO linear probes on frozen embeddings, both predicting Allen CCF
coordinates from `nn.Linear(768, *)`:

  1. **MSE probe**  — `nn.Linear(768, 3)`, MSE loss, point estimate (μ only).
  2. **Gaussian probe** — `nn.Linear(768, 6)` split into (μ, log σ²), heteroscedastic
     Gaussian NLL. Learns per-sample uncertainty (σ_ap, σ_dv, σ_d in µm),
     useful for uncertainty-aware viz and calibration.

Both probes are genuine linear probes — a single linear layer on top of a
frozen backbone. No MLP, no hidden layer. The difference is just the output
shape and loss.

Targets are **per-fine_id centroids** (Variant B): for each of the 309
shared fine_ids, we average all probe_train channel coordinates with that
fine_id and use that as the target for every sample of that region. This
aligns regression with classification's per-region label and gives the
probe a clean `embedding → region centroid xyz` mapping.

LR is symmetrized: targets use `|lr - 5739|` (Allen CCF anatomical midline),
and predictions are decoded to the LEFT hemisphere (`lr = midline - d`)
because IBL coverage is 84% left / 16% right. Allen annotation is symmetric,
so region lookup returns the correct region regardless of hemisphere.

Predicted xyz is turned into a region prediction TWO ways at inference:

    (a) volume_lookup:    ba.ccf2xyz(pred) → ba.get_labels() → region_id
    (b) nearest_centroid: argmin_{c in train_centroids} ||pred - c||

Both are walked up the Allen CCF hierarchy to report depth-1..9 strict and
inclusive accuracy, mirroring eval_hierarchical.py.

Raw per-sample predictions are saved to <out>/predictions/*.npz so spatial
viz scripts can render predicted vs true xyz clouds (with σ from the
Gaussian probe available for uncertainty-aware rendering).

Output JSON (`regression.json`):

    {
      # === MSE probe results (top-level = volume lookup, primary) ===
      "fine":            {"n_classes": 309, "chance": ..., "linear": {mean,std}},
      "d1"..."d9":       same,
      "d1_inclusive"...: same,
      "um_errors":       {"linear": {median: [ap,dv,d], mean: [ap,dv,d]}},
      "baselines":       {"random_uniform": ..., "mean_coord": ..., "spatial_prior": ...},
      "centroid_method": {...},   # nearest-centroid inference version
      "normalizer_stats": {...},
      "shared_fine_ids": 309,

      # === Gaussian probe results (parallel experiment) ===
      "gaussian_probe": {
        "fine", "d1"..."d9", "d1_inclusive"...: same schema,
        "um_errors":   {...},
        "calibration": {
          "within_1sigma_per_axis": [ap,dv,d],  # ideal 68.3% each
          "within_2sigma_per_axis": [ap,dv,d],  # ideal 95.4% each
          "within_{1,2}sigma_joint": float,      # ideal 31.8% / 86.9%
          "nll_um": float,
          "mean_sigma":   [ap,dv,d],  # µm
          "median_sigma": [ap,dv,d],
        },
        "centroid_method": {...},
      }
    }

Usage:
    python eval_regression_probe.py \
        --checkpoint <path/to/ssl_wav2vec2_best_valloss.pt> \
        --eval_data_dir data/ibl/subject_split_v2/eval_v2_100k \
        --output_dir /workspace/ws2/ray_results/eval_v2/models/<name>
"""

import argparse
import copy
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ALPHABRAIN_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ALPHABRAIN_ROOT)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ────────────────────────────────────────────────────────────────────────────
#                           Allen atlas helpers
# ────────────────────────────────────────────────────────────────────────────

def build_region_ancestors(ba):
    """region_id → {depth: acronym} for every Allen region."""
    region_hier = {}
    n, n_fail = 0, 0
    for rid in ba.regions.id:
        rid = int(rid)
        if rid <= 0:
            continue
        try:
            anc = ba.regions.ancestors(rid)
            hier = {}
            for i in range(len(anc.id)):
                d = int(anc.level[i])
                hier[d] = str(anc.acronym[i])
            region_hier[rid] = hier
            n += 1
        except Exception:
            n_fail += 1
    log(f"  Built hierarchy for {n} regions ({n_fail} failed)")
    return region_hier


def coords_to_region_ids(ba, coords_apdvml_um):
    """(N, 3) µm (ap, dv, lr) → (N,) int64 region_ids via annotation volume."""
    xyz_m = ba.ccf2xyz(np.asarray(coords_apdvml_um, dtype=np.float64),
                       ccf_order="apdvml")
    return np.atleast_1d(ba.get_labels(xyz_m, mode="clip")).astype(np.int64)


# ────────────────────────────────────────────────────────────────────────────
#                   Centroid targets + normalization
# ────────────────────────────────────────────────────────────────────────────

def compute_region_centroids(coords, fine_ids, keep_ids):
    """Compute per-fine_id centroid from (coords, fine_ids), restricted to keep_ids.

    Returns a dict {fine_id: (ap, dv, lr)} in µm.
    """
    centroids = {}
    keep_set = set(int(x) for x in keep_ids)
    for fid in sorted(keep_set):
        mask = (fine_ids == fid)
        if mask.sum() == 0:
            continue
        centroids[int(fid)] = coords[mask].mean(axis=0)
    return centroids


ALLEN_MIDLINE_UM = 5739.0  # Allen CCF anatomical midline (fixed, not data-derived)


class CentroidNormalizer:
    """Normalize (ap, dv, |lr - midline|) to [0, 1] using CENTROIDS bounding box.

    LR is symmetrized: the target is distance from the Allen CCF midline
    (5739 µm), so both hemispheres fold to the same value. This avoids
    the 84/16 hemisphere coverage asymmetry in the IBL data and halves the
    LR regression range. At decode time, lr is reconstructed as
    midline + d (right hemisphere); Allen annotation is symmetric so
    region lookup is correct regardless of original hemisphere.

    AP and DV are normalized as raw coordinates (no folding).
    """

    def __init__(self, centroids_dict):
        c = np.stack(list(centroids_dict.values()), axis=0)  # (K, 3)
        self.ap_min, self.ap_max = float(c[:, 0].min()), float(c[:, 0].max())
        self.dv_min, self.dv_max = float(c[:, 1].min()), float(c[:, 1].max())
        self.midline = ALLEN_MIDLINE_UM
        d = np.abs(c[:, 2] - self.midline)
        self.d_min, self.d_max = float(d.min()), float(d.max())

    def stats(self):
        return {
            "ap_min": self.ap_min, "ap_max": self.ap_max,
            "dv_min": self.dv_min, "dv_max": self.dv_max,
            "midline": self.midline,
            "d_min": self.d_min, "d_max": self.d_max,
        }

    def encode(self, coords):
        """(N, 3) µm (ap, dv, lr) → (N, 3) normalized (ap_n, dv_n, d_n)."""
        coords = np.asarray(coords, dtype=np.float64)
        ap_n = (coords[:, 0] - self.ap_min) / max(self.ap_max - self.ap_min, 1e-9)
        dv_n = (coords[:, 1] - self.dv_min) / max(self.dv_max - self.dv_min, 1e-9)
        d_n = (np.abs(coords[:, 2] - self.midline) - self.d_min) \
              / max(self.d_max - self.d_min, 1e-9)
        return np.stack([ap_n, dv_n, d_n], axis=1).astype(np.float32)

    def decode(self, normalized):
        """(N, 3) normalized → (N, 3) µm (ap, dv, lr).

        lr is reconstructed as midline - d (LEFT hemisphere). IBL coverage is
        84% left / 16% right, so placing predictions on the left hemisphere
        minimizes µm error vs true coords. Allen CCF annotation is symmetric,
        so region lookup returns the correct region regardless of hemisphere.
        """
        arr = np.asarray(normalized, dtype=np.float64)
        ap = arr[:, 0] * (self.ap_max - self.ap_min) + self.ap_min
        dv = arr[:, 1] * (self.dv_max - self.dv_min) + self.dv_min
        d = arr[:, 2] * (self.d_max - self.d_min) + self.d_min
        lr = self.midline - d
        return np.stack([ap, dv, lr], axis=1)


# ────────────────────────────────────────────────────────────────────────────
#                        Probe training
# ────────────────────────────────────────────────────────────────────────────

def train_linear_regression_probe(train_emb, train_tgt, val_emb, val_tgt,
                                  device, epochs=100, patience=10,
                                  batch_size=512, lr=1e-3):
    """Train a single linear regression probe with MSE + early stopping on val MSE."""
    hidden_dim = train_emb.shape[1]
    out_dim = train_tgt.shape[1]

    model = nn.Linear(hidden_dim, out_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    tr_x = torch.from_numpy(train_emb).float().to(device)
    tr_y = torch.from_numpy(train_tgt).float().to(device)
    va_x = torch.from_numpy(val_emb).float().to(device)
    va_y = torch.from_numpy(val_tgt).float().to(device)

    best_mse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(tr_x), device=device)
        for s in range(0, len(tr_x), batch_size):
            b = idx[s:s + batch_size]
            loss = criterion(model(tr_x[b]), tr_y[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mse = float(criterion(model(va_x), va_y))
        if val_mse < best_mse - 1e-5:
            best_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_mse


# Log-variance clamp range for Gaussian probe (prevents numerical issues).
LOG_VAR_MIN = -10.0  # σ² ≥ ~4.5e-5 in normalized space
LOG_VAR_MAX =  10.0  # σ² ≤ ~22026


class GaussianLinearProbe(nn.Module):
    """Single linear layer predicting (μ, log σ²) for 3D coordinates.

    Output: 6 floats — 3 means + 3 log-variances (both in normalized space).
    Same parameter class as `nn.Linear(768, 3)` — just 768 → 6 instead of 768 → 3,
    so it qualifies as a linear probe.
    """

    def __init__(self, input_dim=768):
        super().__init__()
        self.mu = nn.Linear(input_dim, 3)
        self.log_var = nn.Linear(input_dim, 3)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x).clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        return mu, log_var


class XYZMLPHead(nn.Module):
    """Nonlinear coordinate head matched to frozen-probe / full-FT MLP arms."""

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
    """MLP analogue of GaussianLinearProbe (μ + log σ² via shared trunk)."""

    def __init__(self, input_dim=768):
        super().__init__()
        self.net = XYZMLPHead(input_dim, output_dim=6)

    def forward(self, x):
        out = self.net(x)
        return out[:, :3], out[:, 3:].clamp(LOG_VAR_MIN, LOG_VAR_MAX)


def make_xyz_heads_for_eval(head_type, input_dim, device):
    if head_type == "mlp":
        return (XYZMLPHead(input_dim).to(device),
                GaussianMLPProbe(input_dim).to(device))
    return (nn.Linear(input_dim, 3).to(device),
            GaussianLinearProbe(input_dim=input_dim).to(device))


class GaussianNLL3D(nn.Module):
    """Heteroscedastic Gaussian NLL for 3D coordinates.

        L = 0.5 * Σ_d [ log_var_d + (y_d - μ_d)² / exp(log_var_d) ]

    The 0.5 × 3 × log(2π) constant is omitted (doesn't affect optimization).
    """

    def forward(self, mu, log_var, target):
        var = log_var.exp()
        nll = 0.5 * (log_var + (target - mu) ** 2 / var)
        return nll.sum(dim=1).mean()


def train_gaussian_linear_probe(train_emb, train_tgt, val_emb, val_tgt,
                                device, epochs=100, patience=10,
                                batch_size=512, lr=1e-3):
    """Train a single linear Gaussian probe with heteroscedastic NLL + early
    stopping on val NLL.
    """
    hidden_dim = train_emb.shape[1]
    model = GaussianLinearProbe(input_dim=hidden_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = GaussianNLL3D()

    tr_x = torch.from_numpy(train_emb).float().to(device)
    tr_y = torch.from_numpy(train_tgt).float().to(device)
    va_x = torch.from_numpy(val_emb).float().to(device)
    va_y = torch.from_numpy(val_tgt).float().to(device)

    best_nll = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(tr_x), device=device)
        for s in range(0, len(tr_x), batch_size):
            b = idx[s:s + batch_size]
            mu, log_var = model(tr_x[b])
            loss = criterion(mu, log_var, tr_y[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            mu_v, lv_v = model(va_x)
            val_nll = float(criterion(mu_v, lv_v, va_y))
        if val_nll < best_nll - 1e-4:
            best_nll = val_nll
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_nll


def predict_xyz_gaussian(model, emb, normalizer, device, batch_size=4096):
    """embeddings → (pred_xyz_um, pred_sigma_um). Both (N, 3)."""
    model.eval()
    mus, log_vars = [], []
    with torch.no_grad():
        for s in range(0, len(emb), batch_size):
            xb = torch.from_numpy(emb[s:s + batch_size]).float().to(device)
            mu, lv = model(xb)
            mus.append(mu.cpu().numpy())
            log_vars.append(lv.cpu().numpy())
    mu_n = np.concatenate(mus)
    log_var_n = np.concatenate(log_vars)

    # De-normalize μ the same way as the MSE probe
    pred_xyz = normalizer.decode(mu_n)

    # De-normalize σ per axis.
    # σ_raw = σ_normalized × (raw_max - raw_min). Axis ordering:
    #   ap_n → ap range, dv_n → dv range, lr_n → d range (distance-from-midline).
    scale = np.array([
        normalizer.ap_max - normalizer.ap_min,
        normalizer.dv_max - normalizer.dv_min,
        normalizer.d_max - normalizer.d_min,
    ], dtype=np.float64)
    sigma = np.sqrt(np.exp(log_var_n)) * scale  # (N, 3) in µm
    return pred_xyz.astype(np.float32), sigma.astype(np.float32)


def predict_xyz(model, emb, normalizer, device, batch_size=4096):
    """embeddings → predicted (ap, dv, lr) in µm."""
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(emb), batch_size):
            xb = torch.from_numpy(emb[s:s + batch_size]).float().to(device)
            outs.append(model(xb).cpu().numpy())
    pred_norm = np.concatenate(outs)
    return normalizer.decode(pred_norm)


# ────────────────────────────────────────────────────────────────────────────
#           Linear probe with hemisphere head (strict linear — two
#           independent nn.Linear layers, no hidden activations)
# ────────────────────────────────────────────────────────────────────────────

class LinearProbeWithHemisphere(nn.Module):
    """Two independent linear heads:
        coord_head = nn.Linear(768, 3)  →  (ap_n, dv_n, d_n)  in [0, 1]
        side_head  = nn.Linear(768, 1)  →  p_right logit (pre-sigmoid)

    The side head is a binary classifier: p_right > 0.5 → right hemisphere,
    ≤ 0.5 → left. Decoded LR: midline + d × sign(p_right − 0.5).

    Kept strictly linear (no hidden layers, no activations) so it still
    qualifies as a "linear probe" for fair comparison with the existing
    MSE probe — total params = 768×3 + 768 = 3072 (vs 768×3 = 2304 for MSE).
    """

    def __init__(self, input_dim=768):
        super().__init__()
        self.coord_head = nn.Linear(input_dim, 3)
        self.side_head = nn.Linear(input_dim, 1)

    def forward(self, x):
        coord_norm = self.coord_head(x)   # (B, 3)
        side_logit = self.side_head(x).squeeze(-1)  # (B,)
        return coord_norm, side_logit


def train_linear_probe_with_hemisphere(train_emb, train_tgt, train_side,
                                       val_emb, val_tgt, val_side,
                                       device, epochs=100, patience=10,
                                       batch_size=512, lr=1e-3,
                                       hemi_lambda=1.0):
    """Train the combined coord + hemi-side probe.

    Loss = MSE(coord_pred, coord_tgt)  +  hemi_lambda × BCE(side_logit, side_tgt)

    hemi_lambda balances the two objectives. Default 1.0 = treat side as roughly
    equally important as the regression (BCE and MSE in normalized space are
    commensurate orders of magnitude).

    Early stopping uses val total loss (sum of both components).
    """
    hidden_dim = train_emb.shape[1]
    model = LinearProbeWithHemisphere(input_dim=hidden_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    tr_x = torch.from_numpy(train_emb).float().to(device)
    tr_y = torch.from_numpy(train_tgt).float().to(device)
    tr_s = torch.from_numpy(train_side).float().to(device)
    va_x = torch.from_numpy(val_emb).float().to(device)
    va_y = torch.from_numpy(val_tgt).float().to(device)
    va_s = torch.from_numpy(val_side).float().to(device)

    best_total = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    best_metrics = {"mse": float("inf"), "bce": float("inf"), "hemi_acc": 0.0}

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(tr_x), device=device)
        for s in range(0, len(tr_x), batch_size):
            b = idx[s:s + batch_size]
            coord_pred, side_logit = model(tr_x[b])
            loss = mse(coord_pred, tr_y[b]) + hemi_lambda * bce(side_logit, tr_s[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            cp_v, sl_v = model(va_x)
            val_mse = float(mse(cp_v, va_y))
            val_bce = float(bce(sl_v, va_s))
            val_total = val_mse + hemi_lambda * val_bce
            val_hemi_acc = float(((torch.sigmoid(sl_v) > 0.5).float() == va_s).float().mean())
        if val_total < best_total - 1e-5:
            best_total = val_total
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {"mse": val_mse, "bce": val_bce, "hemi_acc": val_hemi_acc}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_metrics


def predict_xyz_with_hemisphere(model, emb, normalizer, device, batch_size=4096):
    """embeddings → (pred_xyz_um, p_right). Both arrays have N rows.

    Decode: lr = midline + d × sign(p_right − 0.5) where d = decoded distance
    from the coord head (distance-from-midline, always ≥ 0). Hard threshold
    at 0.5 keeps the decode clean; p_right is ALSO returned so downstream code
    can do its own thresholding or use soft decode if desired.
    """
    model.eval()
    coord_norms, p_rights = [], []
    with torch.no_grad():
        for s in range(0, len(emb), batch_size):
            xb = torch.from_numpy(emb[s:s + batch_size]).float().to(device)
            cp, sl = model(xb)
            coord_norms.append(cp.cpu().numpy())
            p_rights.append(torch.sigmoid(sl).cpu().numpy())
    coord_norm = np.concatenate(coord_norms)  # (N, 3) in [0, 1] ideally
    p_right = np.concatenate(p_rights)         # (N,)  in [0, 1]

    # Decode ap, dv the normal way. For LR, reconstruct with hemisphere sign.
    ap = coord_norm[:, 0] * (normalizer.ap_max - normalizer.ap_min) + normalizer.ap_min
    dv = coord_norm[:, 1] * (normalizer.dv_max - normalizer.dv_min) + normalizer.dv_min
    d = coord_norm[:, 2] * (normalizer.d_max - normalizer.d_min) + normalizer.d_min
    sign = np.where(p_right > 0.5, 1.0, -1.0)
    lr = normalizer.midline + d * sign
    pred_xyz = np.stack([ap, dv, lr], axis=1).astype(np.float32)
    return pred_xyz, p_right.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
#                  Inference: volume lookup vs nearest centroid
# ────────────────────────────────────────────────────────────────────────────

def nearest_centroid_predict(pred_xyz_um, centroid_coords_um, centroid_fids):
    """For each predicted xyz, return the fine_id of the nearest training centroid.

    pred_xyz_um: (N, 3)
    centroid_coords_um: (K, 3)
    centroid_fids: (K,) int64 — fine_id for each centroid
    """
    # Pairwise L2^2 distance. For typical K=309, N=100K this is cheap (~3M ops)
    # but chunk to be safe on memory.
    N = len(pred_xyz_um)
    out = np.zeros(N, dtype=np.int64)
    chunk = 10000
    c = np.asarray(centroid_coords_um, dtype=np.float64)
    cnorm_sq = (c * c).sum(axis=1)  # (K,)
    for s in range(0, N, chunk):
        p = np.asarray(pred_xyz_um[s:s + chunk], dtype=np.float64)
        pnorm_sq = (p * p).sum(axis=1, keepdims=True)  # (chunk, 1)
        dist_sq = pnorm_sq + cnorm_sq[None, :] - 2.0 * p @ c.T  # (chunk, K)
        nearest = np.argmin(dist_sq, axis=1)
        out[s:s + chunk] = centroid_fids[nearest]
    return out


# ────────────────────────────────────────────────────────────────────────────
#                   Hierarchy check (mirrors eval_hierarchical)
# ────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(pred_region_ids, true_region_ids, region_ancestors,
                         shared_fine_ids, min_samples=10):
    """Walk pred & true region_ids up the hierarchy and compute balanced
    accuracy at each depth 1-9 (strict + inclusive) and at fine level.
    """
    from sklearn.metrics import balanced_accuracy_score

    keep = np.array([int(r) in shared_fine_ids for r in true_region_ids], dtype=bool)
    if keep.sum() == 0:
        return None

    pred = np.asarray(pred_region_ids)[keep]
    true = np.asarray(true_region_ids)[keep]

    results = {}

    if len(set(true.tolist())) > 1:
        fine_bal = balanced_accuracy_score(true, pred)
    else:
        fine_bal = float((pred == true).mean())
    results["fine"] = {"bal_acc": fine_bal, "n_classes": len(shared_fine_ids)}

    def get_at_depth_or_finest(anc, depth):
        if depth in anc:
            return anc[depth]
        for d in range(depth - 1, -1, -1):
            if d in anc:
                return anc[d]
        return None

    for depth in range(1, 10):
        t_strict, p_strict = [], []
        t_incl, p_incl = [], []

        for p, t in zip(pred, true):
            t_anc = region_ancestors.get(int(t), {})
            p_anc = region_ancestors.get(int(p), {})

            ts = t_anc.get(depth)
            ps = p_anc.get(depth)
            if ts is not None and ps is not None:
                t_strict.append(ts)
                p_strict.append(ps)

            ti = get_at_depth_or_finest(t_anc, depth)
            pi = get_at_depth_or_finest(p_anc, depth)
            if ti is not None and pi is not None:
                t_incl.append(ti)
                p_incl.append(pi)

        def _bal(true_list, pred_list):
            if len(true_list) < min_samples:
                return None
            labels = sorted(set(true_list) | set(pred_list))
            remap = {l: i for i, l in enumerate(labels)}
            ti = [remap[l] for l in true_list]
            pi = [remap[l] for l in pred_list]
            if len(set(ti)) > 1:
                return (balanced_accuracy_score(ti, pi), len(set(true_list)), len(true_list))
            return (float(np.mean(np.array(ti) == np.array(pi))),
                    len(set(true_list)), len(true_list))

        r = _bal(t_strict, p_strict)
        if r is not None:
            bal, n_cls, n_samp = r
            results[depth] = {
                "bal_acc": bal, "n_classes": n_cls,
                "n_samples": n_samp, "chance": 1.0 / max(n_cls, 1),
            }
        r = _bal(t_incl, p_incl)
        if r is not None:
            bal, n_cls, n_samp = r
            results[f"{depth}_inclusive"] = {
                "bal_acc": bal, "n_classes": n_cls,
                "n_samples": n_samp, "chance": 1.0 / max(n_cls, 1),
            }

    return results


# ────────────────────────────────────────────────────────────────────────────
#                              Baselines
# ────────────────────────────────────────────────────────────────────────────

def baseline_random_uniform(train_coords, n_samples, rng):
    c = np.asarray(train_coords, dtype=np.float64)
    lo, hi = c.min(axis=0), c.max(axis=0)
    return rng.uniform(lo, hi, size=(n_samples, 3))


def baseline_mean_coord(train_coords, n_samples):
    c = np.asarray(train_coords, dtype=np.float64)
    return np.tile(c.mean(axis=0), (n_samples, 1))


def baseline_spatial_prior(train_coords, n_samples, rng):
    c = np.asarray(train_coords, dtype=np.float64)
    idx = rng.integers(0, len(c), size=n_samples)
    return c[idx]


# ────────────────────────────────────────────────────────────────────────────
#                                main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hemi_lambda", type=float, default=1.0,
                        help="Weight of BCE hemisphere loss relative to "
                             "coord MSE in the hemi-head probe.")
    parser.add_argument(
        "--target_mode", default="region_centroid",
        choices=["region_centroid", "per_channel_xyz"],
        help=("Probe target. `region_centroid` (default, Variant B) = centroid "
              "of the sample's fine_id region (piecewise-constant in depth, "
              "original behaviour). `per_channel_xyz` (Variant A) = sample's "
              "actual channel xyz — gives continuous predictions useful for "
              "smooth PRED-path visualization; fine bal_acc still computed "
              "via nearest-centroid / volume lookup after prediction."),
    )
    parser.add_argument(
        "--use_model_head", action="store_true",
        help=("Skip fitting fresh linear/Gaussian probes; instead load "
              "`mse_head_state_dict` and `gauss_head_state_dict` from the "
              "ckpt and use them directly. Used for end-to-end fine-tuned "
              "models (path A and path B from train_xyz_finetune.py) where "
              "the heads are part of the model and refitting would discard "
              "what the joint optimisation learned."),
    )
    parser.add_argument(
        "--ap_features_npz", default=None,
        help=("Path to pid_to_ap_rms_chan_uV.npz (built by "
              "extract_ap_rms_per_pid.py). When set, each sample's "
              "(probe_key, channel_idx) is looked up and the per-channel AP "
              "RMS scalar (log10-µV, z-scored on probe_train) is concatenated "
              "to the LFP embedding before the probe — i.e. LFP+AP probe. "
              "Probe heads adapt to the new input_dim (768 → 769) via "
              "hidden_dim = train_emb.shape[1]. Incompatible with "
              "--use_model_head (head sized to 768)."),
    )
    parser.add_argument(
        "--ap_row", type=int, default=1, choices=[0, 1],
        help="Which row of the (2, 384) AP RMS file to use (0=raw, 1=destriped).",
    )
    args = parser.parse_args()
    if args.ap_features_npz and args.use_model_head:
        raise SystemExit("--ap_features_npz incompatible with --use_model_head "
                         "(model head was sized to 768-d LFP only).")

    os.makedirs(args.output_dir, exist_ok=True)
    pred_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    from blind_localization.data.lazyloader_dataset import CompactDataset
    from backbones import get_backbone
    from utils.preprocessing import preprocess_batch
    from iblatlas.atlas import AllenAtlas

    # ────── Allen atlas ──────
    log("Loading AllenAtlas(res_um=25)...")
    ba = AllenAtlas(res_um=25)
    log("Building region → ancestors lookup...")
    region_ancestors = build_region_ancestors(ba)

    # ────── Backbone ──────
    log(f"Loading backbone from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    backbone = get_backbone(config.get("backbone", "wav2vec2"), config).to(device)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.eval()

    # ────── Datasets ──────
    compact_kwargs = dict(
        include_labels=True, dataset="IBL", verbose=False, atlas_depth=9,
        already_resampled=False, gpu_resample=True, input_sr=1250, model_sr=16000,
    )
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=4,
                         pin_memory=True, shuffle=False)

    splits = ["probe_train", "probe_val"] + [f"test_fold_{i}" for i in range(5)]
    datasets = {}
    for s in splits:
        d = os.path.join(args.eval_data_dir, s)
        if os.path.isdir(d):
            datasets[s] = CompactDataset(compact_dir=d, **compact_kwargs)
            log(f"  {s}: {len(datasets[s])} samples")

    # ────── Collect coords + fine_ids + meta_str per split ──────
    def _get_meta_str(ds, idx):
        """Get meta_str from raw sample (not exposed by __getitem__)."""
        file_idx, offset = ds._find_file_and_offset(idx)
        samples = ds._load_file(file_idx)
        return str(samples[offset][1])

    def _get_hierarchy(ds, idx):
        """Return the raw atlas hierarchy dict for sample idx.

        Local fallback for CompactDataset.get_hierarchy(), which exists in
        ws2's tree but not ws1's. Reads the raw sample's label slot
        (sample[2]) directly via the same file/offset helpers _get_meta_str
        uses. Returns the dict, or None if the label is not a dict.
        """
        if hasattr(ds, "get_hierarchy"):
            return ds.get_hierarchy(idx)
        file_idx, offset = ds._find_file_and_offset(idx)
        raw_label = ds._load_file(file_idx)[offset][2]
        return raw_label if isinstance(raw_label, dict) else None

    def collect_meta(ds):
        n = len(ds)
        coords = np.zeros((n, 3), dtype=np.float64)
        fids = np.zeros(n, dtype=np.int64)
        meta_strs = []
        for i in range(n):
            _, _, c = ds[i]
            h = _get_hierarchy(ds, i)
            coords[i, 0] = float(c[0])  # ap
            coords[i, 1] = float(c[1])  # dv
            coords[i, 2] = float(c[2])  # lr
            fids[i] = int(h["fine_id"]) if h and "fine_id" in h else 0
            meta_strs.append(_get_meta_str(ds, i))
        return coords, fids, meta_strs

    log("\nCollecting coords + fine_ids + meta_str...")
    meta = {}
    for s, ds in datasets.items():
        log(f"  {s}...")
        meta[s] = collect_meta(ds)

    # Shared fine_ids across all splits
    fid_sets = {s: set(meta[s][1].tolist()) for s in datasets}
    shared_fine_ids = sorted(set.intersection(*fid_sets.values()) - {0})
    log(f"  Shared fine_ids across all splits: {len(shared_fine_ids)}")

    # ────── Compute centroids from probe_train (filtered to shared) ──────
    tr_coords_all, tr_fids_all, tr_mstrs_all = meta["probe_train"]
    train_keep = np.isin(tr_fids_all, shared_fine_ids)
    tr_coords_shared = tr_coords_all[train_keep]
    tr_fids_shared = tr_fids_all[train_keep]

    centroids = compute_region_centroids(tr_coords_shared, tr_fids_shared, shared_fine_ids)
    log(f"  Computed centroids for {len(centroids)} regions")

    # Dense centroid arrays for nearest-centroid inference
    centroid_fids_arr = np.array(sorted(centroids.keys()), dtype=np.int64)
    centroid_coords_arr = np.stack([centroids[int(fid)] for fid in centroid_fids_arr], axis=0)

    normalizer = CentroidNormalizer(centroids)
    log(f"Normalizer stats: {normalizer.stats()}")

    # ────── Extract embeddings ──────
    def extract_emb(ds):
        loader = DataLoader(ds, **loader_kwargs)
        all_emb = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].float().to(device)
                x = preprocess_batch(x, backbone.needs_resampling, backbone.needs_normalization)
                emb = backbone.encode(x)
                all_emb.append(emb.cpu().numpy())
        return np.concatenate(all_emb)

    log("\nExtracting embeddings...")
    embeddings = {}
    for s, ds in datasets.items():
        log(f"  {s}...")
        embeddings[s] = extract_emb(ds)

    del backbone
    torch.cuda.empty_cache()

    # ────── (Optional) Concatenate per-channel AP-RMS scalar ──────
    # When --ap_features_npz is set, look up each sample's (probe_key, ch_idx)
    # in the AP RMS cache and append a single log10-µV z-scored scalar.
    # input_dim grows 768 → 769; probe heads adapt automatically.
    if args.ap_features_npz:
        log(f"\nLoading AP features from {args.ap_features_npz}")
        ap_npz = np.load(args.ap_features_npz)
        ap_cache = {k: ap_npz[k] for k in ap_npz.files}   # key=<eid>_probeNN
        log(f"  {len(ap_cache)} PIDs in AP cache, ap_row={args.ap_row}")

        def parse_pid_ch(meta_str):
            """IBL meta_str: eid_idx_probeNN_pid-chNNNN_region → (pid_key, ch)."""
            parts = str(meta_str).split("_")
            if len(parts) < 4 or "-ch" not in parts[3]:
                return None, None
            try:
                ch = int(parts[3].split("-ch")[1])
            except (ValueError, IndexError):
                return None, None
            return f"{parts[0]}_{parts[2]}", ch

        FLOOR_UV = 1e-3   # log floor (avoid log(0) for dead channels)

        def lookup_ap_uV(meta_strs):
            n = len(meta_strs)
            out = np.full(n, np.nan, dtype=np.float64)
            n_miss_pid, n_miss_parse = 0, 0
            for i, m in enumerate(meta_strs):
                pid, ch = parse_pid_ch(m)
                if pid is None:
                    n_miss_parse += 1; continue
                arr = ap_cache.get(pid)
                if arr is None:
                    n_miss_pid += 1; continue
                out[i] = float(arr[args.ap_row, ch])
            return out, n_miss_pid, n_miss_parse

        # Per-split AP scalar (log10 µV, fill NaN with split mean before z-score)
        ap_logs = {}
        for s in datasets:
            mstrs = meta[s][2]
            uV, miss_pid, miss_parse = lookup_ap_uV(mstrs)
            log(f"  {s}: looked up AP for {len(mstrs)} samples — "
                f"{(~np.isnan(uV)).sum()} hit, {miss_pid} miss-pid, "
                f"{miss_parse} miss-parse")
            log_uV = np.log10(np.maximum(uV, FLOOR_UV))
            ap_logs[s] = log_uV

        # z-score using probe_train statistics
        tr_log = ap_logs["probe_train"]
        tr_log_finite = tr_log[np.isfinite(tr_log)]
        ap_mean = float(tr_log_finite.mean())
        ap_std = float(tr_log_finite.std() + 1e-8)
        log(f"  AP z-score on probe_train: mean(log10_uV)={ap_mean:.3f}, "
            f"std={ap_std:.3f}")

        for s in datasets:
            log_uV = ap_logs[s]
            # Replace NaN (missing PID lookup) with the train mean = z-score 0.
            log_uV = np.where(np.isfinite(log_uV), log_uV, ap_mean)
            ap_z = ((log_uV - ap_mean) / ap_std).astype(np.float32)
            old = embeddings[s]
            embeddings[s] = np.concatenate(
                [old, ap_z[:, None].astype(old.dtype)], axis=1
            )
        log(f"  Embeddings now: {embeddings['probe_train'].shape[1]}-d "
            f"(LFP {old.shape[1]} + AP 1)")

    # ────── Build training data (filter to shared) ──────
    # probe_train (filtered)
    tr_emb = embeddings["probe_train"][train_keep]
    va_coords_all, va_fids_all, _va_mstrs = meta["probe_val"]
    val_keep = np.isin(va_fids_all, shared_fine_ids)
    va_emb = embeddings["probe_val"][val_keep]
    va_fids_shared = va_fids_all[val_keep]
    va_coords_shared_all = va_coords_all[val_keep]   # actual channel xyz for val

    if args.target_mode == "region_centroid":
        # Variant B: target = centroid of sample's fine_id region (original).
        tr_tgt_xyz = np.stack([centroids[int(fid)] for fid in tr_fids_shared], axis=0)
        va_tgt_xyz = np.stack([centroids[int(fid)] for fid in va_fids_shared], axis=0)
        log("Target mode: REGION_CENTROID (Variant B — piecewise-constant).")
    else:
        # Variant A: target = sample's actual channel xyz. Continuous along
        # probe shank → smooth PRED paths, visually comparable to TRUE.
        tr_tgt_xyz = tr_coords_shared.astype(np.float64)
        va_tgt_xyz = va_coords_shared_all.astype(np.float64)
        log("Target mode: PER_CHANNEL_XYZ (Variant A — continuous channel coords).")
        # Widen normalizer bounds: channel coords can extend beyond the
        # centroid bounding box. Extend AP/DV/d ranges to cover actual data.
        combined = np.concatenate([tr_tgt_xyz, va_tgt_xyz], axis=0)
        ap_lo, ap_hi = float(combined[:, 0].min()), float(combined[:, 0].max())
        dv_lo, dv_hi = float(combined[:, 1].min()), float(combined[:, 1].max())
        d_all = np.abs(combined[:, 2] - normalizer.midline)
        d_lo, d_hi = float(d_all.min()), float(d_all.max())
        log(f"  Normalizer extended for Variant A: "
            f"ap[{ap_lo:.0f},{ap_hi:.0f}] dv[{dv_lo:.0f},{dv_hi:.0f}] d[{d_lo:.0f},{d_hi:.0f}]")
        normalizer.ap_min = min(normalizer.ap_min, ap_lo)
        normalizer.ap_max = max(normalizer.ap_max, ap_hi)
        normalizer.dv_min = min(normalizer.dv_min, dv_lo)
        normalizer.dv_max = max(normalizer.dv_max, dv_hi)
        normalizer.d_min  = min(normalizer.d_min,  d_lo)
        normalizer.d_max  = max(normalizer.d_max,  d_hi)

    tr_tgt_norm = normalizer.encode(tr_tgt_xyz)
    va_tgt_norm = normalizer.encode(va_tgt_xyz)

    log(f"\nTrain: {tr_emb.shape} embeddings, {tr_tgt_norm.shape} targets "
        f"(filtered from {len(tr_fids_all)} → {len(tr_fids_shared)})")
    log(f"Val:   {va_emb.shape} embeddings, {va_tgt_norm.shape} targets "
        f"(filtered from {len(va_fids_all)} → {len(va_fids_shared)})")

    # Hemisphere targets (p_right = 1.0 if true lr > midline). Use the FILTERED
    # train/val coords — same samples as tr_emb/va_emb, so the arrays align.
    tr_side = (tr_coords_shared[:, 2] > normalizer.midline).astype(np.float32)
    va_coords_shared = va_coords_all[val_keep]
    va_side = (va_coords_shared[:, 2] > normalizer.midline).astype(np.float32)
    tr_right_frac = float(tr_side.mean())
    va_right_frac = float(va_side.mean())
    log(f"Hemisphere distribution: train right-hem {tr_right_frac:.3f}, "
        f"val right-hem {va_right_frac:.3f} "
        f"(majority-class baseline accuracy: "
        f"{max(1 - tr_right_frac, tr_right_frac):.3f})")

    # ────── MSE + Gaussian probes — fresh-fit OR load from fine-tuned ckpt ──
    if args.use_model_head:
        # End-to-end fine-tune mode: ckpt already has the trained heads.
        # Build matching modules and load saved state dicts. Skip probe fit.
        log("\n--use_model_head: loading mse/gauss heads from ckpt "
            "(skipping fresh probe fitting).")
        if "mse_head_state_dict" not in ckpt or "gauss_head_state_dict" not in ckpt:
            raise SystemExit(
                "--use_model_head requires both `mse_head_state_dict` and "
                "`gauss_head_state_dict` in the ckpt. Got keys: "
                f"{list(ckpt.keys())}"
            )
        hidden_dim = tr_emb.shape[1]
        head_type = ckpt.get("head_type")
        if head_type is None and isinstance(ckpt.get("args"), dict):
            head_type = ckpt["args"].get("head_type", "linear")
        if head_type is None:
            head_type = "linear"
        log(f"  ckpt head_type={head_type}")
        lin_model, gauss_model = make_xyz_heads_for_eval(head_type, hidden_dim, device)
        lin_model.load_state_dict(ckpt["mse_head_state_dict"])
        gauss_model.load_state_dict(ckpt["gauss_head_state_dict"])
        # Compute val MSE / NLL on the saved heads for reporting parity.
        with torch.no_grad():
            va_x = torch.from_numpy(va_emb).float().to(device)
            va_y = torch.from_numpy(va_tgt_norm).float().to(device)
            lin_val_mse = float(nn.MSELoss()(lin_model(va_x), va_y))
            mu_v, lv_v = gauss_model(va_x)
            gauss_val_nll = float(GaussianNLL3D()(mu_v, lv_v, va_y))
        log(f"  Loaded MSE head val MSE: {lin_val_mse:.6f}")
        log(f"  Loaded Gaussian head val NLL: {gauss_val_nll:.6f}")
    else:
        log("\nTraining linear MSE probe (nn.Linear(768, 3), point estimate)...")
        lin_model, lin_val_mse = train_linear_regression_probe(
            tr_emb, tr_tgt_norm, va_emb, va_tgt_norm, device,
            epochs=args.epochs, patience=args.patience, lr=args.lr,
        )
        log(f"  Linear MSE probe best val MSE: {lin_val_mse:.6f}")

        log("\nTraining linear Gaussian probe (nn.Linear(768, 6), NLL heteroscedastic)...")
        gauss_model, gauss_val_nll = train_gaussian_linear_probe(
            tr_emb, tr_tgt_norm, va_emb, va_tgt_norm, device,
            epochs=args.epochs, patience=args.patience, lr=args.lr,
        )
        log(f"  Gaussian probe best val NLL: {gauss_val_nll:.6f}")

    # Hemi probe is independent of fine-tune heads — always fitted fresh.
    # (The fine-tune script does not produce a hemi head.)
    log("\nTraining hemi-head probe (nn.Linear(768, 3) + nn.Linear(768, 1), "
        "MSE + λ·BCE, λ=1.0)...")
    hemi_model, hemi_val_metrics = train_linear_probe_with_hemisphere(
        tr_emb, tr_tgt_norm, tr_side, va_emb, va_tgt_norm, va_side,
        device, epochs=args.epochs, patience=args.patience, lr=args.lr,
        hemi_lambda=args.hemi_lambda,
    )
    log(f"  Hemi probe best: mse={hemi_val_metrics['mse']:.6f}, "
        f"bce={hemi_val_metrics['bce']:.4f}, "
        f"hemi_acc={hemi_val_metrics['hemi_acc']:.4f} "
        f"(majority baseline={max(1 - va_right_frac, va_right_frac):.4f})")

    # Save all three probes' weights
    probe_dir = os.path.join(args.output_dir, "probes")
    os.makedirs(probe_dir, exist_ok=True)
    torch.save({
        "linear_state_dict": lin_model.state_dict(),
        "gaussian_state_dict": gauss_model.state_dict(),
        "hemi_state_dict": hemi_model.state_dict(),
        "normalizer_stats": normalizer.stats(),
        "centroids": {int(k): v.tolist() for k, v in centroids.items()},
        "best_val_mse": {"linear": lin_val_mse},
        "best_val_nll": {"gaussian": gauss_val_nll},
        "best_val_hemi": hemi_val_metrics,
        "hemi_lambda": args.hemi_lambda,
    }, os.path.join(probe_dir, "regression_probes.pt"))

    # ────── Save predictions (all three probes — for spatial viz + probe projection) ──
    def save_predictions(split_name, split_coords_true, split_fids, split_mstrs):
        emb = embeddings[split_name]
        pred_xyz_lin = predict_xyz(lin_model, emb, normalizer, device)
        pred_xyz_gauss, pred_sigma_gauss = predict_xyz_gaussian(
            gauss_model, emb, normalizer, device,
        )
        pred_xyz_hemi, pred_p_right = predict_xyz_with_hemisphere(
            hemi_model, emb, normalizer, device,
        )
        np.savez_compressed(
            os.path.join(pred_dir, f"{split_name}_predictions.npz"),
            # Linear MSE probe (primary — for existing viz/downstream)
            pred_xyz=pred_xyz_lin.astype(np.float32),
            # Gaussian probe
            pred_xyz_gauss=pred_xyz_gauss.astype(np.float32),
            pred_sigma_gauss=pred_sigma_gauss.astype(np.float32),
            # Hemi-head probe — LR decoded to the predicted hemisphere
            # instead of always left. pred_p_right is the per-sample
            # sigmoid probability (pre-threshold) for downstream analysis.
            pred_xyz_hemi=pred_xyz_hemi.astype(np.float32),
            pred_p_right=pred_p_right.astype(np.float32),
            # Ground truth
            true_xyz=split_coords_true.astype(np.float32),
            fine_id=split_fids.astype(np.int64),
            meta_str=np.array(split_mstrs, dtype="U200"),
        )

    log("\nSaving probe_train predictions (for spatial viz)...")
    save_predictions("probe_train", *meta["probe_train"])

    # ────── Evaluate on 5 test folds ──────
    log(f"\n{'=' * 70}")
    log(f"REGRESSION EVAL — 5 test folds, shared_fine_ids={len(shared_fine_ids)}")
    log(f"{'=' * 70}")

    fold_results = {"volume": {"linear": []}, "centroid": {"linear": []}}
    # Separate fold results for the Gaussian probe (same structure).
    fold_results_gauss = {"volume": {"linear": []}, "centroid": {"linear": []}}
    baseline_fold = {
        "volume": defaultdict(lambda: defaultdict(list)),
        "centroid": defaultdict(lambda: defaultdict(list)),
    }
    axis_errors = {"linear": []}
    axis_errors_gauss = {"linear": []}
    # Hemi probe: per-fold axis errors (after hemi decode, so LR error is
    # absolute-space, not folded) + hemi classification accuracy.
    axis_errors_hemi = []
    hemi_accuracies = []
    hemi_right_frac_test = []
    # For Gaussian calibration: accumulate (residuals, sigmas) across folds.
    calib_residuals, calib_sigmas = [], []
    # Folded-residual collections (MSE probe only; Gaussian already stored in
    # calib_residuals). Used for 3D RMSE / median Euclidean error: the MSE
    # probe decodes LR to left-hem, so raw err has ~2×|true_lr−midline| mirror
    # inflation on right-hem samples. Folding both true and pred to
    # |lr−midline| gives a physically meaningful 3D distance.
    folded_errs_mse = []

    for fold in range(5):
        fold_name = f"test_fold_{fold}"
        if fold_name not in datasets:
            continue

        test_true_coords, test_true_fid, test_mstrs = meta[fold_name]
        test_emb = embeddings[fold_name]

        # Save raw predictions for this fold (both probes)
        save_predictions(fold_name, test_true_coords, test_true_fid, test_mstrs)

        # ── MSE probe ──
        pred_xyz = predict_xyz(lin_model, test_emb, normalizer, device)

        pred_regions_vol = coords_to_region_ids(ba, pred_xyz)
        res_vol = evaluate_predictions(
            pred_regions_vol, test_true_fid, region_ancestors, shared_fine_ids,
        )
        if res_vol is not None:
            fold_results["volume"]["linear"].append(res_vol)

        pred_regions_cen = nearest_centroid_predict(
            pred_xyz, centroid_coords_arr, centroid_fids_arr
        )
        res_cen = evaluate_predictions(
            pred_regions_cen, test_true_fid, region_ancestors, shared_fine_ids,
        )
        if res_cen is not None:
            fold_results["centroid"]["linear"].append(res_cen)

        keep = np.isin(test_true_fid, shared_fine_ids)
        err = pred_xyz[keep] - test_true_coords[keep]
        axis_errors["linear"].append(np.abs(err))

        # Folded residuals (MSE probe): fold both pred and true to
        # |lr - midline| on the LR axis so 3D distance is meaningful.
        true_d_mse = np.abs(test_true_coords[keep, 2] - normalizer.midline)
        pred_d_mse = np.abs(pred_xyz[keep, 2] - normalizer.midline)
        folded_errs_mse.append(np.stack([
            np.abs(pred_xyz[keep, 0] - test_true_coords[keep, 0]),
            np.abs(pred_xyz[keep, 1] - test_true_coords[keep, 1]),
            np.abs(pred_d_mse - true_d_mse),
        ], axis=1))

        # ── Gaussian probe ──
        pred_xyz_g, pred_sigma_g = predict_xyz_gaussian(
            gauss_model, test_emb, normalizer, device,
        )

        pred_regions_vol_g = coords_to_region_ids(ba, pred_xyz_g)
        res_vol_g = evaluate_predictions(
            pred_regions_vol_g, test_true_fid, region_ancestors, shared_fine_ids,
        )
        if res_vol_g is not None:
            fold_results_gauss["volume"]["linear"].append(res_vol_g)

        pred_regions_cen_g = nearest_centroid_predict(
            pred_xyz_g, centroid_coords_arr, centroid_fids_arr
        )
        res_cen_g = evaluate_predictions(
            pred_regions_cen_g, test_true_fid, region_ancestors, shared_fine_ids,
        )
        if res_cen_g is not None:
            fold_results_gauss["centroid"]["linear"].append(res_cen_g)

        err_g = pred_xyz_g[keep] - test_true_coords[keep]
        axis_errors_gauss["linear"].append(np.abs(err_g))

        # Calibration: |residual_axis| vs σ_axis in folded (d) space.
        # axis 0,1 = raw (ap, dv). axis 2 = distance-from-midline (left-hem decode).
        true_d = np.abs(test_true_coords[keep, 2] - normalizer.midline)
        pred_d = np.abs(pred_xyz_g[keep, 2] - normalizer.midline)
        resid = np.stack([
            np.abs(pred_xyz_g[keep, 0] - test_true_coords[keep, 0]),
            np.abs(pred_xyz_g[keep, 1] - test_true_coords[keep, 1]),
            np.abs(pred_d - true_d),
        ], axis=1)
        calib_residuals.append(resid)
        calib_sigmas.append(pred_sigma_g[keep])

        # ── Hemi-head probe ──
        pred_xyz_h, pred_p_right = predict_xyz_with_hemisphere(
            hemi_model, test_emb, normalizer, device,
        )
        # Axis errors in ABSOLUTE-SPACE (LR decoded to predicted hemisphere).
        err_h = pred_xyz_h[keep] - test_true_coords[keep]
        axis_errors_hemi.append(np.abs(err_h))
        # Hemi classification accuracy vs true side.
        true_side_fold = (test_true_coords[keep, 2] > normalizer.midline).astype(np.float32)
        pred_side_fold = (pred_p_right[keep] > 0.5).astype(np.float32)
        hemi_accuracies.append(float((pred_side_fold == true_side_fold).mean()))
        hemi_right_frac_test.append(float(true_side_fold.mean()))

        # Baselines
        n_test = len(test_emb)
        baselines_raw = {
            "random_uniform": baseline_random_uniform(tr_coords_all, n_test, rng),
            "mean_coord":     baseline_mean_coord(tr_coords_all, n_test),
            "spatial_prior":  baseline_spatial_prior(tr_coords_all, n_test, rng),
        }
        for bname, bcoords in baselines_raw.items():
            # volume lookup
            pred_vol = coords_to_region_ids(ba, bcoords)
            res = evaluate_predictions(pred_vol, test_true_fid, region_ancestors, shared_fine_ids)
            if res is not None:
                for k, v in res.items():
                    baseline_fold["volume"][bname][k].append(v["bal_acc"])

            # nearest centroid
            pred_cen = nearest_centroid_predict(bcoords, centroid_coords_arr, centroid_fids_arr)
            res = evaluate_predictions(pred_cen, test_true_fid, region_ancestors, shared_fine_ids)
            if res is not None:
                for k, v in res.items():
                    baseline_fold["centroid"][bname][k].append(v["bal_acc"])

    # ────── Class-counts metadata (from first-fold linear volume result) ──────
    class_counts = {}
    if fold_results["volume"]["linear"]:
        first = fold_results["volume"]["linear"][0]
        for k, v in first.items():
            if k == "fine":
                continue
            n_cls = v.get("n_classes", 0)
            chance = v.get("chance", 1.0 / max(n_cls, 1))
            class_counts[k if isinstance(k, int) else k] = (n_cls, chance)
        # Convert int keys to "d{N}" format
        class_counts_fmt = {}
        for k, v in class_counts.items():
            if isinstance(k, int):
                class_counts_fmt[k] = v
                class_counts_fmt[f"d{k}"] = v
            else:
                class_counts_fmt[k] = v
        class_counts = class_counts_fmt

    # ────── Build the final regression.json ──────
    def _summarize(method_results):
        """Turn method_results[probe][fold] → classification.json-like dict."""
        summary = {}
        keys = ["fine"] + list(range(1, 10)) + [f"{d}_inclusive" for d in range(1, 10)]
        for k in keys:
            if k == "fine":
                n_cls, chance = len(shared_fine_ids), 1.0 / max(len(shared_fine_ids), 1)
            else:
                n_cls, chance = class_counts.get(k, (0, 0))

            entry = {"n_classes": n_cls, "chance": chance}
            folds = method_results.get("linear", [])
            vals = [f.get(k, {}).get("bal_acc") for f in folds if k in f]
            vals = [v for v in vals if v is not None]
            if vals:
                entry["linear"] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
                key_out = "fine" if k == "fine" else (f"d{k}" if isinstance(k, int) else f"d{k}")
                summary[key_out] = entry
        return summary

    summary_volume = _summarize(fold_results["volume"])
    summary_centroid = _summarize(fold_results["centroid"])

    # Gaussian probe summaries
    summary_volume_g = _summarize(fold_results_gauss["volume"])
    summary_centroid_g = _summarize(fold_results_gauss["centroid"])

    def _summarize_baselines(baseline_dict):
        out = {}
        for bname, per_depth in baseline_dict.items():
            entry = {}
            for k, vals in per_depth.items():
                if vals:
                    key_out = "fine" if k == "fine" else f"d{k}"
                    entry[key_out] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                    }
            out[bname] = entry
        return out

    baselines_vol = _summarize_baselines(baseline_fold["volume"])
    baselines_cen = _summarize_baselines(baseline_fold["centroid"])

    def _3d_metrics(folded_abs_err):
        """3D metrics from folded per-axis |residuals| (N, 3) [ap, dv, d]:
          - rmse_3d               = sqrt(E[‖err‖²])  headline metric
          - median_euclidean_3d   = median ‖err‖     robust summary
          - rmse_per_axis         = [rmse_ap, rmse_dv, rmse_d] (sqrt of axis MSE)
        """
        err2 = folded_abs_err.astype(np.float64) ** 2
        per_sample_eucl = np.sqrt(err2.sum(axis=1))
        return {
            "rmse_3d":             float(np.sqrt(err2.sum(axis=1).mean())),
            "median_euclidean_3d": float(np.median(per_sample_eucl)),
            "rmse_per_axis":       np.sqrt(err2.mean(axis=0)).tolist(),
        }

    # µm errors (both probes, linear layer)
    mm_errors = {}
    if axis_errors["linear"]:
        all_err = np.concatenate(axis_errors["linear"], axis=0)
        mm_errors["linear"] = {
            "median": np.median(all_err, axis=0).tolist(),
            "mean":   np.mean(all_err, axis=0).tolist(),
        }
        if folded_errs_mse:
            all_folded = np.concatenate(folded_errs_mse, axis=0)
            mm_errors["linear"].update(_3d_metrics(all_folded))
    mm_errors_gauss = {}
    if axis_errors_gauss["linear"]:
        all_err_g = np.concatenate(axis_errors_gauss["linear"], axis=0)
        mm_errors_gauss["linear"] = {
            "median": np.median(all_err_g, axis=0).tolist(),
            "mean":   np.mean(all_err_g, axis=0).tolist(),
        }
        # Gaussian uses the same folded residual as calibration.
        if calib_residuals:
            all_folded_g = np.concatenate(calib_residuals, axis=0)
            mm_errors_gauss["linear"].update(_3d_metrics(all_folded_g))

    # Gaussian calibration metrics
    calibration = {}
    if calib_residuals:
        res = np.concatenate(calib_residuals, axis=0)   # (N, 3) µm, folded space
        sig = np.concatenate(calib_sigmas, axis=0)      # (N, 3) µm
        # Per-axis calibration + joint (box within k·σ on all axes)
        within_1_per_axis = (res < sig).mean(axis=0).tolist()
        within_2_per_axis = (res < 2 * sig).mean(axis=0).tolist()
        within_1_joint = float((res < sig).all(axis=1).mean())
        within_2_joint = float((res < 2 * sig).all(axis=1).mean())
        # Heteroscedastic NLL (constant term omitted)
        log_var_um = 2.0 * np.log(np.clip(sig, 1e-6, None))
        nll = float(0.5 * (log_var_um + res ** 2 / (sig ** 2 + 1e-12)).sum(axis=1).mean())
        calibration = {
            "within_1sigma_per_axis": within_1_per_axis,   # ideal 68.3% each
            "within_2sigma_per_axis": within_2_per_axis,   # ideal 95.4% each
            "within_1sigma_joint":   within_1_joint,        # ideal 31.8%
            "within_2sigma_joint":   within_2_joint,        # ideal 86.9%
            "nll_um": nll,
            "mean_sigma":   np.mean(sig, axis=0).tolist(),   # [σ_ap, σ_dv, σ_d] in µm
            "median_sigma": np.median(sig, axis=0).tolist(),
        }

    # Top-level = volume_lookup (primary), centroid_method nested alongside
    out_json = dict(summary_volume)
    out_json["um_errors"] = mm_errors
    out_json["baselines"] = baselines_vol
    out_json["normalizer_stats"] = normalizer.stats()
    out_json["shared_fine_ids"] = len(shared_fine_ids)
    out_json["head_source"] = "fine_tuned_model" if args.use_model_head else "frozen_probe"
    out_json["centroid_method"] = {
        **summary_centroid,
        "baselines": baselines_cen,
    }

    # Gaussian probe results (parallel experiment, same linear architecture)
    out_json["gaussian_probe"] = {
        **summary_volume_g,
        "um_errors": mm_errors_gauss,
        "calibration": calibration,
        "centroid_method": summary_centroid_g,
    }

    # Hemi-head probe — absolute-space LR errors (decoded to predicted
    # hemisphere instead of always left). Not evaluated with volume lookup
    # here because the pipeline parallel is already covered by the MSE
    # probe; this section is focused on the µm error comparison.
    hemi_results = {}
    if axis_errors_hemi:
        all_err_h = np.concatenate(axis_errors_hemi, axis=0)
        hemi_results = {
            "description": (
                "Strict linear probe with nn.Linear(768, 3) coord head + "
                "nn.Linear(768, 1) side-of-midline logit. LR decoded as "
                "midline + d × sign(p_right − 0.5), so errors are absolute-"
                "space (not folded)."
            ),
            "hemi_lambda": args.hemi_lambda,
            "val_hemi_acc": hemi_val_metrics["hemi_acc"],
            "val_mse": hemi_val_metrics["mse"],
            "val_bce": hemi_val_metrics["bce"],
            "test_hemi_acc_per_fold": hemi_accuracies,
            "test_hemi_acc_mean": float(np.mean(hemi_accuracies)) if hemi_accuracies else None,
            "test_right_frac_per_fold": hemi_right_frac_test,
            "majority_baseline_acc": (
                max(np.mean(hemi_right_frac_test), 1 - np.mean(hemi_right_frac_test))
                if hemi_right_frac_test else None
            ),
            "um_errors_absolute": {
                "median": np.median(all_err_h, axis=0).tolist(),  # [ap, dv, lr]
                "mean":   np.mean(all_err_h, axis=0).tolist(),
                # Hemi probe already decodes to predicted hemisphere, so
                # err_h is absolute-space; 3D metrics from it directly.
                **_3d_metrics(all_err_h),
            },
        }
    out_json["hemi_probe"] = hemi_results

    # ────── Log summary tables ──────
    def _print_table(title, summary):
        log(f"\n{'=' * 70}")
        log(title)
        log(f"{'=' * 70}")
        log(f"{'Depth':>5} {'Cls':>4} {'Chance':>7} {'Linear':>14}")
        log(f"{'-' * 40}")
        for key in ["fine"] + [f"d{d}" for d in range(1, 10)]:
            if key not in summary:
                continue
            e = summary[key]
            n_cls, chance = e.get("n_classes", "?"), e.get("chance", 0)
            lin = e.get("linear", {})
            lm, ls = lin.get("mean", 0) * 100, lin.get("std", 0) * 100
            label = "fine" if key == "fine" else key[1:]
            log(f"{label:>5} {n_cls:>4} {chance * 100:>6.2f}% {lm:>7.2f}±{ls:.2f}%")

    _print_table("[MSE probe] STRICT — volume lookup", summary_volume)
    _print_table("[MSE probe] STRICT — nearest centroid", summary_centroid)
    _print_table("[Gaussian probe] STRICT — volume lookup", summary_volume_g)
    _print_table("[Gaussian probe] STRICT — nearest centroid", summary_centroid_g)

    if calibration:
        log(f"\n{'=' * 70}")
        log("Gaussian probe calibration")
        log(f"{'=' * 70}")
        log(f"  Mean σ (µm):   ap={calibration['mean_sigma'][0]:.0f}, "
            f"dv={calibration['mean_sigma'][1]:.0f}, d={calibration['mean_sigma'][2]:.0f}")
        log(f"  Median σ (µm): ap={calibration['median_sigma'][0]:.0f}, "
            f"dv={calibration['median_sigma'][1]:.0f}, d={calibration['median_sigma'][2]:.0f}")
        pa = calibration['within_1sigma_per_axis']
        log(f"  Within 1σ per-axis: ap={pa[0]:.1%}, dv={pa[1]:.1%}, d={pa[2]:.1%}  (ideal 68.3%)")
        pa = calibration['within_2sigma_per_axis']
        log(f"  Within 2σ per-axis: ap={pa[0]:.1%}, dv={pa[1]:.1%}, d={pa[2]:.1%}  (ideal 95.4%)")
        log(f"  Within 1σ joint: {calibration['within_1sigma_joint']:.1%}  (ideal 31.8%)")
        log(f"  Within 2σ joint: {calibration['within_2sigma_joint']:.1%}  (ideal 86.9%)")
        log(f"  NLL (µm):        {calibration['nll_um']:.1f}")

    log(f"\n{'=' * 70}")
    log("µm error (pred xyz vs TRUE channel xyz)")
    log(f"{'=' * 70}")
    def _log_err_block(tag, v):
        med = v["median"]
        mean = v["mean"]
        log(f"  {tag}:")
        log(f"    median:        ap={med[0]:.0f}µm, dv={med[1]:.0f}µm, lr={med[2]:.0f}µm")
        log(f"    mean:          ap={mean[0]:.0f}µm, dv={mean[1]:.0f}µm, lr={mean[2]:.0f}µm")
        if "rmse_per_axis" in v:
            rp = v["rmse_per_axis"]
            log(f"    rmse_per_axis: ap={rp[0]:.0f}µm, dv={rp[1]:.0f}µm, d={rp[2]:.0f}µm  (folded LR)")
        if "rmse_3d" in v:
            log(f"    ** 3D RMSE:              {v['rmse_3d']:.0f} µm  (folded)")
            log(f"    ** 3D median Euclidean:  {v['median_euclidean_3d']:.0f} µm  (folded)")

    for p, v in mm_errors.items():
        _log_err_block(f"MSE-probe {p}", v)
    for p, v in mm_errors_gauss.items():
        _log_err_block(f"Gaussian-probe {p}", v)

    if hemi_results:
        log(f"\n{'=' * 70}")
        log("Hemi-head probe (absolute LR decode, not folded)")
        log(f"{'=' * 70}")
        log(f"  Hemi classification accuracy (test, 5-fold): "
            f"{hemi_results['test_hemi_acc_mean']:.4f} "
            f"(majority baseline {hemi_results['majority_baseline_acc']:.4f})")
        u = hemi_results["um_errors_absolute"]
        med_h, mean_h = u["median"], u["mean"]
        log(f"  median err:  ap={med_h[0]:.0f}µm, dv={med_h[1]:.0f}µm, lr={med_h[2]:.0f}µm")
        log(f"  mean err:    ap={mean_h[0]:.0f}µm, dv={mean_h[1]:.0f}µm, lr={mean_h[2]:.0f}µm")
        if "rmse_per_axis" in u:
            rp = u["rmse_per_axis"]
            log(f"  rmse_per_axis: ap={rp[0]:.0f}µm, dv={rp[1]:.0f}µm, lr={rp[2]:.0f}µm  (absolute)")
        if "rmse_3d" in u:
            log(f"  ** 3D RMSE:              {u['rmse_3d']:.0f} µm  (absolute)")
            log(f"  ** 3D median Euclidean:  {u['median_euclidean_3d']:.0f} µm  (absolute)")
        # Direct comparison to MSE probe's folded-space LR
        if mm_errors.get("linear"):
            lr_folded = mm_errors["linear"]["median"][2]
            log(f"  → LR median err: folded-decode={lr_folded:.0f}µm (MSE probe) "
                f"vs absolute-decode={med_h[2]:.0f}µm (hemi probe)")

    log(f"\n{'=' * 70}")
    log("Baselines (fine-level, volume lookup)")
    log(f"{'=' * 70}")
    for bname, bdict in baselines_vol.items():
        fine = bdict.get("fine", {})
        m, s = fine.get("mean", 0) * 100, fine.get("std", 0) * 100
        log(f"  {bname:>16}: {m:>6.2f}±{s:.2f}%")

    log(f"\n{'=' * 70}")
    log("Baselines (fine-level, nearest centroid)")
    log(f"{'=' * 70}")
    for bname, bdict in baselines_cen.items():
        fine = bdict.get("fine", {})
        m, s = fine.get("mean", 0) * 100, fine.get("std", 0) * 100
        log(f"  {bname:>16}: {m:>6.2f}±{s:.2f}%")

    out_path = os.path.join(args.output_dir, "regression.json")
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    log(f"\nSaved {out_path}")
    log(f"Saved raw predictions to {pred_dir}/")


if __name__ == "__main__":
    main()
