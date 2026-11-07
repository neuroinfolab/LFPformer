"""Compute per-fold per-class recall CSVs for Allen 75K (matches IBL format).

For each anchor SSL × FT mode and each baseline (ViT, AnyNet, classical),
loads gaussian_predictions.npz, applies posthoc gauss_wcons_line refinement,
voxelizes via Allen 25µm CCF, and writes per-region recall with fold1..fold5
columns to eval_plots/per_region_balacc_75k/<key>_per_region.csv.

Reuses logic from plot_per_region_recall_anchor_vs_baselines_allen_75k.py
but exports CSVs rather than figures.

Container: w2v2_cuda_cu128_np2.sif (needs nrrd + numpy 2.x for some npz).
"""
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import nrrd
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALPHABRAIN_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DSET_DIR = os.path.join(
    ALPHABRAIN_ROOT, "blind_localization/data/atlas_data/compact_atlas_lp_eval"
)
ATLAS_META_PATH = os.path.join(DSET_DIR, "metadata.pkl")
ANNOTATION_PATH = os.path.join(ALPHABRAIN_ROOT, "data/allen_ccf/annotation_25.nrrd")
STRUCTURE_TREE_PATH = os.path.join(ALPHABRAIN_ROOT, "data/allen_ccf/structure_tree.json")
META_CACHE_FID = os.path.join(SCRIPT_DIR, ".cache_lp_eval_test_meta_fid.npz")
FOLD_BOUNDS_CACHE = os.path.join(SCRIPT_DIR, ".cache_lp_eval_fold_bounds.npz")

NEURIPS_DIR = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026')
BASELINE_DIR = os.path.join(NEURIPS_DIR, "baseline")
VSP_BASE = "/workspace/ws3/BrainSTORM/ray_results/atlas_experiments/2026-04-22"
OUT_CSV_DIR = os.path.join(NEURIPS_DIR, "eval_plots/per_region_balacc_75k")

VOXEL_SIZE = 25.0
POSTHOC_KEY = "gauss_wcons_line"

# What we want CSVs for ─────────────────────────────────────────────────────
# Anchor SSL × FT (only those whose gaussian_predictions.npz exists):
SSL_VARIANTS = [
    {"key": "qv3",  "ssl_dir": "w2v2_75k_quantizer_best_v3"},
    {"key": "pdm",  "ssl_dir": "w2v2_75k_probe_dist_v2_matched"},
    {"key": "pdv4", "ssl_dir": "w2v2_75k_ssl_pd_hebo_v4"},
    {"key": "distc","ssl_dir": "w2v2_75k_distc_best"},
]
FT_MODES = [
    {"key": "fullft", "subdir": "finetune_gaussian_fullft"},
    {"key": "cnnfr",  "subdir": "finetune_gaussian"},
    {"key": "fr",     "subdir": "finetune_gaussian_frozen"},
]
BASELINES = [
    {"key": "histgbm_band",  "npz": os.path.join(BASELINE_DIR, "preds_histgbm_band.npz")},
    {"key": "knn_band",      "npz": os.path.join(BASELINE_DIR, "preds_knn_band.npz")},
    {"key": "knn_pca_psd",   "npz": os.path.join(BASELINE_DIR, "preds_knn_pca_psd.npz")},
    {"key": "pca_ridge_psd", "npz": os.path.join(BASELINE_DIR, "preds_pca_ridge_psd.npz")},
    {"key": "vit",           "npz": f"{VSP_BASE}/vit_rival_xyz_hp_allen/gaussian_predictions.npz"},
    {"key": "anynet",        "npz": f"{VSP_BASE}/anynet_rival_xyz_hp_allen/gaussian_predictions.npz"},
]


# ── meta cache + fold boundaries ─────────────────────────────────────────
def build_or_load_meta_cache():
    if os.path.isfile(META_CACHE_FID):
        z = np.load(META_CACHE_FID, allow_pickle=False)
        return z["fine_ids"], z["probes"], z["channels"], z["coords"]
    print(f"[*] building meta cache → {META_CACHE_FID}", flush=True)
    fine_ids, probes, channels, coords = [], [], [], []
    for fold in range(1, 6):
        sd = os.path.join(DSET_DIR, f"test_fold{fold}")
        for f in sorted(os.listdir(sd)):
            if not f.startswith("samples_"):
                continue
            with open(os.path.join(sd, f), "rb") as fh:
                samples = pickle.load(fh)
            for s in samples:
                parts = str(s[1]).split("_")
                fine_ids.append(int(s[2]["fine_id"]))
                probes.append(parts[2])
                channels.append(parts[3])
                coords.append(np.asarray(s[3], dtype=np.float32))
    fine_ids = np.array(fine_ids, dtype=np.int64)
    probes = np.array(probes); channels = np.array(channels)
    coords = np.stack(coords).astype(np.float32)
    np.savez(META_CACHE_FID, fine_ids=fine_ids, probes=probes,
             channels=channels, coords=coords)
    return fine_ids, probes, channels, coords


def build_or_load_fold_bounds():
    if os.path.isfile(FOLD_BOUNDS_CACHE):
        return np.load(FOLD_BOUNDS_CACHE)["bounds"]
    print(f"[*] computing fold sample boundaries …", flush=True)
    sizes = []
    for fold in range(1, 6):
        sd = os.path.join(DSET_DIR, f"test_fold{fold}")
        n = 0
        for f in sorted(os.listdir(sd)):
            if not f.startswith("samples_"):
                continue
            with open(os.path.join(sd, f), "rb") as fh:
                n += len(pickle.load(fh))
        sizes.append(n)
    bounds = np.cumsum([0] + sizes).astype(np.int64)  # length 6
    np.savez(FOLD_BOUNDS_CACHE, bounds=bounds)
    return bounds


# ── atlas voxelization (sid-indexed 116-class) ────────────────────────────
def build_label_volume(canonical_sids):
    print("[*] loading CCF annotation …", flush=True)
    ann, _ = nrrd.read(ANNOTATION_PATH)
    tree = json.load(open(STRUCTURE_TREE_PATH))
    id2parent = {}
    for s in tree:
        path = [int(x) for x in s.get("structure_id_path", "").strip("/").split("/") if x]
        if len(path) >= 2:
            id2parent[s["id"]] = path[-2]
    sid_to_idx = {int(sid): i for i, sid in enumerate(canonical_sids)}
    cache = {0: -1}

    def resolve(sid):
        if sid in cache:
            return cache[sid]
        cur = int(sid); out = -1; seen = set()
        while cur and cur not in seen:
            if cur in sid_to_idx:
                out = sid_to_idx[cur]; break
            seen.add(cur)
            cur = id2parent.get(cur, 0)
        cache[sid] = out
        return out

    max_sid = int(ann.max())
    lookup = np.full(max_sid + 1, -1, dtype=np.int32)
    for s in np.unique(ann):
        lookup[int(s)] = resolve(int(s))
    return lookup[ann]


def voxelize(xyz):
    return np.floor(xyz / VOXEL_SIZE).astype(np.int64)


def label_at_voxel(vox, label_volume):
    shape = np.array(label_volume.shape)
    in_b = np.all((vox >= 0) & (vox < shape), axis=1)
    out = np.full(len(vox), -1, dtype=np.int64)
    if in_b.any():
        v = vox[in_b]
        out[in_b] = label_volume[v[:, 0], v[:, 1], v[:, 2]]
    return out


# ── posthoc ───────────────────────────────────────────────────────────────
def _group_indices(probes, channels):
    g = {}
    for i, key in enumerate(zip(probes, channels)):
        g.setdefault(key, []).append(i)
    return g


def refine_weighted_consensus(pred, stds, probes, channels):
    refined = pred.copy()
    for idxs in _group_indices(probes, channels).values():
        idx = np.asarray(idxs)
        p = pred[idx]; s = stds[idx]
        w = 1.0 / (s ** 2 + 1e-6)
        w_norm = w / w.sum(axis=0, keepdims=True)
        refined[idx] = (p * w_norm).sum(axis=0)
    return refined


def refine_line_projection(pred, probes, channels):
    key_to_idx = _group_indices(probes, channels)
    probe_to = {}
    for (p, c), idxs in key_to_idx.items():
        cent = pred[idxs[0]]
        probe_to.setdefault(p, {})[c] = (cent, np.asarray(idxs))
    refined = pred.copy()
    for pid, chans in probe_to.items():
        cents = np.array([v[0] for v in chans.values()])
        if len(cents) < 3:
            for ch, (cent, idxs) in chans.items():
                refined[idxs] = cent
            continue
        pca = PCA(n_components=1)
        pca.fit(cents)
        center = cents.mean(axis=0)
        direction = pca.components_[0]
        for ch, (cent, idxs) in chans.items():
            t = float(np.dot(cent - center, direction))
            refined[idxs] = center + t * direction
    return refined


def apply_posthoc(pred, stds, probes, channels):
    g = refine_weighted_consensus(pred, stds, probes, channels)
    return refine_line_projection(g, probes, channels)


# ── per-fold per-class recall ─────────────────────────────────────────────
def per_fold_per_region_recall(true_lbl, pred_lbl, fold_bounds, n_regions):
    """Return (n_regions, n_folds) recall matrix; NaN if class absent in fold."""
    n_folds = len(fold_bounds) - 1
    out = np.full((n_regions, n_folds), np.nan, dtype=np.float64)
    for k in range(n_folds):
        a, b = fold_bounds[k], fold_bounds[k + 1]
        tl = true_lbl[a:b]; pl = pred_lbl[a:b]
        valid = tl >= 0
        for c in range(n_regions):
            m = valid & (tl == c)
            n = int(m.sum())
            if n > 0:
                out[c, k] = float((pl[m] == c).sum()) / n
    return out


def compute_one(npz_path, key, coords_cache, probes, channels, fold_bounds,
                label_volume, canonical_sids_acrs):
    if not os.path.isfile(npz_path):
        print(f"[!] missing: {npz_path}")
        return None
    print(f"[*] {key}: loading {npz_path}", flush=True)
    d = np.load(npz_path, allow_pickle=True)
    # Different sources use different key names
    if "preds" in d:
        pred = d["preds"].astype(np.float32)
        true = d["true"].astype(np.float32)
        stds = d["stds"].astype(np.float32)
    elif "pred_xyz" in d:
        pred = d["pred_xyz"].astype(np.float32)
        true = d["true_xyz"].astype(np.float32)
        stds = d["pred_std"].astype(np.float32) if "pred_std" in d else \
               np.ones_like(pred, dtype=np.float32)
    else:
        raise ValueError(f"{key}: unknown npz schema, keys={list(d.files)}")
    # Verify alignment
    if pred.shape != coords_cache.shape:
        raise SystemExit(f"{key}: shape mismatch {pred.shape} vs {coords_cache.shape}")
    diff = np.abs(coords_cache - true).max(axis=1)
    bad = int((diff > 1.0).sum())
    if bad > 0:
        raise SystemExit(f"{key}: meta order mismatch — {bad} samples differ > 1µm")

    pred_post = apply_posthoc(pred, stds, probes, channels)
    true_lbl = label_at_voxel(voxelize(true), label_volume)
    pred_lbl = label_at_voxel(voxelize(pred_post), label_volume)
    n_regions = len(canonical_sids_acrs)
    return per_fold_per_region_recall(true_lbl, pred_lbl, fold_bounds, n_regions)


def write_csv(out_path, mat, region_acrs, region_sids):
    """mat: (n_regions, n_folds=5)."""
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "sid", "mean_recall", "std_recall",
                    "fold1", "fold2", "fold3", "fold4", "fold5"])
        for i, (acr, sid) in enumerate(zip(region_acrs, region_sids)):
            row = mat[i]
            mn = float(np.nanmean(row)) if not np.all(np.isnan(row)) else 0.0
            sd = float(np.nanstd(row))  if not np.all(np.isnan(row)) else 0.0
            vals = [f"{v:.6f}" if not np.isnan(v) else "0.000000" for v in row]
            w.writerow([acr, int(sid), f"{mn:.6f}", f"{sd:.6f}", *vals])
    print(f"[+] wrote {out_path}")


def main():
    t0 = time.time()
    fine_ids, probes, channels, coords_cache = build_or_load_meta_cache()
    fold_bounds = build_or_load_fold_bounds()
    print(f"[*] N={len(fine_ids):,}  folds bounds={fold_bounds.tolist()}")

    # canonical sids = lp_eval atlas labels (sid-sorted)
    meta = pickle.load(open(ATLAS_META_PATH, "rb"))
    fid_to_acr = meta.get("fine_id_to_acr") or meta.get("fine_id_to_acronym")
    canonical_fids = sorted(set(fine_ids.tolist()))
    region_acrs = [fid_to_acr.get(int(f), str(int(f))) for f in canonical_fids]
    label_volume = build_label_volume(canonical_fids)
    n_regions = len(canonical_fids)
    print(f"[*] {n_regions} canonical Allen regions in lp_eval")

    Path(OUT_CSV_DIR).mkdir(parents=True, exist_ok=True)

    # ── SSL anchors ──────────────────────────────────────────────────────
    for ssl in SSL_VARIANTS:
        for ft in FT_MODES:
            npz = os.path.join(NEURIPS_DIR, ssl["ssl_dir"], ft["subdir"],
                               "gaussian_predictions.npz")
            key = f"anchor_{ssl['key']}_{ft['key']}"
            mat = compute_one(npz, key, coords_cache, probes, channels,
                              fold_bounds, label_volume, region_acrs)
            if mat is not None:
                write_csv(os.path.join(OUT_CSV_DIR, f"{key}_per_region.csv"),
                          mat, region_acrs, canonical_fids)

    # ── Baselines ────────────────────────────────────────────────────────
    for b in BASELINES:
        mat = compute_one(b["npz"], b["key"], coords_cache, probes, channels,
                          fold_bounds, label_volume, region_acrs)
        if mat is not None:
            write_csv(os.path.join(OUT_CSV_DIR, f"{b['key']}_per_region.csv"),
                      mat, region_acrs, canonical_fids)

    print(f"[+] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
