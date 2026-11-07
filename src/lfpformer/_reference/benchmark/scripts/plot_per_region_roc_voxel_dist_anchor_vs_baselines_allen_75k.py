"""Allen lp_eval — Voxel-distance ROC (anchor SSL × FT mode vs baselines).

Replaces the centroid-Gaussian scoring with **distance to nearest atlas voxel
of the class**:

    score[i, c] = − min_{v ∈ region_c} ‖v − pred_xyz[i]‖   (µm)

If pred_xyz lands inside region C → score = 0 (max). Outside → −distance.
This respects region shape (no point-region assumption), matches the recall
plot's atlas-lookup convention, and gives perfect score for "lookup says
you're in region C".

Per-class ROC (one-vs-rest), aggregated to macro / weighted / micro AUC.
9 PNG panels (one per anchor) with anchor + 6 baseline ROC curves overlaid.

Implementation: build a `cKDTree` per class once (over its atlas voxel
coordinates in µm), then for each model `tree.query(pred_xyz, k=1)`. Per-class
ROC then proceeds standardly.

Outputs:
    ray_results/Neurips_2026/neurips_2026_plots/allen_75k_roc_voxel_panels/
        roc_voxeldist_<ssl>_<ft>_vs_baselines_75k_allen.png

Container: w2v2_cuda_cu128_ibl.sif (iblatlas + nrrd + scipy.spatial).
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

import nrrd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALPHABRAIN_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DSET_DIR = os.path.join(
    ALPHABRAIN_ROOT, "blind_localization/data/atlas_data/compact_atlas_lp_eval"
)
ATLAS_META_PATH = os.path.join(DSET_DIR, "metadata.pkl")
ANNOTATION_PATH = os.path.join(ALPHABRAIN_ROOT, "data/allen_ccf/annotation_25.nrrd")
STRUCTURE_TREE_PATH = os.path.join(ALPHABRAIN_ROOT, "data/allen_ccf/structure_tree.json")
META_CACHE_FID = os.path.join(SCRIPT_DIR, ".cache_lp_eval_test_meta_fid.npz")

NEURIPS_DIR = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026')
BASELINE_DIR = os.path.join(NEURIPS_DIR, "baseline")

VOXEL_SIZE = 25.0

SSL_VARIANTS = [
    {"key": "qv3",   "ssl_dir": "w2v2_75k_quantizer_best_v3",
     "label": "SSL-masked"},
    {"key": "pdm",   "ssl_dir": "w2v2_75k_probe_dist_v2_matched",
     "label": "within-probe channel dist-contrastive"},
    {"key": "distc", "ssl_dir": "w2v2_75k_distc_best",
     "label": "CCF distance-contrastive"},
    {"key": "pdv4",  "ssl_dir": "w2v2_75k_ssl_pd_hebo_v4",
     "label": "MaskPred + ChanDistc"},
]
FT_MODES = [
    {"key": "fullft", "subdir": "finetune_gaussian_fullft", "label": "Full FT"},
    {"key": "cnnfr",  "subdir": "finetune_gaussian",         "label": "CNN-frozen"},
    {"key": "fr",     "subdir": "finetune_gaussian_frozen",  "label": "Fully frozen"},
]
VSP_BASE = "/workspace/ws3/BrainSTORM/ray_results/atlas_experiments/2026-04-22"

BASELINES = [
    {"key": "histgbm_band",  "label": "HistGBM / band power",  "color": "#FB5607",
     "source": "classical"},
    {"key": "knn_band",      "label": "kNN-5 / band power",    "color": "#8338EC",
     "source": "classical"},
    {"key": "knn_pca_psd",   "label": "kNN-5 / PCA(50) PSD",   "color": "#FFBE0B",
     "source": "classical"},
    {"key": "pca_ridge_psd", "label": "PCA+Ridge / PSD",       "color": "#06A77D",
     "source": "classical"},
    {"key": "vit",           "label": "ViT",                   "color": "#1F78B4",
     "source": "ws3",
     "npz": f"{VSP_BASE}/vit_rival_xyz_hp_allen/gaussian_predictions.npz"},
    {"key": "anynet",        "label": "AnyNet",                "color": "#33A02C",
     "source": "ws3",
     "npz": f"{VSP_BASE}/anynet_rival_xyz_hp_allen/gaussian_predictions.npz"},
]
ANCHOR_COLOR = "#000000"


# ── Per-sample meta cache ────────────────────────────────────────────────
def build_or_load_meta_cache(cache_path=META_CACHE_FID):
    if os.path.isfile(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        return z["fine_ids"], z["probes"], z["channels"], z["coords"]
    raise SystemExit(f"meta cache missing: {cache_path}")


def verify_alignment(coords_cache, true_xyz, model_key, tol_um=1.0):
    if coords_cache.shape != true_xyz.shape:
        raise SystemExit(f"{model_key}: shape {coords_cache.shape} vs {true_xyz.shape}")
    diff = np.abs(coords_cache - true_xyz).max(axis=1)
    bad = int((diff > tol_um).sum())
    if bad > 0:
        raise SystemExit(
            f"{model_key}: meta/coord order mismatch — {bad}/{len(diff)} "
            f"samples differ by > {tol_um}µm. max={diff.max():.2f}µm")


# ── Atlas: 116-class label volume ────────────────────────────────────────
def build_label_volume_sid_indexed(canonical_sids):
    print("[*] loading CCF annotation …", flush=True)
    ann, _ = nrrd.read(ANNOTATION_PATH)
    print(f"    annotation shape: {ann.shape}", flush=True)
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
        cur = int(sid)
        out = -1
        seen = set()
        while cur and cur not in seen:
            if cur in sid_to_idx:
                out = sid_to_idx[cur]
                break
            seen.add(cur)
            cur = id2parent.get(cur, 0)
        cache[sid] = out
        return out

    max_sid = int(ann.max())
    lookup = np.full(max_sid + 1, -1, dtype=np.int32)
    for s in np.unique(ann):
        lookup[int(s)] = resolve(int(s))
    label_volume = lookup[ann]
    return label_volume


def voxelize_clip(pred_xyz, shape):
    """Voxelize µm coords; clip to in-bounds indices."""
    vox = np.floor(pred_xyz / VOXEL_SIZE).astype(np.int64)
    vox[:, 0] = np.clip(vox[:, 0], 0, shape[0] - 1)
    vox[:, 1] = np.clip(vox[:, 1], 0, shape[1] - 1)
    vox[:, 2] = np.clip(vox[:, 2], 0, shape[2] - 1)
    return vox


# ── Posthoc refinement ───────────────────────────────────────────────────
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


def apply_posthoc(pred, stds, probes, channels, key):
    if key == "raw":
        return pred
    g = refine_weighted_consensus(pred, stds, probes, channels)
    if key == "gauss_weighted":
        return g
    if key == "gauss_wcons_line":
        return refine_line_projection(g, probes, channels)
    raise ValueError(f"unknown posthoc key: {key}")


# ── Streaming EDT-based ROC across all models ────────────────────────────
def load_model_pred_voxels(npz_path, coords_cache, probes, channels, model_key,
                           posthoc_key, label_shape):
    d = np.load(npz_path)
    pred = d["preds"].astype(np.float32)
    true = d["true"].astype(np.float32)
    stds = d["stds"].astype(np.float32)
    verify_alignment(coords_cache, true, model_key)
    pred_post = apply_posthoc(pred, stds, probes, channels, posthoc_key)
    return voxelize_clip(pred_post, label_shape)


def compute_all_rocs_streaming(label_volume, n_regions, model_pred_voxels,
                               true_lbl, fpr_grid):
    """Loop over classes; for each class compute EDT once, then accumulate
    per-class ROC contributions across all models. Returns dict:
    model_key -> {mean_tpr, macro_auc, weighted_auc, n_present}."""
    model_keys = list(model_pred_voxels.keys())
    M = len(model_keys)
    auc_per_class  = {k: np.full(n_regions, np.nan) for k in model_keys}
    tpr_per_class  = {k: np.full((n_regions, len(fpr_grid)), np.nan)
                      for k in model_keys}
    counts_per_cls = np.zeros(n_regions, dtype=np.int64)
    print(f"[*] streaming EDT-ROC over {n_regions} classes × {M} models…",
          flush=True)
    t_total = time.time()
    for c in range(n_regions):
        mask_in = (label_volume == c)
        n_vox = int(mask_in.sum())
        if n_vox == 0:
            continue
        yt = (true_lbl == c)
        n_pos = int(yt.sum())
        counts_per_cls[c] = n_pos
        if n_pos == 0 or n_pos == len(yt):
            continue
        t0 = time.time()
        edt = distance_transform_edt(~mask_in, sampling=VOXEL_SIZE)
        t_edt = time.time() - t0
        t1 = time.time()
        for k, vox in model_pred_voxels.items():
            d_at_pred = edt[vox[:, 0], vox[:, 1], vox[:, 2]]
            ys = -d_at_pred.astype(np.float32)
            fpr, tpr, _ = roc_curve(yt, ys)
            tpr_per_class[k][c] = np.interp(fpr_grid, fpr, tpr)
            auc_per_class[k][c] = float(roc_auc_score(yt, ys))
        t_roc = time.time() - t1
        if c < 3 or c % 10 == 0 or c == n_regions - 1:
            print(f"    class {c:3d}/{n_regions}  vox={n_vox:>6,d}  pos={n_pos:>5,d}  "
                  f"edt={t_edt:.2f}s  roc×{M}={t_roc:.2f}s  "
                  f"max_dist={float(edt.max()):.0f}µm",
                  flush=True)
    print(f"[*] streaming done in {time.time()-t_total:.1f}s", flush=True)

    # Aggregate per model
    out = {}
    for k in model_keys:
        a = auc_per_class[k]
        valid = np.isfinite(a)
        if valid.any():
            macro = float(a[valid].mean())
            w = counts_per_cls[valid].astype(np.float64)
            w = w / max(w.sum(), 1.0)
            weighted = float((a[valid] * w).sum())
            mean_tpr = np.nanmean(tpr_per_class[k][valid], axis=0)
            n_present = int(valid.sum())
        else:
            macro = weighted = float("nan")
            mean_tpr = np.full_like(fpr_grid, np.nan)
            n_present = 0
        out[k] = {"macro_auc": macro, "weighted_auc": weighted,
                  "mean_tpr": mean_tpr, "n_present": n_present,
                  "micro_auc": float("nan"), "micro_curve": (None, None)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",
                    default=os.path.join(NEURIPS_DIR,
                                         "neurips_2026_plots",
                                         "allen_75k_roc_voxel_panels"))
    ap.add_argument("--scale_label", default="75k")
    ap.add_argument("--posthoc_key",
                    choices=["gauss_weighted", "gauss_wcons_line", "raw"],
                    default="gauss_wcons_line")
    ap.add_argument("--anchors", default="all",
                    help="comma-separated 'ssl_ft' keys or 'all' for all 9")
    ap.add_argument("--curve", choices=["macro", "micro"], default="macro",
                    help="which ROC curve to plot (AUCs reported regardless)")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Load metadata ────────────────────────────────────────────────────
    print("[*] Loading lp_eval metadata + per-sample cache…", flush=True)
    meta = pickle.load(open(ATLAS_META_PATH, "rb"))
    fid_to_acr = {int(k): str(v) for k, v in meta["fine_id_to_acr"].items()}
    canonical_sids = sorted(fid_to_acr.keys())
    n_regions = len(canonical_sids)

    fine_ids, probes, channels, coords_cache = build_or_load_meta_cache()
    print(f"    cache rows: {len(fine_ids):,}  n_regions={n_regions}", flush=True)

    label_volume = build_label_volume_sid_indexed(canonical_sids)

    sid_to_idx = {int(sid): i for i, sid in enumerate(canonical_sids)}
    true_lbl = np.array([sid_to_idx.get(int(f), -1) for f in fine_ids],
                        dtype=np.int64)

    fpr_grid = np.linspace(0.0, 1.0, 401)

    if args.anchors == "all":
        anchor_pairs = [(s["key"], f["key"]) for s in SSL_VARIANTS for f in FT_MODES]
    else:
        anchor_pairs = []
        for tok in args.anchors.split(","):
            tok = tok.strip()
            ssl, ft = tok.split("_", 1) if "_" in tok else (None, None)
            if ssl is None or ssl not in {s["key"] for s in SSL_VARIANTS} \
                    or ft not in {f["key"] for f in FT_MODES}:
                raise SystemExit(f"bad anchor token: {tok!r}")
            anchor_pairs.append((ssl, ft))

    # ── Load + voxelize all model predictions up front ───────────────────
    print(f"[*] Voxelizing predictions for all models (posthoc={args.posthoc_key})…",
          flush=True)
    pred_voxels = {}
    base_keys = []
    for b in BASELINES:
        if b["source"] == "ws3":
            npz = b["npz"]
        else:
            npz = os.path.join(BASELINE_DIR, f"preds_{b['key']}.npz")
        if not os.path.isfile(npz):
            print(f"[!] missing baseline {npz} — skipping {b['key']}", flush=True)
            continue
        key = f"baseline_{b['key']}"
        pred_voxels[key] = load_model_pred_voxels(
            npz, coords_cache, probes, channels, b["key"],
            args.posthoc_key, label_volume.shape)
        base_keys.append((b, key))
        print(f"    [baseline] {b['key']}", flush=True)

    anchor_keys = []
    for (ssl_key, ft_key) in anchor_pairs:
        ssl = next(s for s in SSL_VARIANTS if s["key"] == ssl_key)
        ft  = next(f for f in FT_MODES     if f["key"] == ft_key)
        npz = os.path.join(NEURIPS_DIR, ssl["ssl_dir"], ft["subdir"],
                           "gaussian_predictions.npz")
        if not os.path.isfile(npz):
            print(f"[!] missing {npz} — skipping anchor {ssl_key}/{ft_key}",
                  flush=True)
            continue
        key = f"anchor_{ssl_key}_{ft_key}"
        pred_voxels[key] = load_model_pred_voxels(
            npz, coords_cache, probes, channels, f"{ssl_key}_{ft_key}",
            args.posthoc_key, label_volume.shape)
        anchor_keys.append((ssl_key, ft_key, ssl, ft, key))
        print(f"    [anchor] {ssl_key}/{ft_key}", flush=True)

    # ── Streaming per-class ROC for all models in lockstep ───────────────
    all_results = compute_all_rocs_streaming(
        label_volume, n_regions, pred_voxels, true_lbl, fpr_grid)
    base_results = {b["key"]: all_results.get(k) for b, k in base_keys}

    # ── Cache npz for compact replots ────────────────────────────────────
    cache_path = os.path.join(
        args.out_dir,
        f"roc_voxeldist_cache_allen_{args.scale_label}_{args.posthoc_key}.npz")
    cache_keys, cache_tprs, cache_aucs, cache_meta = [], [], [], {}
    for b, k in base_keys:
        r = all_results.get(k)
        if r is None:
            continue
        cache_keys.append(b["key"])
        cache_tprs.append(r["mean_tpr"])
        cache_aucs.append(r["macro_auc"])
        cache_meta[b["key"]] = {"label": b["label"], "color": b["color"]}
    for (ssl_key, ft_key, ssl, ft, akey) in anchor_keys:
        r = all_results.get(akey)
        if r is None:
            continue
        anc_k = f"anchor_{ssl_key}_{ft_key}"
        cache_keys.append(anc_k)
        cache_tprs.append(r["mean_tpr"])
        cache_aucs.append(r["macro_auc"])
        cache_meta[anc_k] = {"label": f"{ssl['label']} · {ft['label']}",
                             "color": ANCHOR_COLOR}
    np.savez_compressed(
        cache_path,
        keys=np.array(cache_keys, dtype=object),
        mean_tpr=np.stack(cache_tprs, axis=0),
        macro_auc=np.array(cache_aucs, dtype=np.float64),
        fpr_grid=fpr_grid,
        model_meta=json.dumps(cache_meta),
        n_regions=n_regions,
        posthoc_key=args.posthoc_key,
        scale_label=args.scale_label,
    )
    print(f"[+] wrote cache → {cache_path}", flush=True)

    # ── Render one figure per anchor ─────────────────────────────────────
    for (ssl_key, ft_key, ssl, ft, akey) in anchor_keys:
        anchor_res = all_results[akey]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1.0,
                alpha=0.8, zorder=2, label="chance")

        def fmt_label(name, r):
            parts = [f"macro={r['macro_auc']:.3f}",
                     f"weighted={r['weighted_auc']:.3f}"]
            if np.isfinite(r["micro_auc"]):
                parts.append(f"micro={r['micro_auc']:.3f}")
            return f"{name}  ({', '.join(parts)})"

        for b in BASELINES:
            r = base_results.get(b["key"])
            if r is None:
                continue
            if args.curve == "macro":
                ax.plot(fpr_grid, r["mean_tpr"], color=b["color"], linewidth=1.6,
                        alpha=0.92, zorder=4, label=fmt_label(b["label"], r))
            else:
                fpr_m, tpr_m = r["micro_curve"]
                ax.plot(fpr_m, tpr_m, color=b["color"], linewidth=1.6,
                        alpha=0.92, zorder=4, label=fmt_label(b["label"], r))

        anchor_label = f"{ssl['label']} · {ft['label']}"
        if args.curve == "macro":
            ax.plot(fpr_grid, anchor_res["mean_tpr"], color=ANCHOR_COLOR,
                    linewidth=2.6, alpha=1.0, zorder=6,
                    label=fmt_label(anchor_label, anchor_res))
        else:
            fpr_m, tpr_m = anchor_res["micro_curve"]
            ax.plot(fpr_m, tpr_m, color=ANCHOR_COLOR, linewidth=2.6, alpha=1.0,
                    zorder=6, label=fmt_label(anchor_label, anchor_res))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("False positive rate", fontsize=10)
        ax.set_ylabel("True positive rate", fontsize=10)
        ax.set_title(
            f"Allen lp_eval — Voxel-distance ROC ({args.curve}-averaged)\n"
            f"score = − dist(pred, nearest class voxel)   "
            f"anchor: w2v2 {args.scale_label} · {anchor_label}  "
            f"(posthoc={args.posthoc_key})",
            fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.92)

        out = os.path.join(
            args.out_dir,
            f"roc_voxeldist_{args.curve}_{ssl_key}_{ft_key}_vs_baselines_"
            f"{args.scale_label}_allen.png")
        fig.tight_layout()
        fig.savefig(out, dpi=240, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"[+] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
