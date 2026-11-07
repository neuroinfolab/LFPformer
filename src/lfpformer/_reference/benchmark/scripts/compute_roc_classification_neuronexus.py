"""NeuroNexus 5-class classification ROC (one-vs-rest, macro).

For each SSL variant, load embeddings, apply the same 3-fold session split
as knn_neuronexus_cv.py, fit a KNN classifier, take predict_proba as
per-class score, concat test predictions across folds, compute one-vs-rest
ROC + macro-average, and save a cache npz with the same schema as the
Allen ROC cache so the compact 1×3 plot can read it.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize


REGION_NAMES = {0: "Cortex", 1: "CA1", 2: "CA2", 3: "CA3", 4: "DG"}

CROSS_SPECIES_DIR = (__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026/neurips_2026_plots/cross_species_eval_75kcd'))

# External proba caches (zero-shot regime, matched to SSL variant kNN protocol)
EXTERNAL_PROBA = [
    {"key": "vit",    "label": "ViT",    "color": "#1F78B4",
     "path": f"{CROSS_SPECIES_DIR}/vit_zero_shot_ibl500k/proba_cache.npz"},
    {"key": "anynet", "label": "AnyNet", "color": "#33A02C",
     "path": f"{CROSS_SPECIES_DIR}/anynet_zero_shot_ibl500k/proba_cache.npz"},
    {"key": "histgbm_band", "label": "HistGBM/band", "color": "#FB5607",
     "path": f"{CROSS_SPECIES_DIR}/nn_classical_histgbm_band_ibl75kcd/proba_cache.npz"},
    {"key": "knn_band",     "label": "kNN/band",     "color": "#8338EC",
     "path": f"{CROSS_SPECIES_DIR}/nn_classical_kNN_band_ibl75kcd/proba_cache.npz"},
    {"key": "knn_pca_psd",  "label": "kNN/PCA-PSD",  "color": "#FFBE0B",
     "path": f"{CROSS_SPECIES_DIR}/nn_classical_kNN_pca_psd_ibl75kcd/proba_cache.npz"},
    {"key": "ridge_psd",    "label": "Ridge/PSD",    "color": "#06A77D",
     "path": f"{CROSS_SPECIES_DIR}/nn_classical_ridge_psd_ibl75kcd/proba_cache.npz"},
]
TRAIN_POOL = ["AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_03"]
ROTATING = ["AD_HF01_1", "NN_syn_01", "NN_syn_02"]
FOLDS = []
for i, test_sess in enumerate(ROTATING):
    val_sess = [s for s in ROTATING if s != test_sess]
    FOLDS.append({"fold": i + 1, "train": TRAIN_POOL,
                  "val": val_sess, "test": [test_sess]})

NEURIPS_DIR = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026')
NN_ROOT = os.path.join(NEURIPS_DIR, "neuronexus_zero_shot")

VARIANTS = [
    {"key": "distc",  "subdir": "variant_distc",
     "label": "CCFDistc",  "color": "#882255"},
    {"key": "qv3",    "subdir": "variant_qv3",
     "label": "MaskPred", "color": "#1F78B4"},
    {"key": "pdm",    "subdir": "variant_pdm",
     "label": "ChanDistc", "color": "#33A02C"},
    {"key": "ibl_cd", "subdir": "variant_ibl_chandist_s500_240k",
     "label": "IBL ChanDistc 240k", "color": "#FB5607"},
]
KNN_K = 10
CLASSES = list(REGION_NAMES.keys())


def load_emb(path):
    d = np.load(path, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    lab = d["region_label"].astype(np.int32)
    sess = np.asarray(d["session"]).astype(str)
    keep = lab != 5
    return emb[keep], lab[keep], sess[keep]


def fold_proba(emb, lab, sess, k=KNN_K):
    """Concat per-fold test predict_proba in a fixed class order."""
    score_chunks = []
    label_chunks = []
    for fd in FOLDS:
        tr = np.isin(sess, fd["train"])
        te = np.isin(sess, fd["test"])
        X_tr = normalize(emb[tr], norm="l2", axis=1)
        X_te = normalize(emb[te], norm="l2", axis=1)
        y_tr = lab[tr]
        y_te = lab[te]
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, metric="euclidean")
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        # remap to full 5-class order in case a class is missing
        full = np.zeros((proba.shape[0], len(CLASSES)), dtype=np.float32)
        for j, cl in enumerate(clf.classes_):
            if cl in REGION_NAMES:
                full[:, CLASSES.index(int(cl))] = proba[:, j]
        score_chunks.append(full)
        label_chunks.append(y_te)
    return np.concatenate(score_chunks, axis=0), np.concatenate(label_chunks, axis=0)


def macro_roc(y, scores, fpr_grid):
    """Macro-average one-vs-rest ROC.  scores: (N, C). y: (N,) class ids."""
    tpr_acc = np.zeros_like(fpr_grid, dtype=np.float64)
    aucs = []
    n_present = 0
    for c in CLASSES:
        yt = (y == c).astype(np.int32)
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        ys = scores[:, CLASSES.index(c)]
        fpr, tpr, _ = roc_curve(yt, ys)
        tpr_acc += np.interp(fpr_grid, fpr, tpr)
        aucs.append(roc_auc_score(yt, ys))
        n_present += 1
    if n_present == 0:
        return np.full_like(fpr_grid, np.nan), float("nan")
    return tpr_acc / n_present, float(np.mean(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path",
                    default=os.path.join(NEURIPS_DIR, "neurips_2026_plots",
                                         "may4",
                                         "roc_classification_cache_neuronexus.npz"))
    ap.add_argument("--k", type=int, default=KNN_K)
    args = ap.parse_args()

    fpr_grid = np.linspace(0.0, 1.0, 401)
    keys = []
    tprs = []
    aucs = []
    meta = {}

    # ── w2v2 full-FT anchor (LOO-CV concat across all 7 sessions) ────────
    fullft_dir = os.path.join(CROSS_SPECIES_DIR,
                              "nn_full_finetune_ibl75kcd", "predictions")
    pred_files = sorted(f for f in os.listdir(fullft_dir) if f.endswith(".npz"))
    fullft_logits = []
    fullft_y = []
    for pf in pred_files:
        d = np.load(os.path.join(fullft_dir, pf), allow_pickle=True)
        fullft_logits.append(d["logits"].astype(np.float32))
        fullft_y.append(d["labels"].astype(np.int64))
    logits = np.concatenate(fullft_logits, axis=0)
    y_full = np.concatenate(fullft_y, axis=0)
    # softmax → proba
    z = logits - logits.max(axis=1, keepdims=True)
    proba = np.exp(z); proba /= proba.sum(axis=1, keepdims=True)
    print(f"[*] full-FT anchor: N={len(y_full):,} sessions={len(pred_files)}")
    mean_tpr, macro_auc = macro_roc(y_full, proba, fpr_grid)
    print(f"    macro AUC = {macro_auc:.3f}")
    keys.append("anchor")
    tprs.append(mean_tpr); aucs.append(macro_auc)
    meta["anchor"] = {"label": "w2v2 full-FT (75k chandist)",
                      "color": "#882255"}

    # ── External proba caches (ViT, AnyNet, classical baselines) ────────
    for ext in EXTERNAL_PROBA:
        if not os.path.isfile(ext["path"]):
            print(f"[!] missing {ext['path']} — skipping {ext['key']}")
            continue
        d = np.load(ext["path"], allow_pickle=True)
        proba = d["proba"].astype(np.float32)
        y = d["y"].astype(np.int64)
        print(f"[*] {ext['key']}: proba shape={proba.shape}")
        mean_tpr, macro_auc = macro_roc(y, proba, fpr_grid)
        print(f"    macro AUC = {macro_auc:.3f}")
        keys.append(ext["key"])
        tprs.append(mean_tpr); aucs.append(macro_auc)
        meta[ext["key"]] = {"label": ext["label"], "color": ext["color"]}

    Path(os.path.dirname(args.out_path)).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_path,
        keys=np.array(keys, dtype=object),
        mean_tpr=np.stack(tprs, axis=0),
        macro_auc=np.array(aucs, dtype=np.float64),
        fpr_grid=fpr_grid,
        model_meta=json.dumps(meta),
        n_regions=5,
        scoring="w2v2 full-FT softmax (anchor) + zero-shot baseline proba (macro 5-class OvR ROC)",
        scale_label="full_ft_75kcd",
    )
    print(f"[+] wrote {args.out_path}")


if __name__ == "__main__":
    main()
