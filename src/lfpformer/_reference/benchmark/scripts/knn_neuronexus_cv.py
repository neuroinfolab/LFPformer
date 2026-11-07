"""3-fold session-rotation KNN classification on Neuronexus 5-class labels.

Loads SSL embeddings for one or more variants, applies a session-wise
train/val/test split (rotating test through {AD_HF01_1, NN_syn_01, NN_syn_02}),
fits a KNN classifier on train, evaluates on val and test, and reports
per-fold and aggregated balanced accuracy / macro F1 / confusion matrix.

Drops class 5 (UNK) before splitting. Embeddings L2-normalized.
Sweeps k in {1, 3, 5, 10, 20}.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize


REGION_NAMES = {0: "Cortex", 1: "CA1", 2: "CA2", 3: "CA3", 4: "DG"}

# 3-fold session-rotation. Train pool always = the 5 sessions that are not
# in {AD_HF01_1, NN_syn_01, NN_syn_02}. Test rotates; val = the other two
# of {AD_HF01_1, NN_syn_01, NN_syn_02}.
TRAIN_POOL = ["AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_03"]
ROTATING = ["AD_HF01_1", "NN_syn_01", "NN_syn_02"]
FOLDS = []
for i, test_sess in enumerate(ROTATING):
    val_sess = [s for s in ROTATING if s != test_sess]
    FOLDS.append({
        "fold": i + 1,
        "train": TRAIN_POOL,
        "val": val_sess,
        "test": [test_sess],
    })

KS = [1, 3, 5, 10, 20]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_emb(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    lab = d["region_label"].astype(np.int32)
    sess = np.asarray(d["session"]).astype(str)
    keep = lab != 5
    return emb[keep], lab[keep], sess[keep]


def evaluate_split(X_tr, y_tr, X_ev, y_ev, ks):
    X_tr_n = normalize(X_tr, norm="l2", axis=1)
    X_ev_n = normalize(X_ev, norm="l2", axis=1)
    out = {}
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, metric="euclidean")
        clf.fit(X_tr_n, y_tr)
        pred = clf.predict(X_ev_n)
        out[k] = {
            "bal_acc": float(balanced_accuracy_score(y_ev, pred)),
            "macro_f1": float(f1_score(y_ev, pred, average="macro")),
            "pred": pred,
        }
    return out


def run_variant(variant, npz_path, out_dir):
    log(f"=== Variant: {variant} ===")
    log(f"Loading {npz_path}")
    emb, lab, sess = load_emb(npz_path)
    log(f"  N={len(emb):,}  classes seen={sorted(set(lab.tolist()))}  "
        f"sessions={sorted(set(sess.tolist()))}")

    fold_results = []
    for fold in FOLDS:
        train_mask = np.isin(sess, fold["train"])
        val_mask = np.isin(sess, fold["val"])
        test_mask = np.isin(sess, fold["test"])
        log(f"  Fold {fold['fold']} test={fold['test']}: "
            f"train={train_mask.sum():,}  val={val_mask.sum():,}  "
            f"test={test_mask.sum():,}")

        X_tr, y_tr = emb[train_mask], lab[train_mask]
        X_va, y_va = emb[val_mask], lab[val_mask]
        X_te, y_te = emb[test_mask], lab[test_mask]

        val_res = evaluate_split(X_tr, y_tr, X_va, y_va, KS)
        test_res = evaluate_split(X_tr, y_tr, X_te, y_te, KS)

        # Best k by val bal_acc
        best_k = max(KS, key=lambda k: val_res[k]["bal_acc"])
        cm_test = confusion_matrix(
            y_te, test_res[best_k]["pred"],
            labels=list(REGION_NAMES.keys()))
        log(f"    val[best_k={best_k}]: "
            f"bal_acc={val_res[best_k]['bal_acc']:.4f}  "
            f"f1={val_res[best_k]['macro_f1']:.4f}")
        log(f"    test[best_k={best_k}]: "
            f"bal_acc={test_res[best_k]['bal_acc']:.4f}  "
            f"f1={test_res[best_k]['macro_f1']:.4f}")

        fold_results.append({
            "fold": fold["fold"],
            "test_session": fold["test"][0],
            "best_k": int(best_k),
            "val": {k: {kk: v for kk, v in val_res[k].items() if kk != "pred"}
                    for k in KS},
            "test": {k: {kk: v for kk, v in test_res[k].items() if kk != "pred"}
                     for k in KS},
            "test_confusion": cm_test.tolist(),
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
        })

    # Aggregate across folds at each k (test set)
    agg = {}
    for k in KS:
        ba = [fr["test"][k]["bal_acc"] for fr in fold_results]
        f1 = [fr["test"][k]["macro_f1"] for fr in fold_results]
        agg[k] = {
            "test_bal_acc_mean": float(np.mean(ba)),
            "test_bal_acc_std": float(np.std(ba, ddof=0)),
            "test_macro_f1_mean": float(np.mean(f1)),
            "test_macro_f1_std": float(np.std(f1, ddof=0)),
        }

    # Best k by mean val bal_acc
    val_means = {k: float(np.mean([fr["val"][k]["bal_acc"]
                                   for fr in fold_results])) for k in KS}
    best_k = max(KS, key=lambda k: val_means[k])

    # Summed 5x5 confusion matrix at best_k across folds.
    # Re-fit at best_k (skipped storing predictions earlier to keep JSON small).
    label_order = list(REGION_NAMES.keys())
    cm_sum = np.zeros((len(label_order), len(label_order)), dtype=np.int64)
    for fold in FOLDS:
        train_mask = np.isin(sess, fold["train"])
        test_mask = np.isin(sess, fold["test"])
        Xtr_n = normalize(emb[train_mask], norm="l2", axis=1)
        Xte_n = normalize(emb[test_mask], norm="l2", axis=1)
        clf = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1,
                                   metric="euclidean")
        clf.fit(Xtr_n, lab[train_mask])
        pred = clf.predict(Xte_n)
        cm_sum += confusion_matrix(lab[test_mask], pred, labels=label_order)
    log(f"  Aggregated best_k={best_k}: "
        f"test bal_acc={agg[best_k]['test_bal_acc_mean']:.4f}"
        f"±{agg[best_k]['test_bal_acc_std']:.4f}  "
        f"f1={agg[best_k]['test_macro_f1_mean']:.4f}"
        f"±{agg[best_k]['test_macro_f1_std']:.4f}")

    out = {
        "variant": variant,
        "embeddings_npz": str(npz_path),
        "n_total": int(len(emb)),
        "folds": fold_results,
        "aggregate_test": agg,
        "val_bal_acc_mean_by_k": val_means,
        "best_k_by_val": int(best_k),
        "test_confusion_summed_at_best_k": cm_sum.tolist(),
        "region_names": REGION_NAMES,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"knn_cv_{variant}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"  Saved {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026/neuronexus_zero_shot_75k'),
        help="Root containing variant_<name>/embeddings.npz")
    ap.add_argument("--variants", nargs="+",
                    default=["distc_75k", "qv3_75k", "pdm_75k"])
    ap.add_argument("--out-dir", default=None,
                    help="Defaults to <root>/knn_results")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "knn_results"

    summary = {}
    for v in args.variants:
        npz = root / f"variant_{v}" / "embeddings.npz"
        if not npz.exists():
            log(f"SKIP {v}: missing {npz}")
            continue
        summary[v] = run_variant(v, npz, out_dir)

    # Print summary table
    print()
    log("=== Summary (test set, mean ± std over 3 folds, best k by val) ===")
    print(f"{'variant':<14} {'best_k':>6}  "
          f"{'bal_acc':>16}  {'macro_f1':>16}")
    for v, res in summary.items():
        k = res["best_k_by_val"]
        a = res["aggregate_test"][k]
        print(f"{v:<14} {k:>6}  "
              f"{a['test_bal_acc_mean']:>7.4f}±{a['test_bal_acc_std']:.4f}  "
              f"{a['test_macro_f1_mean']:>7.4f}±{a['test_macro_f1_std']:.4f}")

    summary_path = out_dir / "summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({v: {k: r for k, r in res.items() if k != "folds"}
                   for v, res in summary.items()}, f, indent=2)
    log(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
