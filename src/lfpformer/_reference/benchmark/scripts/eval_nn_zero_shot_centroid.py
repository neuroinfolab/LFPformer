#!/usr/bin/env python3
"""NN zero-shot k-NN eval (IBL 75K chandist) — Lfp2vec protocol.

Cosine k-NN on frozen SSL embeddings; no learned parameters. Mirrors the
Lfp2vec NeuroNexus protocol (see eval_nn_linear_probe.py for details):
  - 7-fold leave-one-session-out (NN_syn_03 excluded)
  - within-session trial split: 36 train / 12 val / 12 test trials (seed=42)
  - random session split of the 6 remaining: 4 train + 2 val (random.seed=42)
  - test = held-out session × 12 test trials only
  - k ∈ {1, 3, 5, 10, 20}, picked by val accuracy
  - primary metric = plain accuracy; balanced accuracy retained as secondary

Outputs:
    <out>/zero_shot_results.json
    <out>/predictions/<test>.npz   preds_at_best_k + per-k preds + labels
    <out>/embeddings_cache.npz     same schema as eval_nn_linear_probe.py
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split as sk_train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parent
ALPHABRAIN_ROOT = BENCH_ROOT.parent
sys.path.insert(0, str(BENCH_ROOT))
sys.path.insert(0, str(ALPHABRAIN_ROOT))

from backbones.wav2vec2_backbone import Wav2Vec2Backbone  # noqa: E402

NN_SESSIONS = [
    "AD_HF01_1", "AD_HF02_2", "AD_HF02_4",
    "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02",
]
CLASS_NAMES = ["Cortex", "CA1", "CA2", "CA3", "DG"]
NUM_CLASSES = 5
K_SWEEP = (1, 3, 5, 10, 20)

DEFAULT_CKPT = (
    __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/benchmarks/w2v2_v2_crossneg_75k_chandist_s500_5653103/checkpoints/ssl_wav2vec2_best_valloss.pt')
)
DEFAULT_LFP_DIR = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws4/region_decoding/data/Neuronexus/lfp')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_session(lfp_dir, session):
    fp = os.path.join(lfp_dir, f"{session}_lfp.pickle")
    with open(fp, "rb") as fh:
        data = pickle.load(fh)
    n = len(data)
    sigs = np.empty((n, 3750), dtype=np.float32)
    labs = np.empty(n, dtype=np.int64)
    trls = np.empty(n, dtype=np.int32)
    for i, s in enumerate(data):
        x = np.asarray(s[0], dtype=np.float32)
        if x.shape[0] != 3750:
            x = x[:3750] if x.shape[0] > 3750 else np.pad(x, (0, 3750 - x.shape[0]))
        sigs[i] = x
        labs[i] = int(s[1])
        trls[i] = int(s[2])
    keep = ~np.all(sigs == 0, axis=1)
    sigs, labs, trls = sigs[keep], labs[keep], trls[keep]
    if labs.min() < 0 or labs.max() >= NUM_CLASSES:
        raise RuntimeError(f"{session}: labels out of range "
                           f"[{labs.min()}, {labs.max()}], expected 0..{NUM_CLASSES-1}")
    log(f"  {session}: N={len(sigs)}  trials={int(trls.max()-trls.min()+1)}  "
        f"classes={np.bincount(labs, minlength=NUM_CLASSES).tolist()}")
    return sigs, labs, trls


def load_session_trials_only(lfp_dir, session):
    fp = os.path.join(lfp_dir, f"{session}_lfp.pickle")
    with open(fp, "rb") as fh:
        data = pickle.load(fh)
    n = len(data)
    keep = np.ones(n, dtype=bool)
    trls = np.empty(n, dtype=np.int32)
    for i, s in enumerate(data):
        x = np.asarray(s[0], dtype=np.float32)
        if x.shape[0] != 3750:
            x = x[:3750] if x.shape[0] > 3750 else np.pad(x, (0, 3750 - x.shape[0]))
        if np.all(x == 0):
            keep[i] = False
        trls[i] = int(s[2])
    return trls[keep]


def compute_trial_split(trial_length=60, fixed_test_count=12, val_size=0.25,
                        random_state=42):
    all_idx = np.arange(trial_length)
    test_tr = np.random.RandomState(seed=random_state).choice(
        all_idx, size=fixed_test_count, replace=False)
    remaining = np.setdiff1d(all_idx, test_tr)
    if trial_length >= fixed_test_count + 4:
        train_tr, val_tr = sk_train_test_split(
            remaining, test_size=val_size, random_state=random_state)
    else:
        train_tr, val_tr = remaining, remaining
    return train_tr, val_tr, test_tr


def compute_session_split(rest_sessions, train_frac=0.8, seed=42):
    random.seed(seed)
    n_train = int(len(rest_sessions) * train_frac)
    train = random.sample(list(rest_sessions), n_train)
    val = [s for s in rest_sessions if s not in train]
    return train, val


def load_backbone(ckpt_path, device):
    log(f"backbone: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "backbone_state_dict" not in ck or "config" not in ck:
        raise RuntimeError("ckpt missing backbone_state_dict / config")
    cfg = dict(ck["config"])
    cfg["rand_init"] = True
    cfg["queue_size"] = 0
    backbone = Wav2Vec2Backbone(cfg)
    missing, unexpected = backbone.load_state_dict(
        ck["backbone_state_dict"], strict=False)
    critical = (
        "model.wav2vec2.encoder",
        "model.wav2vec2.feature_extractor",
        "model.wav2vec2.feature_projection",
        "model.wav2vec2.masked_spec_embed",
    )
    crit_missing = [k for k in missing if k.startswith(critical)]
    if crit_missing:
        raise RuntimeError(
            f"Backbone missing critical keys ({len(crit_missing)}): "
            f"{crit_missing[:5]}")
    log(f"  loaded: missing={len(missing)} unexpected={len(unexpected)} "
        f"epoch={ck.get('epoch')}")
    backbone.to(device).eval()
    return backbone


@torch.no_grad()
def embed(backbone, signals, src_sr, device, batch_size=64):
    n = len(signals)
    out = np.empty((n, 768), dtype=np.float32)
    t0 = time.time()
    total = (n + batch_size - 1) // batch_size
    for i in range(0, n, batch_size):
        wav = torch.from_numpy(signals[i:i + batch_size]).to(device, non_blocking=True)
        if src_sr != 16000:
            wav = torchaudio.functional.resample(wav, src_sr, 16000)
        emb_b = backbone.encode(wav)
        if emb_b.ndim == 3:
            emb_b = emb_b.mean(dim=1)
        out[i:i + len(wav)] = emb_b.cpu().numpy()
        bi = i // batch_size + 1
        if bi % 50 == 0 or bi == total:
            dt = time.time() - t0
            log(f"    embed {bi}/{total}  {dt:.1f}s  ({(i+len(wav))/dt:.0f}/s)")
    return out


def get_or_build_embeddings(cache_path, lfp_dir, sessions, ckpt_path, src_sr, device):
    fingerprint = f"{os.path.abspath(ckpt_path)}|sr={src_sr}|sessions={','.join(sessions)}"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        cached_fp = str(z["fingerprint"]) if "fingerprint" in z.files else ""
        if cached_fp == fingerprint:
            cache = {}
            need_resave = False
            for s in sessions:
                emb = z[f"emb_{s}"]
                lab = z[f"lab_{s}"]
                if f"trl_{s}" in z.files:
                    trl = z[f"trl_{s}"]
                else:
                    log(f"  augmenting cache with trial idx for {s} ...")
                    trl = load_session_trials_only(lfp_dir, s)
                    if len(trl) != len(emb):
                        raise RuntimeError(
                            f"trial-count mismatch for {s}: trl={len(trl)} "
                            f"vs cached emb={len(emb)}")
                    need_resave = True
                cache[s] = (emb, lab, trl)
            if need_resave:
                save_kwargs = {"fingerprint": np.array(fingerprint)}
                for s, (e, l, t) in cache.items():
                    save_kwargs[f"emb_{s}"] = e
                    save_kwargs[f"lab_{s}"] = l
                    save_kwargs[f"trl_{s}"] = t
                np.savez_compressed(cache_path, **save_kwargs)
                log(f"  augmented cache → {cache_path}")
            log(f"loaded cached embeddings: {cache_path}")
            return cache
        raise RuntimeError(
            f"Cache fingerprint mismatch at {cache_path}.\n"
            f"  cached:   {cached_fp}\n  expected: {fingerprint}\n"
            f"Delete cache or pick a different --out-dir.")

    backbone = load_backbone(ckpt_path, device)
    cache = {}
    for s in sessions:
        sigs, labs, trls = load_session(lfp_dir, s)
        e = embed(backbone, sigs, src_sr, device)
        cache[s] = (e, labs, trls)
    save_kwargs = {"fingerprint": np.array(fingerprint)}
    for s, (e, l, t) in cache.items():
        save_kwargs[f"emb_{s}"] = e
        save_kwargs[f"lab_{s}"] = l
        save_kwargs[f"trl_{s}"] = t
    np.savez_compressed(cache_path, **save_kwargs)
    log(f"cached embeddings → {cache_path}")
    del backbone
    torch.cuda.empty_cache()
    return cache


def knn_predict_cosine(emb_train, lab_train, emb_query, k_sweep, device,
                       chunk_size=1024):
    """Cosine k-NN predictions for each k in k_sweep.

    Returns: dict {k: preds (np.int64, len N_query)}.
    Uses GPU batched cosine similarity → torch.topk per chunk → majority vote.
    """
    K_max = max(k_sweep)
    et = torch.from_numpy(emb_train).to(device)
    et = et / et.norm(dim=1, keepdim=True).clamp_min(1e-8)
    lt = torch.from_numpy(lab_train).to(device)

    eq = torch.from_numpy(emb_query)  # CPU; chunks moved to device
    n_q = eq.shape[0]
    topk_idx = torch.empty((n_q, K_max), dtype=torch.long, device=device)
    for i in range(0, n_q, chunk_size):
        q = eq[i:i + chunk_size].to(device)
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sims = q @ et.T          # (b, N_train)
        _, idx = sims.topk(K_max, dim=1)
        topk_idx[i:i + q.shape[0]] = idx

    # Gather neighbour labels once: (N_q, K_max)
    nbr_labels = lt[topk_idx]    # GPU long tensor

    preds_per_k = {}
    for k in k_sweep:
        sub = nbr_labels[:, :k]  # (N_q, k)
        # Majority vote: bincount per row via one-hot sum
        oh = torch.nn.functional.one_hot(sub, num_classes=NUM_CLASSES).sum(dim=1)
        # Tie-break by smallest class id (argmax returns first max → matches sklearn convention)
        preds = oh.argmax(dim=1).cpu().numpy().astype(np.int64)
        preds_per_k[k] = preds

    del et, lt, eq, topk_idx, nbr_labels
    torch.cuda.empty_cache()
    return preds_per_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--lfp-dir", default=DEFAULT_LFP_DIR)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-rate", type=int, default=1250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--knn-chunk", type=int, default=1024)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    log(f"out dir: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    cache_path = out_dir / "embeddings_cache.npz"
    cache = get_or_build_embeddings(
        cache_path, args.lfp_dir, NN_SESSIONS, args.ckpt, args.sample_rate, device)

    train_tr_idx, val_tr_idx, test_tr_idx = compute_trial_split(
        trial_length=60, fixed_test_count=12, val_size=0.25,
        random_state=args.seed)
    log(f"trial split (seed={args.seed}): "
        f"train={len(train_tr_idx)} val={len(val_tr_idx)} test={len(test_tr_idx)} trials")

    def slice_sessions(sess_list, trial_idx_set):
        embs, labs = [], []
        for s in sess_list:
            e, l, t = cache[s]
            m = np.isin(t, trial_idx_set)
            embs.append(e[m]); labs.append(l[m])
        return np.concatenate(embs, axis=0), np.concatenate(labs, axis=0)

    fold_results = []
    for test_sess in NN_SESSIONS:
        log(f"\n=== fold: test={test_sess} ===")
        rest = [s for s in NN_SESSIONS if s != test_sess]
        train_sess, val_sess = compute_session_split(rest, train_frac=0.8,
                                                     seed=args.seed)
        log(f"  train sessions ({len(train_sess)}): {train_sess}")
        log(f"  val   sessions ({len(val_sess)}): {val_sess}")

        emb_tr, lab_tr = slice_sessions(train_sess, train_tr_idx)
        emb_va, lab_va = slice_sessions(val_sess,   val_tr_idx)
        emb_te, lab_te = slice_sessions([test_sess], test_tr_idx)

        log(f"  sizes: train={len(emb_tr)}  val={len(emb_va)}  test={len(emb_te)}")

        # Val-side k selection
        val_preds = knn_predict_cosine(emb_tr, lab_tr, emb_va, K_SWEEP,
                                       device, chunk_size=args.knn_chunk)
        val_acc_per_k = {k: float(accuracy_score(lab_va, val_preds[k]))
                          for k in K_SWEEP}
        val_bal = {k: float(balanced_accuracy_score(lab_va, val_preds[k]))
                   for k in K_SWEEP}
        log(f"  val acc per k: {val_acc_per_k}")
        best_k = max(K_SWEEP, key=lambda kk: val_acc_per_k[kk])
        log(f"  → best_k = {best_k}  (val_acc={val_acc_per_k[best_k]:.4f})")

        # Test predictions for ALL k (so we can later inspect sensitivity)
        test_preds = knn_predict_cosine(emb_tr, lab_tr, emb_te, K_SWEEP,
                                        device, chunk_size=args.knn_chunk)
        test_bal = {k: float(balanced_accuracy_score(lab_te, test_preds[k]))
                    for k in K_SWEEP}
        test_acc = {k: float(accuracy_score(lab_te, test_preds[k])) for k in K_SWEEP}
        log(f"  test acc per k: {test_acc}")

        preds_best = test_preds[best_k]
        cm = confusion_matrix(lab_te, preds_best,
                              labels=list(range(NUM_CLASSES))).tolist()
        log(f"  → @ best_k={best_k}: test_acc={test_acc[best_k]:.4f}  "
            f"test_bal_acc={test_bal[best_k]:.4f}")

        np.savez_compressed(
            pred_dir / f"{test_sess}.npz",
            preds_at_best_k=preds_best, labels=lab_te,
            **{f"preds_k{k}": test_preds[k] for k in K_SWEEP},
            train_sessions=np.array(train_sess), val_sessions=np.array(val_sess),
        )
        fold_results.append({
            "test_session": test_sess,
            "train_sessions": train_sess,
            "val_sessions": val_sess,
            "n_train": int(len(emb_tr)),
            "n_val": int(len(emb_va)),
            "n_test": int(len(emb_te)),
            "best_k": int(best_k),
            "val_acc_per_k": val_acc_per_k,
            "val_bal_acc_per_k": val_bal,
            "test_bal_acc_per_k": test_bal,
            "test_acc_per_k": test_acc,
            "test_acc": test_acc[best_k],
            "test_bal_acc": test_bal[best_k],
            "confusion_matrix": cm,
        })

    summary = {
        "ckpt": args.ckpt,
        "method": "knn_cosine",
        "k_sweep": list(K_SWEEP),
        "class_names": CLASS_NAMES,
        "n_folds": len(fold_results),
        "mean_test_acc":     float(np.mean([f["test_acc"] for f in fold_results])),
        "std_test_acc":      float(np.std([f["test_acc"] for f in fold_results])),
        "mean_test_bal_acc": float(np.mean([f["test_bal_acc"] for f in fold_results])),
        "std_test_bal_acc":  float(np.std([f["test_bal_acc"] for f in fold_results])),
        "folds": fold_results,
        "args": vars(args),
    }
    with open(out_dir / "zero_shot_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSUMMARY:  mean_acc={summary['mean_test_acc']:.4f} ± {summary['std_test_acc']:.4f}  "
        f"mean_bal_acc={summary['mean_test_bal_acc']:.4f} ± {summary['std_test_bal_acc']:.4f}")
    log(f"results → {out_dir / 'zero_shot_results.json'}")


if __name__ == "__main__":
    main()
