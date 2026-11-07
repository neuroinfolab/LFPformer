#!/usr/bin/env python3
"""Linear-probe eval on NeuroNexus 5-class region task — Lfp2vec protocol.

Mirrors the original Lfp2vec NeuroNexus pipeline (`run_wave2vec2.py`):
  - 7-fold leave-one-session-out (NN_syn_03 excluded)
  - within-session trial-level split shared across all sessions:
      36 train trials / 12 val trials / 12 test trials  (seed=42)
  - random session split of the remaining 6 sessions:
      int(0.8 * 6) = 4 train sessions + 2 val sessions  (random.seed=42)
  - test set = held-out session × 12 test trials only
  - primary metric = plain accuracy (sklearn.metrics.accuracy_score);
    balanced accuracy retained as secondary diagnostic

Backbone: 75K IBL pdm ckpt (default).
Decoder:  nn.Linear(768, 5) trained with cross-entropy on mean-pooled
          last_hidden_state. No fine-tuning of the backbone.

Optimization: embeddings are extracted ONCE per session (frozen backbone)
and cached to <out>/embeddings_cache.npz, then re-used across all 7 folds.

Outputs:
    <out>/nn_probe_results.json   per-fold + summary metrics
    <out>/predictions/<test>.npz  logits/labels/preds per held-out session
    <out>/embeddings_cache.npz    per-sample 768-d embeddings (one pass)
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
CLASS_NAMES = ["Cortex", "CA1", "CA2", "CA3", "DG"]  # repo convention: 0=Cortex, 1=CA1, ..., 4=DG
NUM_CLASSES = 5

DEFAULT_CKPT = (
    __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/benchmarks/w2v2_v2_crossneg_75k_chandist_s500_5653103/checkpoints/ssl_wav2vec2_best_valloss.pt')
)
DEFAULT_LFP_DIR = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws4/region_decoding/data/Neuronexus/lfp')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Data ────────────────────────────────────────────────────────────────
def load_session(lfp_dir, session):
    """Returns (signals, labels, trial_idx) — order-preserving, all-zero chunks dropped."""
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
    """Cheap reload of trial idx (matches load_session order, no embedding pass).
    Used to augment a pre-existing embeddings cache that lacks trial info."""
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
    """Mirror of original Lfp2vec `split_trials()`.
    Returns (train_tr_idx, val_tr_idx, test_tr_idx) — these indices are SHARED
    across all 7 sessions (every session has trials 0..59)."""
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
    """Mirror of `random.seed(42); random.sample(session_list, int(N*0.8))`
    from the original Lfp2vec script (per held-out session)."""
    random.seed(seed)
    n_train = int(len(rest_sessions) * train_frac)
    train = random.sample(list(rest_sessions), n_train)
    val = [s for s in rest_sessions if s not in train]
    return train, val


# ── Backbone ────────────────────────────────────────────────────────────
def load_backbone(ckpt_path, device):
    log(f"backbone: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "backbone_state_dict" not in ck or "config" not in ck:
        raise RuntimeError("ckpt missing backbone_state_dict / config")
    cfg = dict(ck["config"])
    cfg["rand_init"] = True   # weights from ckpt; skip HF download
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
        emb = backbone.encode(wav)
        if emb.ndim == 3:                # (B, T, D) → mean over time
            emb = emb.mean(dim=1)
        out[i:i + len(wav)] = emb.cpu().numpy()
        bi = i // batch_size + 1
        if bi % 50 == 0 or bi == total:
            dt = time.time() - t0
            log(f"    embed {bi}/{total}  {dt:.1f}s  ({(i+len(wav))/dt:.0f}/s)")
    return out


def get_or_build_embeddings(cache_path, lfp_dir, sessions, ckpt_path, src_sr, device):
    """Cache schema (per session): emb_<s>, lab_<s>, trl_<s>.
    Pre-existing caches that lack trl_<s> are augmented from pickles in-place
    (no re-embedding needed since chunk order is deterministic)."""
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
                            f"vs cached emb={len(emb)} — order changed?")
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
            f"Delete the cache or pick a different --out-dir.")

    backbone = load_backbone(ckpt_path, device)
    cache = {}
    for s in sessions:
        sigs, labs, trls = load_session(lfp_dir, s)
        emb = embed(backbone, sigs, src_sr, device)
        cache[s] = (emb, labs, trls)
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


# ── Linear probe ────────────────────────────────────────────────────────
def train_probe(emb_tr, lab_tr, emb_va, lab_va, emb_te, lab_te,
                device, epochs=200, lr=1e-3, weight_decay=1e-4,
                batch_size=512, patience=20, seed=42):
    torch.manual_seed(seed)
    probe = nn.Linear(768, NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr = torch.from_numpy(emb_tr).to(device)
    ytr = torch.from_numpy(lab_tr).to(device)
    Xva = torch.from_numpy(emb_va).to(device)
    yva = torch.from_numpy(lab_va).to(device)
    Xte = torch.from_numpy(emb_te).to(device)
    yte_np = lab_te

    best_va_loss = float("inf")
    best_state = None
    bad = 0
    n_tr = len(Xtr)
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(n_tr, device=device)
        ep_loss = 0.0
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            logits = probe(Xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= n_tr

        probe.eval()
        with torch.no_grad():
            va_logits = probe(Xva)
            va_loss = F.cross_entropy(va_logits, yva).item()
            va_acc = (va_logits.argmax(1) == yva).float().mean().item()

        if va_loss < best_va_loss - 1e-5:
            best_va_loss = va_loss
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            log(f"    early stop @ ep{ep+1}  best_va_loss={best_va_loss:.4f}")
            break
        if (ep + 1) % 25 == 0:
            log(f"    ep{ep+1:3d}  tr_loss={ep_loss:.4f}  va_loss={va_loss:.4f}  va_acc={va_acc:.3f}")

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        te_logits = probe(Xte).cpu().numpy()
    preds = te_logits.argmax(1)
    acc = float((preds == yte_np).mean())
    bal = float(balanced_accuracy_score(yte_np, preds))
    cm = confusion_matrix(yte_np, preds, labels=list(range(NUM_CLASSES))).tolist()
    return {
        "test_acc": acc,
        "test_bal_acc": bal,
        "confusion_matrix": cm,
        "best_va_loss": float(best_va_loss),
        "logits": te_logits,
        "preds": preds,
    }


# ── Main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--lfp-dir", default=DEFAULT_LFP_DIR)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-rate", type=int, default=1250)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
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

    # Trial split is shared across all sessions (Lfp2vec convention)
    train_tr_idx, val_tr_idx, test_tr_idx = compute_trial_split(
        trial_length=60, fixed_test_count=12, val_size=0.25,
        random_state=args.seed)
    log(f"trial split (seed={args.seed}): "
        f"train={len(train_tr_idx)} val={len(val_tr_idx)} test={len(test_tr_idx)} trials")

    fold_results = []
    for test_sess in NN_SESSIONS:
        log(f"\n=== fold: test={test_sess} ===")
        rest = [s for s in NN_SESSIONS if s != test_sess]
        train_sess, val_sess = compute_session_split(rest, train_frac=0.8,
                                                     seed=args.seed)
        log(f"  train sessions ({len(train_sess)}): {train_sess}")
        log(f"  val   sessions ({len(val_sess)}): {val_sess}")

        # Slice each session by its trial idx ∈ {train,val,test}_tr_idx
        def slice_sessions(sess_list, trial_idx_set):
            embs, labs = [], []
            for s in sess_list:
                e, l, t = cache[s]
                m = np.isin(t, trial_idx_set)
                embs.append(e[m])
                labs.append(l[m])
            return np.concatenate(embs, axis=0), np.concatenate(labs, axis=0)

        emb_tr, lab_tr = slice_sessions(train_sess, train_tr_idx)
        emb_va, lab_va = slice_sessions(val_sess,   val_tr_idx)
        emb_te, lab_te = slice_sessions([test_sess], test_tr_idx)

        log(f"  sizes: train={len(emb_tr)} val={len(emb_va)} test={len(emb_te)}")
        log(f"  train class dist: {np.bincount(lab_tr, minlength=NUM_CLASSES).tolist()}")
        log(f"  val   class dist: {np.bincount(lab_va, minlength=NUM_CLASSES).tolist()}")
        log(f"  test  class dist: {np.bincount(lab_te, minlength=NUM_CLASSES).tolist()}")

        res = train_probe(
            emb_tr, lab_tr, emb_va, lab_va, emb_te, lab_te, device,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            patience=args.patience, seed=args.seed)

        log(f"  → test_acc={res['test_acc']:.4f} (primary)  "
            f"test_bal_acc={res['test_bal_acc']:.4f} (secondary)")

        np.savez_compressed(
            pred_dir / f"{test_sess}.npz",
            logits=res["logits"], preds=res["preds"], labels=lab_te,
            train_sessions=np.array(train_sess), val_sessions=np.array(val_sess),
        )
        fold_results.append({
            "test_session": test_sess,
            "train_sessions": train_sess,
            "val_sessions": val_sess,
            "n_train": int(len(emb_tr)),
            "n_val": int(len(emb_va)),
            "n_test": int(len(emb_te)),
            "test_acc": res["test_acc"],
            "test_bal_acc": res["test_bal_acc"],
            "best_va_loss": res["best_va_loss"],
            "confusion_matrix": res["confusion_matrix"],
        })

    summary = {
        "ckpt": args.ckpt,
        "class_names": CLASS_NAMES,
        "n_folds": len(fold_results),
        "mean_test_acc":     float(np.mean([f["test_acc"] for f in fold_results])),
        "std_test_acc":      float(np.std([f["test_acc"] for f in fold_results])),
        "mean_test_bal_acc": float(np.mean([f["test_bal_acc"] for f in fold_results])),
        "std_test_bal_acc":  float(np.std([f["test_bal_acc"] for f in fold_results])),
        "folds": fold_results,
        "args": vars(args),
    }
    with open(out_dir / "nn_probe_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSUMMARY:  mean_acc={summary['mean_test_acc']:.4f} ± {summary['std_test_acc']:.4f}  "
        f"mean_bal_acc={summary['mean_test_bal_acc']:.4f} ± {summary['std_test_bal_acc']:.4f}")
    log(f"results → {out_dir / 'nn_probe_results.json'}")


if __name__ == "__main__":
    main()
