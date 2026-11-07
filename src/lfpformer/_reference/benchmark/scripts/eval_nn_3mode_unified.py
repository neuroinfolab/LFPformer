#!/usr/bin/env python3
"""Unified NN 3-mode eval supporting wav2vec2 / ViT-rival / AnyNet-rival.

Runs ONE of {zero_shot, frozen_lp, full_ft} on the canonical Lfp2vec
NeuroNexus protocol (7-fold session rotation, within-session trial split,
seed=42) and writes a result.json compatible with plot_cross_species_bars.py.

Schema parity:
  - mode=zero_shot writes `zero_shot_results.json` with `mean_test_bal_acc`,
    `std_test_bal_acc`, `n_folds`, `folds[]` (matches eval_nn_zero_shot_centroid).
  - mode=frozen_lp writes `nn_probe_results.json` (matches eval_nn_linear_probe).
  - mode=full_ft writes `nn_fullft_results.json` (matches eval_nn_full_finetune).

Reuses helpers from existing eval scripts to avoid duplicating the protocol.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix)

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parent
ALPHABRAIN_ROOT = BENCH_ROOT.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BENCH_ROOT))
sys.path.insert(0, str(ALPHABRAIN_ROOT))

# Reuse data-loading + protocol helpers from the existing zero-shot script.
from eval_nn_zero_shot_centroid import (
    load_session, compute_trial_split, compute_session_split,
    NN_SESSIONS, CLASS_NAMES, NUM_CLASSES,
)
from nn_eval_backbone import (load_backbone, embed_signals,
                              encode_batch_for_train)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------- IO

def load_all_sessions(lfp_dir, sessions):
    out = {}
    for s in sessions:
        sigs, labs, trls = load_session(lfp_dir, s)
        out[s] = (sigs, labs, trls)
    return out


def slice_sigs(sess_data, sess_list, trial_idx):
    sigs, labs, trls = [], [], []
    for s in sess_list:
        sg, lb, tr = sess_data[s]
        m = np.isin(tr, trial_idx)
        sigs.append(sg[m]); labs.append(lb[m]); trls.append(tr[m])
    return (np.concatenate(sigs, axis=0),
            np.concatenate(labs, axis=0),
            np.concatenate(trls, axis=0))


# --------------------------------------------------------------------- ZS / LP

@torch.no_grad()
def cosine_knn(emb_train, lab_train, emb_query, k, return_proba=False):
    """Plain cosine k-NN with majority vote (ties broken by sklearn-style
    minimum-distance, which here means highest cosine sim).

    If `return_proba` is True, also returns per-class vote fraction (Q, NUM_CLASSES).
    """
    et = torch.as_tensor(emb_train, device="cuda", dtype=torch.float32)
    eq = torch.as_tensor(emb_query, device="cuda", dtype=torch.float32)
    et = torch.nn.functional.normalize(et, dim=1)
    eq = torch.nn.functional.normalize(eq, dim=1)
    sim = eq @ et.t()  # (Q, T)
    # top-k
    k_use = min(k, sim.shape[1])
    _, idx = sim.topk(k_use, dim=1, largest=True, sorted=True)
    nbr_lab = torch.as_tensor(lab_train, device="cuda")[idx]  # (Q, k)
    preds = []
    proba = np.zeros((nbr_lab.shape[0], NUM_CLASSES), dtype=np.float32)
    for i, row in enumerate(nbr_lab.cpu().numpy()):
        c = np.bincount(row, minlength=NUM_CLASSES)
        preds.append(int(np.argmax(c)))
        proba[i] = c.astype(np.float32) / max(k_use, 1)
    preds = np.array(preds, dtype=np.int64)
    if return_proba:
        return preds, proba
    return preds


def fold_zero_shot(emb_tr, lab_tr, emb_va, lab_va, emb_te, lab_te,
                   k_sweep=(1, 3, 5, 10, 20)):
    val_per_k = {}
    for k in k_sweep:
        p = cosine_knn(emb_tr, lab_tr, emb_va, k)
        val_per_k[k] = float(balanced_accuracy_score(lab_va, p))
    best_k = max(val_per_k, key=val_per_k.get)
    p_te, proba_te = cosine_knn(emb_tr, lab_tr, emb_te, best_k, return_proba=True)
    return {
        "best_k": int(best_k),
        "val_bal_acc_per_k": val_per_k,
        "test_acc": float(accuracy_score(lab_te, p_te)),
        "test_bal_acc": float(balanced_accuracy_score(lab_te, p_te)),
        "_proba_te": proba_te,    # popped before json.dump
        "_y_te": np.asarray(lab_te, dtype=np.int64),
    }


def fold_frozen_lp(emb_tr, lab_tr, emb_va, lab_va, emb_te, lab_te,
                   embed_dim, device, lr, wd, epochs, patience):
    head = nn.Linear(embed_dim, NUM_CLASSES).to(device)
    nn.init.xavier_uniform_(head.weight); nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(emb_tr).to(device)
    Yt = torch.from_numpy(lab_tr).to(device)
    Xv = torch.from_numpy(emb_va).to(device)
    Yv = torch.from_numpy(lab_va).to(device)
    Xte = torch.from_numpy(emb_te).to(device)

    best_va = float("inf"); best_state = None; bad = 0
    bs = 256
    g = torch.Generator(); g.manual_seed(42)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xt), generator=g).to(device)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            logits = head(Xt[idx])
            loss = crit(logits, Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            va_loss = float(crit(head(Xv), Yv))
        if va_loss < best_va - 1e-5:
            best_va = va_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience: break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        p_te = head(Xte).argmax(dim=1).cpu().numpy()
    return {
        "best_va_loss": float(best_va),
        "test_acc": float(accuracy_score(lab_te, p_te)),
        "test_bal_acc": float(balanced_accuracy_score(lab_te, p_te)),
    }


# --------------------------------------------------------------------- FULL FT

class SignalDataset(Dataset):
    def __init__(self, sigs, labs):
        self.sigs = sigs; self.labs = labs
    def __len__(self): return len(self.sigs)
    def __getitem__(self, i):
        return torch.from_numpy(self.sigs[i]).float(), int(self.labs[i])


def fold_full_ft(args, train_sess, val_sess, test_sess, sess_data,
                 train_tr, val_tr, test_tr, device, embed_dim,
                 native_sr, needs_resample_to, chunk_seconds,
                 backbone_loader_kwargs):
    tr_sigs, tr_labs, _ = slice_sigs(sess_data, train_sess, train_tr)
    va_sigs, va_labs, _ = slice_sigs(sess_data, val_sess, val_tr)
    te_sigs, te_labs, _ = slice_sigs(sess_data, [test_sess], test_tr)
    log(f"  sizes: train={len(tr_sigs)} val={len(va_sigs)} test={len(te_sigs)}")

    train_loader = DataLoader(
        SignalDataset(tr_sigs, tr_labs), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(
        SignalDataset(va_sigs, va_labs), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(
        SignalDataset(te_sigs, te_labs), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Fresh backbone per fold
    backbone, ed, ns, rs, cs = load_backbone(**backbone_loader_kwargs,
                                             device=device)
    assert ed == embed_dim
    head = nn.Linear(embed_dim, NUM_CLASSES).to(device)
    nn.init.xavier_uniform_(head.weight); nn.init.zeros_(head.bias)

    opt = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": args.lr_backbone,
         "weight_decay": args.weight_decay},
        {"params": head.parameters(), "lr": args.lr_head,
         "weight_decay": args.weight_decay},
    ])
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_va = float("inf"); best_bb = None; best_head = None; bad = 0
    for ep in range(args.epochs):
        backbone.train(); head.train()
        t0 = time.time(); ep_loss = 0.0; ep_n = 0
        for wav, y in train_loader:
            wav = wav.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            emb = encode_batch_for_train(backbone, wav, native_sr,
                                          needs_resample_to, chunk_seconds)
            logits = head(emb)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()),
                args.grad_clip)
            opt.step()
            ep_loss += float(loss.detach()) * len(y); ep_n += len(y)
        ep_loss /= max(ep_n, 1)

        backbone.eval(); head.eval()
        va_loss = 0.0; va_n = 0
        with torch.no_grad():
            for wav, y in val_loader:
                wav = wav.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                emb = encode_batch_for_train(backbone, wav, native_sr,
                                              needs_resample_to, chunk_seconds)
                l = crit(head(emb), y)
                va_loss += float(l) * len(y); va_n += len(y)
        va_loss /= max(va_n, 1)
        log(f"  ep{ep:3d} tr_loss={ep_loss:.4f} va_loss={va_loss:.4f} "
            f"({time.time()-t0:.1f}s)")
        if va_loss < best_va - 1e-5:
            best_va = va_loss
            best_bb = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
            best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                log(f"  early stop @ ep{ep+1} best_va={best_va:.4f}")
                break

    backbone.load_state_dict(best_bb); head.load_state_dict(best_head)
    backbone.eval(); head.eval()
    preds_all, labs_all = [], []
    with torch.no_grad():
        for wav, y in test_loader:
            wav = wav.to(device, non_blocking=True)
            emb = encode_batch_for_train(backbone, wav, native_sr,
                                          needs_resample_to, chunk_seconds)
            preds_all.append(head(emb).argmax(dim=1).cpu().numpy())
            labs_all.append(y.numpy())
    p_te = np.concatenate(preds_all); l_te = np.concatenate(labs_all)
    cm = confusion_matrix(l_te, p_te, labels=list(range(NUM_CLASSES))).tolist()
    del backbone, head, opt
    torch.cuda.empty_cache()
    return {
        "best_va_loss": float(best_va),
        "test_acc": float(accuracy_score(l_te, p_te)),
        "test_bal_acc": float(balanced_accuracy_score(l_te, p_te)),
        "confusion_matrix": cm,
    }


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["zero_shot", "frozen_lp", "full_ft"])
    ap.add_argument("--backbone-type", required=True,
                    choices=["wav2vec2", "vit_rival", "anynet_rival"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--lfp-dir",
                    default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws4/region_decoding/data/Neuronexus/lfp'))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-rate", type=int, default=1250,
                    help="LFP source sample rate (Hz)")
    ap.add_argument("--seed", type=int, default=42)
    # LP
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    # Full FT
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone_loader_kwargs = dict(
        backbone_type=args.backbone_type,
        ckpt_path=args.ckpt,
        config_path=args.config,
    )

    log(f"Mode={args.mode}  Backbone={args.backbone_type}  Device={device}")

    # Load all 7 sessions once
    log("Loading sessions ...")
    sess_data = load_all_sessions(args.lfp_dir, NN_SESSIONS)

    # Trial split (shared)
    train_tr, val_tr, test_tr = compute_trial_split()
    log(f"Trial split: |train|={len(train_tr)} |val|={len(val_tr)} "
        f"|test|={len(test_tr)}")

    # Embed ALL sessions once (zs + frozen_lp share this; full_ft re-embeds per-fold internally)
    if args.mode in ("zero_shot", "frozen_lp"):
        backbone, embed_dim, native_sr, needs_resample_to, chunk_s = \
            load_backbone(**backbone_loader_kwargs, device=device)
        emb_per_sess, lab_per_sess, trl_per_sess = {}, {}, {}
        for s in NN_SESSIONS:
            sigs, labs, trls = sess_data[s]
            log(f"  embed {s}: N={len(sigs)} ...")
            emb_per_sess[s] = embed_signals(
                backbone, sigs, args.sample_rate, needs_resample_to,
                chunk_s, embed_dim, device, batch_size=64)
            lab_per_sess[s] = labs; trl_per_sess[s] = trls
        del backbone; torch.cuda.empty_cache()
    else:
        # We still need embed_dim for the FT head
        # Quick load to read dim then drop.
        bb, embed_dim, native_sr, needs_resample_to, chunk_s = \
            load_backbone(**backbone_loader_kwargs, device=device)
        del bb; torch.cuda.empty_cache()

    folds_out = []
    for test_sess in NN_SESSIONS:
        rest = [s for s in NN_SESSIONS if s != test_sess]
        train_sess, val_sess = compute_session_split(rest, seed=args.seed)
        log(f"=== fold: test={test_sess}  "
            f"train={len(train_sess)} val={len(val_sess)} ===")

        if args.mode in ("zero_shot", "frozen_lp"):
            # Build embedding+label arrays restricted to trial subsets
            def stack(sess_list, trial_idx):
                E, L = [], []
                for s in sess_list:
                    m = np.isin(trl_per_sess[s], trial_idx)
                    E.append(emb_per_sess[s][m]); L.append(lab_per_sess[s][m])
                return np.concatenate(E, 0), np.concatenate(L, 0)

            emb_tr, lab_tr = stack(train_sess, train_tr)
            emb_va, lab_va = stack(val_sess, val_tr)
            emb_te, lab_te = stack([test_sess], test_tr)
            log(f"  sizes: train={len(emb_tr)} val={len(emb_va)} "
                f"test={len(emb_te)}")
            log(f"  test class dist: "
                f"{np.bincount(lab_te, minlength=NUM_CLASSES).tolist()}")

            if args.mode == "zero_shot":
                r = fold_zero_shot(emb_tr, lab_tr, emb_va, lab_va,
                                   emb_te, lab_te)
            else:  # frozen_lp
                r = fold_frozen_lp(emb_tr, lab_tr, emb_va, lab_va,
                                   emb_te, lab_te, embed_dim, device,
                                   args.lr, args.weight_decay,
                                   args.epochs, args.patience)
            log(f"  → test_acc={r['test_acc']:.4f}  "
                f"test_bal_acc={r['test_bal_acc']:.4f}")
        else:
            r = fold_full_ft(args, train_sess, val_sess, test_sess,
                              sess_data, train_tr, val_tr, test_tr, device,
                              embed_dim, native_sr, needs_resample_to,
                              chunk_s, backbone_loader_kwargs)
            log(f"  → test_acc={r['test_acc']:.4f}  "
                f"test_bal_acc={r['test_bal_acc']:.4f}")

        r["test_session"] = test_sess
        r["train_sessions"] = train_sess
        r["val_sessions"] = val_sess
        folds_out.append(r)

    # Pop heavy arrays before JSON dump; save as separate npz cache
    proba_chunks = []
    y_chunks = []
    for f in folds_out:
        if "_proba_te" in f:
            proba_chunks.append(f.pop("_proba_te"))
            y_chunks.append(f.pop("_y_te"))
    if proba_chunks:
        cache_path = os.path.join(args.out_dir, "proba_cache.npz")
        np.savez_compressed(
            cache_path,
            proba=np.concatenate(proba_chunks, axis=0),
            y=np.concatenate(y_chunks, axis=0),
            class_names=np.array(CLASS_NAMES, dtype=object),
            mode=args.mode,
            backbone_type=args.backbone_type,
        )
        log(f"[+] wrote proba cache → {cache_path}")

    accs = [f["test_acc"] for f in folds_out]
    bals = [f["test_bal_acc"] for f in folds_out]
    summary = {
        "ckpt": args.ckpt,
        "backbone_type": args.backbone_type,
        "config": args.config,
        "mode": args.mode,
        "class_names": CLASS_NAMES,
        "n_folds": len(folds_out),
        "mean_test_acc": float(np.mean(accs)),
        "std_test_acc": float(np.std(accs)),
        "mean_test_bal_acc": float(np.mean(bals)),
        "std_test_bal_acc": float(np.std(bals)),
        "folds": folds_out,
    }
    fname = {
        "zero_shot": "zero_shot_results.json",
        "frozen_lp": "nn_probe_results.json",
        "full_ft": "nn_fullft_results.json",
    }[args.mode]
    with open(out_dir / fname, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"SUMMARY: mean_acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
        f"mean_bal_acc={np.mean(bals):.4f}±{np.std(bals):.4f}")
    log(f"results → {out_dir / fname}")


if __name__ == "__main__":
    main()
