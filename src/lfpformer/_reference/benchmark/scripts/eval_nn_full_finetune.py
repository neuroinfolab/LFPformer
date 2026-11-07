#!/usr/bin/env python3
"""NN 5-class full fine-tune (IBL 75K chandist) — Lfp2vec protocol.

Lfp2vec NeuroNexus protocol (matches eval_nn_linear_probe.py):
  - 7-fold leave-one-session-out (NN_syn_03 excluded)
  - within-session trial split: 36 train / 12 val / 12 test trials (seed=42)
  - random session split of the 6 remaining: 4 train + 2 val (random.seed=42)
  - test = held-out session × 12 test trials only
  - primary metric = plain accuracy (Lfp2vec convention).

Mirror of eval_nn_linear_probe.py but unfreezes the backbone:
  - one fresh backbone init per fold (loaded from --ckpt each time)
  - Linear(768 → 5) head; AdamW with two LR groups (bb=1e-5, head=1e-3)
  - linear warmup + cosine decay; AMP off (per project policy)
  - early stop on val cross-entropy
  - test-time embeddings (768-d) re-extracted from the fine-tuned backbone
    and saved per fold

Outputs:
    <out>/nn_fullft_results.json
    <out>/predictions/<test>.npz   logits/preds/labels per held-out session
    <out>/embeddings/<test>.npz    post-FT 768-d embeddings for the held-out
                                   session, cached per fold
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

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


class SignalDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.signals[idx]),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
        )


def load_backbone(ckpt_path, device):
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
    log(f"  backbone loaded: missing={len(missing)} unexpected={len(unexpected)} "
        f"epoch={ck.get('epoch')}")
    backbone.to(device)
    return backbone


def encode_batch(backbone, wav, src_sr):
    if src_sr != 16000:
        wav = torchaudio.functional.resample(wav, src_sr, 16000)
    emb = backbone.encode(wav)
    if emb.ndim == 3:
        emb = emb.mean(dim=1)
    return emb


@torch.no_grad()
def extract_embeddings(backbone, loader, src_sr, device):
    backbone.eval()
    embs, labs = [], []
    for wav, y in loader:
        wav = wav.to(device, non_blocking=True)
        emb = encode_batch(backbone, wav, src_sr)
        embs.append(emb.float().cpu().numpy())
        labs.append(y.numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labs, axis=0)


def evaluate(backbone, head, loader, src_sr, device):
    backbone.eval(); head.eval()
    losses, all_logits, all_labels = [], [], []
    with torch.no_grad():
        for wav, y in loader:
            wav = wav.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            emb = encode_batch(backbone, wav, src_sr)
            logits = head(emb)
            losses.append(F.cross_entropy(logits, y, reduction="sum").item())
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(y.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    n = len(labels)
    return logits, labels, sum(losses) / max(n, 1)


def train_one_fold(args, train_sess, val_sess, test_sess,
                   sess_data, train_tr_idx, val_tr_idx, test_tr_idx,
                   device, fold_dir):
    """Train backbone+head from --ckpt init for one fold, return metrics + best state.
    sess_data[s] = (sigs, labs, trls). Trial-level filter applied here:
        train: train_sess × train_tr_idx
        val:   val_sess   × val_tr_idx
        test:  test_sess  × test_tr_idx
    """
    def slice_sigs(sess_list, trial_idx_set):
        sigs, labs = [], []
        for s in sess_list:
            sg, lb, tr = sess_data[s]
            m = np.isin(tr, trial_idx_set)
            sigs.append(sg[m]); labs.append(lb[m])
        return np.concatenate(sigs, axis=0), np.concatenate(labs, axis=0)

    train_sigs, train_labs = slice_sigs(train_sess, train_tr_idx)
    val_sigs,   val_labs   = slice_sigs(val_sess,   val_tr_idx)
    test_sigs,  test_labs  = slice_sigs([test_sess], test_tr_idx)
    log(f"  sizes: train={len(train_sigs)} val={len(val_sigs)} test={len(test_sigs)}")

    train_ds = SignalDataset(train_sigs, train_labs)
    val_ds = SignalDataset(val_sigs, val_labs)
    test_ds = SignalDataset(test_sigs, test_labs)

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        generator=g, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Fresh backbone every fold
    backbone = load_backbone(args.ckpt, device)
    head = nn.Linear(768, NUM_CLASSES).to(device)
    nn.init.xavier_uniform_(head.weight); nn.init.zeros_(head.bias)

    opt = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": args.lr_backbone,
         "weight_decay": args.weight_decay},
        {"params": head.parameters(), "lr": args.lr_head,
         "weight_decay": args.weight_decay},
    ])
    total_steps = max(1, len(train_loader)) * args.epochs
    warmup_steps = max(50, total_steps // 20)
    warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_steps)
    cosine = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps),
                               eta_min=args.lr_backbone * 0.01)
    sched = SequentialLR(opt, schedulers=[warmup, cosine],
                         milestones=[warmup_steps])
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_va_loss = float("inf")
    best_state = None
    bad = 0
    history = []
    for ep in range(args.epochs):
        backbone.train(); head.train()
        t0 = time.time()
        ep_loss = 0.0
        ep_total = 0
        for step, (wav, y) in enumerate(train_loader):
            wav = wav.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            emb = encode_batch(backbone, wav, args.sample_rate)
            logits = head(emb)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()),
                args.grad_clip)
            opt.step(); sched.step()
            ep_loss += float(loss.detach()) * len(y)
            ep_total += len(y)
            if step % 100 == 0:
                lrs = [pg["lr"] for pg in opt.param_groups]
                log(f"    ep{ep} step {step}/{len(train_loader)} "
                    f"loss={float(loss):.4f} "
                    f"lr=[{lrs[0]:.2e},{lrs[1]:.2e}]")
        ep_loss /= max(ep_total, 1)

        _, _, va_loss = evaluate(backbone, head, val_loader,
                                 args.sample_rate, device)
        elapsed = time.time() - t0
        log(f"  ep{ep:3d}  tr_loss={ep_loss:.4f}  va_loss={va_loss:.4f}  "
            f"({elapsed:.1f}s)")
        history.append({"epoch": ep, "train_loss": float(ep_loss),
                        "val_loss": float(va_loss)})

        if va_loss < best_va_loss - 1e-5:
            best_va_loss = va_loss
            best_state = {
                "backbone": {k: v.detach().cpu().clone()
                             for k, v in backbone.state_dict().items()},
                "head": {k: v.detach().cpu().clone()
                         for k, v in head.state_dict().items()},
                "epoch": ep,
            }
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            log(f"  early stop @ ep{ep+1}  best_va_loss={best_va_loss:.4f}")
            break

    # Reload best state for test eval + embedding extraction
    backbone.load_state_dict(best_state["backbone"])
    head.load_state_dict(best_state["head"])

    te_logits, te_labels, _ = evaluate(backbone, head, test_loader,
                                       args.sample_rate, device)
    preds = te_logits.argmax(axis=1)
    acc = float(accuracy_score(te_labels, preds))
    bal = float(balanced_accuracy_score(te_labels, preds))
    cm = confusion_matrix(te_labels, preds, labels=list(range(NUM_CLASSES))).tolist()

    # Save post-FT embeddings on the held-out session
    emb_te, lab_te = extract_embeddings(backbone, test_loader,
                                        args.sample_rate, device)

    # Free GPU before next fold
    del backbone, head, opt, sched, warmup, cosine
    torch.cuda.empty_cache()

    return {
        "test_acc": acc,
        "test_bal_acc": bal,
        "best_va_loss": float(best_va_loss),
        "best_epoch": int(best_state["epoch"]),
        "confusion_matrix": cm,
        "logits": te_logits,
        "preds": preds,
        "labels": te_labels,
        "emb_test": emb_te,
        "lab_test": lab_te,
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--lfp-dir", default=DEFAULT_LFP_DIR)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-rate", type=int, default=1250)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", nargs="+", default=None,
                    help="Optional subset of test sessions (default: all 7).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    emb_dir = out_dir / "embeddings"
    pred_dir.mkdir(exist_ok=True)
    emb_dir.mkdir(exist_ok=True)
    log(f"out dir: {out_dir}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    # Load all 7 sessions once (kept on CPU)
    sess_data = {}
    log("Loading 7 NN sessions ...")
    for s in NN_SESSIONS:
        sess_data[s] = load_session(args.lfp_dir, s)

    train_tr_idx, val_tr_idx, test_tr_idx = compute_trial_split(
        trial_length=60, fixed_test_count=12, val_size=0.25,
        random_state=args.seed)
    log(f"trial split (seed={args.seed}): "
        f"train={len(train_tr_idx)} val={len(val_tr_idx)} test={len(test_tr_idx)} trials")

    folds_to_run = args.folds if args.folds else NN_SESSIONS
    fold_results = []
    for test_sess in folds_to_run:
        if test_sess not in NN_SESSIONS:
            raise ValueError(f"unknown session {test_sess}")
        log(f"\n=== fold: test={test_sess} ===")
        fold_dir = out_dir / f"fold_{test_sess}"
        fold_dir.mkdir(exist_ok=True)
        rest = [s for s in NN_SESSIONS if s != test_sess]
        train_sess, val_sess = compute_session_split(rest, train_frac=0.8,
                                                     seed=args.seed)
        log(f"  train sessions ({len(train_sess)}): {train_sess}")
        log(f"  val   sessions ({len(val_sess)}): {val_sess}")

        res = train_one_fold(args, train_sess, val_sess, test_sess,
                             sess_data, train_tr_idx, val_tr_idx, test_tr_idx,
                             device, fold_dir)

        log(f"  → test_acc={res['test_acc']:.4f} (primary)  "
            f"test_bal_acc={res['test_bal_acc']:.4f} (secondary)  "
            f"(best_ep={res['best_epoch']}, va_loss={res['best_va_loss']:.4f})")

        np.savez_compressed(
            pred_dir / f"{test_sess}.npz",
            logits=res["logits"], preds=res["preds"], labels=res["labels"],
            train_sessions=np.array(train_sess), val_sessions=np.array(val_sess),
        )
        np.savez_compressed(
            emb_dir / f"{test_sess}.npz",
            emb=res["emb_test"], labels=res["lab_test"],
            train_sessions=np.array(train_sess),
        )
        fold_results.append({
            "test_session": test_sess,
            "train_sessions": train_sess,
            "val_sessions": val_sess,
            "n_train": int(sum(int(np.isin(sess_data[s][2], train_tr_idx).sum())
                                 for s in train_sess)),
            "n_val": int(sum(int(np.isin(sess_data[s][2], val_tr_idx).sum())
                              for s in val_sess)),
            "n_test": int(np.isin(sess_data[test_sess][2], test_tr_idx).sum()),
            "test_acc": res["test_acc"],
            "test_bal_acc": res["test_bal_acc"],
            "best_va_loss": res["best_va_loss"],
            "best_epoch": res["best_epoch"],
            "confusion_matrix": res["confusion_matrix"],
            "history": res["history"],
        })
        # Persist after each fold so partial results survive timeouts
        with open(out_dir / "nn_fullft_results.json", "w") as f:
            json.dump({
                "ckpt": args.ckpt,
                "method": "full_finetune",
                "class_names": CLASS_NAMES,
                "n_folds": len(fold_results),
                "mean_test_acc":
                    float(np.mean([f["test_acc"] for f in fold_results])),
                "std_test_acc":
                    float(np.std([f["test_acc"] for f in fold_results])),
                "mean_test_bal_acc":
                    float(np.mean([f["test_bal_acc"] for f in fold_results])),
                "std_test_bal_acc":
                    float(np.std([f["test_bal_acc"] for f in fold_results])),
                "folds": fold_results,
                "args": vars(args),
            }, f, indent=2)

    log(f"\nSUMMARY:  mean_acc="
        f"{np.mean([f['test_acc'] for f in fold_results]):.4f}  "
        f"mean_bal_acc="
        f"{np.mean([f['test_bal_acc'] for f in fold_results]):.4f}")
    log(f"results → {out_dir / 'nn_fullft_results.json'}")


if __name__ == "__main__":
    main()
