#!/usr/bin/env python3
"""Supervised LOSO 5-class classification on NeuroNexus, starting from a
pre-trained SSL backbone (Allen pdm or IBL chandist σ=500, at 75K or 240K).

Split (Script-2 style):
  - Trial pool (60 trials/session): 12 random test trials, then train/val =
    75/25 of the remaining 48 (seed=42).
  - Held-out test session: 1 (passed via --test-session).
  - Train sessions: random.sample(other 6, int(0.8*6)) = 4 (seed=42).
  - Val sessions: the 2 leftovers.
  - Train ⇄ val ⇄ test sessions are mutually disjoint.
  - All splits use disjoint trial pools (train_tr_idx / val_tr_idx / test_tr_idx).

NN classes: {0:Cortex, 1:CA1, 2:CA2, 3:CA3, 4:DG}.
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
from scipy.signal import resample
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from transformers import (AutoFeatureExtractor, AutoModelForAudioClassification,
                          Trainer, TrainingArguments, Wav2Vec2Config)


# ── Constants ────────────────────────────────────────────────────────────
NN_REGION_NAMES = {0: "Cortex", 1: "CA1", 2: "CA2", 3: "CA3", 4: "DG"}
NN_LFP_DIR = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws4/region_decoding/data/Neuronexus/lfp')
DEFAULT_NN_SESSIONS = [
    "AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2",
    "NN_syn_01", "NN_syn_02",
]  # 7-session paper-aligned pool (NN_syn_03 excluded — Cortex-only)

VARIANTS = {
    "allen_pdm_240k": {
        "ssl_ckpt": __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026/w2v2_240k_probe_dist_v2_matched/checkpoints/ssl_wav2vec2_best.pt'),
        "display": "Allen 240K probe_dist_matched",
    },
    "allen_pdm_75k": {
        "ssl_ckpt": __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026/w2v2_75k_probe_dist_v2_matched/checkpoints/ssl_wav2vec2_best_valloss.pt'),
        "display": "Allen 75K probe_dist_matched",
    },
    "ibl_cd500_240k": {
        "ssl_ckpt": __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/benchmarks/w2v2_v2_crossneg_240k_chandist_s500_clean_6896816/checkpoints/ssl_wav2vec2_best.pt'),
        "display": "IBL 240K chandist σ=500",
    },
    "ibl_cd500_75k": {
        "ssl_ckpt": __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ray_results/benchmarks/w2v2_v2_crossneg_75k_chandist_s500_5653103/checkpoints/ssl_wav2vec2_best_valloss.pt'),
        "display": "IBL 75K chandist σ=500",
    },
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Splits (Script-2) ────────────────────────────────────────────────────
def split_trials(trial_length, fixed_test_count=12, val_size=0.25,
                 random_state=42):
    """12 random trials → test, then 75/25 of the rest → train/val."""
    all_idx = np.arange(trial_length)
    rng = np.random.RandomState(seed=random_state)
    test_tr_idx = rng.choice(all_idx, size=fixed_test_count, replace=False)
    remaining = np.setdiff1d(all_idx, test_tr_idx)
    if len(remaining) >= 4:
        train_tr_idx, val_tr_idx = train_test_split(
            remaining, test_size=val_size, random_state=random_state)
    else:
        train_tr_idx, val_tr_idx = remaining, remaining
    return (np.sort(train_tr_idx), np.sort(val_tr_idx), np.sort(test_tr_idx))


def split_sessions(test_session, all_sessions, random_state=42):
    """Script-2 80/20 session split among the 6 non-test sessions → 4 train, 2 val."""
    other = [s for s in all_sessions if s != test_session]
    rng = random.Random(random_state)
    train_sess = rng.sample(other, int(0.8 * len(other)))
    val_sess = [s for s in other if s not in train_sess]
    return train_sess, val_sess


# ── Data loading ─────────────────────────────────────────────────────────
def load_nn_session(pickle_path, session):
    """Return (signals[N,3750], labels[N], trial_idx[N], chan_id[N]) for one session."""
    fname_lfp = f"{pickle_path}/{session}_lfp.pickle"
    fname_raw = f"{pickle_path}/{session}_raw.pickle"
    fname = fname_lfp if os.path.exists(fname_lfp) else fname_raw
    with open(fname, "rb") as f:
        data = pickle.load(f)
    X, y, trial_idx, chan_id = zip(
        *[(d[0], d[1], d[2], d[3]) for d in data])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    trial_idx = np.asarray(trial_idx, dtype=np.int32)
    chan_id = np.asarray(chan_id, dtype=np.int32)
    nz = np.array([i for i, x in enumerate(X) if not np.all(x == 0)])
    return X[nz], y[nz], trial_idx[nz], chan_id[nz]


def preprocess_signals(signals, src_sr=1250, tgt_sr=16000):
    """Resample 1250→16 kHz + per-chunk z-score (matches Script-2 preprocess_data)."""
    n = len(signals)
    n_tgt = int(signals.shape[1] * tgt_sr / src_sr)
    out = np.empty((n, n_tgt), dtype=np.float32)
    eps = 1e-10
    for i, sig in enumerate(signals):
        ups = resample(sig, n_tgt)
        out[i] = ((ups - ups.mean()) / (ups.std() + eps)).astype(np.float32)
    return out


# ── Model loading ────────────────────────────────────────────────────────
def build_classifier_from_ssl_ckpt(ssl_ckpt_path, num_labels, id2label, label2id):
    """Load SSL ckpt → instantiate Wav2Vec2ForSequenceClassification with matching
    architecture, load encoder weights, leave classifier head random."""
    log(f"Loading SSL ckpt: {ssl_ckpt_path}")
    ck = torch.load(ssl_ckpt_path, map_location="cpu", weights_only=False)
    if "backbone_state_dict" not in ck or "config" not in ck:
        raise RuntimeError("ckpt missing 'backbone_state_dict' or 'config'")
    cfg_dict = dict(ck["config"])
    sd = ck["backbone_state_dict"]
    log(f"  ckpt epoch={ck.get('epoch')} val_loss={ck.get('val_loss')} "
        f"probe_acc={ck.get('probe_acc')} state-dict#={len(sd)}")

    # Filter cfg to fields Wav2Vec2Config accepts; drop SSL-specific keys.
    known = set(Wav2Vec2Config().to_dict().keys())
    hf_kwargs = {k: v for k, v in cfg_dict.items() if k in known}
    hf_config = Wav2Vec2Config(**hf_kwargs)
    hf_config.num_labels = num_labels
    hf_config.id2label = id2label
    hf_config.label2id = label2id
    # Disable SpecAugment-style masking for supervised FT — SSL configs may carry
    # apply_spec_augment=True with high mask_time_prob, which would inject random
    # hidden-state masks during classification forward passes and depress accuracy.
    hf_config.apply_spec_augment = False
    hf_config.mask_time_prob = 0.0
    hf_config.mask_time_length = 0
    hf_config.mask_feature_prob = 0.0
    hf_config.mask_feature_length = 0
    log(f"  hf_config: hidden={hf_config.hidden_size} layers={hf_config.num_hidden_layers} "
        f"conv_kernel={tuple(hf_config.conv_kernel)} (mask disabled for FT)")

    model = AutoModelForAudioClassification.from_config(hf_config)

    # Backbone state dict keys are prefixed `model.wav2vec2.*`; HF expects `wav2vec2.*`.
    filtered = {}
    for k, v in sd.items():
        if k.startswith("model."):
            filtered[k[len("model."):]] = v
        # else: skip non-`model.` keys (e.g. xyz_mean, xyz_std, xyz_stats_initialized)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    crit_prefixes = ("wav2vec2.encoder", "wav2vec2.feature_extractor",
                     "wav2vec2.feature_projection", "wav2vec2.masked_spec_embed")
    crit_missing = [k for k in missing if k.startswith(crit_prefixes)]
    if crit_missing:
        raise RuntimeError(
            f"Critical encoder weights missing from ckpt ({len(crit_missing)}): "
            f"{crit_missing[:5]}")
    log(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    log(f"  missing samples: {missing[:3]}")
    log(f"  unexpected samples: {unexpected[:3]}")
    return model


# ── Metrics ──────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": float((preds == labels).mean()),
        "bal_acc": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro",
                                    labels=list(range(5)), zero_division=0)),
    }


# ── Main run ─────────────────────────────────────────────────────────────
def run_one_loso(args):
    sessions_all = list(DEFAULT_NN_SESSIONS)
    assert args.test_session in sessions_all, (
        f"--test-session {args.test_session} not in {sessions_all}")

    variant_cfg = VARIANTS[args.variant]
    output_dir = Path(args.output_base) / args.variant / args.test_session
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"variant={args.variant} ({variant_cfg['display']})")
    log(f"test_session={args.test_session}")
    log(f"output_dir={output_dir}")

    # ── Splits ──
    train_tr_idx, val_tr_idx, test_tr_idx = split_trials(args.trial_length)
    log(f"trial split: train={len(train_tr_idx)} val={len(val_tr_idx)} "
        f"test={len(test_tr_idx)}")
    train_sess, val_sess = split_sessions(args.test_session, sessions_all)
    log(f"train sessions: {train_sess}")
    log(f"val sessions:   {val_sess}")
    log(f"test session:   {args.test_session}")

    # ── Load ──
    log("Loading NN session pickles ...")
    X_tr, y_tr = [], []
    for s in train_sess:
        Xs, ys, trs, _ = load_nn_session(NN_LFP_DIR, s)
        m = np.isin(trs, train_tr_idx)
        X_tr.append(Xs[m]); y_tr.append(ys[m])
        log(f"  train {s}: {m.sum():,}/{len(Xs):,} chunks (trials in train_tr_idx)")
    X_tr = np.concatenate(X_tr); y_tr = np.concatenate(y_tr)

    X_va, y_va = [], []
    for s in val_sess:
        Xs, ys, trs, _ = load_nn_session(NN_LFP_DIR, s)
        m = np.isin(trs, val_tr_idx)
        X_va.append(Xs[m]); y_va.append(ys[m])
        log(f"  val   {s}: {m.sum():,}/{len(Xs):,} chunks (trials in val_tr_idx)")
    X_va = np.concatenate(X_va); y_va = np.concatenate(y_va)

    Xs, ys, trs, _ = load_nn_session(NN_LFP_DIR, args.test_session)
    m_te = np.isin(trs, test_tr_idx)
    X_te = Xs[m_te]; y_te = ys[m_te]
    log(f"  test  {args.test_session}: {m_te.sum():,}/{len(Xs):,} chunks")

    log(f"final sizes: train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")
    if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
        raise RuntimeError("Empty split — aborting")

    # Class distribution
    for split_name, ys_arr in [("train", y_tr), ("val", y_va), ("test", y_te)]:
        u, c = np.unique(ys_arr, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(u, c)}
        log(f"  {split_name} class_dist: {dist}")

    # ── Preprocess ──
    log("Resampling 1250→16 kHz + per-chunk z-score ...")
    t0 = time.time()
    X_tr_up = preprocess_signals(X_tr)
    X_va_up = preprocess_signals(X_va)
    X_te_up = preprocess_signals(X_te)
    log(f"  done in {time.time()-t0:.1f}s   shape={X_tr_up.shape}")

    # Free memory
    del X_tr, X_va, X_te

    # ── Build dataset ──
    label_names = [NN_REGION_NAMES[i] for i in range(5)]
    custom_features = Features({
        "label": ClassLabel(num_classes=5, names=label_names),
        "input_values": Sequence(feature=Value(dtype="float"), length=-1),
    })

    # Avoid `.tolist()` — converts a (N, 48000) float32 array into nested Python
    # lists, which explodes memory. Pass a list of numpy arrays instead so Arrow
    # serializes each row as a list-of-float column without going through Python.
    def make_ds(X_up, y_arr):
        return Dataset.from_dict(
            {"label": y_arr.astype(np.int64).tolist(),
             "input_values": [row for row in X_up]},
            features=custom_features,
        )

    log("Building HF DatasetDict ...")
    dataset = DatasetDict({
        "train": make_ds(X_tr_up, y_tr),
        "validation": make_ds(X_va_up, y_va),
        "test": make_ds(X_te_up, y_te),
    })
    del X_tr_up, X_va_up, X_te_up

    # ── Model ──
    id2label = {str(i): NN_REGION_NAMES[i] for i in range(5)}
    label2id = {NN_REGION_NAMES[i]: str(i) for i in range(5)}
    model = build_classifier_from_ssl_ckpt(
        variant_cfg["ssl_ckpt"], 5, id2label, label2id)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model trainable params: {n_trainable:,}")

    # ── W&B (optional) ──
    if args.wandb:
        import wandb
        wandb.init(
            project="neuronexus_supervised_ft",
            entity=os.environ.get("WANDB_ENTITY"),
            name=f"{args.variant}_{args.test_session}",
            config={
                "variant": args.variant,
                "ssl_ckpt": variant_cfg["ssl_ckpt"],
                "test_session": args.test_session,
                "train_sessions": train_sess,
                "val_sessions": val_sess,
                "trial_length": args.trial_length,
                "lr": args.lr,
                "epochs": args.num_train_epochs,
                "per_device_batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
            },
            reinit=True,
        )

    # ── Trainer ──
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base")
    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_output"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.1,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="bal_acc",
        greater_is_better=True,
        fp16=True,
        report_to=("wandb" if args.wandb else "none"),
        run_name=f"{args.variant}_{args.test_session}",
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"].with_format("torch"),
        eval_dataset=dataset["validation"].with_format("torch"),
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,
    )

    log("=== Training ===")
    trainer.train()

    log("=== Test eval ===")
    test_pred = trainer.predict(dataset["test"].with_format("torch"))
    test_metrics = test_pred.metrics
    log(f"test_metrics={test_metrics}")

    # Confusion
    preds = np.argmax(test_pred.predictions, axis=1)
    cm = confusion_matrix(test_pred.label_ids, preds, labels=list(range(5)))

    # ── Save ──
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        logits=test_pred.predictions,
        labels=test_pred.label_ids,
        confusion_matrix=cm,
    )
    metrics_out = {
        "variant": args.variant,
        "ssl_ckpt": variant_cfg["ssl_ckpt"],
        "test_session": args.test_session,
        "train_sessions": train_sess,
        "val_sessions": val_sess,
        "split_seed": 42,
        "trial_length": args.trial_length,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "test_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating))
                             else v) for k, v in test_metrics.items()},
        "test_class_dist": {int(k): int((np.asarray(y_te) == k).sum())
                             for k in range(5)},
        "label_map": NN_REGION_NAMES,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    log(f"Saved {output_dir}/test_predictions.npz + metrics.json")

    if args.wandb:
        for k, v in test_metrics.items():
            if isinstance(v, (int, float, np.floating)):
                wandb.log({f"test/{k}": float(v)})
        wandb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    ap.add_argument("--test-session", required=True,
                    choices=DEFAULT_NN_SESSIONS,
                    help="Held-out NN session for LOSO eval")
    ap.add_argument("--trial-length", type=int, default=60)
    ap.add_argument("--num-train-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--output-base", type=str,
                    default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026/neuronexus_supervised_ft'))
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.set_defaults(wandb=True)
    ap.add_argument("--lfp-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    global NN_LFP_DIR
    NN_LFP_DIR = args.lfp_dir
    VARIANTS[args.variant]["ssl_ckpt"] = args.checkpoint
    run_one_loso(args)


if __name__ == "__main__":
    main()
