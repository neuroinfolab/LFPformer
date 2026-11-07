"""Diagnostic: is the 1M train/val SSL gap a memorization gap or a measurement artifact?

Loads the latest checkpoint and computes ssl_forward loss under IDENTICAL conditions
(eval mode, same masking machinery) on:
  - TRAIN subset in EVAL mode   (train_eval)
  - TRAIN subset in TRAIN mode   (train_train; reproduces the logged "ssl=" number)
  - VAL subset   in EVAL mode    (val; reproduces the logged "val=" number)

Interpretation:
  train_eval ~= val   -> the gap was within-epoch optimism + dropout/train-mode. NOT memorization.
  train_eval <<  val   -> genuine generalization gap.
"""
import os, sys, argparse, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

# import path identical to the training launcher
sys.path.insert(0, os.environ["BENCH_ROOT"])
sys.path.insert(0, os.environ["ALPHABRAIN_ROOT"])

from utils.preprocessing import preprocess_batch  # reuse exact preprocessing
from backbones import get_backbone
from blind_localization.data.lazyloader_dataset import CompactDataset


def run_loss(backbone_module, loader, device, n_batches, train_mode, seed):
    """Mean ssl_forward loss over n_batches. train_mode toggles .train()/.eval()."""
    backbone_module.train(train_mode)
    torch.manual_seed(seed); np.random.seed(seed)  # make masking realization comparable
    losses = []
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            x = batch[0].float().to(device)
            x = preprocess_batch(x, backbone_module.needs_resampling,
                                 backbone_module.needs_normalization)
            vl = backbone_module.ssl_forward(x)
            if vl is not None and torch.isfinite(vl):
                losses.append(float(vl.item()))
    return float(np.mean(losses)) if losses else float("nan"), len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--compact_data_path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["channel_distance"] = True
    config["rand_init"] = True  # ckpt overwrites all weights; skip HF download (cache locked)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    backbone = get_backbone(config["backbone"], config).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    print(f"Loaded ckpt epoch={ckpt.get('epoch')} "
          f"best_val_loss={ckpt.get('best_val_loss'):.2f}", flush=True)

    compact_kwargs = dict(
        include_labels=True, dataset=config.get("data", "IBL"), verbose=False,
        atlas_depth=int(config.get("atlas_depth", 5)), already_resampled=False,
        gpu_resample=True, input_sr=1250,
        model_sr=16000 if backbone.needs_resampling else 1250,
    )
    train_ds = CompactDataset(compact_dir=os.path.join(args.compact_data_path, "train"), **compact_kwargs)
    val_ds   = CompactDataset(compact_dir=os.path.join(args.compact_data_path, "val"),   **compact_kwargs)
    print(f"train_ds={len(train_ds)}  val_ds={len(val_ds)}", flush=True)

    # plain shuffled loaders (ssl_forward only needs the signal in batch[0])
    g = torch.Generator(); g.manual_seed(0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, generator=g, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, drop_last=True)

    nb = args.n_batches
    train_eval,  ne1 = run_loss(backbone, train_loader, device, nb, train_mode=False, seed=123)
    train_train, ne2 = run_loss(backbone, train_loader, device, nb, train_mode=True,  seed=123)
    val_eval,    ne3 = run_loss(backbone, val_loader,   device, nb, train_mode=False, seed=123)

    print("\n================ TRAIN/VAL GAP DIAGNOSTIC ================", flush=True)
    print(f"  TRAIN (eval mode, no-grad) : {train_eval:8.1f}   [{ne1} batches]")
    print(f"  TRAIN (train mode)         : {train_train:8.1f}   [{ne2} batches]  <- matches logged ssl=")
    print(f"  VAL   (eval mode, no-grad) : {val_eval:8.1f}   [{ne3} batches]  <- matches logged val=")
    print("---------------------------------------------------------")
    if np.isfinite(train_eval) and np.isfinite(val_eval):
        gap = val_eval - train_eval
        ratio = val_eval / train_eval if train_eval else float("nan")
        print(f"  GENUINE gap (val_eval - train_eval) = {gap:8.1f}   ratio={ratio:.2f}x")
        print(f"  Dropout/train-mode effect (train_train - train_eval) = {train_train - train_eval:8.1f}")
        if abs(ratio - 1.0) < 0.15:
            print("  => train_eval ~= val  : NOT a memorization gap (measurement artifact).")
        else:
            print("  => train_eval still << val : genuine generalization gap.")
    print("=========================================================", flush=True)


if __name__ == "__main__":
    main()
