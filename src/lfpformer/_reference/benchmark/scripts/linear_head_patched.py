#!/usr/bin/env python3
"""Per-channel decoder-capacity ablation on frozen whit_a1.0 backbones (4 scales).

Reviewer response: the fig_rmse_vs_scale_whit_a1p0 point uses a strong probe-seq
TRANSFORMER decoder. This shows the SAME frozen representation decoded by simpler
PER-CHANNEL heads, so the arms are comparable end-to-end:

  linear : nn.Linear(768, 6)                -> (mu, log var)
  mlp    : 768 -> 256 -> ReLU -> 6          -> (mu, log var)

Both are GAUSSIAN (heteroscedastic) heads trained with Gaussian NLL, so every arm
emits `pred_sigma_gauss` and can go through the canonical
inverse-variance channel pooling -> per-probe Kabsch pipeline
(kabsch_gauss_chanpool_eval.pool_channels(mode="sigma") + per_probe_kabsch),
exactly like the transformer arm. Seeds match the transformer (8).

`--head mse` runs the SAME trunks/optimiser/seeds/early-stopping with a plain 3-D
MSE loss instead (the P2 reviewer ablation "Gaussian heteroscedastic head vs direct
3D MSE regression"). ONLY the loss changes. MSE arms have no learned sigma, so the
written `pred_sigma_gauss` is the constant per-axis train SD -> inverse-variance
pooling degenerates EXACTLY to the unweighted chunk->channel mean. Output dirs get
an `_mse` suffix.

Targets are z-scored on TRAIN only; mu and sigma are de-normalised back to um.

Input : probe_seq_cache_{scale}_sslraw_whit1.0/{probe_train,probe_val,test_fold_*}_seq.npz
Output: <out>/{linear,mlp}_s<seed>/predictions/test_fold_<i>_predictions.npz
        keys: pred_xyz, pred_xyz_gauss (mu), pred_sigma_gauss, true_xyz, meta_str
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn

NM = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/gcp_neuroframe/eval_out')
TEST_FOLDS = [f"test_fold_{i}" for i in range(5)]
CACHE = None


def load(fn):
    z = np.load(os.path.join(CACHE, fn + ".npz"), allow_pickle=True)
    return (z["emb"].astype(np.float32), z["true_xyz"].astype(np.float32),
            z["meta_str"])


class GaussHead(nn.Module):
    """Emits (mu, log_var) for 3-D coords.

    kind='linear' -> single nn.Linear.
    kind='mlp'    -> mlp_layers hidden ReLU blocks (default 1: 768→256→out).

    head='mse' keeps the SAME mu trunk but drops the log_var branch (returns a
    constant 0 log_var, i.e. sigma=1 in normalised space) so ONLY the loss changes.
    """

    def __init__(self, kind, d=768, h=256, head="gauss", mlp_layers=1):
        super().__init__()
        self.head = head
        if kind == "linear":
            self.body = nn.Identity()
            din = d
        else:
            if mlp_layers < 1:
                raise ValueError(f"mlp_layers must be >=1, got {mlp_layers}")
            blocks = []
            din = d
            width = h
            for i in range(mlp_layers):
                blocks += [nn.Linear(din, width), nn.ReLU()]
                din = width
                width = max(width // 2, 32)
            self.body = nn.Sequential(*blocks)
        self.mu = nn.Linear(din, 3)
        self.log_var = nn.Linear(din, 3) if head == "gauss" else None

    def forward(self, x):
        z = self.body(x)
        mu = self.mu(z)
        if self.head != "gauss":
            return mu, torch.zeros_like(mu)
        # clamp keeps NLL finite early in training
        return mu, self.log_var(z).clamp(-10.0, 10.0)


def gauss_nll(mu, log_var, y):
    return (0.5 * (log_var + (y - mu) ** 2 / log_var.exp())).mean()


def mse_loss(mu, log_var, y):        # log_var ignored; signature matches gauss_nll
    return ((y - mu) ** 2).mean()


def train_probe(kind, Xtr, Ytr, Xva, Yva, device, seed, epochs=600, patience=60,
                head="gauss", mlp_layers=1):
    torch.manual_seed(seed)
    crit = gauss_nll if head == "gauss" else mse_loss
    model = GaussHead(kind, d=Xtr.shape[1], head=head, mlp_layers=mlp_layers).to(device)  # 768 base, 1024 large
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs), eta_min=1e-5)
    Xtr, Ytr, Xva, Yva = (t.to(device) for t in (Xtr, Ytr, Xva, Yva))
    n, bs = Xtr.shape[0], 4096
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            mu, lv = model(Xtr[b])
            crit(mu, lv, Ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mu_v, lv_v = model(Xva)
            vl = crit(mu_v, lv_v, Yva).item()
        sched.step()
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 7, 42])
    ap.add_argument("--scale", default="75k")
    ap.add_argument("--cache-dir", required=True, help="Whitened train/val/test *_seq.npz cache")
    ap.add_argument("--kinds", nargs="+", default=["linear", "mlp"])
    ap.add_argument("--head", default="gauss", choices=["gauss", "mse"],
                    help="gauss = heteroscedastic NLL (default); mse = plain 3-D MSE")
    ap.add_argument("--mlp_layers", type=int, default=1,
                    help="Number of hidden ReLU layers for kind=mlp (1=legacy 768→256→out).")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    CACHE = a.cache_dir
    suff = "" if a.head == "gauss" else "_mse"
    layer_tag = f"_mlplyr{a.mlp_layers}" if ("mlp" in a.kinds and a.mlp_layers != 1) else ""
    a.out = a.out or f"{NM}/perchannel_decoders_{a.scale}_whit1p0{suff}{layer_tag}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device} | scale={a.scale} | head={a.head} | "
          f"mlp_layers={a.mlp_layers}", flush=True)

    Xtr, Ytr_um, _ = load("probe_train_seq")
    Xva, Yva_um, _ = load("probe_val_seq")
    mu_t, sd_t = Ytr_um.mean(0), Ytr_um.std(0) + 1e-8      # TRAIN-only stats
    Ytr = torch.from_numpy((Ytr_um - mu_t) / sd_t)
    Yva = torch.from_numpy((Yva_um - mu_t) / sd_t)
    Xtr_t, Xva_t = torch.from_numpy(Xtr), torch.from_numpy(Xva)
    mu_d = torch.from_numpy(mu_t).to(device)
    sd_d = torch.from_numpy(sd_t).to(device)

    test = {sf: load(sf + "_seq") for sf in TEST_FOLDS}

    for kind in a.kinds:
        for seed in a.seeds:
            model = train_probe(kind, Xtr_t, Ytr, Xva_t, Yva, device, seed,
                                head=a.head, mlp_layers=a.mlp_layers)
            tag = f"{kind}_s{seed}"
            if kind == "mlp" and a.mlp_layers != 1:
                tag = f"{kind}_L{a.mlp_layers}_s{seed}"
            od = os.path.join(a.out, tag, "predictions")
            os.makedirs(od, exist_ok=True)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[params] {kind} mlp_layers={a.mlp_layers} -> {n_params:,}",
                  flush=True)
            with open(os.path.join(a.out, tag, "params.txt"), "w") as pf:
                pf.write(f"{n_params}\n")
            torch.save(
                {"model": model.state_dict(), "kind": kind, "mlp_layers": a.mlp_layers,
                 "head": a.head, "n_params": n_params, "seed": seed,
                 "mu_t": mu_t, "sd_t": sd_t},
                os.path.join(a.out, tag, "best.pt"),
            )
            model.eval()
            # Dump probe_val + test folds so capacity ablations can report val.
            dump = {"probe_val": load("probe_val_seq")}
            for sf in TEST_FOLDS:
                dump[sf] = test[sf]
            for sf, (Xte, Yte_um, meta) in dump.items():
                with torch.no_grad():
                    m, lv = model(torch.from_numpy(Xte).to(device))
                    pred = (m * sd_d + mu_d).cpu().numpy().astype(np.float32)
                    sig = ((0.5 * lv).exp() * sd_d).cpu().numpy().astype(np.float32)
                np.savez(os.path.join(od, f"{sf}_predictions.npz"),
                         pred_xyz=pred, pred_xyz_gauss=pred,
                         pred_sigma_gauss=sig,
                         true_xyz=Yte_um.astype(np.float32), meta_str=meta)
            print(f"[done] {kind} s{seed} -> {od}", flush=True)


if __name__ == "__main__":
    main()
