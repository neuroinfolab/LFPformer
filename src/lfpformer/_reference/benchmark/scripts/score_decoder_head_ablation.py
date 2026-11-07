#!/usr/bin/env python3
"""Score the Gaussian-vs-MSE head ablation (NeurIPS'26 P2) for every decoder arm.

Reuses the canonical helpers in kabsch_gauss_chanpool_eval so the numbers are
computed exactly like the existing decoder-capacity ladder:

  raw    : per-SAMPLE (chunk-row) RMSE, LH-folded, NO alignment   <- PRIMARY
  gavg   : chunk->channel pooling, LH-folded, no alignment
  kabsch : pooled -> per-probe Kabsch (ORACLE: fits on true coords), LH-folded

Pooling mode per head:
  gauss -> inverse-variance ("sigma"), the learned heteroscedastic sigma
  mse   -> unweighted "mean" (an MSE head has no sigma; it writes zeros/constants)

Arms discovered on disk:
  perchannel_decoders_{scale}_whit1p0[_mse]/{linear,mlp}_s{seed}/predictions
  stdz_frozen_{scale}_sslraw/whit_a1.0[_mse]_s{seed}/predictions

Usage:
  python3 score_decoder_head_ablation.py [--scales 25k 301k 1M] [--out <json>]
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kabsch_gauss_chanpool_eval as K

NM = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026/pathb_newmodels')


def score_arm(pred_dir, head):
    """(raw, gavg, kabsch) in um for one seed's prediction dir, or None."""
    K.PRED_DIR = pred_dir
    if not any(os.path.exists(os.path.join(pred_dir, f"test_fold_{f}_predictions.npz"))
               for f in range(5)):
        return None
    mu, sig, true, pids, chans = K.load_all()
    raw = K.rmse_3d(mu, true, fold=True)                     # per-sample, unaligned
    mode = "sigma" if head == "gauss" else "mean"
    P, Q, W, T, PID = K.pool_channels(mu, sig, true, pids, chans, mode)
    gavg = K.rmse_3d(Q, T, fold=True)
    refined, n_ok, _ = K.per_probe_kabsch(P, Q, W, PID, use_w=False)
    kab = K.rmse_3d(refined, T, fold=True)
    return dict(raw=raw, gavg=gavg, kabsch=kab, n_probes=n_ok)


def arm_dirs(scale, head):
    """[(name, seed, pred_dir)] for every decoder at one scale/head."""
    suff = "" if head == "gauss" else "_mse"
    out = []
    pc = f"{NM}/perchannel_decoders_{scale}_whit1p0{suff}"
    for kind in ("linear", "mlp"):
        for seed in (1, 2, 3, 4, 5, 6, 7, 42):
            d = f"{pc}/{kind}_s{seed}/predictions"
            if os.path.isdir(d):
                out.append((kind, seed, d))
    tf = f"{NM}/stdz_frozen_{scale}_sslraw"
    for seed in (1, 2, 3, 4, 5, 6, 7, 42):
        d = f"{tf}/whit_a1.0{suff}_s{seed}/predictions"
        if os.path.isdir(d):
            out.append(("transformer", seed, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=["25k", "75k", "301k", "1M"])
    ap.add_argument("--heads", nargs="+", default=["gauss", "mse"])
    ap.add_argument("--out", default=f"{NM}/decoder_head_ablation.json")
    a = ap.parse_args()

    res = {}
    for scale in a.scales:
        for head in a.heads:
            for name, seed, d in arm_dirs(scale, head):
                s = score_arm(d, head)
                if s is None:
                    continue
                res.setdefault(f"{name}_{scale}_{head}", []).append(dict(seed=seed, **s))
                print(f"  {name:11s} {scale:5s} {head:5s} s{seed:<3d} "
                      f"raw={s['raw']:7.1f}  gavg={s['gavg']:7.1f}  "
                      f"kabsch={s['kabsch']:7.1f}", flush=True)

    agg = {}
    for k, v in res.items():
        agg[k] = {"n": len(v), "seeds": sorted(x["seed"] for x in v)}
        for m in ("raw", "gavg", "kabsch"):
            arr = np.array([x[m] for x in v], float)
            agg[k][f"{m}_mean"] = float(arr.mean())
            agg[k][f"{m}_sd"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        agg[k]["per_seed"] = v
    json.dump(agg, open(a.out, "w"), indent=1)
    print(f"\n[wrote] {a.out}")

    print("\nRAW per-sample um (PRIMARY, no alignment)")
    print(f"{'arm':14s} " + " ".join(f"{s:>16s}" for s in a.scales))
    for name in ("linear", "mlp", "transformer"):
        for head in a.heads:
            cells = []
            for scale in a.scales:
                r = agg.get(f"{name}_{scale}_{head}")
                cells.append(f"{r['raw_mean']:8.1f}±{r['raw_sd']:<5.1f}" if r else f"{'—':>16s}")
            print(f"{name+' '+head:14s} " + " ".join(cells))


if __name__ == "__main__":
    main()
