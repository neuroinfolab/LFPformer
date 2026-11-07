"""NeurIPS 2026 camera-ready Fig 4 — Allen panel: IBL encoder TRANSFER inputs.

The default run scores the IBL-pretrained MaskPred+ManipDistc 75k encoder.
The generic ``--transfer_root``/``--key_prefix`` interface also scores matched
IBL ChanDistc transfers without overwriting the historical ManipDistc files.
Each transfer uses a Gaussian-NLL xyz head trained on Allen lp_eval
(run_fig4cr_allen_transfer.sbatch), then exactly the paper's Allen pipeline:

  (a) bars : compute_allen_per_fold_balacc.compute_one
             (gauss_wcons_line posthoc -> 25 µm CCF voxel lookup, 116 cls,
             per-fold recall over test_fold1..5)
  (b) ROC  : plot_per_region_roc_voxel_dist_anchor_vs_baselines_allen_75k
             (same posthoc -> EDT voxel-distance macro ROC)

HistGBM/band is recomputed as a reproduction check against its published CSV.

--protocol main (DEFAULT since 2026-10-13, user: "all figures use weighted then
Kabsch"): the posthoc for bars AND ROC is the MAIN protocol (main_eval_protocol.
pool_kabsch): Gaussian 1/sigma^2 channel pooling -> per-probe Kabsch of the on-probe
geometry (data/channels.csv probe_horizontal/vertical_position) to the pooled
channels. All available methods are rescored (transfer models + ViT, AnyNet,
HistGBM, kNN band, kNN PCA-PSD, PCA-Ridge PSD) -> fig4_inputs/allen_main/. The
Allen-pretrained pdv4 anchor cannot be rescored: its predictions no longer exist
(only the paper CSV/cache survive). --protocol paper = the original gauss_wcons_line.
Container: w2v2_cuda_cu128_np2.sif (nrrd). CPU only.
"""
import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import compute_allen_per_fold_balacc as A                                  # noqa: E402
import main_eval_protocol as MP                                            # noqa: E402
import plot_per_region_roc_voxel_dist_anchor_vs_baselines_allen_75k as RA  # noqa: E402

NEURIPS = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/ray_results/Neurips_2026')
CR_DIR = (__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/plots/neurips_camera_ready_2026-10-12'))
TRANSFER_ROOT = os.path.join(NEURIPS, "allen_transfer_ibl75k_manip53")
MODELS = {
    "manip_transfer_frozen": os.path.join(TRANSFER_ROOT, "finetune_gaussian_frozen",
                                          "gaussian_predictions.npz"),
    "manip_transfer_fullft": os.path.join(TRANSFER_ROOT, "finetune_gaussian_fullft",
                                          "gaussian_predictions.npz"),
    "histgbm_band_check": os.path.join(NEURIPS, "analysis_outputs", "baseline",
                                       "preds_histgbm_band.npz"),
}
AO = os.path.join(NEURIPS, "analysis_outputs")
MAIN_BASELINES = {   # Allen baselines whose predictions still exist (all have stds)
    "vit": os.path.join(AO, "ws3_allen", "vit_rival_xyz_hp_allen", "gaussian_predictions.npz"),
    "anynet": os.path.join(AO, "ws3_allen", "anynet_rival_xyz_hp_allen", "gaussian_predictions.npz"),
    "histgbm_band": os.path.join(AO, "baseline", "preds_histgbm_band.npz"),
    "knn_band": os.path.join(AO, "baseline", "preds_knn_band.npz"),
    "knn_pca_psd": os.path.join(AO, "baseline", "preds_knn_pca_psd.npz"),
    "pca_ridge_psd": os.path.join(AO, "baseline", "preds_pca_ridge_psd.npz"),
}
CHANNELS_CSV = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/data/channels.csv')
PAPER_CSV = {"histgbm_band_check": os.path.join(
    NEURIPS, "eval_plots/per_region_balacc_75k/histgbm_band_per_region.csv")}


def per_fold_from_csv(path):
    with open(path) as f:
        rows = [[float(r[f"fold{i}"]) for i in range(1, 6)] for r in csv.DictReader(f)]
    return np.asarray(rows).mean(axis=0)


def allen_xy(channels):
    """Allen global channel id -> on-probe (x, depth) µm from data/channels.csv."""
    geo = {}
    with open(CHANNELS_CSV) as f:
        for r in csv.DictReader(f):
            geo[r["id"]] = (float(r["probe_horizontal_position"]),
                            float(r["probe_vertical_position"]))
    return np.array([geo[str(c)] for c in channels], dtype=np.float64)


def main_posthoc(npz, key, coords, probes, channels, xy):
    d = np.load(npz, allow_pickle=True)
    pred, true, stds = d["preds"], d["true"], d["stds"]
    if pred.shape != coords.shape or np.abs(coords - true).max() > 1.0:
        raise SystemExit(f"{key}: meta order mismatch vs the lp_eval meta cache")
    r = MP.pool_kabsch(pred, stds, probes, channels, xy)
    print(f"    {key}: main protocol, {len(r['first'])} channels, {r['n_ok']} probes fitted",
          flush=True)
    return r["sample"].astype(np.float32), true.astype(np.float32)


def transfer_models(args):
    return {
        f"{args.key_prefix}_frozen": os.path.join(
            args.transfer_root, "finetune_gaussian_frozen", "gaussian_predictions.npz"),
        f"{args.key_prefix}_fullft": os.path.join(
            args.transfer_root, "finetune_gaussian_fullft", "gaussian_predictions.npz"),
    }


def transfer_provenance(args, requested):
    """Validate the matched Gaussian-MLP training contract from result JSONs."""
    out = {}
    expected_lr = {"frozen": 0.0, "fullft": 5e-6}
    for variant in ("frozen", "fullft"):
        key = f"{args.key_prefix}_{variant}"
        npz = requested[key]
        result_path = os.path.join(os.path.dirname(npz), "finetune_gaussian_results.json")
        if not os.path.isfile(npz) or not os.path.isfile(result_path):
            if args.require_transfer or args.only_transfer:
                raise SystemExit(f"missing completed {variant} transfer: {npz} / {result_path}")
            continue
        with open(result_path) as f:
            result = json.load(f)
        checks = {
            "loss": result.get("loss") == "gaussian_nll",
            "lr_head": abs(float(result.get("lr_head", -1)) - 1e-3) < 1e-12,
            "lr_backbone": abs(float(result.get("lr_backbone", -1)) - expected_lr[variant]) < 1e-12,
            "lr_cnn": abs(float(result.get("lr_cnn", -1)) - expected_lr[variant]) < 1e-12,
            "batch_size": int(result.get("batch_size", -1)) == 32,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise SystemExit(f"{key}: training-contract mismatch: {failed}; {result_path}")
        out[variant] = {
            "key": key,
            "result_json": result_path,
            "checkpoint": result.get("checkpoint"),
            "best_epoch": result.get("best_epoch"),
            "epochs_trained": result.get("epochs_trained"),
            "allen_training_used": True,
            "encoder_updated": variant == "fullft",
            "decoder": "GaussianXYZHead MLP (768->512->256->128; mean + log-variance)",
            "loss": "heteroscedastic Gaussian NLL",
            "lr_head": 1e-3,
            "lr_backbone": expected_lr[variant],
            "lr_cnn": expected_lr[variant],
            "batch_size": 32,
        }
    return out


def main_protocol(args):
    requested_transfer = transfer_models(args)
    provenance = transfer_provenance(args, requested_transfer)
    requested = dict(requested_transfer)
    if not args.only_transfer:
        requested.update(MAIN_BASELINES)
    models = {k: p for k, p in requested.items() if os.path.isfile(p)}
    for k in set(requested) - set(models):
        print(f"[!] not ready, skipping: {k} ({requested[k]})", flush=True)
    fine_ids, probes, channels, coords = A.build_or_load_meta_cache()
    fold_bounds = A.build_or_load_fold_bounds()
    meta = pickle.load(open(A.ATLAS_META_PATH, "rb"))
    fid_to_acr = meta.get("fine_id_to_acr") or meta.get("fine_id_to_acronym")
    xy = allen_xy(channels)
    canon_fids = sorted(set(fine_ids.tolist()))
    acrs = [fid_to_acr.get(int(f), str(int(f))) for f in canon_fids]
    lv_bars = A.build_label_volume(canon_fids)
    if not args.skip_roc:
        canon_sids = sorted(int(k) for k in meta["fine_id_to_acr"].keys())
        lv_roc = RA.build_label_volume_sid_indexed(canon_sids)
        sid_to_idx = {s: i for i, s in enumerate(canon_sids)}
        true_roc = np.array([sid_to_idx.get(int(f), -1) for f in fine_ids], dtype=np.int64)
    summary = {
        "protocol": "main_sigma_pool_kabsch",
        "models": models,
        "bars": {},
        "transfer": {
            "objective": args.objective,
            "encoder_source": args.encoder_source,
            "checkpoint_sha256": args.checkpoint_sha256,
            "key_prefix": args.key_prefix,
            "result_root": args.transfer_root,
            "variants": provenance,
        },
    }
    vox = {}
    for key, npz in models.items():
        post, true = main_posthoc(npz, key, coords, probes, channels, xy)
        true_lbl = A.label_at_voxel(A.voxelize(true), lv_bars)
        pred_lbl = A.label_at_voxel(A.voxelize(post), lv_bars)
        mat = A.per_fold_per_region_recall(true_lbl, pred_lbl, fold_bounds, len(acrs))
        out_csv = os.path.join(args.out_dir, f"{key}_per_region.csv")
        A.write_csv(out_csv, mat, acrs, canon_fids)
        pf = per_fold_from_csv(out_csv)
        summary["bars"][key] = {"per_fold_balacc": pf.tolist(), "mean": float(pf.mean()),
                                "std_ddof1": float(pf.std(ddof=1))}
        print(f"[bars] {key}: {pf.mean():.4f} ± {pf.std(ddof=1):.4f}", flush=True)
        if not args.skip_roc:
            vox[key] = RA.voxelize_clip(post, lv_roc.shape)
    if not args.skip_roc:
        fpr_grid = np.linspace(0.0, 1.0, 401)
        res = RA.compute_all_rocs_streaming(lv_roc, len(canon_sids), vox, true_roc, fpr_grid)
        keys = list(res)
        np.savez_compressed(os.path.join(
                                args.out_dir, f"fig4cr_allen_roc_cache_{args.output_tag}.npz"),
                            keys=np.array(keys), fpr_grid=fpr_grid,
                            mean_tpr=np.stack([res[k]["mean_tpr"] for k in keys]),
                            macro_auc=np.array([res[k]["macro_auc"] for k in keys]))
        summary["roc"] = {k: {"macro_auc": res[k]["macro_auc"]} for k in keys}
        print(json.dumps(summary["roc"], indent=2), flush=True)
    with open(os.path.join(args.out_dir, f"fig4cr_allen_summary_{args.output_tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["main", "paper"], default="main")
    ap.add_argument("--transfer_root", default=TRANSFER_ROOT,
                    help="frozen/full-FT result root")
    ap.add_argument("--key_prefix", default="manip_transfer",
                    help="output key prefix; e.g. anchor_transfer for ChanDistc")
    ap.add_argument("--only_transfer", action="store_true",
                    help="score only the two requested transfer variants, not shared baselines")
    ap.add_argument("--require_transfer", action="store_true",
                    help="fail instead of skipping if either transfer variant is incomplete")
    ap.add_argument("--output_tag", default=None,
                    help="suffix for compact ROC/summary files (default: main or transfer)")
    ap.add_argument("--objective", default="MaskPred+ManipDistc")
    ap.add_argument("--encoder_source", default="IBL 75k")
    ap.add_argument("--checkpoint_sha256", default="")
    ap.add_argument("--out_dir", default=None,
                    help="default fig4_inputs/allen_main (main) or fig4_inputs/allen (paper)")
    ap.add_argument("--skip_roc", action="store_true")
    args = ap.parse_args()
    if args.output_tag is None:
        args.output_tag = "main" if args.protocol == "main" else "transfer"
    if args.out_dir is None:
        args.out_dir = os.path.join(CR_DIR, "fig4_inputs",
                                    "allen_main" if args.protocol == "main" else "allen")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.protocol == "main":
        return main_protocol(args)

    requested_models = transfer_models(args)
    provenance = transfer_provenance(args, requested_models)
    if not args.only_transfer:
        requested_models["histgbm_band_check"] = MODELS["histgbm_band_check"]
    models = {k: p for k, p in requested_models.items() if os.path.isfile(p)}
    for k in set(requested_models) - set(models):
        print(f"[!] not ready, skipping: {k} ({requested_models[k]})", flush=True)

    fine_ids, probes, channels, coords = A.build_or_load_meta_cache()
    fold_bounds = A.build_or_load_fold_bounds()
    meta = pickle.load(open(A.ATLAS_META_PATH, "rb"))
    fid_to_acr = meta.get("fine_id_to_acr") or meta.get("fine_id_to_acronym")

    # ── (a) bars — identical to compute_allen_per_fold_balacc.main ──
    canon_fids = sorted(set(fine_ids.tolist()))
    acrs = [fid_to_acr.get(int(f), str(int(f))) for f in canon_fids]
    lv_bars = A.build_label_volume(canon_fids)
    summary = {
        "protocol": "paper_gauss_wcons_line",
        "models": models,
        "bars": {},
        "transfer": {
            "objective": args.objective,
            "encoder_source": args.encoder_source,
            "checkpoint_sha256": args.checkpoint_sha256,
            "key_prefix": args.key_prefix,
            "result_root": args.transfer_root,
            "variants": provenance,
        },
    }
    for key, npz in models.items():
        mat = A.compute_one(npz, key, coords, probes, channels, fold_bounds,
                            lv_bars, acrs)
        out_csv = os.path.join(args.out_dir, f"{key}_per_region.csv")
        A.write_csv(out_csv, mat, acrs, canon_fids)
        pf = per_fold_from_csv(out_csv)      # same reader semantics as the plot
        summary["bars"][key] = {"per_fold_balacc": pf.tolist(),
                                "mean": float(pf.mean()),
                                "std_ddof1": float(pf.std(ddof=1))}
        msg = f"[bars] {key}: {pf.mean():.4f} ± {pf.std(ddof=1):.4f}"
        if key in PAPER_CSV:
            ref = per_fold_from_csv(PAPER_CSV[key])
            summary["bars"][key]["paper_mean"] = float(ref.mean())
            msg += f"   (paper CSV {ref.mean():.4f}, max|Δfold|={np.abs(ref-pf).max():.2e})"
        print(msg, flush=True)

    # ── (b) ROC — identical to the Allen voxel-dist ROC script ──
    if not args.skip_roc:
        canon_sids = sorted(int(k) for k in meta["fine_id_to_acr"].keys())
        lv_roc = RA.build_label_volume_sid_indexed(canon_sids)
        sid_to_idx = {s: i for i, s in enumerate(canon_sids)}
        true_lbl = np.array([sid_to_idx.get(int(f), -1) for f in fine_ids], dtype=np.int64)
        fpr_grid = np.linspace(0.0, 1.0, 401)
        vox = {k: RA.load_model_pred_voxels(p, coords, probes, channels, k,
                                            "gauss_wcons_line", lv_roc.shape)
               for k, p in models.items()}
        res = RA.compute_all_rocs_streaming(lv_roc, len(canon_sids), vox, true_lbl, fpr_grid)
        keys = list(res)
        np.savez_compressed(
            os.path.join(args.out_dir, f"fig4cr_allen_roc_cache_{args.output_tag}.npz"),
            keys=np.array(keys), fpr_grid=fpr_grid,
            mean_tpr=np.stack([res[k]["mean_tpr"] for k in keys]),
            macro_auc=np.array([res[k]["macro_auc"] for k in keys]))
        summary["roc"] = {k: {"macro_auc": res[k]["macro_auc"]} for k in keys}
        # reproduction check vs the paper cache (HistGBM entry)
        old = np.load(os.path.join(NEURIPS, "neurips_2026_plots", "allen_75k_roc_voxel_panels",
                                   "roc_voxeldist_cache_allen_75k_gauss_wcons_line.npz"),
                      allow_pickle=True)
        ok = [str(k) for k in old["keys"]]
        for cand in ("histgbm_band", "baseline_histgbm_band"):
            if cand in ok and "histgbm_band_check" in res:
                summary["roc"]["histgbm_band_check"]["paper_macro_auc"] = \
                    float(old["macro_auc"][ok.index(cand)])
        print(json.dumps(summary["roc"], indent=2), flush=True)

    with open(os.path.join(args.out_dir, f"fig4cr_allen_summary_{args.output_tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
