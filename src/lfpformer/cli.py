import argparse, json, os, runpy, sys
from pathlib import Path
import numpy as np
from .pooling import pool_channels
from .geometry import fit_probe_geometry, project_probe_lines
from .metrics import localization_metrics, balanced_accuracy
from .paths import REFERENCE, activate_reference

ROUTES = {
    "pretrain": ("pretraining", "scripts/train_ssl.py"),
    "pretrain-chandistc": ("pretraining", "scripts/train_ssl_channel_dist.py"),
    "pretrain-manipdistc": ("pretraining", "scripts/train_ssl_manip_dist.py"),
    "ibl-preprocess": ("preprocessing", "ibl/raw_data_chunck_fullrec.py"),
    "ibl-compact": ("preprocessing", "ibl/build_atlas_dataset.py"),
    "ibl-eval-split": ("preprocessing", "ibl/build_eval_dataset.py"),
    "ibl-finalize": ("preprocessing", "ibl/finalize_memmap_split.py"),
    "ibl-val-subset": ("preprocessing", "ibl/build_val_subset.py"),
    "ibl-memmap": ("preprocessing", "ibl/convert_blocks_to_memmap.py"),
    "allen-compact": ("preprocessing", "allen/build_atlas_dataset.py"),
    "extract-sequences": ("benchmark", "scripts/extract_probe_sequences.py"),
    "extract-allen-sequences": (
        "benchmark",
        "scripts/extract_probe_sequences_allen.py",
    ),
    "train-transformer": ("benchmark", "scripts/train_probe_sequence_ext.py"),
    "train-linear": ("benchmark", "scripts/linear_head_patched.py"),
    "train-allen-mlp": ("benchmark", "scripts/finetune_xyz_gaussian.py"),
    "train-manip-linear": ("benchmark", "scripts/train_xyz_finetune_manip.py"),
    "eval-manip-linear": ("benchmark", "scripts/eval_manip_ft.py"),
    "neuronexus-unified": ("benchmark", "scripts/eval_nn_3mode_unified.py"),
    "neuronexus-loso": ("benchmark", "scripts/train_neuronexus_supervised_loso.py"),
    "neuronexus-embed": ("benchmark", "scripts/embed_predict_neuronexus.py"),
    "neuronexus-knn": ("benchmark", "scripts/knn_neuronexus_cv.py"),
    "neuronexus-refine": ("benchmark", "scripts/refine_neuronexus_posthoc.py"),
}


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save(path, **data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="lfpformer",
        description="LFPformer preprocessing, embeddings and localization evaluation",
    )
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("chunk")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--probe-id", required=True)
    c.add_argument("--sample-rate", type=int, default=1250)
    c.add_argument("--seconds", type=float, default=3.0)
    c.add_argument("--stride", type=float)
    c.add_argument("--start", type=float, default=0.0)
    c = sub.add_parser("embed")
    c.add_argument("--input", required=True)
    c.add_argument("--config", required=True)
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--batch-size", type=int, default=32)
    c.add_argument("--device", default="cpu")
    c = sub.add_parser("postprocess")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument(
        "--method", choices=["raw", "pool", "ibl-kabsch", "allen-line"], required=True
    )
    c.add_argument("--orientation", choices=["ibl", "free"], default="ibl")
    c = sub.add_parser("score")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--no-fold", action="store_true")
    c.add_argument("--midline", type=float, default=5739.0)
    c = sub.add_parser("atlas-labels")
    c.add_argument("--input", required=True)
    c.add_argument("--volume", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--voxel-um", type=float, default=25.0)
    c = sub.add_parser("balanced-accuracy")
    c.add_argument("--input", required=True)
    c.add_argument("--classes", required=True)
    c.add_argument("--output", required=True)
    c = sub.add_parser(
        "reference", help="Run an included paper implementation with its original CLI"
    )
    c.add_argument("name", choices=sorted(ROUTES))
    c.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if a.command == "reference":
        stack, rel = ROUTES[a.name]
        base = activate_reference(stack)
        if stack == "preprocessing":
            sys.path.insert(0, str((base / rel).parent))
        sys.argv = [str(base / rel)] + (a.args[1:] if a.args[:1] == ["--"] else a.args)
        runpy.run_path(str(base / rel), run_name="__main__")
        return
    if a.command == "chunk":
        from .chunking import chunk_lfp

        rows = list(
            chunk_lfp(np.load(a.input), a.sample_rate, a.seconds, a.stride, a.start)
        )
        if not rows:
            p.error("No complete windows")
        signals, ch, t = zip(*rows)
        save(
            a.output,
            signals=np.stack(signals),
            channel_ids=ch,
            sample_starts=t,
            probe_ids=np.repeat(a.probe_id, len(rows)),
        )
        return
    if a.command == "embed":
        from .inference import extract_embeddings

        extract_embeddings(
            a.input, a.config, a.checkpoint, a.output, a.batch_size, a.device
        )
        return
    d = load(a.input)
    if a.command == "postprocess":
        mu = d.get("pred_xyz_gauss", d.get("mu"))
        sigma = d.get("pred_sigma_gauss", d.get("sigma"))
        ids = d["probe_ids"]
        ch = d["channel_ids"]
        if mu is None:
            p.error("Missing mu or pred_xyz_gauss")
        if a.method == "raw":
            save(a.output, **d)
            return
        z = pool_channels(
            mu,
            ids,
            ch,
            sigma,
            variance_rule="add" if a.method == "allen-line" else "clip",
        )
        q = z["mu"]
        first = z["first"]
        inv = z["inverse"]
        extra = {}
        for k in ["true_xyz", "geometry_xyz", "true_labels"]:
            if k in d:
                if not np.allclose(d[k], d[k][first][inv], atol=1e-3):
                    p.error(k + " varies within a physical channel")
                extra[k] = d[k][first]
        if a.method == "ibl-kabsch":
            if "geometry_xyz" not in extra:
                p.error("Kabsch requires known geometry_xyz; never substitute true_xyz")
            q, fitted = fit_probe_geometry(
                extra["geometry_xyz"], q, z["probe_ids"], a.orientation
            )
            extra["n_fitted_probes"] = fitted
        elif a.method == "allen-line":
            q = project_probe_lines(q, z["probe_ids"])
        save(
            a.output,
            mu=q,
            probe_ids=z["probe_ids"],
            channel_ids=z["channel_ids"],
            counts=z["counts"],
            window_to_channel=inv,
            **extra
        )
        return
    if a.command == "score":
        result = localization_metrics(
            d.get("mu", d.get("pred_xyz_gauss")),
            d["true_xyz"],
            d["probe_ids"],
            not a.no_fold,
            a.midline,
        )
    elif a.command == "atlas-labels":
        from .atlas import lookup_labels

        pred = lookup_labels(
            d.get("mu", d.get("pred_xyz_gauss")), np.load(a.volume), a.voxel_um
        )
        save(a.output, **{**d, "pred_labels": pred})
        return
    else:
        result = balanced_accuracy(
            d["true_labels"], d["pred_labels"], json.loads(Path(a.classes).read_text())
        )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
