#!/usr/bin/env python3
"""Matched direct-XYZ ProbeSeq: vary only availability of manipulator coordinates."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import model_base as base

ROOT = Path(__file__).resolve().parents[1]
NM = ROOT.parent
SCRIPTS = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/Lfp2vec_benchmarks/scripts'))
SEEDS = [1, 2, 3, 4, 5, 6, 7, 42]
ARMS = ["lfp_only", "lfp_manip"]
CONFIG = {
    "cache_dir": str(NM / "probe_seq_cache_75k_manip_sslraw"),
    "manip_dir": __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/one_cache/manip_xyz/eval_v2_100k'),
    "encoder": str(NM / "init_ckpts/init_75k_manipdistc_best_valloss_ep53.pt"),
    "subject_map": str(SCRIPTS / ".cache_ibl_session_to_subject_eval_v2_trainval.json"),
    "subject_val_frac": 0.2, "split_seed": 0,
    "coordinate_convention": "LH-folded manipulator inputs and training targets; AP/DV unchanged; ML=m-abs(ML-m)",
    "parameterization": "direct", "decompose": False, "gaussian": True,
    "max_len": 4096, "batch_size": 8, "epochs": 200,
    "d_model": 256, "nhead": 8, "layers": 4, "ff": 512,
    "dropout": 0.1, "lr": 0.0003, "weight_decay": 0.0001,
    "early_stop": 0, "select_by": "val", "test_every": 0,
    "schedule": "constant", "grad_clip": 1.0, "precision": "float32",
    "normalization": "per-feature center and one global RMS scale, fitted only to training-subject rows",
    "cohort": "same finite-micromanipulator rows in both arms",
    "input_control": "same H+3+2 projection and initialization; three manipulator inputs are zero in lfp_only",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_hash(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def torch_save(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def load_trainval(output):
    a = SimpleNamespace(**CONFIG, out_dir=str(output))
    tr = base.ds_for(a.cache_dir, a.manip_dir, "probe_train", a.max_len)
    va = base.ds_for(a.cache_dir, a.manip_dir, "probe_val", a.max_len)
    base.subject_held_out_split(tr, va, a)
    return tr, va


def ds_signature(ds):
    metas = [str(m) for item in ds.items for m in item["meta"]]
    return {"sequences": len(ds), "rows": len(metas), "meta_sha256": text_hash(metas)}


class MatchedDirectProbeSeq(base.ManipProbeSequenceModel):
    def __init__(self, hidden, mean, scale, arm):
        super().__init__(hidden, CONFIG["d_model"], CONFIG["nhead"], CONFIG["layers"],
                         CONFIG["ff"], CONFIG["dropout"], decompose=False,
                         parameterization="direct", gaussian=True)
        assert arm in ARMS
        self.arm = arm
        self.register_buffer("embedding_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("embedding_scale", torch.as_tensor(scale, dtype=torch.float32))
        # Same direct-output initialization as the selected LFP-only ProbeSeq.
        torch.nn.init.zeros_(self.resid_head.weight)
        torch.nn.init.zeros_(self.resid_head.bias)

    def forward(self, emb, manip, geom, depth, mask):
        emb = (emb - self.embedding_mean) / self.embedding_scale
        manip = base.fold_lr_t(manip)
        if self.arm == "lfp_only":
            manip = torch.zeros_like(manip)
        return super().forward(emb, manip, geom, depth, mask)


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def normalization(tr):
    n = 0
    total = np.zeros(tr.H, np.float64)
    total_sq = np.zeros(tr.H, np.float64)
    for item in tr.items:
        e = item["emb"].astype(np.float64)
        total += e.sum(0)
        total_sq += (e * e).sum(0)
        n += len(e)
    mean = total / n
    variance = np.maximum(total_sq / n - mean * mean, 0)
    scale = float(np.sqrt(variance.mean()))
    assert scale > 0 and np.isfinite(scale)
    return mean.astype(np.float32), np.float32(scale)


def prepare():
    tr, va = load_trainval(ROOT)
    mean, scale = normalization(tr)
    np.savez(ROOT / "normalization.npz", mean=mean, scale=scale)
    split = json.loads((ROOT / "subject_split.json").read_text())
    smap = json.loads(Path(CONFIG["subject_map"]).read_text())
    subjects = lambda ds: sorted({smap[str(it["meta"][0]).split("_")[0]] for it in ds.items})
    split["train_subjects"], split["val_subjects"] = subjects(tr), subjects(va)
    assert not set(split["train_subjects"]) & set(split["val_subjects"])
    train_meta = {str(m) for it in tr.items for m in it["meta"]}
    val_meta = {str(m) for it in va.items for m in it["meta"]}
    assert not train_meta & val_meta
    test_probes, test_subjects, test_rows = set(), set(), 0
    test_folds = {}
    # Counts/alignment only: no model predictions or test scores are used here.
    all_subjects = json.loads((SCRIPTS / ".cache_ibl_session_to_subject.json").read_text())
    for f in range(5):
        name = f"test_fold_{f}"
        ds = base.ds_for(CONFIG["cache_dir"], CONFIG["manip_dir"], name, CONFIG["max_len"])
        test_folds[name] = ds_signature(ds)
        test_rows += test_folds[name]["rows"]
        for it in ds.items:
            test_probes.add(str(it["pid"]))
            for m in it["meta"]:
                assert str(m) not in train_meta and str(m) not in val_meta
            eid = str(it["meta"][0]).split("_")[0]
            test_subjects.add(all_subjects[eid])
    assert not test_subjects & (set(split["train_subjects"]) | set(split["val_subjects"]))
    assert test_rows > 0 and test_probes, "No valid test rows/probes"
    write_json(ROOT / "subject_split.json", split)
    seed_all(1)
    model0 = MatchedDirectProbeSeq(tr.H, mean, scale, "lfp_only")
    seed_all(1)
    model1 = MatchedDirectProbeSeq(tr.H, mean, scale, "lfp_manip")
    assert state_hash(model0) == state_hash(model1)
    assert not any("pose_head" in key for key in model0.state_dict())
    # Nonzero XYZ weights make the no-manip leakage check nontrivial.
    with torch.no_grad():
        model0.resid_head.weight.normal_(0, 0.01)
        model1.load_state_dict(model0.state_dict())
    model0.eval(); model1.eval()
    sample = base.collate_manip([tr[0]])
    emb, manip, geom, depth, true, mask, _ = sample
    emb, manip, geom, depth, mask = [x[:, :32] for x in (emb, manip, geom, depth, mask)]
    with torch.no_grad():
        pred0 = model0(emb, manip, geom, depth, mask)
        perturbed = model0(emb, manip + 12345, geom, depth, mask)
        zero_fusion = model1(emb, torch.zeros_like(manip), geom, depth, mask)
        with_manip = model1(emb, manip, geom, depth, mask)
    for a, b, c in zip(pred0, perturbed, zero_fusion):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a, c, rtol=0, atol=0)
    assert not torch.equal(pred0[0], with_manip[0])
    mirrored = manip.clone()
    mirrored[..., 2] = 2 * base.ALLEN_MIDLINE_UM - mirrored[..., 2]
    with torch.no_grad():
        reflected = model1(emb, mirrored, geom, depth, mask)
    for a, b in zip(with_manip, reflected):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=0.01)
    torch.testing.assert_close(base.fold_lr_t(manip), base.fold_lr_t(mirrored), rtol=0, atol=0.001)
    assert torch.all(pred0[1] > 0)
    train_params = sum(p.numel() for p in model0.parameters())
    sources = {}
    for folder, suffix in [(CONFIG["cache_dir"], "_seq.npz"), (CONFIG["manip_dir"], "_manip_xyz.npy")]:
        for name in ["probe_train", "probe_val"] + [f"test_fold_{i}" for i in range(5)]:
            path = Path(folder) / (name + suffix)
            sources[str(path)] = {"sha256": digest(path), "bytes": path.stat().st_size}
    prepared = {
        "created_utc": now(), "status": "passed", "config": CONFIG, "seeds": SEEDS,
        "arms": ARMS, "embedding_dim": tr.H, "parameters_per_arm": train_params,
        "train": ds_signature(tr), "validation": ds_signature(va),
        "test_folds": test_folds, "test_rows": test_rows, "test_probes": len(test_probes),
        "test_subjects": sorted(test_subjects), "subject_split": split,
        "normalization_sha256": digest(ROOT / "normalization.npz"),
        "matched_initialization": True, "lfp_only_manip_perturbation_invariant": True,
        "zero_manip_fusion_equals_lfp_only": True, "fusion_uses_manipulator_input": True,
        "sources": sources,
    }
    write_json(ROOT / "preflight.json", prepared)
    write_json(ROOT / "protocol.json", {"created_utc": now(), "config": CONFIG, "seeds": SEEDS, "arms": ARMS})
    print(json.dumps({k: prepared[k] for k in ("status", "parameters_per_arm", "train",
                                             "validation", "test_rows", "test_probes", "subject_split")}, indent=2))


def train(arm, seed):
    preflight = json.loads((ROOT / "preflight.json").read_text())
    assert preflight["status"] == "passed" and preflight["config"] == CONFIG
    assert seed in SEEDS and arm in ARMS
    assert torch.cuda.is_available(), "GPU training required"
    device = torch.device("cuda")
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    output = ROOT / "runs" / arm / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        print(f"Already complete: {output}")
        return
    ckpts = output / "checkpoints"
    ckpts.mkdir(exist_ok=True)
    tr, va = load_trainval(output)
    assert ds_signature(tr) == preflight["train"]
    assert ds_signature(va) == preflight["validation"]
    norm = np.load(ROOT / "normalization.npz")
    seed_all(seed)
    model = MatchedDirectProbeSeq(tr.H, norm["mean"], norm["scale"], arm).to(device)
    args = {**CONFIG, "arm": arm, "seed": seed, "out_dir": str(output)}
    initial_sha = state_hash(model)
    write_json(output / "args.json", args)
    write_json(output / "initialization.json", {
        "sha256": initial_sha, "parameters": sum(p.numel() for p in model.parameters()),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "normalization_sha256": digest(ROOT / "normalization.npz"),
    })
    tl = DataLoader(tr, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=base.collate_manip)
    vl = DataLoader(va, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=base.collate_manip)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    first_epoch, best = 0, float("inf")
    latest = ckpts / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        assert state["args"] == args
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        first_epoch, best = state["epoch"] + 1, state["best"]
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        print(f"Resuming after epoch {first_epoch}", flush=True)
    metrics_path = output / "train_metrics.csv"
    if metrics_path.exists() and first_epoch:
        old_rows = list(csv.DictReader(metrics_path.open()))
        old_rows = [r for r in old_rows if int(r["epoch"]) <= first_epoch]
    else:
        old_rows = []
    fields = ["epoch", "train_nll", "train_mse", "val_rmse_fold", "best_val_rmse"]
    with metrics_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(old_rows); stream.flush()
        for epoch in range(first_epoch, CONFIG["epochs"]):
            model.train()
            losses, mses = [], []
            for emb, manip, geom, depth, true, mask, _ in tl:
                emb, manip, geom, depth, true, mask = [
                    x.to(device) for x in (emb, manip, geom, depth, true, mask)]
                opt.zero_grad(set_to_none=True)
                pred, sigma = model(emb, manip, geom, depth, mask)
                loss, mse, *_ = base.seq_loss(pred, base.fold_lr_t(true), depth, mask, 0, 0, sigma_um=sigma)
                assert torch.isfinite(loss), f"Non-finite loss at epoch {epoch + 1}"
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"], error_if_nonfinite=True)
                opt.step()
                losses.append(float(loss.detach())); mses.append(mse)
            value = base.evaluate(model, vl, device)
            assert math.isfinite(value)
            if value < best:
                best = value
                torch_save(ckpts / "best.pt", {
                    "model": model.state_dict(), "args": args, "epoch": epoch,
                    "val_rmse": value, "select_by": "val", "select_metric": value,
                    "initialization_sha256": initial_sha,
                })
            writer.writerow({"epoch": epoch + 1, "train_nll": np.mean(losses),
                             "train_mse": np.mean(mses), "val_rmse_fold": value, "best_val_rmse": best})
            stream.flush()
            torch_save(latest, {
                "model": model.state_dict(), "args": args, "optimizer": opt.state_dict(),
                "epoch": epoch, "best": best, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            })
            print(f"{arm} seed {seed} ep {epoch+1}/200 val={value:.2f} best={best:.2f}", flush=True)
    state = torch.load(ckpts / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    # Verify checkpoint selection by exact validation replay before opening test data.
    replay = base.evaluate(model, vl, device)
    assert abs(replay - state["val_rmse"]) < 0.01, (replay, state["val_rmse"])
    predictions = []
    for i in range(5):
        split = f"test_fold_{i}"
        ds = base.ds_for(CONFIG["cache_dir"], CONFIG["manip_dir"], split, CONFIG["max_len"])
        assert ds_signature(ds) == preflight["test_folds"][split]
        path = base.write_predictions(model, ds, device, str(output), split)
        predictions.append({"path": path, "sha256": digest(path)})
    write_json(output / "complete.json", {
        "completed_utc": now(), "arm": arm, "seed": seed, "epochs_completed": CONFIG["epochs"],
        "best_epoch": state["epoch"] + 1, "validation_rmse": state["val_rmse"],
        "validation_replay_rmse": replay, "test_inference_only_after_checkpoint_selection": True,
        "initialization_sha256": initial_sha, "predictions": predictions,
        "best_checkpoint_sha256": digest(ckpts / "best.pt"),
    })
    print(f"COMPLETE {arm} seed {seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train"])
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    opt = parser.parse_args()
    if opt.command == "prepare":
        prepare()
    else:
        train(opt.arm, opt.seed)
