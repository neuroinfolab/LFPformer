#!/usr/bin/env python3
"""Precompute per-channel micro-manipulator CCF xyz aligned to CompactDataset order.

Used by ManipDistc SSL (global soft-KL contrastive with manip xyz instead of
histology CCF). Mirrors precompute_manip_xyz.py but targets any compact atlas
dir (train/ + val/).

TIP_SIZE_UM=200 is APPLIED here (correct NP1.0 physics). Do NOT reuse these
arrays as inputs to older manip-FT models trained on the uncorrected
eval_v2_100k cache.

Run inside w2v2_cuda_cu128_ibl.sif (needs iblatlas). CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

TIP_SIZE_UM = 200.0  # NP1.0 tip below first electrode; IBL depth = entry→physical tip


def _load_manip_cache(paths):
    manip = {}
    for p in paths:
        if not os.path.isfile(p):
            print(f"[WARN] missing manip cache: {p}")
            continue
        for r in json.load(open(p)):
            manip[f"{r['eid']}_{r['probe']}"] = r
    print(f"manip cache: {len(manip)} probes from {len(paths)} files")
    return manip


def _key_from_meta(meta: str) -> str:
    # Compact meta: "{eid}_{session}_{probe}-ch{N}" (probe may be probe00 / probe00a)
    parts = meta.split("_")
    return f"{parts[0]}_{parts[2]}"


def manip_line(traj, ch, ba, Insertion):
    """Per-channel CCF xyz (ap,dv,lr µm) from surgical log + NP1.0 geometry."""
    ins = Insertion.from_dict(
        dict(
            x=traj["x"], y=traj["y"], z=traj["z"], depth=traj["depth"],
            theta=traj["theta"], phi=traj["phi"],
        ),
        brain_atlas=ba,
    )
    e, t = ins.entry, ins.tip
    span = np.linalg.norm(e - t)
    if not np.isfinite(span) or span < 1e-9:
        raise ValueError(f"degenerate insertion span={span}")
    up = (e - t) / span
    # channel 0 sits TIP_SIZE_UM above physical tip
    ch = np.asarray(ch, dtype=np.int64)
    s = ((ch // 2) * 20.0 + TIP_SIZE_UM) * 1e-6
    pts = t[None, :] + up[None, :] * s[:, None]
    out = ba.xyz2ccf(pts, ccf_order="apdvml", mode="clip")
    if not np.isfinite(out).all():
        raise ValueError("non-finite CCF xyz from manip_line")
    return out


def iter_split_samples(split_dir):
    idx = pickle.load(open(os.path.join(split_dir, "compact_index.pkl"), "rb"))
    for fid in range(idx["n_files"]):
        fpath = os.path.join(split_dir, f"samples_{fid:06d}.pkl")
        for s in pickle.load(open(fpath, "rb")):
            m = s[1]
            key = _key_from_meta(m)
            ch = int(m.split("_")[3].split("-ch")[1])
            yield key, ch


def process_split(split_dir, manip, ba, Insertion, out_prefix):
    keys, chs = [], []
    for key, ch in iter_split_samples(split_dir):
        keys.append(key)
        chs.append(ch)
    keys = np.asarray(keys)
    chs = np.asarray(chs, dtype=int)
    n = len(keys)
    xyz = np.full((n, 3), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    uniq = {}
    for i, (k, ch) in enumerate(zip(keys, chs)):
        if k not in manip:
            continue
        uniq.setdefault(k, []).append((i, ch))

    n_ok_probes = 0
    for k, items in uniq.items():
        traj = manip[k]
        ch_arr = np.asarray([ch for _, ch in items], dtype=int)
        try:
            pts = manip_line(traj, ch_arr, ba, Insertion)
        except Exception as e:
            print(f"[WARN] manip_line failed for {k}: {e}")
            continue
        n_ok_probes += 1
        for (i, _), pt in zip(items, pts):
            xyz[i] = pt
            valid[i] = True

    # Drop residual non-finite rows (degenerate insertions, atlas edge cases).
    finite = np.isfinite(xyz).all(axis=1)
    valid &= finite
    np.save(f"{out_prefix}_manip_xyz.npy", xyz.astype(np.float32))
    np.save(f"{out_prefix}_valid.npy", valid)
    cov = float(valid.mean()) if n else 0.0
    print(
        f"  wrote {out_prefix}_*  n={n} valid={valid.sum()} ({100*cov:.1f}%) "
        f"probes_ok={n_ok_probes}/{len(np.unique(keys))}"
    )
    if valid.any():
        v = xyz[valid]
        print(
            f"  manip xyz range ap[{v[:,0].min():.0f},{v[:,0].max():.0f}] "
            f"dv[{v[:,1].min():.0f},{v[:,1].max():.0f}] "
            f"lr[{v[:,2].min():.0f},{v[:,2].max():.0f}]"
        )
    return int(valid.sum()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--compact_root",
        required=True,
        help="e.g. .../compact_atlas_ibl_v2_75k (must contain train/ and val/)",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="output dir for {train,val}_manip_xyz.npy + _valid.npy",
    )
    ap.add_argument(
        "--manip_json",
        nargs="+",
        default=[
            __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/one_cache/fullrec_manip_trajectories.json'),
            __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/one_cache/train_manip_trajectories.json'),
            __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/one_cache/eval_manip_trajectories.json'),
        ],
    )
    args = ap.parse_args()

    from iblatlas.atlas import AllenAtlas, Insertion

    os.makedirs(args.out_dir, exist_ok=True)
    manip = _load_manip_cache(args.manip_json)
    ba = AllenAtlas(25)
    print(f"TIP_SIZE_UM={TIP_SIZE_UM} (applied)")

    for split in ("train", "val"):
        sd = os.path.join(args.compact_root, split)
        if not os.path.isdir(sd):
            raise FileNotFoundError(sd)
        print(f"===== {split} =====")
        process_split(
            sd, manip, ba, Insertion,
            out_prefix=os.path.join(args.out_dir, split),
        )
    print("DONE", args.out_dir)


if __name__ == "__main__":
    main()
