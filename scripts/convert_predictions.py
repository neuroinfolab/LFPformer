#!/usr/bin/env python3
"""Convert a trusted historical IBL prediction NPZ to the portable explicit schema."""
import argparse
from pathlib import Path
import numpy as np
from lfpformer.geometry import np1_geometry
from lfpformer.paths import activate_reference

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
activate_reference("benchmark")
from recompute_refined_3d_metrics import parse_pid_channel

with np.load(a.input, allow_pickle=True) as z:
    d = {k: z[k] for k in z.files}
parsed = [parse_pid_channel(s) for s in d["meta_str"]]
if any(pid is None or ch is None or ch < 0 for pid, ch in parsed):
    raise ValueError("Unparseable IBL metadata; no rows were silently dropped")
ids, ch = zip(*parsed)
mu = d.get("pred_xyz_gauss", d.get("pred_xyz"))
if mu is None:
    raise ValueError("Missing coordinate predictions")
out = dict(
    mu=mu,
    probe_ids=np.asarray(ids, dtype=str),
    channel_ids=np.asarray(ch, dtype=np.int64),
    geometry_xyz=np1_geometry(np.asarray(ch)),
)
if "pred_sigma_gauss" in d:
    out["sigma"] = d["pred_sigma_gauss"]
for key in ["true_xyz", "true_labels"]:
    if key in d:
        out[key] = d[key]
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.output, **out)
