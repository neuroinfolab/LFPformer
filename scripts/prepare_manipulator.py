#!/usr/bin/env python3
"""Build row-aligned manipulator XYZ from insertion metadata and channel IDs."""
import argparse, json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--rows", required=True, help="NPZ: probe_ids (EID_probeXX), channel_ids"
)
p.add_argument(
    "--trajectories", required=True, help="JSON list: eid,probe,x,y,z,depth,theta,phi"
)
p.add_argument("--out", required=True)
p.add_argument(
    "--valid-output", help="Optional path for a row-aligned boolean validity mask"
)
p.add_argument(
    "--tip-offset-um",
    type=float,
    default=0.0,
    help="0 reproduces historical paper cache; 200 requires rebuilding both train and test caches and retraining",
)
a = p.parse_args()
from iblatlas.atlas import Insertion, AllenAtlas

ba = AllenAtlas(25)
trajectories = {
    f"{r['eid']}_{r['probe']}": r for r in json.loads(Path(a.trajectories).read_text())
}
with np.load(a.rows, allow_pickle=False) as z:
    ids = z["probe_ids"].astype(str)
    channels = z["channel_ids"]
if (
    channels.ndim != 1
    or len(channels) != len(ids)
    or not np.issubdtype(channels.dtype, np.integer)
    or (channels < 0).any()
):
    raise ValueError("Aligned nonnegative integer channel IDs required")
xyz = np.full((len(ids), 3), np.nan, np.float32)
for pid in np.unique(ids):
    t = trajectories.get(pid)
    if t is None or t.get("depth", 0) <= 0:
        continue
    ix = np.flatnonzero(ids == pid)
    ins = Insertion.from_dict(
        {k: t[k] for k in ["x", "y", "z", "depth", "theta", "phi"]}, brain_atlas=ba
    )
    delta = ins.entry - ins.tip
    if np.linalg.norm(delta) == 0:
        continue
    up = delta / np.linalg.norm(delta)
    s = ((channels[ix] // 2) * 20 + a.tip_offset_um) * 1e-6
    xyz[ix] = ba.xyz2ccf(
        ins.tip[None, :] + s[:, None] * up[None, :], ccf_order="apdvml", mode="clip"
    )
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.save(a.out, xyz)
if a.valid_output:
    Path(a.valid_output).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.valid_output, np.isfinite(xyz).all(1))
Path(a.out + ".json").write_text(
    json.dumps(
        {
            "rows": len(ids),
            "valid": int(np.isfinite(xyz).all(1).sum()),
            "tip_offset_um": a.tip_offset_um,
            "units": "um",
            "columns": ["AP", "DV", "ML"],
        },
        indent=2,
    )
    + "\n"
)
