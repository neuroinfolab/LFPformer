#!/usr/bin/env python3
"""Apply the paper's training-only ZCA shrinkage whitening to flat caches."""
import argparse, sys
from pathlib import Path
import numpy as np
from lfpformer.paths import activate_reference

base = activate_reference()
sys.path.insert(0, str(base / "scripts/_frozen"))
from train_probe_sequence_ext_whiten import emb_whiten_stats

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--cache-dir", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
if a.cache_dir.resolve() == a.out.resolve():
    raise ValueError("Use a separate output directory")
mean, W, info = emb_whiten_stats(
    a.cache_dir / "probe_train_seq.npz", mode="zca", shrink=1.0
)
np.savez(a.out / "whitening.npz", mean=mean, W=W)
for path in sorted(a.cache_dir.glob("*_seq.npz")):
    with np.load(path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    d["emb"] = ((d["emb"].astype(np.float32) - mean) @ W.T).astype(np.float16)
    np.savez_compressed(a.out / path.name, **d)
import json

(a.out / "whitening.json").write_text(json.dumps(info, indent=2) + "\n")
