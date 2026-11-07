#!/usr/bin/env python3
"""Exercise prediction post-processing and scoring; not a paper-result benchmark."""
import argparse, json
from pathlib import Path
import numpy as np
from lfpformer.geometry import np1_geometry
from lfpformer.cli import main

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", type=Path, default=Path("outputs/synthetic"))
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
channels = np.tile(np.repeat(np.arange(64), 5), 2)
ids = np.repeat(["probe_a", "probe_b"], 320)
geometry = np1_geometry(channels)
rotation = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
truth = geometry @ rotation.T + np.array([4000, 5000, 3000])
truth[ids == "probe_b", 0] += 500
sigma = rng.uniform(30, 150, truth.shape)
mu = truth + rng.normal(size=truth.shape) * sigma
source = a.out / "window_predictions.npz"
np.savez_compressed(
    source,
    mu=mu,
    sigma=sigma,
    true_xyz=truth,
    geometry_xyz=geometry,
    probe_ids=ids,
    channel_ids=channels,
)
results = {}
for method in ["raw", "pool", "ibl-kabsch", "allen-line"]:
    output = a.out / f"{method}.npz"
    metrics = a.out / f"{method}.json"
    main(
        [
            "postprocess",
            "--input",
            str(source),
            "--method",
            method,
            "--output",
            str(output),
        ]
    )
    main(["score", "--input", str(output), "--output", str(metrics)])
    results[method] = json.loads(metrics.read_text())
assert results["ibl-kabsch"]["n_channels"] == 128
assert (
    results["ibl-kabsch"]["channel_aggregate_rmse_um"]
    < results["raw"]["channel_aggregate_rmse_um"]
)
(a.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
print(f"Synthetic pipeline complete: {a.out.resolve()}")
