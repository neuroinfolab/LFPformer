#!/usr/bin/env python3
"""Export Allen SDK LFP to 3-second compact shards with explicit channel metadata.

Input CSV columns: session_id,probe_id,channel_id,acronym,AP,DV,ML,label.
The CSV defines the split and atlas labels; this script never creates a random
window split. Atlas coordinates are AP,DV,ML in µm. SDK LFP is already 1250 Hz.
"""
import argparse, pickle, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--manifest", required=True)
p.add_argument("--channels-csv", required=True)
p.add_argument(
    "--atlas-metadata", required=True, help="Paper metadata.pkl with fixed label maps"
)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seconds", type=float, default=3.0)
p.add_argument("--start-seconds", type=float, default=0.0)
p.add_argument("--max-windows-per-channel", type=int, default=0)
p.add_argument("--shard-size", type=int, default=5000)
a = p.parse_args()
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

cache = EcephysProjectCache.from_warehouse(manifest=a.manifest)
channels = pd.read_csv(a.channels_csv)
required = {
    "session_id",
    "probe_id",
    "channel_id",
    "acronym",
    "AP",
    "DV",
    "ML",
    "label",
}
if not required.issubset(channels):
    raise ValueError(f"Missing columns: {required-set(channels)}")
if a.shard_size <= 0 or a.seconds <= 0:
    raise ValueError("Positive window and shard sizes required")
a.out.mkdir(parents=True, exist_ok=True)
shutil.copy2(a.atlas_metadata, a.out / "metadata.pkl")
if list(a.out.glob("samples_*.pkl")):
    raise FileExistsError("Output already contains shards")
files = []
counts = []
rows = []


def flush():
    if not rows:
        return
    name = f"samples_{len(files):06d}.pkl"
    with (a.out / name).open("wb") as f:
        pickle.dump(list(rows), f, protocol=4)
    files.append(name)
    counts.append(len(rows))
    rows.clear()


for session_id, session_channels in channels.groupby("session_id", sort=True):
    session = cache.get_session_data(int(session_id))
    for probe_id, table in session_channels.groupby("probe_id", sort=True):
        lfp = session.get_lfp(int(probe_id))
        fs = float(session.probes.loc[int(probe_id)].lfp_sampling_rate)
        if abs(fs - 1250) > 1e-3:
            raise ValueError(
                f"Expected 1250Hz, got {fs}; explicit resampling policy required"
            )
        width = int(round(a.seconds * fs))
        start = int(round(a.start_seconds * fs))
        for row in table.itertuples(index=False):
            x = np.asarray(
                lfp.sel(channel=int(row.channel_id)).values, dtype=np.float32
            )
            coords = np.array([row.AP, row.DV, row.ML], np.float32)
            if not np.isfinite(coords).all():
                raise ValueError("Nonfinite atlas coordinates")
            for chunk, i in enumerate(range(start, len(x) - width + 1, width)):
                if a.max_windows_per_channel and chunk >= a.max_windows_per_channel:
                    break
                signal = x[i : i + width]
                if not np.isfinite(signal).all():
                    continue
                meta = f"{session_id}_{chunk}_{probe_id}_{row.channel_id}_{row.acronym}"
                rows.append((signal, meta, int(row.label), coords))
                if len(rows) >= a.shard_size:
                    flush()
flush()
# CompactDataset uses cumulative_counts with a leading zero.
index = {
    "file_list": files,
    "file_counts": counts,
    "cumulative_counts": np.r_[0, np.cumsum(counts)],
    "n_files": len(files),
    "n_samples": sum(counts),
}
with (a.out / "compact_index.pkl").open("wb") as f:
    pickle.dump(index, f, protocol=4)
(a.out / "preprocessing.json").write_text(
    json.dumps(
        {
            "sample_rate": 1250,
            "seconds": a.seconds,
            "n_samples": sum(counts),
            "channel_metadata": str(Path(a.channels_csv).resolve()),
            "tail": "drop incomplete windows",
            "coordinate_order": "AP,DV,ML",
            "filtering": "Allen SDK LFP, no additional filtering",
        },
        indent=2,
    )
    + "\n"
)
