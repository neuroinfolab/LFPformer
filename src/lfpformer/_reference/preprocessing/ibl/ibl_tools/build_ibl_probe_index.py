"""
build_ibl_probe_index.py

Builds a per-probe index recording which brain regions each probe
covers at each depth (1-9), using mlapdv from probe_channels.npz
directly for atlas lookups via iblatlas.

Output: probe_index.pkl
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from iblatlas.atlas import AllenAtlas

# ── Config ────────────────────────────────────────────────────────────────────
NPZ_PATH    = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/probe_channels.npz'))
PID_EID_CSV = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/pid_eid_new.csv'))
FLAT_DIR    = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/preprocessed_flat'))
OUTPUT_PATH = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/probe_index.pkl'))

# ── Step 1: Load atlas ────────────────────────────────────────────────────────
print("Loading AllenAtlas...")
ba = AllenAtlas(res_um=25)

# ── Step 2: Build hierarchy lookup for all unique structure IDs ───────────────
print("Building hierarchy lookup...")

def get_hierarchy(sid):
    chain = []
    current_id = int(sid)
    while current_id is not None and current_id != 0:
        idx = np.where(ba.regions.id == current_id)[0]
        if len(idx) == 0:
            break
        idx = idx[0]
        acr = ba.regions.acronym[idx]
        chain.append((current_id, acr))
        parent = ba.regions.parent[idx]
        if parent is None or (isinstance(parent, float) and np.isnan(parent)):
            break
        current_id = int(parent) if parent != 0 else None
    chain = list(reversed(chain))
    if not chain:
        return None
    hier = {d: chain[d][1] for d in range(len(chain))}
    hier['fine_id']   = sid
    hier['fine_acr']  = chain[-1][1]
    hier['max_depth'] = len(chain) - 1
    return hier

# Pre-compute hierarchy for ALL structures in ba.regions (not just annotation volume)
unique_ids = [int(sid) for sid in ba.regions.id if sid != 0]
anno_id_to_hierarchy = {}
for uid in unique_ids:
    h = get_hierarchy(uid)
    if h is not None:
        anno_id_to_hierarchy[uid] = h
print(f"  Mapped {len(anno_id_to_hierarchy)} structure IDs")

# ── Step 3: Load probe data ───────────────────────────────────────────────────
print("\nLoading probe_channels.npz...")
npz = np.load(NPZ_PATH, allow_pickle=True)
probe_data = {}
for key in npz.files:
    pid, field = key.split('__', 1)
    if pid not in probe_data:
        probe_data[pid] = {}
    probe_data[pid][field] = npz[key]
print(f"  Loaded {len(probe_data)} probes")

# Load pid → eid/probe_name mapping
pid_eid_df  = pd.read_csv(PID_EID_CSV)
pid_to_info = {row['pid']: row for _, row in pid_eid_df.iterrows()}

# Index available pickles by stem (eid_probename)
pickle_paths = {p.stem: p for p in sorted(FLAT_DIR.glob('*.pickle'))}

# ── Step 4: Build per-probe index ─────────────────────────────────────────────
print("Building per-probe region coverage...")

probe_index     = []
missing_pickles = []

for pid, d in probe_data.items():
    if pid not in pid_to_info:
        print(f"  WARNING: {pid[:8]} not in pid_eid_new.csv, skipping")
        missing_pickles.append(pid)
        continue

    info       = pid_to_info[pid]
    eid        = info['eid']
    probe_name = info['probe_name']
    stem       = f"{eid}_{probe_name}"

    if stem not in pickle_paths:
        print(f"  WARNING: no pickle for {pid[:8]} ({stem}), skipping")
        missing_pickles.append(pid)
        continue

    mlapdv   = d['mlapdv'].astype('float64') / 1e6  # µm → meters
    n_ch     = mlapdv.shape[0]

    # Look up structure ID for each channel
    sids = ba.get_labels(mlapdv, mode='clip')

    # Build regions_by_depth
    regions_by_depth = defaultdict(set)
    n_valid = 0

    for ch_idx, sid in enumerate(sids):
        sid = int(sid)
        if sid == 0 or sid not in anno_id_to_hierarchy:
            continue
        hier = anno_id_to_hierarchy[sid]
        n_valid += 1
        for depth in range(1, 10):
            if depth in hier:
                regions_by_depth[depth].add(hier[depth])

    probe_index.append({
        'pid':              pid,
        'eid':              eid,
        'probe_name':       probe_name,
        'pickle_path':      str(pickle_paths[stem]),
        'regions_by_depth': dict(regions_by_depth),
        'n_channels':       n_ch,
        'n_valid':          n_valid,
    })

print(f"  Probes indexed:  {len(probe_index)}")
print(f"  Probes missing:  {len(missing_pickles)}")

# ── Step 5: Summary ───────────────────────────────────────────────────────────
print("\nRegion coverage summary:")
print(f"  {'Depth':>5}  {'Total Regions':>14}  {'Avg per probe':>14}")
for d in range(1, 10):
    all_regions = set()
    per_probe   = []
    for p in probe_index:
        r = p['regions_by_depth'].get(d, set())
        all_regions |= r
        per_probe.append(len(r))
    print(f"  {d:>5}  {len(all_regions):>14}  {np.mean(per_probe):>14.1f}")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(probe_index, f, protocol=4)
print(f"\nSaved → {OUTPUT_PATH} ({len(probe_index)} probes)")