"""
build_ibl_splits.py

Stratified train/val/test split for IBL BWM probes.

Strategy:
  1. At each depth, find included regions (>= MIN_PROBES_PER_REGION probes)
  2. For each included region, guarantee >= 1 probe in each split
  3. Assign remaining unassigned probes to train (70%) / val (15%) / test (15%)
  4. Write balanced_index_{train/val/test}.pkl

Output format matches build_atlas_dataset.py expectation:
  {'session_path_list': [list of pickle path strings]}
"""

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
PROBE_INDEX_PATH = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/probe_index.pkl'))
OUTPUT_DIR       = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/atlas_index_ibl'))
MIN_PROBES       = 6    # min probes per region to be included
SEED             = 42
DEPTHS           = list(range(1, 10))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

# ── Load probe index ──────────────────────────────────────────────────────────
print("Loading probe index...")
with open(PROBE_INDEX_PATH, 'rb') as f:
    probe_index = pickle.load(f)

all_pids = [p['pid'] for p in probe_index]
pid_to_probe = {p['pid']: p for p in probe_index}
n_probes = len(all_pids)
print(f"  Total probes: {n_probes}")

# ── Step 1: Find included regions at each depth ───────────────────────────────
print("\nFinding included regions (>= {} probes)...".format(MIN_PROBES))

# region_to_pids[depth][region] = list of pids covering that region
region_to_pids = {}
included_regions = {}

for d in DEPTHS:
    r2p = defaultdict(list)
    for p in probe_index:
        for r in p['regions_by_depth'].get(d, set()):
            r2p[r].append(p['pid'])
    region_to_pids[d] = dict(r2p)
    included_regions[d] = {r for r, pids in r2p.items() if len(pids) >= MIN_PROBES}
    n_dropped = len(r2p) - len(included_regions[d])
    print(f"  Depth {d}: {len(included_regions[d])} included, {n_dropped} dropped")

# ── Step 2: Stratified assignment ─────────────────────────────────────────────
print("\nRunning stratified split...")

# Track split assignment per pid
assignments = {}  # pid → 'train' | 'val' | 'test'

def assign(pid, split):
    """Assign a probe to a split if not already assigned."""
    if pid not in assignments:
        assignments[pid] = split
        return True
    return False

# For each depth and each included region, guarantee coverage in all 3 splits
splits = ['train', 'val', 'test']

for d in DEPTHS:
    for region in sorted(included_regions[d]):
        covering_pids = [p for p in region_to_pids[d][region]
                         if p in pid_to_probe]

        # Shuffle for randomness
        covering_pids = list(rng.permutation(covering_pids))

        # Check which splits already have coverage for this region
        covered_splits = set()
        for pid in covering_pids:
            if pid in assignments:
                covered_splits.add(assignments[pid])

        # Assign one probe per missing split
        for split in splits:
            if split not in covered_splits:
                for pid in covering_pids:
                    if pid not in assignments:
                        assign(pid, split)
                        covered_splits.add(split)
                        break

# ── Step 3: Assign remaining probes to train/val/test at 70/15/15 ─────────────
unassigned = [pid for pid in all_pids if pid not in assignments]
unassigned = list(rng.permutation(unassigned))

n_unassigned = len(unassigned)
n_val   = int(0.15 * n_probes) - sum(1 for v in assignments.values() if v == 'val')
n_test  = int(0.15 * n_probes) - sum(1 for v in assignments.values() if v == 'test')
n_val   = max(0, n_val)
n_test  = max(0, n_test)

print(f"\n  Assigned by stratification: {len(assignments)}")
print(f"  Unassigned remaining:       {n_unassigned}")
print(f"  Still need for val:         {n_val}")
print(f"  Still need for test:        {n_test}")

for i, pid in enumerate(unassigned):
    if i < n_val:
        assignments[pid] = 'val'
    elif i < n_val + n_test:
        assignments[pid] = 'test'
    else:
        assignments[pid] = 'train'

# ── Step 4: Final counts ──────────────────────────────────────────────────────
split_pids = {'train': [], 'val': [], 'test': []}
for pid, split in assignments.items():
    split_pids[split].append(pid)

print("\nFinal split sizes:")
for split in splits:
    print(f"  {split}: {len(split_pids[split])} probes "
          f"({100*len(split_pids[split])/n_probes:.1f}%)")

# ── Step 5: Verify coverage at all depths ────────────────────────────────────
print("\nCoverage verification:")
print(f"  {'Depth':>5}  {'Regions':>8}  {'Train':>8}  {'Val':>8}  {'Test':>8}  {'All covered':>12}")
all_good = True
for d in DEPTHS:
    split_regions = {s: set() for s in splits}
    for split in splits:
        for pid in split_pids[split]:
            for r in pid_to_probe[pid]['regions_by_depth'].get(d, set()):
                if r in included_regions[d]:
                    split_regions[split].add(r)

    n_included = len(included_regions[d])
    covered = all(len(split_regions[s]) == n_included for s in splits)
    if not covered:
        all_good = False
        for s in splits:
            missing = included_regions[d] - split_regions[s]
            if missing:
                print(f"  WARNING depth {d} {s} missing: {missing}")

    print(f"  {d:>5}  {n_included:>8}  "
          f"{len(split_regions['train']):>8}  "
          f"{len(split_regions['val']):>8}  "
          f"{len(split_regions['test']):>8}  "
          f"{'✓' if covered else '✗':>12}")

if all_good:
    print("\n  All depths fully covered in all splits ✓")
else:
    print("\n  WARNING: Some depths/splits not fully covered — check above")

# ── Step 5b: Repair pass — fix remaining coverage gaps ───────────────────────
print("\nRunning repair pass...")
max_iterations = 20
for iteration in range(max_iterations):
    # Find all gaps: (depth, region, split) that are missing
    gaps = []
    for d in DEPTHS:
        split_regions = {s: set() for s in splits}
        for split in splits:
            for pid in split_pids[split]:
                for r in pid_to_probe[pid]['regions_by_depth'].get(d, set()):
                    if r in included_regions[d]:
                        split_regions[split].add(r)
        for split in splits:
            for r in included_regions[d]:
                if r not in split_regions[split]:
                    gaps.append((d, r, split))

    if not gaps:
        print(f"  All gaps fixed after {iteration} iterations ✓")
        break

    print(f"  Iteration {iteration+1}: {len(gaps)} gaps remaining")

    for d, region, missing_split in gaps:
        # Find a probe that covers this region but is NOT in missing_split
        donor_pid  = None
        donor_split = None
        for split in splits:
            if split == missing_split:
                continue
            candidates = [pid for pid in split_pids[split]
                         if region in pid_to_probe[pid]['regions_by_depth'].get(d, set())]
            if len(candidates) > 1:  # only swap if split keeps >= 1 probe for this region
                donor_pid   = candidates[0]
                donor_split = split
                break

        if donor_pid is None:
            print(f"  WARNING: Cannot fix depth {d} region {region} in {missing_split} — not enough probes")
            continue

        # Find a probe in missing_split to swap with (one that doesn't break donor_split coverage)
        swap_pid = None
        for pid in split_pids[missing_split]:
            # Check that moving this pid to donor_split won't break missing_split coverage
            # (i.e., missing_split keeps coverage for all its currently covered regions)
            ok = True
            for d2 in DEPTHS:
                for r2 in pid_to_probe[pid]['regions_by_depth'].get(d2, set()):
                    if r2 not in included_regions[d2]:
                        continue
                    # Count how many probes cover r2 in missing_split
                    coverage_count = sum(
                        1 for p in split_pids[missing_split]
                        if r2 in pid_to_probe[p]['regions_by_depth'].get(d2, set())
                    )
                    if coverage_count <= 1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                swap_pid = pid
                break

        if swap_pid is None:
            # No safe swap — just move donor without swapping
            split_pids[donor_split].remove(donor_pid)
            split_pids[missing_split].append(donor_pid)
            assignments[donor_pid] = missing_split
        else:
            # Swap donor_pid ↔ swap_pid
            split_pids[donor_split].remove(donor_pid)
            split_pids[missing_split].remove(swap_pid)
            split_pids[donor_split].append(swap_pid)
            split_pids[missing_split].append(donor_pid)
            assignments[donor_pid] = missing_split
            assignments[swap_pid]  = donor_split
else:
    print(f"  WARNING: Repair did not fully converge after {max_iterations} iterations")

# ── Step 6: Save split files ──────────────────────────────────────────────────
print("\nSaving split files...")
for split in splits:
    pickle_paths = [pid_to_probe[pid]['pickle_path'] for pid in split_pids[split]]
    out = {'session_path_list': pickle_paths}
    out_path = OUTPUT_DIR / f'balanced_index_{split}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    print(f"  Saved: {out_path} ({len(pickle_paths)} probes)")

print("\nDone!")