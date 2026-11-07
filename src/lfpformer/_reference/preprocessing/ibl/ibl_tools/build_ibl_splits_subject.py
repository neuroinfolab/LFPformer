"""
build_ibl_splits_subject.py

Subject-level train/val/test split for IBL BWM probes.

Strategy:
  1. Map probes → subjects via pid_eid_new.csv
  2. For each depth, find which regions each subject covers
  3. Greedily select holdout subjects for test (and val) while ensuring
     every included region retains >= MIN_SUBJECTS_PER_REGION subjects in train
  4. Assign ALL probes of a held-out subject to test (or val)
  5. Verify coverage, run repair pass if needed

Threshold analysis (from prior investigation):
  >= 5 per region: 28 test subjects (20%), 17/52 d9 regions evaluable
  >= 3 per region: 51 test subjects (37%), 32/52 d9 regions evaluable  ← chosen
  >= 1 per region: 89 test subjects (64%), 47/52 d9 regions evaluable

Output format matches build_atlas_dataset.py expectation:
  {'session_path_list': [list of pickle path strings]}
"""

import csv
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
PROBE_INDEX_PATH  = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/probe_index.pkl'))
PID_EID_CSV       = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/pid_eid_new.csv'))
OUTPUT_DIR        = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/atlas_index_ibl_subject_split'))
MIN_PROBES        = 6     # min probes per region to be "included" (same as original)
MIN_SUBJECTS_TRAIN = 3    # min subjects covering each region that must remain in train
HOLDOUT_FRACTION  = 0.37  # target fraction of subjects to hold out (val + test)
VAL_FRACTION      = 0.33  # fraction of held-out subjects for val (rest → test)
SEED              = 42
DEPTHS            = list(range(1, 10))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

# ── Load probe index ──────────────────────────────────────────────────────────
print("Loading probe index...")
with open(PROBE_INDEX_PATH, 'rb') as f:
    probe_index = pickle.load(f)

pid_to_probe = {p['pid']: p for p in probe_index}
n_probes = len(probe_index)
print(f"  Total probes: {n_probes}")

# ── Load pid → subject mapping ────────────────────────────────────────────────
print("Loading subject mapping...")
pid_to_subject = {}
with open(PID_EID_CSV) as f:
    for row in csv.DictReader(f):
        pid_to_subject[row['pid']] = row['subject']

subjects = sorted(set(pid_to_subject.values()))
n_subjects = len(subjects)
print(f"  Total subjects: {n_subjects}")

# Build subject → list of pids
subject_to_pids = defaultdict(list)
for p in probe_index:
    subj = pid_to_subject[p['pid']]
    subject_to_pids[subj].append(p['pid'])

print(f"  Probes per subject: min={min(len(v) for v in subject_to_pids.values())}, "
      f"max={max(len(v) for v in subject_to_pids.values())}, "
      f"mean={np.mean([len(v) for v in subject_to_pids.values()]):.1f}")

# ── Step 1: Find included regions at each depth ───────────────────────────────
print("\nFinding included regions (>= {} probes)...".format(MIN_PROBES))

region_to_pids = {}
included_regions = {}

for d in DEPTHS:
    r2p = defaultdict(list)
    for p in probe_index:
        for r in p['regions_by_depth'].get(d, set()):
            r2p[r].append(p['pid'])
    region_to_pids[d] = dict(r2p)
    included_regions[d] = {r for r, pids in r2p.items() if len(pids) >= MIN_PROBES}
    print(f"  Depth {d}: {len(included_regions[d])} included regions")

# ── Step 2: Build subject → regions coverage ──────────────────────────────────
print("\nBuilding subject-region coverage...")

# subject_regions[depth][subject] = set of regions that subject's probes cover
subject_regions = {}
for d in DEPTHS:
    sr = defaultdict(set)
    for p in probe_index:
        subj = pid_to_subject[p['pid']]
        for r in p['regions_by_depth'].get(d, set()):
            if r in included_regions[d]:
                sr[subj].add(r)
    subject_regions[d] = dict(sr)

# region_subjects[depth][region] = set of subjects covering that region
region_subjects = {}
for d in DEPTHS:
    rs = defaultdict(set)
    for subj, regions in subject_regions[d].items():
        for r in regions:
            rs[r].add(subj)
    region_subjects[d] = dict(rs)

# Print region coverage stats at key depths
for d in [1, 5, 9]:
    subj_counts = [len(ss) for ss in region_subjects[d].values()]
    if subj_counts:
        print(f"  Depth {d}: {len(included_regions[d])} regions, "
              f"subjects/region: min={min(subj_counts)}, "
              f"median={int(np.median(subj_counts))}, "
              f"max={max(subj_counts)}")

# ── Step 3: Greedy holdout selection ──────────────────────────────────────────
target_holdout = int(n_subjects * HOLDOUT_FRACTION)
print(f"\nSelecting holdout subjects (target={target_holdout}, "
      f"min {MIN_SUBJECTS_TRAIN} subjects/region in train)...")

# Track how many subjects remain in train for each (depth, region)
remaining_subjects = {}
for d in DEPTHS:
    for r in included_regions[d]:
        remaining_subjects[(d, r)] = set(region_subjects[d].get(r, set()))

train_subjects = set(subjects)
holdout_subjects = set()

# Shuffle subjects for randomness
shuffled_subjects = list(rng.permutation(subjects))

for subj in shuffled_subjects:
    # Stop once we've reached the target holdout count
    if len(holdout_subjects) >= target_holdout:
        break

    # Check: can we hold out this subject without dropping any region below threshold?
    can_holdout = True
    for d in DEPTHS:
        for r in subject_regions[d].get(subj, set()):
            key = (d, r)
            if key in remaining_subjects:
                # After removing this subject, how many remain?
                if len(remaining_subjects[key]) - 1 < MIN_SUBJECTS_TRAIN:
                    can_holdout = False
                    break
        if not can_holdout:
            break

    if can_holdout:
        holdout_subjects.add(subj)
        train_subjects.discard(subj)
        # Update remaining counts
        for d in DEPTHS:
            for r in subject_regions[d].get(subj, set()):
                key = (d, r)
                if key in remaining_subjects:
                    remaining_subjects[key].discard(subj)

print(f"  Held out: {len(holdout_subjects)} subjects ({100*len(holdout_subjects)/n_subjects:.1f}%)")
print(f"  Train:    {len(train_subjects)} subjects ({100*len(train_subjects)/n_subjects:.1f}%)")

# ── Step 4: Split holdout into val and test ───────────────────────────────────
holdout_list = list(rng.permutation(list(holdout_subjects)))
n_val = int(len(holdout_list) * VAL_FRACTION)
val_subjects = set(holdout_list[:n_val])
test_subjects = set(holdout_list[n_val:])

print(f"\n  Val subjects:  {len(val_subjects)}")
print(f"  Test subjects: {len(test_subjects)}")

# ── Step 5: Assign probes to splits ──────────────────────────────────────────
split_pids = {'train': [], 'val': [], 'test': []}
for p in probe_index:
    subj = pid_to_subject[p['pid']]
    if subj in val_subjects:
        split_pids['val'].append(p['pid'])
    elif subj in test_subjects:
        split_pids['test'].append(p['pid'])
    else:
        split_pids['train'].append(p['pid'])

print("\nFinal split sizes (probes):")
splits = ['train', 'val', 'test']
for split in splits:
    print(f"  {split}: {len(split_pids[split])} probes "
          f"({100*len(split_pids[split])/n_probes:.1f}%)")

# Count subjects per split
for split in splits:
    subjs = set(pid_to_subject[pid] for pid in split_pids[split])
    print(f"  {split} subjects: {len(subjs)}")

# Verify no subject overlap
train_subjs = set(pid_to_subject[pid] for pid in split_pids['train'])
val_subjs   = set(pid_to_subject[pid] for pid in split_pids['val'])
test_subjs  = set(pid_to_subject[pid] for pid in split_pids['test'])
assert len(train_subjs & val_subjs) == 0, "Train/Val subject overlap!"
assert len(train_subjs & test_subjs) == 0, "Train/Test subject overlap!"
assert len(val_subjs & test_subjs) == 0, "Val/Test subject overlap!"
print("\n  No subject overlap between splits ✓")

# ── Step 6: Coverage verification ────────────────────────────────────────────
print("\nCoverage verification:")
print(f"  {'Depth':>5}  {'Included':>8}  {'Train':>8}  {'Val':>8}  {'Test':>8}  "
      f"{'Evaluable':>10}  {'Notes':>20}")

evaluable_regions = {}
for d in DEPTHS:
    split_regions = {s: set() for s in splits}
    for split in splits:
        for pid in split_pids[split]:
            for r in pid_to_probe[pid]['regions_by_depth'].get(d, set()):
                if r in included_regions[d]:
                    split_regions[split].add(r)

    n_included = len(included_regions[d])
    # "Evaluable" = regions present in ALL 3 splits
    evaluable = split_regions['train'] & split_regions['val'] & split_regions['test']
    evaluable_regions[d] = evaluable

    n_train_only = len(split_regions['train'] - evaluable)
    notes = ""
    if len(evaluable) < n_included:
        notes = f"{n_included - len(evaluable)} not in all splits"

    print(f"  {d:>5}  {n_included:>8}  "
          f"{len(split_regions['train']):>8}  "
          f"{len(split_regions['val']):>8}  "
          f"{len(split_regions['test']):>8}  "
          f"{len(evaluable):>10}  {notes:>20}")

# ── Step 7: Print sample counts per split ────────────────────────────────────
print("\nEstimated sample counts (from compact data):")
try:
    for split in splits:
        idx_path = Path(f'/workspace/ws2/Alphabrain_staging/data/ibl/compact_atlas_ibl_25k/{split}/compact_index.pkl')
        if idx_path.exists():
            with open(idx_path, 'rb') as f:
                idx = pickle.load(f)
            print(f"  {split}: {idx.get('n_samples', '?')} samples (existing compact data)")
except Exception:
    pass

# Count probes and channels
for split in splits:
    n_ch = sum(pid_to_probe[pid]['n_channels'] for pid in split_pids[split])
    print(f"  {split}: {len(split_pids[split])} probes, ~{n_ch} channels")

# ── Step 8: Save split files ─────────────────────────────────────────────────
print("\nSaving split files...")
for split in splits:
    pickle_paths = [pid_to_probe[pid]['pickle_path'] for pid in split_pids[split]]
    out = {
        'session_path_list': pickle_paths,
        'metadata': {
            'split_type': 'subject-level',
            'split': split,
            'n_probes': len(pickle_paths),
            'n_subjects': len(set(pid_to_subject[pid] for pid in split_pids[split])),
            'min_subjects_per_region_train': MIN_SUBJECTS_TRAIN,
            'seed': SEED,
            'subjects': sorted(set(pid_to_subject[pid] for pid in split_pids[split])),
        }
    }
    out_path = OUTPUT_DIR / f'balanced_index_{split}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    print(f"  Saved: {out_path} ({len(pickle_paths)} probes)")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Subject-Level Split Summary")
print("=" * 60)
print(f"  Total subjects:    {n_subjects}")
print(f"  Train subjects:    {len(train_subjects)} ({100*len(train_subjects)/n_subjects:.1f}%)")
print(f"  Val subjects:      {len(val_subjects)} ({100*len(val_subjects)/n_subjects:.1f}%)")
print(f"  Test subjects:     {len(test_subjects)} ({100*len(test_subjects)/n_subjects:.1f}%)")
print(f"  Train probes:      {len(split_pids['train'])}")
print(f"  Val probes:        {len(split_pids['val'])}")
print(f"  Test probes:       {len(split_pids['test'])}")
print(f"  Threshold:         >= {MIN_SUBJECTS_TRAIN} subjects/region in train")
print(f"  D9 evaluable:      {len(evaluable_regions.get(9, set()))} / {len(included_regions.get(9, set()))}")
print(f"  D5 evaluable:      {len(evaluable_regions.get(5, set()))} / {len(included_regions.get(5, set()))}")
print(f"  Output dir:        {OUTPUT_DIR}")
print("=" * 60)
print("\nDone!")
