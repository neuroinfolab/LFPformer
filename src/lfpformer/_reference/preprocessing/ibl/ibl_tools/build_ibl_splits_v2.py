"""
build_ibl_splits_v2.py

Three-pool subject-level split for IBL BWM SSL evaluation.

Design:
  Pool 1 — SSL Train (65 subjects): pretrain the SSL model (no labels used)
  Pool 2 — Probe (33 subjects):     train & validate the linear probe (model never saw these)
  Pool 3 — Test (38 subjects):      5×100K folds for final evaluation

Strategy:
  1. Start with the existing subject-level split (88 train / 16 val / 35 test)
  2. Move 7 test→val subjects (from greedy region coverage optimization)
  3. Move 20 train→val/test subjects (greedy, maximizing evaluable leaf regions)
  4. Final: 65 train / 33 val / 38 test subjects

  The "val" pool = probe train + probe val (model never saw these during SSL pretrain)
  The "test" pool = 5×100K test folds

Output: balanced_index_{train,val,test}.pkl in OUTPUT_DIR
  Each contains {'session_path_list': [...], 'metadata': {...}}
"""

import csv
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
PROBE_INDEX_PATH  = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/probe_index.pkl'))
PID_EID_CSV       = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/pid_eid_new.csv'))
OUTPUT_DIR        = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/atlas_index_ibl_v2'))
MIN_PROBES        = 6     # min probes per region to be "included"
MIN_SUBJECTS_TRAIN = 3    # min subjects covering each region that must remain in train
SEED              = 42
DEPTHS            = list(range(1, 10))

# Target split sizes (subjects)
# Starting from 88/16/35 → moving to ~65/33/38
TARGET_VAL = 33
TARGET_TEST = 38

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

# ── Step 1: Find included regions at each depth ─────────────────────────────
print(f"\nFinding included regions (>= {MIN_PROBES} probes)...")

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

subject_regions = {}
for d in DEPTHS:
    sr = defaultdict(set)
    for p in probe_index:
        subj = pid_to_subject[p['pid']]
        for r in p['regions_by_depth'].get(d, set()):
            if r in included_regions[d]:
                sr[subj].add(r)
    subject_regions[d] = dict(sr)

region_subjects = {}
for d in DEPTHS:
    rs = defaultdict(set)
    for subj, regions in subject_regions[d].items():
        for r in regions:
            rs[r].add(subj)
    region_subjects[d] = dict(rs)

# ── Step 3: Start from existing split, then optimize ─────────────────────────
# First, reproduce the original split (same as build_ibl_splits_subject.py with SEED=42)
print("\nReproducing original split...")

remaining_subjects_orig = {}
for d in DEPTHS:
    for r in included_regions[d]:
        remaining_subjects_orig[(d, r)] = set(region_subjects[d].get(r, set()))

train_subjects = set(subjects)
holdout_subjects = set()
target_holdout = int(n_subjects * 0.37)

shuffled_subjects = list(rng.permutation(subjects))
for subj in shuffled_subjects:
    if len(holdout_subjects) >= target_holdout:
        break
    can_holdout = True
    for d in DEPTHS:
        for r in subject_regions[d].get(subj, set()):
            key = (d, r)
            if key in remaining_subjects_orig:
                if len(remaining_subjects_orig[key]) - 1 < MIN_SUBJECTS_TRAIN:
                    can_holdout = False
                    break
        if not can_holdout:
            break
    if can_holdout:
        holdout_subjects.add(subj)
        train_subjects.discard(subj)
        for d in DEPTHS:
            for r in subject_regions[d].get(subj, set()):
                key = (d, r)
                if key in remaining_subjects_orig:
                    remaining_subjects_orig[key].discard(subj)

holdout_list = list(np.random.default_rng(SEED).permutation(list(holdout_subjects)))
n_val_orig = int(len(holdout_list) * 0.33)
val_subjects = set(holdout_list[:n_val_orig])
test_subjects = set(holdout_list[n_val_orig:])

print(f"  Original split: {len(train_subjects)} train / {len(val_subjects)} val / {len(test_subjects)} test")

# ── Step 4: Greedy rebalancing ────────────────────────────────────────────────
# Helper functions
def get_pool_regions(pool_subjects):
    """Get regions covered by a set of subjects at each depth."""
    pool = defaultdict(set)
    for subj in pool_subjects:
        for d in DEPTHS:
            pool[d].update(subject_regions[d].get(subj, set()))
    return dict(pool)

def compute_evaluable(train_s, val_s, test_s):
    """Compute total evaluable regions (union across depths of train∩val∩test)."""
    tr = get_pool_regions(train_s)
    vr = get_pool_regions(val_s)
    ter = get_pool_regions(test_s)
    all_evaluable = set()
    per_depth = {}
    for d in DEPTHS:
        shared = tr.get(d, set()) & vr.get(d, set()) & ter.get(d, set())
        all_evaluable.update(f"d{d}_{r}" for r in shared)  # unique key per depth+region
        per_depth[d] = len(shared)
    return len(all_evaluable), per_depth

total_ev, per_d = compute_evaluable(train_subjects, val_subjects, test_subjects)
print(f"  Original evaluable: {total_ev} (d5={per_d.get(5,0)}, d7={per_d.get(7,0)}, d9={per_d.get(9,0)})")

# Phase A: Move test subjects to val (7 moves from earlier optimization)
print("\nPhase A: Moving test → val subjects...")
for round_num in range(20):
    if len(val_subjects) >= TARGET_VAL and len(test_subjects) <= TARGET_TEST:
        break

    best_subj = None
    best_gain = -1

    for subj in list(test_subjects):
        trial_val = val_subjects | {subj}
        trial_test = test_subjects - {subj}
        new_total, _ = compute_evaluable(train_subjects, trial_val, trial_test)
        gain = new_total - total_ev
        if gain > best_gain:
            best_gain = gain
            best_subj = subj

    if best_gain <= 0:
        print(f"  No more improvements after {round_num} moves")
        break

    test_subjects.discard(best_subj)
    val_subjects.add(best_subj)
    total_ev, per_d = compute_evaluable(train_subjects, val_subjects, test_subjects)
    print(f"  Move {round_num+1}: {best_subj} → val (gain={best_gain}, total={total_ev})")

print(f"  After phase A: {len(train_subjects)}t / {len(val_subjects)}v / {len(test_subjects)}te, evaluable={total_ev}")

# Phase B: Move train subjects to val or test
print("\nPhase B: Moving train → val/test subjects...")
for round_num in range(30):
    best_subj = None
    best_target = None
    best_gain = 0

    for subj in list(train_subjects):
        # Check safety: removing from train won't drop any region below threshold
        can_remove = True
        for d in DEPTHS:
            for r in subject_regions[d].get(subj, set()):
                # Count how many train subjects cover this region
                train_coverage = sum(1 for s in train_subjects if r in subject_regions[d].get(s, set()))
                if train_coverage - 1 < MIN_SUBJECTS_TRAIN:
                    can_remove = False
                    break
            if not can_remove:
                break
        if not can_remove:
            continue

        for target in ['val', 'test']:
            # Respect target sizes
            if target == 'val' and len(val_subjects) >= TARGET_VAL:
                continue
            if target == 'test' and len(test_subjects) >= TARGET_TEST:
                continue

            trial_train = train_subjects - {subj}
            trial_val = val_subjects | ({subj} if target == 'val' else set())
            trial_test = test_subjects | ({subj} if target == 'test' else set())

            new_total, _ = compute_evaluable(trial_train, trial_val, trial_test)
            gain = new_total - total_ev

            if gain > best_gain:
                best_gain = gain
                best_subj = subj
                best_target = target

    if best_gain <= 0:
        print(f"  No more improvements after {round_num} moves")
        break

    train_subjects.discard(best_subj)
    if best_target == 'val':
        val_subjects.add(best_subj)
    else:
        test_subjects.add(best_subj)
    total_ev, per_d = compute_evaluable(train_subjects, val_subjects, test_subjects)
    print(f"  Move {round_num+1}: {best_subj} → {best_target} (gain={best_gain}, total={total_ev})")

print(f"\n  Final: {len(train_subjects)}t / {len(val_subjects)}v / {len(test_subjects)}te, evaluable={total_ev}")

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
print(f"  {'Depth':>5}  {'Total':>8}  {'Train':>8}  {'Val':>8}  {'Test':>8}  "
      f"{'Evaluable':>10}  {'Coverage':>10}")

splits = ['train', 'val', 'test']
evaluable_regions = {}
all_evaluable_names = set()

for d in DEPTHS:
    split_regions = {s: set() for s in splits}
    for split in splits:
        for pid in split_pids[split]:
            for r in pid_to_probe[pid]['regions_by_depth'].get(d, set()):
                if r in included_regions[d]:
                    split_regions[split].add(r)

    evaluable = split_regions['train'] & split_regions['val'] & split_regions['test']
    evaluable_regions[d] = evaluable
    all_evaluable_names.update(evaluable)

    n_total = len(included_regions[d])
    pct = f"{100*len(evaluable)/n_total:.0f}%" if n_total > 0 else "N/A"

    print(f"  {d:>5}  {n_total:>8}  "
          f"{len(split_regions['train']):>8}  "
          f"{len(split_regions['val']):>8}  "
          f"{len(split_regions['test']):>8}  "
          f"{len(evaluable):>10}  {pct:>10}")

# ── Step 7: Leaf region analysis ─────────────────────────────────────────────
print("\nLeaf region analysis:")
# A leaf region at depth d = region that appears at depth d but NOT as a parent at depth d+1
# For our data, a sample's max_depth tells us where the leaf is
# But from probe_index, we can approximate: a region at depth d is a leaf if
# no probe has a region at depth d+1 that is a child of it
# Simpler: count unique regions at each depth and subtract those that appear at d-1

# Actually, we can check from the probe_index which regions are leaves
leaf_regions_by_depth = defaultdict(set)
for p in probe_index:
    rbd = p['regions_by_depth']
    max_d = max(rbd.keys()) if rbd else 0
    for d in range(1, max_d + 1):
        if d in rbd:
            for r in rbd[d]:
                # This region is a leaf at depth d if d == max_d for this probe path
                # Actually need per-region-path check
                pass

# Simpler: just report the per-depth evaluable and the total unique
print(f"\n  Total unique evaluable region names (union across depths): {len(all_evaluable_names)}")

# ── Step 8: Sample count estimates ───────────────────────────────────────────
print("\nEstimated sample counts:")
for split in splits:
    n_ch = sum(pid_to_probe[pid]['n_channels'] for pid in split_pids[split])
    n_pids = len(split_pids[split])
    subjs = set(pid_to_subject[pid] for pid in split_pids[split])
    print(f"  {split}: {len(subjs)} subjects, {n_pids} probes, ~{n_ch} channels")

# ── Step 9: Save split files ─────────────────────────────────────────────────
print("\nSaving split files...")
for split in splits:
    pickle_paths = [pid_to_probe[pid]['pickle_path'] for pid in split_pids[split]]
    out = {
        'session_path_list': pickle_paths,
        'metadata': {
            'split_type': 'subject-level-v2',
            'split_strategy': '3-pool SSL evaluation',
            'pools': {
                'train': 'SSL pretraining (no labels)',
                'val': 'Linear probe train + val (model never saw these)',
                'test': '5x100K test folds (final evaluation)',
            },
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
print("Subject-Level Split v2 Summary (3-Pool SSL Design)")
print("=" * 60)
print(f"  Total subjects:    {n_subjects}")
print(f"  SSL Train:         {len(train_subjects)} subjects ({100*len(train_subjects)/n_subjects:.1f}%)")
print(f"  Probe (val):       {len(val_subjects)} subjects ({100*len(val_subjects)/n_subjects:.1f}%)")
print(f"  Test:              {len(test_subjects)} subjects ({100*len(test_subjects)/n_subjects:.1f}%)")
print(f"  Train probes:      {len(split_pids['train'])}")
print(f"  Probe probes:      {len(split_pids['val'])}")
print(f"  Test probes:       {len(split_pids['test'])}")
print(f"  Threshold:         >= {MIN_SUBJECTS_TRAIN} subjects/region in train")
for d in [1, 3, 5, 7, 9]:
    er = evaluable_regions.get(d, set())
    ir = included_regions.get(d, set())
    print(f"  D{d} evaluable:     {len(er)} / {len(ir)}")
print(f"  Output dir:        {OUTPUT_DIR}")
print("=" * 60)
print("\nDone!")
