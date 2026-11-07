"""Build eval-specific datasets for the 3-pool SSL evaluation design.

Creates:
  - probe_train/  : 100K class-balanced samples from val subjects (29 subjects)
  - probe_val/    : 25K class-balanced samples from val subjects (no overlap with probe_train)
  - test_fold_0/  : 100K class-balanced samples from test subjects (34 subjects)
  - test_fold_1/  : 100K (non-overlapping with fold_0)
  - test_fold_2/  : 100K (non-overlapping with fold_0,1)
  - test_fold_3/  : 100K (non-overlapping with fold_0,1,2)
  - test_fold_4/  : 100K (non-overlapping with fold_0,1,2,3)

All subsampled from the full compact dataset (subject_split_v2/compact_atlas_ibl_v2_full).
Uses class-balanced sampling (by fine_id) to preserve region coverage.

Usage:
    python build_eval_dataset.py \
        --full_data_dir data/ibl/subject_split_v2/compact_atlas_ibl_v2_full \
        --output_dir data/ibl/subject_split_v2/eval_v2 \
        --probe_train_size 100000 \
        --probe_val_size 25000 \
        --test_fold_size 100000 \
        --n_test_folds 5
"""

import argparse
import os
import pickle
import time
from collections import Counter, defaultdict

import numpy as np


def load_all_samples(split_dir, max_files=None):
    """Load all samples from a compact dataset split directory."""
    index_path = os.path.join(split_dir, "compact_index.pkl")
    idx = pickle.load(open(index_path, "rb"))
    n_files = idx["n_files"]
    if max_files:
        n_files = min(n_files, max_files)

    all_samples = []
    for i in range(n_files):
        fname = os.path.join(split_dir, f"samples_{i:06d}.pkl")
        if not os.path.exists(fname):
            continue
        samples = pickle.load(open(fname, "rb"))
        all_samples.extend(samples)
        if (i + 1) % 200 == 0:
            print(f"  ... loaded {i+1}/{n_files} files, {len(all_samples):,} samples")

    print(f"  Loaded {len(all_samples):,} samples from {n_files} files")
    return all_samples


def class_balanced_subsample(samples, target_size, rng, exclude_indices=None):
    """Class-balanced subsample by fine_id.

    Args:
        samples: list of (signal, meta, hierarchy, coords) tuples
        target_size: desired number of samples
        rng: numpy RandomState
        exclude_indices: set of indices to exclude (for non-overlapping folds)

    Returns:
        selected_indices: list of indices into samples
    """
    # Group indices by fine_id
    class_to_indices = defaultdict(list)
    for i, s in enumerate(samples):
        if exclude_indices and i in exclude_indices:
            continue
        fine_id = s[2]["fine_id"]
        class_to_indices[fine_id].append(i)

    n_classes = len(class_to_indices)
    if n_classes == 0:
        return []

    # Compute per-class quota
    base_per_class = target_size // n_classes
    remainder = target_size - base_per_class * n_classes

    selected = []
    # Shuffle classes for fair remainder distribution
    class_ids = list(class_to_indices.keys())
    rng.shuffle(class_ids)

    for ci, cid in enumerate(class_ids):
        indices = class_to_indices[cid]
        rng.shuffle(indices)
        quota = base_per_class + (1 if ci < remainder else 0)
        # Take min(quota, available)
        n_take = min(quota, len(indices))
        selected.extend(indices[:n_take])

    # If we're short (some classes had fewer samples), fill from remaining
    if len(selected) < target_size:
        all_remaining = []
        selected_set = set(selected)
        for cid in class_ids:
            for idx in class_to_indices[cid]:
                if idx not in selected_set:
                    all_remaining.append(idx)
        rng.shuffle(all_remaining)
        need = target_size - len(selected)
        selected.extend(all_remaining[:need])

    return selected[:target_size]


def write_compact(samples, indices, output_dir, chunk_size=5000):
    """Write selected samples to a compact dataset directory."""
    os.makedirs(output_dir, exist_ok=True)

    selected_samples = [samples[i] for i in indices]
    n_files = 0
    cumulative = [0]

    for start in range(0, len(selected_samples), chunk_size):
        chunk = selected_samples[start:start + chunk_size]
        fname = os.path.join(output_dir, f"samples_{n_files:06d}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(chunk, f, protocol=4)
        n_files += 1
        cumulative.append(cumulative[-1] + len(chunk))

    # Build compact_index
    # Count classes per depth
    num_classes_by_depth = {}
    for d in range(10):
        labels = set()
        for s in selected_samples:
            h = s[2]
            if d in h:
                labels.add(h[d])
        num_classes_by_depth[d] = len(labels)

    index = {
        "n_samples": len(selected_samples),
        "n_files": n_files,
        "samples_per_file": chunk_size,
        "file_list": [f"samples_{i:06d}.pkl" for i in range(n_files)],
        "cumulative_counts": cumulative,
        "label_type": "allen_atlas_hierarchical",
        "num_classes_by_depth": num_classes_by_depth,
        "has_coordinates": True,
        "coordinate_system": "allen_ccf_v3",
    }

    with open(os.path.join(output_dir, "compact_index.pkl"), "wb") as f:
        pickle.dump(index, f, protocol=4)

    # Count fine_id distribution
    fine_counts = Counter(samples[i][2]["fine_id"] for i in indices)
    print(f"    Written {len(selected_samples):,} samples in {n_files} files")
    print(f"    {len(fine_counts)} unique fine_ids, "
          f"min={min(fine_counts.values())}, max={max(fine_counts.values())}")

    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_data_dir", required=True,
                        help="Path to compact_atlas_ibl_v2_full")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for eval datasets")
    parser.add_argument("--probe_train_size", type=int, default=100000)
    parser.add_argument("--probe_val_size", type=int, default=25000)
    parser.add_argument("--test_fold_size", type=int, default=100000)
    parser.add_argument("--n_test_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy metadata from full dataset
    meta_src = os.path.join(args.full_data_dir, "metadata.pkl")
    if os.path.exists(meta_src):
        import shutil
        shutil.copy2(meta_src, os.path.join(args.output_dir, "metadata.pkl"))
        print(f"Copied metadata.pkl")

    # ── Build probe datasets from val subjects ────────────────────────────
    print("\n===== Loading val subjects (probe pool) =====")
    t0 = time.time()
    val_samples = load_all_samples(os.path.join(args.full_data_dir, "val"))
    print(f"  Loaded in {time.time()-t0:.0f}s")

    total_probe = args.probe_train_size + args.probe_val_size
    if total_probe > len(val_samples):
        print(f"  WARNING: need {total_probe:,} but only have {len(val_samples):,} val samples")
        # Scale down proportionally
        ratio = len(val_samples) / total_probe
        args.probe_train_size = int(args.probe_train_size * ratio)
        args.probe_val_size = int(args.probe_val_size * ratio)
        print(f"  Scaled to probe_train={args.probe_train_size:,}, probe_val={args.probe_val_size:,}")

    # Probe train
    print(f"\n  Building probe_train ({args.probe_train_size:,} samples)...")
    probe_train_idx = class_balanced_subsample(val_samples, args.probe_train_size, rng)
    write_compact(val_samples, probe_train_idx,
                  os.path.join(args.output_dir, "probe_train"))

    # Probe val (no overlap with probe_train)
    print(f"\n  Building probe_val ({args.probe_val_size:,} samples, no overlap)...")
    probe_val_idx = class_balanced_subsample(
        val_samples, args.probe_val_size, rng,
        exclude_indices=set(probe_train_idx)
    )
    write_compact(val_samples, probe_val_idx,
                  os.path.join(args.output_dir, "probe_val"))

    # Verify no overlap
    overlap = set(probe_train_idx) & set(probe_val_idx)
    print(f"  Probe train/val overlap: {len(overlap)} (should be 0)")
    assert len(overlap) == 0, "Probe train/val overlap detected!"

    del val_samples  # free memory

    # ── Build test folds from test subjects ───────────────────────────────
    print("\n===== Loading test subjects (test pool) =====")
    t0 = time.time()
    test_samples = load_all_samples(os.path.join(args.full_data_dir, "test"))
    print(f"  Loaded in {time.time()-t0:.0f}s")

    total_test = args.test_fold_size * args.n_test_folds
    if total_test > len(test_samples):
        print(f"  WARNING: need {total_test:,} but only have {len(test_samples):,} test samples")
        args.test_fold_size = len(test_samples) // args.n_test_folds
        print(f"  Scaled to test_fold_size={args.test_fold_size:,}")

    used_test_indices = set()
    for fold in range(args.n_test_folds):
        print(f"\n  Building test_fold_{fold} ({args.test_fold_size:,} samples, no overlap)...")
        fold_idx = class_balanced_subsample(
            test_samples, args.test_fold_size, rng,
            exclude_indices=used_test_indices
        )
        write_compact(test_samples, fold_idx,
                      os.path.join(args.output_dir, f"test_fold_{fold}"))
        used_test_indices.update(fold_idx)

    print(f"\n  Total test samples used: {len(used_test_indices):,} / {len(test_samples):,}")

    # ── Region coverage verification ─────────────────────────────────────
    print("\n===== Region Coverage Verification =====")

    def get_leaf_regions(samples, indices):
        leaves = set()
        for i in indices:
            h = samples[i][2]
            md = h.get("max_depth", 9)
            if md in h:
                leaves.add(h[md])
        return leaves

    def get_regions_by_depth(samples, indices):
        rbd = defaultdict(set)
        for i in indices:
            h = samples[i][2]
            md = h.get("max_depth", 9)
            for d in range(min(md + 1, 10)):
                if d in h:
                    rbd[d].add(h[d])
        return rbd

    # Reload val samples for probe coverage check
    val_samples = load_all_samples(os.path.join(args.full_data_dir, "val"))
    probe_train_regions = get_regions_by_depth(val_samples, probe_train_idx)
    probe_train_leaves = get_leaf_regions(val_samples, probe_train_idx)
    del val_samples

    # Reload test samples for fold coverage check
    test_samples = load_all_samples(os.path.join(args.full_data_dir, "test"))

    print(f"\n  Probe train: {len(probe_train_leaves)} leaf regions")

    for fold in range(args.n_test_folds):
        # Recover fold indices from saved files
        fold_dir = os.path.join(args.output_dir, f"test_fold_{fold}")
        fold_idx_pkl = pickle.load(open(os.path.join(fold_dir, "compact_index.pkl"), "rb"))
        # Reload fold samples to check regions
        fold_samples = []
        for i in range(fold_idx_pkl["n_files"]):
            fname = os.path.join(fold_dir, f"samples_{i:06d}.pkl")
            fold_samples.extend(pickle.load(open(fname, "rb")))

        fold_leaves = set()
        fold_rbd = defaultdict(set)
        for s in fold_samples:
            h = s[2]
            md = h.get("max_depth", 9)
            if md in h:
                fold_leaves.add(h[md])
            for d in range(min(md + 1, 10)):
                if d in h:
                    fold_rbd[d].add(h[d])

        shared_leaves = probe_train_leaves & fold_leaves
        print(f"  test_fold_{fold}: {len(fold_leaves)} leaf regions, "
              f"shared with probe_train: {len(shared_leaves)}")

        # Per-depth evaluable
        for d in range(10):
            pt = probe_train_regions.get(d, set())
            ft = fold_rbd.get(d, set())
            shared = pt & ft
            print(f"    depth {d}: probe_train={len(pt)}, fold={len(ft)}, evaluable={len(shared)}")

    del test_samples

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Eval Dataset Summary")
    print("=" * 60)
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Probe train:    {args.probe_train_size:,} samples (from val subjects)")
    print(f"  Probe val:      {args.probe_val_size:,} samples (from val subjects, no overlap)")
    print(f"  Test folds:     {args.n_test_folds} × {args.test_fold_size:,} samples (from test subjects)")
    print(f"  Total test:     {len(used_test_indices):,} samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
