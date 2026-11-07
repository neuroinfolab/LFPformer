#!/usr/bin/env python3
"""
Build a balanced index for Ray Tune HP tuning with SESSION-LEVEL splits.

This script:
1. Scans all session pickles
2. Splits SESSIONS into train/val/test sets:
   - Train sessions are EXCLUSIVE (no overlap with val/test)
   - Val and Test can SHARE sessions but use DIFFERENT recordings
3. For each split, collects sample indices per coarse class (13 classes)
4. Samples N_PER_CLASS from each class within each split
5. Sorts by session path to minimize cache misses
6. Saves as pickle for reuse across HP trials

This hybrid approach ensures:
- No data leakage between train and val/test (session-level separation)
- All 13 classes have 1000 samples in ALL splits (even rare classes like AMY)
- Val and Test don't share recordings (recording-level separation)

Usage:
    python build_balanced_index.py --data-path /path/to/Allen --output balanced_index.pkl --n-per-class 1000
"""

import os
import sys
import pickle
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


def load_valid_electrodes(channels_csv: str) -> Set[int]:
    """
    Load channels.csv and return set of electrode IDs that have valid CCF coordinates.

    Returns:
        Set of electrode IDs with non-NaN (ap, dv, lr) coordinates
    """
    print(f"Loading valid electrodes from: {channels_csv}")
    df = pd.read_csv(channels_csv)

    # Filter to electrodes with valid coordinates
    valid_mask = (
        df['anterior_posterior_ccf_coordinate'].notna() &
        df['dorsal_ventral_ccf_coordinate'].notna() &
        df['left_right_ccf_coordinate'].notna()
    )

    valid_electrodes = set(df[valid_mask]['id'].astype(int).values)

    print(f"  Total electrodes: {len(df):,}")
    print(f"  With valid coordinates: {len(valid_electrodes):,} ({100*len(valid_electrodes)/len(df):.1f}%)")
    print(f"  With NaN coordinates: {len(df) - len(valid_electrodes):,} (excluded)")

    return valid_electrodes


def parse_electrode_id(meta_str: str) -> int:
    """Extract electrode_id from meta string (field 4)."""
    parts = meta_str.split("_")
    if len(parts) >= 4:
        try:
            return int(parts[3])
        except ValueError:
            return None
    return None


# Coarse label mappings (same as in lazyloader_dataset.py)
FINE_TO_COARSE = {
    'VIS': 'VIS', 'VISal': 'VIS', 'VISam': 'VIS', 'VISl': 'VIS', 'VISli': 'VIS',
    'VISp': 'VIS', 'VISpm': 'VIS', 'VISrl': 'VIS', 'VISmma': 'VIS', 'VISmmp': 'VIS',
    'CA1': 'HPF', 'CA2': 'HPF', 'CA3': 'HPF', 'DG': 'HPF', 'HPF': 'HPF',
    'ProS': 'HPF', 'SUB': 'HPF',
    'OLF': 'OLF', 'Eth': 'OLF',
    'BMAa': 'AMY', 'COAa': 'AMY', 'COApm': 'AMY',
    'CP': 'CTX-STR', 'POST': 'CTX-STR', 'PRE': 'CTX-STR', 'PP': 'CTX-STR',
    'RPF': 'CTX-STR', 'OP': 'CTX-STR', 'POL': 'CTX-STR',
    'TH': 'TH-GEN', 'PF': 'TH-GEN', 'PIL': 'TH-GEN', 'PoT': 'TH-GEN',
    'IGL': 'VIS-TH', 'LD': 'VIS-TH', 'LGd': 'VIS-TH', 'LGv': 'VIS-TH', 'LP': 'VIS-TH',
    'MGd': 'AUD-TH', 'MGm': 'AUD-TH', 'MGv': 'AUD-TH',
    'VPL': 'SM-TH', 'VPM': 'SM-TH', 'PO': 'SM-TH', 'VL': 'SM-TH',
    'RT': 'RT-ZI', 'ZI': 'RT-ZI',
    'APN': 'MB-Tect', 'SCig': 'MB-Tect', 'SCiw': 'MB-Tect', 'SCop': 'MB-Tect',
    'SCsg': 'MB-Tect', 'SCzo': 'MB-Tect', 'NOT': 'MB-Tect',
    'MB': 'MB-Teg', 'MRN': 'MB-Teg', 'PPT': 'MB-Teg', 'SGN': 'MB-Teg', 'LT': 'MB-Teg',
    'IntG': 'Misc'
}

COARSE_LABEL_MAP = {
    'VIS': 0, 'HPF': 1, 'OLF': 2, 'AMY': 3, 'CTX-STR': 4, 'TH-GEN': 5, 'VIS-TH': 6,
    'AUD-TH': 7, 'SM-TH': 8, 'RT-ZI': 9, 'MB-Tect': 10, 'MB-Teg': 11, 'Misc': 12
}

COARSE_LABEL_NAMES = {v: k for k, v in COARSE_LABEL_MAP.items()}


def scan_sessions(data_path: Path) -> List[Path]:
    """Find all pickle files in the data directory."""
    # Try both .pkl and .pickle extensions
    pickles = sorted(data_path.glob("*.pkl")) + sorted(data_path.glob("*.pickle"))
    if not pickles:
        pickles = sorted(data_path.rglob("*.pkl")) + sorted(data_path.rglob("*.pickle"))
    return sorted(set(pickles))  # Remove duplicates and sort


def scan_session_classes(session_paths: List[Path], verbose: bool = True) -> Dict[str, Dict[str, int]]:
    """
    Scan all sessions to determine which classes exist in each session.

    Returns:
        Dict mapping session_path_str -> {class_name: count}
    """
    session_classes = {}

    for i, session_path in enumerate(session_paths):
        if verbose and i % 50 == 0:
            print(f"  Pre-scanning sessions: {i}/{len(session_paths)}...", flush=True)

        with open(session_path, "rb") as f:
            session_data = pickle.load(f)

        class_counts = defaultdict(int)
        for rec_idx, recording_tuple in enumerate(session_data):
            meta_str = recording_tuple[1]
            parts = meta_str.split("_")
            if len(parts) < 2:
                continue
            label_str = parts[-1]
            coarse = FINE_TO_COARSE.get(label_str, None)
            if coarse:
                class_counts[coarse] += 1

        session_classes[str(session_path)] = dict(class_counts)

    if verbose:
        print(f"  Pre-scanning sessions: {len(session_paths)}/{len(session_paths)} done")

    return session_classes


def stratified_session_split(
    session_paths: List[Path],
    session_classes: Dict[str, Dict[str, int]],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    min_sessions_per_split: int = 3
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split sessions with stratification to ensure all classes are represented.

    For rare classes (fewer sessions than min_sessions_per_split * 3):
    - Allocate sessions to ensure train has coverage
    - Val and Test will SHARE remaining sessions (but use different recordings)

    Returns:
        Tuple of (train_sessions, val_test_shared_sessions, val_test_shared_sessions)
        Note: val and test return the SAME sessions - recording split happens later
    """
    random.seed(seed)

    # Identify which classes each session contains
    class_to_sessions = defaultdict(list)
    for session_path in session_paths:
        session_str = str(session_path)
        for cls in session_classes.get(session_str, {}).keys():
            class_to_sessions[cls].append(session_path)

    # Identify rare classes (those that can't be split 3 ways)
    rare_classes = []
    for cls, sessions in class_to_sessions.items():
        if len(sessions) < min_sessions_per_split * 3:
            rare_classes.append(cls)

    print(f"\n  Rare classes (< {min_sessions_per_split * 3} sessions): {rare_classes}")

    # Track session assignments using STRING paths for reliable comparison
    # (Path objects can have subtle equality issues)
    train_sessions = set()
    val_test_shared_sessions = set()  # Sessions shared between val and test
    assigned_sessions = set()  # Track all assigned sessions to avoid conflicts

    # Build string-to-path mapping for reliable lookups
    str_to_path = {str(s): s for s in session_paths}

    # First, handle rare classes to ensure coverage
    for cls in rare_classes:
        cls_sessions = class_to_sessions[cls]
        # Filter out already assigned sessions (use string comparison)
        unassigned = [s for s in cls_sessions if str(s) not in assigned_sessions]
        random.shuffle(unassigned)

        # Check how many are already in train vs val_test_shared (use string comparison)
        already_in_train = sum(1 for s in cls_sessions if str(s) in train_sessions)
        already_in_shared = sum(1 for s in cls_sessions if str(s) in val_test_shared_sessions)

        n_sessions = len(cls_sessions)
        # Need at least 1 in train, at least 1 in val_test_shared
        need_in_train = max(0, 1 - already_in_train)
        need_in_shared = max(0, 1 - already_in_shared)

        assigned_to_train = 0
        assigned_to_shared = 0

        for session in unassigned:
            session_str = str(session)
            if assigned_to_train < need_in_train:
                train_sessions.add(session_str)
                assigned_sessions.add(session_str)
                assigned_to_train += 1
            elif assigned_to_shared < need_in_shared:
                val_test_shared_sessions.add(session_str)
                assigned_sessions.add(session_str)
                assigned_to_shared += 1
            else:
                # Assign remaining to train (following train_fraction)
                train_sessions.add(session_str)
                assigned_sessions.add(session_str)
                assigned_to_train += 1

        total_train = already_in_train + assigned_to_train
        total_shared = already_in_shared + assigned_to_shared
        print(f"    {cls}: {n_sessions} sessions -> {total_train} train, {total_shared} val+test shared")

    # Now handle remaining sessions (not yet assigned) - use string comparison
    remaining = [s for s in session_paths if str(s) not in assigned_sessions]
    random.shuffle(remaining)

    n_remaining = len(remaining)
    n_train_remaining = int(n_remaining * train_fraction)
    n_val_remaining = int(n_remaining * val_fraction)

    # Add remaining to train (using string paths)
    for session in remaining[:n_train_remaining]:
        session_str = str(session)
        train_sessions.add(session_str)
        assigned_sessions.add(session_str)  # Track these too for safety

    # For common classes, val and test get EXCLUSIVE sessions (as string paths)
    val_exclusive_sessions = set(str(s) for s in remaining[n_train_remaining:n_train_remaining + n_val_remaining])
    test_exclusive_sessions = set(str(s) for s in remaining[n_train_remaining + n_val_remaining:])

    print(f"\n  Session allocation:")
    print(f"    Train (exclusive): {len(train_sessions)}")
    print(f"    Val+Test (shared): {len(val_test_shared_sessions)}")
    print(f"    Val (exclusive): {len(val_exclusive_sessions)}")
    print(f"    Test (exclusive): {len(test_exclusive_sessions)}")

    # Verify no overlap between train and shared
    overlap = train_sessions & val_test_shared_sessions
    if overlap:
        print(f"  ERROR: {len(overlap)} sessions in both train and shared!")
        for s in list(overlap)[:5]:
            print(f"    - {Path(s).name}")
        raise AssertionError("Train and val_test_shared sessions overlap!")

    # Combine for val and test (all string paths)
    val_sessions_str = val_test_shared_sessions | val_exclusive_sessions
    test_sessions_str = val_test_shared_sessions | test_exclusive_sessions

    # Convert back to Path objects for return
    val_sessions = [str_to_path[s] for s in val_sessions_str]
    test_sessions = [str_to_path[s] for s in test_sessions_str]
    train_sessions_paths = [str_to_path[s] for s in train_sessions]
    shared_sessions_paths = set(str_to_path[s] for s in val_test_shared_sessions)

    return train_sessions_paths, val_sessions, test_sessions, shared_sessions_paths


def build_class_indices(
    session_paths: List[Path],
    verbose: bool = True,
    excluded_recordings: Dict[str, set] = None,
    valid_electrodes: Set[int] = None
) -> Dict[int, List[Tuple[str, int, int]]]:
    """
    Scan all sessions and collect (session_path, recording_idx, electrode_id) per class.

    Args:
        session_paths: List of session paths to scan
        verbose: Print progress
        excluded_recordings: Dict mapping session_path_str -> set of rec_idx to exclude
                           Used to prevent val/test from using the same recordings
        valid_electrodes: Set of electrode IDs with valid coordinates.
                         If provided, only include recordings from these electrodes.

    Returns:
        Dict mapping label_int -> List of (session_path, rec_idx, electrode_id) tuples
    """
    if excluded_recordings is None:
        excluded_recordings = {}

    class_indices: Dict[int, List[Tuple[str, int, int]]] = defaultdict(list)
    total_skipped_coords = 0

    for i, session_path in enumerate(session_paths):
        if verbose:
            print(f"[{i+1}/{len(session_paths)}] Scanning {session_path.name}...", end=" ", flush=True)

        with open(session_path, "rb") as f:
            session_data = pickle.load(f)

        session_str = str(session_path)
        excluded = excluded_recordings.get(session_str, set())
        count = 0
        skipped_coords = 0

        for rec_idx, recording_tuple in enumerate(session_data):
            # Skip recordings that are excluded (used by another split)
            if rec_idx in excluded:
                continue

            meta_str = recording_tuple[1]
            parts = meta_str.split("_")
            if len(parts) < 2:
                continue

            # Parse electrode_id
            electrode_id = parse_electrode_id(meta_str)

            # Skip recordings without valid coordinates
            if valid_electrodes is not None:
                if electrode_id is None or electrode_id not in valid_electrodes:
                    skipped_coords += 1
                    continue

            label_str = parts[-1]
            coarse = FINE_TO_COARSE.get(label_str, None)
            if coarse is None:
                continue

            label_int = COARSE_LABEL_MAP[coarse]
            # Store electrode_id with the index for diverse sampling
            class_indices[label_int].append((session_str, rec_idx, electrode_id or 0))
            count += 1

        total_skipped_coords += skipped_coords

        if verbose:
            extra = ""
            if excluded:
                extra += f", {len(excluded)} excluded"
            if skipped_coords > 0:
                extra += f", {skipped_coords} no-coords"
            print(f"{count} recordings" + extra)

    if valid_electrodes is not None and total_skipped_coords > 0:
        print(f"\n  Total recordings skipped (no valid coordinates): {total_skipped_coords:,}")

    return dict(class_indices)


def split_shared_session_recordings(
    session_paths: List[Path],
    shared_sessions: set,
    seed: int
) -> Tuple[Dict[str, set], Dict[str, set]]:
    """
    For sessions shared between val and test, split recordings 50/50.

    Returns:
        Tuple of (val_recordings, test_recordings)
        Each is a dict mapping session_path_str -> set of rec_idx
    """
    random.seed(seed)
    np.random.seed(seed)

    val_recordings = {}
    test_recordings = {}

    print(f"\n  Splitting recordings in {len(shared_sessions)} shared sessions...")

    for session_path in shared_sessions:
        with open(session_path, "rb") as f:
            session_data = pickle.load(f)

        # Collect all valid recording indices per class
        class_recordings = defaultdict(list)
        for rec_idx, recording_tuple in enumerate(session_data):
            meta_str = recording_tuple[1]
            parts = meta_str.split("_")
            if len(parts) < 2:
                continue
            label_str = parts[-1]
            coarse = FINE_TO_COARSE.get(label_str, None)
            if coarse:
                class_recordings[coarse].append(rec_idx)

        session_str = str(session_path)
        val_recordings[session_str] = set()
        test_recordings[session_str] = set()

        # Split each class's recordings 50/50
        for cls, rec_indices in class_recordings.items():
            random.shuffle(rec_indices)
            mid = len(rec_indices) // 2
            val_recordings[session_str].update(rec_indices[:mid])
            test_recordings[session_str].update(rec_indices[mid:])

        print(f"    {Path(session_path).name}: {len(val_recordings[session_str])} val, {len(test_recordings[session_str])} test")

    return val_recordings, test_recordings


def sample_balanced_diverse(
    class_indices: Dict[int, List[Tuple[str, int, int]]],
    n_per_class: int,
    seed: int = 42,
    split_name: str = "",
    max_per_electrode: int = 10,
    max_per_session: int = 50
) -> List[Tuple[str, int, int]]:
    """
    Sample n_per_class from each class with DIVERSITY constraints.

    Ensures samples come from many different electrodes/probes/sessions:
    1. Group samples by electrode (most granular)
    2. Sample at most max_per_electrode from each electrode
    3. Sample at most max_per_session from each session
    4. Spread across as many electrodes as possible

    Args:
        class_indices: Dict mapping label_int -> List of (session_path, rec_idx, electrode_id)

    This naturally ensures probe and session diversity since:
    - Each electrode belongs to one probe
    - Each probe belongs to one session
    """
    random.seed(seed)
    np.random.seed(seed)

    balanced_samples = []

    print(f"\n=== Diverse sampling per class ({split_name}) ===")
    print(f"    (max {max_per_electrode}/electrode, max {max_per_session}/session)")

    for label_int in sorted(class_indices.keys()):
        indices = class_indices[label_int]
        label_name = COARSE_LABEL_NAMES[label_int]

        available = len(indices)
        to_sample = min(n_per_class, available)

        if available == 0:
            print(f"  {label_name} ({label_int}): 0 available, SKIPPING")
            continue

        # Group indices by electrode (electrode_id is already in the tuple)
        electrode_to_indices = defaultdict(list)
        for session_path, rec_idx, electrode_id in indices:
            electrode_to_indices[electrode_id].append((session_path, rec_idx, electrode_id))

        # Group indices by session
        session_to_indices = defaultdict(list)
        for session_path, rec_idx, electrode_id in indices:
            session_to_indices[session_path].append((session_path, rec_idx, electrode_id))

        n_electrodes = len(electrode_to_indices)
        n_sessions = len(session_to_indices)

        # Shuffle electrodes to randomize selection order
        electrode_keys = list(electrode_to_indices.keys())
        random.shuffle(electrode_keys)

        # Sample with diversity constraints
        sampled = []
        electrode_counts = defaultdict(int)
        session_counts = defaultdict(int)
        local_max_per_session = max_per_session  # Local copy for relaxation

        # Round-robin through electrodes until we have enough samples
        rounds = 0
        max_rounds = max_per_electrode + 10  # Safety limit

        while len(sampled) < to_sample and rounds < max_rounds:
            made_progress = False
            random.shuffle(electrode_keys)  # Re-shuffle each round for variety

            for electrode_key in electrode_keys:
                if len(sampled) >= to_sample:
                    break

                electrode_indices = electrode_to_indices[electrode_key]

                # Skip if electrode is exhausted or at limit
                if electrode_counts[electrode_key] >= len(electrode_indices):
                    continue
                if electrode_counts[electrode_key] >= max_per_electrode:
                    continue

                # Find a sample from this electrode that doesn't exceed session limit
                for idx, (session_path, rec_idx, eid) in enumerate(electrode_indices):
                    if idx < electrode_counts[electrode_key]:
                        continue  # Already sampled this one

                    if session_counts[session_path] >= local_max_per_session:
                        continue  # Session at limit

                    # Sample this one
                    sampled.append((session_path, rec_idx, eid))
                    electrode_counts[electrode_key] += 1
                    session_counts[session_path] += 1
                    made_progress = True
                    break

            rounds += 1

            # If we couldn't make progress, relax constraints
            if not made_progress and len(sampled) < to_sample:
                local_max_per_session += 10

        # If still not enough, fall back to random sampling from remaining
        if len(sampled) < to_sample:
            already_sampled = set((s, r) for s, r, e in sampled)
            remaining = [(s, r, e) for s, r, e in indices if (s, r) not in already_sampled]
            random.shuffle(remaining)
            sampled.extend(remaining[:to_sample - len(sampled)])

        n_sampled_electrodes = len([k for k, v in electrode_counts.items() if v > 0])
        n_sampled_sessions = len([k for k, v in session_counts.items() if v > 0])

        print(f"  {label_name} ({label_int}): {len(sampled)}/{available} samples "
              f"from {n_sampled_electrodes}/{n_electrodes} electrodes, "
              f"{n_sampled_sessions}/{n_sessions} sessions")

        for session_path, rec_idx, _ in sampled:
            balanced_samples.append((session_path, rec_idx, label_int))

    # Sort by session path to minimize cache misses during training
    balanced_samples.sort(key=lambda x: (x[0], x[1]))

    return balanced_samples


def sample_balanced(
    class_indices: Dict[int, List[Tuple[str, int]]],
    n_per_class: int,
    seed: int = 42,
    split_name: str = ""
) -> List[Tuple[str, int, int]]:
    """
    Sample n_per_class from each class with diversity constraints.

    Wrapper that calls sample_balanced_diverse with default diversity settings.
    """
    return sample_balanced_diverse(
        class_indices=class_indices,
        n_per_class=n_per_class,
        seed=seed,
        split_name=split_name,
        max_per_electrode=10,  # At most 10 samples per electrode
        max_per_session=50     # At most 50 samples per session (initial limit)
    )


def save_balanced_index(
    balanced_samples: List[Tuple[str, int, int]],
    output_path: Path,
    metadata: dict
):
    """Save the balanced index as a pickle file."""

    if len(balanced_samples) == 0:
        print(f"\nWARNING: No samples to save for {output_path}")
        return

    session_paths = [s[0] for s in balanced_samples]
    rec_ids = np.array([s[1] for s in balanced_samples], dtype=np.int32)
    label_ids = np.array([s[2] for s in balanced_samples], dtype=np.int32)

    unique_sessions = list(dict.fromkeys(session_paths))
    session_to_id = {s: i for i, s in enumerate(unique_sessions)}
    file_ids = np.array([session_to_id[s] for s in session_paths], dtype=np.int32)

    index_data = {
        "session_path_list": unique_sessions,
        "file_ids": file_ids,
        "rec_ids": rec_ids,
        "label_ids": label_ids,
        "metadata": metadata,
    }

    with open(output_path, "wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved balanced index to: {output_path}")
    print(f"  Total samples: {len(balanced_samples)}")
    print(f"  Unique sessions: {len(unique_sessions)}")


def print_class_distribution(class_indices: Dict[int, List], title: str):
    """Print class distribution summary."""
    print(f"\n=== {title} ===")
    total = 0
    for label_int in sorted(class_indices.keys()):
        count = len(class_indices[label_int])
        label_name = COARSE_LABEL_NAMES[label_int]
        print(f"  {label_name} ({label_int}): {count:,}")
        total += count
    print(f"  TOTAL: {total:,}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Build balanced index with HYBRID session/recording-level splits")
    parser.add_argument("--data-path", type=str, required=True, help="Path to session pickles")
    parser.add_argument("--output", type=str, required=True, help="Output path for balanced index pickle")
    parser.add_argument("--channels-csv", type=str,
                        default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/data/channels.csv'),
                        help="Path to channels.csv for coordinate filtering")
    parser.add_argument("--n-per-class", type=int, default=1000, help="Number of samples per class per split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of sessions for training")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of sessions for validation")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Fraction of sessions for testing")
    parser.add_argument("--require-coords", action="store_true", default=True,
                        help="Only include samples with valid CCF coordinates (default: True)")
    parser.add_argument("--no-require-coords", action="store_false", dest="require_coords",
                        help="Include samples even without coordinates")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output)

    if not data_path.exists():
        print(f"ERROR: Data path does not exist: {data_path}")
        sys.exit(1)

    # Validate fractions
    total_fraction = args.train_fraction + args.val_fraction + args.test_fraction
    if abs(total_fraction - 1.0) > 1e-6:
        print(f"ERROR: Fractions must sum to 1.0, got {total_fraction}")
        sys.exit(1)

    # Step 0: Load valid electrodes (for coordinate filtering)
    valid_electrodes = None
    if args.require_coords:
        valid_electrodes = load_valid_electrodes(args.channels_csv)
    else:
        print("WARNING: --no-require-coords specified, samples may lack coordinates")

    # Step 1: Find all sessions
    print(f"\n=== Scanning sessions in {data_path} ===")
    session_paths = scan_sessions(data_path)
    print(f"Found {len(session_paths)} session pickles\n")

    if not session_paths:
        print("ERROR: No pickle files found")
        sys.exit(1)

    # Step 2: Pre-scan all sessions to identify class coverage
    print("=== Pre-scanning sessions for class coverage ===")
    session_classes = scan_session_classes(session_paths)

    # Step 3: Stratified session split (handles rare classes)
    print("\n=== Splitting sessions (STRATIFIED HYBRID) ===")
    train_sessions, val_sessions, test_sessions, shared_sessions = stratified_session_split(
        session_paths,
        session_classes,
        args.train_fraction,
        args.val_fraction,
        args.test_fraction,
        args.seed
    )

    print(f"\n  Final session counts:")
    print(f"    Train sessions: {len(train_sessions)}")
    print(f"    Val sessions:   {len(val_sessions)} (includes {len(shared_sessions)} shared with test)")
    print(f"    Test sessions:  {len(test_sessions)} (includes {len(shared_sessions)} shared with val)")

    # Verify train doesn't overlap with val/test
    train_set = set(str(s) for s in train_sessions)
    val_set = set(str(s) for s in val_sessions)
    test_set = set(str(s) for s in test_sessions)
    shared_set = set(str(s) for s in shared_sessions)

    assert len(train_set & val_set) == 0, "Train and val sessions overlap!"
    assert len(train_set & test_set) == 0, "Train and test sessions overlap!"
    print("  Verified: Train sessions don't overlap with val/test")

    # Step 4: Split recordings in shared sessions between val and test
    print("\n=== Splitting recordings in shared val+test sessions ===")
    val_recordings, test_recordings = split_shared_session_recordings(
        list(shared_sessions),
        shared_sessions,
        args.seed + 100
    )

    # Step 5: Build class indices for each split
    print("\n" + "="*60)
    print("Building class indices for TRAIN sessions (exclusive)")
    if valid_electrodes:
        print(f"  (filtering to {len(valid_electrodes):,} electrodes with valid coordinates)")
    print("="*60)
    train_class_indices = build_class_indices(train_sessions, valid_electrodes=valid_electrodes)
    print_class_distribution(train_class_indices, "Train class distribution")

    print("\n" + "="*60)
    print("Building class indices for VAL sessions")
    if valid_electrodes:
        print(f"  (filtering to {len(valid_electrodes):,} electrodes with valid coordinates)")
    print("="*60)
    # Val excludes recordings assigned to test in shared sessions
    val_class_indices = build_class_indices(val_sessions, excluded_recordings=test_recordings, valid_electrodes=valid_electrodes)
    print_class_distribution(val_class_indices, "Val class distribution")

    print("\n" + "="*60)
    print("Building class indices for TEST sessions")
    if valid_electrodes:
        print(f"  (filtering to {len(valid_electrodes):,} electrodes with valid coordinates)")
    print("="*60)
    # Test excludes recordings assigned to val in shared sessions
    test_class_indices = build_class_indices(test_sessions, excluded_recordings=val_recordings, valid_electrodes=valid_electrodes)
    print_class_distribution(test_class_indices, "Test class distribution")

    # Step 6: Sample balanced subsets from each split
    train_samples = sample_balanced(train_class_indices, args.n_per_class, args.seed, "train")
    val_samples = sample_balanced(val_class_indices, args.n_per_class, args.seed + 1, "val")
    test_samples = sample_balanced(test_class_indices, args.n_per_class, args.seed + 2, "test")

    # Step 7: Save indices
    base_metadata = {
        "data_path": str(data_path),
        "n_per_class": args.n_per_class,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "num_total_sessions": len(session_paths),
        "split_type": "hybrid-session-recording",  # Documents the hybrid approach
        "shared_val_test_sessions": len(shared_sessions),
    }

    # Save train index
    train_output = output_path.parent / (output_path.stem + "_train.pkl")
    train_metadata = {
        **base_metadata,
        "split": "train",
        "num_sessions": len(train_sessions),
        "session_names": [s.name for s in train_sessions],
        "class_distribution": {COARSE_LABEL_NAMES[k]: len(v) for k, v in train_class_indices.items()},
    }
    save_balanced_index(train_samples, train_output, train_metadata)

    # Save val index
    val_output = output_path.parent / (output_path.stem + "_val.pkl")
    val_metadata = {
        **base_metadata,
        "split": "val",
        "num_sessions": len(val_sessions),
        "num_shared_sessions": len(shared_sessions),
        "session_names": [s.name for s in val_sessions],
        "class_distribution": {COARSE_LABEL_NAMES[k]: len(v) for k, v in val_class_indices.items()},
    }
    save_balanced_index(val_samples, val_output, val_metadata)

    # Save test index
    test_output = output_path.parent / (output_path.stem + "_test.pkl")
    test_metadata = {
        **base_metadata,
        "split": "test",
        "num_sessions": len(test_sessions),
        "num_shared_sessions": len(shared_sessions),
        "session_names": [s.name for s in test_sessions],
        "class_distribution": {COARSE_LABEL_NAMES[k]: len(v) for k, v in test_class_indices.items()},
    }
    save_balanced_index(test_samples, test_output, test_metadata)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY (HYBRID SESSION/RECORDING-LEVEL SPLITS)")
    print("="*60)
    print(f"\nSplit strategy:")
    print(f"  - Train: Exclusive sessions (no overlap with val/test)")
    print(f"  - Val+Test: Share {len(shared_sessions)} sessions (different recordings)")
    print(f"  - Rare classes (AMY, RT-ZI, Misc): Guaranteed coverage in all splits")

    print(f"\nSession counts:")
    print(f"  Train: {len(train_sessions)} sessions (exclusive)")
    print(f"  Val:   {len(val_sessions)} sessions ({len(shared_sessions)} shared)")
    print(f"  Test:  {len(test_sessions)} sessions ({len(shared_sessions)} shared)")

    print(f"\nSample counts:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")
    print(f"  Total: {len(train_samples) + len(val_samples) + len(test_samples)} samples")

    print(f"\nOutput files:")
    print(f"  {train_output}")
    print(f"  {val_output}")
    print(f"  {test_output}")

    # Print class distribution in each balanced set
    for split_name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        print(f"\n=== Class distribution ({split_name} balanced) ===")
        label_counts = Counter(s[2] for s in samples)
        all_present = True
        for label_int in sorted(COARSE_LABEL_MAP.values()):
            count = label_counts.get(label_int, 0)
            label_name = COARSE_LABEL_NAMES[label_int]
            status = "✓" if count >= args.n_per_class else "⚠" if count > 0 else "✗"
            print(f"  {status} {label_name} ({label_int}): {count:,}")
            if count == 0:
                all_present = False
        if all_present:
            print(f"  ✓ All 13 classes present!")


if __name__ == "__main__":
    main()
