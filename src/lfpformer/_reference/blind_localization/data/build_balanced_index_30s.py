#!/usr/bin/env python3
"""
Build a balanced index for 30-second continuous LFP segments.

Each sample is 10 consecutive 3s time windows from the SAME electrode/probe/session,
giving a genuine 30s continuous segment. This avoids padding or chunk-packing for
Whisper's 30s encoder input.

This script:
1. Scans all session pickles
2. Groups recordings by (probe_id, electrode_id) within each session
3. Finds valid 30s segments (10 consecutive time windows per electrode)
4. Filters to electrodes with valid CCF coordinates
5. Splits SESSIONS into train/val/test (reusing stratified_session_split logic)
6. For shared val/test sessions, splits at the ELECTRODE level
7. Samples 1000/class with diversity constraints

Usage:
    python build_balanced_index_30s.py --data-path /path/to/Allen --output balanced_index_30s.pkl
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

# Import shared utilities from the 3s index builder
from build_balanced_index import (
    FINE_TO_COARSE,
    COARSE_LABEL_MAP,
    COARSE_LABEL_NAMES,
    load_valid_electrodes,
    scan_sessions,
    scan_session_classes,
    stratified_session_split,
)

CHUNKS_PER_SEGMENT = 10  # 10 x 3s = 30s


def parse_meta_fields(meta_str: str) -> dict:
    """
    Parse a meta string into its component fields.

    Format: session_id_time_window_idx_probe_id_electrode_id_region
    Example: 715093703_0_810755797_850264150_APN
    """
    parts = meta_str.split("_")
    if len(parts) < 5:
        return None
    try:
        return {
            "session_id": parts[0],
            "time_window_idx": int(parts[1]),
            "probe_id": parts[2],
            "electrode_id": int(parts[3]),
            "region": parts[4],
        }
    except (ValueError, IndexError):
        return None


def scan_session_30s_segments(
    session_path: Path,
    valid_electrodes: Set[int] = None,
) -> List[dict]:
    """
    Scan a session pickle and find all valid 30s segments.

    Groups recordings by (probe_id, electrode_id), then finds runs of 10
    consecutive time windows.

    Returns:
        List of segment descriptors, each containing:
        - session_path: str
        - electrode_id: int
        - electrode_local_idx: int (rec_id of this electrode at time_window=0)
        - start_tw: int (starting time window index)
        - num_electrodes: int (total electrodes in session, for reconstruction)
        - region: str (fine region label)
        - coarse_region: str
        - label_int: int
        - rec_ids: list of 10 rec_ids for the consecutive chunks
        - probe_id: str
    """
    with open(session_path, "rb") as f:
        session_data = pickle.load(f)

    # Group recordings by (probe_id, electrode_id)
    electrode_records = defaultdict(list)  # (probe_id, electrode_id) -> [(rec_id, tw_idx, region)]

    for rec_id, recording_tuple in enumerate(session_data):
        meta_str = recording_tuple[1]
        fields = parse_meta_fields(meta_str)
        if fields is None:
            continue

        electrode_id = fields["electrode_id"]

        # Skip electrodes without valid coordinates
        if valid_electrodes is not None and electrode_id not in valid_electrodes:
            continue

        # Check if region maps to a coarse class
        coarse = FINE_TO_COARSE.get(fields["region"])
        if coarse is None:
            continue

        key = (fields["probe_id"], electrode_id)
        electrode_records[key].append((rec_id, fields["time_window_idx"], fields["region"]))

    # Determine num_electrodes: count distinct electrodes at time_window=0
    electrodes_at_tw0 = set()
    electrode_local_idx_map = {}  # electrode_id -> rec_id at tw=0
    for (probe_id, electrode_id), records in electrode_records.items():
        for rec_id, tw_idx, region in records:
            if tw_idx == 0:
                electrodes_at_tw0.add(electrode_id)
                electrode_local_idx_map[electrode_id] = rec_id
                break

    num_electrodes = len(electrodes_at_tw0)

    # Find valid 30s segments for each electrode
    segments = []
    session_str = str(session_path)

    for (probe_id, electrode_id), records in electrode_records.items():
        # Sort by time window
        records.sort(key=lambda x: x[1])

        # Build tw_idx -> rec_id mapping
        tw_to_rec = {}
        tw_to_region = {}
        for rec_id, tw_idx, region in records:
            tw_to_rec[tw_idx] = rec_id
            tw_to_region[tw_idx] = region

        # Find all valid starting positions for 10 consecutive windows
        all_tw = sorted(tw_to_rec.keys())
        if len(all_tw) < CHUNKS_PER_SEGMENT:
            continue

        for start_idx in range(len(all_tw) - CHUNKS_PER_SEGMENT + 1):
            start_tw = all_tw[start_idx]
            # Check that next 10 windows are truly consecutive
            expected_tws = list(range(start_tw, start_tw + CHUNKS_PER_SEGMENT))
            actual_tws = all_tw[start_idx:start_idx + CHUNKS_PER_SEGMENT]

            if actual_tws != expected_tws:
                continue

            # All 10 windows must have the same region
            regions = set(tw_to_region[tw] for tw in expected_tws)
            if len(regions) > 1:
                continue

            region = tw_to_region[start_tw]
            coarse = FINE_TO_COARSE.get(region)
            if coarse is None:
                continue

            rec_ids = [tw_to_rec[tw] for tw in expected_tws]

            # Get electrode_local_idx (rec_id at tw=0, or first rec_id as fallback)
            electrode_local_idx = electrode_local_idx_map.get(electrode_id, rec_ids[0])

            segments.append({
                "session_path": session_str,
                "electrode_id": electrode_id,
                "electrode_local_idx": electrode_local_idx,
                "start_tw": start_tw,
                "num_electrodes": num_electrodes,
                "region": region,
                "coarse_region": coarse,
                "label_int": COARSE_LABEL_MAP[coarse],
                "rec_ids": rec_ids,
                "probe_id": probe_id,
            })

    return segments


def scan_all_sessions_30s(
    session_paths: List[Path],
    valid_electrodes: Set[int] = None,
    verbose: bool = True,
) -> Dict[str, List[dict]]:
    """
    Scan all sessions and collect 30s segment descriptors.

    Returns:
        Dict mapping session_path_str -> list of segment descriptors
    """
    all_segments = {}

    for i, session_path in enumerate(session_paths):
        if verbose:
            print(f"[{i+1}/{len(session_paths)}] Scanning {session_path.name}...", end=" ", flush=True)

        segments = scan_session_30s_segments(session_path, valid_electrodes)

        all_segments[str(session_path)] = segments

        if verbose:
            # Summarize by class
            class_counts = Counter(s["coarse_region"] for s in segments)
            n_electrodes = len(set(s["electrode_id"] for s in segments))
            print(f"{len(segments)} segments from {n_electrodes} electrodes "
                  f"({', '.join(f'{k}:{v}' for k, v in sorted(class_counts.items()))})")

    return all_segments


def print_feasibility_report(
    all_segments: Dict[str, List[dict]],
    train_sessions: List[Path],
    val_sessions: List[Path],
    test_sessions: List[Path],
):
    """Print a feasibility report showing available 30s segments per class per split."""
    print("\n" + "=" * 70)
    print("FEASIBILITY REPORT: Available 30s segments per class per split")
    print("=" * 70)

    splits = {
        "train": set(str(s) for s in train_sessions),
        "val": set(str(s) for s in val_sessions),
        "test": set(str(s) for s in test_sessions),
    }

    header = f"{'Class':<12}"
    for split_name in ["train", "val", "test"]:
        header += f"  {split_name:>8}"
    header += f"  {'total':>8}"
    print(header)
    print("-" * 50)

    all_feasible = True
    for label_int in sorted(COARSE_LABEL_MAP.values()):
        label_name = COARSE_LABEL_NAMES[label_int]
        row = f"{label_name:<12}"
        total = 0
        for split_name, session_set in splits.items():
            count = sum(
                1 for sp, segs in all_segments.items()
                if sp in session_set
                for s in segs
                if s["label_int"] == label_int
            )
            row += f"  {count:>8,}"
            total += count
            if count < 1000:
                all_feasible = False
        row += f"  {total:>8,}"
        print(row)

    print("-" * 50)
    if all_feasible:
        print("All classes have >= 1000 segments in all splits.")
    else:
        print("WARNING: Some classes have < 1000 segments in some splits!")

    print()


def split_shared_session_electrodes(
    shared_sessions: set,
    all_segments: Dict[str, List[dict]],
    seed: int,
) -> Tuple[Dict[str, set], Dict[str, set]]:
    """
    For sessions shared between val and test, split at the ELECTRODE level.

    All segments from an electrode go to one split. This is different from the
    3s version which splits at the recording level.

    Returns:
        Tuple of (val_electrodes, test_electrodes)
        Each is a dict mapping session_path_str -> set of electrode_ids
    """
    random.seed(seed)

    val_electrodes = {}
    test_electrodes = {}

    print(f"\n  Splitting electrodes in {len(shared_sessions)} shared sessions...")

    for session_path in shared_sessions:
        session_str = str(session_path)
        segments = all_segments.get(session_str, [])

        # Collect unique electrodes per class
        class_electrodes = defaultdict(set)
        for seg in segments:
            class_electrodes[seg["coarse_region"]].add(seg["electrode_id"])

        val_electrodes[session_str] = set()
        test_electrodes[session_str] = set()

        # Split each class's electrodes 50/50
        for cls, electrode_ids in class_electrodes.items():
            electrode_list = sorted(electrode_ids)
            random.shuffle(electrode_list)
            mid = len(electrode_list) // 2
            val_electrodes[session_str].update(electrode_list[:mid])
            test_electrodes[session_str].update(electrode_list[mid:])

        print(f"    {Path(session_path).name}: {len(val_electrodes[session_str])} val electrodes, "
              f"{len(test_electrodes[session_str])} test electrodes")

    return val_electrodes, test_electrodes


def collect_split_segments(
    session_paths: List[Path],
    all_segments: Dict[str, List[dict]],
    excluded_electrodes: Dict[str, set] = None,
) -> Dict[int, List[dict]]:
    """
    Collect segments for a split, grouped by class.

    Args:
        session_paths: Sessions in this split
        all_segments: All scanned segments
        excluded_electrodes: Dict mapping session_path_str -> set of electrode_ids to exclude

    Returns:
        Dict mapping label_int -> list of segment descriptors
    """
    if excluded_electrodes is None:
        excluded_electrodes = {}

    class_segments = defaultdict(list)

    for session_path in session_paths:
        session_str = str(session_path)
        excluded = excluded_electrodes.get(session_str, set())

        for seg in all_segments.get(session_str, []):
            if seg["electrode_id"] in excluded:
                continue
            class_segments[seg["label_int"]].append(seg)

    return dict(class_segments)


def sample_balanced_diverse_30s(
    class_segments: Dict[int, List[dict]],
    n_per_class: int,
    seed: int = 42,
    split_name: str = "",
    max_per_electrode: int = 10,
    max_per_session: int = 50,
) -> List[dict]:
    """
    Sample n_per_class 30s segments from each class with diversity constraints.

    Round-robin across electrodes, max 10/electrode, max 50/session.
    """
    random.seed(seed)
    np.random.seed(seed)

    balanced_segments = []

    print(f"\n=== Diverse sampling per class ({split_name}) ===")
    print(f"    (max {max_per_electrode}/electrode, max {max_per_session}/session)")

    for label_int in sorted(class_segments.keys()):
        segments = class_segments[label_int]
        label_name = COARSE_LABEL_NAMES[label_int]

        available = len(segments)
        to_sample = min(n_per_class, available)

        if available == 0:
            print(f"  {label_name} ({label_int}): 0 available, SKIPPING")
            continue

        # Group by electrode
        electrode_to_segs = defaultdict(list)
        for seg in segments:
            electrode_to_segs[seg["electrode_id"]].append(seg)

        # Group by session
        session_to_segs = defaultdict(list)
        for seg in segments:
            session_to_segs[seg["session_path"]].append(seg)

        n_electrodes = len(electrode_to_segs)
        n_sessions = len(session_to_segs)

        # Shuffle within each electrode group for variety
        for eid in electrode_to_segs:
            random.shuffle(electrode_to_segs[eid])

        electrode_keys = list(electrode_to_segs.keys())
        random.shuffle(electrode_keys)

        sampled = []
        electrode_counts = defaultdict(int)
        session_counts = defaultdict(int)
        local_max_per_session = max_per_session

        rounds = 0
        max_rounds = max_per_electrode + 10

        while len(sampled) < to_sample and rounds < max_rounds:
            made_progress = False
            random.shuffle(electrode_keys)

            for electrode_key in electrode_keys:
                if len(sampled) >= to_sample:
                    break

                electrode_segs = electrode_to_segs[electrode_key]

                if electrode_counts[electrode_key] >= len(electrode_segs):
                    continue
                if electrode_counts[electrode_key] >= max_per_electrode:
                    continue

                # Find a segment that doesn't exceed session limit
                for idx in range(electrode_counts[electrode_key], len(electrode_segs)):
                    seg = electrode_segs[idx]
                    if session_counts[seg["session_path"]] >= local_max_per_session:
                        continue

                    sampled.append(seg)
                    electrode_counts[electrode_key] = idx + 1
                    session_counts[seg["session_path"]] += 1
                    made_progress = True
                    break

            rounds += 1

            if not made_progress and len(sampled) < to_sample:
                local_max_per_session += 10

        # Fallback: random sample from remaining
        if len(sampled) < to_sample:
            already = set(id(s) for s in sampled)
            remaining = [s for s in segments if id(s) not in already]
            random.shuffle(remaining)
            sampled.extend(remaining[:to_sample - len(sampled)])

        n_sampled_electrodes = len([k for k, v in electrode_counts.items() if v > 0])
        n_sampled_sessions = len([k for k, v in session_counts.items() if v > 0])

        print(f"  {label_name} ({label_int}): {len(sampled)}/{available} segments "
              f"from {n_sampled_electrodes}/{n_electrodes} electrodes, "
              f"{n_sampled_sessions}/{n_sessions} sessions")

        balanced_segments.extend(sampled)

    # Sort by session path for cache-friendly loading
    balanced_segments.sort(key=lambda s: (s["session_path"], s["rec_ids"][0]))

    return balanced_segments


def save_balanced_index_30s(
    balanced_segments: List[dict],
    output_path: Path,
    metadata: dict,
):
    """Save the 30s balanced index as a pickle file."""
    if len(balanced_segments) == 0:
        print(f"\nWARNING: No segments to save for {output_path}")
        return

    # Build session path list
    session_paths_ordered = []
    session_to_id = {}
    for seg in balanced_segments:
        sp = seg["session_path"]
        if sp not in session_to_id:
            session_to_id[sp] = len(session_paths_ordered)
            session_paths_ordered.append(sp)

    file_ids = np.array([session_to_id[s["session_path"]] for s in balanced_segments], dtype=np.int32)
    label_ids = np.array([s["label_int"] for s in balanced_segments], dtype=np.int32)

    # Build segment descriptors (compact form for the data extractor)
    segments_list = []
    for seg in balanced_segments:
        segments_list.append({
            "electrode_local_idx": seg["electrode_local_idx"],
            "start_tw": seg["start_tw"],
            "num_electrodes": seg["num_electrodes"],
            "rec_ids": seg["rec_ids"],
            "electrode_id": seg["electrode_id"],
            "probe_id": seg["probe_id"],
            "region": seg["region"],
        })

    index_data = {
        "session_path_list": session_paths_ordered,
        "file_ids": file_ids,
        "segments": segments_list,
        "label_ids": label_ids,
        "metadata": {
            **metadata,
            "chunk_duration_s": 30,
            "chunks_per_segment": CHUNKS_PER_SEGMENT,
            "samples_per_chunk_s": 3,
            "sample_rate_hz": 1250,
            "samples_per_segment": 37500,  # 30s * 1250 Hz
        },
    }

    with open(output_path, "wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved 30s balanced index to: {output_path}")
    print(f"  Total segments: {len(balanced_segments)}")
    print(f"  Unique sessions: {len(session_paths_ordered)}")


def main():
    parser = argparse.ArgumentParser(
        description="Build balanced index for 30-second continuous LFP segments"
    )
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to session pickles")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for balanced index pickle (base name)")
    parser.add_argument("--channels-csv", type=str,
                        default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/data/channels.csv'),
                        help="Path to channels.csv for coordinate filtering")
    parser.add_argument("--n-per-class", type=int, default=1000,
                        help="Number of segments per class per split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--require-coords", action="store_true", default=True)
    parser.add_argument("--no-require-coords", action="store_false", dest="require_coords")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output)

    if not data_path.exists():
        print(f"ERROR: Data path does not exist: {data_path}")
        sys.exit(1)

    total_fraction = args.train_fraction + args.val_fraction + args.test_fraction
    if abs(total_fraction - 1.0) > 1e-6:
        print(f"ERROR: Fractions must sum to 1.0, got {total_fraction}")
        sys.exit(1)

    # Step 0: Load valid electrodes
    valid_electrodes = None
    if args.require_coords:
        valid_electrodes = load_valid_electrodes(args.channels_csv)

    # Step 1: Find all sessions
    print(f"\n=== Scanning sessions in {data_path} ===")
    session_paths = scan_sessions(data_path)
    print(f"Found {len(session_paths)} session pickles\n")

    if not session_paths:
        print("ERROR: No pickle files found")
        sys.exit(1)

    # Step 2: Pre-scan for class coverage (needed for stratified session split)
    print("=== Pre-scanning sessions for class coverage ===")
    session_classes = scan_session_classes(session_paths)

    # Step 3: Stratified session split (reuse existing logic)
    print("\n=== Splitting sessions (STRATIFIED HYBRID) ===")
    train_sessions, val_sessions, test_sessions, shared_sessions = stratified_session_split(
        session_paths,
        session_classes,
        args.train_fraction,
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )

    print(f"\n  Final session counts:")
    print(f"    Train: {len(train_sessions)}")
    print(f"    Val:   {len(val_sessions)} (includes {len(shared_sessions)} shared)")
    print(f"    Test:  {len(test_sessions)} (includes {len(shared_sessions)} shared)")

    # Verify no train/val/test overlap
    train_set = set(str(s) for s in train_sessions)
    val_set = set(str(s) for s in val_sessions)
    test_set = set(str(s) for s in test_sessions)
    assert len(train_set & val_set) == 0, "Train and val sessions overlap!"
    assert len(train_set & test_set) == 0, "Train and test sessions overlap!"
    print("  Verified: Train sessions don't overlap with val/test")

    # Step 4: Scan all sessions for 30s segments
    print("\n" + "=" * 60)
    print("Scanning all sessions for 30s segments")
    print("=" * 60)
    all_segments = scan_all_sessions_30s(session_paths, valid_electrodes)

    total_segments = sum(len(segs) for segs in all_segments.values())
    print(f"\nTotal 30s segments found: {total_segments:,}")

    # Step 5: Print feasibility report
    print_feasibility_report(all_segments, train_sessions, val_sessions, test_sessions)

    # Step 6: Split electrodes in shared sessions
    print("=== Splitting electrodes in shared val+test sessions ===")
    val_excluded_electrodes, test_excluded_electrodes = split_shared_session_electrodes(
        shared_sessions, all_segments, args.seed + 100
    )

    # Step 7: Collect segments per split
    print("\n" + "=" * 60)
    print("Collecting segments for TRAIN")
    print("=" * 60)
    train_class_segs = collect_split_segments(train_sessions, all_segments)
    for label_int in sorted(train_class_segs.keys()):
        print(f"  {COARSE_LABEL_NAMES[label_int]}: {len(train_class_segs[label_int]):,} segments")

    print("\n" + "=" * 60)
    print("Collecting segments for VAL")
    print("=" * 60)
    val_class_segs = collect_split_segments(val_sessions, all_segments, test_excluded_electrodes)
    for label_int in sorted(val_class_segs.keys()):
        print(f"  {COARSE_LABEL_NAMES[label_int]}: {len(val_class_segs[label_int]):,} segments")

    print("\n" + "=" * 60)
    print("Collecting segments for TEST")
    print("=" * 60)
    test_class_segs = collect_split_segments(test_sessions, all_segments, val_excluded_electrodes)
    for label_int in sorted(test_class_segs.keys()):
        print(f"  {COARSE_LABEL_NAMES[label_int]}: {len(test_class_segs[label_int]):,} segments")

    # Step 8: Sample balanced subsets
    train_segments = sample_balanced_diverse_30s(
        train_class_segs, args.n_per_class, args.seed, "train"
    )
    val_segments = sample_balanced_diverse_30s(
        val_class_segs, args.n_per_class, args.seed + 1, "val"
    )
    test_segments = sample_balanced_diverse_30s(
        test_class_segs, args.n_per_class, args.seed + 2, "test"
    )

    # Step 9: Save indices
    base_metadata = {
        "data_path": str(data_path),
        "n_per_class": args.n_per_class,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "num_total_sessions": len(session_paths),
        "split_type": "hybrid-session-electrode-30s",
        "shared_val_test_sessions": len(shared_sessions),
    }

    train_output = output_path.parent / (output_path.stem + "_train.pkl")
    save_balanced_index_30s(train_segments, train_output, {
        **base_metadata,
        "split": "train",
        "num_sessions": len(train_sessions),
        "session_names": [s.name for s in train_sessions],
    })

    val_output = output_path.parent / (output_path.stem + "_val.pkl")
    save_balanced_index_30s(val_segments, val_output, {
        **base_metadata,
        "split": "val",
        "num_sessions": len(val_sessions),
        "num_shared_sessions": len(shared_sessions),
        "session_names": [s.name for s in val_sessions],
    })

    test_output = output_path.parent / (output_path.stem + "_test.pkl")
    save_balanced_index_30s(test_segments, test_output, {
        **base_metadata,
        "split": "test",
        "num_sessions": len(test_sessions),
        "num_shared_sessions": len(shared_sessions),
        "session_names": [s.name for s in test_sessions],
    })

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (30-SECOND CONTINUOUS SEGMENTS)")
    print("=" * 60)

    print(f"\nSegment duration: 30s ({CHUNKS_PER_SEGMENT} x 3s consecutive windows)")
    print(f"Signal length: 37,500 samples (30s x 1250 Hz)")

    print(f"\nSplit strategy:")
    print(f"  - Train: Exclusive sessions")
    print(f"  - Val+Test: Share {len(shared_sessions)} sessions (different ELECTRODES)")
    print(f"  - Rare classes: Guaranteed coverage in all splits")

    print(f"\nSession counts:")
    print(f"  Train: {len(train_sessions)}")
    print(f"  Val:   {len(val_sessions)} ({len(shared_sessions)} shared)")
    print(f"  Test:  {len(test_sessions)} ({len(shared_sessions)} shared)")

    print(f"\nSegment counts:")
    print(f"  Train: {len(train_segments)}")
    print(f"  Val:   {len(val_segments)}")
    print(f"  Test:  {len(test_segments)}")
    print(f"  Total: {len(train_segments) + len(val_segments) + len(test_segments)}")

    print(f"\nOutput files:")
    print(f"  {train_output}")
    print(f"  {val_output}")
    print(f"  {test_output}")

    # Class distribution
    for split_name, segments in [("Train", train_segments), ("Val", val_segments), ("Test", test_segments)]:
        print(f"\n=== Class distribution ({split_name} balanced) ===")
        label_counts = Counter(s["label_int"] for s in segments)
        for label_int in sorted(COARSE_LABEL_MAP.values()):
            count = label_counts.get(label_int, 0)
            label_name = COARSE_LABEL_NAMES[label_int]
            status = "OK" if count >= args.n_per_class else "LOW" if count > 0 else "MISSING"
            print(f"  [{status:>7s}] {label_name} ({label_int}): {count:,}")


if __name__ == "__main__":
    main()
