#!/usr/bin/env python3
"""
Build compact balanced data by pre-extracting only the needed samples.

Session pickles are ~4GB each but we only need ~36 samples per session.
This script extracts just those samples and saves them in small files
that can be copied to /tmp for fast local I/O.

Supports train/val/test splits (session-level splits with no overlap).

Output structure:
- compact_data/
  - train/
    - compact_index.pkl  (metadata + file mapping)
    - samples_0000.pkl   (list of (signal, meta_str, label_id, coords) tuples)
    - samples_0001.pkl
    - ...
  - val/
    - compact_index.pkl
    - samples_0000.pkl
    - ...
  - test/
    - compact_index.pkl
    - samples_0000.pkl
    - ...

Sample tuple format:
    (signal, meta_str, label_id, coords)
    - signal: np.float32[3750] - 3 seconds at 1250 Hz
    - meta_str: "session_idx_probe_electrode_region"
    - label_id: int (0-12, coarse brain region class)
    - coords: (ap, dv, lr) CCF coordinates in micrometers, or None if not found

Meta string format:
    715093703_0_810755797_850264150_APN
        │      │     │         │       │
        │      │     │         │       └── brain_region
        │      │     │         └── electrode_id (lookup key for coordinates)
        │      │     └── probe_id
        │      └── time_window_index
        └── session_id
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd


class NumpyBackwardsCompatUnpickler(pickle.Unpickler):
    """
    Custom unpickler to handle numpy._core -> numpy.core compatibility.
    Needed when loading pickles created with numpy 2.x in numpy 1.x.
    """
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_pickle_compat(filepath: str):
    """Load pickle with numpy backwards compatibility."""
    with open(filepath, 'rb') as f:
        return NumpyBackwardsCompatUnpickler(f).load()


def load_coordinate_lookup(channels_csv: str) -> dict:
    """
    Load channels.csv and create electrode_id -> (ap, dv, lr) lookup.

    Returns:
        dict: electrode_id -> (ap, dv, lr) tuple, or None if coords are NaN
    """
    print(f"Loading coordinates from: {channels_csv}")
    df = pd.read_csv(channels_csv)

    lookup = {}
    valid_count = 0
    for _, row in df.iterrows():
        electrode_id = int(row['id'])
        ap = row['anterior_posterior_ccf_coordinate']
        dv = row['dorsal_ventral_ccf_coordinate']
        lr = row['left_right_ccf_coordinate']

        if pd.notna(ap) and pd.notna(dv) and pd.notna(lr):
            lookup[electrode_id] = (float(ap), float(dv), float(lr))
            valid_count += 1
        else:
            lookup[electrode_id] = None

    print(f"  Loaded {len(lookup):,} electrodes, {valid_count:,} with valid coordinates")
    return lookup


def parse_electrode_id(meta_str: str) -> int:
    """
    Extract electrode_id from meta string.

    Format: session_idx_probe_electrode_region
    Example: 715093703_0_810755797_850264150_APN
    """
    parts = meta_str.split('_')
    if len(parts) >= 4:
        try:
            return int(parts[3])
        except ValueError:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Build compact balanced data")
    parser.add_argument("--train-index", type=str,
                        default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/blind_localization/data/balanced_index_train.pkl'),
                        help="Path to balanced_index_train.pkl")
    parser.add_argument("--val-index", type=str,
                        default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/blind_localization/data/balanced_index_val.pkl'),
                        help="Path to balanced_index_val.pkl")
    parser.add_argument("--test-index", type=str,
                        default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/blind_localization/data/balanced_index_test.pkl'),
                        help="Path to balanced_index_test.pkl")
    parser.add_argument("--channels-csv", type=str,
                        default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/data/channels.csv'),
                        help="Path to channels.csv for CCF coordinates")
    parser.add_argument("--output-dir", type=str,
                        default=__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/blind_localization/data/compact_balanced'),
                        help="Output directory for compact data")
    parser.add_argument("--samples-per-file", type=int, default=200,
                        help="Max samples per output file (default: 200)")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip splits whose index files don't exist (useful for partial rebuilds)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load coordinate lookup from channels.csv
    coord_lookup = load_coordinate_lookup(args.channels_csv)

    # Define splits to process
    splits = [
        ("train", args.train_index),
        ("val", args.val_index),
        ("test", args.test_index),
    ]

    # Track results for summary
    results = {}

    # Process each split
    for split_name, index_path in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split: {index_path}")
        print('='*60)

        if not os.path.exists(index_path):
            if args.skip_missing:
                print(f"WARNING: Index file not found, skipping {split_name} split")
                results[split_name] = {"status": "skipped", "reason": "index not found"}
                continue
            else:
                print(f"ERROR: Index file not found: {index_path}")
                print("Use --skip-missing to skip missing splits")
                sys.exit(1)

        n_samples, total_size, coord_stats = process_split(
            index_path, output_dir / split_name, args.samples_per_file, coord_lookup
        )
        results[split_name] = {
            "status": "success",
            "n_samples": n_samples,
            "size_gb": total_size / 1e9,
            "coords_found": coord_stats["found"],
            "coords_missing": coord_stats["missing"]
        }

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_samples = 0
    total_size = 0
    total_coords_found = 0
    total_coords_missing = 0

    for split_name, result in results.items():
        if result["status"] == "success":
            print(f"  {split_name:6s}: {result['n_samples']:,} samples, {result['size_gb']:.2f} GB, "
                  f"coords: {result['coords_found']}/{result['coords_found']+result['coords_missing']}")
            total_samples += result["n_samples"]
            total_size += result["size_gb"]
            total_coords_found += result["coords_found"]
            total_coords_missing += result["coords_missing"]
        else:
            print(f"  {split_name:6s}: SKIPPED ({result['reason']})")

    print(f"  {'TOTAL':6s}: {total_samples:,} samples, {total_size:.2f} GB")
    coord_pct = 100 * total_coords_found / (total_coords_found + total_coords_missing) if total_samples > 0 else 0
    print(f"  Coordinates: {total_coords_found:,} found ({coord_pct:.1f}%), {total_coords_missing:,} missing")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nSample format: (signal, meta_str, label_id, coords)")
    print(f"  coords = (ap, dv, lr) in micrometers, or None if not found")


def process_split(index_path: str, output_dir: Path, samples_per_file: int, coord_lookup: dict) -> tuple:
    """
    Process one split (train, val, or test).

    Returns:
        Tuple of (n_samples, total_size_bytes, coord_stats)
        coord_stats = {"found": int, "missing": int}
    """

    # Load balanced index
    print(f"Loading balanced index: {index_path}")
    idx = load_pickle_compat(index_path)

    session_paths = idx["session_path_list"]
    file_ids = np.asarray(idx["file_ids"])
    rec_ids = np.asarray(idx["rec_ids"])
    label_ids = np.asarray(idx["label_ids"])

    # Check for session-level split metadata
    metadata = idx.get("metadata", {})
    split_type = metadata.get("split_type", "unknown")
    split_name = metadata.get("split", "unknown")

    print(f"Split type: {split_type}")
    print(f"Split name: {split_name}")

    n_samples = len(file_ids)
    print(f"Total samples: {n_samples}")
    print(f"Total sessions referenced: {len(session_paths)}")

    if n_samples == 0:
        print("WARNING: No samples in this split")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save empty compact index
        compact_index = {
            "n_samples": 0,
            "n_files": 0,
            "samples_per_file": samples_per_file,
            "file_list": [],
            "cumulative_counts": [0],
            "split_type": split_type,
            "split_name": split_name,
            "has_coordinates": True,
            "coordinate_system": "allen_ccf_v3",
            "coordinate_units": "micrometers",
        }
        index_out_path = output_dir / "compact_index.pkl"
        with open(index_out_path, "wb") as f:
            pickle.dump(compact_index, f)
        return 0, 0, {"found": 0, "missing": 0}

    # Group samples by session for efficient loading
    session_to_samples = defaultdict(list)
    for i in range(n_samples):
        session_idx = int(file_ids[i])
        session_to_samples[session_idx].append({
            "global_idx": i,
            "rec_id": int(rec_ids[i]),
            "label_id": int(label_ids[i]),
        })

    print(f"Sessions with samples: {len(session_to_samples)}")

    # Calculate average samples per session
    avg_samples = n_samples / len(session_to_samples) if session_to_samples else 0
    print(f"Average samples per session: {avg_samples:.1f}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each session and extract samples
    all_samples = []
    coords_found = 0
    coords_missing = 0

    for session_idx in tqdm(sorted(session_to_samples.keys()), desc="Extracting samples"):
        samples_info = session_to_samples[session_idx]
        session_path = session_paths[session_idx]

        if not os.path.exists(session_path):
            print(f"\nWARNING: Session not found: {session_path}")
            continue

        # Load session data (this is the slow part - only done once per session)
        try:
            session_data = load_pickle_compat(session_path)
        except Exception as e:
            print(f"\nERROR loading {session_path}: {e}")
            continue

        # Extract only the needed recordings
        for sample_info in samples_info:
            rec_id = sample_info["rec_id"]

            try:
                signal_np, meta_str = session_data[rec_id]

                # Look up coordinates from electrode_id
                electrode_id = parse_electrode_id(meta_str)
                if electrode_id is not None and electrode_id in coord_lookup:
                    coords = coord_lookup[electrode_id]  # (ap, dv, lr) or None
                    if coords is not None:
                        coords_found += 1
                    else:
                        coords_missing += 1
                else:
                    coords = None
                    coords_missing += 1

                all_samples.append({
                    "signal": np.asarray(signal_np, dtype=np.float32),
                    "meta_str": meta_str,
                    "label_id": sample_info["label_id"],
                    "coords": coords,
                    "original_idx": sample_info["global_idx"],
                })
            except Exception as e:
                print(f"\nERROR extracting rec_id={rec_id} from {session_path}: {e}")
                continue

        # Clear memory
        del session_data

    print(f"\nExtracted {len(all_samples)} samples total")
    print(f"Coordinates: {coords_found} found, {coords_missing} missing")

    # Sort by original index to maintain order
    all_samples.sort(key=lambda x: x["original_idx"])

    # Save in chunks
    n_files = (len(all_samples) + samples_per_file - 1) // samples_per_file if all_samples else 0

    compact_index = {
        "n_samples": len(all_samples),
        "n_files": n_files,
        "samples_per_file": samples_per_file,
        "file_list": [],
        "cumulative_counts": [0],  # For fast indexing
        "split_type": split_type,
        "split_name": split_name,
        "has_coordinates": True,
        "coordinate_system": "allen_ccf_v3",
        "coordinate_units": "micrometers",
        "coordinate_columns": ["anterior_posterior", "dorsal_ventral", "left_right"],
        "coords_found": coords_found,
        "coords_missing": coords_missing,
    }

    total_size = 0
    for i in range(n_files):
        start = i * samples_per_file
        end = min((i + 1) * samples_per_file, len(all_samples))
        chunk = all_samples[start:end]

        filename = f"samples_{i:04d}.pkl"
        filepath = output_dir / filename

        # Save as list of (signal, meta_str, label_id, coords) tuples
        chunk_data = [(s["signal"], s["meta_str"], s["label_id"], s["coords"]) for s in chunk]

        with open(filepath, "wb") as f:
            pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size = filepath.stat().st_size
        total_size += file_size

        compact_index["file_list"].append(str(filepath))
        compact_index["cumulative_counts"].append(compact_index["cumulative_counts"][-1] + len(chunk))

        if i < 3 or i == n_files - 1:
            print(f"  {filename}: {len(chunk)} samples, {file_size / 1e6:.1f} MB")
        elif i == 3:
            print(f"  ... ({n_files - 4} more files) ...")

    # Save compact index
    index_out_path = output_dir / "compact_index.pkl"
    with open(index_out_path, "wb") as f:
        pickle.dump(compact_index, f)

    print(f"\nCompact index saved to: {index_out_path}")
    print(f"Total files: {n_files}")
    print(f"Total size: {total_size / 1e9:.2f} GB")
    if total_size > 0:
        print(f"Compression ratio: {1351 / (total_size / 1e9):.1f}x smaller than original")

    return len(all_samples), total_size, {"found": coords_found, "missing": coords_missing}


if __name__ == "__main__":
    main()
