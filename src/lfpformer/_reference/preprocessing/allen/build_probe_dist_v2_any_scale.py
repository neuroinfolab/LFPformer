#!/usr/bin/env python3
"""Build compact_atlas_{SCALE}_probe_dist_v2 dataset.

Combined script: seeds from any compact_atlas, fetches sibling channels from raw
pickles, flattens directly to individual flat samples (no intermediate group format).

Pipeline:
  compact_atlas_{scale}  +  raw Allen pickles  ->  compact_atlas_{scale}_probe_dist_v2

For each seed in the source compact_atlas:
  1. Parse (session, probe, t_win, electrode) from meta_str
  2. Load the corresponding raw pickle once per (session, probe) pair
  3. Collect ALL channels at that time window (seed + siblings)
  4. Add probe_horizontal_position + probe_vertical_position from channels.csv
  5. Flatten to individual (signal, meta_str, label, shank_pos, ccf_coords) samples

Then subsample to TARGET_SAMPLES per split using probe-proportional allocation
(same logic as build_probe_dist_v2_5k.py) to preserve within-probe batching.

Sample format (5-tuple):
    (signal[3750], meta_str, label_dict_or_-1, shank_pos[2], ccf_coords[3])
    - signal:     np.float32[3750]  raw LFP @ 1250 Hz
    - meta_str:   str               session_twin_probe_elec_region
    - label:      dict | -1         atlas hierarchy dict (seed) or -1 (sibling)
    - shank_pos:  np.float32[2]     [probe_horizontal_position, probe_vertical_position] in µm
    - ccf_coords: np.float32[3]     (ap, dv, lr) in µm from Allen CCF

probe_groups.pkl  maps  probe_id -> [global_sample_indices]  for ProbeBatchSampler.

Usage:
    singularity exec /workspace/ws1/containers/w2v2_cuda_cu128.sif \\
        python3 script/build_probe_dist_v2_any_scale.py \\
            --src-dir  blind_localization/data/atlas_data/compact_atlas_25k \\
            --dst-dir  blind_localization/data/atlas_data/compact_atlas_25k_probe_dist_v2 \\
            --target-samples 25000

    singularity exec /workspace/ws1/containers/w2v2_cuda_cu128.sif \\
        python3 script/build_probe_dist_v2_any_scale.py \\
            --src-dir  blind_localization/data/atlas_data/compact_atlas_75k \\
            --dst-dir  blind_localization/data/atlas_data/compact_atlas_75k_probe_dist_v2 \\
            --target-samples 75000
"""

import argparse
import gc
import glob
import os
import pickle
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Channel lookup: electrode_id -> (horiz_µm, vert_µm, ap, dv, lr)
# ---------------------------------------------------------------------------
def build_channel_lookup(channels_csv):
    df = pd.read_csv(channels_csv)
    lookup = {}
    for _, row in df.iterrows():
        eid   = int(row["id"])
        horiz = float(row["probe_horizontal_position"])
        vert  = float(row["probe_vertical_position"])
        ap    = float(row["anterior_posterior_ccf_coordinate"]) \
                if not pd.isna(row["anterior_posterior_ccf_coordinate"]) else float("nan")
        dv    = float(row["dorsal_ventral_ccf_coordinate"]) \
                if not pd.isna(row["dorsal_ventral_ccf_coordinate"]) else float("nan")
        lr    = float(row["left_right_ccf_coordinate"]) \
                if not pd.isna(row["left_right_ccf_coordinate"]) else float("nan")
        lookup[eid] = (horiz, vert, ap, dv, lr)
    print(f"  channel lookup: {len(lookup):,} entries")
    return lookup


# ---------------------------------------------------------------------------
# Index a raw probe pickle for requested time windows only
#   returns: { t_win_str: { elec_str: np.float32[3750] } }
# ---------------------------------------------------------------------------
def index_probe(pickle_path, needed_twins):
    with open(pickle_path, "rb") as f:
        recs = pickle.load(f)
    index = defaultdict(dict)
    for signal, meta in recs:
        parts = meta.split("_")
        if len(parts) < 5:
            continue  # skip malformed entries
        t_win, elec = parts[1], parts[3]
        if t_win in needed_twins:
            index[t_win][elec] = np.asarray(signal, dtype=np.float32)
    return index


# ---------------------------------------------------------------------------
# For one (seed_sample, t_win), collect all sibling channels from probe_index.
# Returns a list of flat (signal, meta_str, label, shank_pos, ccf) tuples,
# capped at max_channels_per_group (uniform vertical coverage, seed always kept).
# Returns None if seed channel is unavailable or has bad signal.
# ---------------------------------------------------------------------------
def collect_flat_samples(seed_sample, t_win, probe_index, ch_lookup,
                         max_channels_per_group, rng):
    parts = seed_sample[1].split("_")
    seed_elec = parts[3]

    if t_win not in probe_index:
        return None  # time window missing from raw pickle

    tw_index = probe_index[t_win]
    if seed_elec not in tw_index:
        return None  # seed electrode missing

    # Validate seed signal
    seed_sig = tw_index[seed_elec]
    if np.any(np.isnan(seed_sig)) or np.all(seed_sig == 0.0):
        return None

    # Validate seed electrode has depth
    seed_eid = int(seed_elec)
    if seed_eid not in ch_lookup or np.isnan(ch_lookup[seed_eid][1]):
        return None

    # Collect all valid channels at this time window
    channels = []  # (horiz, vert, signal, ap, dv, lr, label, meta_str)
    for elec_str, sig in tw_index.items():
        eid = int(elec_str)
        if eid not in ch_lookup:
            continue
        horiz, vert, ap, dv, lr = ch_lookup[eid]
        if np.isnan(vert):
            continue
        if np.any(np.isnan(sig)) or np.all(sig == 0.0):
            continue
        if elec_str == seed_elec:
            label    = seed_sample[2]  # atlas dict from compact_atlas seed
            meta_str = seed_sample[1]
        else:
            label    = -1
            meta_str = f"{parts[0]}_{t_win}_{parts[2]}_{elec_str}_unknown"
        channels.append((horiz, vert, sig, ap, dv, lr, label, meta_str))

    if not channels:
        return None

    # Cap to max_channels_per_group, always keeping seed, uniform vertical coverage
    if len(channels) > max_channels_per_group:
        seed_ch = next(c for c in channels if c[7] == seed_sample[1])
        others  = [c for c in channels if c[7] != seed_sample[1]]
        others.sort(key=lambda x: x[1])  # sort by vertical position
        n_keep = max_channels_per_group - 1
        idxs   = np.linspace(0, len(others) - 1, n_keep, dtype=int)
        others = [others[i] for i in idxs]
        channels = [seed_ch] + others
        rng.shuffle(channels)

    result = []
    for (horiz, vert, sig, ap, dv, lr, label, meta_str) in channels:
        result.append((
            sig,
            meta_str,
            label,
            np.array([horiz, vert], dtype=np.float32),
            np.array([ap, dv, lr], dtype=np.float32),
        ))
    return result


# ---------------------------------------------------------------------------
# Parse probe_id from meta_str: "session_twin_probe_elec_region"
# probe_id = "session_probe"
# ---------------------------------------------------------------------------
def get_probe_id(meta_str):
    parts = meta_str.split("_")
    return f"{parts[0]}_{parts[2]}"


# ---------------------------------------------------------------------------
# Proportional allocation: distribute target_samples across probes
# (mirrors build_probe_dist_v2_5k.py logic)
# ---------------------------------------------------------------------------
def allocate_samples(probe_samples, target_samples, min_per_probe, rng):
    # Filter probes below minimum
    eligible = {pid: samps for pid, samps in probe_samples.items()
                if len(samps) >= min_per_probe}
    print(f"  {len(eligible)} probes with >= {min_per_probe} channels "
          f"({sum(len(v) for v in eligible.values()):,} total channels)")

    if not eligible:
        raise RuntimeError(f"No probes with >= {min_per_probe} channels")

    # Cap number of probes so min_per_probe * n_probes <= target_samples
    max_probes = target_samples // min_per_probe
    if max_probes == 0:
        raise RuntimeError(
            f"target_samples={target_samples} < min_per_probe={min_per_probe}. "
            f"Lower --min-per-probe or raise --target-samples."
        )
    probe_ids  = sorted(eligible.keys())
    if len(probe_ids) > max_probes:
        probe_ids = sorted(probe_ids, key=lambda p: len(eligible[p]), reverse=True)
        probe_ids = sorted(probe_ids[:max_probes])
        print(f"  Capped to {len(probe_ids)} probes "
              f"(max {max_probes} for TARGET={target_samples}, MIN={min_per_probe})")

    total_eligible = sum(len(eligible[pid]) for pid in probe_ids)

    # Proportional allocation, floored to min_per_probe
    allocations = {}
    for pid in probe_ids:
        frac  = len(eligible[pid]) / total_eligible
        alloc = max(min_per_probe, int(round(frac * target_samples)))
        alloc = min(alloc, len(eligible[pid]))
        allocations[pid] = alloc

    # Adjust to hit exactly target_samples
    actual    = sum(allocations.values())
    remaining = target_samples - actual

    if remaining > 0:
        for pid in rng.permutation(probe_ids):
            headroom = len(eligible[pid]) - allocations[pid]
            add      = min(remaining, headroom)
            allocations[pid] += add
            remaining -= add
            if remaining <= 0:
                break
        if remaining > 0:
            actual_total = sum(allocations.values())
            print(f"  WARNING: only {actual_total:,} samples available "
                  f"(target={target_samples:,}, shortfall={remaining:,}). "
                  f"Proceeding with available samples.")
    elif remaining < 0:
        for pid in sorted(probe_ids, key=lambda p: allocations[p], reverse=True):
            excess = allocations[pid] - min_per_probe
            remove = min(-remaining, excess)
            allocations[pid] -= remove
            remaining += remove
            if remaining >= 0:
                break

    return eligible, probe_ids, allocations


# ---------------------------------------------------------------------------
# Process one split
# ---------------------------------------------------------------------------
def process_split(split, src_dir, dst_dir, raw_dir, ch_lookup,
                  target_samples, min_per_probe, max_channels_per_group, rng,
                  exclude_sessions=None):
    """exclude_sessions: set of session_id strings to skip (e.g. lp_eval sessions).
    If all seeds in the native split are excluded, falls back to the train split
    of src_dir so that val/test can still be built without lp_eval contamination."""
    src = os.path.join(src_dir, split)
    dst = os.path.join(dst_dir, split)
    os.makedirs(dst, exist_ok=True)

    # Load all seed samples
    seeds = []
    for fp in sorted(glob.glob(os.path.join(src, "samples_*.pkl"))):
        seeds.extend(pickle.load(open(fp, "rb")))
    print(f"  {len(seeds):,} seed samples")

    # Filter excluded sessions
    if exclude_sessions:
        before = len(seeds)
        seeds = [s for s in seeds if s[1].split("_")[0] not in exclude_sessions]
        removed = before - len(seeds)
        if removed:
            print(f"  excluded {removed:,} seeds from {len(exclude_sessions)} lp_eval sessions "
                  f"({len(seeds):,} remain)")
        if not seeds:
            # Fall back to train split (no lp_eval contamination there)
            fallback_src = os.path.join(src_dir, "train")
            print(f"  [fallback] no seeds remain after exclusion — "
                  f"sourcing from train split ({fallback_src})")
            for fp in sorted(glob.glob(os.path.join(fallback_src, "samples_*.pkl"))):
                seeds.extend(pickle.load(open(fp, "rb")))
            # Re-filter (train seeds should already be clean, but be safe)
            seeds = [s for s in seeds if s[1].split("_")[0] not in exclude_sessions]
            print(f"  [fallback] {len(seeds):,} seeds from train split")

    # Validate seed meta_str format upfront
    bad = [i for i, s in enumerate(seeds) if len(s[1].split("_")) < 5]
    if bad:
        raise ValueError(
            f"{len(bad)} seeds have malformed meta_str (need >=5 '_'-separated fields): "
            f"indices {bad[:5]}{'...' if len(bad) > 5 else ''}"
        )

    # Group seeds by (session, probe) -> needed t_windows
    probe_twins   = defaultdict(set)   # (session, probe) -> {t_win, ...}
    probe_seed_idx = defaultdict(list) # (session, probe) -> [seed indices]
    for i, s in enumerate(seeds):
        parts = s[1].split("_")
        key   = (parts[0], parts[2])
        probe_twins[key].add(parts[1])
        probe_seed_idx[key].append(i)

    print(f"  {len(probe_twins)} unique (session, probe) pairs")

    # Build flat channel list per probe_id
    # probe_samples: probe_id -> list of 5-tuples
    probe_samples = defaultdict(list)
    n_skipped = 0

    for pi, ((session, probe), twins) in enumerate(sorted(probe_twins.items())):
        pkl_path = os.path.join(raw_dir, f"{session}_{probe}.pickle")

        if (pi + 1) % 50 == 0 or pi == 0:
            n_total = sum(len(v) for v in probe_samples.values())
            print(f"  [{pi+1}/{len(probe_twins)}] {session}_{probe} "
                  f"({len(twins)} t-wins) | accumulated {n_total:,} channels")

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Raw pickle not found: {pkl_path}")

        pidx = index_probe(pkl_path, twins)

        for i in probe_seed_idx[(session, probe)]:
            s     = seeds[i]
            t_win = s[1].split("_")[1]
            flat  = collect_flat_samples(s, t_win, pidx, ch_lookup,
                                         max_channels_per_group, rng)
            if flat is None:
                n_skipped += 1
                continue
            probe_id = get_probe_id(s[1])
            probe_samples[probe_id].extend(flat)

        del pidx
        gc.collect()

    total_channels = sum(len(v) for v in probe_samples.values())
    print(f"  {total_channels:,} total channels across {len(probe_samples)} probes "
          f"({n_skipped} seeds skipped)")

    # Allocate target_samples across probes
    eligible, probe_ids, allocations = allocate_samples(
        probe_samples, target_samples, min_per_probe, rng)

    actual_total = sum(allocations.values())
    print(f"  Allocated {actual_total:,} samples across {len(allocations)} probes")

    # Subsample each probe to its allocation
    all_samples  = []
    probe_groups = {}  # probe_id -> [global_sample_indices]
    global_idx   = 0

    for pid in probe_ids:
        n_take = allocations[pid]
        pool   = eligible[pid]
        if n_take < len(pool):
            chosen = rng.choice(len(pool), size=n_take, replace=False)
            chosen.sort()
            samps = [pool[i] for i in chosen]
        else:
            samps = pool

        indices             = list(range(global_idx, global_idx + len(samps)))
        probe_groups[pid]   = indices
        all_samples.extend(samps)
        global_idx         += len(samps)

    print(f"  {len(all_samples):,} final samples, {len(probe_groups)} probes")

    # Stats
    per_probe_counts = [len(v) for v in probe_groups.values()]
    depths = np.array([float(s[3][1]) for s in all_samples])  # shank_pos[1] = vertical
    n_labeled = sum(1 for s in all_samples if s[2] != -1 and s[2] is not None)
    print(f"  per-probe: min={min(per_probe_counts)}, max={max(per_probe_counts)}, "
          f"mean={np.mean(per_probe_counts):.1f}")
    print(f"  depth range: {depths.min():.0f} – {depths.max():.0f} µm")
    print(f"  labeled (seed channels): {n_labeled:,} / {len(all_samples):,}")

    # Save samples
    SAMPLES_PER_FILE = 5000
    file_list, cumulative = [], [0]
    running = n_files = 0

    for start in range(0, len(all_samples), SAMPLES_PER_FILE):
        chunk = all_samples[start:start + SAMPLES_PER_FILE]
        fname = f"samples_{n_files:06d}.pkl"
        fpath = os.path.join(dst, fname)
        with open(fpath, "wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        sz_mb   = os.path.getsize(fpath) / 1e6
        running += len(chunk)
        cumulative.append(running)
        file_list.append(os.path.join(dst_dir, split, fname))
        n_files += 1
        print(f"  saved {fname}: {len(chunk)} samples ({sz_mb:.1f} MB)")

    # Save compact_index.pkl
    idx = {
        "n_samples":         len(all_samples),
        "n_files":           n_files,
        "samples_per_file":  SAMPLES_PER_FILE,
        "file_list":         file_list,
        "cumulative_counts": cumulative,
        "split_name":        split,
        "label_type":        "atlas_hierarchy",
        "has_coordinates":   True,
        "coordinate_system": "probe_shank+CCF",
        "coordinate_units":  "micrometers",
        "item_format": (
            "5-tuple: (signal[3750], meta_str, label_dict_or_-1, shank_pos[2], ccf_coords[3]). "
            "shank_pos[0] = probe_horizontal_position in µm. "
            "shank_pos[1] = probe_vertical_position in µm. "
            "2D on-shank distance = sqrt((h1-h2)^2 + (v1-v2)^2). "
            "ccf_coords = (ap, dv, lr) in µm from Allen CCF. "
            "label is -1 for sibling channels (no atlas label). "
            "Use probe_groups.pkl for same-probe batching."
        ),
    }
    with open(os.path.join(dst, "compact_index.pkl"), "wb") as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save probe_groups.pkl
    with open(os.path.join(dst, "probe_groups.pkl"), "wb") as f:
        pickle.dump(probe_groups, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved probe_groups.pkl ({len(probe_groups)} probes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build compact_atlas_{scale}_probe_dist_v2 from any compact_atlas."
    )
    parser.add_argument("--src-dir",   required=True,
                        help="Source compact_atlas directory (e.g. blind_localization/data/atlas_data/compact_atlas_25k)")
    parser.add_argument("--dst-dir",   required=True,
                        help="Output directory (e.g. blind_localization/data/atlas_data/compact_atlas_25k_probe_dist_v2)")
    parser.add_argument("--target-samples", type=int, required=True,
                        help="Target samples per split (e.g. 25000 or 75000)")
    parser.add_argument("--raw-dir",   default="data/Allen_w2v2/Allen",
                        help="Directory of raw Allen probe pickles")
    parser.add_argument("--channels-csv", default="data/channels.csv",
                        help="Allen channels.csv with CCF coordinates and probe_vertical_position")
    parser.add_argument("--min-per-probe", type=int, default=32,
                        help="Minimum samples per probe (default: 32)")
    parser.add_argument("--max-channels-per-group", type=int, default=32,
                        help="Max sibling channels per time window (default: 32)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-sessions", default="",
                        help="Comma-separated session IDs to exclude (e.g. lp_eval val+test sessions). "
                             "If val/test seeds are fully excluded, falls back to train split.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    exclude_sessions = set(s.strip() for s in args.exclude_sessions.split(",") if s.strip())

    print(f"Source:               {args.src_dir}")
    print(f"Destination:          {args.dst_dir}")
    print(f"Target samples/split: {args.target_samples:,}")
    print(f"Min per probe:        {args.min_per_probe}")
    print(f"Max channels/group:   {args.max_channels_per_group}")
    if exclude_sessions:
        print(f"Excluded sessions:    {len(exclude_sessions)} sessions")
    print()

    print("Loading channel lookup...")
    ch_lookup = build_channel_lookup(args.channels_csv)
    print()

    os.makedirs(args.dst_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"\n{'='*60}")
        print(f"  {split}")
        print(f"{'='*60}")
        process_split(
            split        = split,
            src_dir      = args.src_dir,
            dst_dir      = args.dst_dir,
            raw_dir      = args.raw_dir,
            ch_lookup    = ch_lookup,
            target_samples         = args.target_samples,
            min_per_probe          = args.min_per_probe,
            max_channels_per_group = args.max_channels_per_group,
            rng          = rng,
            exclude_sessions       = exclude_sessions,
        )

    # Copy metadata from source (same sessions / class structure)
    for fname in ["metadata.pkl", "metadata_fixed_105.pkl"]:
        src_f = os.path.join(args.src_dir, fname)
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(args.dst_dir, fname))
            print(f"Copied {fname}")

    print("\nDone!")


if __name__ == "__main__":
    main()
