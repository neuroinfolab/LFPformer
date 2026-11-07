"""Build compact atlas-labeled datasets from raw session pickles.

Two-phase index-first approach:

Phase 1: --build-index
  Scan all raw session pickles and build a lightweight index (metadata only,
  no signal data copied). Records file path, sample offset, channel_id, and
  fine_id for each valid sample. Also builds channel_map (channel_id -> hierarchy
  + coords) and metadata (label_maps, class counts).

  Output: atlas_index/ (~500MB total)
    {split}.pkl      - per-split arrays (file_idx, sample_idx, fine_id, channel_id)
    channel_map.pkl  - channel_id -> {hierarchy, coords} for ~80K channels
    metadata.pkl     - label_maps, class counts by depth, region names

Phase 2: --from-index
  Read the index, optionally subsample (class-balanced via fine_id), then
  read raw files to extract signal data and write compact datasets with
  large chunk size (default 5000 samples/file).

  Output: e.g. compact_atlas_13k/ (~8 files total)

Usage (must run inside container):
  # Phase 1: Build index
  python3 script/build_atlas_dataset.py --build-index

  # Phase 2: Build 13K dataset from index
  python3 script/build_atlas_dataset.py --from-index \
    --atlas-index blind_localization/data/atlas_index \
    --output-dir blind_localization/data/compact_atlas_13k \
    --target-per-split 13000

  # Phase 2: Build 200K dataset with per-split targets
  python3 script/build_atlas_dataset.py --from-index \
    --atlas-index blind_localization/data/atlas_index \
    --output-dir blind_localization/data/compact_atlas_200k \
    --per-split-targets 130000 30000 40000
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from glob import glob

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared helpers (used by both phases)
# ---------------------------------------------------------------------------

def load_atlas(atlas_path, structure_tree_path):
    """Load atlas and build hierarchy lookup via iblatlas — unified with probe index."""
    from iblatlas.atlas import AllenAtlas
    import numpy as np

    print("Loading AllenAtlas (iblatlas)...")
    ba = AllenAtlas(res_um=25)
    anno = ba.label
    print("  Shape: {}, brain voxels: {}".format(anno.shape, (anno > 0).sum()))

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

    # Build hierarchy for ALL structures in ba.regions (not just annotation volume)
    print("Building hierarchy lookup from all {} structures...".format(len(ba.regions.id)))
    anno_id_to_hierarchy = {}
    for sid in ba.regions.id:
        if sid == 0:
            continue
        h = get_hierarchy(int(sid))
        if h is not None:
            anno_id_to_hierarchy[int(sid)] = h
    print("  Mapped {} structure IDs".format(len(anno_id_to_hierarchy)))

    acr_to_name = {ba.regions.acronym[i]: ba.regions.name[i]
                   for i in range(len(ba.regions.id))}

    return anno, anno_id_to_hierarchy, acr_to_name


def build_filtered_channel_map(channels_csv, anno, anno_id_to_hierarchy):
    """Build channel_id -> (coords, hierarchy) using iblatlas for lookups."""
    from iblatlas.atlas import AllenAtlas
    import numpy as np

    print("\nLoading channels.csv...")
    df = pd.read_csv(channels_csv)
    df = df.dropna(subset=['anterior_posterior_ccf_coordinate',
                           'dorsal_ventral_ccf_coordinate',
                           'left_right_ccf_coordinate'])

    ba = AllenAtlas(res_um=25)

    # Load mlapdv from npz for proper iblatlas lookups
    npz_path = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/probe_channels.npz')
    npz = np.load(npz_path, allow_pickle=True)
    probe_npz = {}
    for key in npz.files:
        pid, field = key.split('__', 1)
        if pid not in probe_npz:
            probe_npz[pid] = {}
        probe_npz[pid][field] = npz[key]

    # Build pid_short → full pid lookup
    pid_short_to_full = {pid[:8]: pid for pid in probe_npz}

    channel_map = {}
    n_outside   = 0
    n_oob       = 0

    # Group channels by probe (pid_short) for batch get_labels
    df['pid_short'] = df['id'].str[:8]
    for pid_short, group in df.groupby('pid_short'):
        full_pid = pid_short_to_full.get(pid_short)
        if full_pid is None or full_pid not in probe_npz:
            n_outside += len(group)
            continue

        mlapdv = probe_npz[full_pid]['mlapdv'].astype('float64') / 1e6
        sids   = ba.get_labels(mlapdv, mode='clip')

        for _, row in group.iterrows():
            cid     = str(row['id'])
            ch_idx  = int(row['channel_idx'])
            ap      = float(row['anterior_posterior_ccf_coordinate'])
            dv      = float(row['dorsal_ventral_ccf_coordinate'])
            lr      = float(row['left_right_ccf_coordinate'])

            sid = int(sids[ch_idx])
            if sid == 0 or sid not in anno_id_to_hierarchy:
                n_outside += 1
                continue

            channel_map[cid] = {
                'coords':    (ap, dv, lr),
                'hierarchy': anno_id_to_hierarchy[sid],
            }

    print("  Total with coords:       {}".format(len(df)))
    print("  Filtered (inside brain): {}".format(len(channel_map)))
    print("  Out of bounds:           {}".format(n_oob))
    print("  Outside brain mask:      {}".format(n_outside))

    return channel_map


def load_split_sessions(index_dir):
    """Load session-to-split mapping from balanced index."""
    split_sessions = {}
    for split in ['train', 'val', 'test']:
        idx_path = os.path.join(index_dir, 'balanced_index_{}.pkl'.format(split))
        with open(idx_path, 'rb') as f:
            idx = pickle.load(f)
        sessions = set()
        for p in idx['session_path_list']:
            fname = os.path.basename(p).replace('.pickle', '')
            sessions.add(fname)
        split_sessions[split] = sessions
        print("  {}: {} session files".format(split, len(sessions)))

    overlap = split_sessions['val'] & split_sessions['test']
    if overlap:
        print("  Val/Test overlap: {} files (assigned to both)".format(len(overlap)))

    return split_sessions


# ---------------------------------------------------------------------------
# Phase 1: Build index
# ---------------------------------------------------------------------------

def build_atlas_index(raw_dir, split_sessions, channel_map, output_dir):
    """Scan raw files and build lightweight per-split index arrays.

    For each valid sample, records only metadata (no signal copied):
      - file_idx: index into raw_files list
      - sample_idx: offset within that raw file
      - channel_id: for linking to channel_map
      - fine_id: annotation structure ID (for class-balanced subsampling)

    Also stores per-split depth/class counts for reporting.
    """
    all_files = sorted(glob(os.path.join(raw_dir, '*.pickle')))
    print("\nScanning {} raw session files for index...".format(len(all_files)))

    # Map filename -> split(s)
    file_to_splits = {}
    for f in all_files:
        fname = os.path.basename(f).replace('.pickle', '')
        splits = []
        for split in ['train', 'val', 'test']:
            if fname in split_sessions[split]:
                splits.append(split)
        if splits:
            file_to_splits[f] = splits

    print("  Files with split assignment: {}".format(len(file_to_splits)))
    print("  Files without assignment: {}".format(len(all_files) - len(file_to_splits)))

    # Build ordered list of raw files and file_idx mapping
    raw_files = sorted(file_to_splits.keys())
    file_path_to_idx = {f: i for i, f in enumerate(raw_files)}

    # Per-split accumulators (lists, converted to arrays at end)
    split_data = {}
    for split in ['train', 'val', 'test']:
        split_data[split] = {
            'file_idx': [],
            'sample_idx': [],
            'channel_id': [],
            'fine_id': [],
            'depth_counts': defaultdict(Counter),
            'fine_counts': Counter(),
        }

    stats = {'processed_files': 0, 'total_raw_samples': 0,
             'matched': 0, 'no_channel': 0, 'no_coords': 0}

    t0 = time.time()
    for i, fpath in enumerate(raw_files):
        splits = file_to_splits[fpath]
        fidx = file_path_to_idx[fpath]

        if i % 20 == 0:
            elapsed = time.time() - t0
            print("  [{}/{}] Scanning {} ({:.0f}s elapsed)...".format(
                i + 1, len(raw_files), os.path.basename(fpath), elapsed))
            sys.stdout.flush()

        try:
            with open(fpath, 'rb') as f:
                raw_samples = pickle.load(f)
        except Exception as e:
            print("    ERROR loading {}: {}".format(fpath, e))
            continue

        stats['processed_files'] += 1
        stats['total_raw_samples'] += len(raw_samples)

        for sample_idx, sample in enumerate(raw_samples):
            meta = sample[1]

            # Parse channel_id from meta_str
            parts = meta.split('_')
            if len(parts) < 4:
                stats['no_channel'] += 1
                continue

            try:
                channel_id = str(parts[3]) # change from int to str to adapt IBL
            except ValueError:
                stats['no_channel'] += 1
                continue

            if channel_id not in channel_map:
                stats['no_coords'] += 1
                continue

            info = channel_map[channel_id]
            fine_id = info['hierarchy']['fine_id']
            stats['matched'] += 1

            for split in splits:
                sd = split_data[split]
                sd['file_idx'].append(fidx)
                sd['sample_idx'].append(sample_idx)
                sd['channel_id'].append(channel_id)
                sd['fine_id'].append(fine_id)

                # Track stats
                hier = info['hierarchy']
                sd['fine_counts'][fine_id] += 1
                for d in hier:
                    if isinstance(d, int):
                        sd['depth_counts'][d][hier[d]] += 1

        del raw_samples
        gc.collect()

    # Write index
    os.makedirs(output_dir, exist_ok=True)

    print("\nExtraction stats:")
    print("  Files processed: {}".format(stats['processed_files']))
    print("  Total raw samples: {:,}".format(stats['total_raw_samples']))
    print("  Matched (brain-filtered): {:,}".format(stats['matched']))
    print("  No channel ID: {:,}".format(stats['no_channel']))
    print("  Channel outside brain: {:,}".format(stats['no_coords']))

    # Save raw_files list
    with open(os.path.join(output_dir, 'raw_files.pkl'), 'wb') as f:
        pickle.dump(raw_files, f)

    for split in ['train', 'val', 'test']:
        sd = split_data[split]
        n = len(sd['file_idx'])

        split_index = {
            'file_idx': np.array(sd['file_idx'], dtype=np.uint16),
            'sample_idx': np.array(sd['sample_idx'], dtype=np.uint32),
            # 'channel_id': np.array(sd['channel_id'], dtype=np.int32),
            'channel_id': np.array(sd['channel_id'], dtype = object),
            'fine_id': np.array(sd['fine_id'], dtype=np.int32),
            'n_samples': n,
            'depth_counts': {d: dict(c) for d, c in sd['depth_counts'].items()},
            'fine_counts': dict(sd['fine_counts']),
        }

        out_path = os.path.join(output_dir, '{}.pkl'.format(split))
        with open(out_path, 'wb') as f:
            pickle.dump(split_index, f)

        print("  {}: {:,} samples".format(split, n))

    print("  Raw files indexed: {}".format(len(raw_files)))
    return raw_files, split_data


def build_index_metadata(split_data, channel_map, output_dir, acr_to_name):
    """Write channel_map.pkl and metadata.pkl from index scan results."""
    # Save channel_map
    with open(os.path.join(output_dir, 'channel_map.pkl'), 'wb') as f:
        pickle.dump(channel_map, f)
    print("  Saved channel_map ({} channels)".format(len(channel_map)))

    # Aggregate stats across splits for metadata
    depth_regions_all = defaultdict(Counter)
    fine_ids_all = Counter()

    for split in ['train', 'val', 'test']:
        sd = split_data[split]
        for d, counts in sd['depth_counts'].items():
            depth_regions_all[d].update(counts)
        fine_ids_all.update(sd['fine_counts'])

    # Build label maps (sorted by frequency)
    label_maps = {}
    for d in depth_regions_all:
        sorted_regions = sorted(depth_regions_all[d].items(), key=lambda x: -x[1])
        label_maps[d] = {acr: i for i, (acr, _) in enumerate(sorted_regions)}

    sorted_fine = sorted(fine_ids_all.items(), key=lambda x: -x[1])
    fine_id_map = {sid: i for i, (sid, _) in enumerate(sorted_fine)}

    metadata = {
        'label_type': 'allen_atlas_hierarchical',
        'label_maps': label_maps,
        'fine_id_map': fine_id_map,
        'num_classes_by_depth': {d: len(label_maps[d]) for d in label_maps},
        'region_counts_by_depth': {d: dict(depth_regions_all[d]) for d in depth_regions_all},
        'fine_id_counts': dict(fine_ids_all),
        'atlas_file': 'data/allen_ccf/annotation_25.nrrd',
        'structure_tree': 'data/allen_ccf/structure_tree.json',
        'region_names': acr_to_name,
        'sample_format': '(signal, meta_str, atlas_hierarchy_dict, coords)',
        'atlas_hierarchy_format': '{depth_int: acronym_str, ..., fine_id: int, fine_acr: str, max_depth: int}',
    }
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print("\n  Classes by depth:")
    for d in sorted(label_maps.keys()):
        print("    depth {}: {} classes".format(d, len(label_maps[d])))
    print("  Fine-grained: {} unique structure IDs".format(len(fine_id_map)))


def check_class_coverage(index_dir, channel_map, target_depth=6):
    """Check class coverage across splits at ALL depths, with detail at target_depth.

    Prints a full depth-wise table showing #classes per split and missing classes.
    Returns coverage info at target_depth for session reassignment.
    """
    # Load all split indices
    split_indices = {}
    for split in ['train', 'val', 'test']:
        with open(os.path.join(index_dir, '{}.pkl'.format(split)), 'rb') as f:
            split_indices[split] = pickle.load(f)

    # Find all depths available
    all_depths = set()
    for split in ['train', 'val', 'test']:
        all_depths.update(split_indices[split].get('depth_counts', {}).keys())
    all_depths = sorted(all_depths)

    # Print full depth-wise report
    print("\n" + "=" * 80)
    print("CLASS COVERAGE BY DEPTH")
    print("=" * 80)
    print("{:>5}  {:>8} {:>8} {:>8}  {:>10} {:>10} {:>10}  {}".format(
        "Depth", "Train#", "Val#", "Test#", "TrainSamp", "ValSamp", "TestSamp", "Missing from val/test"))
    print("-" * 80)

    target_split_classes = {}
    target_split_counts = {}
    target_missing = set()

    for d in all_depths:
        row_classes = {}
        row_counts = {}
        for split in ['train', 'val', 'test']:
            dc = split_indices[split].get('depth_counts', {})
            if d in dc:
                counts = Counter(dc[d])
            else:
                counts = Counter()
            row_classes[split] = set(counts.keys())
            row_counts[split] = counts

        missing_val = row_classes['train'] - row_classes['val']
        missing_test = row_classes['train'] - row_classes['test']
        all_missing = missing_val | missing_test

        missing_str = ""
        if all_missing:
            if len(all_missing) <= 5:
                missing_str = ", ".join(sorted(all_missing))
            else:
                missing_str = "{} classes missing".format(len(all_missing))

        marker = " <--" if d == target_depth else ""
        print("{:>5}  {:>8} {:>8} {:>8}  {:>10,} {:>10,} {:>10,}  {}{}".format(
            d,
            len(row_classes['train']), len(row_classes['val']), len(row_classes['test']),
            sum(row_counts['train'].values()),
            sum(row_counts['val'].values()),
            sum(row_counts['test'].values()),
            missing_str, marker))

        if d == target_depth:
            target_split_classes = row_classes
            target_split_counts = row_counts
            target_missing = all_missing

    print("-" * 80)

    # Detailed report at target depth
    print("\nDetailed coverage at depth {} ({} classes in train):".format(
        target_depth, len(target_split_classes.get('train', set()))))

    if target_split_counts:
        # Show per-class counts sorted by train count
        train_counts = target_split_counts.get('train', Counter())
        val_counts = target_split_counts.get('val', Counter())
        test_counts = target_split_counts.get('test', Counter())
        all_classes = sorted(train_counts.keys(), key=lambda x: -train_counts[x])

        print("{:>20}  {:>10} {:>10} {:>10}  {}".format(
            "Class", "Train", "Val", "Test", "Status"))
        print("-" * 65)
        for cls in all_classes:
            tc = train_counts.get(cls, 0)
            vc = val_counts.get(cls, 0)
            xc = test_counts.get(cls, 0)
            status = ""
            if vc == 0 and xc == 0:
                status = "MISSING both"
            elif vc == 0:
                status = "MISSING val"
            elif xc == 0:
                status = "MISSING test"
            print("{:>20}  {:>10,} {:>10,} {:>10,}  {}".format(cls, tc, vc, xc, status))

    if target_missing:
        print("\n  {} classes need session reassignment at depth {}".format(
            len(target_missing), target_depth))
    else:
        print("\n  All classes covered in all splits at depth {}".format(target_depth))

    return target_split_classes, target_split_counts, target_missing


def reassign_sessions_for_coverage(index_dir, raw_files, channel_map, missing_classes,
                                   target_depth=6):
    """Reassign minimum sessions from train to val/test to cover missing classes.

    For each missing class, find train sessions containing that class and move
    some to val/test. Prefer sessions covering multiple missing classes.
    """
    if not missing_classes:
        print("\nNo missing classes — skipping session reassignment.")
        return

    print("\nReassigning sessions to cover {} missing classes: {}".format(
        len(missing_classes), sorted(missing_classes)))

    # Load all three split indices
    splits = {}
    for split in ['train', 'val', 'test']:
        with open(os.path.join(index_dir, '{}.pkl'.format(split)), 'rb') as f:
            splits[split] = pickle.load(f)

    # For each raw file in train, find which missing classes it contributes
    # Build: file_idx -> set of classes at target_depth
    file_to_classes = defaultdict(set)
    train_idx = splits['train']
    for i in range(train_idx['n_samples']):
        fidx = int(train_idx['file_idx'][i])
        cid = str(train_idx['channel_id'][i])
        if cid in channel_map:
            hier = channel_map[cid]['hierarchy']
            if target_depth in hier:
                cls = hier[target_depth]
                if cls in missing_classes:
                    file_to_classes[fidx].add(cls)

    if not file_to_classes:
        print("  ERROR: No train files contain missing classes!")
        return

    # Greedy set cover: pick files covering most missing classes
    remaining = set(missing_classes)
    reassigned_files = []
    while remaining and file_to_classes:
        # Pick file covering most remaining classes
        best_fidx = max(file_to_classes.keys(),
                        key=lambda f: len(file_to_classes[f] & remaining))
        covered = file_to_classes[best_fidx] & remaining
        if not covered:
            break
        reassigned_files.append(best_fidx)
        remaining -= covered
        del file_to_classes[best_fidx]
        print("  Reassigning file {} ({}) covers: {}".format(
            best_fidx, os.path.basename(raw_files[best_fidx]), sorted(covered)))

    if remaining:
        print("  WARNING: Could not cover {} classes: {}".format(
            len(remaining), sorted(remaining)))

    # Move samples from train to val+test (alternate)
    reassigned_set = set(reassigned_files)

    # Collect samples to move
    move_mask = np.zeros(train_idx['n_samples'], dtype=bool)
    for i in range(train_idx['n_samples']):
        if int(train_idx['file_idx'][i]) in reassigned_set:
            move_mask[i] = True

    n_moving = move_mask.sum()
    print("  Moving {:,} samples from {} files".format(n_moving, len(reassigned_files)))

    # Split moved samples alternating between val and test
    move_indices = np.where(move_mask)[0]
    val_add = move_indices[::2]   # even indices -> val
    test_add = move_indices[1::2]  # odd indices -> test

    # Build new indices
    keep_mask = ~move_mask

    for split, add_indices in [('val', val_add), ('test', test_add)]:
        old = splits[split]
        new_file_idx = np.concatenate([old['file_idx'], train_idx['file_idx'][add_indices]])
        new_sample_idx = np.concatenate([old['sample_idx'], train_idx['sample_idx'][add_indices]])
        new_channel_id = np.concatenate([old['channel_id'], train_idx['channel_id'][add_indices]])
        new_fine_id = np.concatenate([old['fine_id'], train_idx['fine_id'][add_indices]])

        # Recompute depth_counts and fine_counts
        depth_counts = defaultdict(Counter)
        fine_counts = Counter()
        for j in range(len(new_fine_id)):
            cid = str(new_channel_id[j])
            if cid in channel_map:
                hier = channel_map[cid]['hierarchy']
                fid = hier['fine_id']
                fine_counts[fid] += 1
                for d in hier:
                    if isinstance(d, int):
                        depth_counts[d][hier[d]] += 1

        new_idx = {
            'file_idx': new_file_idx.astype(np.uint16),
            'sample_idx': new_sample_idx.astype(np.uint32),
            # 'channel_id': new_channel_id.astype(np.int32),
            'channel_id': new_channel_id.astype(object),
            'fine_id': new_fine_id.astype(np.int32),
            'n_samples': len(new_file_idx),
            'depth_counts': {d: dict(c) for d, c in depth_counts.items()},
            'fine_counts': dict(fine_counts),
        }
        with open(os.path.join(index_dir, '{}.pkl'.format(split)), 'wb') as f:
            pickle.dump(new_idx, f)
        print("  {}: {:,} -> {:,} samples (+{:,})".format(
            split, old['n_samples'], new_idx['n_samples'], len(add_indices)))

    # Update train (remove reassigned samples)
    depth_counts_train = defaultdict(Counter)
    fine_counts_train = Counter()
    kept_channel_ids = train_idx['channel_id'][keep_mask]
    for cid in kept_channel_ids:
        cid = str(cid)
        if cid in channel_map:
            hier = channel_map[cid]['hierarchy']
            fid = hier['fine_id']
            fine_counts_train[fid] += 1
            for d in hier:
                if isinstance(d, int):
                    depth_counts_train[d][hier[d]] += 1

    new_train = {
        'file_idx': train_idx['file_idx'][keep_mask].astype(np.uint16),
        'sample_idx': train_idx['sample_idx'][keep_mask].astype(np.uint32),
        # 'channel_id': train_idx['channel_id'][keep_mask].astype(np.int32),
        'channel_id': new_channel_id.astype(object),
        'fine_id': train_idx['fine_id'][keep_mask].astype(np.int32),
        'n_samples': int(keep_mask.sum()),
        'depth_counts': {d: dict(c) for d, c in depth_counts_train.items()},
        'fine_counts': dict(fine_counts_train),
    }
    with open(os.path.join(index_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(new_train, f)
    print("  train: {:,} -> {:,} samples (-{:,})".format(
        train_idx['n_samples'], new_train['n_samples'], n_moving))

    # Update metadata
    with open(os.path.join(index_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    # Recompute label_maps from updated splits
    depth_regions_all = defaultdict(Counter)
    fine_ids_all = Counter()
    for split in ['train', 'val', 'test']:
        with open(os.path.join(index_dir, '{}.pkl'.format(split)), 'rb') as f:
            idx = pickle.load(f)
        for d, counts in idx['depth_counts'].items():
            depth_regions_all[d].update(counts)
        fine_ids_all.update(idx['fine_counts'])

    label_maps = {}
    for d in depth_regions_all:
        sorted_regions = sorted(depth_regions_all[d].items(), key=lambda x: -x[1])
        label_maps[d] = {acr: i for i, (acr, _) in enumerate(sorted_regions)}

    metadata['label_maps'] = label_maps
    metadata['num_classes_by_depth'] = {d: len(label_maps[d]) for d in label_maps}
    metadata['region_counts_by_depth'] = {d: dict(depth_regions_all[d]) for d in depth_regions_all}
    metadata['fine_id_counts'] = dict(fine_ids_all)

    with open(os.path.join(index_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print("  Metadata updated.")


# ---------------------------------------------------------------------------
# Phase 2: Build compact dataset from index
# ---------------------------------------------------------------------------

def build_from_index(index_dir, output_dir, target_per_split=0,
                     per_split_targets=None, chunk_size=5000, seed=42,
                     min_per_class=100, balanced_uniform=False, per_class=0,
                     evaluable_set=None, balance_splits=('train',),
                     only_splits=None, per_region_targets=None):
    """Build compact dataset by reading index then extracting signals from raw files.

    Steps:
    1. Load index for each split
    2. Subsample if target specified (class-balanced: min_per_class guaranteed, rest proportional)
    3. Group selected samples by raw file for sequential I/O
    4. Read each raw file once, extract needed samples
    5. Attach hierarchy + coords from channel_map
    6. Write to chunk files with large chunk_size

    Balanced-uniform mode (balanced_uniform=True): for splits in `balance_splits`,
    select EXACTLY `per_class` samples (uniform) from each fine_id in `evaluable_set`
    (fids with fewer than per_class available contribute all they have; fids absent
    contribute 0). This fills under-sampled evaluable regions instead of the natural
    head-heavy proportional fill. `only_splits` (e.g. {'train'}) restricts which
    splits are built at all.
    """
    print("\n=== Building compact dataset from index ===")
    print("  Index: {}".format(index_dir))
    print("  Output: {}".format(output_dir))
    print("  Chunk size: {}".format(chunk_size))
    if balanced_uniform:
        print("  BALANCED-UNIFORM mode: per_class={}, evaluable_set={} fids, "
              "balance_splits={}".format(per_class,
              len(evaluable_set) if evaluable_set else 'ALL', balance_splits))

    # Load channel_map and raw_files list
    with open(os.path.join(index_dir, 'channel_map.pkl'), 'rb') as f:
        channel_map = pickle.load(f)
    print("  Loaded channel_map ({} channels)".format(len(channel_map)))

    with open(os.path.join(index_dir, 'raw_files.pkl'), 'rb') as f:
        raw_files = pickle.load(f)
    print("  Raw files: {}".format(len(raw_files)))

    with open(os.path.join(index_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    rng = np.random.RandomState(seed)

    # Track stats for metadata
    depth_regions_all = defaultdict(Counter)
    fine_ids_all = Counter()

    for split in ['train', 'val', 'test']:
        if only_splits is not None and split not in only_splits:
            print("\n  {}: skipped (not in only_splits={})".format(split, only_splits))
            continue
        t_split = time.time()

        # Determine target for this split
        if per_split_targets and split in per_split_targets:
            target = per_split_targets[split]
        elif target_per_split > 0:
            target = target_per_split
        else:
            target = 0  # keep all

        # Load split index
        with open(os.path.join(index_dir, '{}.pkl'.format(split)), 'rb') as f:
            idx = pickle.load(f)

        n_total = idx['n_samples']
        file_idx_arr = idx['file_idx']
        sample_idx_arr = idx['sample_idx']
        channel_id_arr = idx['channel_id']

        print("\n  {}: {:,} total samples".format(split, n_total))

        # Balanced selection: per-region target (rare-boost max(natural,F)) OR
        # uniform per_class. per_region_targets={fid: count} takes precedence.
        if balanced_uniform and split in balance_splits:
            fine_id_arr = idx['fine_id']
            if per_region_targets:
                target_fids = sorted(per_region_targets.keys())
            else:
                target_fids = (sorted(evaluable_set) if evaluable_set is not None
                               else sorted(np.unique(fine_id_arr).tolist()))
            selected_list = []
            n_filled = n_short = n_absent = 0
            for fid in target_fids:
                ci = np.where(fine_id_arr == fid)[0]
                if len(ci) == 0:
                    n_absent += 1
                    continue
                tgt = int(per_region_targets[fid]) if per_region_targets else per_class
                take = min(tgt, len(ci))
                if take < tgt:
                    n_short += 1
                else:
                    n_filled += 1
                selected_list.append(rng.choice(ci, take, replace=False))
            selected = np.unique(np.concatenate(selected_list))
            sel_fids = fine_id_arr[selected]
            uniq, cnts = np.unique(sel_fids, return_counts=True)
            mode = "PER-REGION-TARGET" if per_region_targets else "UNIFORM per_class={}".format(per_class)
            print("    BALANCED [{}]: {:,} samples across {} fids "
                  "({} filled, {} short, {} absent)".format(
                      mode, len(selected), len(uniq), n_filled, n_short, n_absent))
            print("    Per-region: min={}, max={}, median={}".format(
                  cnts.min(), cnts.max(), int(np.median(cnts))))
        # Subsample if needed (class-balanced by fine_id)
        elif target > 0 and target < n_total:
            fine_id_arr = idx['fine_id']
            unique_fids = np.unique(fine_id_arr)
            n_classes = len(unique_fids)

            # Group sample indices by fine_id
            class_indices = {}
            for fid in unique_fids:
                class_indices[fid] = np.where(fine_id_arr == fid)[0]

            # Phase 1: guarantee min_per_class from each class
            reserved = min_per_class * n_classes
            if reserved > target:
                # Can't guarantee min_per_class for all; do equal split
                per_class = max(1, target // n_classes)
                selected_list = []
                for fid in unique_fids:
                    ci = class_indices[fid]
                    take = min(per_class, len(ci))
                    selected_list.append(rng.choice(ci, take, replace=False))
                selected = np.concatenate(selected_list)
                print("    Budget too small for min_per_class={}, using {} per class".format(
                    min_per_class, per_class))
            else:
                selected_list = []
                used = set()
                for fid in unique_fids:
                    ci = class_indices[fid]
                    take = min(min_per_class, len(ci))
                    chosen = rng.choice(ci, take, replace=False)
                    selected_list.append(chosen)
                    used.update(chosen.tolist())

                # Phase 2: fill remaining budget proportionally
                remaining = target - len(used)
                if remaining > 0:
                    # Pool of unused indices
                    all_unused = np.array([i for i in range(n_total) if i not in used])
                    if len(all_unused) <= remaining:
                        selected_list.append(all_unused)
                    else:
                        extra = rng.choice(all_unused, remaining, replace=False)
                        selected_list.append(extra)

                selected = np.concatenate(selected_list)

            selected = np.unique(selected)  # deduplicate and sort
            print("    Class-balanced subsample: {:,} samples, {} classes, min_per_class={}".format(
                len(selected), n_classes, min_per_class))

            # Report per-class counts
            sel_fids = fine_id_arr[selected]
            counts = np.bincount(sel_fids, minlength=fine_id_arr.max() + 1)
            nonzero = counts[counts > 0]
            print("    Per-class: min={}, max={}, median={}, classes_with_samples={}".format(
                nonzero.min(), nonzero.max(), int(np.median(nonzero)), len(nonzero)))
        else:
            selected = np.arange(n_total)
            if target > 0:
                print("    Target {:,} >= total {:,}, keeping all".format(target, n_total))

        # Group by raw file for sequential I/O
        # file_idx -> list of (selected_pos, sample_idx_in_raw_file, channel_id)
        file_groups = defaultdict(list)
        for pos in selected:
            fidx = int(file_idx_arr[pos])
            sidx = int(sample_idx_arr[pos])
            cid = str(channel_id_arr[pos])
            file_groups[fidx].append((sidx, cid))

        n_raw_files_needed = len(file_groups)
        print("    Need to read {} raw files".format(n_raw_files_needed))

        # Prepare output
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        buffer = []
        n_out_files = 0
        n_collected = 0
        cumulative = [0]
        file_list = []
        split_depth_counts = defaultdict(Counter)
        split_fine_counts = Counter()

        def flush_buf():
            nonlocal n_out_files, n_collected
            if not buffer:
                return
            out_path = os.path.join(split_dir, 'samples_{:06d}.pkl'.format(n_out_files))
            with open(out_path, 'wb') as f:
                pickle.dump(buffer, f)
            cumulative.append(cumulative[-1] + len(buffer))
            file_list.append(os.path.join(output_dir, split,
                                          'samples_{:06d}.pkl'.format(n_out_files)))
            n_collected += len(buffer)
            n_out_files += 1
            buffer.clear()

        # Process raw files in order
        files_read = 0
        for fidx in sorted(file_groups.keys()):
            fpath = raw_files[fidx]
            samples_needed = file_groups[fidx]
            # Sort by sample_idx for sequential access
            samples_needed.sort(key=lambda x: x[0])
            needed_set = {sidx: cid for sidx, cid in samples_needed}

            try:
                with open(fpath, 'rb') as f:
                    raw_samples = pickle.load(f)
            except Exception as e:
                print("    ERROR loading {}: {}".format(fpath, e))
                continue

            files_read += 1

            for sidx, cid in samples_needed:
                if sidx >= len(raw_samples):
                    continue

                sample = raw_samples[sidx]
                signal, meta = sample[0], sample[1]

                if cid not in channel_map:
                    continue

                info = channel_map[cid]
                hier = info['hierarchy']
                coords = info['coords']
                new_sample = (signal, meta, hier, coords)

                buffer.append(new_sample)

                # Track stats
                fine_id = hier['fine_id']
                split_fine_counts[fine_id] += 1
                fine_ids_all[fine_id] += 1
                for d in hier:
                    if isinstance(d, int):
                        split_depth_counts[d][hier[d]] += 1
                        depth_regions_all[d][hier[d]] += 1

                if len(buffer) >= chunk_size:
                    flush_buf()

            del raw_samples
            gc.collect()

            if files_read % 20 == 0:
                print("    ... read {}/{} files, collected {:,} samples".format(
                    files_read, n_raw_files_needed, n_collected + len(buffer)))
                sys.stdout.flush()

        flush_buf()

        # Write compact_index.pkl
        compact_index = {
            'n_samples': n_collected,
            'n_files': n_out_files,
            'samples_per_file': chunk_size,
            'file_list': file_list,
            'cumulative_counts': cumulative,
            'split_name': split,
            'label_type': 'allen_atlas_hierarchical',
            'num_classes_by_depth': {d: len(split_depth_counts[d])
                                     for d in split_depth_counts},
            'has_coordinates': True,
            'coordinate_system': 'allen_ccf_v3',
            'coordinate_units': 'micrometers',
        }
        with open(os.path.join(split_dir, 'compact_index.pkl'), 'wb') as f:
            pickle.dump(compact_index, f)

        d6 = split_depth_counts.get(6, {})
        top5 = sorted(d6.items(), key=lambda x: -x[1])[:5]
        elapsed = time.time() - t_split
        print("    {:,} samples in {} files ({:.0f}s, read {} raw files)".format(
            n_collected, n_out_files, elapsed, files_read))
        print("    depth-6 top: {}".format(top5))

        gc.collect()

    # Write global metadata for this dataset
    label_maps = {}
    for d in depth_regions_all:
        sorted_regions = sorted(depth_regions_all[d].items(), key=lambda x: -x[1])
        label_maps[d] = {acr: i for i, (acr, _) in enumerate(sorted_regions)}

    sorted_fine = sorted(fine_ids_all.items(), key=lambda x: -x[1])
    fine_id_map = {sid: i for i, (sid, _) in enumerate(sorted_fine)}

    # Build global cumulative effective label maps per depth.
    # For depth D: union of all region names from label_maps[0..D], sorted, contiguous 0..N-1.
    # This ensures train/val/test splits share the same label ID assignments.
    max_depth = max(label_maps.keys()) if label_maps else 0
    effective_label_maps_by_depth = {}
    for target_depth in range(max_depth + 1):
        cumulative = {}
        idx = 0
        for d in range(target_depth + 1):
            for name in sorted(label_maps.get(d, {}).keys()):
                if name not in cumulative:
                    cumulative[name] = idx
                    idx += 1
        effective_label_maps_by_depth[target_depth] = cumulative

    print("\n  Effective (cumulative) classes by depth:")
    for d in sorted(effective_label_maps_by_depth.keys()):
        print("    depth {}: {} classes".format(d, len(effective_label_maps_by_depth[d])))

    out_metadata = {
        'label_type': 'allen_atlas_hierarchical',
        'label_maps': label_maps,
        'fine_id_map': fine_id_map,
        'effective_label_maps_by_depth': effective_label_maps_by_depth,
        'num_classes_by_depth': {d: len(label_maps[d]) for d in label_maps},
        'region_counts_by_depth': {d: dict(depth_regions_all[d]) for d in depth_regions_all},
        'fine_id_counts': dict(fine_ids_all),
        'atlas_file': 'data/allen_ccf/annotation_25.nrrd',
        'structure_tree': 'data/allen_ccf/structure_tree.json',
        'region_names': metadata.get('region_names', {}),
        'sample_format': '(signal, meta_str, atlas_hierarchy_dict, coords)',
        'atlas_hierarchy_format': '{depth_int: acronym_str, ..., fine_id: int, fine_acr: str, max_depth: int}',
        'source_index': index_dir,
    }
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(out_metadata, f)

    print("\n  Output classes by depth:")
    for d in sorted(label_maps.keys()):
        print("    depth {}: {} classes".format(d, len(label_maps[d])))


# ---------------------------------------------------------------------------
# Legacy: streaming build + subsample (kept for backward compatibility)
# ---------------------------------------------------------------------------

def subsample_from_disk(full_dir, output_dir, target_per_split, min_per_class, acr_to_name,
                        per_split_targets=None):
    """Subsample a full dataset on disk to create a smaller balanced version.

    Uses single-pass random file sampling — shuffles chunk files and reads only
    as many as needed to hit the target. Much faster than scanning all files.

    Args:
        per_split_targets: dict like {'train': 130000, 'val': 30000, 'test': 40000}.
                          If set, overrides target_per_split for each split.
    """
    if per_split_targets:
        print("\nBuilding datasets from {} -> {}".format(full_dir, output_dir))
        print("  Targets: train={}, val={}, test={}".format(
            per_split_targets.get('train', target_per_split),
            per_split_targets.get('val', target_per_split),
            per_split_targets.get('test', target_per_split)))
    else:
        print("\nBuilding dataset from {} -> {} (target {}/split)...".format(
            full_dir, output_dir, target_per_split))
    rng = np.random.RandomState(42)

    depth_regions_all = defaultdict(Counter)
    fine_ids_all = Counter()

    for split in ['train', 'val', 'test']:
        t_split = time.time()
        split_target = target_per_split
        if per_split_targets and split in per_split_targets:
            split_target = per_split_targets[split]

        split_src = os.path.join(full_dir, split)
        split_dst = os.path.join(output_dir, split)
        os.makedirs(split_dst, exist_ok=True)

        # Get file list (use os.listdir instead of glob for speed on large dirs)
        all_names = [f for f in os.listdir(split_src) if f.startswith('samples_') and f.endswith('.pkl')]
        all_names.sort()
        n_total_files = len(all_names)
        # Each chunk has ~200 samples
        total_est = n_total_files * 200

        print("  {}: ~{:,} samples in {} files (target: {:,})".format(
            split, total_est, n_total_files, split_target))

        if total_est <= split_target:
            import shutil
            print("    Target >= total, copying all files...")
            for fn in all_names:
                shutil.copy2(os.path.join(split_src, fn), os.path.join(split_dst, fn))
            if os.path.exists(os.path.join(split_src, 'compact_index.pkl')):
                shutil.copy2(os.path.join(split_src, 'compact_index.pkl'),
                            os.path.join(split_dst, 'compact_index.pkl'))
            sample_fns = rng.choice(all_names, min(500, len(all_names)), replace=False)
            for fn in sample_fns:
                with open(os.path.join(split_src, fn), 'rb') as f:
                    samples = pickle.load(f)
                for s in samples:
                    hier = s[2]
                    fine_ids_all[hier['fine_id']] += 1
                    for d in hier:
                        if isinstance(d, int):
                            depth_regions_all[d][hier[d]] += 1
                del samples
            print("    Done ({:.0f}s)".format(time.time() - t_split))
            continue

        sampling_rate = min(1.0, split_target * 1.05 / total_est)
        shuffled_names = list(all_names)
        rng.shuffle(shuffled_names)

        chunk_size = 200
        n_out_files = 0
        n_collected = 0
        cumulative = [0]
        file_list = []
        buffer = []
        split_depth_counts = defaultdict(Counter)
        class_counts = Counter()

        def flush_buf():
            nonlocal n_out_files, n_collected
            if not buffer:
                return
            out_path = os.path.join(split_dst, 'samples_{:04d}.pkl'.format(n_out_files))
            with open(out_path, 'wb') as f:
                pickle.dump(buffer, f)
            cumulative.append(cumulative[-1] + len(buffer))
            file_list.append(os.path.join(output_dir, split, 'samples_{:04d}.pkl'.format(n_out_files)))
            n_collected += len(buffer)
            n_out_files += 1
            buffer.clear()

        files_read = 0
        for fn in shuffled_names:
            if n_collected >= split_target:
                break
            with open(os.path.join(split_src, fn), 'rb') as f:
                samples = pickle.load(f)
            files_read += 1
            for s in samples:
                if n_collected + len(buffer) >= split_target:
                    break
                if rng.random() > sampling_rate:
                    continue
                buffer.append(s)
                hier = s[2]
                cls = hier.get('fine_acr', 'unknown')
                class_counts[cls] += 1
                fine_ids_all[hier['fine_id']] += 1
                for d in hier:
                    if isinstance(d, int):
                        depth_regions_all[d][hier[d]] += 1
                        split_depth_counts[d][hier[d]] += 1
                if len(buffer) >= chunk_size:
                    flush_buf()
            del samples
            if files_read % 500 == 0:
                print("    ... read {}/{} files, collected {:,}/{:,} samples".format(
                    files_read, n_total_files, n_collected + len(buffer), split_target))
                sys.stdout.flush()

        flush_buf()

        index = {
            'n_samples': n_collected,
            'n_files': n_out_files,
            'samples_per_file': chunk_size,
            'file_list': file_list,
            'cumulative_counts': cumulative,
            'split_name': split,
            'label_type': 'allen_atlas_hierarchical',
            'num_classes_by_depth': {d: len(split_depth_counts[d]) for d in split_depth_counts},
            'has_coordinates': True,
            'coordinate_system': 'allen_ccf_v3',
            'coordinate_units': 'micrometers',
        }
        with open(os.path.join(split_dst, 'compact_index.pkl'), 'wb') as f:
            pickle.dump(index, f)

        d6 = split_depth_counts.get(6, {})
        top5 = sorted(d6.items(), key=lambda x: -x[1])[:5]
        n_classes = len(class_counts)
        elapsed = time.time() - t_split
        print("    {:,} samples in {} files, {} fine classes ({:.0f}s, read {}/{} files)".format(
            n_collected, n_out_files, n_classes, elapsed, files_read, n_total_files))
        print("    depth-6 top: {}".format(top5))
        gc.collect()

    # Write global metadata
    label_maps = {}
    for d in depth_regions_all:
        sorted_regions = sorted(depth_regions_all[d].items(), key=lambda x: -x[1])
        label_maps[d] = {acr: i for i, (acr, _) in enumerate(sorted_regions)}

    sorted_fine = sorted(fine_ids_all.items(), key=lambda x: -x[1])
    fine_id_map = {sid: i for i, (sid, _) in enumerate(sorted_fine)}

    out_metadata = {
        'label_type': 'allen_atlas_hierarchical',
        'label_maps': label_maps,
        'fine_id_map': fine_id_map,
        'num_classes_by_depth': {d: len(label_maps[d]) for d in label_maps},
        'region_counts_by_depth': {d: dict(depth_regions_all[d]) for d in depth_regions_all},
        'fine_id_counts': dict(fine_ids_all),
        'atlas_file': 'data/allen_ccf/annotation_25.nrrd',
        'structure_tree': 'data/allen_ccf/structure_tree.json',
        'region_names': acr_to_name,
        'sample_format': '(signal, meta_str, atlas_hierarchy_dict, coords)',
        'atlas_hierarchy_format': '{depth_int: acronym_str, ..., fine_id: int, fine_acr: str, max_depth: int}',
    }
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(out_metadata, f)

    print("\n  Classes by depth:")
    for d in sorted(label_maps.keys()):
        print("    depth {}: {} classes".format(d, len(label_maps[d])))


def process_and_save_streaming(raw_dir, split_sessions, channel_map, output_dir):
    """Process raw files and write samples to disk in streaming fashion.

    Returns per-split sample counts and class statistics (without holding samples in RAM).
    """
    all_files = sorted(glob(os.path.join(raw_dir, '*.pickle')))
    print("\nProcessing {} raw session files (streaming)...".format(len(all_files)))

    # Map filename -> split(s)
    file_to_splits = {}
    for f in all_files:
        fname = os.path.basename(f).replace('.pickle', '')
        splits = []
        for split in ['train', 'val', 'test']:
            if fname in split_sessions[split]:
                splits.append(split)
        if splits:
            file_to_splits[f] = splits

    print("  Files with split assignment: {}".format(len(file_to_splits)))
    print("  Files without assignment: {}".format(len(all_files) - len(file_to_splits)))

    # Prepare output directories and buffers
    chunk_size = 200
    split_state = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        split_state[split] = {
            'buffer': [],
            'n_files': 0,
            'n_samples': 0,
            'cumulative': [0],
            'file_list': [],
            'depth_counts': defaultdict(Counter),
            'fine_counts': Counter(),
        }

    stats = {'processed_files': 0, 'total_raw_samples': 0,
             'matched': 0, 'no_channel': 0, 'no_coords': 0}

    def flush_buffer(split):
        """Write buffer to disk as a chunk file."""
        state = split_state[split]
        if not state['buffer']:
            return
        split_dir = os.path.join(output_dir, split)
        out_path = os.path.join(split_dir, 'samples_{:04d}.pkl'.format(state['n_files']))
        with open(out_path, 'wb') as f:
            pickle.dump(state['buffer'], f)
        state['cumulative'].append(state['cumulative'][-1] + len(state['buffer']))
        state['file_list'].append(os.path.join(output_dir, split, 'samples_{:04d}.pkl'.format(state['n_files'])))
        state['n_files'] += 1
        state['buffer'] = []

    for i, (fpath, splits) in enumerate(sorted(file_to_splits.items())):
        if i % 20 == 0:
            print("  [{}/{}] Processing {}...".format(
                i + 1, len(file_to_splits), os.path.basename(fpath)))
            sys.stdout.flush()

        try:
            with open(fpath, 'rb') as f:
                raw_samples = pickle.load(f)
        except Exception as e:
            print("    ERROR loading {}: {}".format(fpath, e))
            continue

        stats['processed_files'] += 1
        stats['total_raw_samples'] += len(raw_samples)

        for sample in raw_samples:
            signal, meta = sample[0], sample[1]

            # Parse channel_id from meta_str
            parts = meta.split('_')
            if len(parts) < 4:
                stats['no_channel'] += 1
                continue

            try:
                channel_id = int(parts[3])
            except ValueError:
                stats['no_channel'] += 1
                continue

            if channel_id not in channel_map:
                stats['no_coords'] += 1
                continue

            info = channel_map[channel_id]
            new_sample = (signal, meta, info['hierarchy'], info['coords'])

            for split in splits:
                state = split_state[split]
                state['buffer'].append(new_sample)
                state['n_samples'] += 1

                # Track stats (lightweight — just counters)
                hier = info['hierarchy']
                state['fine_counts'][hier['fine_id']] += 1
                for d in hier:
                    if isinstance(d, int):
                        state['depth_counts'][d][hier[d]] += 1

                # Flush when buffer is full
                if len(state['buffer']) >= chunk_size:
                    flush_buffer(split)

            stats['matched'] += 1

        # Free memory after each file
        del raw_samples
        gc.collect()

    # Flush remaining buffers
    for split in ['train', 'val', 'test']:
        flush_buffer(split)

    print("\nExtraction stats:")
    print("  Files processed: {}".format(stats['processed_files']))
    print("  Total raw samples: {:,}".format(stats['total_raw_samples']))
    print("  Matched (brain-filtered): {:,}".format(stats['matched']))
    print("  No channel ID: {:,}".format(stats['no_channel']))
    print("  Channel outside brain: {:,}".format(stats['no_coords']))

    for split in ['train', 'val', 'test']:
        state = split_state[split]
        print("  {}: {:,} samples in {} files".format(split, state['n_samples'], state['n_files']))

    return split_state


def write_index_and_metadata(split_state, output_dir, acr_to_name):
    """Write compact_index.pkl per split and global metadata.pkl."""
    print("\nWriting indices and metadata...")

    # Aggregate stats across splits
    depth_regions_all = defaultdict(Counter)
    fine_ids_all = Counter()

    for split in ['train', 'val', 'test']:
        state = split_state[split]

        for d, counts in state['depth_counts'].items():
            depth_regions_all[d].update(counts)
        fine_ids_all.update(state['fine_counts'])

        index = {
            'n_samples': state['n_samples'],
            'n_files': state['n_files'],
            'samples_per_file': 200,
            'file_list': state['file_list'],
            'cumulative_counts': state['cumulative'],
            'split_name': split,
            'label_type': 'allen_atlas_hierarchical',
            'num_classes_by_depth': {d: len(state['depth_counts'][d]) for d in state['depth_counts']},
            'has_coordinates': True,
            'coordinate_system': 'allen_ccf_v3',
            'coordinate_units': 'micrometers',
        }
        with open(os.path.join(output_dir, split, 'compact_index.pkl'), 'wb') as f:
            pickle.dump(index, f)

        # Show distribution at depth 6
        d6 = state['depth_counts'].get(6, {})
        top5 = sorted(d6.items(), key=lambda x: -x[1])[:5]
        print("  {}: {} samples, depth-6 top: {}".format(split, state['n_samples'], top5))

    # Build global label maps
    label_maps = {}
    for d in depth_regions_all:
        sorted_regions = sorted(depth_regions_all[d].items(), key=lambda x: -x[1])
        label_maps[d] = {acr: i for i, (acr, _) in enumerate(sorted_regions)}

    sorted_fine = sorted(fine_ids_all.items(), key=lambda x: -x[1])
    fine_id_map = {sid: i for i, (sid, _) in enumerate(sorted_fine)}

    metadata = {
        'label_type': 'allen_atlas_hierarchical',
        'label_maps': label_maps,
        'fine_id_map': fine_id_map,
        'num_classes_by_depth': {d: len(label_maps[d]) for d in label_maps},
        'region_counts_by_depth': {d: dict(depth_regions_all[d]) for d in depth_regions_all},
        'fine_id_counts': dict(fine_ids_all),
        'atlas_file': 'data/allen_ccf/annotation_25.nrrd',
        'structure_tree': 'data/allen_ccf/structure_tree.json',
        'region_names': acr_to_name,
        'sample_format': '(signal, meta_str, atlas_hierarchy_dict, coords)',
        'atlas_hierarchy_format': '{depth_int: acronym_str, ..., fine_id: int, fine_acr: str, max_depth: int}',
    }
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print("\n  Classes by depth:")
    for d in sorted(label_maps.keys()):
        print("    depth {}: {} classes".format(d, len(label_maps[d])))
    print("  Fine-grained: {} unique structure IDs".format(len(fine_id_map)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Build atlas-labeled dataset from raw session pickles (index-first approach)')

    # Shared args
    parser.add_argument('--raw-dir', default='data/Allen_w2v2/Allen')
    parser.add_argument('--index-dir', default='blind_localization/data/130k_index',
                        help='Directory with balanced_index_{split}.pkl (session splits)')
    parser.add_argument('--channels-csv', default='data/channels.csv')
    parser.add_argument('--atlas', default='data/allen_ccf/annotation_25.nrrd')
    parser.add_argument('--structure-tree', default='data/allen_ccf/structure_tree.json')

    # Phase 1: Build index
    parser.add_argument('--build-index', action='store_true',
                        help='Phase 1: Scan raw files and build lightweight index')
    parser.add_argument('--atlas-index', default='blind_localization/data/atlas_index',
                        help='Directory for atlas index output (Phase 1) / input (Phase 2)')
    parser.add_argument('--coverage-depth', type=int, default=6,
                        help='Depth to check class coverage at (for session reassignment)')

    # Phase 2: Build from index
    parser.add_argument('--from-index', action='store_true',
                        help='Phase 2: Build compact dataset from pre-built index')
    parser.add_argument('--output-dir', default='blind_localization/data/compact_atlas_13k')
    parser.add_argument('--target-per-split', type=int, default=0,
                        help='Target samples per split (0 = keep all)')
    parser.add_argument('--per-split-targets', type=int, nargs=3,
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='Per-split targets: TRAIN VAL TEST')
    parser.add_argument('--chunk-size', type=int, default=5000,
                        help='Samples per output chunk file (default 5000)')
    parser.add_argument('--seed', type=int, default=42)

    # Legacy mode (backward compat)
    parser.add_argument('--legacy', action='store_true',
                        help='Use old streaming build (200 samples/file)')
    parser.add_argument('--balanced-uniform', action='store_true',
                        help='Uniform per-class selection (fill under-sampled evaluable regions)')
    parser.add_argument('--per-class', type=int, default=0,
                        help='Samples per fine_id in balanced-uniform mode')
    parser.add_argument('--evaluable-set', default='',
                        help='Pickle of allowed fine_ids (set) for balanced-uniform mode')
    parser.add_argument('--per-region-targets', default='',
                        help='Pickle dict {fine_id: target_count} for rare-boost max(natural,F) mode')
    parser.add_argument('--balance-splits', default='train',
                        help='Comma-separated splits to apply balanced-uniform to (default train)')
    parser.add_argument('--only-splits', default='',
                        help='Comma-separated splits to build at all (default all)')
    parser.add_argument('--min-per-class', type=int, default=100,
                        help='Minimum samples per fine_id class (Phase 2 and legacy)')
    parser.add_argument('--subsample-output-dir', default='')
    parser.add_argument('--subsample-target', type=int, default=13000)
    parser.add_argument('--subsample-targets', type=int, nargs=3,
                        metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--subsample-only', action='store_true')

    args = parser.parse_args()
    t0 = time.time()

    if args.build_index:
        # ---- Phase 1: Build index ----
        print("=== Phase 1: Building atlas index ===\n")

        anno, anno_id_to_hierarchy, acr_to_name = load_atlas(args.atlas, args.structure_tree)
        channel_map = build_filtered_channel_map(args.channels_csv, anno, anno_id_to_hierarchy)
        del anno, anno_id_to_hierarchy
        gc.collect()

        print("\nLoading split assignments...")
        split_sessions = load_split_sessions(args.index_dir)

        raw_files, split_data = build_atlas_index(
            args.raw_dir, split_sessions, channel_map, args.atlas_index)
        build_index_metadata(split_data, channel_map, args.atlas_index, acr_to_name)

        # DISABLED — coverage already guaranteed by build_ibl_splits.py
        print("\nSkipping auto-reassignment (IBL splits pre-balanced).")

        # # Check class coverage and reassign at multiple depths (6, 7, 8, 9)
        # # Moving a file adds ALL its samples across all depths, so fixing
        # # deeper levels often fixes shallower ones too. Go coarse-to-fine.
        # coverage_depths = [6, 7, 8, 9]
        # total_reassigned = 0
        # for depth in coverage_depths:
        #     split_classes, split_counts, missing = check_class_coverage(
        #         args.atlas_index, channel_map, target_depth=depth)

        #     if missing:
        #         reassign_sessions_for_coverage(
        #             args.atlas_index, raw_files, channel_map, missing,
        #             target_depth=depth)
        #         total_reassigned += len(missing)

        # if total_reassigned > 0:
        #     # Final summary across all depths
        #     print("\n" + "=" * 80)
        #     print("FINAL COVERAGE CHECK (after all reassignments)")
        #     print("=" * 80)
        #     for depth in coverage_depths:
        #         check_class_coverage(
        #             args.atlas_index, channel_map, target_depth=depth)
        # else:
        #     print("\nNo reassignments needed at any depth.")

    elif args.from_index:
        # ---- Phase 2: Build from index ----
        print("=== Phase 2: Building compact dataset from index ===\n")

        per_split_targets = None
        if args.per_split_targets:
            per_split_targets = {
                'train': args.per_split_targets[0],
                'val': args.per_split_targets[1],
                'test': args.per_split_targets[2],
            }

        evaluable_set = None
        if args.evaluable_set:
            with open(args.evaluable_set, 'rb') as f:
                evaluable_set = set(int(x) for x in pickle.load(f))
        only_splits = (set(args.only_splits.split(',')) if args.only_splits else None)
        per_region_targets = None
        if args.per_region_targets:
            with open(args.per_region_targets, 'rb') as f:
                per_region_targets = {int(k): int(v) for k, v in pickle.load(f).items()}
        build_from_index(
            args.atlas_index, args.output_dir,
            target_per_split=args.target_per_split,
            per_split_targets=per_split_targets,
            chunk_size=args.chunk_size,
            seed=args.seed,
            min_per_class=args.min_per_class,
            balanced_uniform=args.balanced_uniform,
            per_class=args.per_class,
            evaluable_set=evaluable_set,
            balance_splits=tuple(args.balance_splits.split(',')),
            only_splits=only_splits,
            per_region_targets=per_region_targets)

    elif args.subsample_only or args.legacy:
        # ---- Legacy modes ----
        per_split_targets = None
        if args.subsample_targets:
            per_split_targets = {
                'train': args.subsample_targets[0],
                'val': args.subsample_targets[1],
                'test': args.subsample_targets[2],
            }

        if args.subsample_only:
            if not args.subsample_output_dir:
                print("ERROR: --subsample-only requires --subsample-output-dir")
                sys.exit(1)
            with open(args.structure_tree) as f:
                structures = json.load(f)
            acr_to_name = {s['acronym']: s['name'] for s in structures}

            subsample_from_disk(args.output_dir, args.subsample_output_dir,
                               args.subsample_target, args.min_per_class, acr_to_name,
                               per_split_targets=per_split_targets)
        else:
            anno, anno_id_to_hierarchy, acr_to_name = load_atlas(args.atlas, args.structure_tree)
            channel_map = build_filtered_channel_map(args.channels_csv, anno, anno_id_to_hierarchy)
            del anno, anno_id_to_hierarchy
            gc.collect()

            print("\nLoading split assignments...")
            split_sessions = load_split_sessions(args.index_dir)

            split_state = process_and_save_streaming(
                args.raw_dir, split_sessions, channel_map, args.output_dir)
            write_index_and_metadata(split_state, args.output_dir, acr_to_name)

            if args.subsample_output_dir:
                subsample_from_disk(args.output_dir, args.subsample_output_dir,
                                   args.subsample_target, args.min_per_class, acr_to_name,
                                   per_split_targets=per_split_targets)
    else:
        parser.print_help()
        print("\nERROR: Specify --build-index (Phase 1) or --from-index (Phase 2)")
        sys.exit(1)

    elapsed = time.time() - t0
    print("\nTotal time: {:.1f} minutes".format(elapsed / 60))


if __name__ == '__main__':
    main()
