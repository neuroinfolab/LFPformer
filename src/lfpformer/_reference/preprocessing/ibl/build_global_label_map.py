"""Build global effective label maps across ALL datasets and ALL splits.

Scans all 4 atlas datasets (25K/75K/240K/1M) and all 3 splits (train/val/test),
takes the INTERSECTION of regions across everything, and creates ONE canonical
mapping per depth. Only classes present in EVERY dataset's EVERY split are included.

Outputs:
  1. blind_localization/data/atlas_data/global_effective_label_maps.pkl
  2. Patches each dataset's metadata.pkl

Usage:
  python3 script/build_global_label_map.py
"""

import os
import pickle


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_pickle(path):
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            return NumpyCompatUnpickler(f).load()


def get_cumulative_label(hier, depth):
    effective_depth = min(depth, hier.get('max_depth', 0))
    return hier.get(effective_depth, None)


def scan_split(ds_path, split):
    """Return {depth: set(region_names)} for one split of one dataset."""
    split_dir = os.path.join(ds_path, split)
    index_path = os.path.join(split_dir, 'compact_index.pkl')
    if not os.path.exists(index_path):
        return None

    idx = load_pickle(index_path)
    regions = {d: set() for d in range(10)}

    for fpath in idx['file_list']:
        if not os.path.exists(fpath):
            # Fall back to filename in split_dir (handles path mismatches)
            fpath = os.path.join(split_dir, os.path.basename(fpath))
        samples = load_pickle(fpath)
        for s in samples:
            hier = s[2]
            if not isinstance(hier, dict):
                continue
            for depth in range(10):
                label = get_cumulative_label(hier, depth)
                if label is not None:
                    regions[depth].add(label)

    return regions, idx['n_samples']


def main():
    base = 'blind_localization/data'
    datasets = [
        __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/compact_atlas_ibl_25k'),
        # os.path.join(base, 'atlas_data/compact_atlas_25k'),
        # os.path.join(base, 'atlas_data/compact_atlas_75k'),
        # os.path.join(base, 'atlas_data/compact_atlas_240k'),
        # os.path.join(base, 'atlas_data/compact_atlas_1M'),
    ]
    splits = ['train', 'val', 'test']

    # Collect regions per dataset per split: {ds: {split: {depth: set(names)}}}
    per_ds_regions = {}
    # Also collect union across datasets for val/test reporting
    split_regions = {s: {d: set() for d in range(10)} for s in splits}

    for ds_path in datasets:
        ds_name = os.path.basename(ds_path)
        if not os.path.exists(ds_path):
            print("SKIP {} (not found)".format(ds_name))
            continue

        per_ds_regions[ds_name] = {}
        for split in splits:
            result = scan_split(ds_path, split)
            if result is None:
                continue
            regions, n = result
            print("Scanned {}/{} ({:,} samples)".format(ds_name, split, n))
            per_ds_regions[ds_name][split] = regions
            for depth in range(10):
                split_regions[split][depth] |= regions[depth]

    # Build ONE global map from INTERSECTION across ALL datasets AND ALL splits.
    # Only classes present in every dataset's every split are included.
    ds_names = sorted(per_ds_regions.keys())

    global_regions = {d: None for d in range(10)}
    for ds_name in ds_names:
        for split in splits:
            if split not in per_ds_regions[ds_name]:
                continue
            for depth in range(10):
                ds_regions = per_ds_regions[ds_name][split][depth]
                if global_regions[depth] is None:
                    global_regions[depth] = set(ds_regions)
                else:
                    global_regions[depth] &= ds_regions

    global_label_map = {}    # {depth: {name: id}}
    global_id_to_name = {}   # {depth: {id: name}}
    global_num_classes = {}   # {depth: int}

    for depth in range(10):
        names = sorted(global_regions[depth]) if global_regions[depth] else []
        lm = {name: i for i, name in enumerate(names)}
        global_label_map[depth] = lm
        global_id_to_name[depth] = {i: name for name, i in lm.items()}
        global_num_classes[depth] = len(names)

    # Print summary
    print("\n=== Global Effective Label Maps (intersection of ALL datasets × ALL splits) ===\n")
    print("| Depth | Classes |")
    print("|:-----:|:-------:|")
    for depth in range(10):
        print("| {} | {} |".format(depth, global_num_classes[depth]))

    # Show excluded regions per split at deeper depths
    for depth in [6, 7, 8, 9]:
        global_set = set(global_label_map[depth].keys())
        for split in splits:
            extra = sorted(split_regions[split][depth] - global_set)
            if extra:
                print("\n  Depth {} {} — {} regions excluded: {}".format(
                    depth, split, len(extra), extra))

    # Save
    # out_path = os.path.join(base, 'atlas_data/global_effective_label_maps.pkl')
    out_path = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/global_effective_label_maps.pkl')
    payload = {
        'label_maps': global_label_map,      # {depth: {name: id}}
        'id_to_name': global_id_to_name,     # {depth: {id: name}}
        'num_classes': global_num_classes,    # {depth: int}
        'source_datasets': [os.path.basename(d) for d in datasets if os.path.exists(d)],
        'description': (
            'Global effective label maps built from the INTERSECTION of ALL splits '
            '(train/val/test) across ALL atlas datasets (25K/75K/240K/1M). One map '
            'per depth, shared by all splits and all datasets. Only classes present '
            'in every dataset every split are included. Samples with regions not in '
            'map get label=-1. Region names sorted alphabetically, contiguous IDs 0..N-1.'
        ),
    }
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f, protocol=4)
    print("\nSaved: {}".format(out_path))

    # Patch each dataset's metadata.pkl
    print("\nPatching dataset metadata.pkl files...")
    for ds_path in datasets:
        ds_name = os.path.basename(ds_path)
        meta_path = os.path.join(ds_path, 'metadata.pkl')
        if not os.path.exists(meta_path):
            print("  SKIP {} (no metadata.pkl)".format(ds_name))
            continue

        meta = load_pickle(meta_path)
        meta['effective_label_maps_by_depth'] = global_label_map
        meta['effective_id_to_name_by_depth'] = global_id_to_name
        meta['effective_num_classes_by_depth'] = global_num_classes
        meta['global_label_map_source'] = 'global_effective_label_maps.pkl'
        meta['label_map_type'] = 'all_intersection'

        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f, protocol=4)
        print("  {} — patched (d9={} classes, same for all splits)".format(
            ds_name, global_num_classes[9]))

    # Verify: per-dataset coverage and samples excluded
    print("\n=== Verification: per-dataset coverage (depth 9) ===\n")
    print("| Dataset | Split | Actual | Map | Coverage | Excluded |")
    print("|---------|-------|:------:|:---:|:--------:|:--------:|")
    for ds_path in datasets:
        ds_name = os.path.basename(ds_path)
        if not os.path.exists(ds_path):
            continue
        for split in splits:
            result = scan_split(ds_path, split)
            if result is None:
                continue
            regions, _ = result
            map_keys = set(global_label_map[9].keys())
            actual_in_map = len(regions[9] & map_keys)
            excluded = len(regions[9] - map_keys)
            total = global_num_classes[9]
            pct = 100.0 * actual_in_map / total if total > 0 else 0
            print("| {} | {} | {} | {} | {:.0f}% | {} |".format(
                ds_name, split, actual_in_map, total, pct, excluded))

    print("\nDone!")


if __name__ == '__main__':
    main()
