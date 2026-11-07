#!/usr/bin/env python3
"""ManipDistc labels for the memmap datasets: {split}_manip_xyz.npy (float32 [N,3],
CCF ap,dv,lr µm) + {split}_valid.npy (bool [N]), in exactly the memmap sample order
(parts in memmap_index.json order, rows in part order).

Per-sample value = manip_line(trajectory of that probe, channel index), the same
function (NP1.0 geometry, TIP_SIZE_UM=200) as precompute_manip_xyz_compact.py,
which produced the 75k/1M ManipDistc arrays. It depends only on (probe, channel),
so it's computed once per probe for its channels and expanded via ch_idx.
Probes without a trajectory -> valid=False (dropped by the ManipDistc wrapper).
Run inside the ibl container (iblatlas).
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'val_subset'])
    ap.add_argument('--manip-json', nargs='+', required=True)
    ap.add_argument('--precompute-script', required=True, help='path to precompute_manip_xyz_compact.py (for manip_line)')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.precompute_script)))
    import precompute_manip_xyz_compact as pm
    from iblatlas.atlas import AllenAtlas, Insertion

    manip = pm._load_manip_cache(args.manip_json)
    ba = AllenAtlas(25)
    print(f'TIP_SIZE_UM={pm.TIP_SIZE_UM}')
    os.makedirs(args.out_dir, exist_ok=True)
    for split in args.splits:
        sd = os.path.join(args.root, split)
        idx = json.load(open(os.path.join(sd, 'memmap_index.json')))
        N = idx['n_samples']
        xyz = np.lib.format.open_memmap(os.path.join(args.out_dir, f'{split}_manip_xyz.npy'), mode='w+',
                                        dtype=np.float32, shape=(N, 3))
        valid = np.lib.format.open_memmap(os.path.join(args.out_dir, f'{split}_valid.npy'), mode='w+',
                                          dtype=bool, shape=(N,))
        off, n_ok, n_probe_ok, n_probe_missing = 0, 0, 0, []
        for p in idx['parts']:
            name, n = p['name'], p['n']
            ch = np.fromfile(os.path.join(sd, 'parts', name, 'ch_idx.i16'), dtype=np.int16).astype(np.int64)
            assert len(ch) == n
            sl = slice(off, off + n)
            xyz[sl] = np.nan
            valid[sl] = False
            if name in manip:
                uch = np.unique(ch)
                try:
                    pts = pm.manip_line(manip[name], uch, ba, Insertion).astype(np.float32)
                    lut = np.full(int(uch.max()) + 1, -1, dtype=np.int64)
                    lut[uch] = np.arange(len(uch))
                    rows = pts[lut[ch]]
                    fin = np.isfinite(rows).all(axis=1)
                    xyz[sl] = rows
                    valid[sl] = fin
                    n_ok += int(fin.sum())
                    n_probe_ok += 1
                except Exception as e:  # noqa: BLE001
                    print(f'[WARN] manip_line failed for {name}: {e}')
                    n_probe_missing.append(name)
            else:
                n_probe_missing.append(name)
            off += n
        assert off == N
        xyz.flush(); valid.flush()
        print(f'{split}: N={N:,} valid={n_ok:,} ({100*n_ok/max(N,1):.1f}%) probes_ok={n_probe_ok}/{len(idx["parts"])} '
              f'missing={len(n_probe_missing)} {n_probe_missing[:12]}', flush=True)
    print('DONE', args.out_dir)


if __name__ == '__main__':
    main()
