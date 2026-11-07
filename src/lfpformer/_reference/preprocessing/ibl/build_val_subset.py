#!/usr/bin/env python3
"""Build the fixed validation subset (memmap split 'val_subset') from the full val split.

Deterministic: for every val probe (part), take all in-brain channels at 3 chunk
times located at the 10% / 50% / 90% quantiles of that probe's chunk indices
(i.e. early / middle / late in the recording). ~3 x ~366 rows per probe x 141
probes ~= 150k samples, covering every val probe, region and time range.
Rows are copied from the val parts (same bytes), so the subset is exact data.
"""
import argparse
import json
import os
import shutil

import numpy as np

FILES = [('signals.f32', np.float32, 3750), ('coords.f32', np.float32, 3), ('fine_id.i32', np.int32, 1),
         ('chunk_idx.i32', np.int32, 1), ('ch_idx.i16', np.int16, 1), ('region_idx.u16', np.uint16, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dataset root containing val/')
    ap.add_argument('--out-split', default='val_subset')
    ap.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.5, 0.9])
    args = ap.parse_args()

    src = os.path.join(args.root, 'val')
    dst = os.path.join(args.root, args.out_split)
    idx = json.load(open(os.path.join(src, 'memmap_index.json')))
    os.makedirs(os.path.join(dst, 'parts'), exist_ok=True)
    rows_out = []
    for p in idx['parts']:
        name, n = p['name'], p['n']
        sp = os.path.join(src, 'parts', name)
        chunk = np.fromfile(os.path.join(sp, 'chunk_idx.i32'), dtype=np.int32)
        assert len(chunk) == n
        uniq = np.unique(chunk)
        picks = sorted({int(uniq[min(int(q * len(uniq)), len(uniq) - 1)]) for q in args.quantiles})
        sel = np.flatnonzero(np.isin(chunk, picks))
        dp = os.path.join(dst, 'parts', name)
        os.makedirs(dp, exist_ok=True)
        for fn, dt, w in FILES:
            a = np.memmap(os.path.join(sp, fn), dtype=dt, mode='r', shape=(n, w) if w > 1 else (n,))
            np.ascontiguousarray(a[sel]).tofile(os.path.join(dp, fn))
        pj = json.load(open(os.path.join(sp, 'part.json')))
        pj.update(n=int(len(sel)), subset_of='val', subset_chunks=picks, blocks=pj.get('blocks', []))
        json.dump(pj, open(os.path.join(dp, 'part.json'), 'w'), indent=1)
        rows_out.append(dict(name=name, n=int(len(sel))))
    shutil.copy(os.path.join(src, 'hier_by_fine_id.pkl'), os.path.join(dst, 'hier_by_fine_id.pkl'))
    total = sum(r['n'] for r in rows_out)
    json.dump(dict(format='ibl_memmap_v1', split=args.out_split, n_samples=total, n_parts=len(rows_out),
                   chunk_len=3750, parts=rows_out, subset_rule=f'all channels at chunk quantiles {args.quantiles} per val probe'),
              open(os.path.join(dst, 'memmap_index.json'), 'w'), indent=1)
    print(f'{args.out_split}: {len(rows_out)} probes, {total:,} samples')


if __name__ == '__main__':
    main()
