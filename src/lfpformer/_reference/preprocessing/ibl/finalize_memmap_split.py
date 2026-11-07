#!/usr/bin/env python3
"""Index all memmap parts of one split into a dataset dir readable by the
memmap-aware CompactDataset (gcp_neuroframe/train/code/pl/blind_localization/
data/lazyloader_dataset.py).

Writes into <out_root>/<split>/:
  memmap_index.json   {n_samples, parts:[{name, n}], ...} (parts sorted by name)
  hier_by_fine_id.pkl {fine_id: hierarchy dict} from Lawrence's channel_map.pkl
and copies the label metadata (global effective label maps, so label ids match
the NeurIPS datasets) to <out_root>/metadata.pkl, where CompactDataset looks for it
(the parent dir of the split dir).
"""
import argparse
import json
import os
import pickle
import shutil
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--channel-map', required=True)
    ap.add_argument('--metadata', required=True,
                    help='compact_atlas_ibl_v2_full/metadata.pkl (has effective_label_maps_by_depth)')
    ap.add_argument('--expect-parts', type=int, default=0, help='fail if fewer parts are present')
    args = ap.parse_args()

    split_dir = os.path.join(args.out_root, args.split)
    pdir = os.path.join(split_dir, 'parts')
    names = sorted(d for d in os.listdir(pdir) if os.path.isfile(os.path.join(pdir, d, 'part.json')))
    if args.expect_parts and len(names) < args.expect_parts:
        raise SystemExit(f'only {len(names)} parts, expected {args.expect_parts}')
    parts, total, dropped = [], 0, 0
    for nm in names:
        pj = json.load(open(os.path.join(pdir, nm, 'part.json')))
        sz = os.path.getsize(os.path.join(pdir, nm, 'signals.f32'))
        assert sz == pj['n'] * pj['chunk_len'] * 4, (nm, sz, pj['n'])
        parts.append(dict(name=nm, n=pj['n']))
        total += pj['n']
        dropped += pj['n_dropped_no_channel']

    channel_map = pickle.load(open(args.channel_map, 'rb'))
    hier = {}
    for v in channel_map.values():
        h = v['hierarchy']
        hier.setdefault(int(h['fine_id']), h)
    pickle.dump(hier, open(os.path.join(split_dir, 'hier_by_fine_id.pkl'), 'wb'), protocol=4)

    meta_dst = os.path.join(args.out_root, 'metadata.pkl')
    if not os.path.exists(meta_dst):
        shutil.copy(args.metadata, meta_dst)

    json.dump(dict(format='ibl_memmap_v1', split=args.split, n_samples=total, n_parts=len(parts),
                   n_dropped_no_channel=dropped, chunk_len=3750, parts=parts,
                   created=time.strftime('%Y-%m-%dT%H:%M:%S')),
              open(os.path.join(split_dir, 'memmap_index.json'), 'w'), indent=1)
    print(f'{args.split}: {len(parts)} parts, {total:,} samples ({dropped} dropped no-channel)')


if __name__ == '__main__':
    main()
