#!/usr/bin/env python3
"""Convert one probe's step-1 block pickles into a memory-mapped dataset "part".

Replaces build_atlas_dataset.py steps 5-6 + per-block compact shards for the
full-recording set (CompactDataset's pickle shards do not scale to 257M; see
GCP_PRETRAINING.md §2f). Values are unchanged:
  * signal   = the step-1 float32[3750] (exact bytes)
  * coords   = channel_map[channel_id]['coords'] (CCF ap,dv,lr µm), from
               Lawrence's own atlas_index_ibl_v2/channel_map.pkl, the same object
               build_atlas_dataset used, stored as float32 (what CompactDataset
               returns after torch.tensor(..., float32))
  * hierarchy = channel_map[channel_id]['hierarchy'] -> stored as fine_id; the
               dict is recoverable from hier_by_fine_id.pkl
  * samples whose channel_id is not in channel_map are dropped, exactly like
    build_atlas_index ("Channel outside brain").
Order = block order, then sample order within each block pickle (block 0 order
== Lawrence's compact order for that probe).

Part layout  <out_root>/<split>/parts/<eid>_<probe>/
  signals.f32   raw float32, shape (n, 3750)
  coords.f32    raw float32, shape (n, 3)
  fine_id.i32   int32 (n,)
  chunk_idx.i32 int32 (n,)   global chunk index (block*200 + local)
  ch_idx.i16    int16 (n,)
  region_idx.u16 uint16 (n,) index into part.json region_vocab (meta_str region)
  part.json     {n, eid, probe, pid8, blocks, region_vocab, n_dropped_no_channel}
"""
import argparse
import glob
import json
import os
import pickle
import time

import numpy as np

CHUNK_LEN = 3750


def convert_probe(eid, probe, blocks_root, channel_map, out_dir, delete_pickles=False):
    files = sorted(glob.glob(os.path.join(blocks_root, 'b[0-9][0-9]', f'{eid}_{probe}.pickle')))
    if not files:
        raise FileNotFoundError(f'no block pickles for {eid}_{probe} under {blocks_root}')
    tmp_dir = out_dir + '.tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    fs = {k: open(os.path.join(tmp_dir, k), 'wb') for k in
          ('signals.f32', 'coords.f32', 'fine_id.i32', 'chunk_idx.i32', 'ch_idx.i16', 'region_idx.u16')}
    vocab, n, n_drop, blocks, pid8 = {}, 0, 0, [], None
    for f in files:
        samples = pickle.load(open(f, 'rb'))
        sig, crd, fid, chk, chx, reg = [], [], [], [], [], []
        for signal, meta in samples:
            parts = meta.split('_')
            # meta = {eid}_{chunk}_{probe}_{pid8}-chNNNN_{region}; eid has no '_'
            channel_id = parts[3]
            info = channel_map.get(channel_id)
            if info is None:
                n_drop += 1
                continue
            assert parts[0] == eid and parts[2] == probe, meta
            region = '_'.join(parts[4:])
            pid8 = channel_id.split('-ch')[0]
            sig.append(np.asarray(signal, dtype=np.float32))
            crd.append(info['coords'])
            fid.append(int(info['hierarchy']['fine_id']))
            chk.append(int(parts[1]))
            chx.append(int(channel_id.split('-ch')[1]))
            reg.append(vocab.setdefault(region, len(vocab)))
        if sig:
            s = np.stack(sig)
            assert s.shape[1] == CHUNK_LEN and s.dtype == np.float32
            fs['signals.f32'].write(s.tobytes())
            fs['coords.f32'].write(np.asarray(crd, dtype=np.float64).astype(np.float32).tobytes())
            fs['fine_id.i32'].write(np.asarray(fid, dtype=np.int32).tobytes())
            fs['chunk_idx.i32'].write(np.asarray(chk, dtype=np.int32).tobytes())
            fs['ch_idx.i16'].write(np.asarray(chx, dtype=np.int16).tobytes())
            fs['region_idx.u16'].write(np.asarray(reg, dtype=np.uint16).tobytes())
            n += len(sig)
        blocks.append(dict(file=os.path.relpath(f, blocks_root), n_in=len(samples), n_kept=len(sig)))
        del samples, sig
    for fh in fs.values():
        fh.close()
    json.dump(dict(n=n, eid=eid, probe=probe, pid8=pid8, blocks=blocks,
                   region_vocab=sorted(vocab, key=vocab.get), n_dropped_no_channel=n_drop,
                   chunk_len=CHUNK_LEN, created=time.strftime('%Y-%m-%dT%H:%M:%S')),
              open(os.path.join(tmp_dir, 'part.json'), 'w'), indent=1)
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    os.rename(tmp_dir, out_dir)  # atomic publish
    if delete_pickles:
        for f in files:
            os.remove(f)
    return n, n_drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks-root', required=True, help='dir containing bBB/{eid}_{probe}.pickle')
    ap.add_argument('--channel-map', required=True, help='atlas_index_ibl_v2/channel_map.pkl')
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--probes', nargs='+', required=True, help='{eid}_{probe} names')
    ap.add_argument('--delete-pickles', action='store_true')
    args = ap.parse_args()

    channel_map = pickle.load(open(args.channel_map, 'rb'))
    for name in args.probes:
        eid, probe = name.split('_', 1)
        out_dir = os.path.join(args.out_root, args.split, 'parts', name)
        if os.path.exists(os.path.join(out_dir, 'part.json')):
            print(f'{name}: exists, skip', flush=True)
            continue
        t0 = time.time()
        n, n_drop = convert_probe(eid, probe, args.blocks_root, channel_map, out_dir, args.delete_pickles)
        print(f'{name}: {n:,} samples kept, {n_drop} dropped (no channel), {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
