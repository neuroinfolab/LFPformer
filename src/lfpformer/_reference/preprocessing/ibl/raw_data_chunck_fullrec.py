#!/usr/bin/env python3
"""Full-recording version of Lawrence's (ws2) IBL step 1, `raw_data_chunck.py`.

Signal processing is copied verbatim from
/workspace/ws2/Alphabrain_staging/script/ibl_preprocessing/raw_data_chunck.py
(detect/interpolate dead channels -> destripe_lfp(nv=1) or manual HP+CAR for
non-384ch -> decimate q=2 zero-phase -> 3 s chunks, drop out-of-brain channels).

Only two things differ:
  1. Input is the local LF .cbin (Zusanna's full-recording copy) read with
     spikeglx.Reader, instead of ONE Streamer. Same file, same volts.
  2. The recording is processed as consecutive 600 s blocks starting at t=30 s
     ([30,630), [630,1230), ...), each exactly like Lawrence's single block.
     Block b is written to  <out_dir>/b{b:02d}/{eid}_{probe}.pickle  so the
     filename -> split matching in build_atlas_dataset.py works unchanged, and
     chunk_idx in meta_str is global (b*200 + local) so it stays unique.
     Block 0 reproduces the existing preprocessed_v2 pickle.
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spikeglx
from ibldsp.voltage import destripe_lfp
from scipy.signal import butter, decimate, sosfiltfilt

T_START = 30       # skip first 30s
T_DUR = 600        # block length (Lawrence's whole window)
CHUNK_SIZE = 3     # seconds
TARGET_FS = 1250   # Hz
CHUNKS_PER_BLOCK = T_DUR // CHUNK_SIZE  # 200

outside = {'root', 'void', 'nan', '', 'out of brain', 'x', 'unknown', 'pitching'}


# ── Helpers: verbatim from raw_data_chunck.py ──────────────────────────────────
def detect_bad_channels(raw, fs, dead_ratio=0.1, neighbor_radius=10):
    """Detect dead channels via local variance comparison on raw data."""
    n_ch = raw.shape[0]
    r = neighbor_radius
    ch_var = np.var(raw, axis=1)
    local_median_var = np.array([
        np.median(ch_var[max(0, i - r):min(n_ch, i + r + 1)])
        for i in range(n_ch)
    ])
    dead = np.where(ch_var < dead_ratio * local_median_var)[0]
    return dead


def interpolate_bad_channels(raw, bad_channels):
    """Replace bad channels with mean of nearest good neighbors."""
    if len(bad_channels) == 0:
        return raw
    n_ch = raw.shape[0]
    good_mask = np.ones(n_ch, dtype=bool)
    good_mask[bad_channels] = False
    good_indices = np.where(good_mask)[0]
    for bc in bad_channels:
        above = good_indices[good_indices < bc]
        below = good_indices[good_indices > bc]
        neighbors = []
        if len(above) > 0:
            neighbors.append(above[-1])
        if len(below) > 0:
            neighbors.append(below[0])
        if len(neighbors) > 0:
            raw[bc] = raw[neighbors].mean(axis=0)
    return raw


def manual_preprocess(raw, fs):
    sos_hp = butter(3, 1 / (fs / 2), btype='high', output='sos')
    raw = sosfiltfilt(sos_hp, raw, axis=1)
    raw = raw - raw.mean(axis=0, keepdims=True)  # CAR
    return raw.astype('float32')


def preprocess(raw, fs, n_ch):
    if n_ch == 384:
        return destripe_lfp(raw, fs=fs, neuropixel_version=1)
    else:
        return manual_preprocess(raw, fs)


def downsample(raw, native_fs, target_fs):
    """Decimate by integer factor, handling non-exact rates (e.g. 2499.99 Hz)."""
    q = round(native_fs / target_fs)  # should be 2
    return decimate(raw, q=q, axis=1, zero_phase=True).astype('float32')


# ── One block: same steps as Lawrence's per-probe body ─────────────────────────
def process_block(sr, b, pid, eid, probe_name, acronyms):
    s0 = int((T_START + b * T_DUR) * sr.fs)
    s1 = int((T_START + (b + 1) * T_DUR) * sr.fs)
    s1 = min(s1, sr.ns)  # clip to actual recording length
    raw = sr[s0:s1, :-sr.nsync].T.astype('float32')  # (n_ch, n_samples)
    n_ch = raw.shape[0]

    bad_ch = detect_bad_channels(raw, sr.fs)
    if len(bad_ch) > 0:
        raw = interpolate_bad_channels(raw, bad_ch)

    raw = preprocess(raw, sr.fs, n_ch)
    raw = downsample(raw, sr.fs, TARGET_FS)

    if acronyms is None:
        acronyms = np.array(['unknown'] * n_ch)

    chunk_len = CHUNK_SIZE * TARGET_FS   # 3750 samples
    n_chunks = raw.shape[1] // chunk_len
    samples = []
    for local_idx in range(n_chunks):
        t0 = local_idx * chunk_len
        chunk = raw[:, t0:t0 + chunk_len]  # (n_ch, 3750)
        chunk_idx = b * CHUNKS_PER_BLOCK + local_idx
        for ch_idx in range(n_ch):
            region = str(acronyms[ch_idx]).strip()
            if region in outside:
                continue
            signal = chunk[ch_idx].copy()  # float32[3750]
            chan_id = f'{pid[:8]}-ch{ch_idx:04d}'
            meta_str = f'{eid}_{chunk_idx}_{probe_name}_{chan_id}_{region}'
            samples.append((signal, meta_str))
    return samples, dict(n_ch=n_ch, n_bad=int(len(bad_ch)), n_chunks=int(n_chunks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, help='raw_lf_manifest JSON (key -> {cbin, eid, pname, split, ...})')
    ap.add_argument('--cbin-root-from', default='', help='prefix in manifest cbin paths to replace')
    ap.add_argument('--cbin-root-to', default='', help='local prefix to substitute')
    ap.add_argument('--pid-csv', required=True, help='pid_eid_new.csv (eid, pid, probe_name)')
    ap.add_argument('--npz', required=True, help='probe_channels.npz ({pid}__acronyms)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--keys', nargs='*', default=None, help='manifest keys to process (default: all in --splits)')
    ap.add_argument('--splits', nargs='*', default=['train', 'val'])
    ap.add_argument('--max-blocks', type=int, default=-1, help='-1 = whole recording')
    ap.add_argument('--task-id', type=int, default=int(os.environ.get('TASK_ID', 0)))
    ap.add_argument('--n-tasks', type=int, default=int(os.environ.get('N_TASKS', 1)))
    ap.add_argument('--report', default=None)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    df = pd.read_csv(args.pid_csv)
    pid_of = {(r.eid, r.probe_name): r.pid for r in df.itertuples()}
    npz = np.load(args.npz, allow_pickle=True)

    keys = args.keys if args.keys else sorted(k for k, v in man.items() if v['split'] in args.splits)
    keys = keys[args.task_id::args.n_tasks]
    print(f'Task {args.task_id}/{args.n_tasks}: {len(keys)} probes', flush=True)

    out_dir = Path(args.out_dir)
    report = []
    for i, key in enumerate(keys):
        e = man[key]
        eid, probe_name = e['eid'], e['pname']
        pid = pid_of[(eid, probe_name)]
        cbin = e['cbin'].replace(args.cbin_root_from, args.cbin_root_to, 1) if args.cbin_root_from else e['cbin']
        acr_key = f'{pid}__acronyms'
        acronyms = npz[acr_key] if acr_key in npz.files else None
        try:
            sr = spikeglx.Reader(cbin)
            n_blocks = int(np.ceil(max(sr.ns / sr.fs - T_START, 0) / T_DUR))
            if args.max_blocks > 0:
                n_blocks = min(n_blocks, args.max_blocks)
            print(f'[{i+1}/{len(keys)}] {key} pid={pid[:8]} fs={sr.fs:.2f} nc={sr.nc} nsync={sr.nsync} '
                  f'dur={sr.ns/sr.fs/60:.1f} min -> {n_blocks} blocks', flush=True)
            for b in range(n_blocks):
                out_path = out_dir / f'b{b:02d}' / f'{eid}_{probe_name}.pickle'
                if out_path.exists():  # resumable, like Lawrence's script
                    print(f'   b{b:02d} SKIP (exists)', flush=True)
                    continue
                t0 = time.time()
                samples, info = process_block(sr, b, pid, eid, probe_name, acronyms)
                if not samples:
                    print(f'   b{b:02d} no full 3 s chunk, nothing written', flush=True)
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_path.with_suffix('.pickle.tmp')
                with open(tmp, 'wb') as f:
                    pickle.dump(samples, f, protocol=4)
                os.replace(tmp, out_path)
                dt = time.time() - t0
                rec = dict(key=key, pid=pid, block=b, n_samples=len(samples), wall_s=dt,
                           bytes=out_path.stat().st_size, **info)
                report.append(rec)
                print(f'   b{b:02d} {info["n_chunks"]} chunks x {info["n_ch"]} ch '
                      f'(bad {info["n_bad"]}) -> {len(samples):,} samples, '
                      f'{rec["bytes"]/1e9:.2f} GB, {dt:.0f} s', flush=True)
                del samples
        except Exception as exc:  # noqa: BLE001  (Lawrence skipped failed probes too)
            import traceback
            print(f'  {key} FAILED: {exc}', flush=True)
            traceback.print_exc()
            report.append(dict(key=key, pid=pid, error=str(exc)))

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        json.dump(report, open(args.report, 'w'), indent=1)
    print('Done!', flush=True)


if __name__ == '__main__':
    main()
