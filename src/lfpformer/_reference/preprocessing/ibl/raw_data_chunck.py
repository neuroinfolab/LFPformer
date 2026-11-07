from pathlib import Path
from one.api import ONE
from brainbox.io.spikeglx import Streamer
from ibldsp.voltage import destripe_lfp
import numpy as np
import pandas as pd
import pickle
import os
from scipy.signal import decimate, butter, sosfiltfilt

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH   = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/pid_eid_new.csv'))
NPZ_PATH   = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/probe_channels.npz'))
OUT_DIR    = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/preprocessed_v2'))
T_START    = 30       # skip first 30s
T_DUR      = 600      # 10 minutes
CHUNK_SIZE = 3        # seconds
TARGET_FS  = 1250     # Hz

OUT_DIR.mkdir(parents=True, exist_ok=True)

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international',
          cache_dir=Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/ibl_cache')),
          silent=True)

# ── Load acronyms from npz ────────────────────────────────────────────────────
npz = np.load(NPZ_PATH, allow_pickle=True)
probe_data = {}
for key in npz.files:
    pid, field = key.split('__', 1)
    if pid not in probe_data:
        probe_data[pid] = {}
    probe_data[pid][field] = npz[key]

# ── Helpers ───────────────────────────────────────────────────────────────────
outside = {'root', 'void', 'nan', '', 'out of brain', 'x', 'unknown', 'pitching'}

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

# ── Main loop ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# SLURM array task: each task processes a slice of probes
task_id  = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_tasks  = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
df = df.iloc[task_id::n_tasks].reset_index(drop=True)
print(f"Task {task_id}/{n_tasks}: processing {len(df)} probes")

total = len(df)

for idx, row in df.iterrows():
    pid        = row['pid']
    eid        = row['eid']
    probe_name = row['probe_name']

    # ── Resumable: skip if pickle already exists ──────────────────────────────
    out_path = OUT_DIR / pid / f'{eid}_{probe_name}.pickle'
    if out_path.exists():
        print(f"[{idx+1}/{total}] {pid[:8]} SKIP (already done)")
        continue

    print(f"[{idx+1}/{total}] {pid[:8]} | {probe_name} | eid: {eid[:8]}...")

    try:
        # ── Stream raw LFP ────────────────────────────────────────────────────
        sr   = Streamer(pid=pid, one=one, typ='lf', remove_cached=True)
        s0   = int(T_START * sr.fs)
        s1   = int((T_START + T_DUR) * sr.fs)
        s1   = min(s1, sr.ns)  # clip to actual recording length
        raw  = sr[s0:s1, :-sr.nsync].T.astype('float32')  # (n_ch, n_samples)
        n_ch = raw.shape[0]
        print(f"  Streamed: {raw.shape}, native_fs={sr.fs:.2f} Hz")

        # ── Detect + interpolate bad channels on RAW (before destripe) ────────
        bad_ch = detect_bad_channels(raw, sr.fs)
        if len(bad_ch) > 0:
            raw = interpolate_bad_channels(raw, bad_ch)
            print(f"  Bad channels: {len(bad_ch)} -> {bad_ch.tolist()[:20]}")

        # ── Preprocess (destripe / manual) ────────────────────────────────────
        raw = preprocess(raw, sr.fs, n_ch)

        # ── Downsample to 1250 Hz ─────────────────────────────────────────────
        raw = downsample(raw, sr.fs, TARGET_FS)
        print(f"  Downsampled: {raw.shape}")

        # ── Get acronyms ──────────────────────────────────────────────────────
        if pid in probe_data and 'acronyms' in probe_data[pid]:
            acronyms = probe_data[pid]['acronyms']
        else:
            acronyms = np.array(['unknown'] * n_ch)

        # ── Chunk into 3s windows ─────────────────────────────────────────────
        chunk_len  = CHUNK_SIZE * TARGET_FS   # 3750 samples
        n_chunks   = raw.shape[1] // chunk_len
        samples    = []

        for chunk_idx in range(n_chunks):
            t0 = chunk_idx * chunk_len
            t1 = t0 + chunk_len
            chunk = raw[:, t0:t1]  # (n_ch, 3750)

            for ch_idx in range(n_ch):
                region = str(acronyms[ch_idx]).strip()
                if region in outside:
                    continue

                signal   = chunk[ch_idx].copy()  # float32[3750]
                # channel_id must match channels_ibl.csv 'id' column
                chan_id  = f'{pid[:8]}-ch{ch_idx:04d}'
                meta_str = f'{eid}_{chunk_idx}_{probe_name}_{chan_id}_{region}'
                samples.append((signal, meta_str))

        print(f"  Chunks: {n_chunks} | Samples: {len(samples)}")

        # ── Write pickle ──────────────────────────────────────────────────────
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'wb') as f:
            pickle.dump(samples, f, protocol=4)
        print(f"  Saved -> {out_path}")

    except Exception as e:
        print(f"  {pid[:8]} FAILED: {e}")
        import traceback; traceback.print_exc()

print("Done!")
