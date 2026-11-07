from pathlib import Path
from one.api import ONE
from iblatlas.atlas import AllenAtlas
import numpy as np
import pandas as pd
import shutil

CSV_PATH    = Path(__import__('lfpformer.paths',fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/pid_eid_new.csv'))
OUTPUT_PATH = Path(__import__('lfpformer.paths',fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/probe_channels.npz'))
CACHE_DIR   = Path(__import__('lfpformer.paths',fromlist=['legacy_path']).legacy_path('/workspace/ws2/ibl_cache'))

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international',
          cache_dir=CACHE_DIR,
          silent=True)
ba = AllenAtlas(res_um=25)

df   = pd.read_csv(CSV_PATH)
pids = df['pid'].tolist()
eids = df['eid'].tolist()
print(f"Loading channels for {len(pids)} PIDs...")

# Load existing npz if it exists so we can skip already-fetched PIDs
if OUTPUT_PATH.exists():
    existing = np.load(OUTPUT_PATH, allow_pickle=True)
    done_pids = set(k.split('__')[0] for k in existing.files)
    print(f"Already have {len(done_pids)} PIDs, skipping...")
else:
    existing  = {}
    done_pids = set()

all_data = {}

for pid, eid in zip(pids, eids):
    if pid in done_pids:
        continue
    try:
        # Load ONLY channel locations — much lighter than full spike sorting
        channels = one.load_object(eid, 'channels',
                                   collection=f'alf/{df[df.pid==pid].probe_name.values[0]}/pykilosort')

        # Get coordinates
        if 'mlapdv' in channels:
            mlapdv = channels['mlapdv']
        elif all(k in channels for k in ['x', 'y', 'z']):
            mlapdv = np.column_stack([channels['x'], channels['y'], channels['z']])
        else:
            print(f"  {pid[:8]}... FAILED: no coords. Keys: {list(channels.keys())}")
            continue

        # Get region acronyms
        if 'acronym' in channels:
            acronyms = np.array(channels['acronym'], dtype=str)
        elif 'brainLocationIds_ccf_2017' in channels:
            acronyms = ba.regions.id2acronym(channels['brainLocationIds_ccf_2017'])
        else:
            acronyms = np.array(['unknown'] * mlapdv.shape[0])

        # With a more robust acronym fetch:
        if 'acronym' in channels:
            acronyms = np.array(channels['acronym'], dtype=str)
        elif 'brainLocationIds_ccf_2017' in channels:
            ids = channels['brainLocationIds_ccf_2017']
            acronyms = np.array(ba.regions.id2acronym(ids), dtype=str)
        elif 'atlas_id' in channels:
            ids = channels['atlas_id']
            acronyms = np.array(ba.regions.id2acronym(ids), dtype=str)
        else:
            print(f"  {pid[:8]}... no acronym key found. Available: {list(channels.keys())}")
            acronyms = np.array(['unknown'] * mlapdv.shape[0])

        all_data[pid] = {
            'mlapdv':   mlapdv.astype('float32'),
            'acronyms': acronyms,
        }
        print(f"  {pid[:8]}... -> {mlapdv.shape[0]} channels ✓")

        # Clean up cache for this session to save disk space
        eid_path = one.eid2path(eid)
        if eid_path and eid_path.exists():
            shutil.rmtree(eid_path)

    except Exception as e:
        print(f"  {pid[:8]}... FAILED: {e}")

# Merge with existing data and save
save_dict = {}
if existing:
    save_dict.update({k: existing[k] for k in existing.files})
save_dict.update({
    f"{pid}__{key}": val
    for pid, d in all_data.items()
    for key, val in d.items()
})

np.savez(OUTPUT_PATH, **save_dict)
print(f"\nSaved {len([k for k in save_dict if k.endswith('__mlapdv')])} total PIDs to {OUTPUT_PATH}")