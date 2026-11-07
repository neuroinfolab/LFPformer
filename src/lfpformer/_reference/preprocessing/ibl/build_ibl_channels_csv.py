import numpy as np
import pandas as pd
from pathlib import Path
from iblatlas.atlas import AllenAtlas

ba       = AllenAtlas(res_um=25)
NPZ_PATH = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/probe_channels.npz'))
CSV_OUT  = Path(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/channels_ibl.csv'))

data = np.load(NPZ_PATH, allow_pickle=True)
probe_data = {}
for key in data.files:
    pid, field = key.split('__', 1)
    if pid not in probe_data:
        probe_data[pid] = {}
    probe_data[pid][field] = data[key]

outside = {'root', 'void', 'nan', '', 'out of brain', 'x', 'unknown', 'pitching'}

# ── Build acronym → atlas_id lookup ONCE ─────────────────────────────────────
print("Building acronym lookup table...")
acr_to_id = {acr: int(aid) for acr, aid in zip(ba.regions.acronym, ba.regions.id)}
print(f"Lookup table: {len(acr_to_id)} entries")

rows = []
skipped_channels = 0

for pid, d in probe_data.items():
    mlapdv   = d['mlapdv'].astype('float64')
    acronyms = d['acronyms']

    # Convert IBL µm → CCF µm (output cols: ML, AP, DV)
    ccf = ba.xyz2ccf(mlapdv / 1e6, mode='extrapolate')

    for i in range(len(ccf)):
        acr = str(acronyms[i]).strip()

        if acr in outside:
            skipped_channels += 1
            continue

        atlas_id = acr_to_id.get(acr, 0)

        rows.append({
            'id':                                f'{pid[:8]}-ch{i:04d}',
            'anterior_posterior_ccf_coordinate': float(ccf[i, 1]),
            'dorsal_ventral_ccf_coordinate':     float(ccf[i, 2]),
            'left_right_ccf_coordinate':         float(ccf[i, 0]),
            'ecephys_structure_id':              atlas_id,
            'ecephys_structure_acronym':         acr,
            'pid':                               pid,
            'channel_idx':                       i,
        })

df = pd.DataFrame(rows)
df.to_csv(CSV_OUT, index=False)

print(f"Total probes:            {len(probe_data)}")
print(f"Total channels:          {len(rows)}")
print(f"Skipped (outside brain): {skipped_channels}")
print(f"Atlas ID zeros (lookup failed): {(df.ecephys_structure_id == 0).sum()}")
print(f"\nSample rows:")
print(df.head(5).to_string())