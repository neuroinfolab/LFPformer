from one.api import ONE
import pandas as pd
from pathlib import Path

OUTPUT_CSV = Path(__import__('lfpformer.paths',fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/pid_eid_new.csv'))

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)

# Single tag covers all 699 insertions
all_pids = [str(p) for p in one.search_insertions(tag='2024_Q2_IBL_et_al_BWM_iblsort')]
print(f"Total BWM insertions: {len(all_pids)}")  # should be 699

# Load existing CSV to avoid re-fetching
if OUTPUT_CSV.exists():
    existing = pd.read_csv(OUTPUT_CSV)
    known_pids = set(existing['pid'].tolist())
    print(f"Already have {len(known_pids)} PIDs in CSV")
else:
    existing = pd.DataFrame()
    known_pids = set()

# Only fetch metadata for new PIDs
new_pids = [p for p in all_pids if p not in known_pids]
print(f"Fetching metadata for {len(new_pids)} new PIDs...")

rows = []
for pid in new_pids:
    try:
        ins = one.alyx.rest('insertions', 'read', id=pid)
        rows.append({
            'eid':        str(ins['session']),  # also cast eid to str
            'pid':        pid,
            'probe_name': ins['name'],
            'subject':    ins['session_info']['subject'],
            'lab':        ins['session_info']['lab'],
        })
        print(f"  {pid[:8]}... subject={ins['session_info']['subject']} probe={ins['name']}")
    except Exception as e:
        print(f"  {pid[:8]}... FAILED: {e}")

# Append and save
updated = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
updated.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(updated)} total rows to {OUTPUT_CSV}")
print(f"  BWM total:  {len(all_pids)}")
print(f"  In CSV:     {len(updated)}")