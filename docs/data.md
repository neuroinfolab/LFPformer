# Data preparation

All coordinates in this release are `(AP, DV, ML)` in micrometres. Input signals are single-channel, 3-second windows (3,750 samples at 1,250 Hz). An atlas label map, channel coordinate map and subject/session split are scientific inputs; they are not inferred from test predictions. Download the recordings through the respective IBL ONE, Allen SDK or NeuroNexus data access process. Record dataset versions and checksums alongside each experiment. Recordings, credentials and weights are not shipped here.

## IBL full recordings

Create `data/recordings.json` for your recordings and provide session-to-subject maps for subject-separated training.

A manifest is a JSON object keyed by a recording name, for example:

```json
{
  "recording01": {
    "cbin": "data/raw/recording01.lf.cbin",
    "eid": "SESSION_UUID",
    "pname": "probe00",
    "pid": "SESSION_UUID_probe00",
    "subject": "animal01",
    "lab": "lab01",
    "split": "train"
  }
}
```

A session-to-subject map is a JSON object such as
`{"SESSION_UUID": "animal01"}`. Supply maps covering all sessions used in
training/validation and evaluation to the fusion preparation command.

The preserved `raw_data_chunck_fullrec.py` reads SpikeGLX LF `.cbin` files with their companion metadata. The run manifest maps a recording key to `cbin`, `eid`, `pname`, `split` and recording metadata. `pid_eid_new.csv` maps `eid,pid,probe_name`; `probe_channels.npz` supplies `{pid}__acronyms`. Assign train/validation/test splits before extracting windows.

```bash
lfpformer reference ibl-preprocess -- --manifest data/recordings.json \
  --pid-csv data/pid_eid_new.csv --npz data/probe_channels.npz \
  --out-dir data/ibl_blocks --splits train val --max-blocks -1
lfpformer reference ibl-memmap -- --blocks-root data/ibl_blocks \
  --channel-map data/atlas_index/channel_map.pkl --out-root data/ibl_fullrec \
  --split train --probes EID_probe00
lfpformer reference ibl-finalize -- --out-root data/ibl_fullrec --split train \
  --channel-map data/atlas_index/channel_map.pkl --metadata data/atlas_index/metadata.pkl
```

Replace `EID_probe00` and the block root with the actual output recording names and block directory. Repeat conversion/finalization for validation, then create the small deterministic validation subset with `lfpformer reference ibl-val-subset -- --root data/ibl_fullrec`. The reference `build_ibl_channels_csv.py`, `build_ibl_hierarchy.py` and `build_global_label_map.py` preserve channel/atlas metadata construction. The older `ibl-compact` and `ibl-eval-split` routes preserve the indexed compact-dataset builders; inspect their `--help` because their historical defaults also mention Allen directories.

The full-recording pipeline starts at 30 seconds and processes 600-second blocks. It applies IBL LFP destriping for a 384-channel probe, identifies/interpolates dead channels, and decimates 2,500 Hz to 1,250 Hz with zero-phase filtering. Its non-384-channel branch uses a 1-Hz high-pass filter and common-average reference. Incomplete windows and channels outside the channel map are excluded. Preserve these branches when reproducing a run rather than adding an extra preprocessing pass.

Memory-map parts contain `signals.f32`, `coords.f32`, `fine_id.i32`, `chunk_idx.i32`, `ch_idx.i16`, `region_idx.u16` and `part.json`. `finalize_memmap_split.py` writes the index consumed by the included `CompactDataset`. `build_manip_memmap.py` supplies aligned manipulation coordinates for the full-recording representation; the compact equivalent is `precompute_manip_xyz_compact.py`.

## Compact pickle format

Each shard is a list of `(signal, meta_str, raw_label[, xyz])`. `compact_index.pkl` contains `file_list`, `cumulative_counts` **including a leading zero**, and `n_samples`. The fixed atlas label maps are in `metadata.pkl` in the dataset root or split directory. A common layout is:

```text
data_root/
  metadata.pkl
  train/{compact_index.pkl,samples_000000.pkl,...}
  val/{compact_index.pkl,samples_000000.pkl,...}
```

Historical IBL metadata strings encode session, temporal chunk, probe, channel and region. Preserve them when generating sequence caches. Never reconstruct a physical-channel grouping using array row number.

## Small-array interface

`lfpformer chunk` accepts a preprocessed `channels × time` `.npy`, emits time-major then channel-major windows, and drops incomplete trailing windows. It preserves `probe_ids`, `channel_ids` and `sample_starts`. This convenience command materializes windows in memory. Use the block/memory-map scripts for full recordings.

A 500,000-window evaluation sample is different from evaluation over all 119 probe recordings. The number of probes alone does not specify the number of windows evaluated. Record both counts and the selected window indices. Also distinguish cumulative training sample presentations from unique recordings or windows.
