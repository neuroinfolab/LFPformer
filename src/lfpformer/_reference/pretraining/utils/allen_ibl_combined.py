"""Allen-compatible wrappers for combined IBL + Allen SSL training.

The Allen and IBL compact datasets share the SAME on-disk format
(`compact_index.pkl` + `samples_XXXXXX.pkl` with tuples of
`(signal, meta_str, hierarchy_dict, coords)`), so our existing
`CompactDataset` reads Allen without changes.

The only real difference is the channel identifier inside `meta_str`:
  - IBL:    `eid_idx_probeNN_pid-chNNNN_region`  → channel_idx via `-ch`
  - Allen:  `session_timeidx_probe_channelID_region`  → channel_id is an
            Allen DB id (~10-digit int) that maps to a 0-383 local slot
            via `channels.csv` (`id` → `local_index`).

Allen uses the same NP1.0 4-column geometry (x ∈ {11,27,43,59} µm, 20 µm
row pitch), so once we have the 0-383 local index our existing
`ChannelDistanceContrastiveLoss` and `channel_position()` are unchanged.
The `SameProbeBatchSampler`'s probe-key convention `parts[0]_parts[2]`
also works for both (IBL: eid_probeNN, Allen: session_probeID) — no
override needed.
"""

import os
import pickle
import pandas as pd
import torch

from Lfp2vec_benchmarks.utils.probe_distance_sampler import (
    ChannelAwareDataset,
    SameProbeBatchSampler,
)


def load_allen_channel_map(csv_path):
    """Allen channel DB `id` -> 0-383 `local_index`. Call once at init."""
    df = pd.read_csv(csv_path, usecols=["id", "local_index"])
    return dict(zip(df["id"].astype(int), df["local_index"].astype(int)))


class AllenChannelAwareDataset(ChannelAwareDataset):
    """Same interface as ChannelAwareDataset, but resolves Allen channel DB
    ids via `channels.csv` instead of the IBL `-ch` suffix.
    """

    def __init__(self, base_dataset, allen_channels_csv):
        self._channel_map = load_allen_channel_map(allen_channels_csv)
        super().__init__(base_dataset)  # calls self._build_channel_index

    def _build_channel_index(self):
        compact_dir = self.base.compact_dir
        idx_data = pickle.load(open(os.path.join(compact_dir, "compact_index.pkl"), "rb"))
        channels = []
        n_missing = 0
        for fid in range(idx_data["n_files"]):
            fpath = os.path.join(compact_dir, f"samples_{fid:06d}.pkl")
            samples = pickle.load(open(fpath, "rb"))
            for s in samples:
                meta = s[1] if len(s) >= 2 and isinstance(s[1], str) else ""
                parts = meta.split("_")
                local_idx = -1
                if len(parts) >= 4:
                    try:
                        ch_id = int(parts[3])
                        local_idx = self._channel_map.get(ch_id, -1)
                    except ValueError:
                        pass
                if local_idx < 0:
                    n_missing += 1
                channels.append(local_idx)
        if n_missing:
            print(f"[AllenChannelAwareDataset] {n_missing} samples with "
                  f"unresolved channel id (of {len(channels)})")
        return channels


def build_combined_train_loaders(
    ibl_base_dataset,
    allen_base_dataset,
    allen_channels_csv,
    samples_per_probe=16,
    num_probes=4,
    num_workers=4,
    seed=42,
):
    """Two DataLoaders — one for IBL, one for Allen — that yield same-probe
    batches of size `num_probes * samples_per_probe` each. Train loop
    alternates between them.

    Returns (ibl_loader, allen_loader).
    """
    from torch.utils.data import DataLoader

    ibl_train = ChannelAwareDataset(ibl_base_dataset)
    allen_train = AllenChannelAwareDataset(allen_base_dataset, allen_channels_csv)

    ibl_sampler = SameProbeBatchSampler(
        ibl_base_dataset, samples_per_probe=samples_per_probe,
        num_probes=num_probes, seed=seed,
    )
    allen_sampler = SameProbeBatchSampler(
        allen_base_dataset, samples_per_probe=samples_per_probe,
        num_probes=num_probes, seed=seed + 1,
    )

    ibl_loader = DataLoader(
        ibl_train, batch_sampler=ibl_sampler,
        num_workers=num_workers, pin_memory=True,
    )
    allen_loader = DataLoader(
        allen_train, batch_sampler=allen_sampler,
        num_workers=num_workers, pin_memory=True,
    )
    return ibl_loader, allen_loader


def interleave_loaders(ibl_loader, allen_loader):
    """Round-robin iterator: yield (source_tag, batch) alternating between
    IBL and Allen. Stops when both loaders are exhausted.

    If one loader is shorter, its iterator is restarted so the longer one
    continues to be consumed — each source runs through its full epoch.
    """
    iters = {"ibl": iter(ibl_loader), "allen": iter(allen_loader)}
    done = {"ibl": False, "allen": False}
    tag = "ibl"
    while not all(done.values()):
        if done[tag]:
            tag = "allen" if tag == "ibl" else "ibl"
            continue
        try:
            yield tag, next(iters[tag])
        except StopIteration:
            done[tag] = True
            continue
        tag = "allen" if tag == "ibl" else "ibl"
