"""Channel Distance Contrastive — custom batch sampler and loss.

Groups samples from the SAME probe into each batch, enabling a spatial
contrastive loss based on channel distances along the probe shaft.

Neuropixels 1.0 electrode geometry (all 699 IBL BWM probes):
  - 960 electrode sites total, 384 recorded simultaneously (bank 0)
  - 4 columns at x = 11, 27, 43, 59 µm
  - 2 sites per row, alternating between column pairs:
      Even rows (0, 2, 4...): x = 11, 43 µm  (columns 0, 2)
      Odd rows  (1, 3, 5...): x = 27, 59 µm  (columns 1, 3)
  - 192 rows, 20 µm vertical pitch between rows
  - Total probe span: ~3820 µm vertically, 48 µm horizontally
  - Channel numbering: ch = row * 2 + col_in_row (col_in_row: 0=left, 1=right)

  Row 0: ch0 @ (11, 0)    ch1 @ (43, 0)
  Row 1: ch2 @ (27, 20)   ch3 @ (59, 20)
  Row 2: ch4 @ (11, 40)   ch5 @ (43, 40)
  Row 3: ch6 @ (27, 60)   ch7 @ (59, 60)
  ...

Channels far apart on the probe record from different brain regions.
Channels close together record from the same local circuit.
This is purely hardware-based — NO atlas coordinates needed.

Usage:
    sampler = SameProbeBatchSampler(dataset, samples_per_probe=16, num_probes=4)
    loader = DataLoader(dataset, batch_sampler=sampler)

    loss_fn = ChannelDistanceContrastiveLoss(sigma=1000.0, temperature=0.2,
                                              samples_per_probe=16)
    loss = loss_fn(embeddings, channel_numbers)
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Sampler


def parse_channel_id(meta_str):
    """Extract channel number from meta string.

    Meta format: eid_chunk_probe_channel_region
    Channel format: xxxx-chNNNN (e.g., '2aea57ac-ch0368')

    Returns channel number as int, or -1 if unparseable.
    """
    parts = meta_str.split("_")
    if len(parts) < 4:
        return -1
    ch_part = parts[3]  # e.g., '2aea57ac-ch0368'
    if "-ch" in ch_part:
        try:
            return int(ch_part.split("-ch")[1])
        except (ValueError, IndexError):
            return -1
    return -1


def channel_position(ch):
    """Get (x, y) position in µm for a Neuropixels 1.0 channel.

    Layout: 4 columns at x = 11, 27, 43, 59 µm.
    Even rows use columns 0, 2 (x=11, 43). Odd rows use columns 1, 3 (x=27, 59).
    2 channels per row, 20 µm vertical pitch, 192 rows for 384 channels.
    """
    row = ch // 2
    col_in_row = ch % 2  # 0 = left site, 1 = right site
    if row % 2 == 0:  # even row: columns 0, 2
        x = 11.0 if col_in_row == 0 else 43.0
    else:              # odd row: columns 1, 3
        x = 27.0 if col_in_row == 0 else 59.0
    y = row * 20.0
    return x, y


def channel_distance(ch_i, ch_j):
    """Compute physical distance between two Neuropixels 1.0 channels.

    Uses the 4-column geometry: x = {11, 27, 43, 59} µm, 20 µm row pitch.

    Args:
        ch_i, ch_j: channel numbers (0-383)

    Returns:
        distance in µm
    """
    xi, yi = channel_position(ch_i)
    xj, yj = channel_position(ch_j)
    return np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)


class SameProbeBatchSampler(Sampler):
    """Batch sampler that groups samples by probe for channel distance loss.

    Supports two modes:
    - Single probe: each batch = 1 probe × samples_per_probe samples
    - Multi-probe: each batch = num_probes × samples_per_probe samples
      (e.g., 4 probes × 16 = 64 total, matching pure SSL batch size)

    Multi-probe mode keeps the same total batch size as pure SSL training
    so that cross-sample negatives have enough diversity. Channel distance
    loss is computed within each probe group separately.

    Args:
        dataset: CompactDataset with meta strings containing probe/channel info
        samples_per_probe: samples from each probe per batch
        num_probes: number of probes per batch (default 1)
        seed: random seed
    """

    def __init__(self, dataset, samples_per_probe=16, num_probes=1, seed=42):
        self.dataset = dataset
        self.samples_per_probe = samples_per_probe
        self.num_probes = num_probes
        self.batch_size = samples_per_probe * num_probes
        self.rng = np.random.RandomState(seed)

        # Build index: probe_key → [sample indices]
        self.probe_index = defaultdict(list)
        self._build_index()

        # Filter to probes with enough samples
        self.valid_probes = [
            k for k, v in self.probe_index.items()
            if len(v) >= samples_per_probe
        ]

        if not self.valid_probes:
            # Fallback: use probes with at least half the required samples
            min_samples = max(2, samples_per_probe // 2)
            self.valid_probes = [
                k for k, v in self.probe_index.items()
                if len(v) >= min_samples
            ]

        total_samples = sum(len(self.probe_index[k]) for k in self.valid_probes)
        self.n_batches = max(1, total_samples // self.batch_size)

        print(f"[SameProbeBatchSampler] {len(self.valid_probes)} valid probes "
              f"(from {len(self.probe_index)} total), "
              f"{num_probes} probes/batch × {samples_per_probe} samples/probe "
              f"= {self.batch_size} batch_size, ~{self.n_batches} batches/epoch")

    def _build_index(self):
        """Scan compact files to build probe → sample index."""
        import pickle as _pkl
        import os

        compact_dir = self.dataset.compact_dir if hasattr(self.dataset, 'compact_dir') else None
        if compact_dir is None:
            print("[SameProbeBatchSampler] WARNING: no compact_dir")
            return

        idx_data = _pkl.load(open(os.path.join(compact_dir, 'compact_index.pkl'), 'rb'))
        sample_idx = 0

        for fid in range(idx_data['n_files']):
            fpath = os.path.join(compact_dir, f'samples_{fid:06d}.pkl')
            samples = _pkl.load(open(fpath, 'rb'))

            for s in samples:
                meta = s[1] if len(s) >= 2 and isinstance(s[1], str) else ""
                parts = meta.split("_")
                if len(parts) >= 3:
                    eid = parts[0]
                    probe_name = parts[2]
                    probe_key = f"{eid}_{probe_name}"
                    self.probe_index[probe_key].append(sample_idx)
                sample_idx += 1

            if (fid + 1) % 100 == 0:
                print(f"  ... indexed {fid+1}/{idx_data['n_files']} files, "
                      f"{sample_idx} samples, {len(self.probe_index)} probes")

    def __iter__(self):
        for _ in range(self.n_batches):
            # Select num_probes different probes
            selected_probes = self.rng.choice(
                self.valid_probes, self.num_probes, replace=False
            ) if self.num_probes <= len(self.valid_probes) else self.rng.choice(
                self.valid_probes, self.num_probes, replace=True
            )

            batch_indices = []
            for probe_key in selected_probes:
                indices = self.probe_index[probe_key]
                if len(indices) >= self.samples_per_probe:
                    selected = self.rng.choice(indices, self.samples_per_probe, replace=False)
                else:
                    selected = self.rng.choice(indices, self.samples_per_probe, replace=True)
                batch_indices.extend(selected.tolist())

            yield batch_indices

    def __len__(self):
        return self.n_batches


class DistributedSameProbeBatchSampler(SameProbeBatchSampler):
    """DDP-aware variant of SameProbeBatchSampler.

    Splits each *global* same-probe batch across ranks so the union of all
    ranks reconstructs the original ``num_probes × samples_per_probe`` batch
    composition, with each rank owning a DISJOINT slice of probes:

        global batch = num_probes probes  →  probes_per_rank = num_probes // world_size
        rank r gets probes[r*ppr : (r+1)*ppr]   (ppr ≥ 1)

    Why this shape (and not just "different seed per rank"):
    - The channel-distance loss is computed WITHIN each probe group. Giving
      each rank its own distinct probes means every rank computes a valid CD
      loss over a full probe group; DDP then averages the per-rank gradients,
      which is equivalent to one larger multi-probe batch.
    - DDP does not gather activations across ranks, so SSL cross-sample
      negatives are drawn from this rank's per-rank batch only (e.g. 2 probes
      × 16 = 32 samples on 2 GPUs). This matches train_ssl.py's DDP behavior
      (per-rank batch = global // world_size) — an accepted DDP trade-off.

    RNG lockstep: every rank uses the SAME ``seed + epoch`` RNG and performs
    ALL the global random draws (probe choice + within-probe sample choice for
    every probe in the batch). Each rank only *keeps* the indices for its own
    probe slice. Because all ranks draw identically, the RNG stays in sync and
    the per-rank slices are guaranteed disjoint + jointly equal to the original
    batch. ``n_batches`` is unchanged, so all ranks do the same number of steps
    (and thus the same number of DDP all-reduces) per epoch.

    Call ``set_epoch(epoch)`` once per epoch (like ``DistributedSampler``).
    """

    def __init__(self, dataset, samples_per_probe=16, num_probes=4, seed=42,
                 num_replicas=1, rank=0):
        super().__init__(dataset, samples_per_probe=samples_per_probe,
                         num_probes=num_probes, seed=seed)
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.probes_per_rank = max(num_probes // self.num_replicas, 1)
        # Total probes drawn per batch globally (== num_probes when divisible).
        self.total_probes = self.probes_per_rank * self.num_replicas
        self.per_rank_batch = self.probes_per_rank * samples_per_probe
        if rank == 0:
            print(f"[DistributedSameProbeBatchSampler] world_size={self.num_replicas}, "
                  f"probes_per_rank={self.probes_per_rank}, total_probes/batch={self.total_probes} "
                  f"(orig num_probes={num_probes}), per_rank_batch={self.per_rank_batch}, "
                  f"n_batches={self.n_batches}")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        # Identical RNG on every rank → identical global draws → RNG stays in
        # lockstep even though each rank keeps only its own probe slice.
        g = np.random.RandomState(self.seed + self.epoch)
        lo = self.rank * self.probes_per_rank
        hi = lo + self.probes_per_rank

        for _ in range(self.n_batches):
            replace_probes = self.total_probes > len(self.valid_probes)
            selected_probes = g.choice(self.valid_probes, self.total_probes,
                                       replace=replace_probes)

            batch_indices = []
            for p_i, probe_key in enumerate(selected_probes):
                indices = self.probe_index[probe_key]
                if len(indices) >= self.samples_per_probe:
                    samp = g.choice(indices, self.samples_per_probe, replace=False)
                else:
                    samp = g.choice(indices, self.samples_per_probe, replace=True)
                # Keep only the draws belonging to THIS rank's probe slice; the
                # draw itself still happens on every rank so the RNG advances
                # identically across ranks.
                if lo <= p_i < hi:
                    batch_indices.extend(samp.tolist())

            yield batch_indices

    def __len__(self):
        return self.n_batches


class ChannelAwareDataset(torch.utils.data.Dataset):
    """Wrapper that adds channel number to each sample.

    Wraps an existing CompactDataset and appends the parsed channel number
    to each sample tuple. This avoids modifying the original dataset class.

    Returns: (signal, label, coords, channel_number) or (signal, label, channel_number)
    """

    def __init__(self, base_dataset):
        self.base = base_dataset
        # Pre-parse channel numbers from compact files for speed
        self._channel_numbers = self._build_channel_index()

    def _build_channel_index(self):
        """Parse channel numbers from all samples."""
        import pickle as _pkl
        import os

        compact_dir = self.base.compact_dir if hasattr(self.base, 'compact_dir') else None
        if compact_dir is None:
            return None

        idx_data = _pkl.load(open(os.path.join(compact_dir, 'compact_index.pkl'), 'rb'))
        channels = []

        for fid in range(idx_data['n_files']):
            fpath = os.path.join(compact_dir, f'samples_{fid:06d}.pkl')
            samples = _pkl.load(open(fpath, 'rb'))
            for s in samples:
                meta = s[1] if len(s) >= 2 and isinstance(s[1], str) else ""
                channels.append(parse_channel_id(meta))

        return channels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        ch = self._channel_numbers[idx] if self._channel_numbers else -1
        # Append channel number to the sample tuple
        if isinstance(sample, tuple):
            return sample + (ch,)
        return (sample, ch)


class ChannelDistanceContrastiveLoss(torch.nn.Module):
    """Soft distance-weighted contrastive loss based on channel distance.

    For each pair of samples from the SAME probe, computes target similarity
    from hardware channel distance and pushes embedding similarity to match
    via KL-divergence. Cross-probe pairs are ignored.

    Supports multi-probe batches: if the batch contains samples from N probes
    (each with samples_per_probe samples), loss is computed within each probe
    group and averaged. Cross-probe pairs do not contribute to the loss.

    L = KL(target_distribution || predicted_distribution)

    where:
        predicted = softmax(cosine_sim(emb_i, emb_j) / temperature)  [within probe]
        target = softmax(exp(-channel_dist² / 2σ²))                  [within probe]

    Args:
        sigma: Gaussian kernel width in µm (controls spatial scale)
            - σ=500: only very nearby channels (~25 channels apart) are similar
            - σ=1000: channels within ~50 rows are similar
            - σ=2000: most channels on same probe are somewhat similar
        temperature: temperature for embedding similarity softmax
        samples_per_probe: number of samples per probe in each batch.
            If set, batch is split into groups of this size, and loss is
            computed within each group. If None, entire batch is one probe.
    """

    def __init__(self, sigma=1000.0, temperature=0.2, samples_per_probe=None):
        super().__init__()
        self.sigma = sigma
        self.temperature = temperature
        self.samples_per_probe = samples_per_probe

    def _compute_loss_single_probe(self, embeddings, channel_numbers):
        """Compute loss for a single probe group."""
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Normalize embeddings for cosine similarity
        z = F.normalize(embeddings, dim=-1)

        # Pairwise cosine similarity / temperature
        sim_matrix = z @ z.T / self.temperature  # [B, B]

        # Mask diagonal
        diag_mask = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(diag_mask, float("-inf"))
        log_probs = F.log_softmax(sim_matrix, dim=-1)

        # Target distribution from channel distances (NP1.0 4-column geometry)
        with torch.no_grad():
            ch = channel_numbers.long()
            rows = ch // 2           # row index (0-191)
            col_in_row = ch % 2      # 0=left, 1=right within row

            # X position depends on row parity and col_in_row
            # Even rows: x = 11 (left) or 43 (right)  → columns 0, 2
            # Odd rows:  x = 27 (left) or 59 (right)  → columns 1, 3
            even_row = (rows % 2 == 0)
            x = torch.where(
                even_row,
                torch.where(col_in_row == 0,
                             torch.tensor(11.0, device=ch.device),
                             torch.tensor(43.0, device=ch.device)),
                torch.where(col_in_row == 0,
                             torch.tensor(27.0, device=ch.device),
                             torch.tensor(59.0, device=ch.device)),
            ).float()
            y = rows.float() * 20.0  # 20 µm row pitch

            # Pairwise squared distance
            dx = x.unsqueeze(0) - x.unsqueeze(1)  # [B, B]
            dy = y.unsqueeze(0) - y.unsqueeze(1)
            dist_sq = dx ** 2 + dy ** 2

            # Gaussian kernel
            soft_target = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))
            soft_target.masked_fill_(diag_mask, 0.0)

            # Normalize to distribution
            row_sums = soft_target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            soft_target = soft_target / row_sums

        # KL divergence
        log_probs = log_probs.masked_fill(diag_mask, 0.0)
        loss = -(soft_target * log_probs).sum(dim=-1).mean()

        return loss

    def forward(self, embeddings, channel_numbers):
        """Compute channel distance contrastive loss.

        Args:
            embeddings: [B, D] mean-pooled encoder output
            channel_numbers: [B] tensor of channel numbers (0-383)

        Returns:
            loss: scalar (averaged across probe groups if multi-probe)
        """
        spp = self.samples_per_probe
        if spp is None or spp >= embeddings.shape[0]:
            # Single probe — entire batch is one group
            return self._compute_loss_single_probe(embeddings, channel_numbers)

        # Multi-probe: split batch into probe groups and average loss
        B = embeddings.shape[0]
        num_groups = B // spp
        losses = []
        for g in range(num_groups):
            start = g * spp
            end = start + spp
            loss_g = self._compute_loss_single_probe(
                embeddings[start:end], channel_numbers[start:end]
            )
            losses.append(loss_g)

        return torch.stack(losses).mean()
