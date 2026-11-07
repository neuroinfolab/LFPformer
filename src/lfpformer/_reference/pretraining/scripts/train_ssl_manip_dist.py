#!/usr/bin/env python3
"""SSL + ManipDistc launcher.

Same recipe as CCFDistc (train_ssl.py + wav2vec2_ibl_v2_*_ssl_ccf_dist_fresh.yaml)
but replaces CompactDataset histology CCF coords (batch[2]) with micro-manipulator
xyz from MANIP_XYZ_DIR/{train,val}_manip_xyz.npy. Samples without manip are dropped.

Env:
  MANIP_XYZ_DIR  required — precompute_manip_xyz_compact.py output

Launch note:
  Running this file puts its directory on sys.path[0]. That would shadow PL's
  train_ssl with the SD copy (which passes channels_csv=). We re-order sys.path
  so PL scripts win, matching the CCFDistc fair-match recipe, and also strip
  kwargs the parent CompactDataset does not accept.
"""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import torch

MANIP_XYZ_DIR = os.environ.get("MANIP_XYZ_DIR", "")
if not MANIP_XYZ_DIR or not os.path.isdir(MANIP_XYZ_DIR):
    raise SystemExit(
        f"MANIP_XYZ_DIR must point to precomputed manip xyz dir (got {MANIP_XYZ_DIR!r})"
    )

# Prefer PYTHONPATH entries (PL train_ssl + adaptive backbone) over this script's dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_HERE]

# Patch CompactDataset BEFORE train_ssl imports it inside its train() body.
import blind_localization.data.lazyloader_dataset as ll  # noqa: E402

_OrigCompact = ll.CompactDataset
_ORIG_INIT_PARAMS = set(inspect.signature(_OrigCompact.__init__).parameters)


class ManipCoordCompactDataset(_OrigCompact):
    """CompactDataset with histology CCF replaced by manip xyz; invalid rows dropped."""

    def __init__(self, compact_dir, *args, **kwargs):
        # SD train_ssl may pass channels_csv=; PL CompactDataset rejects it.
        filtered = {k: v for k, v in kwargs.items() if k in _ORIG_INIT_PARAMS}
        dropped = sorted(set(kwargs) - set(filtered))
        if dropped:
            print(f"[ManipDistc] dropping unsupported CompactDataset kwargs: {dropped}")
        super().__init__(compact_dir, *args, **filtered)
        split = os.path.basename(os.path.abspath(compact_dir.rstrip(os.sep)))
        xyz_path = os.path.join(MANIP_XYZ_DIR, f"{split}_manip_xyz.npy")
        valid_path = os.path.join(MANIP_XYZ_DIR, f"{split}_valid.npy")
        if not os.path.isfile(xyz_path) or not os.path.isfile(valid_path):
            raise FileNotFoundError(
                f"missing manip arrays for split={split}: {xyz_path} / {valid_path}"
            )
        # GCP full-recording run: memory-map (257M x 3 float32 = 3 GB) so all ranks share the page cache
        manip = np.load(xyz_path, mmap_mode="r")
        if manip.dtype != np.float32:
            manip = manip.astype(np.float32)
        valid = np.load(valid_path, mmap_mode="r").astype(bool)
        if len(manip) != super().__len__():
            raise RuntimeError(
                f"manip len {len(manip)} != CompactDataset len {super().__len__()} "
                f"(split={split}). Re-run precompute on this compact root."
            )
        self._manip_xyz = manip
        self._valid_idx = np.flatnonzero(valid)
        if len(self._valid_idx) == 0:
            raise RuntimeError(f"no valid manip samples in split={split}")
        print(
            f"[ManipDistc] {split}: using {len(self._valid_idx)}/{super().__len__()} "
            f"samples with manip xyz ({100.0 * len(self._valid_idx) / super().__len__():.1f}%)"
        )

    def __len__(self):
        return int(len(self._valid_idx))

    def __getitem__(self, idx):
        real = int(self._valid_idx[idx])
        sample = super().__getitem__(real)
        # CompactDataset may return 3-tuple (x, y, coords) or longer.
        x, y = sample[0], sample[1]
        coords = torch.from_numpy(np.array(self._manip_xyz[real], dtype=np.float32))
        if len(sample) == 3:
            return x, y, coords
        return (x, y, coords) + tuple(sample[3:])


ll.CompactDataset = ManipCoordCompactDataset

# Run PL train_ssl with same CLI (sbatch puts PL scripts first on PYTHONPATH).
import train_ssl  # noqa: E402

if __name__ == "__main__":
    print(f"[ManipDistc] MANIP_XYZ_DIR={MANIP_XYZ_DIR}")
    print(f"[ManipDistc] train_ssl={getattr(train_ssl, '__file__', '?')}")
    print(f"[ManipDistc] CompactDataset base={_OrigCompact}")
    print("[ManipDistc] CompactDataset patched → manip xyz coords (global CCF-mode loss)")
    sys.argv[0] = "train_ssl_manip_dist.py"
    train_ssl.main()
