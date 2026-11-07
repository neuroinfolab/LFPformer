#!/usr/bin/env python3
"""Extract frozen Allen lp_eval embeddings for the ProbeSeq XYZ decoder.

The generic :mod:`extract_probe_sequences` script is tied to the IBL eval-v2
directory and metadata conventions.  Allen lp_eval instead stores
``train/``, ``val/`` and ``test_fold1/`` ... ``test_fold5/`` and identifies
geometry with Allen global channel IDs.  This extractor writes the identical
flat cache schema expected by ``train_probe_sequence_ext.py`` while preserving
the Allen split boundaries.

No labels from validation or test are used to transform the embeddings.  The
downstream decoder computes its whitening transform from ``probe_train`` only.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


BENCH = Path(__file__).resolve().parents[1]
ROOT = BENCH.parent
for _path in (str(BENCH / "scripts"), str(BENCH), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backbones import get_backbone  # noqa: E402
from blind_localization.data.lazyloader_dataset import CompactDataset  # noqa: E402
from train_ssl import log  # noqa: E402
from utils.preprocessing import preprocess_batch  # noqa: E402


def channel_geometry(path):
    """Return global channel ID -> (probe ID, local index, x, depth)."""
    out = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            out[str(row["id"])] = (
                str(row["ecephys_probe_id"]),
                int(row["local_index"]),
                float(row["probe_horizontal_position"]),
                float(row["probe_vertical_position"]),
            )
    return out


def raw_sample(ds, index):
    file_index, offset = ds._find_file_and_offset(index)
    return ds._load_file(file_index)[offset]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True,
                        help="Allen compact_atlas_lp_eval root")
    parser.add_argument("--channels_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_sequence_rows",
        type=int,
        default=4096,
        help=("maximum repeated-window tokens in one Transformer sequence; "
              "long Allen probes are deterministically divided into balanced "
              "full-coverage blocks"),
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    config["rand_init"] = True
    backbone = get_backbone(config["backbone"], config).to(device)
    checkpoint = torch.load(args.init_ckpt, map_location=device, weights_only=False)
    missing, unexpected = backbone.load_state_dict(
        checkpoint["backbone_state_dict"], strict=False
    )
    critical_prefixes = (
        "model.wav2vec2.encoder",
        "model.wav2vec2.feature_extractor",
        "model.wav2vec2.feature_projection",
    )
    critical_missing = [
        key for key in missing if key.startswith(critical_prefixes)
    ]
    # HuggingFace only instantiates this training-time vector when masking is
    # enabled.  The saved SSL checkpoint has it, while the deterministic
    # embedding-only configuration correctly does not.
    allowed_unexpected = {"model.wav2vec2.masked_spec_embed"}
    bad_unexpected = [key for key in unexpected if key not in allowed_unexpected]
    if critical_missing or bad_unexpected:
        raise RuntimeError(
            "Unsafe backbone checkpoint mismatch: "
            f"critical missing={critical_missing}; unexpected={bad_unexpected}"
        )
    if missing or unexpected:
        log(f"Non-critical checkpoint mismatch: missing={missing}; unexpected={unexpected}")
    backbone.eval()
    hidden = int(getattr(backbone, "hidden_dim", 768))
    log(f"Backbone loaded from epoch {checkpoint.get('epoch', '?')}; H={hidden}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "Allen compact_atlas_lp_eval",
        "checkpoint": str(Path(args.init_ckpt).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": str(Path(args.config).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "channels_csv": str(Path(args.channels_csv).resolve()),
        "hidden_dim": hidden,
        "splits": {},
    }
    del checkpoint

    geometry = channel_geometry(args.channels_csv)
    split_dirs = [("probe_train", "train"), ("probe_val", "val")]
    split_dirs.extend((f"test_fold_{i}", f"test_fold{i + 1}") for i in range(5))
    dataset_kwargs = dict(
        include_labels=True,
        dataset="Allen",
        verbose=False,
        atlas_depth=9,
        already_resampled=False,
        gpu_resample=True,
        input_sr=1250,
        model_sr=16000 if backbone.needs_resampling else 1250,
    )

    for output_name, input_name in split_dirs:
        split_path = Path(args.data_dir) / input_name
        if not split_path.is_dir():
            raise FileNotFoundError(split_path)
        ds = CompactDataset(compact_dir=str(split_path), **dataset_kwargs)
        n_rows = len(ds)
        emb = np.empty((n_rows, hidden), dtype=np.float16)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        offset = 0
        with torch.no_grad():
            for batch in loader:
                signal = batch[0].float().to(device, non_blocking=True)
                encoded = backbone.encode(preprocess_batch(
                    signal, backbone.needs_resampling, backbone.needs_normalization
                ))
                batch_rows = len(signal)
                emb[offset:offset + batch_rows] = encoded.cpu().numpy().astype(np.float16)
                offset += batch_rows
        if offset != n_rows:
            raise RuntimeError(f"{output_name}: encoded {offset} rows, expected {n_rows}")

        true_xyz = np.empty((n_rows, 3), dtype=np.float32)
        xcol = np.empty(n_rows, dtype=np.float32)
        depth = np.empty(n_rows, dtype=np.float32)
        channel = np.empty(n_rows, dtype=np.int64)
        pids = np.empty(n_rows, dtype="U64")
        metas = np.empty(n_rows, dtype="U200")
        local_index = np.empty(n_rows, dtype=np.int16)

        missing = []
        for index in range(n_rows):
            sample = raw_sample(ds, index)
            meta = str(sample[1])
            parts = meta.split("_")
            if len(parts) < 4:
                raise ValueError(f"cannot parse Allen metadata: {meta!r}")
            probe_id, channel_id = parts[2], parts[3]
            if channel_id not in geometry:
                missing.append(channel_id)
                continue
            geo_probe, local, horizontal, vertical = geometry[channel_id]
            if geo_probe != probe_id:
                raise ValueError(
                    f"channel {channel_id}: metadata probe={probe_id}, channels.csv probe={geo_probe}"
                )
            coords = np.asarray(sample[3], dtype=np.float32)
            if coords.shape != (3,):
                raise ValueError(f"{meta}: expected xyz shape (3,), got {coords.shape}")
            true_xyz[index] = coords
            xcol[index] = horizontal
            depth[index] = vertical
            channel[index] = int(channel_id)
            local_index[index] = local
            pids[index] = probe_id
            metas[index] = meta
        if missing:
            raise KeyError(f"{output_name}: {len(missing)} channel rows absent from channels.csv")

        # Allen contains substantially more repeated LFP windows per physical
        # probe than IBL (up to ~11k versus ~3.3k).  Preserve every row while
        # keeping the matched 4096-token ProbeSeq limit: distribute each
        # channel's repeated windows round-robin over balanced blocks.  Thus
        # each block contains the probe's depth range, and later scoring still
        # pools all predictions by the original probe/channel from meta_str.
        original_pids = pids.copy()
        sequence_ids = pids.copy()
        for probe_id in np.unique(original_pids):
            probe_rows = np.flatnonzero(original_pids == probe_id)
            n_blocks = int(np.ceil(len(probe_rows) / args.max_sequence_rows))
            if n_blocks <= 1:
                continue
            while True:
                block_for_row = np.empty(len(probe_rows), dtype=np.int64)
                for channel_id in np.unique(channel[probe_rows]):
                    within = np.flatnonzero(channel[probe_rows] == channel_id)
                    block_for_row[within] = np.arange(len(within)) % n_blocks
                block_sizes = np.bincount(block_for_row, minlength=n_blocks)
                if int(block_sizes.max()) <= args.max_sequence_rows:
                    break
                n_blocks += 1
            for block in range(n_blocks):
                sequence_ids[probe_rows[block_for_row == block]] = f"{probe_id}#b{block}"
        sequence_counts = np.unique(sequence_ids, return_counts=True)[1]
        if int(sequence_counts.max()) > args.max_sequence_rows:
            raise RuntimeError(
                f"{output_name}: sequence block has {sequence_counts.max()} rows "
                f"> limit {args.max_sequence_rows}"
            )

        output_path = out_dir / f"{output_name}_seq.npz"
        np.savez_compressed(
            output_path,
            emb=emb,
            true_xyz=true_xyz,
            xcol=xcol,
            depth=depth,
            channel=channel,
            local_index=local_index,
            pid=sequence_ids,
            probe_id=original_pids,
            meta_str=metas,
        )
        physical_counts = np.unique(original_pids, return_counts=True)[1]
        manifest["splits"][output_name] = {
            "source": input_name,
            "rows": n_rows,
            "physical_probes": int(len(physical_counts)),
            "sequences": int(len(sequence_counts)),
            "max_physical_probe_rows": int(physical_counts.max()),
            "max_sequence_rows": int(sequence_counts.max()),
            "sequence_block_limit": args.max_sequence_rows,
            "output": str(output_path),
        }
        log(
            f"{output_name}: {n_rows} rows, {len(physical_counts)} probes -> "
            f"{len(sequence_counts)} sequences, max={sequence_counts.max()} -> {output_path}"
        )

    with open(out_dir / "extract_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    log(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
