#!/usr/bin/env python3
"""Cache per-channel backbone embeddings + on-probe geometry for the
probe-as-sequence model (train_probe_sequence.py).

Runs the 75k Path-B backbone (frozen) once over probe_train / probe_val /
test_fold_0..4 and saves FLAT arrays per split:
  emb (M,H) float16, true_xyz (M,3), xcol (M,), depth (M,), channel (M,),
  pid (M,) str, meta_str (M,).
The trainer groups rows by pid into depth-ordered sequences. Caching the frozen
embeddings lets the sequence architecture be iterated cheaply (seconds/run) with
NO backbone re-encode. For end-to-end FT later, train_probe_sequence.py has a
--backbone_ckpt path that bypasses the cache (TODO hook).

on-probe geometry (xcol, depth) comes from channel_xy_onprobe — the SAME known
straight-probe coordinates the headline Kabsch uses; this is geometry, not labels.

Run in w2v2_cuda_cu128_ibl.sif.
"""
import argparse, os, sys
import numpy as np
import torch

PB_BENCH = os.environ.get("PB_BENCH", __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/Lfp2vec_benchmarks'))
PB_ROOT = os.path.dirname(PB_BENCH)
for _p in (os.path.join(PB_BENCH, "scripts"), PB_BENCH, PB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbones import get_backbone
from utils.preprocessing import preprocess_batch
from train_ssl import log
from blind_localization.data.lazyloader_dataset import CompactDataset
from torch.utils.data import DataLoader
from recompute_refined_3d_metrics import parse_pid_channel, channel_xy_onprobe
from lora_wav2vec2 import inject_lora, parse_lora_targets
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init_ckpt", required=True, help="backbone ckpt (75k path-b 7741664)")
    p.add_argument("--config", required=True)
    p.add_argument("--probe_data_path", required=True, help="dir with probe_train/ probe_val/")
    p.add_argument("--eval_data_dir", required=True, help="dir with test_fold_0..4/")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # Force rand_init: we immediately overwrite weights from --init_ckpt below,
    # so skip Wav2Vec2ForPreTraining.from_pretrained (dodges the HF offline
    # dual-snapshot crash: 'NoneType' object has no attribute 'endswith').
    config["rand_init"] = True
    backbone = get_backbone(config["backbone"], config).to(device)
    ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
    lora_rank = int(ckpt.get("lora_rank", 0) or 0)
    if lora_rank > 0:
        # Inject empty adapters so state_dict keys match, then load full weights.
        alpha = float(ckpt.get("lora_alpha", 8.0))
        targets = parse_lora_targets(ckpt.get("lora_targets", "q_proj,v_proj"))
        inject_lora(backbone, rank=lora_rank, alpha=alpha, targets=targets)
        backbone.load_state_dict(ckpt["backbone_state_dict"], strict=True)
        # Belt-and-suspenders: adapters must share device with encode() inputs.
        backbone.to(device)
        log(
            f"LoRA reinjected: rank={lora_rank} alpha={alpha} targets={targets}"
        )
    else:
        backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.eval()
    H = backbone.hidden_dim if hasattr(backbone, "hidden_dim") else 768
    log(f"Backbone {config['backbone']} loaded (epoch {ckpt.get('epoch','?')}), H={H}")
    del ckpt
    os.makedirs(args.out_dir, exist_ok=True)

    ck = dict(include_labels=True, dataset="IBL", verbose=False, atlas_depth=9,
              already_resampled=False, gpu_resample=True,
              input_sr=1250, model_sr=16000 if backbone.needs_resampling else 1250)

    splits = [("probe_train", os.path.join(args.probe_data_path, "probe_train")),
              ("probe_val", os.path.join(args.probe_data_path, "probe_val"))]
    splits += [(f"test_fold_{i}", os.path.join(args.eval_data_dir, f"test_fold_{i}")) for i in range(5)]

    def meta_of(ds, i):
        fi, off = ds._find_file_and_offset(i)
        return str(ds._load_file(fi)[off][1])

    for name, path in splits:
        if not os.path.isdir(path):
            log(f"  skip {name}: {path} missing"); continue
        ds = CompactDataset(compact_dir=path, **ck)
        n = len(ds)
        emb = np.zeros((n, H), np.float16)
        true = np.zeros((n, 3), np.float32)
        xcol = np.zeros(n, np.float32); depth = np.zeros(n, np.float32)
        chan = np.full(n, -1, np.int32); pids = np.empty(n, object); metas = np.empty(n, object)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        off = 0
        with torch.no_grad():
            for batch in loader:
                sig = batch[0].float().to(device, non_blocking=True); b = sig.shape[0]
                x = preprocess_batch(sig, backbone.needs_resampling, backbone.needs_normalization)
                emb[off:off+b] = backbone.encode(x).cpu().numpy().astype(np.float16)
                off += b
        for i in range(n):
            _, _, c = ds[i]; true[i] = (float(c[0]), float(c[1]), float(c[2]))
            m = meta_of(ds, i); metas[i] = m
            pid, chn = parse_pid_channel(m)
            pids[i] = pid
            if chn is not None:
                chan[i] = int(chn); xc, dp = channel_xy_onprobe(int(chn))
                xcol[i] = float(xc); depth[i] = float(dp)
        out = os.path.join(args.out_dir, f"{name}_seq.npz")
        np.savez_compressed(out, emb=emb, true_xyz=true, xcol=xcol, depth=depth,
                            channel=chan, pid=pids.astype("U64"), meta_str=metas.astype("U200"))
        nprobe = len({x for x in pids if x is not None})
        log(f"  {name}: {n} rows, {nprobe} probes -> {out}")

    log(f"\nDone. -> {args.out_dir}")


if __name__ == "__main__":
    main()
