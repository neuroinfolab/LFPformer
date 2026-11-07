"""Precompute per-channel micro-manipulator CCF positions, ALIGNED to the
CompactDataset global index order, for every split of eval_v2_100k.

Why offline: the manip line needs iblatlas (AllenAtlas), which we don't want to
load inside the GPU finetune/eval job. We dump one (N,3) µm array per split that
the training/eval scripts load and index by dataset idx — no iblatlas at train
time.

Alignment: CompactDataset maps global idx → (file, offset) via fid-major /
offset-minor order (cumulative_counts). We iterate samples_{fid:06d}.pkl in the
SAME order, so manip_xyz[i] lines up with dataset[i] (exactly how
LabeledProbeTrainDataset._build_channel_index aligns channel numbers).

Sanity check: folded manip-vs-true RMSE per split should land near the ~920µm
manipulator headline (proves alignment + manip_line geometry are correct).

Run in w2v2_cuda_cu128_ibl.sif (needs iblatlas). CPU only.
"""
import os, sys, glob, json, pickle
import numpy as np

SD = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SD)
if not hasattr(np, "_core"):
    import numpy.core as _nc
    sys.modules["numpy._core"] = _nc
    sys.modules["numpy._core.multiarray"] = getattr(_nc, "multiarray", _nc)
from iblatlas.atlas import Insertion, AllenAtlas
from apply_kabsch_posthoc_ibl import fold_lr

EVAL_ROOT = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/data/ibl/subject_split_v2/eval_v2_100k')
CACHES = [__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/one_cache/train_manip_trajectories.json'),
          __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/one_cache/eval_manip_trajectories.json')]
OUT_DIR = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/one_cache/manip_xyz/eval_v2_100k')
SPLITS = ["probe_train", "probe_val"] + [f"test_fold_{i}" for i in range(5)]
os.makedirs(OUT_DIR, exist_ok=True)
ba = AllenAtlas(25)

# combined manip cache keyed by eid_probe
manip = {}
for c in CACHES:
    for r in json.load(open(c)):
        manip[f"{r['eid']}_{r['probe']}"] = r
print(f"manip cache: {len(manip)} probes")


def manip_line(traj, ch):
    """Per-channel CCF xyz from the surgical log: entry (x,y,z) + depth + angles (theta,phi),
    plus the NP1.0 site geometry (2 sites/row, 20 µm row pitch).

    ⛔ KNOWN BUG (found 2026-07-14, NOT fixed here on purpose — read before "fixing" it):
    `s` starts at 0, i.e. channel 0 is placed AT the probe's physical tip. It belongs
    TIP_SIZE_UM = 200 µm ABOVE it: an NP1.0 shank has a ~200 µm taper below its first
    electrode (constant lives in the `neuropixel` package, "209 / 206 NP1 / NP2"), and IBL's
    `depth` is entry -> the PHYSICAL TIP (Insertion.from_track sets tip = deepest traced point,
    then depth = |entry - tip|). ibllib's own histology applies it:
        xyz_channels = interpolate_along_track(xyz, (channels_positions[:,1] + TIP_SIZE_UM)/1e6)
    So every channel here sits 200 µm too deep. The correct line is `s = ((ch // 2) * 20 + 200)`.
    Verified against the histology-derived ground truth (494k test channels): the truth sits a
    mean +178 µm HIGHER up the shank, the RMSE-minimizing offset is +180 µm, and applying +200 µm
    takes the pooled manipulator baseline from 920.0 -> 902.7 µm.

    ⚠️ WHY IT IS STILL HERE: manip+full-FT (the 827 µm model) was TRAINED on this cache. Rebuilding
    it with the +200 µm would shift that model's INPUT by 200 µm at eval time — a train/test skew
    that would corrupt its predictions. Fixing this properly means rebuilding the cache AND
    retraining manip+full-FT. Until then the correction is applied downstream, where only the
    manipulator BASELINE is affected: see `_tip_correct()` in viz_probe_lines_3d_splits.py.
    """
    ins = Insertion.from_dict(
        dict(x=traj["x"], y=traj["y"], z=traj["z"], depth=traj["depth"],
             theta=traj["theta"], phi=traj["phi"]), brain_atlas=ba)
    e, t = ins.entry, ins.tip
    up = (e - t) / np.linalg.norm(e - t)
    s = (ch // 2) * 20 * 1e-6                 # BUG: should be ((ch // 2) * 20 + 200) * 1e-6
    pts = t[None, :] + up[None, :] * s[:, None]
    return ba.xyz2ccf(pts, ccf_order="apdvml", mode="clip")  # (n,3)=ap,dv,lr


def iter_split_samples(split_dir):
    """Yield (eid_probe_key, channel, true_xyz) in CompactDataset idx order."""
    idx = pickle.load(open(os.path.join(split_dir, "compact_index.pkl"), "rb"))
    for fid in range(idx["n_files"]):
        fpath = os.path.join(split_dir, f"samples_{fid:06d}.pkl")
        for s in pickle.load(open(fpath, "rb")):
            m = s[1]
            key = f"{m.split('_')[0]}_{m.split('_')[2]}"
            ch = int(m.split('_')[3].split('-ch')[1])
            c = s[2]  # hierarchy dict OR coords; grab coords from s if present
            # CompactDataset coords come from s[3] (ap,dv,lr) in these pickles
            xyz = s[3] if len(s) >= 4 else None
            yield key, ch, xyz


for split in SPLITS:
    sd = os.path.join(EVAL_ROOT, split)
    keys, chs, trues = [], [], []
    for key, ch, xyz in iter_split_samples(sd):
        keys.append(key); chs.append(ch); trues.append(xyz)
    keys = np.array(keys); chs = np.array(chs, dtype=int)
    n = len(keys)
    mx = np.full((n, 3), np.nan, dtype=np.float64)
    for key in np.unique(keys):
        if key not in manip:
            continue
        sel = keys == key
        ml = manip_line(manip[key], chs[sel])
        if np.isfinite(ml).all():
            mx[sel] = ml
    valid = np.isfinite(mx).all(1)
    np.save(os.path.join(OUT_DIR, f"{split}_manip_xyz.npy"), mx.astype(np.float32))
    np.save(os.path.join(OUT_DIR, f"{split}_valid.npy"), valid)

    # sanity: folded manip-vs-true RMSE (should be ~900-950 µm)
    msg = f"{split:14s} N={n:7d} valid={valid.sum():7d} ({valid.mean()*100:5.1f}%)"
    if trues[0] is not None:
        true = np.array(trues, dtype=np.float64)
        if true.shape == mx.shape:
            v = valid
            a, b = fold_lr(mx[v]), fold_lr(true[v])
            rmse = float(np.sqrt(((a - b) ** 2).sum(1).mean()))
            msg += f"  folded manip-vs-true RMSE = {rmse:7.1f} µm"
    print(msg, flush=True)

print(f"\nsaved manip_xyz npy -> {OUT_DIR}")
