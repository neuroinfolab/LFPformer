"""Pinned, reviewed ProbeSeq helpers for the matched direct-XYZ experiment.
Shared model and dataset helpers for folded fusion training and inference.
"""
import math, os, sys
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
UM = 1000.0
ALLEN_MIDLINE_UM = 5739.0


def log(*a):
    print(*a, flush=True)


def depth_pos_encoding(depth, d_model, max_depth=8000.0):
    """Sinusoidal positional encoding keyed on physical depth (µm)."""
    device = depth.device
    pos = (depth / max_depth).unsqueeze(-1)  # (B,L,1) in [0,~1]
    div = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(*depth.shape, d_model, device=device)
    pe[..., 0::2] = torch.sin(pos * div * math.pi)
    pe[..., 1::2] = torch.cos(pos * div * math.pi)
    return pe


def rot6d_to_matrix(r6):
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def seq_loss(pred_um, true_um, depth, mask, lam_smooth, lam_mono, sigma_um=None,
             lam_var=0.0):
    valid = (~mask).float().unsqueeze(-1)
    pk, tk = pred_um / UM, true_um / UM
    mse = ((pk - tk) ** 2 * valid).sum() / valid.sum().clamp(min=1) / 3  # reported (kµm²)
    if sigma_um is None:
        loss = mse                                          # plain MSE objective
    else:
        # diagonal Gaussian NLL (kµm): 0.5*((μ-y)/σ)² + log σ  (drops const 0.5·log2π)
        sk = sigma_um / UM
        nll = (0.5 * ((pk - tk) / sk) ** 2 + torch.log(sk))
        loss = (nll * valid).sum() / valid.sum().clamp(min=1) / 3
    sm = mo = torch.tensor(0.0, device=pred_um.device)
    if lam_smooth > 0 or lam_mono > 0:
        # adjacent-channel diffs along the (depth-sorted) sequence
        pair = (~mask[:, 1:]) & (~mask[:, :-1])
        dxyz = (pk[:, 1:] - pk[:, :-1])                      # (B,L-1,3)
        ddepth = (depth[:, 1:] - depth[:, :-1]) / UM         # (B,L-1)
        if lam_smooth > 0:
            sm = (dxyz.pow(2).sum(-1) * pair.float()).sum() / pair.float().sum().clamp(min=1)
            loss = loss + lam_smooth * sm
        if lam_mono > 0:
            # monotonic-depth penalty: predicted step should align with depth direction
            step = dxyz.norm(dim=-1)                          # (B,L-1)
            mono_viol = F.relu(-(step) * torch.sign(ddepth))
            mo = (mono_viol * pair.float()).sum() / pair.float().sum().clamp(min=1)
            loss = loss + lam_mono * mo
    # anti-collapse / anti-centroid spread-matching penalty (VICReg-style).
    # Pool valid tokens across the batch, per axis (in kµm), and penalise the
    # predicted std falling BELOW the target std -> hinge(std_true - std_pred).
    # Zero when predictions keep healthy spread; fires when the model regresses
    # its outputs toward the (constant) dataset centroid.
    vr = torch.tensor(0.0, device=pred_um.device)
    if lam_var > 0:
        vm = valid                                           # (B,L,1)
        cnt = vm.sum().clamp(min=1)
        mean_p = (pk * vm).sum(dim=(0, 1)) / cnt              # (3,)
        mean_t = (tk * vm).sum(dim=(0, 1)) / cnt
        std_p = torch.sqrt(((pk - mean_p) ** 2 * vm).sum(dim=(0, 1)) / cnt + 1e-6)
        std_t = torch.sqrt(((tk - mean_t) ** 2 * vm).sum(dim=(0, 1)) / cnt + 1e-6)
        vr = F.relu(std_t - std_p).mean()
        loss = loss + lam_var * vr
    return loss, mse.item(), float(sm), float(mo), float(vr)


def fold_lr_t(xyz):
    out = xyz.clone()
    out[..., 2] = ALLEN_MIDLINE_UM - (out[..., 2] - ALLEN_MIDLINE_UM).abs()
    return out


class ManipProbeSequenceDataset(Dataset):
    def __init__(self, cache_npz, manip_npy, max_len=384,
                 emb_mean=None, emb_std=None, emb_W=None):
        d = np.load(cache_npz, allow_pickle=True)
        emb = d["emb"].astype(np.float32); true = d["true_xyz"].astype(np.float32)
        # Re-apply the SAME fixed emb affine the decoder was trained with (global-scale /
        # whitening via emb_W, or per-dim z-score via emb_mean/emb_std). Mirrors the transform
        # in train_probe_sequence_ext.py's ProbeSequenceDataset. Applied BEFORE padding/grouping.
        if emb_W is not None:
            if emb_mean is None:
                raise ValueError("emb_W given without emb_mean -- whitening/global-scale needs the mean")
            emb = (emb - emb_mean.astype(np.float32)) @ emb_W.astype(np.float32).T
        elif emb_mean is not None and emb_std is not None:
            emb = (emb - emb_mean.astype(np.float32)) / emb_std.astype(np.float32)
        xcol = d["xcol"].astype(np.float32); depth = d["depth"].astype(np.float32)
        pid = d["pid"].astype(str); meta = d["meta_str"].astype(str)
        manip = np.load(manip_npy).astype(np.float32)            # (N,3) CCF µm, row-aligned
        assert manip.shape[0] == emb.shape[0], f"manip {manip.shape} vs cache {emb.shape}"
        ok = np.isfinite(manip).all(1)                            # drop channels w/o manip
        groups = defaultdict(list)
        for i, p in enumerate(pid):
            if p and p != "None" and ok[i]:
                groups[p].append(i)
        self.items = []; self.H = emb.shape[1]
        for p, idxs in groups.items():
            idxs = np.asarray(idxs)
            idxs = idxs[np.argsort(depth[idxs])]               # depth-ordered
            if len(idxs) > max_len:
                idxs = idxs[np.linspace(0, len(idxs) - 1, max_len).astype(int)]
            self.items.append(dict(pid=p, emb=emb[idxs], manip=manip[idxs], true=true[idxs],
                                   xcol=xcol[idxs], depth=depth[idxs], meta=meta[idxs]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_manip(batch):
    B = len(batch); L = max(len(it["emb"]) for it in batch); H = batch[0]["emb"].shape[1]
    emb = torch.zeros(B, L, H); manip = torch.zeros(B, L, 3); geom = torch.zeros(B, L, 2)
    depth = torch.zeros(B, L); true = torch.zeros(B, L, 3); mask = torch.ones(B, L, dtype=torch.bool)
    metas = []
    for b, it in enumerate(batch):
        n = len(it["emb"])
        emb[b, :n] = torch.from_numpy(it["emb"]); manip[b, :n] = torch.from_numpy(it["manip"])
        geom[b, :n, 0] = torch.from_numpy(it["xcol"]); geom[b, :n, 1] = torch.from_numpy(it["depth"])
        depth[b, :n] = torch.from_numpy(it["depth"]); true[b, :n] = torch.from_numpy(it["true"])
        mask[b, :n] = False; metas.append(it)
    return emb, manip, geom, depth, true, mask, metas


class ManipProbeSequenceModel(nn.Module):
    def __init__(self, H, d_model=256, nhead=8, layers=4, ff=512,
                 dropout=0.1, decompose=True, parameterization=None, gaussian=False):
        super().__init__()
        # Backward compatibility: legacy callers specify only --decompose.
        if parameterization is None:
            parameterization = "decomposed" if decompose else "residual"
        if parameterization not in {"direct", "residual", "decomposed"}:
            raise ValueError(f"unknown parameterization={parameterization!r}")
        self.parameterization = parameterization
        self.decompose = parameterization == "decomposed"
        self.gaussian = bool(gaussian)
        self.in_proj = nn.Sequential(nn.Linear(H + 3 + 2, d_model), nn.GELU())  # emb ⊕ manip ⊕ geom
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.resid_head = nn.Linear(d_model, 3)
        if self.gaussian:
            # per-channel aleatoric σ (kµm via softplus); bias init so σ≈1 kµm at start
            self.sigma_head = nn.Linear(d_model, 3)
            nn.init.zeros_(self.sigma_head.weight)
            nn.init.constant_(self.sigma_head.bias, 0.5413)  # softplus(0.5413)≈1.0
        if self.decompose:
            self.pose_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 9))
            nn.init.zeros_(self.pose_head[-1].weight); nn.init.zeros_(self.pose_head[-1].bias)
            with torch.no_grad():
                self.pose_head[-1].bias[0] = 1.0; self.pose_head[-1].bias[4] = 1.0   # R init = I
        if self.parameterization != "direct":
            # Residual variants start exactly at the manipulator prior.
            nn.init.zeros_(self.resid_head.weight)
            nn.init.zeros_(self.resid_head.bias)

    def forward(self, emb, manip, geom, depth, mask):
        manip_k = manip / UM; geom_k = geom / UM                  # kµm
        tok = self.in_proj(torch.cat([emb, manip_k, geom_k], dim=-1))
        tok = tok + depth_pos_encoding(depth, tok.shape[-1])
        ctx = self.encoder(tok, src_key_padding_mask=mask)
        resid = self.resid_head(ctx)                              # (B,L,3) kµm
        if self.parameterization == "direct":
            mu_um = resid * UM                                   # direct atlas XYZ (µm)
        elif self.parameterization == "residual":
            mu_um = manip + resid * UM                            # manip anchor + residual (µm)
        else:
            valid = (~mask).float().unsqueeze(-1)
            pooled = (ctx * valid).sum(1) / valid.sum(1).clamp(min=1)
            pose = self.pose_head(pooled)
            R = rot6d_to_matrix(pose[:, :6]); t = pose[:, 6:9]    # per-probe rigid correction (kµm)
            rigid = torch.einsum('bij,blj->bli', R, manip_k) + t.unsqueeze(1)
            mu_um = (rigid + resid) * UM                          # R·manip + t + residual (µm)
        if self.gaussian:
            sigma_um = (F.softplus(self.sigma_head(ctx)) + 1e-3) * UM  # (B,L,3) µm, >0
            return mu_um, sigma_um
        return mu_um


def _unpack_pred(out):
    """Return (mu_um, sigma_um_or_None) from model forward."""
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


@torch.no_grad()
def evaluate(model, loaders, device):
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]
    model.eval(); se = 0.0; n = 0
    for loader in loaders:
        for emb, manip, geom, depth, true, mask, _ in loader:
            emb, manip, geom, depth, true, mask = [x.to(device) for x in (emb, manip, geom, depth, true, mask)]
            pred, _sigma = _unpack_pred(model(emb, manip, geom, depth, mask))
            valid = (~mask).float().unsqueeze(-1)
            a = fold_lr_t(pred); t = fold_lr_t(true)
            se += (((a - t) ** 2).sum(-1, keepdim=True) * valid).sum().item()
            n += valid.sum().item()
    return math.sqrt(se / max(n, 1))


@torch.no_grad()
def write_predictions(model, ds, device, out_dir, split, batch_size=4):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_manip)
    preds, sigmas, trues, fids, metas = [], [], [], [], []
    for emb, manip, geom, depth, true, mask, items in loader:
        emb, manip, geom, depth, mask = [x.to(device) for x in (emb, manip, geom, depth, mask)]
        pred, sigma = _unpack_pred(model(emb, manip, geom, depth, mask))
        pred = pred.cpu().numpy()
        sigma = (None if sigma is None else sigma.cpu().numpy())
        for b, it in enumerate(items):
            n = len(it["emb"])
            preds.append(pred[b, :n]); trues.append(it["true"])
            metas.append(it["meta"]); fids.append(np.zeros(n, np.int64))
            if sigma is None:
                sigmas.append(np.zeros((n, 3), np.float32))
            else:
                sigmas.append(sigma[b, :n])
    P = np.concatenate(preds).astype(np.float32); T = np.concatenate(trues).astype(np.float32)
    S = np.concatenate(sigmas).astype(np.float32)
    M = np.concatenate(metas); Fd = np.concatenate(fids)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    out = os.path.join(out_dir, "predictions", f"{split}_predictions.npz")
    np.savez_compressed(out, pred_xyz=P, pred_xyz_gauss=P, pred_sigma_gauss=S,
                        true_xyz=T, fine_id=Fd, meta_str=M.astype("U200"))
    return out


def subject_held_out_split(tr, va, args):
    """In place: pool tr.items + va.items (same insertions, different time chunks)
    and re-split by SUBJECT. Subject list + seed match train_xyz_finetune_manip.py,
    so both fusion decoders hold out the same subjects."""
    import json as _json
    smap = _json.load(open(args.subject_map))

    def subj(it):
        eid = str(it["meta"][0]).split("_")[0]
        if eid not in smap:
            raise ValueError(f"session {eid} missing from {args.subject_map}")
        return smap[eid]

    pool = tr.items + va.items
    s_all = [subj(it) for it in pool]
    subjects = sorted(set(s_all))
    order = np.random.RandomState(args.split_seed).permutation(len(subjects))
    n_val = max(1, int(round(args.subject_val_frac * len(subjects))))
    val_subj = sorted(subjects[i] for i in order[:n_val])
    tr.items = [it for it, s in zip(pool, s_all) if s not in val_subj]
    va.items = [it for it, s in zip(pool, s_all) if s in val_subj]
    assert not ({subj(it) for it in tr.items} & {subj(it) for it in va.items})
    log(f"SUBJECT-held-out val (frac={args.subject_val_frac}, split_seed={args.split_seed}): "
        f"train {len(subjects) - n_val} subj / {len(tr.items)} seqs, "
        f"val {n_val} subj / {len(va.items)} seqs; val subjects = {val_subj}")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "subject_split.json"), "w") as f:
        _json.dump({"val_subjects": val_subj, "n_subjects": len(subjects),
                    "n_train_seqs": len(tr.items), "n_val_seqs": len(va.items),
                    "split_seed": args.split_seed, "frac": args.subject_val_frac}, f, indent=1)


def ds_for(cache_dir, manip_dir, split, max_len, emb_mean=None, emb_std=None, emb_W=None):
    return ManipProbeSequenceDataset(os.path.join(cache_dir, f"{split}_seq.npz"),
                                     os.path.join(manip_dir, f"{split}_manip_xyz.npy"), max_len=max_len,
                                     emb_mean=emb_mean, emb_std=emb_std, emb_W=emb_W)

