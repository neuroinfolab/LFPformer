#!/usr/bin/env python3
"""PROBE-AS-SEQUENCE model — Transformer over depth-ordered channel embeddings.

Per-channel regression treats each channel independently; the headline Kabsch
then snaps the cloud to a rigid line. This model instead consumes a probe's
channels as an ORDERED 1-D sequence (sorted by on-probe depth) and predicts the
whole CCF trajectory jointly, so it can (a) denoise via cross-channel context and
(b) learn the (mildly non-rigid) trajectory shape that rigid Kabsch cannot.

  tokens   = proj([emb_i ⊕ geom_feats_i])  + depth-positional-encoding   (per channel i)
  context  = TransformerEncoder(tokens, key_padding_mask)
  output (two heads, --decompose toggles):
    direct : xyz_i = Linear(context_i)                      # free per-channel
    rigid+ : (R,t) = pose_head(mean_pool(context))          # per-probe rigid (Kabsch-like)
             xyz_i = R @ geom_i + t + Linear(context_i)     # rigid line + learned non-rigid residual

  loss = masked MSE(xyz, true)/UM²  + λ_smooth·‖Δdepth xyz‖  + λ_mono·relu(-d(depth-proj)/di)

Trains on CACHED frozen embeddings (extract_probe_sequences.py) for fast
architecture iteration. Eval writes test_fold_*_predictions.npz (standard keys)
so kabsch_eval_1M_pathb.py runs unchanged. CPU-light; 1 GPU optional.

NOTE: this is LFP-only — it will still face the per-probe OOD pose shrinkage wall
(~1585µm post-hoc). Its hypothesis is that learned non-rigid shape + cross-channel
denoising beats per-channel+rigid-Kabsch on SHAPE, not that it removes the absolute
pose bias (that needs anchors/manip). Compare folded Kabsch RMSE vs 1585/1628 wall.

Run in w2v2_cuda_cu128_ibl.sif.
"""
import argparse, math, os, sys
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PL = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws2/Alphabrain_staging/Lfp2vec_benchmarks/scripts')
if PL not in sys.path:
    sys.path.insert(0, PL)
from recompute_refined_3d_metrics import ALLEN_MIDLINE_UM  # noqa

UM = 1000.0


def log(*a):
    print(*a, flush=True)


def emb_perdim_stats(npz_path):
    """Per-dim mean/std of the frozen embeddings, computed on TRAIN only (applied to
    every split -> no leakage). Fixes the scaled-SSL conditioning problem: 301k/1M
    embeddings have ~60x smaller rms than 75k, so the raw decoder can't use them
    (linear-probe R2 raw 1M 0.28 vs standardized 0.66). Diagnosed 2026-07-11."""
    E = np.load(npz_path, allow_pickle=True)["emb"].astype(np.float32)
    return E.mean(0), E.std(0) + 1e-6


def _ledoit_wolf_alpha(Xc, S, tr_over_d):
    """Analytic Ledoit-Wolf shrinkage intensity toward (tr(S)/D)*I. Closed form, so
    NOTHING is selected on any metric -- we may not tune on val (does not predict
    localization) nor on test_fold* (winner's curse)."""
    n = Xc.shape[0]
    s_fro2 = float((S ** 2).sum())
    d2 = s_fro2 - 2.0 * tr_over_d * float(np.trace(S)) + Xc.shape[1] * tr_over_d ** 2
    # sum_i ||x_i x_i^T - S||_F^2  ==  sum_i ||x_i||^4 - n*||S||_F^2   (since S = X^T X / n)
    sq = (Xc ** 2).sum(1, dtype=np.float64)
    b2 = float((sq ** 2).sum() - n * s_fro2) / (n ** 2)
    b2 = min(max(b2, 0.0), d2)
    return 0.0 if d2 <= 0 else float(min(1.0, b2 / d2))


def emb_whiten_stats(npz_path, mode="zca", shrink="lw", eps=1e-8):
    """Full-covariance whitening of the frozen embeddings, TRAIN-only stats.

    z-score (emb_perdim_stats) is a DIAGONAL rescale: it fixes the magnitude mismatch
    against the O(1) geometry inputs but leaves the covariance anisotropy intact. These
    embeddings are variance-CONCENTRATED, not rank-collapsed (participation ratio 4-7 of
    768 dims), so a handful of directions still dominate the decoder's input. Whitening
    decorrelates and equalises every direction: x' = (x - mu) @ W^T, W = S^-1/2.

    The hazard: S is near-singular, so an unregularised S^-1/2 amplifies ~760 near-noise
    directions to unit variance. Hence shrinkage S_a = (1-a)S + a*(tr(S)/D)*I, with `a`
    from the analytic Ledoit-Wolf estimator by default (no metric is consulted).

    mode: 'zca' -> W = U diag(l^-1/2) U^T  (minimal rotation; directly comparable to z-score)
          'pca' -> W =   diag(l^-1/2) U^T  (axis-aligned to principal directions)
    Returns (mean[D], W[D,D], info dict).
    """
    E = np.load(npz_path, allow_pickle=True)["emb"].astype(np.float32)
    mu = E.mean(0)
    Xc = (E - mu).astype(np.float64)
    n, D = Xc.shape
    S = (Xc.T @ Xc) / n
    tr_over_d = float(np.trace(S)) / D
    a = _ledoit_wolf_alpha(Xc, S, tr_over_d) if shrink == "lw" else float(shrink)
    S_a = (1.0 - a) * S + a * tr_over_d * np.eye(D)
    lam, U = np.linalg.eigh(S_a)
    lam = np.clip(lam, eps * max(float(lam.max()), eps), None)
    inv_sqrt = 1.0 / np.sqrt(lam)
    W = (U * inv_sqrt) @ U.T if mode == "zca" else (inv_sqrt[:, None] * U.T)
    # Renormalize W to UNIT TRAIN OUTPUT RMS. Without this, the whitened input scale is a
    # function of the shrinkage `a`: whitened cov is W S W^T, whose eigenvalues are
    # lam/((1-a)lam + a*tr(S)/D), so as `a` grows the ~760 low-variance directions (PR ~6/768)
    # are pushed FAR below unit variance and the global rms falls — measured 0.97 (lw) ->
    # 0.64 (a=0.01) -> 0.42 (a=0.1). That would confound the two things the alpha sweep is
    # meant to separate: conditioning (improves ~2000x with a) and INPUT SCALE (falls with a).
    # Scale is not a detail in this decoder — in_proj is a bare Linear with NO input LayerNorm
    # and a unit-amplitude positional encoding is ADDED to its output, so the emb:geom:pos ratio
    # at the token rides directly on this rms. A global scalar leaves the rotation and the
    # per-direction reweighting (i.e. the actual shrinkage) untouched; it only removes the
    # a-dependent drift. Applied to every mode, so the whole whit family lands at rms 1.0,
    # scale-matched to --standardize_emb ⇒ whit-vs-std is a PURE DECORRELATION contrast.
    out_rms = float(np.sqrt(max(np.einsum("ij,ji->", W @ S, W.T), 0.0) / D))
    W = W / max(out_rms, 1e-30)
    lam_raw = np.clip(np.linalg.eigvalsh(S), 0.0, None)
    pr = float(lam_raw.sum() ** 2 / max((lam_raw ** 2).sum(), 1e-30))   # participation ratio
    info = dict(alpha=a, mode=mode, participation_ratio=pr, dim=D, n_train=n,
                cond_raw=float(lam_raw.max() / max(lam_raw.min(), 1e-30)),
                cond_shrunk=float(lam.max() / lam.min()),
                rms_before_rescale=out_rms)
    return mu.astype(np.float32), W.astype(np.float32), info


# ───────────────────────────── dataset ─────────────────────────────────────
class ProbeSequenceDataset(Dataset):
    """One item == one probe: depth-sorted (emb, geom, true_xyz) + length.
    Variable length; collate pads to batch max with a key-padding mask."""
    def __init__(self, npz_path, max_len=384, emb_mean=None, emb_std=None, emb_W=None):
        d = np.load(npz_path, allow_pickle=True)
        emb = d["emb"].astype(np.float32); true = d["true_xyz"].astype(np.float32)
        if emb_W is not None:
            # full-covariance whitening with TRAIN stats: decorrelates AND equalises every
            # direction, unlike the diagonal z-score below. See emb_whiten_stats().
            if emb_mean is None:
                raise ValueError("emb_W given without emb_mean -- whitening needs the mean")
            emb = (emb - emb_mean.astype(np.float32)) @ emb_W.astype(np.float32).T
        elif emb_mean is not None and emb_std is not None:
            # per-dim z-score with TRAIN stats: scaled-SSL embeddings are ~60x smaller
            # than the geom features, so the raw decoder ignores them; standardizing
            # unlocks their (superior) location content. See emb_perdim_stats().
            emb = (emb - emb_mean.astype(np.float32)) / emb_std.astype(np.float32)
        xcol = d["xcol"].astype(np.float32); depth = d["depth"].astype(np.float32)
        pid = d["pid"].astype(str); meta = d["meta_str"].astype(str); chan = d["channel"]
        groups = defaultdict(list)
        for i, p in enumerate(pid):
            if p and p != "None":
                groups[p].append(i)
        self.items = []
        self.H = emb.shape[1]
        for p, idxs in groups.items():
            idxs = np.asarray(idxs)
            order = np.argsort(depth[idxs])  # depth-ordered
            idxs = idxs[order]
            if len(idxs) > max_len:   # subsample evenly if a probe is huge
                sel = np.linspace(0, len(idxs) - 1, max_len).astype(int)
                idxs = idxs[sel]
            self.items.append(dict(
                pid=p, emb=emb[idxs], true=true[idxs],
                xcol=xcol[idxs], depth=depth[idxs], chan=chan[idxs],
                meta=meta[idxs], idx=idxs))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    B = len(batch); L = max(len(it["emb"]) for it in batch); H = batch[0]["emb"].shape[1]
    emb = torch.zeros(B, L, H); geom = torch.zeros(B, L, 2); true = torch.zeros(B, L, 3)
    depth = torch.zeros(B, L); mask = torch.ones(B, L, dtype=torch.bool)  # True == PAD
    metas = []
    for b, it in enumerate(batch):
        n = len(it["emb"])
        emb[b, :n] = torch.from_numpy(it["emb"])
        geom[b, :n, 0] = torch.from_numpy(it["xcol"]); geom[b, :n, 1] = torch.from_numpy(it["depth"])
        depth[b, :n] = torch.from_numpy(it["depth"])
        true[b, :n] = torch.from_numpy(it["true"])
        mask[b, :n] = False
        metas.append(it)
    return emb, geom, depth, true, mask, metas


# ───────────────────────────── model ───────────────────────────────────────
def depth_pos_encoding(depth, d_model, max_depth=8000.0):
    """Sinusoidal positional encoding keyed on physical depth (µm)."""
    device = depth.device
    pos = (depth / max_depth).unsqueeze(-1)  # (B,L,1) in [0,~1]
    div = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(*depth.shape, d_model, device=device)
    pe[..., 0::2] = torch.sin(pos * div * math.pi)
    pe[..., 1::2] = torch.cos(pos * div * math.pi)
    return pe


class ProbeSequenceModel(nn.Module):
    def __init__(self, H, d_model=256, nhead=8, layers=4, ff=512, dropout=0.1,
                 decompose=True, gaussian=False):
        super().__init__()
        self.decompose = decompose
        self.gaussian = gaussian
        self.in_proj = nn.Sequential(nn.Linear(H + 2, d_model), nn.GELU())  # +geom(xcol,depth)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.resid_head = nn.Linear(d_model, 3)   # per-channel residual (kµm)
        if gaussian:
            # per-channel aleatoric σ (kµm via softplus); bias init so σ≈1 kµm at start
            self.sigma_head = nn.Linear(d_model, 3)
            nn.init.zeros_(self.sigma_head.weight)
            nn.init.constant_(self.sigma_head.bias, 0.5413)  # softplus(0.5413)≈1.0
        if decompose:
            self.pose_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                           nn.Linear(d_model, 9))  # 6D rot + t
            nn.init.zeros_(self.pose_head[-1].weight); nn.init.zeros_(self.pose_head[-1].bias)
            with torch.no_grad():
                self.pose_head[-1].bias[0] = 1.0; self.pose_head[-1].bias[4] = 1.0
        nn.init.zeros_(self.resid_head.weight); nn.init.zeros_(self.resid_head.bias)

    def forward(self, emb, geom, depth, mask, return_ctx=False):
        # geom in µm -> scale to kµm for stable input
        geom_k = geom / UM
        tok = self.in_proj(torch.cat([emb, geom_k], dim=-1))
        tok = tok + depth_pos_encoding(depth, tok.shape[-1])
        ctx = self.encoder(tok, src_key_padding_mask=mask)  # (B,L,d)
        resid = self.resid_head(ctx)                        # (B,L,3) kµm
        if not self.decompose:
            mu_um = resid * UM  # direct xyz (µm); resid head IS the predictor
        else:
            # rigid pose from masked-mean-pooled context
            valid = (~mask).float().unsqueeze(-1)               # (B,L,1)
            pooled = (ctx * valid).sum(1) / valid.sum(1).clamp(min=1)
            pose = self.pose_head(pooled)
            R = rot6d_to_matrix(pose[:, :6])                    # (B,3,3)
            t = pose[:, 6:9]                                    # (B,3) kµm
            geom3 = torch.stack([geom_k[..., 0], torch.zeros_like(geom_k[..., 0]),
                                 geom_k[..., 1]], dim=-1)       # (B,L,3) [xcol,0,depth] kµm
            rigid = torch.einsum('bij,blj->bli', R, geom3) + t.unsqueeze(1)
            mu_um = (rigid + resid) * UM                         # µm
        if self.gaussian:
            sigma_um = (F.softplus(self.sigma_head(ctx)) + 1e-3) * UM  # (B,L,3) µm, >0
            out = (mu_um, sigma_um)
        else:
            out = mu_um
        # return_ctx exposes the pre-head transformer context for SIGReg
        # (anti-collapse). Default False keeps every eval/inference caller unchanged.
        return (out, ctx) if return_ctx else out


def rot6d_to_matrix(r6):
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


# ───────────────────────────── loss ────────────────────────────────────────
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


def sigreg_loss(z, n_slices=256, n_points=17, t_max=5.0, max_tokens=8192):
    """SIGReg — Sketched Isotropic Gaussian Regularization (LeJEPA, Balestriero &
    LeCun 2025, arXiv:2511.08544). Anti-collapse penalty: pushes the token-embedding
    cloud `z` (N,D) toward an isotropic Gaussian N(0,I) so the regression heads read
    from a full-rank, non-degenerate feature space (counters shrink-to-centroid).

    By Cramér–Wold, a distribution is N(0,I) iff every 1-D projection is N(0,1). We
    sketch `n_slices` random unit directions, project, and score each projection
    against N(0,1) with the differentiable Epps–Pulley statistic (squared distance
    between the empirical characteristic function and e^{-t²/2}, integrated against a
    Gaussian window w(t)=e^{-t²/2} via trapezoid over t∈[-t_max,t_max]). NOT
    standardized — the penalty itself enforces zero mean + unit variance + Gaussianity.
    """
    N, D = z.shape
    if N < 8:
        return z.new_zeros(())
    if N > max_tokens:                       # stochastic subsample -> bound N*M memory
        idx = torch.randperm(N, device=z.device)[:max_tokens]
        z = z[idx]; N = max_tokens
    z = z.float()
    A = torch.randn(D, n_slices, device=z.device, dtype=z.dtype)
    A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-8)   # unit-norm 1-D directions
    proj = z @ A                                          # (N, M) sliced samples
    t = torch.linspace(-t_max, t_max, n_points, device=z.device, dtype=z.dtype)
    phi = torch.exp(-0.5 * t * t)                         # std-normal CF == window w(t)
    # loop over the 17 frequencies so the forward never materialises the (N,M,T)
    # tensor; autograd still saves each (N,M) `tp` for backward (~O(N*M*T) total,
    # ~140MB at the 8192-token cap — negligible). Each term is differentiable.
    diff = []
    for i in range(n_points):
        tp = t[i] * proj                                  # (N, M)
        re = torch.cos(tp).mean(0)                        # (M,) Re of empirical CF
        im = torch.sin(tp).mean(0)                        # (M,) |Im|; squared below
        diff.append((re - phi[i]) ** 2 + im ** 2)         # (M,) |ecf(t)-N(0,1)|²
    diff_sq = torch.stack(diff, dim=1)                    # (M, T)
    T_m = torch.trapz(diff_sq * phi, t, dim=1)            # (M,) per-slice Epps–Pulley
    return T_m.mean()                                     # sketch average


# ───────────────────────────── eval ────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loaders, device):
    """Aggregate over one loader or a list of loaders.
    Returns (mse_kµm²  [same scale as the train loss], folded_RMSE_µm)."""
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]
    model.eval(); se_fold = 0.0; sse = 0.0; n = 0
    for loader in loaders:
        for emb, geom, depth, true, mask, _ in loader:
            emb, geom, depth, true, mask = [x.to(device) for x in (emb, geom, depth, true, mask)]
            out = model(emb, geom, depth, mask)
            pred = out[0] if isinstance(out, tuple) else out   # μ
            valid = (~mask).float().unsqueeze(-1)
            a = fold_lr_t(pred); t = fold_lr_t(true)
            se_fold += (((a - t) ** 2).sum(-1, keepdim=True) * valid).sum().item()
            pk = pred / UM; tk = true / UM
            sse += (((pk - tk) ** 2).sum(-1, keepdim=True) * valid).sum().item()
            n += valid.sum().item()
    n = max(n, 1)
    return sse / n / 3.0, math.sqrt(se_fold / n)  # (mse kµm², folded RMSE µm)


def fold_lr_t(xyz):
    out = xyz.clone()
    out[..., 2] = ALLEN_MIDLINE_UM - (out[..., 2] - ALLEN_MIDLINE_UM).abs()
    return out


@torch.no_grad()
def write_predictions(model, ds, device, out_dir, split, batch_size=4):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    preds, trues, fids, metas, sigmas = [], [], [], [], []
    # fine_id is not in the seq cache; recover from meta via dataset rows if present
    for emb, geom, depth, true, mask, items in loader:
        emb, geom, depth, mask = [x.to(device) for x in (emb, geom, depth, mask)]
        out = model(emb, geom, depth, mask)
        mu, sg = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
        mu = mu.cpu().numpy(); sg = sg.cpu().numpy() if sg is not None else None
        for b, it in enumerate(items):
            n = len(it["emb"])
            preds.append(mu[b, :n]); trues.append(it["true"])
            metas.append(it["meta"]); fids.append(np.zeros(n, np.int64))  # fid filled below=0 (Kabsch ignores)
            sigmas.append(sg[b, :n] if sg is not None else np.zeros((n, 3), np.float32))
    P = np.concatenate(preds).astype(np.float32); T = np.concatenate(trues).astype(np.float32)
    M = np.concatenate(metas); Fd = np.concatenate(fids); S = np.concatenate(sigmas).astype(np.float32)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    out = os.path.join(out_dir, "predictions", f"{split}_predictions.npz")
    np.savez_compressed(out, pred_xyz=P, pred_xyz_gauss=P,
                        pred_sigma_gauss=S,
                        true_xyz=T, fine_id=Fd, meta_str=M.astype("U200"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True, help="extract_probe_sequences.py output")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--decompose", action="store_true", help="rigid-pose + residual head")
    p.add_argument("--lam_smooth", type=float, default=0.0)
    p.add_argument("--lam_mono", type=float, default=0.0)
    p.add_argument("--lam_var", type=float, default=0.0,
                   help="anti-collapse spread-matching penalty weight (kµm hinge)")
    p.add_argument("--lam_sigreg", type=float, default=0.0,
                   help="SIGReg (LeJEPA isotropic-Gaussian) anti-collapse weight on the "
                        "transformer context embeddings. 0 disables. Try 0.05-0.3.")
    p.add_argument("--sigreg_slices", type=int, default=256,
                   help="# random 1-D projections for the SIGReg sketch")
    p.add_argument("--sigreg_points", type=int, default=17,
                   help="# Epps-Pulley quadrature frequencies (t in [-5,5])")
    p.add_argument("--sigreg_max_tokens", type=int, default=8192,
                   help="cap on valid tokens fed to SIGReg per step (memory bound)")
    p.add_argument("--standardize_emb", action="store_true",
                   help="per-dim z-score the frozen embeddings with TRAIN stats before the "
                        "decoder. Fixes scaled-SSL conditioning (301k/1M rms ~60x < 75k); "
                        "raw decoder ignores the tiny embeddings. Stats saved in ckpt.")
    p.add_argument("--whiten_emb", action="store_true",
                   help="FULL-COVARIANCE whitening (TRAIN stats) instead of the diagonal "
                        "z-score: x' = (x-mu) @ W^T, W = S^-1/2. z-score fixes magnitude only; "
                        "whitening also decorrelates the variance-concentrated embeddings "
                        "(participation ratio 4-7 of 768). Mutually exclusive with "
                        "--standardize_emb. Stats saved in ckpt.")
    p.add_argument("--whiten_mode", choices=["zca", "pca"], default="zca",
                   help="zca = minimal-rotation whitening (comparable to z-score); "
                        "pca = axis-aligned to principal directions")
    p.add_argument("--whiten_shrink", default="lw",
                   help="'lw' = analytic Ledoit-Wolf intensity (default; selects nothing on "
                        "any metric), or a float in [0,1] for a fixed shrinkage toward "
                        "(tr(S)/D)*I. Needed because S is near-singular: unregularised S^-1/2 "
                        "would amplify ~760 near-noise directions to unit variance.")
    p.add_argument("--early_stop", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_len", type=int, default=384,
                   help="max channels per probe sequence (even-subsampled if longer). "
                        "Raise above the deepest probe (test max 3338) for FULL coverage.")
    p.add_argument("--eval_ckpt", type=str, default="",
                   help="if set: skip training, load this ckpt, just write test predictions "
                        "(re-score an existing model at a different --max_len / full coverage).")
    p.add_argument("--test_every", type=int, default=5,
                   help="every N epochs, also evaluate the eval_v2 test folds (train/val/test "
                        "curves -> train_metrics.csv). 0 disables. Pre-Kabsch per-sample metric.")
    p.add_argument("--gaussian", action="store_true",
                   help="predict per-channel (μ, σ) and train with diagonal Gaussian NLL "
                        "(writes real pred_sigma_gauss). Default: point head + MSE.")
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Device {device}  decompose={args.decompose}  d_model={args.d_model} layers={args.layers}")

    # per-dim embedding standardization stats (from TRAIN; from ckpt in eval mode).
    # In eval, a ckpt trained standardized MUST be re-scored with its saved stats even
    # if --standardize_emb is omitted, else raw/standardized mismatch silently corrupts.
    emb_mean = emb_std = emb_W = None
    if args.standardize_emb and args.whiten_emb:
        raise SystemExit("--standardize_emb and --whiten_emb are mutually exclusive")
    train_npz = os.path.join(args.cache_dir, "probe_train_seq.npz")
    if args.eval_ckpt:
        _st = torch.load(args.eval_ckpt, map_location="cpu")
        if _st.get("emb_W") is not None:
            emb_mean = np.asarray(_st["emb_mean"], np.float32)
            emb_W = np.asarray(_st["emb_W"], np.float32)
            log("[whiten_emb] applying whitening matrix saved in eval_ckpt")
        elif _st.get("emb_mean") is not None:
            emb_mean = np.asarray(_st["emb_mean"], np.float32)
            emb_std = np.asarray(_st["emb_std"], np.float32)
            log("[standardize_emb] applying per-dim stats saved in eval_ckpt")
        elif args.standardize_emb or args.whiten_emb:
            log("[emb_transform] WARNING: transform IGNORED — eval_ckpt was trained RAW "
                "(no saved stats); transforming eval input would mismatch the model.")
        del _st
    elif args.whiten_emb:
        _shrink = args.whiten_shrink if args.whiten_shrink == "lw" else float(args.whiten_shrink)
        emb_mean, emb_W, _info = emb_whiten_stats(train_npz, mode=args.whiten_mode, shrink=_shrink)
        log(f"[whiten_emb] {_info['mode'].upper()} whitening ON  shrink_alpha={_info['alpha']:.4g} "
            f"({'Ledoit-Wolf' if _shrink == 'lw' else 'fixed'})  "
            f"participation_ratio={_info['participation_ratio']:.1f}/{_info['dim']}  "
            f"cond: {_info['cond_raw']:.3g} -> {_info['cond_shrunk']:.3g}  n_train={_info['n_train']}  "
            f"out_rms {_info['rms_before_rescale']:.3f} -> 1.000 (rescaled: alpha must not move input scale)")
    elif args.standardize_emb:
        emb_mean, emb_std = emb_perdim_stats(train_npz)
        _rms = float(np.sqrt((emb_std**2 + emb_mean**2).mean()))   # true emb rms (E[x^2]=var+mean^2)
        log(f"[standardize_emb] per-dim z-score ON  (train emb rms ~{_rms:.4f})")
    EK = dict(emb_mean=emb_mean, emb_std=emb_std, emb_W=emb_W)  # -> every ProbeSequenceDataset

    tr = ProbeSequenceDataset(train_npz, max_len=args.max_len, **EK)
    va = ProbeSequenceDataset(os.path.join(args.cache_dir, "probe_val_seq.npz"), max_len=args.max_len, **EK)
    log(f"train probes {len(tr)}  val probes {len(va)}  H={tr.H}  max_len={args.max_len}")
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = ProbeSequenceModel(tr.H, args.d_model, args.nhead, args.layers,
                               dropout=args.dropout, decompose=args.decompose,
                               gaussian=args.gaussian).to(device)
    log(f"params {sum(p.numel() for p in model.parameters()):,}  gaussian={args.gaussian}")
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)

    if args.eval_ckpt:
        # eval-only: re-score an already-trained model (e.g. at full coverage)
        state = torch.load(args.eval_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        log(f"[eval-only] loaded {args.eval_ckpt}  "
            f"(val_rmse={state.get('val_rmse', '?')}  ep={state.get('epoch', '?')})  "
            f"writing test predictions at max_len={args.max_len}…")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # eval_v2 test-fold loaders for train/val/test monitoring (periodic, pre-Kabsch)
        test_loaders = []
        if args.test_every > 0:
            for i in range(5):
                fp = os.path.join(args.cache_dir, f"test_fold_{i}_seq.npz")
                if os.path.exists(fp):
                    test_loaders.append(DataLoader(
                        ProbeSequenceDataset(fp, max_len=args.max_len, **EK),
                        batch_size=args.batch_size, shuffle=False, collate_fn=collate))
            log(f"test monitoring: {len(test_loaders)} eval_v2 fold(s) every {args.test_every} ep")
        mf = open(os.path.join(args.out_dir, "train_metrics.csv"), "w")
        mf.write("epoch,train_mse,val_mse,val_rmse_fold,test_mse,test_rmse_fold,sigreg\n"); mf.flush()
        best = float("inf"); no_imp = 0
        for ep in range(args.epochs):
            model.train(); tot = 0.0; nb = 0; totsr = 0.0
            for emb, geom, depth, true, mask, _ in tl:
                emb, geom, depth, true, mask = [x.to(device) for x in (emb, geom, depth, true, mask)]
                opt.zero_grad(set_to_none=True)
                out = model(emb, geom, depth, mask, return_ctx=(args.lam_sigreg > 0))
                if args.lam_sigreg > 0:
                    out, ctx = out
                mu, sigma = out if isinstance(out, tuple) else (out, None)
                loss, mse, sm, mo, vr = seq_loss(mu, true, depth, mask, args.lam_smooth,
                                                 args.lam_mono, sigma_um=sigma,
                                                 lam_var=args.lam_var)
                if args.lam_sigreg > 0:
                    sr = sigreg_loss(ctx[~mask], n_slices=args.sigreg_slices,
                                     n_points=args.sigreg_points,
                                     max_tokens=args.sigreg_max_tokens)
                    loss = loss + args.lam_sigreg * sr
                    totsr += float(sr)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); tot += mse; nb += 1
            train_mse = tot / max(nb, 1); sigreg_ep = totsr / max(nb, 1)
            val_mse, v = evaluate(model, vl, device)
            tmse = trmse = None
            if test_loaders and ((ep + 1) % args.test_every == 0 or ep == args.epochs - 1):
                tmse, trmse = evaluate(model, test_loaders, device)
            msg = (f"ep {ep+1}/{args.epochs}  train_mse={train_mse:.4f}  "
                   f"val_mse={val_mse:.4f} val_RMSE_fold={v:.1f}µm")
            if args.lam_sigreg > 0:
                msg += f"  sigreg={sigreg_ep:.4f}"
            if trmse is not None:
                msg += f"  | test_mse={tmse:.4f} test_RMSE_fold={trmse:.1f}µm"
            log(msg)
            mf.write(f"{ep+1},{train_mse:.5f},{val_mse:.5f},{v:.3f},"
                     f"{'' if tmse is None else format(tmse, '.5f')},"
                     f"{'' if trmse is None else format(trmse, '.3f')},"
                     f"{sigreg_ep:.5f}\n"); mf.flush()
            if v < best - 1e-3:
                best = v; no_imp = 0
                torch.save({"model": model.state_dict(), "args": vars(args), "val_rmse": v, "epoch": ep,
                            # torch tensors (not numpy) so best.pt stays weights_only-safe to load
                            "emb_mean": None if emb_mean is None else torch.from_numpy(emb_mean),
                            "emb_std": None if emb_std is None else torch.from_numpy(emb_std),
                            "emb_W": None if emb_W is None else torch.from_numpy(emb_W)},
                           os.path.join(ckpt_dir, "best.pt"))
                log(f"  best val_RMSE={best:.1f} — saved")
            else:
                no_imp += 1
                if args.early_stop > 0 and no_imp >= args.early_stop:
                    log(f"early stop ep {ep+1}"); break
        mf.close()
        log(f"train/val/test curves -> {os.path.join(args.out_dir, 'train_metrics.csv')}")

        # eval test folds with best ckpt
        state = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device)
        model.load_state_dict(state["model"])
        log(f"\nBest val_RMSE_fold={state['val_rmse']:.1f}µm (ep {state['epoch']}). Writing test predictions…")
    _splits = ["probe_train", "probe_val"] + [f"test_fold_{i}" for i in range(5)]
    for split in _splits:
        fp = os.path.join(args.cache_dir, f"{split}_seq.npz")
        if not os.path.exists(fp):
            continue
        ds = ProbeSequenceDataset(fp, max_len=args.max_len, **EK)
        out = write_predictions(model, ds, device, args.out_dir, split)
        log(f"  {split}: {len(ds)} probes -> {out}")
    log(f"\nDone. Next: PRED_DIR={args.out_dir}/predictions python3 kabsch_eval_1M_pathb.py")


if __name__ == "__main__":
    main()
