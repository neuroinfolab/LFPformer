"""Label-free inference with the released sequence and fusion checkpoint formats."""
import sys
from pathlib import Path
import numpy as np
import torch
from lfpformer.paths import REFERENCE, activate_reference


def predict(input_path, checkpoint, output, kind, whitening=None, device='cpu'):
    with np.load(input_path, allow_pickle=False) as z:
        data = dict(z)
    x = data.pop('emb').astype(np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty embeddings required')
    n, hidden = x.shape
    for key in ('probe_ids', 'channel_ids'):
        if key not in data or data[key].shape != (n,):
            raise ValueError(f'Aligned {key} required')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    args = ck.get('args', {})
    activate_reference('benchmark')
    if kind == 'fusion-linear':
        if whitening is not None:
            raise ValueError('Historical linear fusion uses raw embeddings, not whitening')
        from eval_manip_ft import make_heads, normalizer_from_stats, apply_coordinate_parameterization
        if not ck.get('manip_conditioned'):
            raise ValueError('Expected manipulator-conditioned checkpoint')
        _, model = make_heads(ck.get('head_type', 'linear'), hidden + 3, device)
        model.load_state_dict(ck['gauss_head_state_dict'], strict=True)
        nz = normalizer_from_stats(ck['normalizer_stats'])
        m = data.get('manip_xyz')
        if m is None or m.shape != (n, 3) or not np.isfinite(m).all():
            raise ValueError('Finite row-aligned manip_xyz (N,3) required')
        mn = nz.encode(m)
        scales = np.array([nz.ap_max-nz.ap_min, nz.dv_max-nz.dv_min, nz.d_max-nz.d_min])
        model.eval()
        mus, sigmas = [], []
        with torch.inference_mode():
            for start in range(0, n, 1024):
                end = start + 1024
                mt = torch.as_tensor(mn[start:end], device=device)
                feat = torch.as_tensor(np.concatenate([x[start:end], mn[start:end]], axis=1), device=device)
                mu, lv = model(feat)
                mu = apply_coordinate_parameterization(mu, mt, ck.get('parameterization', 'residual'))
                mus.append(nz.decode(mu.cpu().numpy()))
                sigmas.append(np.exp(.5 * lv.cpu().numpy()) * scales)
        mu, sigma = np.concatenate(mus), np.concatenate(sigmas)
    else:
        g = data.get('geometry_xyz')
        if g is None or g.shape != (n,3) or not np.isfinite(g).all():
            raise ValueError('Finite geometry_xyz (N,3), columns x,0,depth required')
        if not np.allclose(g[:,1], 0):
            raise ValueError('Sequence geometry convention requires middle column zero')
        if kind == 'transformer':
            from train_probe_sequence_ext import ProbeSequenceModel
            if args.get('arch', 'transformer') != 'transformer' or not args.get('gaussian', False):
                raise ValueError('Expected Gaussian ProbeSequenceModel checkpoint')
            model = ProbeSequenceModel(hidden, args['d_model'], args['nhead'], args['layers'],
                ff=args.get('ff') if args.get('ff') is not None else 512, dropout=args.get('dropout', .1),
                decompose=args.get('decompose', False), gaussian=True)
            has_norm = ck.get('emb_mean') is not None
            if whitening and has_norm:
                raise ValueError('Checkpoint already contains embedding normalization; do not apply whitening twice')
            if has_norm:
                x = x - np.asarray(ck['emb_mean'], dtype=np.float32)
                if ck.get('emb_W') is not None:
                    x = x @ np.asarray(ck['emb_W'], dtype=np.float32).T
                else:
                    x = x / np.asarray(ck['emb_std'], dtype=np.float32)
            elif whitening:
                with np.load(whitening) as w:
                    x = ((x-w['mean']) @ w['W'].T).astype(np.float16).astype(np.float32)
        elif kind == 'fusion-transformer':
            if whitening:
                raise ValueError('Fusion checkpoint includes embedding normalization')
            sys.path.insert(0, str(REFERENCE / 'fusion'))
            from model_base import ManipProbeSequenceModel, fold_lr_t
            state = ck['model']
            if 'embedding_mean' not in state or 'embedding_scale' not in state:
                raise ValueError('Expected folded MatchedDirectProbeSeq checkpoint with saved normalization')
            if args.get('arm') not in ('lfp_only', 'lfp_manip'):
                raise ValueError('Checkpoint must identify lfp_only or lfp_manip arm')
            model = ManipProbeSequenceModel(hidden, args['d_model'], args['nhead'], args['layers'],
                args['ff'], args.get('dropout', .1), decompose=False, parameterization='direct', gaussian=True)
            x = (x-np.asarray(state['embedding_mean']))/np.asarray(state['embedding_scale'])
            m = data.get('manip_xyz')
            if args['arm'] == 'lfp_only':
                m = np.zeros((n,3), dtype=np.float32)
            elif m is None or m.shape != (n,3) or not np.isfinite(m).all():
                raise ValueError('Finite row-aligned manip_xyz (N,3) required')
        else:
            raise ValueError(f'Unknown decoder kind: {kind}')
        state = {k:v for k,v in ck['model'].items() if k not in ('embedding_mean','embedding_scale')}
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        mu, sigma = np.empty((n,3)), np.empty((n,3))
        # Never subsample silently or split a probe into different attention contexts.
        groups = data.get('sequence_ids', data['probe_ids']).astype(str)
        if groups.shape != (n,):
            raise ValueError('sequence_ids must have one value per window')
        with torch.inference_mode():
            for group in np.unique(groups):
                ix = np.flatnonzero(groups == group)
                if len(np.unique(data['probe_ids'][ix])) != 1:
                    raise ValueError('A sequence may contain only one physical probe')
                ix = ix[np.argsort(g[ix,2])]
                if len(ix) > args.get('max_len', 4096):
                    raise ValueError('Sequence exceeds checkpoint max_len; provide explicit sequence_ids for intended contexts')
                emb = torch.as_tensor(x[ix][None], dtype=torch.float32, device=device)
                geom = torch.as_tensor(g[ix][:,[0,2]][None], dtype=torch.float32, device=device)
                depth = geom[...,1]
                mask = torch.zeros((1,len(ix)), dtype=torch.bool, device=device)
                if kind == 'transformer':
                    pred, uncertainty = model(emb, geom, depth, mask)
                else:
                    mt = torch.as_tensor(m[ix][None], dtype=torch.float32, device=device)
                    if args['arm'] == 'lfp_manip':
                        mt = fold_lr_t(mt)
                    pred, uncertainty = model(emb, mt, geom, depth, mask)
                mu[ix], sigma[ix] = pred[0].cpu().numpy(), uncertainty[0].cpu().numpy()
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError('Decoder produced invalid coordinates or uncertainty')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, mu=mu, sigma=sigma, **data)
