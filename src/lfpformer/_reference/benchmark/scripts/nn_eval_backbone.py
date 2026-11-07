"""Unified NN-eval backbone loader supporting wav2vec2 (ws1) and
ViT-rival / AnyNet-rival (ws3).

Each backbone exposes the same `.encode(input_values)` interface and a
common `embed(signals, ...)` driver that handles:
  - wav2vec2: 3750-sample (3s @ 1250 Hz) chunks resampled to 16 kHz.
  - vit_rival / anynet_rival: native-SR (1250 Hz) waveform; trained on 1s
    windows. We split the 3750-sample chunks into 3 × 1250 sub-windows,
    encode each, mean-pool — this matches the training-distribution length
    while preserving NN's 3-second protocol.

Returns embeddings as float32 numpy arrays of shape (N, D).
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

VSP7230_ROOT = "/workspace/ws3/BrainSTORM"
SD5963_BENCH = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/Lfp2vec_benchmarks')
SD5963_ROOT = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging')


def _load_module_from_path(mod_name, file_path):
    """Load a Python module from an absolute file path, bypassing sys.path
    so it cannot be shadowed by an already-imported package of the same
    parent name (e.g. `backbones` from ws1's tree clobbering ws3's
    `backbones.vit_rival_backbone`)."""
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {mod_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_paths(backbone_type):
    """Add the right tree to sys.path for each backbone family."""
    if backbone_type == "wav2vec2":
        for p in (SD5963_BENCH, SD5963_ROOT):
            if p not in sys.path:
                sys.path.insert(0, p)
    elif backbone_type in ("vit_rival", "anynet_rival"):
        # ws3 BrainSTORM has its own backbones/__init__.py.
        # An ws1 `backbones` package may already be cached from an
        # earlier top-level import (eval_nn_zero_shot_centroid imports
        # backbones.wav2vec2_backbone at module load); evict it so the
        # ws3 tree wins on re-import.
        for k in [m for m in list(sys.modules)
                  if m == "backbones" or m.startswith("backbones.")]:
            del sys.modules[k]
        for p in (SD5963_BENCH, SD5963_ROOT):
            while p in sys.path:
                sys.path.remove(p)
        if VSP7230_ROOT not in sys.path:
            sys.path.insert(0, VSP7230_ROOT)
    else:
        raise ValueError(f"unknown backbone_type {backbone_type}")


def load_backbone(backbone_type, ckpt_path, config_path, device,
                   rand_init=False):
    """Return (backbone, embed_dim, native_sr_in, needs_resample_to,
                chunk_seconds_for_encode) for the requested backbone.

    If rand_init=True, build the backbone from `config_path` (yaml) only and
    skip ckpt loading entirely. Used for from-scratch comparisons.
    """
    _ensure_paths(backbone_type)
    if rand_init:
        _log(f"loading backbone={backbone_type}  RAND-INIT (no ckpt)")
        ck = None
    else:
        _log(f"loading backbone={backbone_type}  ckpt={ckpt_path}")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if backbone_type == "wav2vec2":
        from backbones.wav2vec2_backbone import Wav2Vec2Backbone
        if rand_init:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        else:
            if "backbone_state_dict" not in ck or "config" not in ck:
                raise RuntimeError(
                    "wav2vec2 ckpt missing backbone_state_dict / config")
            cfg = dict(ck["config"])
        cfg["rand_init"] = True
        cfg["queue_size"] = 0
        backbone = Wav2Vec2Backbone(cfg)
        if rand_init:
            missing, unexpected = [], []
        else:
            missing, unexpected = backbone.load_state_dict(
                ck["backbone_state_dict"], strict=False)
            critical = (
                "model.wav2vec2.encoder",
                "model.wav2vec2.feature_extractor",
                "model.wav2vec2.feature_projection",
                "model.wav2vec2.masked_spec_embed",
            )
            crit_missing = [k for k in missing if k.startswith(critical)]
            if crit_missing:
                raise RuntimeError(
                    f"wav2vec2 backbone missing critical keys "
                    f"({len(crit_missing)}): {crit_missing[:5]}")
        embed_dim = 768
        native_sr_in = 1250
        needs_resample_to = 16000
        chunk_seconds = None  # use full 3s chunk

    elif backbone_type == "vit_rival":
        _vit_mod = _load_module_from_path(
            "ws3_vit_rival_backbone",
            os.path.join(VSP7230_ROOT, "backbones",
                         "vit_rival_backbone.py"))
        ViTRivalBackbone = _vit_mod.ViTRivalBackbone
        if config_path is None:
            raise ValueError("vit_rival requires --config")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        backbone = ViTRivalBackbone(cfg)
        if rand_init:
            missing, unexpected = [], []
        else:
            if "backbone_state_dict" not in ck:
                raise RuntimeError("vit_rival ckpt missing backbone_state_dict")
            missing, unexpected = backbone.load_state_dict(
                ck["backbone_state_dict"], strict=True)
        embed_dim = int((ck or {}).get("hidden_dim", cfg.get("hidden_size", 256)))
        native_sr_in = int(cfg.get("data_sr", 1250))
        needs_resample_to = None
        chunk_seconds = 1.0

    elif backbone_type == "anynet_rival":
        from backbones.anynet_rival_backbone import AnyNetRivalBackbone
        if config_path is None:
            raise ValueError("anynet_rival requires --config")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        backbone = AnyNetRivalBackbone(cfg)
        if rand_init:
            missing, unexpected = [], []
        else:
            if "backbone_state_dict" not in ck:
                raise RuntimeError(
                    "anynet_rival ckpt missing backbone_state_dict")
            missing, unexpected = backbone.load_state_dict(
                ck["backbone_state_dict"], strict=True)
        embed_dim = int((ck or {}).get("hidden_dim", cfg.get("hidden_size", 320)))
        native_sr_in = int(cfg.get("data_sr", 1250))
        needs_resample_to = None
        chunk_seconds = 1.0

    else:
        raise ValueError(f"unknown backbone_type {backbone_type}")

    _log(f"  loaded: missing={len(missing)} unexpected={len(unexpected)} "
         f"embed_dim={embed_dim}")
    backbone.to(device).eval()
    return backbone, embed_dim, native_sr_in, needs_resample_to, chunk_seconds


@torch.no_grad()
def embed_signals(backbone, signals, src_sr, needs_resample_to,
                  chunk_seconds, embed_dim, device, batch_size=64):
    """Produce (N, embed_dim) float32 embeddings.

    For wav2vec2: full 3s chunk -> resample to 16 kHz -> backbone.encode().
    For ViT/AnyNet: split each 3s chunk into floor(3 / chunk_seconds)
                    sub-windows of (chunk_seconds * src_sr) samples,
                    encode each, mean-pool over sub-windows.
    """
    n = len(signals)
    out = np.empty((n, embed_dim), dtype=np.float32)
    t0 = time.time()
    total = (n + batch_size - 1) // batch_size

    for i in range(0, n, batch_size):
        wav = torch.from_numpy(signals[i:i + batch_size]).to(
            device, non_blocking=True)

        if needs_resample_to is not None and src_sr != needs_resample_to:
            wav = torchaudio.functional.resample(wav, src_sr,
                                                 needs_resample_to)

        if chunk_seconds is None:
            emb_b = backbone.encode(wav)
            if emb_b.ndim == 3:
                emb_b = emb_b.mean(dim=1)
        else:
            # Sub-window mean-pool path (rivals).
            sub_len = int(round(chunk_seconds * src_sr))
            T = wav.shape[1]
            n_sub = T // sub_len
            if n_sub < 1:
                raise RuntimeError(
                    f"signal too short ({T}) for sub_len={sub_len}")
            B = wav.shape[0]
            sub = wav[:, : n_sub * sub_len].reshape(B * n_sub, sub_len)
            emb_sub = backbone.encode(sub)
            if emb_sub.ndim == 3:
                emb_sub = emb_sub.mean(dim=1)
            emb_b = emb_sub.reshape(B, n_sub, -1).mean(dim=1)

        out[i:i + len(wav)] = emb_b.float().cpu().numpy()
        bi = i // batch_size + 1
        if bi % 50 == 0 or bi == total:
            dt = time.time() - t0
            _log(f"    embed {bi}/{total}  {dt:.1f}s  "
                 f"({(i+len(wav))/dt:.0f}/s)")
    return out


def encode_batch_for_train(backbone, wav, src_sr, needs_resample_to,
                           chunk_seconds):
    """Differentiable forward for full-FT training. Returns (B, D)."""
    if needs_resample_to is not None and src_sr != needs_resample_to:
        wav = torchaudio.functional.resample(wav, src_sr, needs_resample_to)

    if chunk_seconds is None:
        emb = backbone.encode(wav)
        if emb.ndim == 3:
            emb = emb.mean(dim=1)
        return emb

    sub_len = int(round(chunk_seconds * src_sr))
    T = wav.shape[1]
    n_sub = T // sub_len
    B = wav.shape[0]
    sub = wav[:, : n_sub * sub_len].reshape(B * n_sub, sub_len)
    emb_sub = backbone.encode(sub)
    if emb_sub.ndim == 3:
        emb_sub = emb_sub.mean(dim=1)
    return emb_sub.reshape(B, n_sub, -1).mean(dim=1)
