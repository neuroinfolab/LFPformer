"""Shared preprocessing utilities for all backbones."""

import torch
import torchaudio.functional as AF

INPUT_SR = 1250
MODEL_SR = 16000
ZSCORE_EPS = 1e-8

# Global amplitude scaling factor for Whisper backbone.
# IBL LFP signals have std ≈ 3.8e-5 (median across 25K dataset).
# Whisper's log-mel spectrogram uses per-sample `max - 8.0 dB` clipping,
# which preserves intra-sample structure but places tiny LFP signals in a
# range that overlaps with the silence/padding floor.
#
# Multiplying by this constant maps the median LFP std to ~0.1 (audio-like),
# bringing signals into Whisper's natural operating range while preserving
# amplitude differences between samples (unlike per-sample z-score which
# destroys cross-sample amplitude information).
#
# Computed from compact_atlas_ibl_25k: median_std=3.8e-5, target_std=0.1
# → scale = 0.1 / 3.8e-5 ≈ 2661, rounded to 3000.
LFP_GLOBAL_SCALE = 3000.0


def resample_to_16k(x: torch.Tensor) -> torch.Tensor:
    """Resample from 1250 Hz to 16000 Hz on GPU."""
    return AF.resample(x, orig_freq=INPUT_SR, new_freq=MODEL_SR)


def zscore_per_sample(x: torch.Tensor, eps: float = ZSCORE_EPS) -> torch.Tensor:
    """Z-score each sample independently over time dimension.
    x: [B, T] → mean≈0, std≈1 per sample.

    WARNING: This destroys cross-sample amplitude information. Prefer
    global_scale() for Whisper backbone unless you specifically need
    per-sample normalization.
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False)
    return (x - mean) / std.clamp_min(eps)


def global_scale(x: torch.Tensor, scale: float = LFP_GLOBAL_SCALE) -> torch.Tensor:
    """Scale LFP amplitudes to audio-like range for Whisper mel spectrogram.

    Unlike z-score, this preserves amplitude differences between samples
    (e.g., different brain regions with different baseline noise levels).

    Args:
        x: [B, T] raw waveform
        scale: multiplicative factor (default: 3000.0 for IBL LFP)

    Returns:
        [B, T] scaled waveform
    """
    return x * scale


def preprocess_batch(
    x: torch.Tensor, needs_resampling: bool, normalize: bool = False
) -> torch.Tensor:
    """Preprocess a batch based on backbone requirements.

    Args:
        x: [B, T] raw waveform at 1250 Hz
        needs_resampling: True for wav2vec2/WavLM (→16kHz), False for PatchTST/MOMENT/TTM
        normalize: If True, apply global amplitude scaling for Whisper backbone.
            Brings IBL LFP amplitudes (~0.0001) into Whisper's expected range
            while preserving cross-sample amplitude differences. Not needed for
            wav2vec2/WavLM which have learned CNN feature extractors.

    Returns:
        [B, T'] preprocessed waveform
    """
    if normalize:
        x = global_scale(x)
    if needs_resampling:
        x = resample_to_16k(x)
    return x
