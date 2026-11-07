"""Windowing of already preprocessed channel × time LFP arrays."""

import numpy as np


def chunk_lfp(
    signal, sample_rate=1250, seconds=3.0, stride_seconds=None, start_seconds=0.0
):
    """Yield (window, channel, sample_start); incomplete trailing windows are dropped.

    This operation does not destripe/filter raw recordings. Use dataset preprocessing
    first. Windows are emitted time-major, then channel-major, with exact sample indices.
    """
    x = np.asarray(signal)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite channels × samples")
    if sample_rate <= 0 or seconds <= 0 or start_seconds < 0:
        raise ValueError("Invalid sampling/window parameters")

    def samples(t):
        n = t * sample_rate
        if abs(n - round(n)) > 1e-7:
            raise ValueError("Time does not map to an integer sample count")
        return int(round(n))

    width = samples(seconds)
    stride = samples(seconds if stride_seconds is None else stride_seconds)
    start = samples(start_seconds)
    if width < 1 or stride < 1:
        raise ValueError("Window and stride must be positive")
    for i in range(start, x.shape[1] - width + 1, stride):
        for ch in range(len(x)):
            yield x[ch, i : i + width].astype(np.float32), ch, i
