"""Pool repeated windows by physical (probe, channel), separately per XYZ axis."""

import numpy as np
from .geometry import _xyz


def pool_channels(
    mu, probe_ids, channel_ids, sigma=None, variance_rule="clip", epsilon=1e-6
):
    """Return pooled arrays, counts and an index broadcasting channels to windows.

    IBL uses max(sigma², epsilon); historical Allen uses sigma² + epsilon.
    Missing uncertainty means an ordinary arithmetic mean, not invented precision.
    """
    mu = _xyz(mu)
    p = np.asarray(probe_ids).astype(str)
    c = np.asarray(channel_ids)
    if len(mu) != len(p) or len(mu) != len(c) or not len(mu):
        raise ValueError("Nonempty aligned rows required")
    if not np.issubdtype(c.dtype, np.integer):
        raise ValueError("channel_ids must be integers")
    if (c < 0).any():
        raise ValueError("channel_ids must be nonnegative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if variance_rule not in ("clip", "add"):
        raise ValueError("variance_rule must be clip or add")
    if sigma is None:
        w = np.ones_like(mu)
    else:
        s = _xyz(sigma)
        if s.shape != mu.shape or (s <= 0).any():
            raise ValueError("sigma must be positive standard deviation, shape (N,3)")
        var = s * s
        w = 1 / (np.maximum(var, epsilon) if variance_rule == "clip" else var + epsilon)
    key = np.rec.fromarrays([p, c], names="probe,channel")
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    precision = np.zeros((len(first), 3))
    weighted = np.zeros_like(precision)
    np.add.at(precision, inv, w)
    np.add.at(weighted, inv, w * mu)
    return dict(
        mu=weighted / precision,
        probe_ids=p[first],
        channel_ids=c[first],
        counts=np.bincount(inv),
        inverse=inv,
        first=first,
        precision=precision,
    )
