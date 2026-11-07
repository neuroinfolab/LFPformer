"""Anatomical coordinates are (AP, DV, ML), in micrometres."""

import numpy as np

MIDLINE_UM = 5739.0


def _xyz(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError("Expected finite array of shape (N, 3)")
    return a


def fold_left(xyz, midline_um=MIDLINE_UM):
    out = _xyz(xyz).copy()
    out[:, 2] = midline_um - np.abs(out[:, 2] - midline_um)
    return out


def rigid_fit(geometry_xyz, predicted_xyz):
    """Fit known probe geometry to predictions; never to ground-truth anatomy."""
    p, q = _xyz(geometry_xyz), _xyz(predicted_xyz)
    if p.shape != q.shape or not len(p):
        raise ValueError("Geometry and predictions must align")
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
    r = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return r, qc - r @ pc


def fit_probe_geometry(
    geometry_xyz, predicted_xyz, probe_ids, orientation="ibl", min_channels=4
):
    """Unweighted per-probe Kabsch, after uncertainty-weighted channel pooling.

    IBL constrains increasing shank depth to point toward decreasing atlas DV.
    'free' omits this orientation correction. Small probes retain pooled predictions.
    """
    p, q = _xyz(geometry_xyz), _xyz(predicted_xyz)
    ids = np.asarray(probe_ids)
    if p.shape != q.shape or len(ids) != len(p):
        raise ValueError("Rows must align")
    if orientation not in ("ibl", "free"):
        raise ValueError("orientation must be ibl or free")
    out = q.copy()
    fitted = 0
    for pid in np.unique(ids):
        ix = np.flatnonzero(ids == pid)
        if len(ix) < min_channels:
            continue
        r, t = rigid_fit(p[ix], q[ix])
        if orientation == "ibl" and (r @ np.array([0.0, 0.0, 1.0]))[1] > 0:
            r = r @ np.diag([1.0, -1.0, -1.0])
            t = q[ix].mean(0) - r @ p[ix].mean(0)
        out[ix] = p[ix] @ r.T + t
        fitted += 1
    return out, fitted


def project_probe_lines(predicted_xyz, probe_ids, min_channels=3):
    """Allen protocol: unweighted PCA line through pooled channel predictions.

    Unlike Kabsch, this does not impose known electrode spacing. No true XYZ is used.
    """
    q = _xyz(predicted_xyz)
    ids = np.asarray(probe_ids)
    if len(ids) != len(q):
        raise ValueError("Rows must align")
    out = q.copy()
    for pid in np.unique(ids):
        ix = np.flatnonzero(ids == pid)
        if len(ix) < min_channels:
            continue
        center = q[ix].mean(0)
        _, _, vt = np.linalg.svd(q[ix] - center, full_matrices=False)
        axis = vt[0]
        out[ix] = center + np.outer((q[ix] - center) @ axis, axis)
    return out


def np1_geometry(channel_ids):
    """Paper's NP1 four-column checkerboard, in µm (x, 0, shaft depth)."""
    c = np.asarray(channel_ids)
    if not np.issubdtype(c.dtype, np.integer) or (c < 0).any():
        raise ValueError("Nonnegative integer channel IDs required")
    row = c // 2
    x = 11.0 + 32.0 * (c % 2) + 16.0 * (row % 2)
    return np.column_stack([x, np.zeros(len(c)), row * 20.0])
