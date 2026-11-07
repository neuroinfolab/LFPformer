"""Explicit atlas lookup: input coordinates AP,DV,ML in µm; no clipping."""

import numpy as np


def lookup_labels(xyz, label_volume, voxel_um=25.0, origin_um=(0.0, 0.0, 0.0)):
    from .geometry import _xyz

    x = _xyz(xyz)
    v = np.asarray(label_volume)
    if v.ndim != 3 or voxel_um <= 0:
        raise ValueError(
            "Expected 3D label volume in AP,DV,ML order and positive resolution"
        )
    ijk = np.floor((x - np.asarray(origin_um)) / voxel_um).astype(np.int64)
    valid = ((ijk >= 0) & (ijk < np.asarray(v.shape))).all(1)
    labels = np.full(len(x), -1, dtype=np.int64)
    labels[valid] = v[tuple(ijk[valid].T)]
    return labels
