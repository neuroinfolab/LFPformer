"""Keep channel-aggregated RMSE separate from Figure 3's equal-probe statistic."""

import numpy as np
from .geometry import _xyz, fold_left


def localization_metrics(pred, true, probe_ids, fold=True, midline_um=5739.0):
    p, t = _xyz(pred), _xyz(true)
    ids = np.asarray(probe_ids)
    if p.shape != t.shape or len(ids) != len(p) or not len(p):
        raise ValueError("Nonempty aligned arrays required")
    if fold:
        p, t = fold_left(p, midline_um), fold_left(t, midline_um)
    e2 = ((p - t) ** 2).sum(1)
    probes = np.unique(ids)
    per = np.array([np.sqrt(e2[ids == x].mean()) for x in probes])
    return dict(
        channel_aggregate_rmse_um=float(np.sqrt(e2.mean())),
        equal_probe_mean_rmse_um=float(per.mean()),
        equal_probe_sd_rmse_um=float(per.std(ddof=1)) if len(per) > 1 else None,
        n_probes=len(probes),
        n_channels=len(p),
        probes=probes.astype(str).tolist(),
        per_probe_rmse_um=per.tolist(),
    )


def seed_average_probe_metrics(per_seed_probe_rmse):
    a = np.asarray(per_seed_probe_rmse, float)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError("Expected seeds × aligned probes")
    p = a.mean(0)
    return dict(
        mean_um=float(p.mean()),
        sd_across_probes_um=float(p.std(ddof=1)) if len(p) > 1 else None,
        n_seeds=len(a),
        n_probes=a.shape[1],
    )


def balanced_accuracy(true_labels, pred_labels, class_ids):
    """Macro recall over present true classes in the supplied fixed class universe.

    Predicted -1 (out of atlas) is an error. Unknown true classes are rejected.
    """
    y = np.asarray(true_labels)
    p = np.asarray(pred_labels)
    classes = np.asarray(class_ids)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Aligned nonempty labels required")
    if not np.isin(y, classes).all():
        raise ValueError("True labels outside declared class universe")
    recalls = {
        str(c): float((p[y == c] == c).mean()) for c in classes if (y == c).any()
    }
    return dict(
        balanced_accuracy=float(np.mean(list(recalls.values()))),
        recall_by_present_class=recalls,
        n_present_classes=len(recalls),
        n_declared_classes=len(classes),
    )
