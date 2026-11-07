import ast
from collections import defaultdict
import numpy as np
import pytest
from lfpformer.geometry import (
    fold_left,
    rigid_fit,
    fit_probe_geometry,
    project_probe_lines,
    np1_geometry,
)
from lfpformer.pooling import pool_channels
from lfpformer.metrics import (
    localization_metrics,
    seed_average_probe_metrics,
    balanced_accuracy,
)
from lfpformer.atlas import lookup_labels
from lfpformer.paths import REFERENCE


def original_functions(filename, names, namespace):
    tree = ast.parse(filename.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(selected) == len(names)
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(filename), "exec"),
        namespace,
    )
    return namespace


def test_fold_only_ml_and_idempotent():
    x = np.array([[100, 200, 7000], [9000, 5000, 2000.0]])
    y = fold_left(x)
    np.testing.assert_array_equal(y[:, :2], x[:, :2])
    np.testing.assert_array_equal(y[:, 2], [4478, 2000])
    np.testing.assert_array_equal(fold_left(y), y)
    assert x[0, 2] == 7000


@pytest.mark.parametrize("flip", [False, True])
def test_kabsch_and_pool_match_original_paper(flip):
    ns = {"np": np, "defaultdict": defaultdict, "ALLEN_MIDLINE_UM": 5739.0}
    original_functions(
        REFERENCE / "benchmark/scripts/recompute_refined_3d_metrics.py",
        {"channel_xy_onprobe"},
        ns,
    )
    original_functions(
        REFERENCE / "benchmark/scripts/kabsch_gauss_chanpool_eval.py",
        {"pool_channels", "kabsch_weighted", "per_probe_kabsch"},
        ns,
    )
    rng = np.random.default_rng(11)
    ch = np.tile(np.repeat(np.arange(24), 3), 2)
    ids = np.repeat(["a", "b"], 72)
    P = np1_geometry(ch)
    R = np.array([[1, 0, 0], [0, 0, 1 if flip else -1], [0, -1 if flip else 1, 0]])
    true = P @ R.T + [4000, 5000, 3000]
    mu = true + rng.normal(0, 25, true.shape)
    sig = rng.uniform(0.001, 200, true.shape)
    p, q, w, t, pids = ns["pool_channels"](mu, sig, true, ids, ch, "sigma")
    z = pool_channels(mu, ids, ch, sig)
    np.testing.assert_allclose(z["mu"], q, rtol=1e-12)
    np.testing.assert_allclose(np1_geometry(z["channel_ids"]), p)
    expected, n, _ = ns["per_probe_kabsch"](p, q, w, pids, False)
    actual, n_actual = fit_probe_geometry(p, z["mu"], z["probe_ids"])
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    assert n_actual == n == 2


def test_rigid_fit_rotation_and_translation():
    p = np.random.default_rng(2).normal(size=(30, 3))
    r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    q = p @ r.T + [1, 2, 3]
    rr, t = rigid_fit(p, q)
    np.testing.assert_allclose(p @ rr.T + t, q, atol=1e-12)
    assert np.linalg.det(rr) == pytest.approx(1)


def test_small_probe_preserves_prediction():
    p = np1_geometry(np.arange(3))
    q = p + 8
    out, count = fit_probe_geometry(p, q, ["a"] * 3)
    np.testing.assert_array_equal(out, q)
    assert count == 0


def test_line_projection_is_principal_axis_and_preserves_center():
    p = np.array([[0, 0, 1], [1, 1, -1], [2, 2, -1], [3, 3, 1.0]])
    out = project_probe_lines(p, ["a"] * 4)
    np.testing.assert_allclose(out.mean(0), p.mean(0), atol=1e-12)
    assert np.linalg.matrix_rank(out - out.mean(0), tol=1e-10) == 1
    np.testing.assert_allclose(project_probe_lines(out, ["a"] * 4), out, atol=1e-12)


@pytest.mark.parametrize("rule", ["clip", "add"])
def test_axis_specific_weighting_and_broadcast(rule):
    mu = np.array([[0, 0, 0], [10, 10, 10], [8, 9, 10.0]])
    sigma = np.array([[1, 2, 0.0001], [2, 1, 0.0002], [2, 2, 2]])
    z = pool_channels(
        mu, ["z", "z", "a"], np.array([7, 7, 8]), sigma, variance_rule=rule
    )
    var = sigma[:2] ** 2
    weights = 1 / (np.maximum(var, 1e-6) if rule == "clip" else var + 1e-6)
    expected = (mu[:2] * weights).sum(0) / weights.sum(0)
    np.testing.assert_allclose(z["mu"][z["inverse"][:2]], [expected, expected])
    np.testing.assert_array_equal(z["counts"], [1, 2])


@pytest.mark.parametrize(
    "sigma", [np.zeros((2, 3)), np.full((2, 3), np.nan), np.full((2, 3), -1)]
)
def test_bad_sigma_rejected(sigma):
    with pytest.raises(ValueError):
        pool_channels(np.ones((2, 3)), ["a", "a"], np.array([0, 0]), sigma)


def test_unweighted_pool():
    z = pool_channels([[0, 1, 2], [4, 5, 6]], ["a", "a"], np.array([0, 0]))
    np.testing.assert_array_equal(z["mu"], [[2, 3, 4]])


def test_equal_probe_is_not_channel_aggregate():
    q = np.zeros((5, 3))
    p = q.copy()
    p[:4, 0] = 1
    p[4, 0] = 9
    m = localization_metrics(p, q, ["a"] * 4 + ["b"], fold=False)
    assert m["equal_probe_mean_rmse_um"] == 5
    assert m["channel_aggregate_rmse_um"] == pytest.approx(np.sqrt(17))
    assert m["equal_probe_sd_rmse_um"] == pytest.approx(np.sqrt(32))


def test_seed_average_before_across_probe_sd():
    m = seed_average_probe_metrics([[1, 9], [3, 7]])
    assert m["mean_um"] == 5
    assert m["sd_across_probes_um"] == pytest.approx(np.sqrt(18))


def test_outside_atlas_not_clipped():
    volume = np.arange(8).reshape(2, 2, 2)
    labels = lookup_labels(
        [[0, 0, 0], [25, 25, 25], [-0.01, 0, 0], [50, 0, 0]], volume, 25
    )
    np.testing.assert_array_equal(labels, [0, 7, -1, -1])


def test_balanced_accuracy_counts_failed_lookups():
    m = balanced_accuracy([1, 1, 1, 2], [1, 1, -1, 2], [1, 2, 3])
    assert m["balanced_accuracy"] == pytest.approx(5 / 6)
    assert m["n_present_classes"] == 2
    with pytest.raises(ValueError):
        balanced_accuracy([99], [1], [1, 2])
