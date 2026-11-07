import json
import numpy as np
import pytest
from lfpformer.chunking import chunk_lfp
from lfpformer.cli import main, ROUTES
from lfpformer.paths import REFERENCE


def test_chunk_full_endpoint_tail_order():
    x = np.arange(22).reshape(2, 11)
    rows = list(chunk_lfp(x, sample_rate=2, seconds=2))
    assert [(ch, start) for _, ch, start in rows] == [(0, 0), (1, 0), (0, 4), (1, 4)]
    np.testing.assert_array_equal(rows[-1][0], x[1, 4:8])
    assert rows[0][0].dtype == np.float32


def test_chunk_rejects_fractional_samples():
    with pytest.raises(ValueError):
        list(chunk_lfp(np.zeros((2, 10)), sample_rate=3, seconds=0.5))


def test_all_reference_routes_exist():
    for stack, path in ROUTES.values():
        assert (REFERENCE / stack / path).is_file()


def test_cli_kabsch_never_uses_truth_for_fit(tmp_path):
    from lfpformer.geometry import np1_geometry

    ch = np.repeat(np.arange(12), 2)
    geom = np1_geometry(ch)
    mu = geom + [3000, 2000, 5000]
    for shift in [0, 10000]:
        source = tmp_path / f"in{shift}.npz"
        output = tmp_path / f"out{shift}.npz"
        np.savez(
            source,
            mu=mu,
            sigma=np.ones_like(mu),
            geometry_xyz=geom,
            probe_ids=np.repeat("a", len(ch)),
            channel_ids=ch,
            true_xyz=mu + shift,
        )
        main(
            [
                "postprocess",
                "--input",
                str(source),
                "--output",
                str(output),
                "--method",
                "ibl-kabsch",
            ]
        )
    with np.load(tmp_path / "out0.npz") as a, np.load(tmp_path / "out10000.npz") as b:
        np.testing.assert_array_equal(a["mu"], b["mu"])
        assert len(a["mu"]) == 12
    main(
        [
            "score",
            "--input",
            str(tmp_path / "out0.npz"),
            "--output",
            str(tmp_path / "metrics.json"),
        ]
    )
    assert json.loads((tmp_path / "metrics.json").read_text())["n_channels"] == 12


def test_cli_rejects_truth_varying_within_channel(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(
        p,
        mu=np.zeros((2, 3)),
        true_xyz=np.array([[0, 0, 0], [1, 0, 0]]),
        probe_ids=["a", "a"],
        channel_ids=[1, 1],
    )
    with pytest.raises(SystemExit):
        main(
            [
                "postprocess",
                "--input",
                str(p),
                "--output",
                str(tmp_path / "out.npz"),
                "--method",
                "pool",
            ]
        )
