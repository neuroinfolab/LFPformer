import ast, importlib.util, pickle
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from lfpformer.heads import GaussianXYZHead, gaussian_nll
from lfpformer.paths import REFERENCE, activate_reference


def test_actual_allen_mlp_state_and_forward_match():
    path = REFERENCE / "benchmark/scripts/finetune_xyz_gaussian.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "GaussianXYZHead"
    )
    ns = {"nn": torch.nn, "LOG_VAR_MIN": -10.0, "LOG_VAR_MAX": 10.0}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    paper = ns["GaussianXYZHead"](32).eval()
    new = GaussianXYZHead(32).eval()
    new.load_state_dict(paper.state_dict(), strict=True)
    x = torch.randn(4, 32)
    for a, b in zip(paper(x), new(x)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    loss = gaussian_nll(*new(x), torch.randn(4, 3))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters()
    )


def test_compact_shard_boundary_and_coordinates(tmp_path):
    activate_reference("benchmark")
    from blind_localization.data.lazyloader_dataset import CompactDataset

    for i in range(2):
        rows = [
            (
                np.full(3750, 2 * i + j, np.float32),
                f"probe_{j}",
                0,
                np.array([2 * i + j, 1, 2]),
            )
            for j in range(2)
        ]
        with (tmp_path / f"samples_{i}.pkl").open("wb") as f:
            pickle.dump(rows, f)
    with (tmp_path / "compact_index.pkl").open("wb") as f:
        pickle.dump(
            {
                "file_list": ["samples_0.pkl", "samples_1.pkl"],
                "cumulative_counts": np.array([0, 2, 4]),
                "n_samples": 4,
            },
            f,
        )
    ds = CompactDataset(
        str(tmp_path), include_labels=False, gpu_resample=True, verbose=False
    )
    assert len(ds) == 4
    for i in range(4):
        signal, _, coords = ds[i]
        assert signal.shape[-1] == 3750
        torch.testing.assert_close(signal, torch.full_like(signal, i))
        assert coords[0].item() == i


def test_linear_checkpoint_predictor_roundtrip(tmp_path):
    activate_reference("benchmark")
    from linear_head_patched import GaussHead

    file = Path(__file__).resolve().parents[1] / "scripts/predict_head.py"
    spec = importlib.util.spec_from_file_location("predict_head", file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = GaussHead("linear", d=16).eval()
    x = np.random.default_rng(3).normal(size=(7, 16)).astype(np.float32)
    mean = np.array([100, 200, 300], np.float32)
    std = np.array([10, 20, 30], np.float32)
    ck = tmp_path / "best.pt"
    torch.save(
        {"model": model.state_dict(), "kind": "linear", "mu_t": mean, "sd_t": std}, ck
    )
    np.savez(
        tmp_path / "input.npz",
        emb=x,
        probe_ids=np.repeat("p", len(x)),
        channel_ids=np.arange(len(x)),
    )
    mod.predict(tmp_path / "input.npz", ck, tmp_path / "pred.npz", batch_size=3)
    mu, lv = model(torch.from_numpy(x))
    with np.load(tmp_path / "pred.npz") as z:
        np.testing.assert_allclose(z["mu"], mu.detach().numpy() * std + mean, rtol=1e-6)
        np.testing.assert_allclose(
            z["sigma"], np.exp(0.5 * lv.detach().numpy()) * std, rtol=1e-6
        )


def test_offline_embedding_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("torchaudio")
    import yaml
    from lfpformer.inference import extract_embeddings

    activate_reference("benchmark")
    from backbones import get_backbone

    torch.set_num_threads(2)
    cfg = dict(
        backbone="wav2vec2",
        rand_init=True,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        needs_resampling=False,
        mask_time_length=2,
    )
    model = get_backbone("wav2vec2", cfg).eval()
    ck = tmp_path / "encoder.pt"
    torch.save({"backbone_state_dict": model.state_dict()}, ck)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    signals = np.random.default_rng(8).normal(size=(2, 3750)).astype(np.float32)
    np.savez(
        tmp_path / "windows.npz",
        signals=signals,
        probe_ids=["p", "p"],
        channel_ids=[0, 1],
    )
    extract_embeddings(
        tmp_path / "windows.npz", config, ck, tmp_path / "emb.npz", batch_size=1
    )
    with np.load(tmp_path / "emb.npz") as z:
        assert z["emb"].shape == (2, 32)
        assert z["emb"].dtype == np.float16
        assert np.isfinite(z["emb"]).all()
        np.testing.assert_array_equal(z["channel_ids"], [0, 1])
        with torch.inference_mode():
            expected = (
                model.encode(torch.from_numpy(signals)).numpy().astype(np.float16)
            )
        np.testing.assert_allclose(z["emb"], expected, rtol=0.005, atol=0.001)


def test_unlabelled_end_to_end_inference(tmp_path, monkeypatch):
    pytest.importorskip('transformers')
    pytest.importorskip('torchaudio')
    import yaml
    from lfpformer.geometry import np1_geometry
    activate_reference('benchmark')
    from backbones import get_backbone
    from linear_head_patched import GaussHead
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('infer_lfp', scripts / 'infer_lfp.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = dict(backbone='wav2vec2', rand_init=True, hidden_size=32,
               num_hidden_layers=1, num_attention_heads=4, intermediate_size=64,
               needs_resampling=False, mask_time_length=2)
    torch.set_num_threads(2)
    encoder = get_backbone('wav2vec2', cfg)
    head = GaussHead('linear', d=32)
    torch.save({'backbone_state_dict': encoder.state_dict()}, tmp_path / 'encoder.pt')
    torch.save({'model': head.state_dict(), 'kind': 'linear',
                'mu_t': np.array([4000, 5000, 3000]), 'sd_t': np.ones(3)*100},
               tmp_path / 'head.pt')
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(cfg))
    channels = np.tile(np.arange(4), 2)
    np.savez(tmp_path / 'windows.npz',
             signals=np.random.default_rng(8).normal(size=(8,3750)).astype(np.float32),
             probe_ids=np.repeat('p', 8), channel_ids=channels,
             geometry_xyz=np1_geometry(channels))
    result = mod.run(tmp_path / 'windows.npz', tmp_path / 'config.yaml',
                     tmp_path / 'encoder.pt', tmp_path / 'head.pt', tmp_path / 'out')
    with np.load(result) as z:
        assert z['mu'].shape == (4,3)
        assert np.isfinite(z['mu']).all()
        assert z['n_fitted_probes'] == 1
        np.testing.assert_array_equal(z['counts'], [2,2,2,2])
        assert 'true_xyz' not in z
        expected = z['mu'].copy()
    # Anatomical labels must not influence embeddings, decoding or refinement.
    with np.load(tmp_path / 'windows.npz') as z:
        labelled = dict(z)
    labelled['true_xyz'] = np.tile([9000., 100., 10000.], (8, 1))
    labelled['true_labels'] = np.repeat(999, 8)
    np.savez(tmp_path / 'labelled.npz', **labelled)
    result = mod.run(tmp_path / 'labelled.npz', tmp_path / 'config.yaml',
                     tmp_path / 'encoder.pt', tmp_path / 'head.pt', tmp_path / 'labelled_out')
    with np.load(result) as z:
        np.testing.assert_array_equal(z['mu'], expected)
