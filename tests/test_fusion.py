"""Behavioural checks for the actual released fusion decoder implementations."""
import ast
import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from lfpformer.paths import REFERENCE


def test_fusion_folding_input_control_and_padding(monkeypatch):
    monkeypatch.syspath_prepend(str(REFERENCE / "fusion"))
    spec = importlib.util.spec_from_file_location(
        "fusion_audit_experiment", REFERENCE / "fusion/experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    fusion = module.MatchedDirectProbeSeq(768, np.zeros(768), 1., "lfp_manip").eval()
    control = module.MatchedDirectProbeSeq(768, np.zeros(768), 1., "lfp_only").eval()
    # Avoid the zero-initialized output head making invariance checks trivial.
    torch.nn.init.normal_(fusion.resid_head.weight, std=.01)
    control.load_state_dict(fusion.state_dict())
    assert sum(p.numel() for p in fusion.parameters()) == 2308102
    emb = torch.randn(1, 6, 768)
    manip = torch.tensor([4000., 3000., 2500.]).expand(1, 6, 3).clone()
    geom = torch.randn(1, 6, 2)
    depth = torch.arange(6).float()[None] * 20
    mask = torch.tensor([[False, False, False, False, True, True]])
    with torch.no_grad():
        mu, sigma = fusion(emb, manip, geom, depth, mask)
        assert torch.isfinite(mu).all() and (sigma > 0).all()
        mirrored = manip.clone()
        mirrored[..., 2] = 2 * 5739 - mirrored[..., 2]
        for a, b in zip((mu, sigma), fusion(emb, mirrored, geom, depth, mask)):
            torch.testing.assert_close(a, b)
        baseline = control(emb, manip, geom, depth, mask)
        changed = control(emb, manip + 3000, geom, depth, mask)
        for a, b in zip(baseline, changed):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert not torch.allclose(mu[:, :4], baseline[0][:, :4])
        changed_emb = emb.clone()
        changed_emb[:, 4:] += 1000
        padded = fusion(changed_emb, manip, geom, depth, mask)
        for a, b in zip((mu, sigma), padded):
            torch.testing.assert_close(a[:, :4], b[:, :4])


def test_linear_fusion_consumes_manip_coordinates_independently():
    path = REFERENCE / "benchmark/scripts/eval_regression_probe.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "GaussianLinearProbe")
    ns = {"nn": torch.nn, "LOG_VAR_MIN": -10., "LOG_VAR_MAX": 10.}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    head = ns["GaussianLinearProbe"](771).eval()
    assert sum(p.numel() for p in head.parameters()) == 4632
    x = torch.randn(5, 771)
    mu, lv = head(x)
    changed = x.clone()
    changed[0, -3:] += 10
    new_mu, new_lv = head(changed)
    assert not torch.allclose(new_mu[0], mu[0])
    torch.testing.assert_close(new_mu[1:], mu[1:])
    torch.testing.assert_close(new_lv[1:], lv[1:])
    (mu.square().mean() + lv.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in head.parameters())
