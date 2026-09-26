import pytest
import torch
from torch import nn
from torch.nn import functional as F

from riftlora.baselines import (
    aggregate_spectral_factors, spectral_async_aggregate, attach_florg_adapters,
    capture_florg_state, load_florg_state, florg_aggregate_state,
)
from riftlora.lowrank import LowRankMatrix, compact_svd
from riftlora.scale import FactorSnapshot


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def clients():
    g = torch.Generator().manual_seed(121)
    return [{"q": FactorSnapshot(torch.randn(r, 7, generator=g, dtype=torch.float64),
                                torch.randn(5, r, generator=g, dtype=torch.float64), scale)}
            for r, scale in [(2, .5), (3, 2.)]]


@pytest.mark.parametrize("method", ["flexlora", "florist"])
def test_weighted_products_match_independent_dense_svd(method):
    states = clients()
    dense = sum(w * c["q"].scaling * (c["q"].b @ c["q"].a) for c, w in zip(states, [.3, .7]))
    u, s, vh = torch.linalg.svd(dense, full_matrices=False)
    actual, diagnostics = aggregate_spectral_factors(states, weights=[.3, .7], method=method, max_rank=2, energy=1.)
    torch.testing.assert_close(actual["q"].dense(), (u[:, :2] * s[:2]) @ vh[:2])
    assert diagnostics["spectral_cap_binding_fraction"] == 1


def test_florist_uses_squared_energy_and_keeps_boundary_component():
    # Squares: 9,4,1; tau=.8 requires two components (not one, not three).
    state = {"q": FactorSnapshot(torch.eye(3), torch.diag(torch.tensor([3., 2., 1.])), 1.)}
    result, d = aggregate_spectral_factors([state], weights=[1.], method="florist", energy=.8)
    assert result["q"].rank == 2
    torch.testing.assert_close(result["q"].dense(), torch.diag(torch.tensor([3., 2., 0.])))
    assert d["spectral_mean_retained_energy"] == pytest.approx(13 / 14)
    capped, d = aggregate_spectral_factors([state], weights=[1.], method="florist", energy=.8, max_rank=1)
    assert capped["q"].rank == 1 and d["spectral_cap_binding_fraction"] == 1
    assert d["spectral_mean_retained_energy"] < .8


@pytest.mark.parametrize("method", ["flexlora", "florist"])
def test_async_mixes_full_client_state_not_innovation(method):
    first, second = clients()
    server = {"q": compact_svd(LowRankMatrix(first["q"].b * first["q"].scaling, first["q"].a))}
    result, _ = spectral_async_aggregate(server, second, active_rank=3, weight=.4,
                                        method=method, max_rank=5, energy=1., rank_rtol=1e-10)
    expected = .6 * server["q"].dense() + .4 * second["q"].scaling * second["q"].b @ second["q"].a
    torch.testing.assert_close(result["q"].dense(), expected)


@pytest.mark.parametrize("method", ["flexlora", "florist"])
def test_zero_products_and_invalid_parameters(method):
    state = {"q": FactorSnapshot(torch.ones(2, 3), torch.zeros(4, 2), 1.)}
    result, d = aggregate_spectral_factors([state], weights=[1.], method=method)
    assert result["q"].rank == 0 and d["spectral_mean_retained_energy"] == 1
    for kwargs in ({"weights": [-1.]}, {"weights": [.5]}, {"energy": float("nan")}, {"max_rank": 0}):
        with pytest.raises(ValueError):
            aggregate_spectral_factors([state], **({"weights": [1.], "method": method} | kwargs))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_florg_device_forward_gradient_and_rank_dispatch(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = nn.ModuleDict({"q_proj": nn.Linear(7, 5, bias=False)}).to(device)
    model.requires_grad_(False)
    attach_florg_adapters(model, target_modules=["q_proj"], rank=3, seed=7)
    adapter = model.florg_adapters["layer_0"]
    assert adapter.a.device == model.q_proj.weight.device
    x = torch.randn(2, 4, 7, device=device)
    actual = model.q_proj(x)
    expected = F.linear(x, model.q_proj.weight) + F.linear(x, adapter.delta())
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.square().sum(), adapter.a)[0]
    ge = torch.autograd.grad(expected.square().sum(), adapter.a)[0]
    torch.testing.assert_close(ga, ge, rtol=2e-5, atol=1e-6)
    assert ga.count_nonzero() and adapter.left.grad is None
    state = capture_florg_state(model)
    load_florg_state(model, state, active_rank=1)
    assert adapter.a[1:].count_nonzero() == 0
    load_florg_state(model, state)
    torch.testing.assert_close(capture_florg_state(model)["q_proj"], state["q_proj"])


def test_florg_rectangular_procrustes_matches_dense_equation():
    torch.manual_seed(2)
    old, client = torch.randn(2, 5, dtype=torch.float64), torch.randn(2, 5, dtype=torch.float64)
    gram = .6 * old.T @ old + .4 * client.T @ client
    eigenvalues, vectors = torch.linalg.eigh(gram)
    keep = eigenvalues > 1e-10
    candidate = eigenvalues[keep].sqrt()[:, None] * vectors[:, keep].T
    u, _, vh = torch.linalg.svd(old @ candidate.T, full_matrices=False)
    expected = u @ vh @ candidate
    actual = florg_aggregate_state({"q": old}, {"q": client}, weight=.4)["q"]
    torch.testing.assert_close(actual, expected)
    # At fixed row budget r=2 a rank-4 Gram cannot be preserved exactly.
    assert not torch.allclose(actual.T @ actual, gram)


def test_florg_seed_controls_latent_initialization():
    values = []
    for seed in [3, 3, 4]:
        model = nn.ModuleDict({"q_proj": nn.Linear(7, 5)})
        attach_florg_adapters(model, target_modules=["q_proj"], rank=2, seed=seed)
        values.append(capture_florg_state(model)["q_proj"])
    torch.testing.assert_close(values[0], values[1])
    assert not torch.equal(values[0], values[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_florg_on_real_nf4_linear_backpropagates():
    import bitsandbytes as bnb
    linear = bnb.nn.Linear4bit(32, 16, bias=False, compute_dtype=torch.float16, quant_type="nf4")
    model = nn.ModuleDict({"q_proj": linear}).to("cuda")
    model.requires_grad_(False)
    attach_florg_adapters(model, target_modules=["q_proj"], rank=4, seed=17)
    adapter = model.florg_adapters["layer_0"]
    x = torch.randn(2, 3, 32, device="cuda", dtype=torch.float16)
    result = model.q_proj(x)
    assert result.dtype == torch.float16 and adapter.a.dtype == torch.float32
    result.float().square().mean().backward()
    assert adapter.a.grad is not None and torch.isfinite(adapter.a.grad).all()
    assert adapter.a.grad.count_nonzero()
    assert model.q_proj.weight.grad is None
