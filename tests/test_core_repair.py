import pytest
import torch
from torch import nn

from riftlora.lowrank import CompactSVD
from riftlora.scale.core_repair import CoreRepairConfig, repair_compact_core


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 2, bias=False)})
        self.scaling = {"default": 1.0}
        nn.init.zeros_(self.base.weight)

    def forward(self, x):
        return self.base(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Layer()

    def forward(self, x):
        return self.layer(x)


def mse(model, batch):
    return (model(batch["x"]) - batch["target"]).square().mean()


def run(model, batches, **kwargs):
    return repair_compact_core(
        model, {"layer": CompactSVD(torch.eye(2), torch.ones(2), torch.eye(2))},
        {"layer": torch.ones(2)}, batches, loss_fn=mse,
        server_weight=kwargs.pop("server_weight", 1.0),
        staleness=kwargs.pop("staleness", 0), **kwargs,
    )


def dense(result):
    u = result.updates["layer"]
    return (u.u * u.s) @ u.v.T


def test_full_core_fits_rotation_that_diagonal_cannot():
    # A diagonal filter cannot send the first input coordinate to output two.
    batch = {"x": torch.eye(2), "target": torch.tensor([[0., 1.], [1., 0.]])}
    config = CoreRepairConfig(steps=50, learning_rate=.05, radius=2., proximal_weight=0.)
    full = run(Model(), [(batch, 2.)], config=config)
    diag = run(Model(), [(batch, 2.)], config=config, diagonal_only=True)
    full_loss = (dense(full).T - batch["target"]).square().mean()
    diag_loss = (dense(diag).T - batch["target"]).square().mean()
    assert full_loss < .01
    assert diag_loss >= .5
    assert full.diagnostics["core_mean_offdiagonal_norm"] > 0
    assert diag.diagnostics["core_mean_offdiagonal_norm"] == 0


def test_microbatches_equal_full_objective_and_restore_model():
    model = Model()
    model.train()
    model.layer.base.eval()
    model.layer.base.weight.requires_grad_(False)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    modes = [m.training for m in model.modules()]
    flags = [p.requires_grad for p in model.parameters()]
    batch = {"x": torch.eye(2), "target": torch.tensor([[.3, 1.], [.5, .2]])}
    micro = [({k: v[i:i+1] for k, v in batch.items()}, 1.) for i in range(2)]
    first = run(model, [(batch, 2.)], server_weight=.3)
    second = run(model, micro, server_weight=.3)
    assert torch.allclose(dense(first), dense(second), atol=1e-6)
    assert modes == [m.training for m in model.modules()]
    assert flags == [p.requires_grad for p in model.parameters()]
    assert not model.layer._forward_hooks
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


def test_zero_radius_recovers_filter_and_delay_contracts_trust_region():
    batch = {"x": torch.eye(2), "target": torch.zeros(2, 2)}
    zero = run(Model(), [(batch, 2.)], config=CoreRepairConfig(radius=0.))
    assert torch.allclose(dense(zero), torch.eye(2))
    stale = run(Model(), [(batch, 2.)], staleness=24, config=CoreRepairConfig(radius=.1))
    assert stale.diagnostics["core_radius"] == pytest.approx(.05)
    assert stale.diagnostics["core_max_distance"] <= .050001


def test_failure_cleans_hooks_and_flags():
    model = Model()
    flags = [p.requires_grad for p in model.parameters()]
    batch = {"x": torch.eye(2), "target": torch.full((2, 2), float("nan"))}
    with pytest.raises(ValueError, match="nonfinite"):
        run(model, [(batch, 2.)])
    assert model.training
    assert flags == [p.requires_grad for p in model.parameters()]
    assert not model.layer._forward_hooks


@pytest.mark.parametrize("config", [CoreRepairConfig(steps=0), CoreRepairConfig(radius=-1), CoreRepairConfig(learning_rate=float("nan"))])
def test_reject_bad_config(config):
    with pytest.raises(ValueError):
        config.validate()
