from __future__ import annotations

import torch
from torch import nn
import pytest

from riftlora.lowrank import CompactSVD
from riftlora.scale.objective import (
    filter_compact_by_scores,
    score_compact_components_microbatched,
    score_compact_components_with_hooks,
)


class ToyLoRALinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(2, 1, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})
        self.scaling = {"default": 1.0}
        with torch.no_grad():
            self.base.weight.zero_()
            self.lora_A["default"].weight.zero_()
            self.lora_B["default"].weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = ToyLoRALinear()

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        output = self.layer(x)
        loss = (output - target).square().mean()
        return type("Output", (), {"loss": loss})()


def test_component_hook_scores_descent_direction() -> None:
    model = ToyModel()
    innovation = CompactSVD(
        torch.tensor([[1.0]]),
        torch.tensor([1.0]),
        torch.tensor([[1.0], [0.0]]),
    )
    result = score_compact_components_with_hooks(
        model,
        {"layer": innovation},
        {"x": torch.tensor([[1.0, 0.0]]), "target": torch.tensor([[1.0]])},
    )

    assert result.scores["layer"].item() > 0.0
    assert result.retained_rank == 1
    filtered = filter_compact_by_scores({"layer": innovation}, result.scores)
    assert filtered["layer"].rank == 1


def test_microbatched_component_scores_match_full_batch() -> None:
    model = ToyModel()
    innovation = CompactSVD(
        torch.tensor([[1.0]]),
        torch.tensor([1.0]),
        torch.tensor([[1.0], [0.0]]),
    )
    innovations = {"layer": innovation}
    first = {"x": torch.tensor([[1.0, 0.0]]), "target": torch.tensor([[1.0]])}
    second = {
        "x": torch.tensor([[2.0, 0.0], [3.0, 0.0]]),
        "target": torch.tensor([[0.5], [2.0]]),
    }
    full = {
        "x": torch.cat([first["x"], second["x"]]),
        "target": torch.cat([first["target"], second["target"]]),
    }

    expected = score_compact_components_with_hooks(model, innovations, full)
    actual = score_compact_components_microbatched(
        model,
        innovations,
        [(first, 1.0), (second, 2.0)],
    )

    assert torch.allclose(actual.scores["layer"], expected.scores["layer"])
    assert actual.calibration_loss == pytest.approx(expected.calibration_loss)


def test_filter_retains_global_component_gain_mass() -> None:
    innovations = {
        "first": CompactSVD(torch.eye(3), torch.ones(3), torch.eye(3)),
        "second": CompactSVD(torch.eye(2), torch.ones(2), torch.eye(2)),
    }
    scores = {
        "first": torch.tensor([4.0, 3.0, -1.0]),
        "second": torch.tensor([2.0, 1.0]),
    }

    filtered = filter_compact_by_scores(
        innovations,
        scores,
        retained_gain_mass=0.6,
    )

    assert filtered["first"].rank == 2
    assert filtered["second"].rank == 0


def test_filter_rejects_invalid_gain_mass() -> None:
    innovation = CompactSVD(torch.eye(1), torch.ones(1), torch.eye(1))

    with pytest.raises(ValueError, match="retained_gain_mass"):
        filter_compact_by_scores(
            {"layer": innovation},
            {"layer": torch.ones(1)},
            retained_gain_mass=0.0,
        )

