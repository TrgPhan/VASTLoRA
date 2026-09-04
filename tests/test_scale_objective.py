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

