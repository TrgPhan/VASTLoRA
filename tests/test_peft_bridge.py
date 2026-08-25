from types import SimpleNamespace

import torch
from torch import nn

from vastlora.lowrank import LowRankMatrix, compact_svd
from vastlora.scale import (
    FactorSnapshot,
    capture_factor_snapshot,
    compact_factor_innovations,
    empty_adapter_state,
    fedrot_aggregate_factor_state,
    load_compact_adapter_state,
    mask_inactive_rank_gradients,
)


class FakePeftLoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, out_features, bias=False)})
        self.scaling = {"default": 2.0}


def _model() -> nn.Module:
    return nn.ModuleDict({"projection": FakePeftLoRA(7, 9, 4)})


def test_load_compact_state_reconstructs_server_delta() -> None:
    model = _model()
    generator = torch.Generator().manual_seed(2)
    expected = compact_svd(
        LowRankMatrix(
            torch.randn(9, 3, generator=generator),
            torch.randn(3, 7, generator=generator),
        ),
        rtol=1e-7,
    )
    state = {"projection": expected}

    load_compact_adapter_state(model, state, active_rank=4, initialize_free_directions=False)

    module = model["projection"]
    actual = module.scaling["default"] * (
        module.lora_B["default"].weight @ module.lora_A["default"].weight
    )
    torch.testing.assert_close(actual, expected.dense(), rtol=1e-5, atol=1e-5)


def test_factor_innovation_matches_exact_delta_difference() -> None:
    model = _model()
    state = empty_adapter_state(model)
    load_compact_adapter_state(model, state, active_rank=3, seed=9)
    before = capture_factor_snapshot(model)

    module = model["projection"]
    with torch.no_grad():
        module.lora_A["default"].weight[:3].add_(0.1)
        module.lora_B["default"].weight[:, :3].add_(0.2)
    after = capture_factor_snapshot(model)
    updates = compact_factor_innovations(before, after, active_rank=3, rank_rtol=1e-7)

    expected = before["projection"].scaling * (
        after["projection"].b[:, :3] @ after["projection"].a[:3]
        - before["projection"].b[:, :3] @ before["projection"].a[:3]
    )
    torch.testing.assert_close(updates["projection"].dense(), expected, rtol=1e-5, atol=1e-5)


def test_inactive_rank_gradients_are_masked() -> None:
    model = _model()
    module = model["projection"]
    module.lora_A["default"].weight.grad = torch.ones_like(
        module.lora_A["default"].weight
    )
    module.lora_B["default"].weight.grad = torch.ones_like(
        module.lora_B["default"].weight
    )

    mask_inactive_rank_gradients(model, active_rank=2)

    assert torch.all(module.lora_A["default"].weight.grad[:2] == 1)
    assert torch.all(module.lora_A["default"].weight.grad[2:] == 0)
    assert torch.all(module.lora_B["default"].weight.grad[:, :2] == 1)
    assert torch.all(module.lora_B["default"].weight.grad[:, 2:] == 0)


def test_fedrot_factor_aggregation_aligns_rotated_client() -> None:
    generator = torch.Generator().manual_seed(17)
    b = torch.randn(9, 4, generator=generator)
    a = torch.randn(4, 7, generator=generator)
    q, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
    compact = compact_svd(LowRankMatrix(2.0 * b, a), rtol=1e-7, max_rank=4)
    client = {
        "projection": FactorSnapshot(
            a=q.T @ a,
            b=b @ q,
            scaling=2.0,
        )
    }

    aggregated = fedrot_aggregate_factor_state(
        {"projection": compact},
        client,
        active_rank=4,
        weight=1.0,
        max_rank=4,
        rank_rtol=1e-7,
    )

    torch.testing.assert_close(
        aggregated["projection"].dense(),
        compact.dense(),
        rtol=1e-5,
        atol=1e-5,
    )
