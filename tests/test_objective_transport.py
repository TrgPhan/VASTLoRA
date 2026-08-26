import pytest
import torch

from vastlora.diagnostics import filter_rankwise_by_gradient, paired_loss_gate
from vastlora.lowrank import LowRankMatrix


def test_rankwise_filter_keeps_only_predicted_descent_components() -> None:
    innovation = LowRankMatrix(torch.eye(2), torch.eye(2))
    gradient = torch.diag(torch.tensor([-2.0, 3.0]))

    result = filter_rankwise_by_gradient(
        {"layer": innovation},
        {"layer": gradient},
        rank_rtol=1e-7,
    )

    torch.testing.assert_close(
        result.updates["layer"], torch.diag(torch.tensor([1.0, 0.0]))
    )
    assert result.retained_rank == 1
    assert result.total_rank == 2
    assert result.predicted_gain == pytest.approx(2.0)


def test_rankwise_filter_can_apply_global_component_budget() -> None:
    innovation = LowRankMatrix(torch.eye(3), torch.eye(3))
    gradient = torch.diag(torch.tensor([-1.0, -3.0, -2.0]))

    result = filter_rankwise_by_gradient(
        {"layer": innovation},
        {"layer": gradient},
        max_components=2,
        rank_rtol=1e-7,
    )

    torch.testing.assert_close(
        result.updates["layer"], torch.diag(torch.tensor([0.0, 1.0, 1.0]))
    )
    assert result.retained_rank == 2
    assert result.predicted_gain == pytest.approx(5.0)


def test_rankwise_filter_can_keep_all_components_for_gate_only_ablation() -> None:
    innovation = LowRankMatrix(torch.eye(2), torch.eye(2))
    gradient = torch.diag(torch.tensor([-2.0, 3.0]))

    result = filter_rankwise_by_gradient(
        {"layer": innovation},
        {"layer": gradient},
        keep_nonpositive=True,
        rank_rtol=1e-7,
    )

    torch.testing.assert_close(result.updates["layer"], torch.eye(2))
    assert result.retained_rank == 2
    assert result.predicted_gain == pytest.approx(-1.0)


def test_paired_gate_accounts_for_example_level_uncertainty() -> None:
    current = torch.tensor([1.0, 1.0, 1.0, 1.0])
    candidate = torch.tensor([0.8, 0.8, 1.1, 1.1])

    permissive = paired_loss_gate(current, candidate, z_value=0.0)
    guarded = paired_loss_gate(current, candidate, z_value=1.96)

    assert permissive.accepted is True
    assert guarded.accepted is False
    assert guarded.upper_bound > permissive.upper_bound
