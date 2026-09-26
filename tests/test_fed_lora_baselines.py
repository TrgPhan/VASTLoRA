from __future__ import annotations

import torch

from riftlora.baselines import (
    capture_florg_state,
    fedavg_aggregate_factor_state,
    fedex_aggregate_factor_state,
    florg_aggregate_state,
    load_florg_state,
    flora_stack_aggregate_state,
    flora_stack_factor_states,
)
from riftlora.lowrank import LowRankMatrix, compact_svd
from riftlora.scale import FactorSnapshot


def _state():
    u = torch.tensor([[1.0], [0.0]])
    v = torch.tensor([[1.0], [0.0]])
    return {"projection": compact_svd(LowRankMatrix(2.0 * u, v.T), max_rank=2)}


def _client(a, b, scaling=1.0):
    return {"projection": FactorSnapshot(
        a=torch.tensor(a, dtype=torch.float32),
        b=torch.tensor(b, dtype=torch.float32),
        scaling=scaling,
    )}


def test_fedavg_averages_factors_not_products():
    server = _client([[1.0, 0.0]], [[2.0], [0.0]])
    client = _client([[2.0, 0.0]], [[3.0], [0.0]])
    result = fedavg_aggregate_factor_state(
        server, client, active_rank=1, weight=0.5, max_rank=1
    )
    expected = (0.5 * 2.0 + 0.5 * 3.0) * (0.5 * 1.0 + 0.5 * 2.0)
    value = result["projection"]
    torch.testing.assert_close(value.scaling * value.b @ value.a, torch.tensor([[expected, 0.0], [0.0, 0.0]]))


def test_fedex_residual_recovers_weighted_product_exactly():
    server = _state()
    client = _client([[2.0, 0.0]], [[3.0], [0.0]])
    zero = {"projection": torch.zeros((2, 2))}
    state, residual = fedex_aggregate_factor_state(
        server, client, current_residual=zero, stale_residual=zero,
        active_rank=1, weight=0.5, max_rank=1, rank_rtol=1e-7
    )
    effective = state["projection"].dense() + residual["projection"]
    ideal = 0.5 * server["projection"].dense() + 0.5 * torch.tensor([[6.0, 0.0], [0.0, 0.0]])
    torch.testing.assert_close(effective, ideal, rtol=1e-5, atol=1e-5)


def test_florg_gram_aggregation_preserves_returned_gram():
    server = {"layer": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])}
    client = {"layer": torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])}
    result = florg_aggregate_state(server, client, weight=0.5)
    expected = 0.5 * (server["layer"].T @ server["layer"]) + 0.5 * (client["layer"].T @ client["layer"])
    torch.testing.assert_close(result["layer"].T @ result["layer"], expected, rtol=1e-5, atol=1e-5)


def test_flora_stacking_preserves_weighted_products_with_heterogeneous_ranks():
    client_a = _client([[1.0, 2.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 1.0]])
    client_b = _client([[1.0, 2.0]], [[3.0], [4.0]], scaling=0.5)
    result = flora_stack_factor_states(
        [client_a, client_b], weights=[0.25, 0.75], active_ranks=[2, 1],
        max_rank=3, rank_rtol=1e-7,
    )
    expected = (
        0.25 * (client_a["projection"].scaling * client_a["projection"].b @ client_a["projection"].a)
        + 0.75 * (client_b["projection"].scaling * client_b["projection"].b @ client_b["projection"].a)
    )
    torch.testing.assert_close(result["projection"].dense(), expected, rtol=1e-5, atol=1e-5)


def test_flora_async_stack_is_product_space_not_factor_average():
    server = _state()
    innovation = compact_svd(
        LowRankMatrix(torch.tensor([[0.0], [2.0]]), torch.tensor([[1.0, 0.0]])),
        max_rank=2,
    )
    result = flora_stack_aggregate_state(
        server["projection"], innovation, weight=0.5, max_rank=2, rank_rtol=1e-7
    )
    expected = server["projection"].dense() + 0.5 * innovation.dense()
    torch.testing.assert_close(result.dense(), expected, rtol=1e-5, atol=1e-5)
