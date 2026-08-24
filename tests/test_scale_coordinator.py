import math

import torch

from vastlora.lowrank import LowRankMatrix, compact_svd
from vastlora.scale import (
    TransportConfig,
    aggregate_compact_state,
    transport_compact_update,
    zero_compact,
)


def _svd(seed: int, rows: int = 9, columns: int = 7, rank: int = 3):
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(rows, rank, generator=generator, dtype=torch.float64)
    right = torch.randn(rank, columns, generator=generator, dtype=torch.float64)
    return compact_svd(LowRankMatrix(left, right), rtol=1e-12)


def test_freshness_transport_matches_dense_oracle() -> None:
    update = _svd(1)
    config = TransportConfig(freshness_lambda=0.2, rank_rtol=1e-12)

    result = transport_compact_update(
        update,
        [],
        method="freshness",
        staleness=3,
        config=config,
        max_rank=7,
    )

    expected = math.exp(-0.6) * update.dense()
    torch.testing.assert_close(result.update.dense(), expected, rtol=1e-10, atol=1e-10)


def test_mtip_adaptive_is_compact_and_reports_selected_ranks() -> None:
    history = [_svd(seed) for seed in (2, 3, 4)]
    update = _svd(5)
    config = TransportConfig(
        adaptive_energy=0.8,
        adaptive_min_rank=2,
        adaptive_max_rank=5,
        adaptive_singular_power=1.0,
        rank_rtol=1e-12,
    )

    result = transport_compact_update(
        update,
        history,
        method="mtip_adaptive",
        staleness=4,
        config=config,
        max_rank=5,
    )

    assert 2 <= result.left_rank <= 5
    assert 2 <= result.right_rank <= 5
    assert result.update.rank <= 5
    assert result.left_retained_energy is not None
    assert result.right_retained_energy is not None
    assert 0.0 <= result.rho <= 1.0


def test_compact_aggregation_matches_dense_sum() -> None:
    server = _svd(6, rank=2)
    update = _svd(7, rank=3)

    aggregated = aggregate_compact_state(
        server,
        update,
        weight=0.25,
        max_rank=7,
        rank_rtol=1e-12,
    )

    torch.testing.assert_close(
        aggregated.dense(),
        server.dense() + 0.25 * update.dense(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_zero_state_can_receive_first_update() -> None:
    update = _svd(8)
    server = zero_compact(update.shape, dtype=torch.float64)

    aggregated = aggregate_compact_state(server, update, weight=0.5, max_rank=4)

    torch.testing.assert_close(aggregated.dense(), 0.5 * update.dense())
