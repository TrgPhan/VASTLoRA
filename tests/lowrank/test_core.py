import torch

from riftlora.lowrank import (
    LowRankMatrix,
    build_adaptive_temporal_reference,
    build_temporal_reference,
    compatibility_scores,
    compact_svd,
    exact_lora_innovation,
    project_to_reference,
    recompress,
    weighted_sum,
)


def test_compact_svd_zero_matrix_has_zero_rank() -> None:
    result = compact_svd(
        LowRankMatrix(torch.zeros(3, 1), torch.zeros(1, 4))
    )

    assert result.rank == 0
    torch.testing.assert_close(result.dense(), torch.zeros(3, 4))


def test_compatibility_scores_match_dense_projection_energy() -> None:
    torch.manual_seed(41)
    update = compact_svd(LowRankMatrix(_randn(7, 3), _randn(3, 6)))
    q_left, _ = torch.linalg.qr(_randn(7, 2), mode="reduced")
    q_right, _ = torch.linalg.qr(_randn(6, 2), mode="reduced")

    rho_left, rho_right, rho_two = compatibility_scores(update, q_left, q_right)
    dense = update.dense()
    denominator = torch.sum(dense.square())
    left_dense = q_left @ q_left.T @ dense
    right_dense = dense @ q_right @ q_right.T
    two_dense = q_left @ q_left.T @ dense @ q_right @ q_right.T

    torch.testing.assert_close(rho_left, torch.sum(left_dense.square()) / denominator)
    torch.testing.assert_close(rho_right, torch.sum(right_dense.square()) / denominator)
    torch.testing.assert_close(rho_two, torch.sum(two_dense.square()) / denominator)


def _randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float64)


def test_exact_lora_innovation_matches_dense_difference() -> None:
    torch.manual_seed(1)
    b0 = _randn(9, 3)
    a0 = _randn(3, 7)
    b1 = _randn(9, 3)
    a1 = _randn(3, 7)

    innovation = exact_lora_innovation(b1, a1, b0, a0)

    expected = b1 @ a1 - b0 @ a0
    assert innovation.shape == expected.shape
    assert innovation.factor_rank == 6
    torch.testing.assert_close(innovation.dense(), expected)


def test_compact_svd_reconstructs_lowrank_matrix_with_rectangular_middle() -> None:
    torch.manual_seed(2)
    left = _randn(11, 5)
    right = _randn(5, 8)
    update = LowRankMatrix(left, right)

    svd = compact_svd(update, rtol=1e-12)

    assert svd.rank <= 5
    torch.testing.assert_close(svd.dense(), update.dense(), rtol=1e-10, atol=1e-10)


def test_compact_svd_is_gauge_invariant_in_dense_result_and_spectrum() -> None:
    torch.manual_seed(3)
    b = _randn(10, 4)
    a = _randn(4, 6)
    q = _randn(4, 4)
    while torch.linalg.det(q).abs() < 1e-3:
        q = _randn(4, 4)

    original = LowRankMatrix(b, a)
    gauged = LowRankMatrix(b @ q, torch.linalg.solve(q, a))

    svd_original = compact_svd(original, rtol=1e-12)
    svd_gauged = compact_svd(gauged, rtol=1e-12)

    torch.testing.assert_close(gauged.dense(), original.dense(), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(svd_gauged.dense(), svd_original.dense(), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(svd_gauged.s, svd_original.s, rtol=1e-9, atol=1e-9)


def test_weighted_sum_and_recompress_match_dense_oracle() -> None:
    torch.manual_seed(4)
    updates = [
        LowRankMatrix(_randn(12, 3), _randn(3, 9)),
        LowRankMatrix(_randn(12, 2), _randn(2, 9)),
        LowRankMatrix(_randn(12, 4), _randn(4, 9)),
    ]
    weights = [0.2, -0.5, 1.3]

    summed = weighted_sum(updates, weights)
    expected = sum(weight * update.dense() for weight, update in zip(weights, updates))
    torch.testing.assert_close(summed.dense(), expected)

    full_rank = recompress(summed, max_rank=9, rtol=1e-12)
    torch.testing.assert_close(full_rank.dense(), expected, rtol=1e-10, atol=1e-10)

    low_rank = recompress(summed, max_rank=3, rtol=1e-12)
    dense_error = torch.linalg.matrix_norm(expected - low_rank.dense(), ord="fro")
    assert dense_error > 0
    assert low_rank.rank == 3


def test_temporal_reference_projection_matches_dense_oracle() -> None:
    torch.manual_seed(5)
    history = [
        compact_svd(LowRankMatrix(_randn(10, 3), _randn(3, 8)), rtol=1e-12),
        compact_svd(LowRankMatrix(_randn(10, 4), _randn(4, 8)), rtol=1e-12),
    ]
    update = compact_svd(LowRankMatrix(_randn(10, 5), _randn(5, 8)), rtol=1e-12)

    q_left, q_right = build_temporal_reference(history, left_rank=4, right_rank=3, decay=0.2)
    projected, core, rho = project_to_reference(update, q_left, q_right)

    expected = q_left @ core @ q_right.T
    torch.testing.assert_close(projected.dense(), expected, rtol=1e-10, atol=1e-10)
    assert 0.0 <= float(rho) <= 1.0


def test_temporal_reference_can_prioritize_high_energy_modes() -> None:
    u = torch.eye(3, dtype=torch.float64)
    v = torch.eye(3, dtype=torch.float64)
    history = [
        compact_svd(
            LowRankMatrix(u, torch.diag(torch.tensor([10.0, 2.0, 1.0], dtype=torch.float64)))
        )
    ]

    q_left, q_right = build_temporal_reference(
        history,
        left_rank=1,
        right_rank=1,
        singular_power=1.0,
    )

    assert torch.abs(q_left[0, 0]) > 0.999
    assert torch.abs(q_right[0, 0]) > 0.999


def test_full_reference_projection_has_unit_compatibility() -> None:
    torch.manual_seed(6)
    update = compact_svd(LowRankMatrix(_randn(7, 4), _randn(4, 6)), rtol=1e-12)

    projected, _, rho = project_to_reference(update, update.u, update.v)

    torch.testing.assert_close(projected.dense(), update.dense(), rtol=1e-10, atol=1e-10)
    assert float(rho) > 1.0 - 1e-10


def test_adaptive_temporal_reference_selects_each_side_independently() -> None:
    u = torch.eye(4, dtype=torch.float64)
    v = torch.eye(4, dtype=torch.float64)
    history = [
        compact_svd(
            LowRankMatrix(
                u,
                torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5], dtype=torch.float64)),
            ),
            rtol=1e-12,
        )
    ]

    reference = build_adaptive_temporal_reference(
        history,
        energy_threshold=0.9,
        min_rank=1,
        max_rank=4,
        singular_power=1.0,
    )

    assert reference.left_rank == 2
    assert reference.right_rank == 2
    assert reference.left_retained_energy >= 0.9
    assert reference.right_retained_energy >= 0.9


def test_adaptive_temporal_reference_honors_rank_bounds() -> None:
    torch.manual_seed(7)
    history = [
        compact_svd(LowRankMatrix(_randn(8, 6), _randn(6, 7)), rtol=1e-12)
    ]

    reference = build_adaptive_temporal_reference(
        history,
        energy_threshold=0.999,
        min_rank=2,
        max_rank=3,
    )

    assert reference.left_rank == 3
    assert reference.right_rank == 3

