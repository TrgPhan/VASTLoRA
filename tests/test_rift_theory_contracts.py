"""Check scoped identities and counterexamples, not empirical effectiveness."""

import math

import pytest
import torch

from riftlora.lowrank import CompactSVD
from riftlora.scale.objective import filter_compact_by_scores


def dense(compact):
    return (compact.u * compact.s) @ compact.v.T


@pytest.mark.parametrize("rank", [1, 2, 4])
def test_core_isometry_rank_and_projected_triangle_bound(rank):
    generator = torch.Generator().manual_seed(57 + rank)

    def normal(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    u = torch.linalg.qr(normal(8, rank)).Q
    v = torch.linalg.qr(normal(7, rank)).Q
    anchor = torch.diag(torch.arange(1, rank + 1, dtype=torch.float64))
    delta = normal(rank, rank)
    delta *= 0.3 / delta.norm()
    energy, eta = 2.0, 0.5
    innovation = energy * u @ (anchor + delta) @ v.T
    filtered = energy * u @ anchor @ v.T
    torch.testing.assert_close((innovation - filtered).norm(), energy * delta.norm())
    assert torch.linalg.matrix_rank(innovation) <= rank
    server = normal(8, 7)
    a, b = server + eta * innovation, server + eta * filtered
    pa, pb = rank_projection(a, rank), rank_projection(b, rank)
    bound = eta * energy * delta.norm() + (pa - a).norm() + (pb - b).norm()
    assert (pa - pb).norm() <= bound + 1e-12


def rank_projection(matrix, rank):
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank]


def test_offdiagonal_descent_identity_does_not_require_new_basis():
    gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    offdiagonal = gradient - torch.diag(gradient.diag())
    direction = -offdiagonal
    torch.testing.assert_close(
        (gradient * direction).sum(), -offdiagonal.square().sum()
    )
    assert direction.diag().count_nonzero() == 0


def test_rank_projection_is_not_globally_nonexpansive():
    # Both matrices have unique top singular vectors; the gap is merely small.
    a = torch.diag(torch.tensor([1.01, 1.0], dtype=torch.float64))
    b = torch.diag(torch.tensor([1.0, 1.01], dtype=torch.float64))
    before = (a - b).norm()
    after = (rank_projection(a, 1) - rank_projection(b, 1)).norm()
    assert after / before == pytest.approx(101.0)


def test_positive_filter_is_not_invariant_to_degenerate_svd_rotations():
    identity = torch.eye(2, dtype=torch.float64)
    rotation = torch.tensor([[1.0, -1.0], [1.0, 1.0]], dtype=torch.float64) / math.sqrt(
        2
    )
    gradient = torch.diag(torch.tensor([-2.0, 1.0], dtype=torch.float64))
    outputs = []
    for basis in (identity, rotation):
        svd = CompactSVD(basis, torch.ones(2, dtype=torch.float64), basis)
        torch.testing.assert_close(dense(svd), identity)
        scores = -(basis.T @ gradient @ basis).diag()
        outputs.append(
            filter_compact_by_scores({"layer": svd}, {"layer": scores})["layer"]
        )
    torch.testing.assert_close(
        dense(outputs[0]), torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    )
    torch.testing.assert_close(dense(outputs[1]), identity)


def test_decreasing_relative_radius_does_not_bound_absolute_change_across_events():
    def bound(energy, age):
        return energy * 0.5 / math.sqrt(1 + age / 8)

    assert bound(100.0, 24) > bound(1.0, 0)


def test_calibration_descent_can_increase_test_loss():
    # A different target suffices; a gate cannot eliminate distribution shift.
    initial, candidate = 0.0, 0.5
    calibration_loss = lambda x: (x - 1.0) ** 2
    test_loss = lambda x: (x + 1.0) ** 2
    assert calibration_loss(candidate) < calibration_loss(initial)
    assert test_loss(candidate) > test_loss(initial)
