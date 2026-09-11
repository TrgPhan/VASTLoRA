import pytest
import torch

from riftlora.scale.alignfed_reference import (
    AlignFedBuffer, FactorReturn, RankLinearTransform, center_version_groups,
    fairness_weights, fit_semantic_transform, representation_penalty,
)


def _return(client, version, value):
    return FactorReturn(client, version, {"a": torch.tensor([[float(value)]])})


def test_singleton_groups_are_zero_and_pair_is_zero_sum():
    centered = center_version_groups([_return("a", 0, 2), _return("b", 0, 4), _return("c", 1, 10)])
    assert [x.delta["a"].item() for x in centered] == [-1, 1, 0]


def test_fairness_matches_equation_and_handles_large_staleness():
    items = [_return("a", 4, 2), _return("b", 2, 4)]
    actual = fairness_weights(items, current_version=4, upload_counts={"a": 0, "b": 3}, gamma=0.2)
    weights = torch.tensor([1 / (2 + 1e-8), torch.exp(torch.tensor(-0.4)).item() / (2 * (4 + 1e-8))], dtype=torch.float64)
    torch.testing.assert_close(actual, weights / weights.sum())
    huge = fairness_weights(items, current_version=1000000, upload_counts={}, gamma=0.3)
    assert torch.isfinite(huge).all()
    assert huge.sum().item() == pytest.approx(1.)


def test_singleton_buffer_is_identity_not_an_accuracy_improvement():
    buffer = AlignFedBuffer(max_pending=1, max_wait=10)
    buffer.add(_return("a", 0, 8), now=1)
    state = {"a": torch.tensor([[5.]])}
    result, diag = buffer.flush(state, current_version=0, now=1, align_group=lambda v, d: d)
    torch.testing.assert_close(result["a"], state["a"])
    assert diag["singleton_groups"] == diag["zero_centered_returns"] == 1


def test_equal_weight_centered_pair_cancels_without_staleness():
    buffer = AlignFedBuffer(max_pending=2, max_wait=10)
    buffer.add(_return("a", 0, 2), now=1)
    buffer.add(_return("b", 0, 4), now=2)
    result, _ = buffer.flush({"a": torch.zeros(1, 1)}, current_version=0, now=2,
                             align_group=lambda v, d: pytest.fail("fresh group must use identity"))
    assert result["a"].item() == 0


def test_timer_flush_and_rejected_early_flush_do_not_lose_updates():
    buffer = AlignFedBuffer(max_pending=4, max_wait=5)
    assert not buffer.add(_return("a", 0, 1), now=1)
    with pytest.raises(ValueError, match="trigger"):
        buffer.flush({"a": torch.zeros(1, 1)}, current_version=0, now=2, align_group=lambda v, d: d)
    assert len(buffer.pending) == 1
    assert buffer.ready(5)
    buffer.flush({"a": torch.zeros(1, 1)}, current_version=1, now=5, align_group=lambda v, d: d)
    assert not buffer.pending
    with pytest.raises(ValueError, match="monotonic"):
        buffer.ready(4)


def test_future_version_and_nonlinear_zero_repair_are_rejected():
    with pytest.raises(ValueError, match="future"):
        fairness_weights([_return("a", 2, 1)], current_version=1, upload_counts={})
    buffer = AlignFedBuffer(max_pending=1, max_wait=1)
    buffer.add(_return("a", 0, 2), now=1)
    with pytest.raises(ValueError, match="zero increment"):
        buffer.flush({"a": torch.zeros(1, 1)}, current_version=1, now=1,
                     align_group=lambda v, d: [{"a": torch.ones(1, 1)}])
    assert len(buffer.pending) == 1


def test_penalty_detaches_snapshot_and_uses_squared_l2_not_element_mean():
    local = torch.tensor([[1., 2.]], requires_grad=True)
    snapshot = torch.zeros_like(local, requires_grad=True)
    loss = representation_penalty(local, snapshot, weight=0.1)
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    torch.testing.assert_close(local.grad, torch.tensor([[0.2, 0.4]]))
    assert snapshot.grad is None


def test_transform_has_independent_a_b_maps_and_is_linear():
    delta = {"a": torch.randn(2, 4), "b": torch.randn(3, 2)}
    transform = RankLinearTransform(delta, {"a": "a", "b": "b"})
    for k, v in delta.items():
        torch.testing.assert_close(transform(delta)[k], v)
    with torch.no_grad():
        transform.maps[0].mul_(2)
        transform.maps[1].mul_(3)
    edited = transform(delta)
    torch.testing.assert_close(edited["a"], delta["a"] * 2)
    torch.testing.assert_close(edited["b"], delta["b"] * 3)


def test_semantic_fit_improves_calibration_and_preserves_source_factors():
    source = {"a": torch.zeros(1, 1)}
    deltas = [{"a": torch.ones(1, 1)}]
    transform = fit_semantic_transform(source, deltas, torch.tensor([[2.]]), lambda state: state["a"],
                                      factor_sides={"a": "a"}, steps=40, learning_rate=0.05)
    assert abs(transform(deltas[0])["a"].item() - 2) < 0.15
    assert source["a"].item() == 0
    assert deltas[0]["a"].item() == 1


def test_disconnected_feature_function_fails_instead_of_fake_training():
    with pytest.raises(RuntimeError, match="preserve gradients"):
        fit_semantic_transform({"a": torch.zeros(1, 1)}, [{"a": torch.ones(1, 1)}], torch.ones(1, 1),
                               lambda state: state["a"].detach(), factor_sides={"a": "a"},
                               steps=1, learning_rate=0.01)
