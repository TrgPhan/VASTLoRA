"""Executable AlignFed equations, not an official end-to-end reproduction.

arXiv:2606.08197 equations (3)-(8). Factor coordinates must remain fixed across
versions; compact-SVD reloading changes the coordinates of factor increments.
The rank-linear parameterization of T is an explicit implementation assumption.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import torch


FactorState = dict[str, torch.Tensor]


@dataclass(frozen=True)
class FactorReturn:
    client_id: str
    base_version: int
    delta: FactorState


def _validate_state(state: Mapping[str, torch.Tensor]) -> None:
    if not state:
        raise ValueError("factor state must not be empty")
    for name, tensor in state.items():
        if tensor.ndim != 2 or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid factor tensor: {name}")


def factor_difference(after: FactorState, before: FactorState) -> FactorState:
    _validate_state(before)
    _validate_state(after)
    if set(before) != set(after) or any(before[k].shape != after[k].shape for k in before):
        raise ValueError("factor coordinates/shapes must match; heterogeneous rank needs a separate protocol")
    return {k: (after[k] - before[k]).detach().clone() for k in before}


def center_version_groups(returns: Sequence[FactorReturn]) -> list[FactorReturn]:
    """Literal equation (5). Singleton groups are zero, without a fallback."""
    groups: dict[int, list[int]] = defaultdict(list)
    if not returns:
        raise ValueError("at least one return is required")
    reference = returns[0].delta
    for index, item in enumerate(returns):
        factor_difference(item.delta, reference)
        if item.base_version < 0:
            raise ValueError("base_version must be non-negative")
        groups[item.base_version].append(index)
    centered = list(returns)
    for indices in groups.values():
        mean = {k: torch.stack([returns[i].delta[k] for i in indices]).mean(0) for k in reference}
        for i in indices:
            item = returns[i]
            centered[i] = FactorReturn(item.client_id, item.base_version, factor_difference(item.delta, mean))
    return centered


def fairness_weights(
    centered: Sequence[FactorReturn], *, current_version: int,
    upload_counts: Mapping[str, int], gamma: float = 0.2, epsilon: float = 1e-8,
) -> torch.Tensor:
    """Equation (7), using the norm BEFORE transformation and a stable softmax."""
    if not centered or not math.isfinite(gamma) or gamma < 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("invalid weight settings or empty buffer")
    logits = []
    for item in centered:
        _validate_state(item.delta)
        tau = current_version - item.base_version
        count = upload_counts.get(item.client_id, 0)
        if tau < 0 or count < 0:
            raise ValueError("future versions and negative upload counts are invalid")
        norm = math.sqrt(sum(float(t.double().square().sum()) for t in item.delta.values()))
        logits.append(-gamma * tau - 0.5 * math.log(count + 1) - math.log(norm + epsilon))
    return torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=0)


def representation_penalty(local: torch.Tensor, snapshot: torch.Tensor, *, weight: float) -> torch.Tensor:
    """Equation (3): mean over examples of squared L2 feature discrepancy.

    Caller supplies penultimate features using an explicit, shared token pooling
    rule. Snapshot features are frozen and must correspond to the dispatch model.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("representation weight must be finite and non-negative")
    if local.shape != snapshot.shape or local.ndim != 2 or local.shape[0] == 0:
        raise ValueError("features must be matching nonempty [examples, hidden] tensors")
    return weight * (local.float() - snapshot.detach().to(local).float()).square().sum(-1).mean()


class RankLinearTransform(torch.nn.Module):
    """Separate linear maps on PEFT A (r,in) and B (out,r).

    T_A(delta A)=R_A delta A; T_B(delta B)=delta B R_B. This is a
    low-cost choice consistent with linearity, not a paper-specified architecture.
    """
    def __init__(self, delta: FactorState, factor_sides: Mapping[str, str]):
        super().__init__()
        _validate_state(delta)
        if set(factor_sides) != set(delta) or set(factor_sides.values()) - {"a", "b"}:
            raise ValueError("declare side a or b for every factor")
        self.names = tuple(sorted(delta))
        self.sides = dict(factor_sides)
        self.maps = torch.nn.ParameterList([
            torch.nn.Parameter(torch.eye(
                delta[k].shape[0 if self.sides[k] == "a" else 1],
                device=delta[k].device, dtype=delta[k].dtype,
            )) for k in self.names
        ])

    def forward(self, delta: FactorState) -> FactorState:
        if set(delta) != set(self.names):
            raise ValueError("transform/factor keys must match")
        return {
            k: matrix @ delta[k] if self.sides[k] == "a" else delta[k] @ matrix
            for k, matrix in zip(self.names, self.maps)
        }


def fit_semantic_transform(
    source: FactorState, centered_deltas: Sequence[FactorState], target_features: torch.Tensor,
    feature_fn: Callable[[FactorState], torch.Tensor], *, factor_sides: Mapping[str, str],
    steps: int, learning_rate: float,
) -> RankLinearTransform:
    """Optimize equation (6), shared T per source version.

    feature_fn must DIFFERENTIABLY evaluate the mean calibration penultimate
    feature at the supplied factor state (e.g. torch.func.functional_call).
    It must use only calibration inputs and freeze the backbone. Copying weights
    with load_state_dict severs this gradient and is deliberately rejected.
    """
    if not centered_deltas or steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("alignment needs deltas, positive steps and learning rate")
    _validate_state(source)
    source = {k: v.detach() for k, v in source.items()}
    deltas = [{k: v.detach() for k, v in delta.items()} for delta in centered_deltas]
    for delta in deltas:
        factor_difference(source, delta)
    transform = RankLinearTransform(deltas[0], factor_sides)
    optimizer = torch.optim.Adam(transform.parameters(), lr=learning_rate)
    target = target_features.detach()
    # Retain the identity candidate if optimization worsens the calibration
    # objective. No task labels or held-out measurements enter this selection.
    best_loss = math.inf
    best_state = {k: v.detach().clone() for k, v in transform.state_dict().items()}
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for delta in deltas:
            aligned = transform(delta)
            features = feature_fn({k: source[k] + aligned[k] for k in source})
            if features.shape != target.shape:
                raise ValueError("source and target mean features must have identical shapes")
            loss = (features - target.to(features)).float().square().sum() / len(deltas)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite semantic alignment loss")
            loss_total += float(loss.detach())
            if step < steps:
                if not loss.requires_grad:
                    raise RuntimeError("feature_fn must preserve gradients to the transform")
                gradients = torch.autograd.grad(loss, tuple(transform.parameters()), allow_unused=True)
                if all(gradient is None for gradient in gradients):
                    raise RuntimeError("feature_fn disconnected from the transform")
                for parameter, gradient in zip(transform.parameters(), gradients):
                    if gradient is not None:
                        parameter.grad = gradient if parameter.grad is None else parameter.grad + gradient
        if loss_total < best_loss:
            best_loss = loss_total
            best_state = {k: v.detach().clone() for k, v in transform.state_dict().items()}
        if step < steps:
            optimizer.step()
    transform.load_state_dict(best_state)
    return transform


class AlignFedBuffer:
    """Buffer/group/center/transform/weight/update equations (4)-(8).

    The caller must invoke flush at the deadline even if no upload arrives.
    ``max_pending`` is a count threshold (>=), an explicit boundary convention.
    Upload counts include arrivals in the current buffer, before aggregation.
    """
    def __init__(self, *, max_pending: int, max_wait: float, start_time: float = 0.0):
        if not isinstance(max_pending, int) or max_pending < 1 or not math.isfinite(max_wait) or max_wait <= 0:
            raise ValueError("buffer count and timeout must be positive")
        if not math.isfinite(start_time):
            raise ValueError("start_time must be finite")
        self.max_pending, self.max_wait = max_pending, max_wait
        self.last_aggregation = start_time
        self.last_time = start_time
        self.pending: list[FactorReturn] = []
        self.upload_counts: Counter[str] = Counter()

    def _check_time(self, now: float) -> None:
        if not math.isfinite(now) or now < self.last_time:
            raise ValueError("buffer time must be finite and monotonic")

    def ready(self, now: float) -> bool:
        self._check_time(now)
        return bool(self.pending) and (
            len(self.pending) >= self.max_pending or now - self.last_aggregation >= self.max_wait
        )

    def add(self, item: FactorReturn, *, now: float) -> bool:
        self._check_time(now)
        _validate_state(item.delta)
        if item.base_version < 0:
            raise ValueError("base_version must be non-negative")
        if self.pending:
            factor_difference(item.delta, self.pending[0].delta)
        self.pending.append(FactorReturn(item.client_id, item.base_version, {
            k: v.detach().clone() for k, v in item.delta.items()
        }))
        self.upload_counts[item.client_id] += 1
        self.last_time = now
        return self.ready(now)

    def flush(
        self, current: FactorState, *, current_version: int, now: float,
        align_group: Callable[[int, Sequence[FactorState]], Sequence[FactorState]],
        gamma: float = 0.2,
    ) -> tuple[FactorState, dict]:
        if not self.ready(now):
            raise ValueError("buffer has not reached count or timeout trigger")
        _validate_state(current)
        for item in self.pending:
            factor_difference(item.delta, current)
        centered = center_version_groups(self.pending)
        weights = fairness_weights(centered, current_version=current_version,
                                   upload_counts=self.upload_counts, gamma=gamma)
        groups: dict[int, list[int]] = defaultdict(list)
        for i, item in enumerate(centered):
            groups[item.base_version].append(i)
        transformed = [item.delta for item in centered]
        for version, indices in groups.items():
            if version == current_version:
                continue
            aligned = list(align_group(version, [centered[i].delta for i in indices]))
            if len(aligned) != len(indices):
                raise ValueError("alignment must preserve group size")
            for i, delta in zip(indices, aligned):
                factor_difference(delta, current)
                # A linear map must preserve zero. Reject additive repair masquerading as T.
                if all(torch.count_nonzero(v) == 0 for v in centered[i].delta.values()) and any(
                    torch.count_nonzero(v) != 0 for v in delta.values()
                ):
                    raise ValueError("linear alignment must map a zero increment to zero")
                transformed[i] = delta
        result = {
            k: current[k].detach() + sum(float(w) * delta[k].detach().to(current[k])
                                       for w, delta in zip(weights, transformed))
            for k in current
        }
        _validate_state(result)
        diagnostics = {
            "buffer_returns": len(centered), "version_group_sizes": {str(k): len(v) for k, v in groups.items()},
            "singleton_groups": sum(len(v) == 1 for v in groups.values()),
            "zero_centered_returns": sum(all(torch.count_nonzero(v) == 0 for v in x.delta.values()) for x in centered),
            "weights": weights.tolist(), "new_server_version": current_version + 1,
        }
        self.pending.clear()
        self.last_aggregation = self.last_time = now
        return result, diagnostics
