from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from vastlora.lowrank import LowRankMatrix, compact_svd


@dataclass(frozen=True)
class RankwiseFilterResult:
    """Objective-aware filtering of intrinsic rank-one innovation components."""

    updates: dict[str, torch.Tensor]
    predicted_gain: float
    retained_rank: int
    total_rank: int
    positive_fraction: float
    layer_scores: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class PairedGateResult:
    """Paired uncertainty estimate for a candidate's calibration-loss change."""

    mean_delta: float
    standard_error: float
    upper_bound: float
    accepted: bool


def filter_rankwise_by_gradient(
    innovations: Mapping[str, LowRankMatrix],
    gradients: Mapping[str, torch.Tensor],
    *,
    minimum_predicted_gain: float = 0.0,
    max_components: int | None = None,
    rank_rtol: float = 1e-5,
    keep_nonpositive: bool = False,
) -> RankwiseFilterResult:
    """Keep intrinsic SVD components predicted to decrease calibration loss.

    For D = sum_j sigma_j u_j v_j^T and calibration gradient G, the
    first-order predicted gain of component j is -sigma_j u_j^T G v_j.
    The score is invariant to the sign ambiguity of singular vectors.
    """

    if set(innovations) != set(gradients):
        raise ValueError("innovation and gradient keys must match")
    if minimum_predicted_gain < 0.0:
        raise ValueError("minimum_predicted_gain must be non-negative")
    if max_components is not None and max_components <= 0:
        raise ValueError("max_components must be positive when provided")

    components: list[tuple[float, str, int]] = []
    compact_by_layer = {}
    scores_by_layer: dict[str, tuple[float, ...]] = {}
    for name, innovation in innovations.items():
        compact = compact_svd(innovation, rtol=rank_rtol)
        gradient = gradients[name].to(device=compact.u.device, dtype=compact.u.dtype)
        if gradient.shape != compact.shape:
            raise ValueError(f"gradient shape mismatch for {name}")
        compact_by_layer[name] = compact
        if compact.rank == 0:
            scores = torch.empty(0, device=compact.u.device, dtype=compact.u.dtype)
        else:
            contractions = torch.diagonal(compact.u.T @ gradient @ compact.v)
            scores = -compact.s * contractions
        scores_by_layer[name] = tuple(float(value.item()) for value in scores)
        components.extend(
            (float(score.item()), name, index)
            for index, score in enumerate(scores)
            if keep_nonpositive or float(score.item()) > minimum_predicted_gain
        )

    components.sort(key=lambda value: value[0], reverse=True)
    if max_components is not None:
        components = components[:max_components]
    selected = {(name, index) for _, name, index in components}

    updates: dict[str, torch.Tensor] = {}
    for name, compact in compact_by_layer.items():
        keep = torch.tensor(
            [(name, index) in selected for index in range(compact.rank)],
            dtype=torch.bool,
            device=compact.u.device,
        )
        if torch.any(keep):
            updates[name] = (
                compact.u[:, keep] * compact.s[keep].unsqueeze(0)
            ) @ compact.v[:, keep].T
        else:
            updates[name] = torch.zeros(
                compact.shape,
                dtype=compact.u.dtype,
                device=compact.u.device,
            )

    total_rank = sum(compact.rank for compact in compact_by_layer.values())
    retained_rank = len(selected)
    return RankwiseFilterResult(
        updates=updates,
        predicted_gain=sum(score for score, _, _ in components),
        retained_rank=retained_rank,
        total_rank=total_rank,
        positive_fraction=retained_rank / total_rank if total_rank else 0.0,
        layer_scores=scores_by_layer,
    )


def paired_loss_gate(
    current_losses: torch.Tensor,
    candidate_losses: torch.Tensor,
    *,
    z_value: float = 1.0,
    max_mean_increase: float = 0.0,
) -> PairedGateResult:
    """Accept when an upper confidence bound permits the candidate step."""

    if current_losses.ndim != 1 or candidate_losses.ndim != 1:
        raise ValueError("paired losses must be one-dimensional")
    if current_losses.shape != candidate_losses.shape or current_losses.numel() == 0:
        raise ValueError("paired losses must have the same non-empty shape")
    if z_value < 0.0:
        raise ValueError("z_value must be non-negative")

    deltas = candidate_losses.detach().float() - current_losses.detach().float()
    mean_delta = float(deltas.mean().item())
    if deltas.numel() > 1:
        standard_error = float(
            (deltas.std(unbiased=True) / math.sqrt(deltas.numel())).item()
        )
    else:
        standard_error = 0.0
    upper_bound = mean_delta + z_value * standard_error
    return PairedGateResult(
        mean_delta=mean_delta,
        standard_error=standard_error,
        upper_bound=upper_bound,
        accepted=upper_bound <= max_mean_increase,
    )
