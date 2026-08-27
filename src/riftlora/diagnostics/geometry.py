from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from riftlora.lowrank import (
    CompactSVD,
    LowRankMatrix,
    build_temporal_reference,
    compact_svd,
    compatibility_scores,
    project_to_reference,
)


@dataclass(frozen=True)
class GeometryResult:
    fro_norm: float
    effective_rank: int
    rho_left: float
    rho_right: float
    rho_two_sided: float
    compact_updates: dict[str, CompactSVD]
    projected_left_updates: dict[str, torch.Tensor]
    projected_right_updates: dict[str, torch.Tensor]
    projected_updates: dict[str, torch.Tensor]


@dataclass(frozen=True)
class PersistentProjectionResult:
    projected_updates: dict[str, torch.Tensor]
    left_ranks: dict[str, int]
    right_ranks: dict[str, int]


def analyze_innovation_geometry(
    innovations: Mapping[str, LowRankMatrix],
    history: Mapping[str, Sequence[CompactSVD]],
    *,
    reference_rank: int,
    history_size: int,
    reference_decay: float,
    reference_singular_power: float = 0.0,
    rank_rtol: float = 1e-5,
) -> GeometryResult:
    if not innovations:
        raise ValueError("innovations must not be empty")
    if set(innovations) != set(history):
        raise ValueError("innovation and history keys must match")
    if history_size <= 0:
        raise ValueError("history_size must be positive")

    total_energy = torch.tensor(0.0, device=next(iter(innovations.values())).device)
    left_energy = torch.zeros_like(total_energy)
    right_energy = torch.zeros_like(total_energy)
    two_sided_energy = torch.zeros_like(total_energy)
    effective_rank = 0
    compact_updates: dict[str, CompactSVD] = {}
    projected_left_updates: dict[str, torch.Tensor] = {}
    projected_right_updates: dict[str, torch.Tensor] = {}
    projected_updates: dict[str, torch.Tensor] = {}

    for name, innovation in innovations.items():
        layer_history = list(history[name])[-history_size:]
        if not layer_history:
            raise ValueError("at least one history update is required for every layer")
        update = compact_svd(innovation, rtol=rank_rtol)
        compact_updates[name] = update
        effective_rank += update.rank
        denominator = update.fro_norm_sq()
        total_energy = total_energy + denominator

        q_left, q_right = build_temporal_reference(
            layer_history,
            left_rank=reference_rank,
            right_rank=reference_rank,
            decay=reference_decay,
            singular_power=reference_singular_power,
        )
        layer_left, layer_right, layer_two = compatibility_scores(
            update,
            q_left,
            q_right,
        )
        left_energy = left_energy + layer_left * denominator
        right_energy = right_energy + layer_right * denominator
        two_sided_energy = two_sided_energy + layer_two * denominator
        projection, _, _ = project_to_reference(update, q_left, q_right)
        dense = update.dense()
        projected_left_updates[name] = q_left @ (q_left.T @ dense)
        projected_right_updates[name] = (dense @ q_right) @ q_right.T
        projected_updates[name] = projection.dense()

    denominator = torch.clamp(total_energy, min=1e-12)
    return GeometryResult(
        fro_norm=float(torch.sqrt(total_energy).item()),
        effective_rank=effective_rank,
        rho_left=float((left_energy / denominator).item()),
        rho_right=float((right_energy / denominator).item()),
        rho_two_sided=float((two_sided_energy / denominator).item()),
        compact_updates=compact_updates,
        projected_left_updates=projected_left_updates,
        projected_right_updates=projected_right_updates,
        projected_updates=projected_updates,
    )


def subspace_lattice_transport(
    innovations: Mapping[str, LowRankMatrix],
    left_projections: Mapping[str, torch.Tensor],
    right_projections: Mapping[str, torch.Tensor],
    two_sided_projections: Mapping[str, torch.Tensor],
    *,
    single_side_weight: float,
    neither_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Weight the four blocks induced by left/right temporal projectors."""

    if not 0.0 <= single_side_weight <= 1.0:
        raise ValueError("single_side_weight must be in [0, 1]")
    if not 0.0 <= neither_weight <= 1.0:
        raise ValueError("neither_weight must be in [0, 1]")
    keys = set(innovations)
    if keys != set(left_projections) or keys != set(right_projections):
        raise ValueError("innovation and one-sided projection keys must match")
    if keys != set(two_sided_projections):
        raise ValueError("innovation and two-sided projection keys must match")

    transported: dict[str, torch.Tensor] = {}
    for name, update in innovations.items():
        dense = update.dense()
        both = two_sided_projections[name]
        single = left_projections[name] + right_projections[name] - 2.0 * both
        neither = dense - left_projections[name] - right_projections[name] + both
        transported[name] = (
            both + single_side_weight * single + neither_weight * neither
        )
    return transported


def persistent_temporal_projection(
    innovations: Mapping[str, LowRankMatrix],
    history: Mapping[str, Sequence[CompactSVD]],
    *,
    max_rank: int,
    short_history_size: int,
    long_history_size: int,
    overlap_threshold: float,
    reference_decay: float,
    rank_rtol: float = 1e-5,
) -> PersistentProjectionResult:
    """Project onto directions shared by short- and long-timescale references."""

    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    if not 0 < short_history_size < long_history_size:
        raise ValueError("history sizes must satisfy 0 < short < long")
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be in [0, 1]")
    if set(innovations) != set(history):
        raise ValueError("innovation and history keys must match")

    projected: dict[str, torch.Tensor] = {}
    left_ranks: dict[str, int] = {}
    right_ranks: dict[str, int] = {}
    for name, innovation in innovations.items():
        layer_history = list(history[name])[-long_history_size:]
        if len(layer_history) < long_history_size:
            raise ValueError("persistent projection requires a full long history")
        short_left, short_right = build_temporal_reference(
            layer_history[-short_history_size:],
            left_rank=max_rank,
            right_rank=max_rank,
            decay=reference_decay,
        )
        long_left, long_right = build_temporal_reference(
            layer_history,
            left_rank=max_rank,
            right_rank=max_rank,
            decay=reference_decay,
        )
        stable_left = _persistent_basis(short_left, long_left, overlap_threshold)
        stable_right = _persistent_basis(short_right, long_right, overlap_threshold)
        update = compact_svd(innovation, rtol=rank_rtol)
        projection, _, _ = project_to_reference(update, stable_left, stable_right)
        projected[name] = projection.dense()
        left_ranks[name] = stable_left.shape[1]
        right_ranks[name] = stable_right.shape[1]
    return PersistentProjectionResult(projected, left_ranks, right_ranks)


def _persistent_basis(short: Tensor, long: Tensor, threshold: float) -> Tensor:
    _, singular_values, vh = torch.linalg.svd(short.T @ long, full_matrices=False)
    if singular_values.numel() == 0:
        return long[:, :0]
    keep = singular_values.square() >= threshold
    if not torch.any(keep):
        keep[torch.argmax(singular_values)] = True
    return long @ vh.T[:, keep]


def transport_innovations(
    innovations: Mapping[str, LowRankMatrix],
    projections: Mapping[str, torch.Tensor],
    *,
    freshness: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= freshness <= 1.0:
        raise ValueError("freshness must be in [0, 1]")
    if set(innovations) != set(projections):
        raise ValueError("innovation and projection keys must match")
    return {
        name: freshness * update.dense() + (1.0 - freshness) * projections[name]
        for name, update in innovations.items()
    }


def residual_budget_transport(
    innovations: Mapping[str, LowRankMatrix],
    projections: Mapping[str, torch.Tensor],
    *,
    freshness: float,
    residual_budget: float,
    projection_scale_cap: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Transport with projection-energy compensation and a bounded stale residual."""

    if not 0.0 <= freshness <= 1.0:
        raise ValueError("freshness must be in [0, 1]")
    if residual_budget < 0:
        raise ValueError("residual_budget must be non-negative")
    if projection_scale_cap < 1.0:
        raise ValueError("projection_scale_cap must be at least one")
    if set(innovations) != set(projections):
        raise ValueError("innovation and projection keys must match")

    transported: dict[str, torch.Tensor] = {}
    eps = 1e-12
    for name, update in innovations.items():
        dense = update.dense()
        projection = projections[name]
        residual = dense - projection
        dense_norm = torch.linalg.matrix_norm(dense, ord="fro")
        projection_norm = torch.linalg.matrix_norm(projection, ord="fro")
        residual_norm = torch.linalg.matrix_norm(residual, ord="fro")

        target_scale = torch.clamp(
            dense_norm / torch.clamp(projection_norm, min=eps),
            min=1.0,
            max=projection_scale_cap,
        )
        projection_scale = 1.0 + (1.0 - freshness) * (target_scale - 1.0)
        trust_ratio = residual_budget * projection_norm / torch.clamp(
            residual_norm, min=eps
        )
        residual_scale = freshness * torch.clamp(trust_ratio, min=0.0, max=1.0)
        transported[name] = projection_scale * projection + residual_scale * residual
    return transported

