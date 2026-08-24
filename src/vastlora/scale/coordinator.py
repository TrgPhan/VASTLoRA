from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch

from vastlora.lowrank import (
    CompactSVD,
    build_adaptive_temporal_reference,
    build_temporal_reference,
    project_to_reference,
    recompress,
    weighted_sum,
)


Method = Literal["freshness", "vast", "mtip", "mtip_adaptive"]


@dataclass(frozen=True)
class TransportConfig:
    freshness_lambda: float = 0.17328679513998632
    reference_rank: int = 4
    reference_decay: float = 0.1
    reference_singular_power: float = 0.0
    adaptive_energy: float = 0.9
    adaptive_min_rank: int = 2
    adaptive_max_rank: int = 16
    adaptive_singular_power: float = 1.0
    rank_rtol: float = 1e-5


@dataclass(frozen=True)
class TransportResult:
    update: CompactSVD
    freshness: float
    rho: float
    left_rank: int
    right_rank: int
    left_retained_energy: float | None = None
    right_retained_energy: float | None = None


def zero_compact(
    shape: tuple[int, int],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> CompactSVD:
    rows, columns = shape
    return CompactSVD(
        torch.empty((rows, 0), dtype=dtype, device=device),
        torch.empty((0,), dtype=dtype, device=device),
        torch.empty((columns, 0), dtype=dtype, device=device),
    )


def transport_compact_update(
    update: CompactSVD,
    history: Sequence[CompactSVD],
    *,
    method: Method,
    staleness: int,
    config: TransportConfig,
    max_rank: int,
) -> TransportResult:
    """Transport one update without materializing its dense matrix."""

    if staleness < 0:
        raise ValueError("staleness must be non-negative")
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")

    freshness = math.exp(-config.freshness_lambda * staleness)
    if method == "freshness" or not history:
        transported = recompress(
            update.as_lowrank().scaled(freshness),
            max_rank=max_rank,
            rtol=config.rank_rtol,
        )
        return TransportResult(transported, freshness, 1.0, 0, 0)

    if method == "mtip_adaptive":
        reference = build_adaptive_temporal_reference(
            history,
            energy_threshold=config.adaptive_energy,
            min_rank=config.adaptive_min_rank,
            max_rank=config.adaptive_max_rank,
            decay=config.reference_decay,
            singular_power=config.adaptive_singular_power,
        )
        q_left = reference.q_left
        q_right = reference.q_right
        left_energy = reference.left_retained_energy
        right_energy = reference.right_retained_energy
    else:
        q_left, q_right = build_temporal_reference(
            history,
            left_rank=config.reference_rank,
            right_rank=config.reference_rank,
            decay=config.reference_decay,
            singular_power=config.reference_singular_power,
        )
        left_energy = None
        right_energy = None

    projected, _, rho_tensor = project_to_reference(update, q_left, q_right)
    if method in {"mtip", "mtip_adaptive"}:
        candidate = projected
    elif method == "vast":
        residual = weighted_sum([update.as_lowrank(), projected], [1.0, -1.0])
        candidate = weighted_sum([projected, residual], [1.0, freshness])
    else:
        raise ValueError(f"unknown transport method: {method}")

    transported = recompress(candidate, max_rank=max_rank, rtol=config.rank_rtol)
    return TransportResult(
        transported,
        freshness,
        float(rho_tensor.item()),
        q_left.shape[1],
        q_right.shape[1],
        left_energy,
        right_energy,
    )


def aggregate_compact_state(
    server: CompactSVD,
    update: CompactSVD,
    *,
    weight: float,
    max_rank: int,
    rank_rtol: float = 1e-5,
) -> CompactSVD:
    """Apply an update to a compact server adapter under a rank budget."""

    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    if server.shape != update.shape:
        raise ValueError("server and update shapes must match")
    if server.rank == 0:
        combined = update.as_lowrank().scaled(weight)
    else:
        combined = weighted_sum([server.as_lowrank(), update.as_lowrank()], [1.0, weight])
    return recompress(combined, max_rank=max_rank, rtol=rank_rtol)
