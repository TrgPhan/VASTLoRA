from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class LowRankMatrix:
    """Matrix represented as left @ right without dense materialization."""

    left: Tensor
    right: Tensor

    def __post_init__(self) -> None:
        if self.left.ndim != 2 or self.right.ndim != 2:
            raise ValueError("left and right must be matrices")
        if self.left.shape[1] != self.right.shape[0]:
            raise ValueError(
                "incompatible low-rank factors: "
                f"{tuple(self.left.shape)} and {tuple(self.right.shape)}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return self.left.shape[0], self.right.shape[1]

    @property
    def factor_rank(self) -> int:
        return self.left.shape[1]

    @property
    def dtype(self) -> torch.dtype:
        return self.left.dtype

    @property
    def device(self) -> torch.device:
        return self.left.device

    def dense(self) -> Tensor:
        return self.left @ self.right

    def fro_norm_sq(self) -> Tensor:
        gram_left = self.left.T @ self.left
        gram_right = self.right @ self.right.T
        return torch.sum(gram_left * gram_right)

    def scaled(self, weight: float | Tensor) -> "LowRankMatrix":
        return LowRankMatrix(self.left * weight, self.right)


@dataclass(frozen=True)
class CompactSVD:
    """Compact SVD represented as U, singular values, and V."""

    u: Tensor
    s: Tensor
    v: Tensor

    def __post_init__(self) -> None:
        if self.u.ndim != 2 or self.v.ndim != 2 or self.s.ndim != 1:
            raise ValueError("u and v must be matrices; s must be a vector")
        if self.u.shape[1] != self.s.shape[0] or self.v.shape[1] != self.s.shape[0]:
            raise ValueError("compact SVD dimensions do not match")

    @property
    def rank(self) -> int:
        return self.s.numel()

    @property
    def shape(self) -> tuple[int, int]:
        return self.u.shape[0], self.v.shape[0]

    def dense(self) -> Tensor:
        return (self.u * self.s.unsqueeze(0)) @ self.v.T

    def as_lowrank(self) -> LowRankMatrix:
        return LowRankMatrix(self.u * self.s.unsqueeze(0), self.v.T)

    def fro_norm_sq(self) -> Tensor:
        return torch.sum(self.s.square())


@dataclass(frozen=True)
class AdaptiveReference:
    """Two-sided temporal basis with independently selected ranks."""

    q_left: Tensor
    q_right: Tensor
    left_retained_energy: float
    right_retained_energy: float

    @property
    def left_rank(self) -> int:
        return self.q_left.shape[1]

    @property
    def right_rank(self) -> int:
        return self.q_right.shape[1]


def exact_lora_innovation(
    b_final: Tensor,
    a_final: Tensor,
    b_initial: Tensor,
    a_initial: Tensor,
) -> LowRankMatrix:
    """Return D = B_final A_final - B_initial A_initial as exact low-rank factors."""

    _check_lora_pair(b_final, a_final, "final")
    _check_lora_pair(b_initial, a_initial, "initial")
    if b_final.shape != b_initial.shape or a_final.shape != a_initial.shape:
        raise ValueError("final and initial LoRA factors must have the same shapes")

    left = torch.cat([b_final, b_initial], dim=1)
    right = torch.cat([a_final, -a_initial], dim=0)
    return LowRankMatrix(left, right)


def compact_svd(
    matrix: LowRankMatrix,
    *,
    rtol: float = 1e-6,
    max_rank: int | None = None,
) -> CompactSVD:
    """Compute compact SVD of left @ right through QR and a small SVD."""

    q_left, t_left = torch.linalg.qr(matrix.left, mode="reduced")
    q_right, t_right = torch.linalg.qr(matrix.right.T, mode="reduced")
    middle = t_left @ t_right.T

    p, s, qh = torch.linalg.svd(middle, full_matrices=False)
    if s.numel() == 0:
        keep = torch.zeros(0, dtype=torch.bool, device=s.device)
    else:
        keep = s >= (rtol * s[0])
    if max_rank is not None:
        rank_mask = torch.arange(s.numel(), device=s.device) < max_rank
        keep = keep & rank_mask

    if not torch.any(keep):
        empty = 0
        return CompactSVD(
            q_left[:, :empty],
            s[:empty],
            q_right[:, :empty],
        )

    p = p[:, keep]
    s = s[keep]
    q = qh.T[:, keep]
    return CompactSVD(q_left @ p, s, q_right @ q)


def weighted_sum(
    matrices: Sequence[LowRankMatrix],
    weights: Sequence[float | Tensor] | None = None,
) -> LowRankMatrix:
    """Represent sum_i weights_i * matrices_i by concatenating factors."""

    if not matrices:
        raise ValueError("at least one low-rank matrix is required")

    shape = matrices[0].shape
    for matrix in matrices:
        if matrix.shape != shape:
            raise ValueError("all low-rank matrices must have the same dense shape")

    if weights is None:
        weights = [1.0] * len(matrices)
    if len(weights) != len(matrices):
        raise ValueError("weights and matrices must have the same length")

    left_parts = [matrix.left * weight for matrix, weight in zip(matrices, weights)]
    right_parts = [matrix.right for matrix in matrices]
    return LowRankMatrix(torch.cat(left_parts, dim=1), torch.cat(right_parts, dim=0))


def recompress(
    matrix: LowRankMatrix,
    *,
    max_rank: int,
    rtol: float = 1e-6,
) -> CompactSVD:
    """Recompress a low-rank matrix to a rank budget."""

    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    return compact_svd(matrix, rtol=rtol, max_rank=max_rank)


def build_temporal_reference(
    history: Sequence[CompactSVD],
    *,
    left_rank: int,
    right_rank: int,
    decay: float = 0.0,
    singular_power: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Build Q_L^t and Q_R^t from recent accepted compact SVD updates."""

    if not history:
        raise ValueError("history must contain at least one compact SVD")
    if left_rank <= 0 or right_rank <= 0:
        raise ValueError("reference ranks must be positive")
    if singular_power < 0:
        raise ValueError("singular_power must be non-negative")

    weights = _recency_weights(len(history), decay, history[0].s)
    left_blocks: list[Tensor] = []
    right_blocks: list[Tensor] = []
    for svd, weight in zip(reversed(history), weights):
        scale = torch.sqrt(weight) * svd.s.pow(singular_power)
        left_blocks.append(svd.u * scale.unsqueeze(0))
        right_blocks.append(svd.v * scale.unsqueeze(0))

    left_matrix = torch.cat(left_blocks, dim=1)
    right_matrix = torch.cat(right_blocks, dim=1)
    q_left = _top_left_singular_vectors(left_matrix, left_rank)
    q_right = _top_left_singular_vectors(right_matrix, right_rank)
    return q_left, q_right


def build_adaptive_temporal_reference(
    history: Sequence[CompactSVD],
    *,
    energy_threshold: float = 0.9,
    min_rank: int = 1,
    max_rank: int | None = None,
    decay: float = 0.0,
    singular_power: float = 0.0,
) -> AdaptiveReference:
    """Choose left and right temporal ranks from cumulative spectral energy."""

    if not history:
        raise ValueError("history must contain at least one compact SVD")
    if not 0.0 < energy_threshold <= 1.0:
        raise ValueError("energy_threshold must be in (0, 1]")
    if min_rank <= 0:
        raise ValueError("min_rank must be positive")
    if max_rank is not None and max_rank < min_rank:
        raise ValueError("max_rank must be at least min_rank")
    if singular_power < 0:
        raise ValueError("singular_power must be non-negative")

    weights = _recency_weights(len(history), decay, history[0].s)
    left_blocks: list[Tensor] = []
    right_blocks: list[Tensor] = []
    for svd, weight in zip(reversed(history), weights):
        scale = torch.sqrt(weight) * svd.s.pow(singular_power)
        left_blocks.append(svd.u * scale.unsqueeze(0))
        right_blocks.append(svd.v * scale.unsqueeze(0))

    q_left, left_energy = _adaptive_left_singular_vectors(
        torch.cat(left_blocks, dim=1),
        energy_threshold=energy_threshold,
        min_rank=min_rank,
        max_rank=max_rank,
    )
    q_right, right_energy = _adaptive_left_singular_vectors(
        torch.cat(right_blocks, dim=1),
        energy_threshold=energy_threshold,
        min_rank=min_rank,
        max_rank=max_rank,
    )
    return AdaptiveReference(q_left, q_right, left_energy, right_energy)


def project_to_reference(
    update: CompactSVD,
    q_left_ref: Tensor,
    q_right_ref: Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[LowRankMatrix, Tensor, Tensor]:
    """Project update to Q_L C Q_R^T and return projection, core, and rho."""

    if q_left_ref.ndim != 2 or q_right_ref.ndim != 2:
        raise ValueError("reference bases must be matrices")
    if q_left_ref.shape[0] != update.u.shape[0] or q_right_ref.shape[0] != update.v.shape[0]:
        raise ValueError("reference bases are incompatible with update shape")

    left_coords = q_left_ref.T @ update.u
    right_coords = update.v.T @ q_right_ref
    core = (left_coords * update.s.unsqueeze(0)) @ right_coords
    projected = LowRankMatrix(q_left_ref @ core, q_right_ref.T)

    numerator = torch.sum(core.square())
    denominator = update.fro_norm_sq() + eps
    rho = torch.clamp(numerator / denominator, min=0.0, max=1.0)
    return projected, core, rho


def compatibility_scores(
    update: CompactSVD,
    q_left_ref: Tensor,
    q_right_ref: Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return left, right, and two-sided retained-energy compatibility."""

    if q_left_ref.ndim != 2 or q_right_ref.ndim != 2:
        raise ValueError("reference bases must be matrices")
    if q_left_ref.shape[0] != update.u.shape[0] or q_right_ref.shape[0] != update.v.shape[0]:
        raise ValueError("reference bases are incompatible with update shape")

    left_coords = q_left_ref.T @ update.u
    right_coords = update.v.T @ q_right_ref
    left_energy = torch.sum((left_coords * update.s.unsqueeze(0)).square())
    right_energy = torch.sum((update.s.unsqueeze(1) * right_coords).square())
    core = (left_coords * update.s.unsqueeze(0)) @ right_coords
    two_sided_energy = torch.sum(core.square())
    denominator = update.fro_norm_sq() + eps
    return tuple(
        torch.clamp(value / denominator, min=0.0, max=1.0)
        for value in (left_energy, right_energy, two_sided_energy)
    )


def _check_lora_pair(b_factor: Tensor, a_factor: Tensor, name: str) -> None:
    if b_factor.ndim != 2 or a_factor.ndim != 2:
        raise ValueError(f"{name} LoRA factors must be matrices")
    if b_factor.shape[1] != a_factor.shape[0]:
        raise ValueError(
            f"{name} LoRA factors have incompatible shapes: "
            f"{tuple(b_factor.shape)} and {tuple(a_factor.shape)}"
        )


def _recency_weights(count: int, decay: float, like: Tensor) -> Tensor:
    offsets = torch.arange(count, dtype=like.dtype, device=like.device)
    raw = torch.exp(-decay * offsets)
    return raw / raw.sum()


def _top_left_singular_vectors(matrix: Tensor, rank: int) -> Tensor:
    u, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return u[:, : min(rank, u.shape[1])]


def _adaptive_left_singular_vectors(
    matrix: Tensor,
    *,
    energy_threshold: float,
    min_rank: int,
    max_rank: int | None,
) -> tuple[Tensor, float]:
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    available = s.numel()
    rank_cap = available if max_rank is None else min(max_rank, available)
    if rank_cap == 0:
        return u[:, :0], 0.0

    energy = s.square()
    total = energy.sum()
    if float(total) == 0.0:
        selected_rank = min(min_rank, rank_cap)
        return u[:, :selected_rank], 0.0

    cumulative = torch.cumsum(energy, dim=0) / total
    threshold = torch.tensor(energy_threshold, device=s.device, dtype=s.dtype)
    selected_rank = int(torch.searchsorted(cumulative, threshold).item()) + 1
    selected_rank = max(min_rank, min(selected_rank, rank_cap))
    retained = float(cumulative[selected_rank - 1].item())
    return u[:, :selected_rank], retained
