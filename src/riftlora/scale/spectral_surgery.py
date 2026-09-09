from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

import torch

from riftlora.lowrank import CompactSVD


SpectralPolicy = Literal["abs_select", "smooth_abs"]


@dataclass(frozen=True)
class SpectralSurgeryConfig:
    """Magnitude-based Spectral Surgery settings from Tian et al. (2026)."""

    policy: SpectralPolicy = "smooth_abs"
    core_frac: float = 0.2
    noise_frac: float = 0.2
    min_core_k: int = 1
    amp_factor: float = 1.25
    sup_factor: float = 0.8
    mid_factor: float = 1.0
    smooth_temperature: float = 0.35
    smooth_center_q: float = 0.5
    preserve_energy: Literal["l1", "none"] = "l1"
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.policy not in {"abs_select", "smooth_abs"}:
            raise ValueError("spectral policy must be 'abs_select' or 'smooth_abs'")
        for name, value in (
            ("core_frac", self.core_frac),
            ("noise_frac", self.noise_frac),
            ("smooth_center_q", self.smooth_center_q),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.min_core_k < 0:
            raise ValueError("min_core_k must be non-negative")
        for name, value in (
            ("amp_factor", self.amp_factor),
            ("sup_factor", self.sup_factor),
            ("mid_factor", self.mid_factor),
            ("smooth_temperature", self.smooth_temperature),
            ("epsilon", self.epsilon),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.preserve_energy not in {"l1", "none"}:
            raise ValueError("preserve_energy must be 'l1' or 'none'")


def reweight_compact_spectra(
    updates: Mapping[str, CompactSVD],
    sensitivities: Mapping[str, torch.Tensor],
    *,
    config: SpectralSurgeryConfig,
) -> dict[str, CompactSVD]:
    """Reweight singular values while preserving every singular direction."""

    config.validate()
    if set(updates) != set(sensitivities):
        raise ValueError("updates and sensitivities must have the same keys")

    edited: dict[str, CompactSVD] = {}
    for name, update in updates.items():
        values = sensitivities[name].detach().to(
            device=update.s.device, dtype=update.s.dtype
        ).flatten()
        if values.numel() != update.rank:
            raise ValueError(f"sensitivity rank mismatch for {name}")
        if not torch.isfinite(values).all() or torch.any(values < 0.0):
            raise ValueError(f"sensitivities must be finite and non-negative for {name}")
        if update.rank == 0:
            edited[name] = update
            continue

        normalized = values / values.mean().clamp_min(config.epsilon)
        if config.policy == "abs_select":
            factors = _hard_selection_factors(normalized, config)
        else:
            factors = _smooth_factors(normalized, config)

        singular_values = (update.s * factors).clamp_min(0.0)
        if config.preserve_energy == "l1":
            original_mass = update.s.sum()
            edited_mass = singular_values.sum()
            if float(original_mass.item()) > 0.0 and float(edited_mass.item()) > 0.0:
                singular_values = singular_values * (original_mass / edited_mass)
        edited[name] = CompactSVD(update.u, singular_values, update.v)
    return edited


def _hard_selection_factors(
    normalized: torch.Tensor,
    config: SpectralSurgeryConfig,
) -> torch.Tensor:
    rank = normalized.numel()
    core_count = min(
        rank,
        max(_round_half_up(rank * config.core_frac), config.min_core_k),
    )
    noise_count = min(
        rank - core_count,
        _round_half_up(rank * config.noise_frac),
    )
    order = torch.argsort(normalized, descending=True, stable=True)
    factors = torch.full_like(normalized, config.mid_factor)
    if core_count:
        factors[order[:core_count]] = config.amp_factor
    if noise_count:
        factors[order[-noise_count:]] = config.sup_factor
    return factors


def _smooth_factors(
    normalized: torch.Tensor,
    config: SpectralSurgeryConfig,
) -> torch.Tensor:
    low_q = config.noise_frac
    high_q = 1.0 - config.core_frac
    if high_q <= low_q:
        low_q, high_q = 0.25, 0.75
    center = torch.quantile(normalized, config.smooth_center_q)
    spread = torch.quantile(normalized, high_q) - torch.quantile(normalized, low_q)
    if float(spread.abs().item()) <= config.epsilon:
        return torch.full_like(normalized, config.mid_factor)
    temperature = config.smooth_temperature * spread
    gate = torch.sigmoid((normalized - center) / temperature)
    return config.sup_factor + (config.amp_factor - config.sup_factor) * gate


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))
