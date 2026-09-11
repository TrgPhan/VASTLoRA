from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

import torch

from riftlora.lowrank import CompactSVD


SpectralPolicy = Literal["abs_select", "smooth_abs", "random_index", "grad_direction"]


@dataclass(frozen=True)
class SpectralSurgeryConfig:
    """Spectral Surgery, arXiv:2603.03995, section 3.4.

    Signed step sizes below are explicit reproduction choices: the paper gives
    their equations but does not publish numerical defaults for them.
    """

    policy: SpectralPolicy = "smooth_abs"
    core_frac: float = 0.2
    noise_frac: float = 0.2
    min_core_k: int = 1
    amp_factor: float = 1.25
    sup_factor: float = 0.8
    mid_factor: float = 1.0
    smooth_temperature: float = 0.35
    smooth_center_q: float = 0.5
    smooth_align_mid: bool = False
    asymmetric_update: bool = True
    eta: float = 0.1
    eta_suppress: float = 0.1
    eta_enhance: float = 0.1
    positive_power: float = 1.0
    sigma_clip_min: float = 0.0
    random_seed: int = 0
    preserve_energy: Literal["l1", "none"] = "l1"
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.policy not in {"abs_select", "smooth_abs", "random_index", "grad_direction"}:
            raise ValueError("unsupported spectral policy")
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
            ("positive_power", self.positive_power),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("eta", "eta_suppress", "eta_enhance", "sigma_clip_min"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.smooth_align_mid and not self.sup_factor < self.mid_factor < self.amp_factor:
            raise ValueError("smooth_align_mid requires sup_factor < mid_factor < amp_factor")
        if not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if self.preserve_energy not in {"l1", "none"}:
            raise ValueError("preserve_energy must be 'l1' or 'none'")


def reweight_compact_spectra(
    updates: Mapping[str, CompactSVD],
    sensitivities: Mapping[str, torch.Tensor],
    *,
    config: SpectralSurgeryConfig,
    signed_sensitivities: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, CompactSVD]:
    """Reweight singular values while preserving every singular direction."""

    config.validate()
    if set(updates) != set(sensitivities):
        raise ValueError("updates and sensitivities must have the same keys")
    if config.policy == "grad_direction" and (
        signed_sensitivities is None or set(updates) != set(signed_sensitivities)
    ):
        raise ValueError("grad_direction requires signed_sensitivities for every module")

    edited: dict[str, CompactSVD] = {}
    generator = torch.Generator(device="cpu").manual_seed(config.random_seed)
    for name in sorted(updates):
        update = updates[name]
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
        elif config.policy == "smooth_abs":
            factors = _smooth_factors(normalized, config)
        elif config.policy == "random_index":
            order = torch.randperm(update.rank, generator=generator).to(update.s.device)
            factors = _hard_selection_factors(normalized, config, order=order)
        else:
            signed = signed_sensitivities[name].detach().to(update.s).flatten()
            if signed.numel() != update.rank or not torch.isfinite(signed).all():
                raise ValueError(f"invalid signed sensitivities for {name}")
            signed = signed / signed.abs().mean().clamp_min(config.epsilon)
            if config.asymmetric_update:
                effective = (
                    config.eta_suppress * signed.clamp_min(0).pow(config.positive_power)
                    - config.eta_enhance * (-signed).clamp_min(0)
                )
            else:
                effective = config.eta * signed
            factors = torch.exp(-effective)

        singular_values = (update.s * factors).clamp_min(config.sigma_clip_min)
        if not torch.isfinite(singular_values).all():
            raise ValueError(f"non-finite edited spectrum for {name}; reduce signed step sizes")
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
    *,
    order: torch.Tensor | None = None,
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
    if order is None:
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
    if config.smooth_align_mid:
        midpoint = (config.mid_factor - config.sup_factor) / (config.amp_factor - config.sup_factor)
        center = center - temperature * math.log(midpoint / (1.0 - midpoint))
    gate = torch.sigmoid((normalized - center) / temperature)
    return config.sup_factor + (config.amp_factor - config.sup_factor) * gate


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def edit_trained_adapter(model, state, weighted_batches, *, config, target_modules, loss_fn=None):
    """Edit a loaded, trained adapter once; preserve unselected module updates.

    ``loss_fn=None`` uses the model's answer-token teacher-forcing loss. The
    caller supplies one example per weighted batch and loads ``state`` first.
    """
    from riftlora.scale.objective import score_compact_component_sensitivities_microbatched

    selected = {name for name in state if name.rsplit(".", 1)[-1] in target_modules}
    found = {name.rsplit(".", 1)[-1] for name in selected}
    if not target_modules or found != set(target_modules):
        raise ValueError("every spectral edit target must exist in the trained adapter")
    directions = {
        name: compact if name in selected else CompactSVD(compact.u[:, :0], compact.s[:0], compact.v[:, :0])
        for name, compact in state.items()
    }
    scores = score_compact_component_sensitivities_microbatched(
        model, directions, weighted_batches, loss_fn=loss_fn
    )
    edited = reweight_compact_spectra(
        directions, scores.sensitivities, config=config,
        signed_sensitivities=scores.signed_sensitivities,
    )
    return {name: edited[name] if name in selected else compact for name, compact in state.items()}
