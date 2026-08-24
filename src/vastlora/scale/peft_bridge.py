from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn

from vastlora.lowrank import CompactSVD, LowRankMatrix, compact_svd
from vastlora.scale.coordinator import zero_compact


@dataclass(frozen=True)
class FactorSnapshot:
    a: torch.Tensor
    b: torch.Tensor
    scaling: float


def named_peft_lora_modules(
    model: nn.Module,
    *,
    adapter_name: str = "default",
) -> dict[str, nn.Module]:
    modules: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is not None and lora_b is not None:
            if adapter_name in lora_a and adapter_name in lora_b:
                modules[name] = module
    if not modules:
        raise ValueError(f"no PEFT LoRA modules found for adapter {adapter_name!r}")
    return modules


def empty_adapter_state(
    model: nn.Module,
    *,
    adapter_name: str = "default",
) -> dict[str, CompactSVD]:
    state: dict[str, CompactSVD] = {}
    for name, module in named_peft_lora_modules(
        model, adapter_name=adapter_name
    ).items():
        a, b = _factor_weights(module, adapter_name)
        state[name] = zero_compact((b.shape[0], a.shape[1]))
    return state


@torch.no_grad()
def load_compact_adapter_state(
    model: nn.Module,
    state: Mapping[str, CompactSVD],
    *,
    active_rank: int,
    adapter_name: str = "default",
    seed: int = 0,
    initialize_free_directions: bool = True,
) -> None:
    """Load a compact server state into fixed-width PEFT LoRA factors."""

    modules = named_peft_lora_modules(model, adapter_name=adapter_name)
    if set(modules) != set(state):
        missing = sorted(set(modules) - set(state))
        extra = sorted(set(state) - set(modules))
        raise ValueError(f"adapter state mismatch; missing={missing}, extra={extra}")

    for offset, (name, module) in enumerate(modules.items()):
        a, b = _factor_weights(module, adapter_name)
        if not 0 < active_rank <= a.shape[0]:
            raise ValueError(
                f"active_rank={active_rank} is invalid for {name} with rank {a.shape[0]}"
            )
        compact = state[name]
        if compact.shape != (b.shape[0], a.shape[1]):
            raise ValueError(f"state shape mismatch for {name}")

        a.zero_()
        b.zero_()
        represented_rank = min(compact.rank, active_rank)
        if represented_rank:
            scaling = _scaling(module, adapter_name)
            u = compact.u[:, :represented_rank].to(device=b.device, dtype=b.dtype)
            s = compact.s[:represented_rank].to(device=b.device, dtype=b.dtype)
            v = compact.v[:, :represented_rank].to(device=a.device, dtype=a.dtype)
            b[:, :represented_rank].copy_(u * s.unsqueeze(0))
            a[:represented_rank, :].copy_(v.T / scaling)

        if initialize_free_directions and represented_rank < active_rank:
            generator = torch.Generator(device="cpu").manual_seed(seed + offset)
            free = torch.empty(
                (active_rank - represented_rank, a.shape[1]),
                dtype=torch.float32,
                device="cpu",
            )
            nn.init.kaiming_uniform_(free, a=math.sqrt(5), generator=generator)
            a[represented_rank:active_rank, :].copy_(
                free.to(device=a.device, dtype=a.dtype)
            )


def capture_factor_snapshot(
    model: nn.Module,
    *,
    adapter_name: str = "default",
) -> dict[str, FactorSnapshot]:
    snapshots: dict[str, FactorSnapshot] = {}
    for name, module in named_peft_lora_modules(
        model, adapter_name=adapter_name
    ).items():
        a, b = _factor_weights(module, adapter_name)
        snapshots[name] = FactorSnapshot(
            a.detach().to(device="cpu", dtype=torch.float32).clone(),
            b.detach().to(device="cpu", dtype=torch.float32).clone(),
            _scaling(module, adapter_name),
        )
    return snapshots


def compact_factor_innovations(
    before: Mapping[str, FactorSnapshot],
    after: Mapping[str, FactorSnapshot],
    *,
    active_rank: int,
    rank_rtol: float = 1e-5,
) -> dict[str, CompactSVD]:
    if set(before) != set(after):
        raise ValueError("before and after snapshots must contain the same modules")

    updates: dict[str, CompactSVD] = {}
    for name in before:
        initial = before[name]
        final = after[name]
        if initial.scaling != final.scaling:
            raise ValueError(f"LoRA scaling changed during training for {name}")
        b_initial = initial.b[:, :active_rank]
        a_initial = initial.a[:active_rank, :]
        b_final = final.b[:, :active_rank]
        a_final = final.a[:active_rank, :]
        lowrank = LowRankMatrix(
            torch.cat([b_final, b_initial], dim=1) * initial.scaling,
            torch.cat([a_final, -a_initial], dim=0),
        )
        updates[name] = compact_svd(lowrank, rtol=rank_rtol)
    return updates


def mask_inactive_rank_gradients(
    model: nn.Module,
    *,
    active_rank: int,
    adapter_name: str = "default",
) -> None:
    for module in named_peft_lora_modules(
        model, adapter_name=adapter_name
    ).values():
        a, b = _factor_weights(module, adapter_name)
        if a.grad is not None:
            a.grad[active_rank:, :].zero_()
        if b.grad is not None:
            b.grad[:, active_rank:].zero_()


def _factor_weights(
    module: nn.Module,
    adapter_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return module.lora_A[adapter_name].weight, module.lora_B[adapter_name].weight


def _scaling(module: nn.Module, adapter_name: str) -> float:
    scaling = module.scaling[adapter_name]
    if isinstance(scaling, torch.Tensor):
        return float(scaling.item())
    return float(scaling)
