from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn

from riftlora.lowrank import CompactSVD, LowRankMatrix, compact_svd
from riftlora.scale.coordinator import zero_compact


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


def fedrot_aggregate_factor_state(
    server: Mapping[str, CompactSVD],
    client_after: Mapping[str, FactorSnapshot],
    *,
    active_rank: int,
    weight: float,
    max_rank: int,
    rank_rtol: float = 1e-5,
) -> dict[str, CompactSVD]:
    """Aggregate final client LoRA factors after orthogonal Procrustes alignment.

    This is the matched-simulator FedRot-LoRA baseline: the client keeps the
    same represented update after rotation, but its latent rank coordinates are
    aligned to the current server factors before factor-space averaging.
    """

    if not 0.0 <= weight <= 1.0:
        raise ValueError("FedRot factor averaging weight must be in [0, 1]")
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    if set(server) != set(client_after):
        missing = sorted(set(server) - set(client_after))
        extra = sorted(set(client_after) - set(server))
        raise ValueError(f"state mismatch for FedRot aggregation; missing={missing}, extra={extra}")

    aggregated: dict[str, CompactSVD] = {}
    for name, compact in server.items():
        client = client_after[name]
        if not 0 < active_rank <= client.a.shape[0]:
            raise ValueError(f"active_rank={active_rank} is invalid for {name}")
        if max_rank > client.a.shape[0]:
            raise ValueError(f"max_rank={max_rank} exceeds available LoRA rank for {name}")

        server_b, server_a = _compact_to_lora_factors(
            compact,
            rank=max_rank,
            scaling=client.scaling,
        )
        client_b = _pad_columns(client.b[:, :active_rank], max_rank)
        client_a = _pad_rows(client.a[:active_rank, :], max_rank)

        if compact.rank:
            rotation = _orthogonal_procrustes(client_b, server_b)
            client_b = client_b @ rotation
            client_a = rotation.T @ client_a

        next_b = (1.0 - weight) * server_b + weight * client_b
        next_a = (1.0 - weight) * server_a + weight * client_a
        aggregated[name] = compact_svd(
            LowRankMatrix(next_b * client.scaling, next_a),
            rtol=rank_rtol,
            max_rank=max_rank,
        )
    return aggregated


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


def _compact_to_lora_factors(
    compact: CompactSVD,
    *,
    rank: int,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = compact.shape
    b = torch.zeros((rows, rank), dtype=compact.u.dtype, device=compact.u.device)
    a = torch.zeros((rank, columns), dtype=compact.u.dtype, device=compact.u.device)
    represented = min(compact.rank, rank)
    if represented:
        root = compact.s[:represented].sqrt()
        b[:, :represented] = compact.u[:, :represented] * root.unsqueeze(0)
        a[:represented, :] = root.unsqueeze(1) * compact.v[:, :represented].T / scaling
    return b, a


def _pad_columns(matrix: torch.Tensor, columns: int) -> torch.Tensor:
    if matrix.shape[1] > columns:
        raise ValueError("matrix already has more columns than requested")
    if matrix.shape[1] == columns:
        return matrix.to(dtype=torch.float32, device="cpu").clone()
    padding = torch.zeros(
        (matrix.shape[0], columns - matrix.shape[1]),
        dtype=torch.float32,
        device="cpu",
    )
    return torch.cat([matrix.to(dtype=torch.float32, device="cpu"), padding], dim=1)


def _pad_rows(matrix: torch.Tensor, rows: int) -> torch.Tensor:
    if matrix.shape[0] > rows:
        raise ValueError("matrix already has more rows than requested")
    if matrix.shape[0] == rows:
        return matrix.to(dtype=torch.float32, device="cpu").clone()
    padding = torch.zeros(
        (rows - matrix.shape[0], matrix.shape[1]),
        dtype=torch.float32,
        device="cpu",
    )
    return torch.cat([matrix.to(dtype=torch.float32, device="cpu"), padding], dim=0)


def _orthogonal_procrustes(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape != target.shape:
        raise ValueError("Procrustes source and target must have the same shape")
    if source.ndim != 2:
        raise ValueError("Procrustes inputs must be matrices")
    if float(torch.linalg.matrix_norm(source).item()) == 0.0:
        return torch.eye(source.shape[1], dtype=source.dtype, device=source.device)
    if float(torch.linalg.matrix_norm(target).item()) == 0.0:
        return torch.eye(source.shape[1], dtype=source.dtype, device=source.device)
    u, _, vh = torch.linalg.svd(source.T @ target, full_matrices=False)
    return u @ vh

