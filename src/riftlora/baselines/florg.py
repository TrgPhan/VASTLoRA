"""FLoRG single-matrix Gram aggregation for the 1.5B matched runner.

FLoRG replaces the two LoRA factors with a trainable ``A`` whose Gram matrix
``A.T @ A`` is the server state. Fixed semi-orthogonal L/R bases map that
state back to each target weight shape. The implementation keeps L/R frozen,
aggregates Gram matrices, then applies the closed-form Procrustes alignment
to the next decomposition.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FlorgState:
    a: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    scaling: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.left.shape[0], self.right.shape[1]

    @property
    def rank(self) -> int:
        return self.a.shape[0]


class _FlorgAdapter(nn.Module):
    def __init__(self, left: torch.Tensor, right: torch.Tensor, rank: int, scaling: float,
                 generator: torch.Generator) -> None:
        super().__init__()
        self.register_buffer("left", left)
        self.register_buffer("right", right)
        self.scaling = float(scaling)
        self.a = nn.Parameter(torch.randn((rank, left.shape[1]), generator=generator) * 0.01)

    def delta(self) -> torch.Tensor:
        return self.left @ (self.a.T @ self.a) @ self.right * self.scaling

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Algebraically L A^T A R, without allocating a k-by-k Gram each token.
        right_factor = self.a @ self.right
        left_factor = self.left @ self.a.T
        return F.linear(F.linear(inputs.to(self.a.dtype), right_factor), left_factor) * self.scaling


def _semi_orthogonal(rows: int, columns: int, generator: torch.Generator) -> torch.Tensor:
    if rows >= columns:
        q, _ = torch.linalg.qr(torch.randn((rows, columns), generator=generator), mode="reduced")
        return q
    q, _ = torch.linalg.qr(torch.randn((columns, rows), generator=generator), mode="reduced")
    return q.T


def _target_modules(model: nn.Module, suffixes: Sequence[str]) -> dict[str, nn.Module]:
    found = {name: module for name, module in model.named_modules()
             if name and any(name.endswith(suffix) for suffix in suffixes)
             and hasattr(module, "in_features") and hasattr(module, "out_features")}
    if not found:
        raise ValueError("no FLoRG target modules found")
    return found


def attach_florg_adapters(
    model: nn.Module,
    *,
    target_modules: Sequence[str],
    rank: int,
    alpha: float | None = None,
    seed: int = 0,
) -> dict[str, FlorgState]:
    """Attach trainable single-matrix FLoRG adapters to a base model."""
    if rank <= 0:
        raise ValueError("FLoRG rank must be positive")
    if hasattr(model, "florg_adapters"):
        raise ValueError("FLoRG adapters are already attached")
    if alpha is not None and (not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("FLoRG alpha must be positive and finite")
    modules = _target_modules(model, target_modules)
    adapters = nn.ModuleDict()
    names: dict[str, str] = {}
    states: dict[str, FlorgState] = {}
    for index, (name, module) in enumerate(modules.items()):
        out_features = int(module.out_features)
        in_features = int(module.in_features)
        k = min(out_features, in_features)
        generator = torch.Generator(device="cpu").manual_seed(seed + index * 7919)
        left = _semi_orthogonal(out_features, k, generator)
        right = _semi_orthogonal(in_features, k, generator).T
        key = f"layer_{index}"
        names[name] = key
        adapters[key] = _FlorgAdapter(left, right, rank, alpha / rank if alpha is not None else 1.0, generator)
        # Quantized target weights may be uint8; trainable adapters stay FP32.
        adapters[key].to(device=module.weight.device, dtype=torch.float32)
        states[name] = FlorgState(adapters[key].a.detach().cpu().clone(), left, right, adapters[key].scaling)

    model.add_module("florg_adapters", adapters)
    handles = []
    for name, module in modules.items():
        adapter = adapters[names[name]]

        def hook(_module, inputs, output, adapter=adapter):
            if not inputs or not torch.is_tensor(output):
                raise TypeError("FLoRG hook requires tensor input/output")
            return output + adapter(inputs[0]).to(output.dtype)

        handles.append(module.register_forward_hook(hook))
    model._florg_target_names = names
    model._florg_handles = handles
    return states


def _adapters(model: nn.Module) -> nn.ModuleDict:
    value = getattr(model, "florg_adapters", None)
    if not isinstance(value, nn.ModuleDict):
        raise ValueError("FLoRG adapters are not attached to this model")
    return value


def capture_florg_state(model: nn.Module) -> dict[str, torch.Tensor]:
    adapters = _adapters(model)
    names = getattr(model, "_florg_target_names")
    return {name: adapters[key].a.detach().cpu().clone() for name, key in names.items()}


def load_florg_state(model: nn.Module, state: Mapping[str, torch.Tensor], *, active_rank: int | None = None) -> None:
    adapters = _adapters(model)
    names = getattr(model, "_florg_target_names")
    if set(names) != set(state):
        raise ValueError("FLoRG state keys do not match model targets")
    for name, key in names.items():
        target = adapters[key].a
        value = state[name].to(device=target.device, dtype=target.dtype)
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"FLoRG state shape mismatch for {name}")
        if active_rank is not None:
            if not 0 < active_rank <= target.shape[0]:
                raise ValueError("invalid FLoRG active rank")
            if active_rank < target.shape[0]:
                # Heterogeneous-budget extension: top-r Gram approximation.
                _, s, vh = torch.linalg.svd(value, full_matrices=False)
                candidate = s[:active_rank, None] * vh[:active_rank]
                u, _, vt = torch.linalg.svd(value[:active_rank] @ candidate.T, full_matrices=False)
                value = torch.zeros_like(target)
                value[:active_rank] = u @ vt @ candidate
        with torch.no_grad():
            target.copy_(value)


def mask_florg_gradients(model: nn.Module, *, active_rank: int) -> None:
    """Apply the shared heterogeneous-rank budget to the latent FLoRG matrix."""
    if active_rank <= 0:
        raise ValueError("active_rank must be positive")
    for adapter in _adapters(model).values():
        if active_rank > adapter.a.shape[0]:
            raise ValueError("active_rank exceeds the FLoRG server rank")
        if adapter.a.grad is not None:
            adapter.a.grad[active_rank:, :].zero_()


def florg_aggregate_state(
    server: Mapping[str, torch.Tensor],
    client_after: Mapping[str, torch.Tensor],
    *,
    weight: float,
) -> dict[str, torch.Tensor]:
    """Aggregate Gram matrices and align the new factor to the old one."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("FLoRG aggregation weight must be in [0, 1]")
    if set(server) != set(client_after):
        raise ValueError("FLoRG state keys must match")
    result: dict[str, torch.Tensor] = {}
    for name, previous in server.items():
        client = client_after[name]
        if previous.ndim != 2 or client.shape != previous.shape:
            raise ValueError(f"invalid FLoRG matrix shape for {name}")
        # SVD of the stacked factors equals the eigendecomposition of the Gram.
        # Keep the complete numerical Gram rank before paper Eq. (11).
        stacked = torch.cat([previous * math.sqrt(1.0 - weight), client * math.sqrt(weight)], dim=0)
        _, values, vh = torch.linalg.svd(stacked, full_matrices=False)
        keep = values > values[0] * 1e-6
        candidate = values[keep, None] * vh[keep]
        if candidate.shape[0] == 0:
            result[name] = torch.zeros_like(previous)
            continue
        correlation = previous @ candidate.T
        u, _, vh = torch.linalg.svd(correlation, full_matrices=False)
        rotation = u @ vh
        # When Gram rank exceeds row budget this rectangular projection cannot
        # preserve the whole Gram. Do not claim exactness in that case.
        result[name] = rotation @ candidate
    return result
