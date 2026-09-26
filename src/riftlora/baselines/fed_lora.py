"""FedAvg exports and the legacy compact-state FedEx-LoRA adaptation.

The runner uses an immediate-arrival adaptation of the paper round: the
server interpolates the current state with one returned client state. FedAvg
averages LoRA factors. FedEx averages the factors and folds the exact
``mean(product) - product(mean)`` correction into a non-trainable residual
path. The residual path is kept separate from PEFT weights so quantized base
weights are not overwritten.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from riftlora.lowrank import CompactSVD, LowRankMatrix, compact_svd
from .factor_averaging import fedavg_aggregate_factor_state
from riftlora.scale.peft_bridge import (
    FactorSnapshot,
    _compact_to_lora_factors,
    _pad_columns,
    _pad_rows,
    named_peft_lora_modules,
)


def _validate_common(
    server: Mapping[str, CompactSVD],
    client_after: Mapping[str, FactorSnapshot],
    *,
    active_rank: int,
    weight: float,
    max_rank: int,
) -> None:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("aggregation weight must be in [0, 1]")
    if active_rank <= 0 or max_rank <= 0 or active_rank > max_rank:
        raise ValueError("invalid active/max rank")
    if set(server) != set(client_after):
        raise ValueError("server and client factor keys must match")


def _client_factors(
    compact: CompactSVD,
    client: FactorSnapshot,
    *,
    active_rank: int,
    max_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_rank > client.a.shape[0] or active_rank > client.a.shape[0]:
        raise ValueError("requested rank exceeds client LoRA rank")
    server_b, server_a = _compact_to_lora_factors(
        compact, rank=max_rank, scaling=client.scaling
    )
    client_b = _pad_columns(client.b[:, :active_rank], max_rank)
    client_a = _pad_rows(client.a[:active_rank, :], max_rank)
    return server_b, server_a, client_b, client_a


def fedex_aggregate_factor_state(
    server: Mapping[str, CompactSVD],
    client_after: Mapping[str, FactorSnapshot],
    *,
    current_residual: Mapping[str, torch.Tensor],
    stale_residual: Mapping[str, torch.Tensor],
    active_rank: int,
    weight: float,
    max_rank: int,
    rank_rtol: float = 1e-5,
) -> tuple[dict[str, CompactSVD], dict[str, torch.Tensor]]:
    """Aggregate factors and exact FedEx residuals for one returned client.

    For an immediate stale arrival, the effective target is

    ``(1-w)(W0 + R_current + B_s A_s)``
    ``+ w(W0 + R_stale + B_c A_c)``.

    The returned residual makes this target exactly equal to
    ``W0 + R_next + B_next A_next`` despite the factor averaging.
    """
    _validate_common(
        server, client_after, active_rank=active_rank, weight=weight, max_rank=max_rank
    )
    if set(current_residual) != set(server) or set(stale_residual) != set(server):
        raise ValueError("FedEx residual keys must match adapter state keys")
    result: dict[str, CompactSVD] = {}
    residuals: dict[str, torch.Tensor] = {}
    for name, compact in server.items():
        client = client_after[name]
        server_b, server_a, client_b, client_a = _client_factors(
            compact, client, active_rank=active_rank, max_rank=max_rank
        )
        next_b = (1.0 - weight) * server_b + weight * client_b
        next_a = (1.0 - weight) * server_a + weight * client_a
        scale = client.scaling
        server_product = scale * (server_b @ server_a)
        client_product = scale * (client_b @ client_a)
        next_product = scale * (next_b @ next_a)
        residual = (
            (1.0 - weight) * current_residual[name].to(torch.float32)
            + weight * stale_residual[name].to(torch.float32)
            + (1.0 - weight) * server_product.to(torch.float32)
            + weight * client_product.to(torch.float32)
            - next_product.to(torch.float32)
        )
        result[name] = compact_svd(
            LowRankMatrix(next_b * scale, next_a),
            rtol=rank_rtol,
            max_rank=max_rank,
        )
        residuals[name] = residual.cpu()
    return result, residuals


class FedExResidualController:
    """Adds the FedEx frozen-base residual without mutating quantized weights."""

    def __init__(self, model: nn.Module, *, adapter_name: str = "default") -> None:
        self.modules = named_peft_lora_modules(model, adapter_name=adapter_name)
        self._state: dict[str, torch.Tensor] = {}
        self._hooks = []
        for name, module in self.modules.items():
            self._state[name] = torch.zeros(
                (module.lora_B[adapter_name].weight.shape[0],
                 module.lora_A[adapter_name].weight.shape[1]),
                dtype=torch.float32,
            )
            self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def add_residual(module, inputs, output):
            if not inputs:
                return output
            hidden = inputs[0]
            residual = self._state[name]
            if residual.numel() == 0 or not torch.count_nonzero(residual):
                return output
            addition = F.linear(
                hidden,
                residual.to(device=hidden.device, dtype=hidden.dtype),
            )
            if not torch.is_tensor(output):
                raise TypeError("FedEx residual hook requires tensor module output")
            return output + addition

        return add_residual

    def state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self._state.items()}

    def set_state(self, state: Mapping[str, torch.Tensor]) -> None:
        if set(state) != set(self._state):
            raise ValueError("FedEx residual state keys do not match model modules")
        for name, value in state.items():
            expected = self._state[name].shape
            if tuple(value.shape) != tuple(expected):
                raise ValueError(f"FedEx residual shape mismatch for {name}")
            self._state[name] = value.detach().to(device="cpu", dtype=torch.float32).clone()

    def zero_state(self) -> dict[str, torch.Tensor]:
        return {name: torch.zeros_like(value) for name, value in self._state.items()}

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
