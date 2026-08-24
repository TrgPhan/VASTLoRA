from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F

from vastlora.lowrank import LowRankMatrix, exact_lora_innovation


AdapterState = dict[str, torch.Tensor]


class DiagnosticLoRALinear(nn.Module):
    """Frozen linear layer with a server adapter and a trainable local innovation."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.register_buffer("server_delta", torch.zeros_like(base.weight))
        self.register_buffer("initial_lora_a", torch.empty(0, base.in_features))
        self.register_buffer("initial_lora_b", torch.empty(base.out_features, 0))
        self.lora_a = nn.Parameter(torch.empty(0, base.in_features))
        self.lora_b = nn.Parameter(torch.empty(base.out_features, 0))
        self.scaling = 1.0

    @property
    def rank(self) -> int:
        return self.lora_a.shape[0]

    def reset_local(self, rank: int, *, alpha: float, seed: int) -> None:
        if rank <= 0:
            raise ValueError("rank must be positive")
        generator = torch.Generator(device=self.server_delta.device)
        generator.manual_seed(seed)
        a = torch.empty(
            rank,
            self.base.in_features,
            device=self.server_delta.device,
            dtype=self.server_delta.dtype,
        )
        bound = math.sqrt(6.0 / self.base.in_features)
        a.uniform_(-bound, bound, generator=generator)
        b = torch.zeros(
            self.base.out_features,
            rank,
            device=self.server_delta.device,
            dtype=self.server_delta.dtype,
        )
        self._set_local_factors(a, b, alpha=alpha)

    def reset_local_from_server(self, rank: int, *, alpha: float, seed: int) -> None:
        """Initialize client factors from the dispatched server adapter geometry."""

        if rank <= 0:
            raise ValueError("rank must be positive")
        scaling = float(alpha) / rank
        u, s, vh = torch.linalg.svd(self.server_delta, full_matrices=False)
        threshold = 1e-6 * s[0] if s.numel() and s[0] > 0 else torch.inf
        active = min(rank, int(torch.sum(s > threshold).item()))
        a = torch.empty(
            rank,
            self.base.in_features,
            device=self.server_delta.device,
            dtype=self.server_delta.dtype,
        )
        b = torch.zeros(
            self.base.out_features,
            rank,
            device=self.server_delta.device,
            dtype=self.server_delta.dtype,
        )
        if active:
            roots = torch.sqrt(s[:active] / scaling)
            b[:, :active] = u[:, :active] * roots.unsqueeze(0)
            a[:active] = roots.unsqueeze(1) * vh[:active]
        if active < rank:
            generator = torch.Generator(device=self.server_delta.device)
            generator.manual_seed(seed)
            bound = math.sqrt(6.0 / self.base.in_features)
            a[active:].uniform_(-bound, bound, generator=generator)
        self._set_local_factors(a, b, alpha=alpha)

    def restore_initial(self) -> None:
        with torch.no_grad():
            self.lora_a.copy_(self.initial_lora_a)
            self.lora_b.copy_(self.initial_lora_b)

    def innovation(self) -> LowRankMatrix:
        exact = exact_lora_innovation(
            self.lora_b,
            self.lora_a,
            self.initial_lora_b,
            self.initial_lora_a,
        )
        return exact.scaled(self.scaling)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.base.weight + self.server_delta
        if self.rank:
            local_delta = self.lora_b @ self.lora_a
            initial_delta = self.initial_lora_b @ self.initial_lora_a
            weight = weight + self.scaling * (local_delta - initial_delta)
        return F.linear(inputs, weight, self.base.bias)

    def _set_local_factors(self, a: torch.Tensor, b: torch.Tensor, *, alpha: float) -> None:
        self.lora_a = nn.Parameter(a)
        self.lora_b = nn.Parameter(b)
        self.initial_lora_a = a.detach().clone()
        self.initial_lora_b = b.detach().clone()
        self.scaling = float(alpha) / a.shape[0]


def inject_diagnostic_lora(
    model: nn.Module,
    *,
    target_suffixes: Sequence[str],
) -> tuple[str, ...]:
    """Replace matching linear layers and return their stable module names."""

    if not target_suffixes:
        raise ValueError("target_suffixes must not be empty")
    model.requires_grad_(False)
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and any(
                full_name.endswith(suffix) for suffix in target_suffixes
            ):
                setattr(parent, child_name, DiagnosticLoRALinear(child))
                replaced.append(full_name)
            else:
                visit(child, full_name)

    visit(model, "")
    if not replaced:
        raise ValueError(f"no linear layers matched {tuple(target_suffixes)}")
    return tuple(replaced)


def named_lora_modules(model: nn.Module) -> dict[str, DiagnosticLoRALinear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, DiagnosticLoRALinear)
    }


def reset_local_adapters(model: nn.Module, rank: int, *, alpha: float, seed: int) -> None:
    for offset, module in enumerate(named_lora_modules(model).values()):
        module.reset_local(rank, alpha=alpha, seed=seed + offset)


def reset_local_adapters_from_server(
    model: nn.Module,
    rank: int,
    *,
    alpha: float,
    seed: int,
) -> None:
    for offset, module in enumerate(named_lora_modules(model).values()):
        module.reset_local_from_server(rank, alpha=alpha, seed=seed + offset)


def local_adapter_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for module in named_lora_modules(model).values():
        yield module.lora_a
        yield module.lora_b


def get_server_adapter_state(model: nn.Module, *, cpu: bool = False) -> AdapterState:
    state = {
        name: module.server_delta.detach().clone()
        for name, module in named_lora_modules(model).items()
    }
    if cpu:
        state = {name: value.cpu() for name, value in state.items()}
    return state


def set_server_adapter_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    modules = named_lora_modules(model)
    if set(modules) != set(state):
        raise ValueError("adapter state keys do not match injected LoRA modules")
    with torch.no_grad():
        for name, module in modules.items():
            module.server_delta.copy_(state[name].to(module.server_delta))


def get_local_innovations(model: nn.Module, *, cpu: bool = False) -> dict[str, LowRankMatrix]:
    updates = {
        name: module.innovation()
        for name, module in named_lora_modules(model).items()
    }
    if cpu:
        updates = {
            name: LowRankMatrix(update.left.cpu(), update.right.cpu())
            for name, update in updates.items()
        }
    return updates


def zero_local_adapters(model: nn.Module) -> None:
    """Restore dispatched factors so the local innovation is exactly zero."""

    for module in named_lora_modules(model).values():
        module.restore_initial()


def add_dense_innovation(
    state: Mapping[str, torch.Tensor],
    innovation: Mapping[str, LowRankMatrix | torch.Tensor],
    *,
    weight: float = 1.0,
) -> AdapterState:
    if set(state) != set(innovation):
        raise ValueError("state and innovation keys must match")
    result: AdapterState = {}
    for name, current in state.items():
        update = innovation[name]
        dense = update.dense() if isinstance(update, LowRankMatrix) else update
        result[name] = current + weight * dense.to(current)
    return result
