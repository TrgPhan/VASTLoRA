"""Unrestricted LoRA A/B fitting controls, not a published FL method."""

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import torch

from riftlora.lowrank import CompactSVD
from riftlora.scale.peft_bridge import (
    capture_factor_snapshot,
    compact_factor_innovations,
    load_compact_adapter_state,
    named_peft_lora_modules,
)
from riftlora.scale.objective import scale_compact_update


@dataclass(frozen=True)
class ServerAdaptationConfig:
    steps: int = 3
    learning_rate: float = 0.0002
    gradient_clip_norm: float = 1.0

    def validate(self) -> None:
        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, int)
            or self.steps < 1
        ):
            raise ValueError("server adaptation steps must be a positive integer")
        for value in (self.learning_rate, self.gradient_clip_norm):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "server adaptation rate/clip must be finite and positive"
                )


def fit_server_adapter(
    model: torch.nn.Module,
    current: Mapping[str, CompactSVD],
    initial: Mapping[str, CompactSVD],
    weighted_batches: Sequence,
    *,
    loss_fn: Callable,
    rank: int,
    server_weight: float,
    seed: int,
    rank_rtol: float,
    config: ServerAdaptationConfig,
) -> tuple[dict[str, CompactSVD], dict[str, float]]:
    """Fit ordinary A/B from initial; return (fitted-current)/server_weight.

    The gate subsequently multiplies by server_weight, so scale=1 recovers
    the fitted state rather than applying the aggregation weight twice.
    No incoming-subspace constraint or Core proximal penalty is used.
    """
    config.validate()
    if not math.isfinite(server_weight) or server_weight <= 0:
        raise ValueError("server_weight must be positive")
    batches = list(weighted_batches)
    if not batches or any(not math.isfinite(w) or w <= 0 for _, w in batches):
        raise ValueError("server adaptation needs positive finite batch weights")
    modules = named_peft_lora_modules(model)
    factors = [
        p
        for m in modules.values()
        for p in (m.lora_A["default"].weight, m.lora_B["default"].weight)
    ]
    saved = [(p, p.detach().clone(), p.grad) for p in factors]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(m, m.training) for m in model.modules()]
    total_weight = sum(w for _, w in batches)
    losses = []
    try:
        model.eval()  # Match deterministic Core fitting; no dropout benefit.
        for p, _ in flags:
            p.requires_grad_(False)
        for p in factors:
            p.requires_grad_(True)
            p.grad = None
        load_compact_adapter_state(
            model, current, active_rank=rank, initialize_free_directions=False
        )
        before = capture_factor_snapshot(model)
        load_compact_adapter_state(
            model, initial, active_rank=rank, seed=seed, initialize_free_directions=True
        )
        optimizer = torch.optim.Adam(factors, lr=config.learning_rate)
        for _ in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            value = 0.0
            for batch, weight in batches:
                loss = loss_fn(model, batch)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite server adaptation loss")
                (loss * (weight / total_weight)).backward()
                value += float(loss.detach()) * weight / total_weight
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in factors
            ):
                raise ValueError("nonfinite server adaptation gradient")
            torch.nn.utils.clip_grad_norm_(factors, config.gradient_clip_norm)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in factors):
                raise ValueError("nonfinite server adaptation parameter")
            losses.append(value)
        after = capture_factor_snapshot(model)
        updates = compact_factor_innovations(
            before, after, active_rank=rank, rank_rtol=rank_rtol
        )
        return (
            {
                name: scale_compact_update(u, 1.0 / server_weight)
                for name, u in updates.items()
            },
            {
                "server_fit_steps": float(config.steps),
                "server_fit_backward_calls": float(config.steps * len(batches)),
                "server_fit_weight_per_step": float(total_weight),
                "server_fit_parameters": float(sum(p.numel() for p in factors)),
                "server_fit_loss_initial": losses[0],
                "server_fit_loss_last_pre_step": losses[-1],
            },
        )
    finally:
        with torch.no_grad():
            for p, value, grad in saved:
                p.copy_(value)
                p.grad = grad
        for p, flag in flags:
            p.requires_grad_(flag)
        for m, mode in modes:
            m.training = mode
