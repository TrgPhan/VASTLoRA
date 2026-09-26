"""Persistent-factor FedAvg and Freeze-A LoRA (Sun et al., ICLR 2024).

The multi-client mean is the homogeneous-rank paper operator. Immediate
server/client interpolation and prefix/zero-padding are explicit async and
heterogeneous-rank extensions, not the original synchronous protocol.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from riftlora.lowrank import LowRankMatrix, compact_svd
from riftlora.scale.peft_bridge import FactorSnapshot, named_peft_lora_modules, _scaling


FACTOR_IMPLEMENTATION = "persistent_factors_v2"
FACTOR_METHODS = frozenset({"fedavg_lora", "ffa_lora"})


def average_factor_states(
    states: Sequence[Mapping[str, FactorSnapshot]],
    *,
    weights: Sequence[float],
    freeze_a: bool = False,
) -> dict[str, FactorSnapshot]:
    """Weighted parameter mean; weights may be client sample counts.

    FFA requires the same A0 for every participant, including the server.
    The returned tensors are detached CPU clones, never aliases of inputs.
    """
    if not states or len(states) != len(weights):
        raise ValueError("states and weights must have equal nonzero lengths")
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must be finite, nonnegative and have positive sum")
    total = sum(weights)
    if not math.isfinite(total):
        raise ValueError("sum of weights must be finite")
    weights = [w / total for w in weights]
    first = states[0]
    if not first or any(set(state) != set(first) for state in states):
        raise ValueError("factor module keys must match and be nonempty")
    result = {}
    for name, reference in first.items():
        if not isinstance(reference, FactorSnapshot):
            raise TypeError("persistent FactorSnapshot state is required, not CompactSVD")
        a0 = reference.a.detach().cpu()
        b0 = reference.b.detach().cpu()
        if a0.ndim != 2 or b0.ndim != 2 or a0.shape[0] != b0.shape[1]:
            raise ValueError(f"invalid factor shapes for {name}")
        if not math.isfinite(reference.scaling) or reference.scaling <= 0:
            raise ValueError(f"invalid LoRA scaling for {name}")
        mean_a, mean_b = torch.zeros_like(a0), torch.zeros_like(b0)
        for state, weight in zip(states, weights):
            value = state[name]
            if not isinstance(value, FactorSnapshot):
                raise TypeError("persistent FactorSnapshot state is required, not CompactSVD")
            a, b = value.a.detach().cpu(), value.b.detach().cpu()
            if a.shape != a0.shape or b.shape != b0.shape or value.scaling != reference.scaling:
                raise ValueError(f"factor shape/scaling mismatch for {name}")
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError(f"nonfinite factors for {name}")
            if freeze_a and not torch.equal(a, a0):
                raise ValueError(f"FFA-LoRA A0 changed for {name}")
            if not freeze_a:
                mean_a.add_(a, alpha=weight)
            mean_b.add_(b, alpha=weight)
        result[name] = FactorSnapshot(a0.clone() if freeze_a else mean_a, mean_b, reference.scaling)
    return result


def fedavg_aggregate_factor_state(
    server: Mapping[str, FactorSnapshot],
    client_after: Mapping[str, FactorSnapshot],
    *,
    active_rank: int,
    weight: float,
    max_rank: int,
    freeze_a: bool = False,
) -> dict[str, FactorSnapshot]:
    """Async whole-state interpolation without SVD or factor reinitialization.

    For heterogeneous clients, inactive returned B columns are zero. FedAvg
    also zero-pads A rows; FFA keeps the shared full A0 (only B is aggregated).
    This missing-coordinate policy is an extension and must be reported.
    """
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be in [0, 1]")
    if not 0 < active_rank <= max_rank or set(server) != set(client_after):
        raise ValueError("invalid ranks or factor module keys")
    padded = {}
    for name, value in client_after.items():
        if value.a.shape[0] != max_rank or value.b.shape[1] != max_rank:
            raise ValueError("client snapshot must use the server factor width")
        a, b = value.a.clone(), value.b.clone()
        b[:, active_rank:] = 0
        if not freeze_a:
            a[active_rank:] = 0
        padded[name] = FactorSnapshot(a, b, value.scaling)
    return average_factor_states([server, padded], weights=[1 - weight, weight], freeze_a=freeze_a)


@torch.no_grad()
def load_factor_state(model, state: Mapping[str, FactorSnapshot], *, active_rank: int, freeze_a=False):
    """Load persistent factors directly; mask B, never mutate frozen A0."""
    modules = named_peft_lora_modules(model)
    if set(modules) != set(state):
        raise ValueError("factor module keys do not match model")
    for name, module in modules.items():
        a, b = module.lora_A["default"].weight, module.lora_B["default"].weight
        value = state[name]
        if not 0 < active_rank <= a.shape[0] or a.shape != value.a.shape or b.shape != value.b.shape:
            raise ValueError(f"invalid rank/factor shape for {name}")
        if _scaling(module, "default") != value.scaling:
            raise ValueError(f"LoRA scaling mismatch for {name}")
        source_a = value.a.to(device=a.device, dtype=a.dtype)
        if freeze_a:
            if a.requires_grad or not torch.equal(a, source_a):
                raise ValueError(f"FFA-LoRA requires immutable frozen A0 for {name}")
        else:
            a.copy_(source_a)
            a[active_rank:] = 0
        b.copy_(value.b.to(device=b.device, dtype=b.dtype))
        b[:, active_rank:] = 0


def factors_to_compact(state: Mapping[str, FactorSnapshot], *, rank_rtol: float):
    """Diagnostic/evaluation export only; never load it back into factor training."""
    return {
        name: compact_svd(LowRankMatrix(value.b * value.scaling, value.a),
                          rtol=rank_rtol, max_rank=value.a.shape[0])
        for name, value in state.items()
    }
