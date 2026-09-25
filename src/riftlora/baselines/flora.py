"""FLoRA stacking primitives for the matched federated-LoRA runner.

The official FLoRA aggregation represents a weighted sum of client LoRA
products by concatenating the weighted ``B`` factors horizontally and the
``A`` factors vertically.  This keeps heterogeneous client ranks valid and
avoids averaging factors that do not share a gauge.

The Week 9 runner receives one asynchronous return at a time, so its runner
adapter stacks the current server product with the exact client innovation
before applying the configured server rank budget.  The pure client-state
function below is also exposed and tested against the paper equation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch

from riftlora.lowrank import CompactSVD, LowRankMatrix, compact_svd, recompress, weighted_sum
from riftlora.scale.peft_bridge import FactorSnapshot


def flora_stack_factor_states(
    client_states: Sequence[Mapping[str, FactorSnapshot]],
    *,
    weights: Sequence[float] | None = None,
    active_ranks: Sequence[int] | None = None,
    max_rank: int | None = None,
    rank_rtol: float = 1e-5,
) -> dict[str, CompactSVD]:
    """Aggregate complete client LoRA states with the FLoRA stack equation.

    For each layer, the returned product is exactly

    ``sum_i weights[i] * scaling_i * B_i @ A_i``

    before optional SVD truncation.  ``active_ranks`` permits heterogeneous
    client ranks; when omitted, each snapshot's full factor width is used.
    """

    if not client_states:
        raise ValueError("at least one client state is required")
    if weights is None:
        weights = [1.0 / len(client_states)] * len(client_states)
    if len(weights) != len(client_states):
        raise ValueError("weights and client states must have the same length")
    if active_ranks is not None and len(active_ranks) != len(client_states):
        raise ValueError("active_ranks and client states must have the same length")
    weight_values = [float(value) for value in weights]
    if any(not math.isfinite(value) or value < 0.0 for value in weight_values):
        raise ValueError("FLoRA weights must be finite and non-negative")
    if not math.isclose(sum(weight_values), 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("FLoRA weights must sum to one")
    keys = set(client_states[0])
    if any(set(state) != keys for state in client_states[1:]):
        raise ValueError("client factor keys must match")

    result: dict[str, CompactSVD] = {}
    for name in client_states[0]:
        left_parts: list[torch.Tensor] = []
        right_parts: list[torch.Tensor] = []
        for index, state in enumerate(client_states):
            snapshot = state[name]
            rank = (
                int(active_ranks[index])
                if active_ranks is not None
                else min(snapshot.b.shape[1], snapshot.a.shape[0])
            )
            if rank <= 0 or rank > min(snapshot.b.shape[1], snapshot.a.shape[0]):
                raise ValueError(f"invalid FLoRA client rank {rank} for {name}")
            # Weighting B only is the official stacking construction:
            # concat(w_i * scaling_i * B_i) and concat(A_i) yields the
            # weighted sum of client products without factor averaging.
            left_parts.append(snapshot.b[:, :rank] * (weight_values[index] * snapshot.scaling))
            right_parts.append(snapshot.a[:rank, :])
        stacked = LowRankMatrix(
            torch.cat(left_parts, dim=1),
            torch.cat(right_parts, dim=0),
        )
        result[name] = compact_svd(stacked, rtol=rank_rtol, max_rank=max_rank)
    return result


def flora_stack_aggregate_state(
    server: CompactSVD,
    innovation: CompactSVD,
    *,
    weight: float,
    max_rank: int,
    rank_rtol: float = 1e-5,
) -> CompactSVD:
    """Apply one async FLoRA stack: ``server + weight * innovation``.

    This is the exact stacking result before the shared server rank budget is
    imposed.  Keeping the recompression explicit makes rank loss observable
    instead of silently turning the method into factor-space averaging.
    """

    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("FLoRA aggregation weight must be in [0, 1]")
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    if server.shape != innovation.shape:
        raise ValueError("server and innovation shapes must match")
    if server.rank == 0:
        combined = innovation.as_lowrank().scaled(weight)
    elif innovation.rank == 0:
        combined = server.as_lowrank()
    else:
        combined = weighted_sum(
            [server.as_lowrank(), innovation.as_lowrank()], [1.0, weight]
        )
    return recompress(combined, max_rank=max_rank, rtol=rank_rtol)


def flora_stack_aggregate_states(
    server: Mapping[str, CompactSVD],
    innovations: Mapping[str, CompactSVD],
    *,
    weight: float,
    max_rank: int,
    rank_rtol: float = 1e-5,
) -> dict[str, CompactSVD]:
    """Map ``flora_stack_aggregate_state`` over a complete adapter state."""

    if set(server) != set(innovations):
        raise ValueError("server and innovation keys must match")
    return {
        name: flora_stack_aggregate_state(
            server[name], innovation, weight=weight, max_rank=max_rank,
            rank_rtol=rank_rtol,
        )
        for name, innovation in innovations.items()
    }
