"""FlexLoRA and FLoRIST operators, with an explicit immediate-async wrapper.

FlexLoRA: arXiv:2402.11505, aggregation and rank-specific SVD redistribution.
FLoRIST: arXiv:2506.09199v2, equations (1)-(4) and squared-energy truncation.
Both consume complete client products, not client innovations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch

from riftlora.lowrank import CompactSVD, LowRankMatrix, compact_svd
from riftlora.scale.peft_bridge import FactorSnapshot


def aggregate_spectral_factors(
    clients: Sequence[Mapping[str, FactorSnapshot]],
    *,
    weights: Sequence[float],
    method: str,
    max_rank: int | None = None,
    energy: float = 0.9,
    rank_rtol: float = 1e-5,
) -> tuple[dict[str, CompactSVD], dict[str, float]]:
    """Round-level weighted product aggregation; scaling is included once.

    FlexLoRA uses exact QR/core-SVD instead of a dense weight SVD. The product
    and rank truncation are equivalent; runtime is not a dense-paper benchmark.
    FLoRIST uses the paper's two factor SVDs and intermediate SVD explicitly.
    ``max_rank`` is an optional simulator budget, not the FLoRIST energy rule.
    """
    if method not in {"flexlora", "florist"}:
        raise ValueError("method must be flexlora or florist")
    if not clients or len(clients) != len(weights):
        raise ValueError("clients and weights must be nonempty and match")
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1., abs_tol=1e-6):
        raise ValueError("weights must be finite, nonnegative and sum to one")
    if not 0 < energy <= 1 or not math.isfinite(rank_rtol) or not 0 <= rank_rtol < 1:
        raise ValueError("invalid energy or rank tolerance")
    if max_rank is not None and (not isinstance(max_rank, int) or max_rank <= 0):
        raise ValueError("max_rank must be a positive integer")
    keys = set(clients[0])
    if not keys or any(set(client) != keys for client in clients):
        raise ValueError("client layer keys must be nonempty and match")
    result = {}
    retained, energy_ranks, capped = [], [], []
    for name in clients[0]:
        blocks_b, blocks_a = [], []
        first = clients[0][name]
        shape = (first.b.shape[0], first.a.shape[1])
        for client, weight in zip(clients, weights):
            item = client[name]
            if (item.a.ndim != 2 or item.b.ndim != 2 or item.b.shape[1] != item.a.shape[0]
                    or (item.b.shape[0], item.a.shape[1]) != shape
                    or not math.isfinite(item.scaling) or item.scaling <= 0):
                raise ValueError(f"invalid factors for {name}")
            blocks_b.append(item.b)
            blocks_a.append(item.a * (weight * item.scaling))
        b, a = torch.cat(blocks_b, dim=1), torch.cat(blocks_a, dim=0)
        if method == "flexlora":
            full = compact_svd(LowRankMatrix(b, a), rtol=rank_rtol)
        else:
            ub, sb, vbh = torch.linalg.svd(b, full_matrices=False)
            ua, sa, vah = torch.linalg.svd(a, full_matrices=False)
            middle = sb[:, None] * (vbh @ ua) * sa[None, :]
            up, s, vph = torch.linalg.svd(middle, full_matrices=False)
            full = CompactSVD(ub @ up, s, (vph @ vah).T)
        squares = full.s.square()
        total = squares.sum()
        if not full.rank or total.item() == 0:
            chosen = 0
        elif method == "florist":
            # searchsorted includes the boundary component; tau=1 keeps all.
            chosen = min(full.rank, int(torch.searchsorted(squares.cumsum(0), energy * total).item()) + 1)
        else:
            chosen = full.rank
        energy_ranks.append(chosen)
        keep = chosen if max_rank is None else min(chosen, max_rank)
        capped.append(float(keep < chosen))
        retained.append(float(squares[:keep].sum() / total) if total.item() else 1.)
        result[name] = CompactSVD(full.u[:, :keep], full.s[:keep], full.v[:, :keep])
    return result, {
        "spectral_mean_retained_energy": sum(retained) / len(retained),
        "spectral_rank_before_cap": float(sum(energy_ranks)),
        "spectral_rank_after_cap": float(sum(s.rank for s in result.values())),
        "spectral_cap_binding_fraction": sum(capped) / len(capped),
    }


def spectral_async_aggregate(
    server: Mapping[str, CompactSVD],
    client: Mapping[str, FactorSnapshot],
    *,
    active_rank: int,
    weight: float,
    method: str,
    max_rank: int,
    energy: float = 0.9,
    rank_rtol: float = 1e-5,
) -> tuple[dict[str, CompactSVD], dict[str, float]]:
    """Async adaptation: SVD((1-weight)*server + weight*returned client).

    The convex mix is the async protocol choice, not a synchronous paper round.
    Client dispatch uses the top client-rank singular components of its base.
    """
    if set(server) != set(client) or not isinstance(active_rank, int) or active_rank <= 0:
        raise ValueError("invalid layer keys or active_rank")
    server_factors, client_factors = {}, {}
    for name, state in server.items():
        item = client[name]
        if active_rank > min(item.b.shape[1], item.a.shape[0]):
            raise ValueError("active_rank exceeds available factors")
        server_factors[name] = FactorSnapshot(a=state.v.T, b=state.u * state.s, scaling=1.)
        client_factors[name] = FactorSnapshot(
            a=item.a[:active_rank], b=item.b[:, :active_rank], scaling=item.scaling,
        )
    return aggregate_spectral_factors(
        [server_factors, client_factors], weights=[1. - weight, weight],
        method=method, max_rank=max_rank, energy=energy, rank_rtol=rank_rtol,
    )
