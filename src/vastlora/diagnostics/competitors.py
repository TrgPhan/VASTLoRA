from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from vastlora.lora import DiagnosticFactorSnapshot
from vastlora.lowrank import CompactSVD, LowRankMatrix, compact_svd


@dataclass(frozen=True)
class ProjectedCompetitorUpdate:
    updates: dict[str, torch.Tensor]
    ranks: dict[str, int]


def fedrot_aggregate_diagnostic_state(
    server: Mapping[str, torch.Tensor],
    client_after: Mapping[str, DiagnosticFactorSnapshot],
    *,
    active_rank: int,
    weight: float,
    max_rank: int,
    align_matrix: str = "b",
    rank_rtol: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """FedRot factor alignment followed by matched asynchronous interpolation.

    The Procrustes rotation is the FedRot operator. Convex interpolation with
    the current server factors is the explicit adaptation needed when one
    delayed client arrives at a time instead of a synchronous client cohort.
    """

    if not 0.0 <= weight <= 1.0:
        raise ValueError("FedRot interpolation weight must be in [0, 1]")
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    if align_matrix not in {"a", "b"}:
        raise ValueError("align_matrix must be 'a' or 'b'")
    if set(server) != set(client_after):
        raise ValueError("server and client factor keys must match")

    result: dict[str, torch.Tensor] = {}
    for name, current in server.items():
        client = client_after[name]
        if not 0 < active_rank <= client.a.shape[0]:
            raise ValueError(f"invalid active_rank={active_rank} for {name}")
        if max_rank < active_rank:
            raise ValueError("max_rank must cover every active client rank")

        server_b, server_a = _dense_to_lora_factors(
            current.to(dtype=torch.float32, device="cpu"),
            rank=max_rank,
            scaling=client.scaling,
            rank_rtol=rank_rtol,
        )
        client_b = _pad_columns(client.b[:, :active_rank], max_rank)
        client_a = _pad_rows(client.a[:active_rank], max_rank)
        reference = server_a if align_matrix == "a" else server_b
        source = client_a if align_matrix == "a" else client_b
        rotation = _orthogonal_procrustes(source, reference, align_matrix)
        aligned_b = client_b @ rotation
        aligned_a = rotation.T @ client_a

        next_b = (1.0 - weight) * server_b + weight * aligned_b
        next_a = (1.0 - weight) * server_a + weight * aligned_a
        dense = client.scaling * (next_b @ next_a)
        result[name] = dense.to(device=current.device, dtype=current.dtype)
    return result


def glora_cached_consensus_projection(
    innovations: Mapping[str, LowRankMatrix],
    client_cache: Mapping[str, Mapping[str, CompactSVD]],
    *,
    server_rank: int,
) -> ProjectedCompetitorUpdate:
    """Project an async update onto a GLoRA-style cached consensus column space."""

    if server_rank <= 0:
        raise ValueError("server_rank must be positive")
    updates: dict[str, torch.Tensor] = {}
    ranks: dict[str, int] = {}
    for name, innovation in innovations.items():
        bases = [
            state[name].u
            for state in client_cache.values()
            if name in state and state[name].rank > 0
        ]
        if not bases:
            updates[name] = innovation.dense()
            ranks[name] = 0
            continue
        scale = len(bases) ** -0.5
        projector_factor = torch.cat([basis * scale for basis in bases], dim=1)
        reference, _, _ = torch.linalg.svd(projector_factor, full_matrices=False)
        rank = min(server_rank, reference.shape[1])
        reference = reference[:, :rank]
        dense = innovation.dense()
        updates[name] = reference @ (reference.T @ dense)
        ranks[name] = rank
    return ProjectedCompetitorUpdate(updates=updates, ranks=ranks)


def fedsteer_cached_vector_projection(
    innovations: Mapping[str, LowRankMatrix],
    client_cache: Mapping[str, Mapping[str, CompactSVD]],
    *,
    subspace_rank: int,
    exclude_client: str | None = None,
) -> ProjectedCompetitorUpdate:
    """Project onto a dynamic cache of normalized client-update vectors.

    This is a matched LoRA/async operator inspired by FedSteer's dynamic
    gradient subspace. It is not the paper's inactive-client replay protocol.
    """

    if subspace_rank <= 0:
        raise ValueError("subspace_rank must be positive")
    updates: dict[str, torch.Tensor] = {}
    ranks: dict[str, int] = {}
    for name, innovation in innovations.items():
        vectors: list[torch.Tensor] = []
        for client_id, state in client_cache.items():
            if client_id == exclude_client or name not in state or state[name].rank == 0:
                continue
            vector = state[name].dense().reshape(-1)
            norm = torch.linalg.vector_norm(vector)
            if float(norm.item()) > 0.0:
                vectors.append(vector / norm)
        dense = innovation.dense()
        if not vectors:
            updates[name] = dense
            ranks[name] = 0
            continue
        cache_matrix = torch.stack(vectors, dim=1)
        basis, _, _ = torch.linalg.svd(cache_matrix, full_matrices=False)
        rank = min(subspace_rank, basis.shape[1])
        basis = basis[:, :rank]
        vector = dense.reshape(-1)
        updates[name] = (basis @ (basis.T @ vector)).reshape_as(dense)
        ranks[name] = rank
    return ProjectedCompetitorUpdate(updates=updates, ranks=ranks)


def dense_state_difference(
    after: Mapping[str, torch.Tensor],
    before: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if set(after) != set(before):
        raise ValueError("state keys must match")
    return {name: after[name] - before[name] for name in before}


def _dense_to_lora_factors(
    dense: torch.Tensor,
    *,
    rank: int,
    scaling: float,
    rank_rtol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    compact = compact_svd(
        LowRankMatrix(dense, torch.eye(dense.shape[1], dtype=dense.dtype)),
        rtol=rank_rtol,
        max_rank=rank,
    )
    b = torch.zeros((dense.shape[0], rank), dtype=dense.dtype)
    a = torch.zeros((rank, dense.shape[1]), dtype=dense.dtype)
    represented = compact.rank
    if represented:
        root = compact.s.sqrt()
        b[:, :represented] = compact.u * root.unsqueeze(0)
        a[:represented] = root.unsqueeze(1) * compact.v.T / scaling
    return b, a


def _orthogonal_procrustes(
    source: torch.Tensor,
    target: torch.Tensor,
    align_matrix: str,
) -> torch.Tensor:
    if source.shape != target.shape:
        raise ValueError("Procrustes source and target must have equal shapes")
    if float(torch.linalg.matrix_norm(source).item()) == 0.0:
        return torch.eye(source.shape[0 if align_matrix == "a" else 1])
    if float(torch.linalg.matrix_norm(target).item()) == 0.0:
        return torch.eye(source.shape[0 if align_matrix == "a" else 1])
    correlation = source @ target.T if align_matrix == "a" else source.T @ target
    u, _, vh = torch.linalg.svd(correlation, full_matrices=False)
    return u @ vh


def _pad_columns(matrix: torch.Tensor, columns: int) -> torch.Tensor:
    value = matrix.detach().to(dtype=torch.float32, device="cpu")
    if value.shape[1] > columns:
        raise ValueError("matrix has more columns than requested")
    if value.shape[1] == columns:
        return value.clone()
    return torch.cat(
        [value, torch.zeros((value.shape[0], columns - value.shape[1]))], dim=1
    )


def _pad_rows(matrix: torch.Tensor, rows: int) -> torch.Tensor:
    value = matrix.detach().to(dtype=torch.float32, device="cpu")
    if value.shape[0] > rows:
        raise ValueError("matrix has more rows than requested")
    if value.shape[0] == rows:
        return value.clone()
    return torch.cat(
        [value, torch.zeros((rows - value.shape[0], value.shape[1]))], dim=0
    )
