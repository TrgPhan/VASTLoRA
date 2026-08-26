import torch

from vastlora.diagnostics.competitors import (
    fedrot_aggregate_diagnostic_state,
    fedsteer_cached_vector_projection,
    glora_cached_consensus_projection,
)
from vastlora.lora import DiagnosticFactorSnapshot
from vastlora.lowrank import LowRankMatrix, compact_svd


def test_fedrot_alignment_preserves_equivalent_rotated_factorization() -> None:
    root_two = 2.0**0.5
    b = torch.tensor([[root_two, 0.0], [0.0, 1.0], [0.0, 0.0]])
    a = torch.tensor([[root_two, 0.0], [0.0, 1.0]])
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    server = {"layer": b @ a}
    client = {
        "layer": DiagnosticFactorSnapshot(
            a=rotation.T @ a,
            b=b @ rotation,
            scaling=1.0,
        )
    }

    result = fedrot_aggregate_diagnostic_state(
        server,
        client,
        active_rank=2,
        weight=0.5,
        max_rank=2,
        align_matrix="b",
    )

    torch.testing.assert_close(result["layer"], server["layer"], rtol=1e-5, atol=1e-5)


def test_glora_cached_consensus_uses_left_projector() -> None:
    cached = compact_svd(LowRankMatrix(torch.tensor([[1.0], [0.0]]), torch.tensor([[1.0, 0.0]])))
    current = LowRankMatrix(torch.eye(2), torch.eye(2))

    result = glora_cached_consensus_projection(
        {"layer": current},
        {"c0": {"layer": cached}},
        server_rank=1,
    )

    torch.testing.assert_close(
        result.updates["layer"], torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    )
    assert result.ranks["layer"] == 1


def test_fedsteer_cached_vector_projection_filters_orthogonal_component() -> None:
    cached = compact_svd(LowRankMatrix(torch.tensor([[1.0], [0.0]]), torch.tensor([[1.0, 0.0]])))
    current = LowRankMatrix(torch.eye(2), torch.eye(2))

    result = fedsteer_cached_vector_projection(
        {"layer": current},
        {"c0": {"layer": cached}},
        subspace_rank=1,
    )

    torch.testing.assert_close(
        result.updates["layer"], torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    )
    assert result.ranks["layer"] == 1
