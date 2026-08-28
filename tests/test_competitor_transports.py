import torch

from riftlora.diagnostics.competitors import (
    COMPETITOR_SPECS,
    fedex_exact_diagnostic_state,
    fedrot_aggregate_diagnostic_state,
    fedsteer_cached_vector_projection,
    glora_cached_consensus_projection,
)
from riftlora.lora import DiagnosticFactorSnapshot
from riftlora.lowrank import LowRankMatrix, compact_svd


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


def test_fedex_exact_state_is_dense_difference() -> None:
    before = {"layer": torch.tensor([[1.0, 2.0]])}
    after = {"layer": torch.tensor([[2.5, 1.0]])}

    result = fedex_exact_diagnostic_state(after, before)

    torch.testing.assert_close(result["layer"], torch.tensor([[1.5, -1.0]]))


def test_competitor_registry_exposes_documented_labels() -> None:
    assert COMPETITOR_SPECS["fedex"].fidelity.startswith("faithful exact")
    assert COMPETITOR_SPECS["fedrot"].fidelity.startswith("matched FedRot")
    assert COMPETITOR_SPECS["alignfed_calibration"].fidelity.endswith("AlignFed")


def test_glora_cached_consensus_uses_left_projector() -> None:
    cached = compact_svd(
        LowRankMatrix(torch.tensor([[1.0], [0.0]]), torch.tensor([[1.0, 0.0]]))
    )
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
    cached = compact_svd(
        LowRankMatrix(torch.tensor([[1.0], [0.0]]), torch.tensor([[1.0, 0.0]]))
    )
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
