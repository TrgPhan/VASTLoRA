import pandas as pd
import torch

from vastlora.diagnostics import (
    analyze_innovation_geometry,
    residual_budget_transport,
    persistent_temporal_projection,
    subspace_lattice_transport,
    transport_innovations,
    validate_diagnostic_dataframe,
)
from vastlora.lowrank import LowRankMatrix, compact_svd


def test_multilayer_geometry_uses_energy_weighted_compatibility() -> None:
    update_a = LowRankMatrix(torch.eye(3)[:, :1], torch.eye(3)[:1])
    update_b = LowRankMatrix(2 * torch.eye(3)[:, 1:2], torch.eye(3)[1:2])
    history = {
        "a": [compact_svd(update_a)],
        "b": [compact_svd(update_b)],
    }

    result = analyze_innovation_geometry(
        {"a": update_a, "b": update_b},
        history,
        reference_rank=1,
        history_size=2,
        reference_decay=0.0,
    )

    assert result.effective_rank == 2
    assert abs(result.fro_norm - 5**0.5) < 1e-6
    assert abs(result.rho_two_sided - 1.0) < 1e-6


def test_transport_keeps_projection_and_attenuates_residual() -> None:
    update = LowRankMatrix(torch.eye(2), torch.eye(2))
    transported = transport_innovations(
        {"layer": update},
        {"layer": torch.diag(torch.tensor([1.0, 0.0]))},
        freshness=0.25,
    )
    torch.testing.assert_close(transported["layer"], torch.diag(torch.tensor([1.0, 0.25])))


def test_residual_budget_transport_caps_residual_and_compensates_projection() -> None:
    update = LowRankMatrix(torch.eye(2), torch.eye(2))
    transported = residual_budget_transport(
        {"layer": update},
        {"layer": torch.diag(torch.tensor([1.0, 0.0]))},
        freshness=0.5,
        residual_budget=0.5,
        projection_scale_cap=2.0,
    )

    expected_projection_scale = 1.0 + 0.5 * (2**0.5 - 1.0)
    torch.testing.assert_close(
        transported["layer"],
        torch.diag(torch.tensor([expected_projection_scale, 0.25])),
    )


def test_subspace_lattice_transport_weights_four_orthogonal_blocks() -> None:
    dense = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    update = LowRankMatrix(dense, torch.eye(2))
    left = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    right = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    both = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

    transported = subspace_lattice_transport(
        {"layer": update},
        {"layer": left},
        {"layer": right},
        {"layer": both},
        single_side_weight=0.5,
        neither_weight=0.25,
    )

    torch.testing.assert_close(
        transported["layer"], torch.tensor([[1.0, 1.0], [1.5, 1.0]])
    )


def test_persistent_projection_keeps_cross_timescale_common_direction() -> None:
    common = torch.tensor([[1.0], [0.0], [0.0]])
    histories = []
    for index in range(8):
        transient = torch.eye(3)[:, 1 + index % 2 : 2 + index % 2]
        histories.append(
            compact_svd(
                LowRankMatrix(torch.cat([common, transient], dim=1), torch.eye(2, 3))
            )
        )
    update = LowRankMatrix(torch.eye(3), torch.eye(3))

    result = persistent_temporal_projection(
        {"layer": update},
        {"layer": histories},
        max_rank=2,
        short_history_size=4,
        long_history_size=8,
        overlap_threshold=0.9,
        reference_decay=0.0,
    )

    assert 1 <= result.left_ranks["layer"] <= 2
    assert 1 <= result.right_ranks["layer"] <= 2
    assert torch.isfinite(result.projected_updates["layer"]).all()


def test_schema_rejects_inconsistent_staleness() -> None:
    row = {
        "run_id": "r",
        "regime": "iid",
        "seed": 1,
        "update_id": 1,
        "client_id": "c00",
        "base_version": 1,
        "current_version": 3,
        "tau": 1,
        "rank": 2,
        "num_samples": 8,
        "virtual_latency": 1.0,
        "update_fro_norm": 1.0,
        "effective_rank": 1,
        "rho_left": 0.5,
        "rho_right": 0.5,
        "rho_two_sided": 0.25,
        "raw_update_utility": 0.1,
        "freshness_update_utility": 0.1,
        "vast_update_utility": 0.1,
        "current_loss": 0.5,
        "raw_candidate_loss": 0.4,
        "dataset_name": "x",
        "dataset_fingerprint_sha256": "abc",
        "partition_seed": 1,
        "partition_artifact": "bundle.pt#partitions",
        "client_indices_artifact": "bundle.pt#partitions.c00",
        "base_snapshot_id": "v1",
        "current_snapshot_id": "v3",
        "update_artifact_id": "bundle.pt#u1",
        "validation_split": "validation",
        "validation_indices_sha256": "def",
        "metric": "loss",
    }
    result = validate_diagnostic_dataframe(pd.DataFrame([row]), min_stale_updates=0)
    assert result["valid"] is False
    assert any("tau" in error for error in result["errors"])
