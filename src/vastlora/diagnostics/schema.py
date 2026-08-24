from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_DIAGNOSTIC_COLUMNS = (
    "run_id",
    "regime",
    "seed",
    "update_id",
    "client_id",
    "base_version",
    "current_version",
    "tau",
    "rank",
    "num_samples",
    "virtual_latency",
    "update_fro_norm",
    "effective_rank",
    "rho_left",
    "rho_right",
    "rho_two_sided",
    "raw_update_utility",
    "freshness_update_utility",
    "vast_update_utility",
    "current_loss",
    "raw_candidate_loss",
    "dataset_name",
    "dataset_fingerprint_sha256",
    "partition_seed",
    "partition_artifact",
    "client_indices_artifact",
    "base_snapshot_id",
    "current_snapshot_id",
    "update_artifact_id",
    "validation_split",
    "validation_indices_sha256",
    "metric",
)


def validate_diagnostic_dataframe(
    frame: pd.DataFrame,
    *,
    min_stale_updates: int = 100,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(set(REQUIRED_DIAGNOSTIC_COLUMNS) - set(frame.columns))
    if missing:
        errors.append(f"missing columns: {missing}")
        return {"valid": False, "errors": errors, "warnings": warnings}
    if frame.empty:
        errors.append("diagnostic dataframe is empty")
        return {"valid": False, "errors": errors, "warnings": warnings}

    stale_count = int((frame["tau"] > 0).sum())
    if stale_count < min_stale_updates:
        errors.append(
            f"only {stale_count} stale updates; expected at least {min_stale_updates}"
        )
    if not (frame["tau"] == frame["current_version"] - frame["base_version"]).all():
        errors.append("tau does not equal current_version - base_version")
    if frame[list(REQUIRED_DIAGNOSTIC_COLUMNS)].isnull().any().any():
        errors.append("required diagnostic fields contain null values")
    if not frame["rho_left"].between(0.0, 1.0).all():
        errors.append("rho_left is outside [0, 1]")
    if not frame["rho_right"].between(0.0, 1.0).all():
        errors.append("rho_right is outside [0, 1]")
    if not frame["rho_two_sided"].between(0.0, 1.0).all():
        errors.append("rho_two_sided is outside [0, 1]")
    if (frame["update_fro_norm"] <= 0).any():
        errors.append("one or more innovations have zero norm")
    if (frame["effective_rank"] <= 0).any():
        errors.append("one or more innovations have zero effective rank")

    observed = set(frame.loc[frame["tau"] > 0, "tau"].astype(int))
    target_coverage = {
        "tau_1_2": bool(observed & {1, 2}),
        "tau_3_7": bool(observed & {3, 4, 5, 6, 7}),
        "tau_8_plus": any(value >= 8 for value in observed),
    }
    if not all(target_coverage.values()):
        warnings.append(f"staleness coverage is incomplete: {target_coverage}")

    if artifact_root is not None:
        artifact_paths = {
            str(value).split("#", maxsplit=1)[0]
            for value in frame["update_artifact_id"].unique()
        }
        missing_artifacts = [
            value for value in artifact_paths if not (artifact_root / value).exists()
        ]
        if missing_artifacts:
            errors.append(f"missing replay artifacts: {missing_artifacts}")

    return {
        "valid": not errors,
        "rows": len(frame),
        "stale_updates": stale_count,
        "runs": int(frame["run_id"].nunique()),
        "staleness_coverage": target_coverage,
        "errors": errors,
        "warnings": warnings,
    }
