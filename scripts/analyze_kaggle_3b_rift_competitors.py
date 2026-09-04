from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

from riftlora.diagnostics import competitor_fidelity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs/week8_rift_classification_matrix.json"


FIDELITY = {
    "raw": "matched async baseline",
    "fedex": competitor_fidelity("fedex"),
    "freshness": "faithful scalar staleness weighting baseline",
    "fedrot": competitor_fidelity("fedrot"),
    "vast": "legacy VAST residual transport",
    "mtip": "projection-only transport baseline",
    "mtip_adaptive": "adaptive projection transport baseline",
    "spectral_filter": "Spectral-Surgery-style gradient component filter; not official paper implementation",
    "alignfed_calibration": competitor_fidelity("alignfed_calibration"),
    "rift": "proposed rank-wise objective filter plus paired gate",
}

SKIPPED = {
    "AdaLoRA": "PEFT rank-allocation/training method, not an async server aggregation rule.",
    "Spectral Surgery official": "No official public code found; paper is post-hoc adapter refinement, so the notebook uses a clearly labeled matched ablation only.",
    "FLoRG": "Requires single-matrix/Gram LoRA reparameterization, incompatible with the current PEFT factor simulator without changing client training.",
    "GLoRA official": "No official public code found in the search; current runner lacks synchronous cohort consensus protocol.",
    "FedSteer official": "Official code exists, but its inactive-client replay protocol differs from delayed returned LoRA updates in this simulator.",
    "AlignFed full": "Requires version groups, semantic transform, and fairness weighting; current runner includes only whole-update calibration control.",
    "OrthoFL full": "Requires maintaining separate global/client model progress and calibrated merge state, not a simple LoRA aggregation operator.",
}
DEFAULT_WEEK8_GATE = {
    "minimum_paired_seeds": 6,
    "minimum_acceptance_rate": 0.30,
    "minimum_client_return_coverage": 1.0,
    "minimum_late_events": 6,
    "hard_regimes": ["noniid_high_staleness"],
    "accuracy_noninferiority_margin_pp": -0.5,
    "nll_noninferiority_margin": -0.005,
    "requires_positive_late_harm_reduction": True,
    "requires_positive_cumulative_late_harm_reduction": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Kaggle 3B RIFT competitor runs")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="rift")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_results(args.input_dir)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    gate = resolve_week8_gate(matrix)
    completeness_errors = validate_matrix_completeness(frame, matrix)
    completeness_errors.extend(validate_result_provenance(frame, matrix))
    try:
        validate_seed_alignment(frame, target=args.target)
    except ValueError as exc:
        completeness_errors.append(str(exc))
    summary = summarize(frame)
    paired = paired_against(frame, args.target)
    verdict = week8_verdict(
        paired,
        gate=gate,
        completeness_errors=completeness_errors,
    )
    report = render_report(summary, paired, verdict, target=args.target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "runs.csv", index=False)
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_target.csv", index=False)
    pd.DataFrame(verdict["hard_slice_checks"]).to_csv(
        args.output_dir / "week8_hard_slice_checks.csv", index=False
    )
    (args.output_dir / "week8_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "week8_verdict.md").write_text(
        render_report(summary, paired, verdict, target=args.target), encoding="utf-8"
    )
    (args.output_dir / "competitor_report.md").write_text(report, encoding="utf-8")
    (args.output_dir / "skipped_competitors.json").write_text(
        json.dumps(SKIPPED, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(report)


def validate_seed_alignment(
    frame: pd.DataFrame,
    *,
    target: str = "rift",
) -> dict[tuple[str, str], list[int]]:
    aligned: dict[tuple[str, str], list[int]] = {}
    for (task, regime), group in frame.groupby(["task", "regime"]):
        target_seeds = sorted(group[group["method"] == target]["seed"].unique().tolist())
        if not target_seeds:
            raise ValueError(f"missing target method {target!r} for {task}/{regime}")
        for method, method_group in group.groupby("method"):
            seeds = sorted(method_group["seed"].unique().tolist())
            if seeds != target_seeds:
                raise ValueError(
                    f"seed mismatch for {method} in {task}/{regime}: "
                    f"expected={target_seeds}, actual={seeds}"
                )
        aligned[(task, regime)] = target_seeds
    return aligned


def resolve_week8_gate(matrix: dict[str, Any]) -> dict[str, Any]:
    gate = dict(DEFAULT_WEEK8_GATE)
    gate.update(matrix.get("gates", {}))
    gate["hard_regimes"] = [str(value) for value in gate["hard_regimes"]]
    return gate


def matrix_fingerprint(matrix: dict[str, Any]) -> str:
    payload = json.dumps(
        matrix,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_result_provenance(
    frame: pd.DataFrame,
    matrix: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    key_columns = ["task", "regime", "method", "seed"]
    duplicates = frame[frame.duplicated(key_columns, keep=False)]
    if not duplicates.empty:
        duplicate_keys = duplicates[key_columns].drop_duplicates().to_dict("records")
        errors.append(f"duplicate run keys: {duplicate_keys}")

    required_schema = int(matrix.get("required_schema_version", 3))
    old_schema = frame[frame["schema_version"] < required_schema]
    if not old_schema.empty:
        errors.append(
            f"schema_version must be >= {required_schema}; "
            f"found={sorted(old_schema['schema_version'].unique().tolist())}"
        )

    expected_matrix_sha = matrix_fingerprint(matrix)
    if bool(frame["matrix_sha256"].isna().any()):
        errors.append("one or more runs are missing matrix_sha256")
    actual_matrix_shas = set(frame["matrix_sha256"].dropna().astype(str))
    if actual_matrix_shas != {expected_matrix_sha}:
        errors.append(
            "matrix fingerprint mismatch: "
            f"expected={expected_matrix_sha}, actual={sorted(actual_matrix_shas)}"
        )

    missing_config_hash = frame["config_fingerprint"].isna() | (
        frame["config_fingerprint"].astype(str).str.len() == 0
    )
    if bool(missing_config_hash.any()):
        errors.append("one or more runs are missing config_fingerprint")

    commits = set(frame["git_commit"].dropna().astype(str))
    if bool(frame["git_commit"].isna().any()) or len(commits) != 1 or "unknown" in commits:
        errors.append(f"runs must share one known git commit; actual={sorted(commits)}")

    for (task, regime), group in frame.groupby(["task", "regime"]):
        hashes = set(group["config_fingerprint"].dropna().astype(str))
        if len(hashes) != 1:
            errors.append(
                f"config fingerprint mismatch within {task}/{regime}: "
                f"actual={sorted(hashes)}"
            )

    expected_returns = int(matrix.get("experiment", {}).get("collected_returns", 0))
    if expected_returns:
        if bool(frame["configured_collected_returns"].isna().any()):
            errors.append("one or more runs are missing configured_collected_returns")
        actual_returns = set(frame["configured_collected_returns"].dropna().astype(int))
        if actual_returns != {expected_returns}:
            errors.append(
                "collected_returns mismatch: "
                f"expected={expected_returns}, actual={sorted(actual_returns)}"
            )
        if bool(frame["measured_event_count"].isna().any()):
            errors.append("one or more runs are missing measured_event_count")
        measured_counts = set(frame["measured_event_count"].dropna().astype(int))
        if measured_counts != {expected_returns}:
            errors.append(
                "measured event count mismatch: "
                f"expected={expected_returns}, actual={sorted(measured_counts)}"
            )
    return errors


def validate_matrix_completeness(
    frame: pd.DataFrame,
    matrix: dict[str, Any],
) -> list[str]:
    expected_tasks = {
        str(task.get("run_name", task["name"])) for task in matrix["tasks"]
    }
    expected_regimes = {str(regime["name"]) for regime in matrix["regimes"]}
    expected_methods = {str(method) for method in matrix["methods"]}
    expected_seeds = {int(seed) for seed in matrix["seeds"]}
    errors: list[str] = []
    for task in sorted(expected_tasks):
        for regime in sorted(expected_regimes):
            group = frame[(frame["task"] == task) & (frame["regime"] == regime)]
            if group.empty:
                errors.append(f"missing task/regime: {task}/{regime}")
                continue
            actual_methods = set(group["method"].astype(str))
            for method in sorted(expected_methods - actual_methods):
                errors.append(f"missing method: {task}/{regime}/{method}")
            for method in sorted(expected_methods & actual_methods):
                actual_seeds = set(
                    group[group["method"] == method]["seed"].astype(int)
                )
                if actual_seeds != expected_seeds:
                    errors.append(
                        f"seed set mismatch: {task}/{regime}/{method}: "
                        f"expected={sorted(expected_seeds)}, "
                        f"actual={sorted(actual_seeds)}"
                    )
    return errors


def load_results(input_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*_seed*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        config = payload.get("config", {})
        experiment = config.get("experiment", {})
        provenance = payload.get("provenance", config.get("provenance", {}))
        records.append(
            {
                "source_path": str(path),
                "schema_version": int(payload.get("schema_version", 0)),
                "method": payload["method"],
                "variant": payload.get("variant", payload["method"]),
                "seed": int(payload["seed"]),
                "model": payload["model"],
                "task": payload.get("task", payload.get("config", {}).get("dataset", {}).get("subset", "unknown")),
                "regime": payload.get("regime", payload.get("config", {}).get("experiment", {}).get("regime_name", "default")),
                "git_commit": payload["git_commit"],
                "matrix_sha256": provenance.get("matrix_sha256"),
                "config_fingerprint": payload.get("config_fingerprint"),
                "configured_collected_returns": experiment.get("collected_returns"),
                **metrics,
                # Schema v2 supplies label NLL explicitly.  Fall back to the
                # old total NLL so historical pilot outputs remain readable.
                "classification_nll": metrics.get(
                    "final_label_nll", metrics["final_nll"]
                ),
            }
        )
    if not records:
        raise ValueError(f"no result.json files found under {input_dir}")
    return pd.DataFrame(records).sort_values(["task", "regime", "method", "seed"]).reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, regime, method), group in frame.groupby(["task", "regime", "method"]):
        best_accuracy_index = group["final_accuracy"].astype(float).idxmax()
        best_nll_index = group["classification_nll"].astype(float).idxmin()
        rows.append(
            {
                "task": task,
                "regime": regime,
                "method": method,
                "fidelity": FIDELITY.get(method, "unknown"),
                "seeds": int(group["seed"].nunique()),
                "final_accuracy_mean": float(group["final_accuracy"].mean()),
                "final_accuracy_std": float(group["final_accuracy"].std(ddof=1))
                if len(group) > 1
                else 0.0,
                "best_final_accuracy": float(group.loc[best_accuracy_index, "final_accuracy"]),
                "best_final_accuracy_seed": int(group.loc[best_accuracy_index, "seed"]),
                # Classification comparisons use label NLL.  Sequence NLL
                # includes EOS and is retained as a diagnostic only.
                "final_nll_mean": float(group["classification_nll"].mean()),
                "sequence_nll_mean": float(group["final_nll"].mean()),
                "best_final_nll": float(group.loc[best_nll_index, "classification_nll"]),
                "best_final_nll_seed": int(group.loc[best_nll_index, "seed"]),
                "final_balanced_accuracy_mean": float(
                    group.get("final_balanced_accuracy", group["final_accuracy"]).mean()
                ),
                "final_binary_nll_mean": float(group["final_binary_nll"].mean()),
                "final_brier_mean": float(group["final_brier"].mean()),
                "harmful_update_rate": float(group.get("harmful_update_rate", pd.Series([0.0])).mean()),
                "late_harmful_update_rate": float(
                    group.get("late_harmful_update_rate", pd.Series([0.0])).mean()
                ),
                "extreme_harmful_update_rate": float(
                    group.get("extreme_harmful_update_rate", pd.Series([0.0])).mean()
                ),
                "monitor_loss_change": float(
                    group.get("monitor_loss_change", pd.Series([0.0])).mean()
                ),
                "acceptance_rate": float(
                    group.get("acceptance_rate", pd.Series([float("nan")])).mean()
                ),
                "late_event_count_mean": float(
                    group.get("late_event_count", pd.Series([float("nan")])).mean()
                ),
                "extreme_event_count_mean": float(
                    group.get("extreme_event_count", pd.Series([float("nan")])).mean()
                ),
                "client_return_coverage_mean": float(
                    group.get("client_return_coverage", pd.Series([float("nan")])).mean()
                ),
                "min_client_returns_min": float(
                    group.get("min_client_returns", pd.Series([float("nan")])).min()
                ),
                "cumulative_late_harm_mean": float(
                    group.get("cumulative_late_harm", pd.Series([float("nan")])).mean()
                ),
                "normalized_cumulative_late_harm_mean": float(
                    group.get(
                        "normalized_cumulative_late_harm",
                        pd.Series([float("nan")]),
                    ).mean()
                ),
                "cumulative_extreme_harm_mean": float(
                    group.get("cumulative_extreme_harm", pd.Series([float("nan")])).mean()
                ),
                "worst_step_loss_increase_mean": float(
                    group.get("worst_step_loss_increase", pd.Series([float("nan")])).mean()
                ),
                "utility_per_accepted_update_mean": float(
                    group.get(
                        "utility_per_accepted_update", pd.Series([float("nan")])
                    ).mean()
                ),
                "utility_per_returned_update_mean": float(
                    group.get(
                        "utility_per_returned_update", pd.Series([float("nan")])
                    ).mean()
                ),
                "rank_filtered_route_rate_mean": float(
                    group.get("rank_filtered_route_rate", pd.Series([float("nan")])).mean()
                ),
                "freshness_fallback_route_rate_mean": float(
                    group.get(
                        "freshness_fallback_route_rate", pd.Series([float("nan")])
                    ).mean()
                ),
                "rejection_rate_mean": float(
                    group.get("rejection_rate", pd.Series([float("nan")])).mean()
                ),
                "mean_retained_fraction": float(
                    group.get("mean_retained_fraction", pd.Series([float("nan")])).mean()
                ),
                "runtime_seconds_mean": float(group["runtime_seconds"].mean()),
                "peak_cuda_memory_gib_mean": float(group["peak_cuda_memory_gib"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["task", "regime", "final_accuracy_mean", "final_nll_mean"],
        ascending=[True, True, False, True],
    )


def paired_against(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for (task, regime, method), group in frame.groupby(["task", "regime", "method"]):
        if method == target:
            continue
        joined = group.set_index("seed").join(
            frame[
                (frame["method"] == target)
                & (frame["task"] == task)
                & (frame["regime"] == regime)
            ].set_index("seed"),
            lsuffix="_candidate",
            rsuffix="_target",
            how="inner",
        )
        if joined.empty:
            continue
        accuracy_delta = 100.0 * (
            joined["final_accuracy_target"] - joined["final_accuracy_candidate"]
        )
        nll_delta = (
            joined["classification_nll_candidate"]
            - joined["classification_nll_target"]
        )
        binary_target = pd.to_numeric(
            joined["final_binary_nll_target"], errors="coerce"
        )
        binary_candidate = pd.to_numeric(
            joined["final_binary_nll_candidate"], errors="coerce"
        )
        binary_nll_delta = binary_candidate - binary_target
        harmful_delta = (
            joined["harmful_update_rate_candidate"]
            - joined["harmful_update_rate_target"]
            if "harmful_update_rate_candidate" in joined
            else pd.Series([0.0])
        )
        late_harmful_delta = (
            joined["late_harmful_update_rate_candidate"]
            - joined["late_harmful_update_rate_target"]
            if "late_harmful_update_rate_candidate" in joined
            else pd.Series([0.0])
        )
        balanced_accuracy_delta = 100.0 * (
            joined.get("final_balanced_accuracy_target", joined["final_accuracy_target"])
            - joined.get(
                "final_balanced_accuracy_candidate",
                joined["final_accuracy_candidate"],
            )
        )
        cumulative_late_harm_delta = (
            joined.get("cumulative_late_harm_candidate", pd.Series(0.0, index=joined.index))
            - joined.get("cumulative_late_harm_target", pd.Series(0.0, index=joined.index))
        )
        normalized_late_harm_delta = (
            joined.get(
                "normalized_cumulative_late_harm_candidate",
                pd.Series(0.0, index=joined.index),
            )
            - joined.get(
                "normalized_cumulative_late_harm_target",
                pd.Series(0.0, index=joined.index),
            )
        )
        worst_step_delta = (
            joined.get("worst_step_loss_increase_candidate", pd.Series(0.0, index=joined.index))
            - joined.get("worst_step_loss_increase_target", pd.Series(0.0, index=joined.index))
        )
        acc_mean, acc_low, acc_high = mean_ci95(accuracy_delta)
        balanced_mean, balanced_low, balanced_high = mean_ci95(
            balanced_accuracy_delta
        )
        nll_mean, nll_low, nll_high = mean_ci95(nll_delta)
        binary_mean, binary_low, binary_high = mean_ci95(binary_nll_delta)
        harmful_mean, harmful_low, harmful_high = mean_ci95(harmful_delta)
        late_mean, late_low, late_high = mean_ci95(late_harmful_delta)
        cumulative_mean, cumulative_low, cumulative_high = mean_ci95(
            cumulative_late_harm_delta
        )
        normalized_mean, normalized_low, normalized_high = mean_ci95(
            normalized_late_harm_delta
        )
        worst_mean, worst_low, worst_high = mean_ci95(worst_step_delta)
        target_acceptance_rate = (
            float(joined["acceptance_rate_target"].mean())
            if "acceptance_rate_target" in joined
            else float("nan")
        )
        rows.append(
            {
                "task": task,
                "regime": regime,
                "method": method,
                "paired_seeds": int(joined.shape[0]),
                "target_accuracy_gain_pp": acc_mean,
                "target_accuracy_gain_ci95_low": acc_low,
                "target_accuracy_gain_ci95_high": acc_high,
                "target_accuracy_wins": int(
                    (
                        joined["final_accuracy_target"]
                        > joined["final_accuracy_candidate"]
                    ).sum()
                ),
                "target_balanced_accuracy_gain_pp": balanced_mean,
                "target_balanced_accuracy_gain_ci95_low": balanced_low,
                "target_balanced_accuracy_gain_ci95_high": balanced_high,
                "target_nll_reduction": nll_mean,
                "target_nll_reduction_ci95_low": nll_low,
                "target_nll_reduction_ci95_high": nll_high,
                "target_nll_wins": int(
                    (
                        joined["classification_nll_target"]
                        < joined["classification_nll_candidate"]
                    ).sum()
                ),
                "target_binary_nll_reduction": binary_mean,
                "target_binary_nll_reduction_ci95_low": binary_low,
                "target_binary_nll_reduction_ci95_high": binary_high,
                "target_harmful_reduction": harmful_mean,
                "target_harmful_reduction_ci95_low": harmful_low,
                "target_harmful_reduction_ci95_high": harmful_high,
                "target_late_harmful_reduction": late_mean,
                "target_late_harmful_reduction_ci95_low": late_low,
                "target_late_harmful_reduction_ci95_high": late_high,
                "target_cumulative_late_harm_reduction": cumulative_mean,
                "target_cumulative_late_harm_reduction_ci95_low": cumulative_low,
                "target_cumulative_late_harm_reduction_ci95_high": cumulative_high,
                "target_normalized_late_harm_reduction": normalized_mean,
                "target_normalized_late_harm_reduction_ci95_low": normalized_low,
                "target_normalized_late_harm_reduction_ci95_high": normalized_high,
                "target_worst_step_harm_reduction": worst_mean,
                "target_worst_step_harm_reduction_ci95_low": worst_low,
                "target_worst_step_harm_reduction_ci95_high": worst_high,
                "target_acceptance_rate": target_acceptance_rate,
                "target_client_return_coverage": (
                    float(joined["client_return_coverage_target"].mean())
                    if "client_return_coverage_target" in joined
                    else float("nan")
                ),
                "target_min_client_returns": (
                    float(joined["min_client_returns_target"].min())
                    if "min_client_returns_target" in joined
                    else float("nan")
                ),
                "target_late_event_count": (
                    float(joined["late_event_count_target"].mean())
                    if "late_event_count_target" in joined
                    else float("nan")
                ),
                "target_cumulative_late_harm": (
                    float(joined["cumulative_late_harm_target"].mean())
                    if "cumulative_late_harm_target" in joined
                    else float("nan")
                ),
                "target_worst_step_loss_increase": (
                    float(joined["worst_step_loss_increase_target"].mean())
                    if "worst_step_loss_increase_target" in joined
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["task", "regime", "target_accuracy_gain_pp", "target_nll_reduction"],
        ascending=[True, True, False, False],
    )


def week8_verdict(
    paired: pd.DataFrame,
    *,
    gate: dict[str, Any] | None = None,
    completeness_errors: list[str] | None = None,
) -> dict[str, Any]:
    gate = dict(DEFAULT_WEEK8_GATE if gate is None else gate)
    completeness_errors = list(completeness_errors or [])
    hard = paired[paired["regime"].isin(gate["hard_regimes"])].copy()
    checks: list[dict[str, Any]] = []
    if hard.empty:
        return {
            "status": "INCONCLUSIVE",
            "reason": "No hard-slice paired rows are available.",
            "gate": gate,
            "hard_slice_checks": checks,
            "completeness_errors": completeness_errors,
        }

    for _, row in hard.iterrows():
        seed_ok = int(row["paired_seeds"]) >= int(gate["minimum_paired_seeds"])
        accuracy_ok = (
            float(row["target_accuracy_gain_ci95_low"])
            >= float(gate["accuracy_noninferiority_margin_pp"])
        )
        nll_ok = (
            float(row["target_nll_reduction_ci95_low"])
            >= float(gate["nll_noninferiority_margin"])
        )
        acceptance_rate = float(row.get("target_acceptance_rate", float("nan")))
        acceptance_observed = math.isfinite(acceptance_rate)
        acceptance_ok = (
            acceptance_observed
            and acceptance_rate >= float(gate["minimum_acceptance_rate"])
        )
        client_coverage = float(
            row.get("target_client_return_coverage", float("nan"))
        )
        client_coverage_ok = math.isfinite(client_coverage) and client_coverage >= float(
            gate.get("minimum_client_return_coverage", 1.0)
        )
        late_event_count = float(row.get("target_late_event_count", float("nan")))
        late_events_ok = math.isfinite(late_event_count) and late_event_count >= int(
            gate.get("minimum_late_events", 1)
        )
        late_harm_ok = float(row["target_late_harmful_reduction"]) > 0.0
        if gate["requires_positive_late_harm_reduction"]:
            late_harm_ok = late_harm_ok and float(
                row["target_late_harmful_reduction_ci95_low"]
            ) >= 0.0
        cumulative_harm_ok = True
        if gate.get("requires_positive_cumulative_late_harm_reduction", False):
            cumulative_harm_ok = float(
                row.get("target_cumulative_late_harm_reduction", 0.0)
            ) > 0.0
            cumulative_harm_ok = cumulative_harm_ok and float(
                row.get("target_cumulative_late_harm_reduction_ci95_low", 0.0)
            ) >= 0.0
        checks.append(
            {
                "task": row["task"],
                "regime": row["regime"],
                "opponent": row["method"],
                "paired_seeds": int(row["paired_seeds"]),
                "seed_ok": seed_ok,
                "accuracy_noninferior": accuracy_ok,
                "nll_noninferior": nll_ok,
                "acceptance_observed": acceptance_observed,
                "acceptance_noncollapse": acceptance_ok,
                "client_coverage_ok": client_coverage_ok,
                "late_event_count_ok": late_events_ok,
                "late_harm_improved": late_harm_ok,
                "cumulative_late_harm_improved": cumulative_harm_ok,
                "pass": seed_ok
                and accuracy_ok
                and nll_ok
                and acceptance_ok
                and client_coverage_ok
                and late_events_ok
                and late_harm_ok
                and cumulative_harm_ok,
            }
        )

    if completeness_errors:
        status = "INCONCLUSIVE"
        reason = "Matrix is incomplete: " + "; ".join(completeness_errors)
    elif all(check["pass"] for check in checks):
        status = "GO"
        reason = "RIFT improves hard-slice late harm and is non-inferior on accuracy/loss."
    elif any(
        not check["accuracy_noninferior"]
        or not check["nll_noninferior"]
        for check in checks
    ):
        status = "NO_GO"
        reason = "RIFT fails hard-slice accuracy or loss non-inferiority."
    elif any(not check["acceptance_observed"] for check in checks):
        status = "INCONCLUSIVE"
        reason = "Acceptance metrics are missing for at least one hard-slice comparison."
    elif any(not check["acceptance_noncollapse"] for check in checks):
        status = "NO_GO"
        reason = "RIFT fails the minimum acceptance-rate gate."
    elif any(not check["client_coverage_ok"] for check in checks):
        status = "INCONCLUSIVE"
        reason = "Measured client-return coverage is incomplete."
    elif any(not check["late_event_count_ok"] for check in checks):
        status = "INCONCLUSIVE"
        reason = "Too few late events are available for the hard-slice claim."
    else:
        status = "INCONCLUSIVE"
        reason = "Hard-slice safety improvement or seed count is not yet strong enough."
    return {
        "status": status,
        "reason": reason,
        "gate": gate,
        "hard_slice_checks": checks,
        "completeness_errors": completeness_errors,
    }


def mean_ci95(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return 0.0, 0.0, 0.0
    mean = float(clean.mean())
    if len(clean) < 2:
        return mean, mean, mean
    half_width = float(
        stats.t.ppf(0.975, len(clean) - 1) * clean.std(ddof=1) / math.sqrt(len(clean))
    )
    return mean, mean - half_width, mean + half_width


def render_report(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    verdict: dict[str, Any] | None = None,
    *,
    target: str,
) -> str:
    verdict = verdict or week8_verdict(paired)
    lines = [
        "# Kaggle 3B RIFT competitor report",
        "",
        f"Week 8 verdict: **{verdict['status']}**. {verdict['reason']}",
        "",
    ]
    if verdict.get("completeness_errors"):
        lines.extend(["## Completeness Errors", ""])
        lines.extend(f"- {error}" for error in verdict["completeness_errors"])
        lines.append("")
    lines.extend(
        [
            "## Method Summary",
            "",
            "The Loss column is label NLL; sequence NLL is retained as a diagnostic.",
            "",
            "| Task | Regime | Method | Fidelity | Seeds | Accuracy | Loss | Harmful | Late harmful | Acceptance |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['task']} | {row['regime']} | {row['method']} | {row['fidelity']} | "
            f"{int(row['seeds'])} | "
            f"{100 * row['final_accuracy_mean']:.3f}% | "
            f"{row['final_nll_mean']:.6f} | "
            f"{100 * row['harmful_update_rate']:.2f}% | "
            f"{100 * row['late_harmful_update_rate']:.2f}% | "
            f"{100 * row['acceptance_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Best Observed Run (Descriptive Only)",
            "",
            "Best values may come from different seeds and are not used by the GO gate.",
            "",
            "| Task | Regime | Method | Best accuracy | Seed | Best label NLL | Seed |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['task']} | {row['regime']} | {row['method']} | "
            f"{100 * row['best_final_accuracy']:.3f}% | "
            f"{int(row['best_final_accuracy_seed'])} | "
            f"{row['best_final_nll']:.6f} | "
            f"{int(row['best_final_nll_seed'])} |"
        )
    lines.extend(
        [
            "",
            f"## Paired Gains For `{target}`",
            "",
            "| Task | Regime | Opponent | Paired seeds | Acc gain | Acc wins | Loss reduction | Loss wins | Late harmful reduction | Cumulative late-harm reduction |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in paired.iterrows():
        lines.append(
            f"| {row['task']} | {row['regime']} | {row['method']} | "
            f"{int(row['paired_seeds'])} | "
            f"{row['target_accuracy_gain_pp']:+.3f} pp "
            f"[{row['target_accuracy_gain_ci95_low']:+.3f}, "
            f"{row['target_accuracy_gain_ci95_high']:+.3f}] | "
            f"{int(row['target_accuracy_wins'])}/{int(row['paired_seeds'])} | "
            f"{row['target_nll_reduction']:+.6f} "
            f"[{row['target_nll_reduction_ci95_low']:+.6f}, "
            f"{row['target_nll_reduction_ci95_high']:+.6f}] | "
            f"{int(row['target_nll_wins'])}/{int(row['paired_seeds'])} | "
            f"{100 * row['target_late_harmful_reduction']:+.2f} pp "
            f"[{100 * row['target_late_harmful_reduction_ci95_low']:+.2f}, "
            f"{100 * row['target_late_harmful_reduction_ci95_high']:+.2f}] | "
            f"{row['target_cumulative_late_harm_reduction']:+.6f} "
            f"[{row['target_cumulative_late_harm_reduction_ci95_low']:+.6f}, "
            f"{row['target_cumulative_late_harm_reduction_ci95_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Week 8 Hard-Slice Gate",
            "",
            "| Task | Regime | Opponent | Seeds | Accuracy NI | Loss NI | Acceptance | Coverage | Late N | Late rate | Cumulative harm | Pass |",
            "|---|---|---|---:|---|---|---|---|---|---|---|---|",
        ]
    )
    for check in verdict["hard_slice_checks"]:
        lines.append(
            f"| {check['task']} | {check['regime']} | {check['opponent']} | "
            f"{check['paired_seeds']} | "
            f"{check['accuracy_noninferior']} | "
            f"{check['nll_noninferior']} | "
            f"{check['acceptance_noncollapse']} | "
            f"{check['client_coverage_ok']} | "
            f"{check['late_event_count_ok']} | "
            f"{check['late_harm_improved']} | "
            f"{check['cumulative_late_harm_improved']} | "
            f"{check['pass']} |"
        )
    lines.extend(
        [
            "",
            "## Skipped Or Not-Full Competitors",
            "",
        ]
    )
    for name, reason in SKIPPED.items():
        lines.append(f"- **{name}**: {reason}")
    lines.extend(
        [
            "",
            "Do not use this report to claim victory over skipped full paper protocols.",
            "It is a matched Kaggle 3B simulator comparison plus explicit fidelity labels.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
