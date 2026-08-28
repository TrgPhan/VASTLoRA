from __future__ import annotations

import argparse
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
WEEK8_GATE = {
    "minimum_paired_seeds": 6,
    "minimum_acceptance_rate": 0.30,
    "hard_regimes": ["noniid_high_staleness"],
    "accuracy_noninferiority_margin_pp": -0.5,
    "nll_noninferiority_margin": -0.005,
    "requires_positive_late_harm_reduction": True,
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
    completeness_errors = validate_matrix_completeness(frame, matrix)
    try:
        validate_seed_alignment(frame, target=args.target)
    except ValueError as exc:
        completeness_errors.append(str(exc))
    summary = summarize(frame)
    paired = paired_against(frame, args.target)
    verdict = week8_verdict(paired, completeness_errors=completeness_errors)
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
        records.append(
            {
                "method": payload["method"],
                "variant": payload.get("variant", payload["method"]),
                "seed": int(payload["seed"]),
                "model": payload["model"],
                "task": payload.get("task", payload.get("config", {}).get("dataset", {}).get("subset", "unknown")),
                "regime": payload.get("regime", payload.get("config", {}).get("experiment", {}).get("regime_name", "default")),
                "git_commit": payload["git_commit"],
                **payload["metrics"],
            }
        )
    if not records:
        raise ValueError(f"no result.json files found under {input_dir}")
    return pd.DataFrame(records).sort_values(["task", "regime", "method", "seed"]).reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, regime, method), group in frame.groupby(["task", "regime", "method"]):
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
                "final_nll_mean": float(group["final_nll"].mean()),
                "final_binary_nll_mean": float(group["final_binary_nll"].mean()),
                "final_brier_mean": float(group["final_brier"].mean()),
                "harmful_update_rate": float(group.get("harmful_update_rate", pd.Series([0.0])).mean()),
                "late_harmful_update_rate": float(
                    group.get("late_harmful_update_rate", pd.Series([0.0])).mean()
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
                "cumulative_late_harm_mean": float(
                    group.get("cumulative_late_harm", pd.Series([float("nan")])).mean()
                ),
                "worst_step_loss_increase_mean": float(
                    group.get("worst_step_loss_increase", pd.Series([float("nan")])).mean()
                ),
                "utility_per_accepted_update_mean": float(
                    group.get(
                        "utility_per_accepted_update", pd.Series([float("nan")])
                    ).mean()
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
        nll_delta = joined["final_nll_candidate"] - joined["final_nll_target"]
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
        acc_mean, acc_low, acc_high = mean_ci95(accuracy_delta)
        nll_mean, nll_low, nll_high = mean_ci95(nll_delta)
        binary_mean, binary_low, binary_high = mean_ci95(binary_nll_delta)
        harmful_mean, harmful_low, harmful_high = mean_ci95(harmful_delta)
        late_mean, late_low, late_high = mean_ci95(late_harmful_delta)
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
                "target_nll_reduction": nll_mean,
                "target_nll_reduction_ci95_low": nll_low,
                "target_nll_reduction_ci95_high": nll_high,
                "target_nll_wins": int(
                    (joined["final_nll_target"] < joined["final_nll_candidate"]).sum()
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
                "target_acceptance_rate": target_acceptance_rate,
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
    completeness_errors: list[str] | None = None,
) -> dict[str, Any]:
    completeness_errors = list(completeness_errors or [])
    hard = paired[paired["regime"].isin(WEEK8_GATE["hard_regimes"])].copy()
    checks: list[dict[str, Any]] = []
    if hard.empty:
        return {
            "status": "INCONCLUSIVE",
            "reason": "No hard-slice paired rows are available.",
            "gate": WEEK8_GATE,
            "hard_slice_checks": checks,
            "completeness_errors": completeness_errors,
        }

    for _, row in hard.iterrows():
        seed_ok = int(row["paired_seeds"]) >= WEEK8_GATE["minimum_paired_seeds"]
        accuracy_ok = (
            float(row["target_accuracy_gain_ci95_low"])
            >= WEEK8_GATE["accuracy_noninferiority_margin_pp"]
        )
        nll_ok = (
            float(row["target_nll_reduction_ci95_low"])
            >= WEEK8_GATE["nll_noninferiority_margin"]
        )
        acceptance_rate = float(row.get("target_acceptance_rate", float("nan")))
        acceptance_observed = math.isfinite(acceptance_rate)
        acceptance_ok = (
            acceptance_observed
            and acceptance_rate >= WEEK8_GATE["minimum_acceptance_rate"]
        )
        late_harm_ok = float(row["target_late_harmful_reduction"]) > 0.0
        if WEEK8_GATE["requires_positive_late_harm_reduction"]:
            late_harm_ok = late_harm_ok and float(
                row["target_late_harmful_reduction_ci95_low"]
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
                "late_harm_improved": late_harm_ok,
                "pass": seed_ok
                and accuracy_ok
                and nll_ok
                and acceptance_ok
                and late_harm_ok,
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
    else:
        status = "INCONCLUSIVE"
        reason = "Hard-slice safety improvement or seed count is not yet strong enough."
    return {
        "status": status,
        "reason": reason,
        "gate": WEEK8_GATE,
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
            f"## Paired Gains For `{target}`",
            "",
            "| Task | Regime | Opponent | Paired seeds | Acc gain | Acc wins | Loss reduction | Loss wins | Harmful reduction | Late harmful reduction |",
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
            f"{100 * row['target_harmful_reduction']:+.2f} pp "
            f"[{100 * row['target_harmful_reduction_ci95_low']:+.2f}, "
            f"{100 * row['target_harmful_reduction_ci95_high']:+.2f}] | "
            f"{100 * row['target_late_harmful_reduction']:+.2f} pp "
            f"[{100 * row['target_late_harmful_reduction_ci95_low']:+.2f}, "
            f"{100 * row['target_late_harmful_reduction_ci95_high']:+.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Week 8 Hard-Slice Gate",
            "",
            "| Task | Regime | Opponent | Seeds | Accuracy NI | Loss NI | Acceptance | Late harm improved | Pass |",
            "|---|---|---|---:|---|---|---|---|---|",
        ]
    )
    for check in verdict["hard_slice_checks"]:
        lines.append(
            f"| {check['task']} | {check['regime']} | {check['opponent']} | "
            f"{check['paired_seeds']} | "
            f"{check['accuracy_noninferior']} | "
            f"{check['nll_noninferior']} | "
            f"{check['acceptance_noncollapse']} | "
            f"{check['late_harm_improved']} | "
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
