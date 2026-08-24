from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats


EXPECTED_METHODS = {"freshness", "vast", "mtip", "mtip_adaptive"}


def summarize_results(
    input_dir: Path,
    *,
    target_variant: str = "mtip_adaptive",
    development_status: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    records = []
    for path in sorted(input_dir.glob("*_seed*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "method": payload["method"],
                "variant": payload.get("variant", payload["method"]),
                "seed": payload["seed"],
                "model": payload["model"],
                "git_commit": payload["git_commit"],
                **payload["metrics"],
            }
        )
    if not records:
        raise ValueError(f"no result.json files found under {input_dir}")

    frame = pd.DataFrame(records).sort_values(["seed", "variant"]).reset_index(drop=True)
    comparisons = _paired_comparisons(frame)
    missing = sorted(EXPECTED_METHODS - set(frame["method"]))
    verdict = _build_verdict(
        frame,
        comparisons,
        missing,
        target_variant=target_variant,
        development_status=development_status,
    )
    return frame, comparisons, verdict


def render_verdict_markdown(verdict: dict[str, Any]) -> str:
    lines = [f"## 3B verdict: {verdict['status']}", "", verdict["reason"]]
    if "target_accuracy_gain_vs_freshness_pp" in verdict:
        lines.extend(
            [
                "",
                f"- Target variant: `{verdict['target_variant']}`",
                f"- Accuracy gain: {verdict['target_accuracy_gain_vs_freshness_pp']:+.3f} pp",
                f"- Sequence NLL gain: {verdict['target_nll_gain_vs_freshness']:+.6f}",
                f"- Accuracy wins: {verdict['target_accuracy_wins']}/{verdict['seed_count']}",
                f"- Best mean-accuracy method: `{verdict['best_method_by_mean_accuracy']}`",
                "",
                (
                    "`PILOT_GO` is provisional. A full `GO` requires at least "
                    f"{verdict['gate']['minimum_seeds_for_full_go']} seeds."
                ),
            ]
        )
        binary_gain = verdict.get("target_binary_nll_gain_vs_freshness")
        if binary_gain is not None:
            lines.insert(-3, f"- Binary candidate NLL gain: {binary_gain:+.6f}")
    return "\n".join(lines) + "\n"


def write_summary(
    input_dir: Path,
    output_dir: Path,
    *,
    target_variant: str = "mtip_adaptive",
    development_status: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary, comparisons, verdict = summarize_results(
        input_dir,
        target_variant=target_variant,
        development_status=development_status,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    (output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "verdict.md").write_text(
        render_verdict_markdown(verdict), encoding="utf-8"
    )
    return summary, verdict


def _paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, group in frame.groupby("seed"):
        baseline_rows = group[group["variant"] == "freshness"]
        if baseline_rows.empty:
            baseline_rows = group[group["method"] == "freshness"]
        if len(baseline_rows) != 1:
            continue
        baseline = baseline_rows.iloc[0]
        for _, candidate in group[group.index != baseline.name].iterrows():
            row = {
                    "seed": int(seed),
                    "method": candidate["method"],
                    "variant": candidate["variant"],
                    "accuracy_gain_vs_freshness_pp": 100.0
                    * (candidate["final_accuracy"] - baseline["final_accuracy"]),
                    "nll_gain_vs_freshness": baseline["final_nll"] - candidate["final_nll"],
                    "runtime_ratio_vs_freshness": candidate["runtime_seconds"]
                    / baseline["runtime_seconds"],
                    "memory_delta_vs_freshness_gib": candidate["peak_cuda_memory_gib"]
                    - baseline["peak_cuda_memory_gib"],
                }
            if "final_binary_nll" in frame.columns:
                row["binary_nll_gain_vs_freshness"] = (
                    baseline["final_binary_nll"] - candidate["final_binary_nll"]
                )
                row["binary_nll_relative_change_vs_freshness"] = (
                    candidate["final_binary_nll"] / baseline["final_binary_nll"] - 1.0
                )
            if "final_balanced_accuracy" in frame.columns:
                row["balanced_accuracy_gain_vs_freshness_pp"] = 100.0 * (
                    candidate["final_balanced_accuracy"]
                    - baseline["final_balanced_accuracy"]
                )
            if "final_brier" in frame.columns:
                row["brier_relative_change_vs_freshness"] = (
                    candidate["final_brier"] / baseline["final_brier"] - 1.0
                )
            row["nll_relative_change_vs_freshness"] = (
                candidate["final_nll"] / baseline["final_nll"] - 1.0
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _build_verdict(
    frame: pd.DataFrame,
    comparisons: pd.DataFrame,
    missing: list[str],
    *,
    target_variant: str,
    development_status: str | None,
) -> dict[str, Any]:
    if missing:
        return {
            "status": "INCOMPLETE",
            "reason": f"Missing methods: {', '.join(missing)}",
            "missing_methods": missing,
            "seed_count": int(frame["seed"].nunique()),
        }

    available = set(comparisons["variant"])
    if target_variant not in available:
        return {
            "status": "INCOMPLETE",
            "reason": f"Target variant is missing: {target_variant}",
            "missing_target_variant": target_variant,
            "seed_count": int(frame["seed"].nunique()),
        }
    target = comparisons[comparisons["variant"] == target_variant]
    target_method = str(target["method"].iloc[0])
    mean_accuracy_gain = float(target["accuracy_gain_vs_freshness_pp"].mean())
    mean_nll_gain = float(target["nll_gain_vs_freshness"].mean())
    wins = int((target["accuracy_gain_vs_freshness_pp"] > 0).sum())
    seed_count = int(target["seed"].nunique())
    mean_binary_nll_gain = None
    if "binary_nll_gain_vs_freshness" in target.columns:
        value = target["binary_nll_gain_vs_freshness"].mean()
        if pd.notna(value):
            mean_binary_nll_gain = float(value)
    robust_metrics = {
        "balanced_accuracy_gain_vs_freshness_pp",
        "binary_nll_relative_change_vs_freshness",
        "brier_relative_change_vs_freshness",
        "nll_relative_change_vs_freshness",
    }.issubset(target.columns)
    best_method = (
        frame.groupby("variant")["final_accuracy"].mean().sort_values(ascending=False).index[0]
    )

    if robust_metrics:
        balanced_mean, balanced_lower, balanced_upper = _mean_ci(
            target["balanced_accuracy_gain_vs_freshness_pp"]
        )
        sequence_relative_mean, _, sequence_relative_upper = _mean_ci(
            target["nll_relative_change_vs_freshness"]
        )
        binary_relative_mean, _, binary_relative_upper = _mean_ci(
            target["binary_nll_relative_change_vs_freshness"]
        )
        brier_relative_mean, _, brier_relative_upper = _mean_ci(
            target["brier_relative_change_vs_freshness"]
        )
        balanced_wins = int(
            (target["balanced_accuracy_gain_vs_freshness_pp"] > 0.0).sum()
        )
        clears_metric_gate = (
            balanced_mean >= 0.5
            and (seed_count < 2 or balanced_lower > 0.0)
            and sequence_relative_upper <= 0.10
            and binary_relative_upper <= 0.05
            and brier_relative_upper <= 0.05
            and target["balanced_accuracy_gain_vs_freshness_pp"].min() >= -0.5
            and balanced_wins >= math.ceil(0.8 * seed_count)
        )
        minimum_full_seeds = 5
    else:
        balanced_mean = balanced_lower = balanced_upper = None
        sequence_relative_mean = sequence_relative_upper = None
        binary_relative_mean = binary_relative_upper = None
        brier_relative_mean = brier_relative_upper = None
        balanced_wins = None
        clears_nll_gate = mean_nll_gain >= 0.0 and (
            mean_binary_nll_gain is None or mean_binary_nll_gain >= 0.0
        )
        clears_metric_gate = mean_accuracy_gain >= 0.5 and clears_nll_gate
        minimum_full_seeds = 3
    wins_every_seed = wins == seed_count
    if clears_metric_gate and (robust_metrics or wins_every_seed):
        status = "PILOT_GO" if seed_count < minimum_full_seeds else "GO"
        reason = f"{target_variant} clears the preregistered accuracy/calibration Pareto gate."
    elif mean_accuracy_gain <= -0.5 and mean_nll_gain < 0.0:
        status = "NO_GO"
        reason = f"{target_variant} is materially worse than freshness on accuracy and NLL."
    else:
        status = "INCONCLUSIVE"
        reason = "Accuracy and NLL do not provide a consistent margin over freshness."
    if development_status == "DEV_GATE_MISS" and status in {"PILOT_GO", "GO"}:
        status = "INCONCLUSIVE"
        reason = (
            "Confirmation metrics pass, but the frozen target did not clear the "
            "development gate; GO is blocked by protocol."
        )

    verdict = {
        "status": status,
        "reason": reason,
        "best_method_by_mean_accuracy": str(best_method),
        "target_method": target_method,
        "target_variant": target_variant,
        "development_status": development_status,
        "target_accuracy_gain_vs_freshness_pp": mean_accuracy_gain,
        "target_nll_gain_vs_freshness": mean_nll_gain,
        "target_binary_nll_gain_vs_freshness": mean_binary_nll_gain,
        "target_accuracy_wins": wins,
        "target_balanced_accuracy_gain_pp": balanced_mean,
        "target_balanced_accuracy_ci95": [balanced_lower, balanced_upper]
        if balanced_lower is not None
        else None,
        "target_sequence_nll_relative_change": sequence_relative_mean,
        "target_sequence_nll_relative_ci95_upper": sequence_relative_upper,
        "target_binary_nll_relative_change": binary_relative_mean,
        "target_binary_nll_relative_ci95_upper": binary_relative_upper,
        "target_brier_relative_change": brier_relative_mean,
        "target_brier_relative_ci95_upper": brier_relative_upper,
        "target_balanced_accuracy_wins": balanced_wins,
        "seed_count": seed_count,
        "gate": {
            "minimum_accuracy_gain_pp": 0.5,
            "requires_nonnegative_nll_gain": True,
            "requires_nonnegative_binary_nll_gain_when_available": True,
            "requires_accuracy_win_every_seed": True,
            "minimum_balanced_accuracy_gain_pp": 0.5,
            "requires_positive_balanced_accuracy_ci95_lower": robust_metrics,
            "maximum_sequence_nll_relative_ci95_upper": 0.10,
            "maximum_binary_nll_relative_ci95_upper": 0.05,
            "maximum_brier_relative_ci95_upper": 0.05,
            "minimum_seeds_for_full_go": minimum_full_seeds,
        },
    }
    if target_variant == "mtip_adaptive":
        verdict.update(
            {
                "adaptive_accuracy_gain_vs_freshness_pp": mean_accuracy_gain,
                "adaptive_nll_gain_vs_freshness": mean_nll_gain,
                "adaptive_accuracy_wins": wins,
            }
        )
    return verdict


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna().astype(float)
    mean = float(clean.mean())
    if len(clean) < 2:
        return mean, mean, mean
    half_width = float(
        stats.t.ppf(0.975, len(clean) - 1) * clean.std(ddof=1) / math.sqrt(len(clean))
    )
    return mean, mean - half_width, mean + half_width
