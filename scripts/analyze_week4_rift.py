from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunMetrics:
    method: str
    seed: int
    events: int
    initial_accuracy: float
    final_accuracy: float
    initial_loss: float
    final_loss: float
    loss_progress: float
    mean_update_utility: float
    harmful_update_rate: float
    late_harmful_update_rate: float
    worst_update_utility: float
    cumulative_harm: float
    acceptance_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RIFT safety-progress frontier")
    parser.add_argument("--target-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Matched baseline result directory; may be repeated",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/week4_rift_analysis"
    )
    parser.add_argument("--late-threshold", type=int, default=8)
    parser.add_argument("--harm-tolerance", type=float, default=0.0)
    parser.add_argument("--min-seeds", type=int, default=6)
    parser.add_argument("--max-late-harm-rate", type=float, default=0.10)
    parser.add_argument("--min-acceptance-rate", type=float, default=0.20)
    parser.add_argument("--accuracy-noninferiority-pp", type=float, default=-0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dirs = _parse_baselines(args.baseline)
    target_frames = _load_runs(args.target_dir)
    baseline_frames = {
        name: _load_runs(directories) for name, directories in baseline_dirs.items()
    }
    verdict, run_metrics, paired = analyze_rift(
        target_frames,
        baseline_frames,
        late_threshold=args.late_threshold,
        harm_tolerance=args.harm_tolerance,
        min_seeds=args.min_seeds,
        max_late_harm_rate=args.max_late_harm_rate,
        min_acceptance_rate=args.min_acceptance_rate,
        accuracy_noninferiority_pp=args.accuracy_noninferiority_pp,
    )

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(row) for row in run_metrics]).to_csv(
        output_dir / "run_metrics.csv", index=False
    )
    pd.DataFrame(paired).to_csv(output_dir / "paired_comparisons.csv", index=False)
    (output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    (output_dir / "verdict.md").write_text(
        render_verdict(verdict), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2))


def analyze_rift(
    target_frames: dict[int, pd.DataFrame],
    baseline_frames: dict[str, dict[int, pd.DataFrame]],
    *,
    late_threshold: int = 8,
    harm_tolerance: float = 0.0,
    min_seeds: int = 6,
    max_late_harm_rate: float = 0.10,
    min_acceptance_rate: float = 0.20,
    accuracy_noninferiority_pp: float = -0.50,
) -> tuple[dict[str, object], list[RunMetrics], list[dict[str, object]]]:
    if not target_frames:
        raise ValueError("no target runs were found")
    if not baseline_frames:
        raise ValueError("at least one baseline is required")

    target_metrics = {
        seed: _summarize_run(
            frame,
            method="rift",
            utility_column="rift_update_utility",
            late_threshold=late_threshold,
            harm_tolerance=harm_tolerance,
        )
        for seed, frame in target_frames.items()
    }
    all_metrics = list(target_metrics.values())
    paired_rows: list[dict[str, object]] = []
    comparison_summary: dict[str, dict[str, object]] = {}

    for baseline_name, frames in baseline_frames.items():
        shared_seeds = sorted(set(target_frames) & set(frames))
        if not shared_seeds:
            raise ValueError(f"baseline {baseline_name!r} has no seed paired with RIFT")
        baseline_metrics: dict[int, RunMetrics] = {}
        for seed in shared_seeds:
            utility_column = _accepted_utility_column(frames[seed])
            metric = _summarize_run(
                frames[seed],
                method=baseline_name,
                utility_column=utility_column,
                late_threshold=late_threshold,
                harm_tolerance=harm_tolerance,
            )
            baseline_metrics[seed] = metric
            all_metrics.append(metric)

            target = target_metrics[seed]
            paired_rows.append(
                {
                    "baseline": baseline_name,
                    "seed": seed,
                    "accuracy_gain_pp": 100.0
                    * (target.final_accuracy - metric.final_accuracy),
                    "loss_gain": metric.final_loss - target.final_loss,
                    "late_harm_rate_reduction_pp": 100.0
                    * (
                        metric.late_harmful_update_rate
                        - target.late_harmful_update_rate
                    ),
                    "worst_utility_gain": target.worst_update_utility
                    - metric.worst_update_utility,
                }
            )

        rows = [row for row in paired_rows if row["baseline"] == baseline_name]
        accuracy_gains = [float(row["accuracy_gain_pp"]) for row in rows]
        loss_gains = [float(row["loss_gain"]) for row in rows]
        late_harm_reductions = [
            float(row["late_harm_rate_reduction_pp"]) for row in rows
        ]
        comparison_summary[baseline_name] = {
            "paired_seeds": shared_seeds,
            "mean_accuracy_gain_pp": _mean(accuracy_gains),
            "accuracy_wins": sum(value > 0.0 for value in accuracy_gains),
            "mean_loss_gain": _mean(loss_gains),
            "loss_wins": sum(value > 0.0 for value in loss_gains),
            "mean_late_harm_rate_reduction_pp": _mean(late_harm_reductions),
            "accuracy_gain_95ci": _mean_ci(accuracy_gains),
            "loss_gain_95ci": _mean_ci(loss_gains),
        }

    target_values = list(target_metrics.values())
    target_summary = {
        "seeds": sorted(target_metrics),
        "mean_final_accuracy": _mean(row.final_accuracy for row in target_values),
        "mean_final_loss": _mean(row.final_loss for row in target_values),
        "mean_loss_progress": _mean(row.loss_progress for row in target_values),
        "loss_progress_wins": sum(row.loss_progress > 0.0 for row in target_values),
        "mean_harmful_update_rate": _mean(
            row.harmful_update_rate for row in target_values
        ),
        "mean_late_harmful_update_rate": _mean(
            row.late_harmful_update_rate for row in target_values
        ),
        "mean_acceptance_rate": _mean(row.acceptance_rate for row in target_values),
        "worst_update_utility": min(row.worst_update_utility for row in target_values),
    }

    enough_seeds = len(target_metrics) >= min_seeds and all(
        len(summary["paired_seeds"]) >= min_seeds
        for summary in comparison_summary.values()
    )
    late_safety = (
        float(target_summary["mean_late_harmful_update_rate"])
        <= max_late_harm_rate
    )
    meaningful_acceptance = (
        float(target_summary["mean_acceptance_rate"]) >= min_acceptance_rate
    )
    makes_progress = (
        float(target_summary["mean_loss_progress"]) > 0.0
        and int(target_summary["loss_progress_wins"])
        >= math.ceil(2 * len(target_metrics) / 3)
    )
    baseline_noninferiority = all(
        float(summary["mean_accuracy_gain_pp"]) >= accuracy_noninferiority_pp
        for summary in comparison_summary.values()
    )
    baseline_loss_superiority = all(
        int(summary["loss_wins"]) >= math.ceil(2 * len(summary["paired_seeds"]) / 3)
        and float(summary["mean_loss_gain"]) > 0.0
        for summary in comparison_summary.values()
    )
    improves_late_safety = all(
        float(summary["mean_late_harm_rate_reduction_pp"]) > 0.0
        for summary in comparison_summary.values()
    )
    gates = {
        "enough_paired_seeds": enough_seeds,
        "late_harm_rate_at_most_threshold": late_safety,
        "acceptance_rate_at_least_threshold": meaningful_acceptance,
        "positive_training_progress": makes_progress,
        "accuracy_noninferior_to_all_baselines": baseline_noninferiority,
        "loss_superior_to_all_baselines": baseline_loss_superiority,
        "late_safety_better_than_all_baselines": improves_late_safety,
    }
    verdict = {
        "verdict": "GO" if all(gates.values()) else "INCONCLUSIVE",
        "claim_scope": (
            "research GO for late-update safety in non-IID, high-staleness Async FedLoRA; "
            "not a task-general or all-model superiority claim"
        ),
        "thresholds": {
            "late_threshold": late_threshold,
            "harm_tolerance": harm_tolerance,
            "min_seeds": min_seeds,
            "max_late_harm_rate": max_late_harm_rate,
            "min_acceptance_rate": min_acceptance_rate,
            "accuracy_noninferiority_pp": accuracy_noninferiority_pp,
        },
        "gates": gates,
        "rift": target_summary,
        "comparisons": comparison_summary,
    }
    return verdict, all_metrics, paired_rows


def _summarize_run(
    frame: pd.DataFrame,
    *,
    method: str,
    utility_column: str,
    late_threshold: int,
    harm_tolerance: float,
) -> RunMetrics:
    required = {
        "seed",
        "tau",
        "current_accuracy",
        "accepted_accuracy",
        "current_loss",
        "accepted_loss",
        utility_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"run is missing columns: {missing}")
    if frame.empty or frame["seed"].nunique() != 1:
        raise ValueError("each run must contain one non-empty seed trajectory")

    utility = frame[utility_column].astype(float)
    late_utility = utility[frame["tau"].astype(int) >= late_threshold]
    harmful = utility < -harm_tolerance
    late_harmful = late_utility < -harm_tolerance
    accepted = (
        frame["rift_step_scale"].astype(float) > 0.0
        if method == "rift" and "rift_step_scale" in frame
        else pd.Series(True, index=frame.index)
    )
    return RunMetrics(
        method=method,
        seed=int(frame["seed"].iloc[0]),
        events=len(frame),
        initial_accuracy=float(frame["current_accuracy"].iloc[0]),
        final_accuracy=float(frame["accepted_accuracy"].iloc[-1]),
        initial_loss=float(frame["current_loss"].iloc[0]),
        final_loss=float(frame["accepted_loss"].iloc[-1]),
        loss_progress=float(
            frame["current_loss"].iloc[0] - frame["accepted_loss"].iloc[-1]
        ),
        mean_update_utility=float(utility.mean()),
        harmful_update_rate=float(harmful.mean()),
        late_harmful_update_rate=(
            float(late_harmful.mean()) if not late_harmful.empty else 0.0
        ),
        worst_update_utility=float(utility.min()),
        cumulative_harm=float((-utility[utility < 0.0]).sum()),
        acceptance_rate=float(accepted.mean()),
    )


def _accepted_utility_column(frame: pd.DataFrame) -> str:
    if "accepted_method" not in frame or frame.empty:
        raise ValueError("baseline run does not identify its accepted method")
    methods = set(frame["accepted_method"].astype(str))
    if len(methods) != 1:
        raise ValueError("baseline run mixes accepted methods")
    column = f"{next(iter(methods))}_update_utility"
    if column not in frame:
        raise ValueError(f"baseline utility column {column!r} is missing")
    return column


def _load_runs(directories: Iterable[Path]) -> dict[int, pd.DataFrame]:
    runs: dict[int, pd.DataFrame] = {}
    for directory in directories:
        resolved = directory if directory.is_absolute() else ROOT / directory
        for path in sorted(resolved.glob("runs/*.csv")):
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            seed = int(frame["seed"].iloc[0])
            if seed in runs:
                raise ValueError(f"duplicate seed {seed} in {resolved}")
            runs[seed] = frame
    return runs


def _parse_baselines(values: list[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"baseline must use NAME=DIR syntax: {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"invalid baseline: {value!r}")
        result.setdefault(name, []).append(Path(path))
    return result


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot average an empty sequence")
    return sum(materialized) / len(materialized)


def _mean_ci(values: list[float]) -> list[float]:
    if len(values) < 2:
        mean = _mean(values)
        return [mean, mean]
    series = pd.Series(values, dtype=float)
    half_width = 1.96 * float(series.std(ddof=1)) / math.sqrt(len(series))
    mean = float(series.mean())
    return [mean - half_width, mean + half_width]


def render_verdict(verdict: dict[str, object]) -> str:
    rift = verdict["rift"]
    comparisons = verdict["comparisons"]
    lines = [
        "# RIFT-LoRA safety-progress verdict",
        "",
        f"**{verdict['verdict']}** - {verdict['claim_scope']}.",
        "",
        "## RIFT summary",
        "",
        f"- Mean final accuracy: {100.0 * rift['mean_final_accuracy']:.3f}%",
        f"- Mean final loss: {rift['mean_final_loss']:.6f}",
        f"- Mean late harmful-update rate: {100.0 * rift['mean_late_harmful_update_rate']:.2f}%",
        f"- Mean acceptance rate: {100.0 * rift['mean_acceptance_rate']:.2f}%",
        f"- Positive-loss-progress seeds: {rift['loss_progress_wins']}/{len(rift['seeds'])}",
        "",
        "## Matched comparisons",
        "",
        "| Baseline | Accuracy gain (pp) | Loss gain | Late-harm reduction (pp) |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in comparisons.items():
        lines.append(
            f"| {name} | {summary['mean_accuracy_gain_pp']:+.3f} | "
            f"{summary['mean_loss_gain']:+.6f} | "
            f"{summary['mean_late_harm_rate_reduction_pp']:+.2f} |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: {name}"
        for name, passed in verdict["gates"].items()
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
