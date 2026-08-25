from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


TAU_BANDS = ((0, 2, "0-2"), (3, 7, "3-7"), (8, None, "8+"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 3B VAST slice-matrix runs")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-method", default="vast")
    parser.add_argument("--baseline-method", default="freshness")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir / "slice_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.input_dir)
    if runs.empty:
        raise ValueError(f"no result.json files found under {args.input_dir}")
    events = load_events(args.input_dir)
    paired = paired_vs_baseline(
        runs,
        target_method=args.target_method,
        baseline_method=args.baseline_method,
    )
    regime_summary = summarize_pairs(paired)
    event_summary = summarize_events(events)
    verdict = build_verdict(
        paired,
        regime_summary,
        events,
        target_method=args.target_method,
        baseline_method=args.baseline_method,
    )

    runs.to_csv(output_dir / "method_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)
    event_summary.to_csv(output_dir / "event_slice_summary.csv", index=False)
    (output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "verdict.md").write_text(render_verdict(verdict), encoding="utf-8")

    print(runs.to_string(index=False))
    print()
    print(paired.to_string(index=False))
    print()
    print(regime_summary.to_string(index=False))
    print()
    print(render_verdict(verdict))


def load_runs(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(input_dir.glob("*_seed*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = result["metrics"]
        config = result.get("config", {})
        experiment = config.get("experiment", {})
        variant = result.get("variant") or result_path.parent.name.rsplit("_seed", 1)[0]
        regime = str(result.get("regime") or _variant_regime(variant))
        method = str(result["method"])
        rows.append(
            {
                "regime": regime,
                "method": method,
                "variant": variant,
                "seed": int(result["seed"]),
                "model": result.get("model"),
                "git_commit": result.get("git_commit"),
                "partition_mode": experiment.get("partition_mode", "label_shard"),
                "client_ranks": "/".join(str(rank) for rank in experiment.get("client_ranks", [])),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["regime", "seed", "method"]).reset_index(drop=True)


def load_events(input_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for events_path in sorted(input_dir.glob("*_seed*/events.csv")):
        variant = events_path.parent.name.rsplit("_seed", 1)[0]
        result_path = events_path.parent / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(events_path)
        frame["regime"] = str(result.get("regime") or _variant_regime(variant))
        frame["variant"] = variant
        frame["run_method"] = str(result["method"])
        frame["seed"] = int(result["seed"])
        frame["tau_band"] = frame["staleness"].map(tau_band)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def paired_vs_baseline(
    runs: pd.DataFrame,
    *,
    target_method: str,
    baseline_method: str,
) -> pd.DataFrame:
    baseline = runs[runs["method"] == baseline_method].set_index(["regime", "seed"])
    rows: list[dict[str, Any]] = []
    for _, row in runs[runs["method"] != baseline_method].iterrows():
        key = (row["regime"], row["seed"])
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        rows.append(
            {
                "regime": row["regime"],
                "seed": int(row["seed"]),
                "method": row["method"],
                "variant": row["variant"],
                "accuracy_gain_pp": 100.0 * (row["final_accuracy"] - base["final_accuracy"]),
                "balanced_accuracy_gain_pp": 100.0
                * (row["final_balanced_accuracy"] - base["final_balanced_accuracy"]),
                "sequence_nll_gain": base["final_nll"] - row["final_nll"],
                "sequence_nll_relative_change": row["final_nll"] / base["final_nll"] - 1.0,
                "binary_nll_gain": base["final_binary_nll"] - row["final_binary_nll"],
                "binary_nll_relative_change": row["final_binary_nll"]
                / base["final_binary_nll"]
                - 1.0,
                "brier_relative_change": row["final_brier"] / base["final_brier"] - 1.0,
                "accuracy_win": row["final_accuracy"] > base["final_accuracy"],
                "sequence_nll_win": row["final_nll"] < base["final_nll"],
                "binary_nll_win": row["final_binary_nll"] < base["final_binary_nll"],
            }
        )
    return pd.DataFrame(rows).sort_values(["regime", "method", "seed"]).reset_index(drop=True)


def summarize_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if paired.empty:
        return pd.DataFrame(rows)
    for (regime, method), group in paired.groupby(["regime", "method"], sort=True):
        rows.append(
            {
                "regime": regime,
                "method": method,
                "seeds": len(group),
                "mean_accuracy_gain_pp": group["accuracy_gain_pp"].mean(),
                "mean_balanced_accuracy_gain_pp": group["balanced_accuracy_gain_pp"].mean(),
                "mean_sequence_nll_gain": group["sequence_nll_gain"].mean(),
                "mean_sequence_nll_relative_change": group[
                    "sequence_nll_relative_change"
                ].mean(),
                "mean_binary_nll_relative_change": group[
                    "binary_nll_relative_change"
                ].mean(),
                "mean_brier_relative_change": group["brier_relative_change"].mean(),
                "accuracy_wins": int(group["accuracy_win"].sum()),
                "sequence_nll_wins": int(group["sequence_nll_win"].sum()),
                "binary_nll_wins": int(group["binary_nll_win"].sum()),
                "sequence_nll_relative_ci95": ci95(
                    group["sequence_nll_relative_change"].tolist()
                ),
                "balanced_accuracy_gain_ci95": ci95(
                    group["balanced_accuracy_gain_pp"].tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in events.groupby(["regime", "run_method", "tau_band", "client_rank"], sort=True):
        regime, method, band, rank = keys
        rows.append(
            {
                "regime": regime,
                "method": method,
                "tau_band": band,
                "client_rank": int(rank),
                "events": len(group),
                "mean_staleness": group["staleness"].mean(),
                "mean_rho": group["rho"].mean(),
                "mean_residual_scale": group["residual_scale"].mean(),
                "mean_local_loss": group["local_loss"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_verdict(
    paired: pd.DataFrame,
    regime_summary: pd.DataFrame,
    events: pd.DataFrame,
    *,
    target_method: str,
    baseline_method: str,
) -> dict[str, Any]:
    hard_regime = "noniid_high_staleness"
    target = regime_summary[
        (regime_summary["regime"] == hard_regime)
        & (regime_summary["method"] == target_method)
    ]
    target_rows = paired[
        (paired["regime"] == hard_regime) & (paired["method"] == target_method)
    ]
    if target.empty:
        return {
            "status": "INCOMPLETE",
            "reason": f"missing {target_method} paired results for {hard_regime}",
            "target_method": target_method,
            "baseline_method": baseline_method,
        }
    target_summary = target.iloc[0].to_dict()
    late_events = events[
        (events["regime"] == hard_regime)
        & (events["run_method"] == target_method)
        & (events["staleness"] >= 8)
    ]
    late_event_count = int(len(late_events))
    seed_count = int(target_summary["seeds"])
    nll_ci = target_summary["sequence_nll_relative_ci95"]
    acc_ci = target_summary["balanced_accuracy_gain_ci95"]
    sequence_nll_wins = int(target_summary["sequence_nll_wins"])
    binary_nll_wins = int(target_summary["binary_nll_wins"])
    mean_balanced_accuracy_gain = float(target_summary["mean_balanced_accuracy_gain_pp"])
    mean_sequence_nll_relative_change = float(
        target_summary["mean_sequence_nll_relative_change"]
    )
    mean_binary_nll_relative_change = float(
        target_summary["mean_binary_nll_relative_change"]
    )
    mean_brier_relative_change = float(target_summary["mean_brier_relative_change"])

    go = (
        seed_count >= 3
        and late_event_count > 0
        and mean_balanced_accuracy_gain >= -0.5
        and acc_ci[0] >= -1.0
        and mean_sequence_nll_relative_change <= -0.05
        and nll_ci[1] < 0.0
        and mean_binary_nll_relative_change <= 0.05
        and mean_brier_relative_change <= 0.05
        and sequence_nll_wins == seed_count
        and binary_nll_wins >= max(1, seed_count - 1)
    )
    if go:
        status = "GO"
        reason = (
            f"{target_method} improves sequence NLL in the hard 3B slice while "
            "preserving balanced accuracy and calibration non-inferiority."
        )
    elif mean_sequence_nll_relative_change < 0.0 and mean_balanced_accuracy_gain >= -1.0:
        status = "INCONCLUSIVE"
        reason = (
            f"{target_method} has some likelihood signal in the hard 3B slice, "
            "but it does not clear the preregistered NLL/calibration gate."
        )
    else:
        status = "NO_GO"
        reason = (
            f"{target_method} does not provide a useful NLL/calibration trade-off "
            f"against {baseline_method} in the hard 3B slice."
        )
    return {
        "status": status,
        "reason": reason,
        "target_method": target_method,
        "baseline_method": baseline_method,
        "hard_regime": hard_regime,
        "seed_count": seed_count,
        "late_event_count": late_event_count,
        "mean_balanced_accuracy_gain_pp": mean_balanced_accuracy_gain,
        "balanced_accuracy_gain_ci95": acc_ci,
        "mean_sequence_nll_relative_change": mean_sequence_nll_relative_change,
        "sequence_nll_relative_ci95": nll_ci,
        "mean_binary_nll_relative_change": mean_binary_nll_relative_change,
        "mean_brier_relative_change": mean_brier_relative_change,
        "sequence_nll_wins": sequence_nll_wins,
        "binary_nll_wins": binary_nll_wins,
        "target_rows": target_rows.to_dict(orient="records"),
        "gate": {
            "minimum_seeds": 3,
            "requires_late_events": True,
            "minimum_mean_balanced_accuracy_gain_pp": -0.5,
            "minimum_balanced_accuracy_ci95_low_pp": -1.0,
            "maximum_mean_sequence_nll_relative_change": -0.05,
            "maximum_sequence_nll_relative_ci95_high": 0.0,
            "maximum_binary_nll_relative_change": 0.05,
            "maximum_brier_relative_change": 0.05,
            "requires_sequence_nll_win_every_seed": True,
            "requires_binary_nll_wins_at_least": "seed_count - 1",
        },
    }


def render_verdict(verdict: dict[str, Any]) -> str:
    if verdict["status"] == "INCOMPLETE":
        return f"## 3B slice verdict: INCOMPLETE\n\n{verdict['reason']}\n"
    return "\n".join(
        [
            f"## 3B slice verdict: {verdict['status']}",
            "",
            verdict["reason"],
            "",
            f"- Target method: `{verdict['target_method']}`",
            f"- Baseline method: `{verdict['baseline_method']}`",
            f"- Hard regime: `{verdict['hard_regime']}`",
            f"- Seeds: {verdict['seed_count']}",
            f"- Late events (`tau >= 8`): {verdict['late_event_count']}",
            f"- Balanced-accuracy gain: {verdict['mean_balanced_accuracy_gain_pp']:.3f} pp",
            f"- Sequence NLL relative change: {100.0 * verdict['mean_sequence_nll_relative_change']:.2f}%",
            f"- Binary NLL relative change: {100.0 * verdict['mean_binary_nll_relative_change']:.2f}%",
            f"- Brier relative change: {100.0 * verdict['mean_brier_relative_change']:.2f}%",
            f"- Sequence NLL wins: {verdict['sequence_nll_wins']}/{verdict['seed_count']}",
            f"- Binary NLL wins: {verdict['binary_nll_wins']}/{verdict['seed_count']}",
            "",
        ]
    )


def tau_band(value: float | int) -> str:
    tau = int(value)
    for low, high, label in TAU_BANDS:
        if tau >= low and (high is None or tau <= high):
            return label
    raise ValueError(f"unsupported tau value: {value}")


def ci95(values: list[float]) -> list[float]:
    if not values:
        return [math.nan, math.nan]
    mean = sum(values) / len(values)
    if len(values) == 1:
        return [mean, mean]
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    t_critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    return [mean - t_critical * stderr, mean + t_critical * stderr]


def _variant_regime(variant: str) -> str:
    for suffix in ("_freshness", "_vast", "_mtip", "_mtip_adaptive"):
        if variant.endswith(suffix):
            return variant[: -len(suffix)]
    parts = variant.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else variant


if __name__ == "__main__":
    main()
