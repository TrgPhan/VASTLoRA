from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


FIDELITY = {
    "raw": "matched async baseline",
    "fedex": "faithful exact intrinsic innovation in matched simulator",
    "freshness": "faithful scalar staleness weighting baseline",
    "fedrot": "matched FedRot-LoRA Procrustes operator",
    "vast": "legacy VAST residual transport",
    "mtip": "projection-only transport baseline",
    "mtip_adaptive": "adaptive projection transport baseline",
    "spectral_filter": "Spectral-Surgery-style gradient component filter; not official paper implementation",
    "alignfed_calibration": "whole-update calibration control; not full AlignFed",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Kaggle 3B RIFT competitor runs")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="rift")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_results(args.input_dir)
    summary = summarize(frame)
    paired = paired_against(frame, args.target)
    report = render_report(summary, paired, target=args.target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "runs.csv", index=False)
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_target.csv", index=False)
    (args.output_dir / "competitor_report.md").write_text(report, encoding="utf-8")
    (args.output_dir / "skipped_competitors.json").write_text(
        json.dumps(SKIPPED, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(report)


def load_results(input_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*_seed*/result.json")):
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
        rows.append(
            {
                "task": task,
                "regime": regime,
                "method": method,
                "paired_seeds": int(joined.shape[0]),
                "target_accuracy_gain_pp": float(
                    100.0
                    * (
                        joined["final_accuracy_target"]
                        - joined["final_accuracy_candidate"]
                    ).mean()
                ),
                "target_accuracy_wins": int(
                    (
                        joined["final_accuracy_target"]
                        > joined["final_accuracy_candidate"]
                    ).sum()
                ),
                "target_nll_reduction": float(
                    (
                        joined["final_nll_candidate"]
                        - joined["final_nll_target"]
                    ).mean()
                ),
                "target_nll_wins": int(
                    (joined["final_nll_target"] < joined["final_nll_candidate"]).sum()
                ),
                "target_binary_nll_reduction": float(
                    (
                        joined["final_binary_nll_candidate"]
                        - joined["final_binary_nll_target"]
                    ).mean()
                ),
                "target_harmful_reduction": float(
                    (
                        joined["harmful_update_rate_candidate"]
                        - joined["harmful_update_rate_target"]
                    ).mean()
                )
                if "harmful_update_rate_candidate" in joined
                else 0.0,
                "target_late_harmful_reduction": float(
                    (
                        joined["late_harmful_update_rate_candidate"]
                        - joined["late_harmful_update_rate_target"]
                    ).mean()
                )
                if "late_harmful_update_rate_candidate" in joined
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["task", "regime", "target_accuracy_gain_pp", "target_nll_reduction"],
        ascending=[True, True, False, False],
    )


def render_report(summary: pd.DataFrame, paired: pd.DataFrame, *, target: str) -> str:
    lines = [
        "# Kaggle 3B RIFT competitor report",
        "",
        "## Method Summary",
        "",
        "| Task | Regime | Method | Fidelity | Seeds | Accuracy | Loss | Harmful | Late harmful |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['task']} | {row['regime']} | {row['method']} | {row['fidelity']} | "
            f"{int(row['seeds'])} | "
            f"{100 * row['final_accuracy_mean']:.3f}% | "
            f"{row['final_nll_mean']:.6f} | "
            f"{100 * row['harmful_update_rate']:.2f}% | "
            f"{100 * row['late_harmful_update_rate']:.2f}% |"
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
            f"{row['target_accuracy_gain_pp']:+.3f} pp | "
            f"{int(row['target_accuracy_wins'])}/{int(row['paired_seeds'])} | "
            f"{row['target_nll_reduction']:+.6f} | "
            f"{int(row['target_nll_wins'])}/{int(row['paired_seeds'])} | "
            f"{100 * row['target_harmful_reduction']:+.2f} pp | "
            f"{100 * row['target_late_harmful_reduction']:+.2f} pp |"
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
