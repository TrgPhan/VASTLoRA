from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_METHODS = {
    "rift": [
        ROOT / "outputs/week4_rift_confirmation_a",
        ROOT / "outputs/week4_rift_confirmation_b",
    ],
    "freshness": [
        ROOT / "outputs/week4_rift_confirmation_a_freshness",
        ROOT / "outputs/week4_rift_confirmation_b_freshness",
    ],
    "fedex": [ROOT / "outputs/week4_rift_baseline_raw"],
    "projection": [ROOT / "outputs/week4_rift_baseline_projection"],
    "vast": [ROOT / "outputs/week4_rift_baseline_vast"],
    "fedrot": [ROOT / "outputs/week4_rift_competitor_fedrot"],
    "glora_cache": [ROOT / "outputs/week4_rift_competitor_glora"],
    "fedsteer_cache": [ROOT / "outputs/week4_rift_competitor_fedsteer"],
    "alignfed_calibration": [ROOT / "outputs/week4_rift_competitor_alignfed"],
}

FIDELITY = {
    "rift": "proposed",
    "freshness": "faithful scalar baseline",
    "fedex": "faithful exact-intrinsic operator",
    "projection": "matched temporal projection baseline",
    "vast": "project implementation",
    "fedrot": "faithful Procrustes operator + explicit async interpolation",
    "glora_cache": "async cached adaptation; not synchronous GLoRA",
    "fedsteer_cache": "delayed-arrival adaptation; not inactive-client replay",
    "alignfed_calibration": "whole-update calibration control; not full AlignFed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RIFT with matched competitors")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override defaults; repeat PATH entries to merge split runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/week4_rift_competitor_analysis",
    )
    parser.add_argument("--late-tau", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = _parse_methods(args.method) if args.method else DEFAULT_METHODS
    runs = {name: _load_runs(paths) for name, paths in methods.items()}
    expected_seeds = set(runs["rift"])
    for name, values in runs.items():
        if set(values) != expected_seeds:
            raise ValueError(
                f"seed mismatch for {name}: expected={sorted(expected_seeds)}, "
                f"actual={sorted(values)}"
            )

    rows = [
        _summarize_method(name, values, late_tau=args.late_tau)
        for name, values in runs.items()
    ]
    summary = pd.DataFrame(rows).sort_values(
        ["mean_final_accuracy", "mean_final_loss"],
        ascending=[False, True],
    )
    paired = _paired_against_rift(runs)
    result = {
        "seeds": sorted(expected_seeds),
        "late_tau": args.late_tau,
        "summary": summary.to_dict(orient="records"),
        "paired_against_rift": paired,
        "scope_warning": (
            "Results establish matched-simulator evidence only. GLoRA, FedSteer, "
            "and AlignFed controls are adaptations because their published protocols "
            "do not equal delayed-arrival buffer_size=1 Async LoRA."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "competitor_summary.csv", index=False)
    (args.output_dir / "competitor_analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "competitor_report.md").write_text(
        _render_markdown(summary, paired, sorted(expected_seeds)), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(paired, indent=2))


def _parse_methods(values: Iterable[str]) -> dict[str, list[Path]]:
    methods: dict[str, list[Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"method must have NAME=PATH form: {value}")
        name, raw_path = value.split("=", 1)
        methods.setdefault(name, []).append(Path(raw_path))
    if "rift" not in methods:
        raise ValueError("custom methods must include rift")
    return methods


def _load_runs(paths: Iterable[Path]) -> dict[int, pd.DataFrame]:
    runs: dict[int, pd.DataFrame] = {}
    for path in paths:
        run_dir = path / "runs"
        if not run_dir.is_dir():
            raise FileNotFoundError(f"missing run directory: {run_dir}")
        for csv_path in sorted(run_dir.glob("*.csv")):
            frame = pd.read_csv(csv_path)
            if frame.empty:
                raise ValueError(f"empty run: {csv_path}")
            seed = int(frame["seed"].iloc[0])
            if seed in runs:
                raise ValueError(f"duplicate seed {seed} in {csv_path}")
            runs[seed] = frame.sort_values("current_version").reset_index(drop=True)
    if not runs:
        raise ValueError("no run CSV files found")
    return runs


def _summarize_method(
    name: str,
    runs: dict[int, pd.DataFrame],
    *,
    late_tau: int,
) -> dict[str, object]:
    final_accuracy = pd.Series(
        {seed: float(frame["accepted_accuracy"].iloc[-1]) for seed, frame in runs.items()}
    )
    final_loss = pd.Series(
        {seed: float(frame["accepted_loss"].iloc[-1]) for seed, frame in runs.items()}
    )
    combined = pd.concat(runs.values(), ignore_index=True)
    harmful = combined["accepted_loss"] > combined["current_loss"] + 1e-12
    late = combined["tau"] >= late_tau
    return {
        "method": name,
        "fidelity": FIDELITY.get(name, "unspecified"),
        "seeds": len(runs),
        "updates": len(combined),
        "mean_final_accuracy": float(final_accuracy.mean()),
        "std_final_accuracy": float(final_accuracy.std(ddof=1)),
        "mean_final_loss": float(final_loss.mean()),
        "std_final_loss": float(final_loss.std(ddof=1)),
        "harmful_update_rate": float(harmful.mean()),
        "late_harmful_update_rate": float(harmful[late].mean()),
        "mean_update_utility": float(
            (combined["current_loss"] - combined["accepted_loss"]).mean()
        ),
    }


def _paired_against_rift(
    runs: dict[str, dict[int, pd.DataFrame]],
) -> dict[str, dict[str, float | int]]:
    rift = runs["rift"]
    paired: dict[str, dict[str, float | int]] = {}
    for name, values in runs.items():
        if name == "rift":
            continue
        accuracy_gains = []
        loss_reductions = []
        for seed in sorted(rift):
            rift_final = rift[seed].iloc[-1]
            baseline_final = values[seed].iloc[-1]
            accuracy_gains.append(
                float(rift_final["accepted_accuracy"] - baseline_final["accepted_accuracy"])
            )
            loss_reductions.append(
                float(baseline_final["accepted_loss"] - rift_final["accepted_loss"])
            )
        paired[name] = {
            "mean_rift_accuracy_gain": float(pd.Series(accuracy_gains).mean()),
            "rift_accuracy_wins": int(sum(value > 0.0 for value in accuracy_gains)),
            "accuracy_ties": int(sum(value == 0.0 for value in accuracy_gains)),
            "mean_rift_loss_reduction": float(pd.Series(loss_reductions).mean()),
            "rift_loss_wins": int(sum(value > 0.0 for value in loss_reductions)),
            "loss_ties": int(sum(value == 0.0 for value in loss_reductions)),
        }
    return paired


def _render_markdown(
    summary: pd.DataFrame,
    paired: dict[str, dict[str, float | int]],
    seeds: list[int],
) -> str:
    lines = [
        "# RIFT competitor comparison",
        "",
        f"Seeds: {seeds}. All runs use the same SST-2 non-IID/high-staleness trace design.",
        "",
        "| Method | Fidelity | Final acc | Final loss | Harmful | Late harmful |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['method']} | {row['fidelity']} | "
            f"{100 * row['mean_final_accuracy']:.3f}% | {row['mean_final_loss']:.6f} | "
            f"{100 * row['harmful_update_rate']:.2f}% | "
            f"{100 * row['late_harmful_update_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Paired RIFT gains",
            "",
            "| Baseline | Acc gain | Acc wins | Loss reduction | Loss wins |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, values in paired.items():
        lines.append(
            f"| {name} | {100 * values['mean_rift_accuracy_gain']:.3f} pp | "
            f"{values['rift_accuracy_wins']}/{len(seeds)} | "
            f"{values['mean_rift_loss_reduction']:.6f} | "
            f"{values['rift_loss_wins']}/{len(seeds)} |"
        )
    lines.extend(
        [
            "",
            "GLoRA-cache, FedSteer-cache, and AlignFed-calibration are matched controls, "
            "not claims of reproducing each paper's full published protocol.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
