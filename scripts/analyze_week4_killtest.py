from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from riftlora.diagnostics import (
    analyze_scope,
    decide_gate,
    matched_tau_analysis,
    validate_diagnostic_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Week-4 GO/NO-GO analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/week4/week3_diagnostics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/week4/analysis",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/week4/week4_results_vi.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    validation = validate_diagnostic_dataframe(
        frame,
        min_stale_updates=100,
        artifact_root=args.input.parent,
    )
    if not validation["valid"]:
        raise RuntimeError(f"invalid Week-3 dataframe: {validation['errors']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scopes = {
        regime: analyze_scope(part.reset_index(drop=True))
        for regime, part in frame.groupby("regime")
    }
    scopes["all_regimes"] = analyze_scope(frame)
    gate = decide_gate({name: value for name, value in scopes.items() if name != "all_regimes"})
    matched_frames = []
    for regime, part in frame.groupby("regime"):
        matched = matched_tau_analysis(part)
        matched.insert(0, "regime", regime)
        matched_frames.append(matched)
    matched_all = pd.concat(matched_frames, ignore_index=True)
    matched_all.to_csv(args.output_dir / "matched_tau.csv", index=False)

    summary = {"dataframe_validation": validation, "scopes": scopes, "gate": gate}
    (args.output_dir / "week4_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_scope_table(scopes, args.output_dir / "scope_metrics.csv")
    _write_plots(frame, scopes, args.output_dir)
    report_path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(summary, matched_all), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    print(f"report: {report_path}")


def _write_scope_table(scopes: dict[str, dict[str, Any]], path: Path) -> None:
    rows = []
    for name, values in scopes.items():
        rows.append(
            {
                "scope": name,
                "rows": values["rows"],
                "partial_spearman": values["partial_spearman_rho_utility_given_tau"],
                "partial_ci_low": values["partial_spearman_ci95"][0],
                "partial_ci_high": values["partial_spearman_ci95"][1],
                "cv_r2_gain": values["cv_r2_gain"],
                "auc_gain": values["harmful_auc_gain"],
                "vast_minus_freshness": values["vast_minus_freshness_mean_utility"],
                "freshness_harmful_rate": values["freshness_harmful_rate"],
                "vast_harmful_rate": values["vast_harmful_rate"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_plots(
    frame: pd.DataFrame,
    scopes: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    colors = {
        "iid_homogeneous": "#2a6f97",
        "iid_heterogeneous": "#6a994e",
        "noniid_high_staleness": "#bc4749",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for regime, part in frame.groupby("regime"):
        ax.scatter(part["tau"], part["raw_update_utility"], s=14, alpha=0.45, label=regime, color=colors.get(regime))
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set(xlabel="Version staleness (tau)", ylabel="Raw update utility", title="Utility vs staleness")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "utility_vs_tau.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for regime, part in frame.groupby("regime"):
        ax.scatter(part["rho_two_sided"], part["raw_update_utility"], s=14, alpha=0.45, label=regime, color=colors.get(regime))
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set(xlabel="Two-sided compatibility (rho)", ylabel="Raw update utility", title="Utility vs compatibility")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "utility_vs_rho.png", dpi=180)
    plt.close(fig)

    names = [name for name in scopes if name != "all_regimes"]
    positions = range(len(names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar([value - width for value in positions], [scopes[name]["harmful_auc_tau"] for name in names], width, label="tau")
    ax.bar(list(positions), [scopes[name]["harmful_auc_rho"] for name in names], width, label="rho")
    ax.bar([value + width for value in positions], [scopes[name]["harmful_auc_tau_rho"] for name in names], width, label="tau + rho")
    ax.set_xticks(list(positions), [name.replace("_", "\n") for name in names], fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set(ylabel="Held-seed-out AUROC", title="Harmful-update prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "harmful_auc.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    methods = ("raw", "freshness", "vast")
    for offset, method in enumerate(methods):
        ax.bar(
            [value + (offset - 1) * width for value in positions],
            [scopes[name][f"{method}_mean_utility"] for name in names],
            width,
            label=method,
        )
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(list(positions), [name.replace("_", "\n") for name in names], fontsize=8)
    ax.set(ylabel="Mean post-hoc utility", title="Raw, freshness, and VAST utility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "transport_utility.png", dpi=180)
    plt.close(fig)


def _render_report(summary: dict[str, Any], matched: pd.DataFrame) -> str:
    gate = summary["gate"]
    lines = [
        "# Week 4 kill-test results",
        "",
        f"Decision: **{gate['decision']}**",
        "",
        f"Reason: {gate['reason']}.",
        "",
        "## Preregistered gate metrics",
        "",
        "| Regime | N | Partial Spearman | 95% CI | CV R2 gain | AUROC gain | VAST - freshness utility | Harmful freshness -> VAST | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, values in summary["scopes"].items():
        if name == "all_regimes":
            continue
        ci = values["partial_spearman_ci95"]
        lines.append(
            f"| {name} | {values['rows']} | {values['partial_spearman_rho_utility_given_tau']:.3f} "
            f"| [{ci[0]:.3f}, {ci[1]:.3f}] | {values['cv_r2_gain']:.3f} "
            f"| {values['harmful_auc_gain']:.3f} | {values['vast_minus_freshness_mean_utility']:.6f} "
            f"| {values['freshness_harmful_rate']:.3f} -> {values['vast_harmful_rate']:.3f} "
            f"| {'yes' if gate['scope_pass'].get(name) else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Matched-tau analysis",
            "",
            matched.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Interpretation guardrails",
            "",
            "- The gate uses two-sided rho and thresholds frozen before the full run.",
            "- Validation loss is used only for post-hoc scientific evaluation, not by the transport rule.",
            "- A single SST-2/BERT-tiny kill-test can reject the current hypothesis, but cannot establish the final thesis claim without a second task and larger backbone.",
            "- Full numeric outputs and plots are in `outputs/week4/analysis/`.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

