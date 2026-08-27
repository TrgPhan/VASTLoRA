from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "configs/week1_competitor_targets.json"
EXTERNAL_METHOD_TO_FRAMEWORK = {
    "fedex": "FedEx-LoRA",
    "fedrot": "FedRot-LoRA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Week 1 competitor board from Kaggle runs and literature targets"
    )
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--our-summary-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.targets)
    our_runs = load_our_runs(args.our_summary_dir) if args.our_summary_dir else pd.DataFrame()
    board = build_board(targets, our_runs)
    verdict = build_verdict(targets, our_runs)

    targets.to_csv(args.output_dir / "week1_literature_targets.csv", index=False)
    if not our_runs.empty:
        our_runs.to_csv(args.output_dir / "our_reproduced_runs.csv", index=False)
    board.to_csv(args.output_dir / "week1_competitor_board.csv", index=False)
    (args.output_dir / "competitor_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "competitor_verdict.md").write_text(
        render_verdict(verdict), encoding="utf-8"
    )

    print(board.to_string(index=False))
    print()
    print(render_verdict(verdict))


def load_targets(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["targets"]
    frame = pd.DataFrame(rows)
    frame["row_type"] = "literature_reference"
    frame["comparable_fairly"] = False
    return frame


def load_our_runs(summary_dir: Path) -> pd.DataFrame:
    method_summary_path = summary_dir / "method_summary.csv"
    regime_summary_path = summary_dir / "regime_summary.csv"
    if not method_summary_path.exists():
        raise FileNotFoundError(f"missing {method_summary_path}")
    method_summary = pd.read_csv(method_summary_path)
    if regime_summary_path.exists():
        regime_summary = pd.read_csv(regime_summary_path)
    else:
        regime_summary = pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (regime, method), group in method_summary.groupby(["regime", "method"], sort=True):
        rows.append(
            {
                "framework": f"RIFTLoRA-repro:{method}",
                "dataset": "SST-2",
                "metric": "mean_accuracy",
                "score": float(group["final_accuracy"].mean()),
                "reported_score": f"{100.0 * group['final_accuracy'].mean():.3f}",
                "model_or_backbone": _unique_or_mixed(group["model"]),
                "setting": f"{regime}; {len(group)} seed(s); { _unique_or_mixed(group['client_ranks']) } ranks",
                "code_public": "yes",
                "reproduction_status": "reproduced_in_this_notebook",
                "source": "Kaggle VAST-LoRA run",
                "row_type": "our_reproduced_accuracy",
                "comparable_fairly": True,
                "method": method,
                "regime": regime,
                "mean_accuracy": float(group["final_accuracy"].mean()),
                "mean_balanced_accuracy": float(group["final_balanced_accuracy"].mean()),
                "mean_sequence_nll": float(group["final_nll"].mean()),
                "mean_binary_nll": float(group["final_binary_nll"].mean()),
                "mean_brier": float(group["final_brier"].mean()),
                "seeds": int(group["seed"].nunique()),
            }
        )

    if not regime_summary.empty:
        for _, row in regime_summary.iterrows():
            rows.append(
                {
                    "framework": f"RIFTLoRA-repro:{row['method']}",
                    "dataset": "SST-2",
                    "metric": "gain_vs_freshness",
                    "score": float(row["mean_balanced_accuracy_gain_pp"]),
                    "reported_score": f"{row['mean_balanced_accuracy_gain_pp']:.3f} pp",
                    "model_or_backbone": "same as reproduced run",
                    "setting": f"{row['regime']}; paired vs freshness",
                    "code_public": "yes",
                    "reproduction_status": "reproduced_in_this_notebook",
                    "source": "Kaggle VAST-LoRA run",
                    "row_type": "our_reproduced_paired_gain",
                    "comparable_fairly": True,
                    "method": row["method"],
                    "regime": row["regime"],
                    "mean_balanced_accuracy_gain_pp": float(
                        row["mean_balanced_accuracy_gain_pp"]
                    ),
                    "mean_sequence_nll_relative_change": float(
                        row["mean_sequence_nll_relative_change"]
                    ),
                    "mean_binary_nll_relative_change": float(
                        row["mean_binary_nll_relative_change"]
                    ),
                    "mean_brier_relative_change": float(row["mean_brier_relative_change"]),
                    "seeds": int(row["seeds"]),
                }
            )
    return pd.DataFrame(rows)


def build_board(targets: pd.DataFrame, our_runs: pd.DataFrame) -> pd.DataFrame:
    if our_runs.empty:
        return targets.sort_values(["row_type", "framework", "dataset"]).reset_index(drop=True)
    columns = sorted(set(targets.columns) | set(our_runs.columns))
    return pd.concat(
        [targets.reindex(columns=columns), our_runs.reindex(columns=columns)],
        ignore_index=True,
    ).sort_values(["row_type", "framework", "dataset", "metric"]).reset_index(drop=True)


def build_verdict(targets: pd.DataFrame, our_runs: pd.DataFrame) -> dict[str, Any]:
    public_targets = targets[targets["reproduction_status"].eq("reference_only_public_code")]
    public_reference_count = int(len(public_targets))
    if our_runs.empty:
        return {
            "status": "INCOMPLETE",
            "reason": "No reproduced VAST-LoRA Kaggle summary was provided.",
            "can_claim_breakthrough_vs_week1_opponents": False,
            "public_reference_count": public_reference_count,
            "unreproduced_public_reference_count": public_reference_count,
            "required_next_step": "Run a reproduced baseline matrix or port public-code baselines into the same simulator.",
        }

    accuracy_rows = our_runs[our_runs["row_type"] == "our_reproduced_accuracy"]
    paired_rows = our_runs[our_runs["row_type"] == "our_reproduced_paired_gain"]
    best_accuracy = accuracy_rows.sort_values("mean_accuracy", ascending=False).head(1)
    vast_paired = paired_rows[paired_rows["method"].eq("vast")]
    hard_vast = vast_paired[vast_paired["regime"].eq("noniid_high_staleness")]
    reproduced_external_methods = sorted(
        set(accuracy_rows["method"].dropna()) & set(EXTERNAL_METHOD_TO_FRAMEWORK)
    )
    reproduced_external_frameworks = sorted(
        EXTERNAL_METHOD_TO_FRAMEWORK[method] for method in reproduced_external_methods
    )
    has_public_external_reproduction = bool(reproduced_external_methods)
    unreproduced_public_reference_count = int(
        (~public_targets["framework"].isin(reproduced_external_frameworks)).sum()
    )
    can_claim_breakthrough = False
    vast_vs_external_hard = None

    if not hard_vast.empty:
        row = hard_vast.iloc[0]
        vast_hard_signal = {
            "balanced_accuracy_gain_pp": float(row["mean_balanced_accuracy_gain_pp"]),
            "sequence_nll_relative_change": float(
                row["mean_sequence_nll_relative_change"]
            ),
            "binary_nll_relative_change": float(row["mean_binary_nll_relative_change"]),
            "brier_relative_change": float(row["mean_brier_relative_change"]),
            "seeds": int(row["seeds"]),
        }
    else:
        vast_hard_signal = None

    if has_public_external_reproduction:
        hard_accuracy = accuracy_rows[accuracy_rows["regime"].eq("noniid_high_staleness")]
        hard_vast_accuracy = hard_accuracy[hard_accuracy["method"].eq("vast")]
        hard_external = hard_accuracy[
            hard_accuracy["method"].isin(reproduced_external_methods)
        ]
        if not hard_vast_accuracy.empty and not hard_external.empty:
            vast_row = hard_vast_accuracy.iloc[0]
            best_external = hard_external.sort_values(
                ["mean_accuracy", "mean_balanced_accuracy"],
                ascending=False,
            ).iloc[0]
            vast_vs_external_hard = {
                "best_external_method": str(best_external["method"]),
                "best_external_framework": EXTERNAL_METHOD_TO_FRAMEWORK[
                    str(best_external["method"])
                ],
                "accuracy_gain_pp": 100.0
                * (float(vast_row["mean_accuracy"]) - float(best_external["mean_accuracy"])),
                "balanced_accuracy_gain_pp": 100.0
                * (
                    float(vast_row["mean_balanced_accuracy"])
                    - float(best_external["mean_balanced_accuracy"])
                ),
                "sequence_nll_relative_change": float(vast_row["mean_sequence_nll"])
                / float(best_external["mean_sequence_nll"])
                - 1.0,
                "binary_nll_relative_change": float(vast_row["mean_binary_nll"])
                / float(best_external["mean_binary_nll"])
                - 1.0,
                "brier_relative_change": float(vast_row["mean_brier"])
                / float(best_external["mean_brier"])
                - 1.0,
            }
            can_claim_breakthrough = (
                vast_vs_external_hard["balanced_accuracy_gain_pp"] >= -0.5
                and vast_vs_external_hard["binary_nll_relative_change"] <= 0.05
                and vast_vs_external_hard["brier_relative_change"] <= 0.05
                and vast_vs_external_hard["sequence_nll_relative_change"] < 0.0
                and vast_hard_signal is not None
                and vast_hard_signal["binary_nll_relative_change"] <= 0.05
                and vast_hard_signal["brier_relative_change"] <= 0.05
            )
        reason = (
            "The board now includes matched-simulator ports for "
            f"{', '.join(reproduced_external_frameworks)} plus Week 1 literature targets. "
            "Remaining literature rows are still reference-only, so broad Week 1 superiority "
            "requires more external baselines or a clearly stated limited claim."
        )
    else:
        reason = (
            "The board contains reproduced in-house baselines plus Week 1 literature targets. "
            "Because FedRot/FedEx/FSLoRA/GLoRA are not yet ported into the same simulator, "
            "the literature rows are reference-only and cannot prove a fair breakthrough."
        )
    return {
        "status": "REFERENCE_BOARD_READY",
        "reason": reason,
        "can_claim_breakthrough_vs_week1_opponents": can_claim_breakthrough,
        "has_public_external_reproduction": has_public_external_reproduction,
        "public_reference_count": public_reference_count,
        "unreproduced_public_reference_count": unreproduced_public_reference_count,
        "reproduced_external_methods": reproduced_external_methods,
        "reproduced_external_frameworks": reproduced_external_frameworks,
        "best_reproduced_accuracy_row": (
            best_accuracy.iloc[0].to_dict() if not best_accuracy.empty else None
        ),
        "vast_hard_slice_signal": vast_hard_signal,
        "vast_vs_external_hard_slice": vast_vs_external_hard,
        "minimum_for_breakthrough_claim": [
            "Port at least FedRot-LoRA or FedEx-LoRA into the same dataset/model/client simulator, or run their official code under a matched config.",
            "Report standard task metrics plus NLL/Brier/ECE and staleness/rank slices.",
            "Show VAST is non-inferior on accuracy and clearly better on stale-update reliability in at least one LLM-scale task and one second task.",
        ],
    }


def render_verdict(verdict: dict[str, Any]) -> str:
    lines = [f"## Week 1 competitor verdict: {verdict['status']}", "", verdict["reason"], ""]
    lines.append(
        f"- Can claim breakthrough vs Week 1 opponents: `{verdict['can_claim_breakthrough_vs_week1_opponents']}`"
    )
    lines.append(
        f"- Public-code reference rows not yet reproduced here: {verdict['public_reference_count']}"
    )
    if "unreproduced_public_reference_count" in verdict:
        lines[-1] = (
            "- Public-code reference rows not yet reproduced here: "
            f"{verdict['unreproduced_public_reference_count']} / {verdict['public_reference_count']}"
        )
    if verdict.get("reproduced_external_frameworks"):
        lines.append(
            "- Matched-simulator external ports: "
            + ", ".join(f"`{name}`" for name in verdict["reproduced_external_frameworks"])
        )
    if verdict.get("vast_hard_slice_signal") is not None:
        signal = verdict["vast_hard_slice_signal"]
        lines.extend(
            [
                "- VAST hard-slice signal:",
                f"  - Balanced-accuracy gain: {signal['balanced_accuracy_gain_pp']:.3f} pp",
                f"  - Sequence NLL relative change: {100.0 * signal['sequence_nll_relative_change']:.2f}%",
                f"  - Binary NLL relative change: {100.0 * signal['binary_nll_relative_change']:.2f}%",
                f"  - Brier relative change: {100.0 * signal['brier_relative_change']:.2f}%",
            ]
        )
    if verdict.get("vast_vs_external_hard_slice") is not None:
        external = verdict["vast_vs_external_hard_slice"]
        lines.extend(
            [
                "- VAST vs best external hard-slice port:",
                f"  - Best external: `{external['best_external_framework']}`",
                f"  - Balanced-accuracy gain: {external['balanced_accuracy_gain_pp']:.3f} pp",
                f"  - Sequence NLL relative change: {100.0 * external['sequence_nll_relative_change']:.2f}%",
                f"  - Binary NLL relative change: {100.0 * external['binary_nll_relative_change']:.2f}%",
                f"  - Brier relative change: {100.0 * external['brier_relative_change']:.2f}%",
            ]
        )
    lines.extend(["", "### Minimum for a real breakthrough claim"])
    for item in verdict.get("minimum_for_breakthrough_claim", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _unique_or_mixed(values: pd.Series) -> str:
    unique = [str(value) for value in values.dropna().unique()]
    if not unique:
        return "unknown"
    if len(unique) == 1:
        return unique[0]
    return "mixed"


if __name__ == "__main__":
    main()

