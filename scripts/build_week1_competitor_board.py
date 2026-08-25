from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "configs/week1_competitor_targets.json"


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
                "framework": f"VASTLoRA-repro:{method}",
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
                    "framework": f"VASTLoRA-repro:{row['method']}",
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
    public_reference_count = int(
        targets["reproduction_status"].eq("reference_only_public_code").sum()
    )
    if our_runs.empty:
        return {
            "status": "INCOMPLETE",
            "reason": "No reproduced VAST-LoRA Kaggle summary was provided.",
            "can_claim_breakthrough_vs_week1_opponents": False,
            "public_reference_count": public_reference_count,
            "required_next_step": "Run a reproduced baseline matrix or port public-code baselines into the same simulator.",
        }

    accuracy_rows = our_runs[our_runs["row_type"] == "our_reproduced_accuracy"]
    paired_rows = our_runs[our_runs["row_type"] == "our_reproduced_paired_gain"]
    best_accuracy = accuracy_rows.sort_values("mean_accuracy", ascending=False).head(1)
    vast_paired = paired_rows[paired_rows["method"].eq("vast")]
    hard_vast = vast_paired[vast_paired["regime"].eq("noniid_high_staleness")]
    has_public_external_reproduction = False
    can_claim_breakthrough = False

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
        "best_reproduced_accuracy_row": (
            best_accuracy.iloc[0].to_dict() if not best_accuracy.empty else None
        ),
        "vast_hard_slice_signal": vast_hard_signal,
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
