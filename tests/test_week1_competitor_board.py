from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_week1_competitor_board",
    ROOT / "scripts" / "build_week1_competitor_board.py",
)
assert SPEC is not None and SPEC.loader is not None
competitor_board = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(competitor_board)


def test_competitor_board_keeps_reference_targets_separate(tmp_path: Path) -> None:
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "targets": [
                    {
                        "framework": "FedRot-LoRA",
                        "dataset": "SST-2/QNLI",
                        "metric": "average_accuracy",
                        "score": 0.89,
                        "reported_score": "0.89",
                        "model_or_backbone": "paper",
                        "setting": "paper setting",
                        "code_public": "yes",
                        "reproduction_status": "reference_only_public_code",
                        "source": "week1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    pd.DataFrame(
        [
            {
                "regime": "noniid_high_staleness",
                "method": "fedrot",
                "variant": "noniid_high_staleness_fedrot",
                "seed": 1,
                "model": "qwen",
                "client_ranks": "4/8/16",
                "final_accuracy": 0.92,
                "final_balanced_accuracy": 0.92,
                "final_nll": 4.8,
                "final_binary_nll": 0.195,
                "final_brier": 0.050,
            },
            {
                "regime": "noniid_high_staleness",
                "method": "freshness",
                "variant": "noniid_high_staleness_freshness",
                "seed": 1,
                "model": "qwen",
                "client_ranks": "4/8/16",
                "final_accuracy": 0.90,
                "final_balanced_accuracy": 0.90,
                "final_nll": 5.0,
                "final_binary_nll": 0.2,
                "final_brier": 0.05,
            },
            {
                "regime": "noniid_high_staleness",
                "method": "vast",
                "variant": "noniid_high_staleness_vast",
                "seed": 1,
                "model": "qwen",
                "client_ranks": "4/8/16",
                "final_accuracy": 0.91,
                "final_balanced_accuracy": 0.91,
                "final_nll": 4.5,
                "final_binary_nll": 0.19,
                "final_brier": 0.049,
            },
        ]
    ).to_csv(summary_dir / "method_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "regime": "noniid_high_staleness",
                "method": "vast",
                "seeds": 1,
                "mean_balanced_accuracy_gain_pp": 1.0,
                "mean_sequence_nll_relative_change": -0.1,
                "mean_binary_nll_relative_change": -0.05,
                "mean_brier_relative_change": -0.02,
            }
        ]
    ).to_csv(summary_dir / "regime_summary.csv", index=False)

    targets = competitor_board.load_targets(targets_path)
    our_runs = competitor_board.load_our_runs(summary_dir)
    board = competitor_board.build_board(targets, our_runs)
    verdict = competitor_board.build_verdict(targets, our_runs)

    assert set(board["row_type"]) >= {
        "literature_reference",
        "our_reproduced_accuracy",
        "our_reproduced_paired_gain",
    }
    assert verdict["can_claim_breakthrough_vs_week1_opponents"] is False
    assert verdict["public_reference_count"] == 1
    assert verdict["has_public_external_reproduction"] is True
    assert verdict["unreproduced_public_reference_count"] == 0
    assert verdict["reproduced_external_frameworks"] == ["FedRot-LoRA"]
    assert verdict["vast_vs_external_hard_slice"]["best_external_method"] == "fedrot"
    assert verdict["vast_hard_slice_signal"]["sequence_nll_relative_change"] == -0.1
