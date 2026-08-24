import json
from pathlib import Path

import pandas as pd

from vastlora.scale.nll_debug import write_nll_debug


def _write_run(root: Path, method: str, seed: int, rows: list[dict]) -> None:
    target = root / f"{method}_seed{seed}"
    target.mkdir(parents=True)
    (target / "result.json").write_text(
        json.dumps({"method": method, "seed": seed, "metrics": {}}),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(target / "final_eval_details.csv", index=False)
    pd.DataFrame(
        [
            {
                "event": index,
                "staleness": index,
                "local_loss": 1.0 + index,
                "rho": 0.1 * index,
                "mean_left_rank": 4.0,
                "mean_right_rank": 4.0,
            }
            for index in range(6)
        ]
    ).to_csv(target / "events.csv", index=False)


def _row(index: int, correct: bool, true_nll: float, confidence: float) -> dict:
    true_label = index % 2
    predicted = true_label if correct else 1 - true_label
    return {
        "eval_index": index,
        "text": f"example {index}",
        "true_label": true_label,
        "predicted_label": predicted,
        "is_correct": int(correct),
        "nll_negative": true_nll if true_label == 0 else true_nll + 0.5,
        "nll_positive": true_nll if true_label == 1 else true_nll + 0.5,
        "true_nll": true_nll,
        "wrong_nll": true_nll + 0.5,
        "nll_margin": 0.5 if correct else -0.5,
        "prob_negative": confidence if predicted == 0 else 1.0 - confidence,
        "prob_positive": confidence if predicted == 1 else 1.0 - confidence,
        "true_probability": confidence if correct else 1.0 - confidence,
        "prediction_confidence": confidence,
    }


def test_nll_debug_detects_accuracy_nll_disagreement(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "freshness",
        1,
        [_row(0, True, 0.2, 0.9), _row(1, False, 0.8, 0.8)],
    )
    _write_run(
        tmp_path,
        "mtip_adaptive",
        1,
        [_row(0, True, 5.0, 0.6), _row(1, True, 4.0, 0.55)],
    )

    outputs = write_nll_debug(tmp_path)

    assert outputs.missing_runs == []
    paired = outputs.paired_summary.set_index("method")
    assert paired.loc["mtip_adaptive", "fixed_freshness_errors"] == 1
    assert paired.loc["mtip_adaptive", "mean_delta_nll_vs_freshness"] > 0
    assert (outputs.output_dir / "nll_worst_samples.csv").exists()


def test_nll_debug_reports_missing_detail_artifacts(tmp_path: Path) -> None:
    target = tmp_path / "freshness_seed1"
    target.mkdir()
    (target / "result.json").write_text("{}", encoding="utf-8")

    outputs = write_nll_debug(tmp_path)

    assert outputs.missing_runs == ["freshness_seed1"]
    assert (outputs.output_dir / "nll_debug.md").exists()
