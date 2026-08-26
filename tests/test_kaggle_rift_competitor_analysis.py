from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_kaggle_3b_rift_competitors.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_kaggle_3b_rift_competitors", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_result(root: Path, method: str, task: str, regime: str, seed: int) -> None:
    target = root / f"{task}_{regime}_{method}_seed{seed}"
    target.mkdir(parents=True)
    payload = {
        "method": method,
        "variant": f"{task}_{regime}_{method}",
        "seed": seed,
        "model": "test-3b",
        "task": task,
        "regime": regime,
        "git_commit": "abc123",
        "metrics": {
            "final_accuracy": 0.75 if method == "rift" else 0.70,
            "final_nll": 0.50 if method == "rift" else 0.60,
            "final_binary_nll": 0.20,
            "final_brier": 0.10,
            "harmful_update_rate": 0.10 if method == "rift" else 0.50,
            "late_harmful_update_rate": 0.00 if method == "rift" else 0.40,
            "monitor_loss_change": -0.01,
            "runtime_seconds": 1.0,
            "peak_cuda_memory_gib": 2.0,
        },
    }
    (target / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_kaggle_rift_report_groups_task_regime_and_harm(tmp_path: Path) -> None:
    _write_result(tmp_path, "rift", "qnli", "noniid_high_staleness", 1)
    _write_result(tmp_path, "freshness", "qnli", "noniid_high_staleness", 1)

    frame = MODULE.load_results(tmp_path)
    summary = MODULE.summarize(frame)
    paired = MODULE.paired_against(frame, "rift")
    report = MODULE.render_report(summary, paired, target="rift")

    assert summary.iloc[0]["task"] == "qnli"
    assert summary[summary["method"] == "rift"].iloc[0]["late_harmful_update_rate"] == 0.0
    assert paired.iloc[0]["target_late_harmful_reduction"] == 0.40
    assert "Accuracy | Loss | Harmful | Late harmful" in report
