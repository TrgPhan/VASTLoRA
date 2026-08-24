import json
from pathlib import Path

import pytest

from vastlora.scale.reporting import summarize_results


def _write_result(root: Path, method: str, seed: int, accuracy: float, nll: float) -> None:
    target = root / f"{method}_seed{seed}"
    target.mkdir(parents=True)
    payload = {
        "method": method,
        "seed": seed,
        "model": "test-3b",
        "git_commit": "abc123",
        "metrics": {
            "baseline_accuracy": 0.5,
            "baseline_nll": 1.0,
            "final_accuracy": accuracy,
            "final_nll": nll,
            "accuracy_change_pp": 0.0,
            "nll_change": 0.0,
            "mean_local_loss": 0.5,
            "mean_staleness": 4.0,
            "mean_rho_after_warmup": 0.5,
            "mean_adaptive_left_rank": 4.0,
            "mean_adaptive_right_rank": 4.0,
            "runtime_seconds": 10.0,
            "peak_cuda_memory_gib": 4.0,
        },
    }
    (target / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_summary_marks_single_seed_improvement_as_pilot_go(tmp_path: Path) -> None:
    _write_result(tmp_path, "freshness", 1, 0.70, 0.50)
    _write_result(tmp_path, "vast", 1, 0.69, 0.51)
    _write_result(tmp_path, "mtip", 1, 0.705, 0.49)
    _write_result(tmp_path, "mtip_adaptive", 1, 0.72, 0.45)

    summary, comparisons, verdict = summarize_results(tmp_path)

    assert len(summary) == 4
    assert len(comparisons) == 3
    assert verdict["status"] == "PILOT_GO"
    assert verdict["adaptive_accuracy_gain_vs_freshness_pp"] == pytest.approx(2.0)


def test_summary_refuses_verdict_when_method_is_missing(tmp_path: Path) -> None:
    _write_result(tmp_path, "freshness", 1, 0.70, 0.50)
    _write_result(tmp_path, "mtip_adaptive", 1, 0.72, 0.45)

    _, _, verdict = summarize_results(tmp_path)

    assert verdict["status"] == "INCOMPLETE"
    assert set(verdict["missing_methods"]) == {"vast", "mtip"}
