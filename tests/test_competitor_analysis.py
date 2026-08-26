from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_week4_competitors.py"
SPEC = spec_from_file_location("analyze_week4_competitors", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _run(final_accuracy: float, final_loss: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "current_version": [1, 2],
            "accepted_accuracy": [0.7, final_accuracy],
            "current_loss": [0.62, 0.60],
            "accepted_loss": [0.61, final_loss],
            "tau": [4, 9],
        }
    )


def test_paired_analysis_reports_rift_wins() -> None:
    runs = {
        "rift": {1: _run(0.74, 0.55), 2: _run(0.75, 0.54)},
        "fedrot": {1: _run(0.72, 0.58), 2: _run(0.73, 0.57)},
    }

    paired = MODULE._paired_against_rift(runs)

    assert paired["fedrot"]["rift_accuracy_wins"] == 2
    assert paired["fedrot"]["rift_loss_wins"] == 2
    assert paired["fedrot"]["mean_rift_accuracy_gain"] == pytest.approx(0.02)


def test_method_summary_uses_accepted_trajectory_harm() -> None:
    runs = {1: _run(0.74, 0.61), 2: _run(0.75, 0.59)}

    summary = MODULE._summarize_method("rift", runs, late_tau=8)

    assert summary["harmful_update_rate"] == 0.25
    assert summary["late_harmful_update_rate"] == 0.5
