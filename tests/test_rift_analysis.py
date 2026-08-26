from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_week4_rift.py"
SPEC = importlib.util.spec_from_file_location("analyze_week4_rift", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _run(seed: int, method: str, *, final_loss: float, utility: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed": [seed, seed],
            "tau": [2, 9],
            "current_accuracy": [0.70, 0.71],
            "accepted_accuracy": [0.71, 0.72 if method == "rift" else 0.715],
            "current_loss": [0.60, 0.59],
            "accepted_loss": [0.59, final_loss],
            "accepted_method": [method, method],
            f"{method}_update_utility": [utility, utility],
            "rift_step_scale": [1.0, 0.0],
        }
    )


def test_analysis_requires_safety_and_progress_without_reject_all() -> None:
    target = {
        seed: _run(seed, "rift", final_loss=0.57, utility=0.01)
        for seed in range(6)
    }
    baseline = {
        seed: _run(seed, "freshness", final_loss=0.58, utility=-0.01)
        for seed in range(6)
    }

    verdict, metrics, paired = MODULE.analyze_rift(
        target,
        {"freshness": baseline},
        min_acceptance_rate=0.20,
    )

    assert verdict["verdict"] == "GO"
    assert verdict["gates"]["late_safety_better_than_all_baselines"] is True
    assert verdict["rift"]["mean_acceptance_rate"] == 0.5
    assert len(metrics) == 12
    assert len(paired) == 6


def test_analysis_rejects_trivial_zero_acceptance_solution() -> None:
    target = {
        seed: _run(seed, "rift", final_loss=0.57, utility=0.0).assign(
            rift_step_scale=0.0
        )
        for seed in range(6)
    }
    baseline = {
        seed: _run(seed, "freshness", final_loss=0.58, utility=-0.01)
        for seed in range(6)
    }

    verdict, _, _ = MODULE.analyze_rift(
        target,
        {"freshness": baseline},
        min_acceptance_rate=0.20,
    )

    assert verdict["verdict"] == "INCONCLUSIVE"
    assert verdict["gates"]["acceptance_rate_at_least_threshold"] is False
