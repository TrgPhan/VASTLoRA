import numpy as np
import pandas as pd

from riftlora.diagnostics.analysis import (
    analyze_scope,
    decide_gate,
    matched_tau_analysis,
    partial_spearman,
)


def _signal_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for seed in (17, 31, 43):
        for index in range(60):
            tau = index % 12
            rho = rng.uniform()
            utility = 0.04 * rho - 0.002 * tau + rng.normal(0, 0.003)
            rows.append(
                {
                    "run_id": f"r{seed}",
                    "seed": seed,
                    "tau": tau,
                    "rho_two_sided": rho,
                    "raw_update_utility": utility,
                    "freshness_update_utility": utility - 0.002,
                    "vast_update_utility": utility,
                }
            )
    return pd.DataFrame(rows)


def test_partial_spearman_recovers_signal_beyond_tau() -> None:
    value = partial_spearman(
        _signal_frame(),
        x="rho_two_sided",
        y="raw_update_utility",
        controls=("tau",),
    )
    assert value > 0.8


def test_scope_analysis_and_matched_tau_are_complete() -> None:
    frame = _signal_frame()
    result = analyze_scope(frame)
    matched = matched_tau_analysis(frame)
    assert result["cv_r2_gain"] > 0
    assert result["harmful_auc_tau_rho"] >= 0.5
    assert set(matched["tau_band"]) == {"0-2", "3-7", "8+"}


def test_gate_reports_no_go_without_supported_scope() -> None:
    values = analyze_scope(_signal_frame())
    values["partial_spearman_ci95"] = [-0.1, 0.2]
    result = decide_gate({"iid_homogeneous": values})
    assert result["decision"] == "NO-GO"

