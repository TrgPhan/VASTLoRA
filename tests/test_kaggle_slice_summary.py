from __future__ import annotations

import json
from pathlib import Path
import importlib.util

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_kaggle_3b_slices",
    ROOT / "scripts" / "summarize_kaggle_3b_slices.py",
)
assert SPEC is not None and SPEC.loader is not None
slice_summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(slice_summary)


def test_slice_summary_builds_vast_hard_regime_verdict(tmp_path: Path) -> None:
    for seed, freshness_nll, vast_nll in (
        (1, 10.0, 9.0),
        (2, 11.0, 9.8),
        (3, 9.0, 8.2),
    ):
        _write_run(
            tmp_path,
            variant="noniid_high_staleness_freshness",
            method="freshness",
            seed=seed,
            accuracy=0.70,
            balanced_accuracy=0.70,
            nll=freshness_nll,
            binary_nll=0.20,
            brier=0.05,
            staleness_values=(1, 5, 9),
        )
        _write_run(
            tmp_path,
            variant="noniid_high_staleness_vast",
            method="vast",
            seed=seed,
            accuracy=0.70,
            balanced_accuracy=0.70,
            nll=vast_nll,
            binary_nll=0.19,
            brier=0.049,
            staleness_values=(1, 5, 9),
        )

    runs = slice_summary.load_runs(tmp_path)
    events = slice_summary.load_events(tmp_path)
    paired = slice_summary.paired_vs_baseline(
        runs, target_method="vast", baseline_method="freshness"
    )
    regime_summary = slice_summary.summarize_pairs(paired)
    event_summary = slice_summary.summarize_events(events)
    verdict = slice_summary.build_verdict(
        paired,
        regime_summary,
        events,
        target_method="vast",
        baseline_method="freshness",
    )

    assert len(runs) == 6
    assert not event_summary.empty
    assert verdict["status"] == "GO"
    assert verdict["late_event_count"] == 3
    assert verdict["sequence_nll_wins"] == 3


def _write_run(
    root: Path,
    *,
    variant: str,
    method: str,
    seed: int,
    accuracy: float,
    balanced_accuracy: float,
    nll: float,
    binary_nll: float,
    brier: float,
    staleness_values: tuple[int, ...],
) -> None:
    target = root / f"{variant}_seed{seed}"
    target.mkdir(parents=True)
    result = {
        "method": method,
        "variant": variant,
        "seed": seed,
        "model": "tiny",
        "git_commit": "abc123",
        "config": {
            "experiment": {
                "partition_mode": "label_shard",
                "client_ranks": [4, 8, 16],
            }
        },
        "metrics": {
            "final_accuracy": accuracy,
            "final_balanced_accuracy": balanced_accuracy,
            "final_nll": nll,
            "final_binary_nll": binary_nll,
            "final_brier": brier,
            "baseline_accuracy": 0.69,
            "baseline_balanced_accuracy": 0.69,
            "baseline_nll": 12.0,
            "baseline_binary_nll": 0.25,
            "baseline_brier": 0.06,
            "accuracy_change_pp": 1.0,
            "nll_change": -2.0,
            "binary_nll_change": -0.05,
            "mean_local_loss": 1.0,
            "mean_staleness": sum(staleness_values) / len(staleness_values),
            "mean_rho_after_warmup": 0.1,
            "mean_adaptive_left_rank": 2.0,
            "mean_adaptive_right_rank": 2.0,
            "runtime_seconds": 1.0,
            "peak_cuda_memory_gib": 1.0,
        },
    }
    (target / "result.json").write_text(json.dumps(result), encoding="utf-8")
    pd.DataFrame(
        {
            "event": list(range(len(staleness_values))),
            "client_id": [0] * len(staleness_values),
            "client_rank": [4, 8, 16][: len(staleness_values)],
            "staleness": list(staleness_values),
            "method": [method] * len(staleness_values),
            "local_loss": [1.0] * len(staleness_values),
            "freshness": [0.5] * len(staleness_values),
            "rho": [0.1] * len(staleness_values),
            "residual_scale": [0.2] * len(staleness_values),
            "mean_left_rank": [2.0] * len(staleness_values),
            "mean_right_rank": [2.0] * len(staleness_values),
        }
    ).to_csv(target / "events.csv", index=False)
