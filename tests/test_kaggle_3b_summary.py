import json
from pathlib import Path

import pytest

from riftlora.scale.reporting import summarize_results


def _write_result(
    root: Path,
    method: str,
    seed: int,
    accuracy: float,
    nll: float,
    binary_nll: float | None = None,
    variant: str | None = None,
    balanced_accuracy: float | None = None,
    brier: float | None = None,
) -> None:
    target_name = variant or method
    target = root / f"{target_name}_seed{seed}"
    target.mkdir(parents=True)
    payload = {
        "method": method,
        "variant": target_name,
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
    if binary_nll is not None:
        payload["metrics"]["final_binary_nll"] = binary_nll
    if balanced_accuracy is not None:
        payload["metrics"]["final_balanced_accuracy"] = balanced_accuracy
    if brier is not None:
        payload["metrics"]["final_brier"] = brier
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


def test_full_go_requires_adaptive_to_win_every_seed(tmp_path: Path) -> None:
    for seed, freshness_accuracy, adaptive_accuracy in (
        (1, 0.70, 0.73),
        (2, 0.70, 0.73),
        (3, 0.70, 0.695),
    ):
        _write_result(tmp_path, "freshness", seed, freshness_accuracy, 0.50)
        _write_result(tmp_path, "vast", seed, 0.69, 0.51)
        _write_result(tmp_path, "mtip", seed, 0.70, 0.50)
        _write_result(tmp_path, "mtip_adaptive", seed, adaptive_accuracy, 0.45)

    _, _, verdict = summarize_results(tmp_path)

    assert verdict["adaptive_accuracy_gain_vs_freshness_pp"] > 0.5
    assert verdict["adaptive_accuracy_wins"] == 2
    assert verdict["status"] == "INCONCLUSIVE"


def test_routed_method_must_clear_both_nll_gates(tmp_path: Path) -> None:
    for seed in (1, 2, 3):
        _write_result(tmp_path, "freshness", seed, 0.70, 0.50, 0.20)
        _write_result(tmp_path, "vast", seed, 0.69, 0.48, 0.19)
        _write_result(tmp_path, "mtip", seed, 0.71, 0.60, 0.25)
        _write_result(tmp_path, "mtip_adaptive", seed, 0.71, 0.55, 0.22)
        _write_result(
            tmp_path,
            "mtip_routed",
            seed,
            0.72,
            0.49,
            0.21,
            variant="routed_c4_t1",
        )

    _, _, verdict = summarize_results(tmp_path, target_variant="routed_c4_t1")

    assert verdict["target_method"] == "mtip_routed"
    assert verdict["target_variant"] == "routed_c4_t1"
    assert verdict["target_nll_gain_vs_freshness"] > 0.0
    assert verdict["target_binary_nll_gain_vs_freshness"] < 0.0
    assert verdict["status"] == "INCONCLUSIVE"


def test_five_seed_target_can_clear_robust_pareto_gate(tmp_path: Path) -> None:
    for seed in (1, 2, 3, 4, 5):
        _write_result(
            tmp_path,
            "freshness",
            seed,
            0.70,
            0.50,
            0.20,
            balanced_accuracy=0.70,
            brier=0.10,
        )
        _write_result(tmp_path, "vast", seed, 0.69, 0.49, 0.19)
        _write_result(tmp_path, "mtip", seed, 0.70, 0.60, 0.25)
        _write_result(tmp_path, "mtip_adaptive", seed, 0.70, 0.55, 0.22)
        _write_result(
            tmp_path,
            "mtip_hybrid",
            seed,
            0.71,
            0.525,
            0.204,
            variant="hybrid_beta020",
            balanced_accuracy=0.71,
            brier=0.102,
        )

    _, _, verdict = summarize_results(tmp_path, target_variant="hybrid_beta020")

    assert verdict["status"] == "GO"
    assert verdict["target_balanced_accuracy_gain_pp"] == pytest.approx(1.0)
    assert verdict["gate"]["minimum_seeds_for_full_go"] == 5

    _, _, blocked = summarize_results(
        tmp_path,
        target_variant="hybrid_beta020",
        development_status="DEV_GATE_MISS",
    )
    assert blocked["status"] == "INCONCLUSIVE"
    assert "blocked by protocol" in blocked["reason"]


def test_reporting_refuses_unfrozen_target_variant(tmp_path: Path) -> None:
    _write_result(tmp_path, "freshness", 1, 0.70, 0.50)
    _write_result(tmp_path, "vast", 1, 0.69, 0.51)
    _write_result(tmp_path, "mtip", 1, 0.70, 0.50)
    _write_result(tmp_path, "mtip_adaptive", 1, 0.71, 0.49)

    _, _, verdict = summarize_results(tmp_path, target_variant="hybrid_beta020")

    assert verdict["status"] == "INCOMPLETE"
    assert verdict["missing_target_variant"] == "hybrid_beta020"

