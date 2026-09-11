import json
import math
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_week9_generation import analyze, paired_interval
from run_week9_generation import load_matrix, specs
import run_week8_classification_matrix as runner


def test_paired_interval_not_standard_deviation_of_unpaired_scores():
    mean, low, high = paired_interval([0.1] * 6)
    assert mean == pytest.approx(0.1)
    assert low == pytest.approx(mean) and high == pytest.approx(mean)
    assert paired_interval([0.1]) == (None, None, None)


def test_incomplete_confirmation_cannot_pass(tmp_path):
    (tmp_path / "matrix.json").write_text(json.dumps(load_matrix("confirmation")))
    report = analyze(tmp_path)
    assert report["status"] == "INCOMPLETE_OR_UNVERIFIED"
    assert len(report["missing"]) == 72


def test_complete_paired_cohort_then_corrupted_details(tmp_path):
    matrix = load_matrix("confirmation")
    (tmp_path / "matrix.json").write_text(json.dumps(matrix))
    for method, seed, config, path in specs(matrix, tmp_path):
        nll = 0.8 if method == "rift_core" else 1.0
        payload = {
            "schema_version": 5, "method": method, "seed": seed, "config": config,
            "config_fingerprint": runner._runner_config_fingerprint(config),
            "provenance": config["provenance"], "git_worktree_dirty": False, "git_commit": "a" * 40,
            "metrics": {"final_token_nll": nll, "final_perplexity": math.exp(nll), "final_rouge_l": 0.1,
                        "final_exact_match": 0, "baseline_token_nll": 1.1, "harmful_update_rate": 0,
                        "baseline_perplexity": math.exp(1.1), "baseline_rouge_l": 0.1,
                        "baseline_exact_match": 0, "baseline_generation_limit_rate": 0,
                        "late_harmful_update_rate": 0, "late_event_count": 15, "acceptance_rate": 1,
                        "measured_event_count": 64,
                        "runtime_seconds": 1, "peak_cuda_memory_gib": 1, "final_generation_limit_rate": 0},
            "data_diagnostics": {"generation": {"selected_source_ids": {
                "clients": list(range(512)), "gradient": list(range(512, 536)),
                "gate": list(range(536, 584)), "monitor": list(range(584, 632)),
                "evaluation": list(range(632, 888))}}},
        }
        selected = payload["data_diagnostics"]["generation"]["selected_source_ids"]
        payload["data_diagnostics"]["generation"]["selected_group_ids"] = {
            key: [str(i) for i in values] for key, values in selected.items()}
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload))
        for stage, loss in (("baseline", 1.1), ("final", nll)):
            pd.DataFrame({"nll_sum": [loss] * 256, "response_tokens": [1] * 256,
                          "response_nll": [loss] * 256, "reference": ["answer"] * 256,
                          "rouge_l": [0.1] * 256, "exact_match": [0] * 256,
                          "generated_tokens": [2] * 256, "hit_generation_limit": [False] * 256,
                          "source_id": list(range(632, 888))}).to_csv(path.parent / f"{stage}_eval_details.csv", index=False)
        pd.DataFrame({"event": range(72), "measured": [False] * 8 + [True] * 64,
                      "current_loss": [1.] * 72, "accepted_loss": [1.] * 72,
                      "staleness": [0] * 57 + [4] * 15, "update_accepted": [1] * 72,
                      "client_id": [0] * 72, "base_version": [0] * 72, "arrival_version": [0] * 72,
                      "harmful_update": [False] * 72, "late_harmful_update": [False] * 72,
                      }).to_csv(path.parent / "events.csv", index=False)
    assert analyze(tmp_path)["status"] == "PASS_WEEK9_NLL_GATE_ONLY"
    assert analyze(tmp_path, target="rift")["status"] == "EXPLORATORY_TARGET_ONLY"
    original = path.read_text()
    payload["metrics"]["late_event_count"] = 999
    path.write_text(json.dumps(payload))
    assert any("late event count" in issue for issue in analyze(tmp_path)["issues"])
    path.write_text(original)
    events_path = path.parent / "events.csv"
    original_events = events_path.read_text()
    events = pd.read_csv(events_path)
    events.loc[20, "current_loss"] = float("nan")
    events.to_csv(events_path, index=False)
    assert any("monitor observations" in issue for issue in analyze(tmp_path)["issues"])
    events_path.write_text(original_events)
    details = pd.read_csv(path.parent / "final_eval_details.csv")
    details["nll_sum"] = 50
    details.to_csv(path.parent / "final_eval_details.csv", index=False)
    report = analyze(tmp_path)
    assert report["status"] == "INCOMPLETE_OR_UNVERIFIED"
    assert any("does not match" in issue for issue in report["issues"])
