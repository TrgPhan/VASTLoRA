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
from week9_artifacts import EVAL_COLUMNS, EVENT_COLUMNS


def test_paired_interval_not_standard_deviation_of_unpaired_scores():
    mean, low, high = paired_interval([0.1] * 6)
    assert mean == pytest.approx(0.1)
    assert low == pytest.approx(mean) and high == pytest.approx(mean)
    assert paired_interval([0.1]) == (None, None, None)


def test_incomplete_confirmation_cannot_pass(tmp_path):
    (tmp_path / "matrix.json").write_text(json.dumps(load_matrix("confirmation")))
    report = analyze(tmp_path)
    assert report["status"] == "INCOMPLETE_OR_UNVERIFIED"
    assert len(report["missing"]) == 84


def _write_cohort(tmp_path, first_only=False, spectral_nll=1.0):
    matrix = load_matrix("confirmation")
    (tmp_path / "matrix.json").write_text(json.dumps(matrix))
    for method, seed, config, path in specs(matrix, tmp_path):
        nll = 0.8 if method == "rift_core" else 1.0
        if method == "spectral_surgery":
            nll = spectral_nll
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
        if first_only:
            break
    return matrix, path


def test_spectral_control_participates_in_confirmation_verdict(tmp_path):
    _write_cohort(tmp_path, spectral_nll=0.6)
    report = analyze(tmp_path)
    assert not report["issues"] and not report["missing"]
    assert report["status"] == "NO_GO_NLL"
    spectral_pairs = [p for p in report["pairs"] if p["baseline"] == "spectral_surgery"]
    assert len(spectral_pairs) == 2
    assert all(p["paired_seeds"] == 6 and not p["nll_noninferior"] for p in spectral_pairs)
    assert all(p["nll_noninferior"] for p in report["pairs"] if p not in spectral_pairs)


def test_complete_paired_cohort_then_corrupted_details(tmp_path):
    _, path = _write_cohort(tmp_path)
    assert analyze(tmp_path)["status"] == "PASS_WEEK9_NLL_GATE_ONLY"
    assert analyze(tmp_path, target="rift")["status"] == "EXPLORATORY_TARGET_ONLY"
    original = path.read_text()
    payload = json.loads(original)
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


@pytest.mark.parametrize("filename,column", [
    *(("events.csv", c) for c in EVENT_COLUMNS),
    *((f"{stage}_eval_details.csv", c) for stage in ("baseline", "final") for c in EVAL_COLUMNS),
])
def test_missing_csv_columns_report_issue_and_allow_explicit_retry(tmp_path, monkeypatch, filename, column):
    from run_week9_generation import checked_result, IncompleteRunError
    matrix, path = _write_cohort(tmp_path, first_only=True)
    csv = path.parent / filename
    pd.read_csv(csv).drop(columns=[column]).to_csv(csv, index=False)
    report = analyze(tmp_path)
    assert report["status"] == "INCOMPLETE_OR_UNVERIFIED"
    assert not report["rows"]
    assert any(filename in issue and "missing required columns" in issue and column in issue
               for issue in report["issues"])
    monkeypatch.setattr(runner, "_completed_result_matches", lambda *args, **kwargs: True)
    job = next(specs(matrix, tmp_path))
    with pytest.raises(IncompleteRunError, match="missing required columns"):
        checked_result(job, matrix)


@pytest.mark.parametrize("filename", ["events.csv", "final_eval_details.csv"])
@pytest.mark.parametrize("contents", ["", '"unterminated'])
def test_empty_or_malformed_csv_is_an_issue_not_a_crash(tmp_path, filename, contents):
    _, path = _write_cohort(tmp_path, first_only=True)
    (path.parent / filename).write_text(contents)
    report = analyze(tmp_path)
    assert report["status"] == "INCOMPLETE_OR_UNVERIFIED"
    assert report["issues"] and not report["rows"]


@pytest.mark.parametrize("target", ["rift_core", "rift_diag", "rift"])
def test_export_replaces_previous_runs_when_no_valid_rows_remain(tmp_path, monkeypatch, target):
    import analyze_week9_generation as analyzer
    _, path = _write_cohort(tmp_path, first_only=True)
    monkeypatch.setattr(sys, "argv", ["analyze", "--input-dir", str(tmp_path), "--target", target])
    analyzer.main()
    output = tmp_path / f"analysis_{target}"
    assert len(pd.read_csv(output / "runs.csv")) == 1
    csv = path.parent / "events.csv"
    pd.read_csv(csv).drop(columns=["measured"]).to_csv(csv, index=False)
    analyzer.main()
    runs = pd.read_csv(output / "runs.csv")
    assert runs.empty and list(runs.columns) == list(analyzer.RUN_COLUMNS)
    verdict = json.loads((output / "verdict.json").read_text())
    assert not verdict["rows"] and verdict["issues"]
