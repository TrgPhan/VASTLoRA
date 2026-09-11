import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spectral_v3", ROOT / "scripts/run_spectral_v3_confirmation.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_supplement_preserves_every_task_and_training_setting(tmp_path):
    original = json.loads(RUNNER.SOURCE.read_text(encoding="utf-8"))
    derived = RUNNER.build_matrix()
    for key in ("tasks", "regimes", "runner", "seeds"):
        assert derived[key] == original[key]
    jobs = RUNNER.build_jobs(derived, tmp_path)
    assert len(jobs) == 24
    assert len({j["result"] for j in jobs}) == 24
    for job in jobs:
        task = next(t for t in original["tasks"] if t["name"] == job["task"])
        base = json.loads((ROOT / task["base_config"]).read_text(encoding="utf-8"))
        old = RUNNER.matrix_runner._build_config(base, task, original["regimes"][0], original)
        new = job["config"]
        assert new["model"] == old["model"]
        assert new["dataset"] == old["dataset"]
        changed = {k for k in new["experiment"] if new["experiment"][k] != old["experiment"].get(k)}
        assert changed == {"methods", "spectral_edit_target_modules", "spectral_posthoc_base_method"}
        assert new["experiment"]["spectral_edit_target_modules"] == ["q_proj", "v_proj"]
        assert new["experiment"]["server_update_weight"] == 0.5


@pytest.mark.parametrize("corruption", [None, "seed", "commit", "fingerprint", "csv", "metric"])
def test_resume_rejects_mismatched_or_incomplete_results(tmp_path, monkeypatch, corruption):
    matrix = RUNNER.build_matrix()
    job = RUNNER.build_jobs(matrix, tmp_path)[0]
    module = RUNNER.matrix_runner._RUNNER_MODULE
    monkeypatch.setattr(module, "_git_commit", lambda: "fixed-commit")
    config = job["config"]
    payload = {
        "schema_version": 5, "method": RUNNER.METHOD, "seed": job["seed"],
        "provenance": config["provenance"], "config_fingerprint": module._config_fingerprint(config),
        "git_commit": "fixed-commit", "git_worktree_dirty": False,
        "metrics": {"final_accuracy": 0.75, "final_class_nll": 0.6},
        "spectral_posthoc": {"edit_objective": "answer_token_nll_including_eos"},
    }
    if corruption == "seed":
        payload["seed"] += 1
    if corruption == "commit":
        payload["git_commit"] = "other"
    if corruption == "fingerprint":
        payload["config_fingerprint"] = "other"
    if corruption == "metric":
        payload["metrics"]["final_accuracy"] = 7.5
    path = job["result"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    for name in ("baseline_eval_details", "final_eval_details", "events"):
        count = 40 if name == "events" else 1024
        if corruption == "csv" and name == "final_eval_details":
            count -= 1
        pd.DataFrame({"index": range(count)}).to_csv(path.parent / f"{name}.csv", index=False)
    assert RUNNER.valid_result(job, matrix) is (corruption is None)
