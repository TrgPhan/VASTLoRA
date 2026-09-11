import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_week9_generation as runner


def jobs(tmp_path):
    matrix = runner.load_matrix(smoke=True)
    return matrix, list(runner.specs(matrix, tmp_path))[:2]


def test_matrix_budgets_and_plan_only_do_not_launch(monkeypatch, tmp_path):
    for phase, count, split in (("development", 42, "validation"), ("confirmation", 84, "test")):
        matrix = runner.load_matrix(phase)
        specs = list(runner.specs(matrix, tmp_path))
        assert len(specs) == count
        assert all(s[2]["dataset"]["eval_split"] == split for s in specs)
        assert all(s[2]["dataset"]["prompt_format"] == "chat_v1" for s in specs)
    matrix, specs = jobs(tmp_path)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("must not train"))
    runner.execute_jobs(specs, matrix, tmp_path, gpus=[0, 1], plan_only=True)
    plan = json.loads((tmp_path / "job_plan.json").read_text())
    assert len(plan) == 2
    assert all(all(isinstance(arg, str) for arg in entry["command"]) for entry in plan)


class Process:
    pid = 123
    def __init__(self, code):
        self.code = code
        self.terminated = False
    def poll(self):
        return self.code
    def terminate(self):
        self.terminated = True
        self.code = -15
    def wait(self, **kwargs):
        return self.code


def test_worker_failure_stops_other_job_without_retry(monkeypatch, tmp_path):
    matrix, specs = jobs(tmp_path)
    processes = [Process(None), Process(1)]
    calls = []
    def popen(command, **kwargs):
        calls.append(kwargs)
        return processes[len(calls) - 1]
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="worker failed"):
        runner.execute_jobs(specs, matrix, tmp_path, gpus=[0, 1])
    assert len(calls) == 2
    assert [c["env"]["CUDA_VISIBLE_DEVICES"] for c in calls] == ["0", "1"]
    assert processes[0].terminated
    assert all(c["stdout"].closed for c in calls)


def test_partial_run_never_overwritten_without_explicit_retry(monkeypatch, tmp_path):
    matrix, specs = jobs(tmp_path)
    directory = specs[0][3].parent
    directory.mkdir(parents=True)
    (directory / "events.csv").write_text("partial")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("must not train"))
    with pytest.raises(RuntimeError, match="incomplete run"):
        runner.execute_jobs(specs, matrix, tmp_path, gpus=[0])
    assert (directory / "events.csv").read_text() == "partial"


def test_matching_results_skip_and_max_jobs_only_limits_new_jobs(monkeypatch, tmp_path):
    matrix, specs = jobs(tmp_path)
    specs[0][3].parent.mkdir(parents=True)
    specs[0][3].write_text("{}")
    monkeypatch.setattr(runner, "checked_result", lambda *a, **k: None)
    calls = []
    def popen(command, **kwargs):
        calls.append(command)
        specs[1][3].parent.mkdir(parents=True)
        return Process(0)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner, "validate_run", lambda *a, **k: None)
    runner.execute_jobs(specs, matrix, tmp_path, gpus=[0], max_jobs=1)
    assert len(calls) == 1
    assert (specs[1][3].parent / "launcher.json").exists()


def test_resume_import_validates_before_copy(monkeypatch, tmp_path):
    output, source = tmp_path / "new", tmp_path / "old"
    output.mkdir()
    matrix, specs = jobs(output)
    candidate = source / specs[0][3].relative_to(output)
    candidate.parent.mkdir(parents=True)
    candidate.write_text("invalid")
    def reject(*args, **kwargs):
        raise ValueError("bad provenance")
    monkeypatch.setattr(runner, "checked_result", reject)
    with pytest.raises(ValueError, match="bad provenance"):
        runner.execute_jobs(specs, matrix, output, gpus=[0], resume_roots=[source])
    assert not specs[0][3].exists()


def test_explicit_retry_archives_corrupt_matching_artifacts(monkeypatch, tmp_path):
    matrix, specs = jobs(tmp_path)
    specs = specs[:1]
    result = specs[0][3]
    result.parent.mkdir(parents=True)
    result.write_text("matching identity with corrupt CSV")
    def incomplete(*args, **kwargs):
        raise runner.IncompleteRunError("bad CSV")
    monkeypatch.setattr(runner, "checked_result", incomplete)
    def popen(*args, **kwargs):
        assert not result.parent.exists()
        result.parent.mkdir(parents=True)
        return Process(0)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner, "validate_run", lambda *a, **k: None)
    runner.execute_jobs(specs, matrix, tmp_path, gpus=[0], retry_incomplete=True)
    archived = list((tmp_path / "incomplete").rglob("result.json"))
    assert len(archived) == 1 and archived[0].read_text() == "matching identity with corrupt CSV"
