"""Regression coverage for same-kernel import and report-only recovery."""
import csv
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from week8_spectral_suite import build_matrix, job_list
import run_week8_classification_matrix as matrix_runner

NOTEBOOK = ROOT / "notebooks/kaggle_qwen_1_5b_week8_heldout_classification.ipynb"


def cell_containing(text):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return next("".join(c["source"]) for c in notebook["cells"]
                if c["cell_type"] == "code" and text in "".join(c["source"]))


def test_import_bootstrap_works_without_processing_editable_install_pth():
    source = cell_containing("import riftlora")
    start = source.index("sys.path.insert(0, str(REPO_DIR / 'src'))")
    end = source.index("from week8_spectral_suite import")
    bootstrap = source[start:end]
    # -S disables site/.pth processing, like the already-running notebook kernel
    # which has not seen a newly installed editable package.
    command = ("import sys\nfrom pathlib import Path\n"
               f"REPO_DIR = Path({str(ROOT)!r})\n" + bootstrap)
    subprocess.run([sys.executable, "-S", "-c", command], check=True, cwd=ROOT.parent)


def recovery_scope(tmp_path):
    matrix = build_matrix(suite="factor_only")
    source = tmp_path / "previous_output"
    source.mkdir()
    out = tmp_path / "new_output"
    out.mkdir()
    jobs = job_list(matrix)[:3]
    return {"matrix": matrix, "REPO_DIR": ROOT, "OUTPUT_ROOT": out, "all_jobs": jobs,
            "jobs": jobs[:], "METHODS": tuple(matrix["methods"]), "RESUME_ROOTS": [source],
            "REQUIRE_RESUME": True, "SHARD_INDEX": 0, "Path": Path, "json": json,
            "csv": csv, "shutil": shutil}


def fake_artifacts(directory, job, *, complete):
    task, regime, method, seed = job
    path = directory / task / regime / method / f"{method}_seed{seed}"
    path.mkdir(parents=True)
    (path / "result.json").write_text(json.dumps(dict(task=task, regime=regime, method=method, seed=seed)))
    for name, count in (("events.csv", 40 if complete else 39),
                        ("baseline_eval_details.csv", 1024), ("final_eval_details.csv", 1024)):
        with (path / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index"])
            writer.writerows([[i] for i in range(count)])


def test_recovery_imports_valid_runs_and_only_queues_missing_or_incomplete(tmp_path, monkeypatch):
    scope = recovery_scope(tmp_path)
    jobs = scope["jobs"][:]
    fake_artifacts(scope["RESUME_ROOTS"][0], jobs[0], complete=True)
    fake_artifacts(scope["RESUME_ROOTS"][0], jobs[1], complete=False)
    # Config/commit validity has independent tests; isolate artifact completeness here.
    monkeypatch.setattr(matrix_runner, "_completed_result_matches", lambda *a, **k: True)
    exec(compile(cell_containing("def completed_job("), str(NOTEBOOK), "exec"), scope)
    assert scope["completed_jobs"] == [jobs[0]]
    assert scope["jobs"] == jobs[1:]
    assert not scope["result_path_for"](jobs[1]).exists()
    pending = json.loads((scope["OUTPUT_ROOT"] / "pending_jobs_shard0.json").read_text())
    assert pending == [list(j) for j in jobs[1:]]


def test_missing_resume_input_blocks_unintended_training(tmp_path):
    scope = recovery_scope(tmp_path)
    with pytest.raises(RuntimeError, match="No valid previous results"):
        exec(compile(cell_containing("def completed_job("), str(NOTEBOOK), "exec"), scope)


def test_empty_queue_does_not_cache_model_or_start_workers():
    scope = {"RUN_TRAINING": True, "jobs": []}
    for marker in ("from huggingface_hub import snapshot_download", "subprocess.Popen("):
        exec(compile(cell_containing(marker), str(NOTEBOOK), "exec"), scope)
    assert "torch" not in scope and "snapshot_download" not in scope


def test_launcher_skips_complete_and_only_forces_incomplete_results(tmp_path, monkeypatch):
    import torch
    jobs = job_list(build_matrix(suite="factor_only"))[:3]
    broken = tmp_path / "incomplete.json"
    broken.write_text("{}")
    commands = []

    class Process:
        returncode = 0
        def poll(self):
            return 0

    def popen(command, **kwargs):
        commands.append(command)
        return Process()

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    scope = {"RUN_TRAINING": True, "jobs": jobs, "GPU_IDS": [0, 1], "OUTPUT_ROOT": tmp_path,
             "METHODS": ("fedavg_lora", "ffa_lora"), "SHARD_INDEX": 0, "REPO_DIR": ROOT,
             "MATRIX": tmp_path / "matrix.json", "RUNNER": ROOT / "scripts/run_week8_classification_matrix.py",
             "completed_job": lambda job: job == jobs[0],
             "result_path_for": lambda job: broken if job == jobs[1] else tmp_path / "missing.json",
             "sys": sys, "os": os, "json": json,
             "subprocess": types.SimpleNamespace(Popen=popen, STDOUT=subprocess.STDOUT)}
    exec(compile(cell_containing("subprocess.Popen("), str(NOTEBOOK), "exec"), scope)
    assert len(commands) == 2
    assert [c[c.index("--seed") + 1] for c in commands] == [str(j[3]) for j in jobs[1:]]
    assert "--force" in commands[0] and "--force" not in commands[1]


def test_recovery_keeps_training_commit_and_defaults_to_report_only():
    source = cell_containing("REPO_REF =")
    assert "REPO_REF = 'db4ca69024fdd0697f7ed667efbf10f025c558a3'" in source
    assert "RUN_TRAINING = False" in source
    assert "REQUIRE_RESUME = True" in source
