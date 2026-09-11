import ast
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle_qwen_1_5b_rift_week9_generation.ipynb"


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_week9_notebook_cells_compile_and_contain_no_fabricated_outputs():
    for cell in code_cells():
        compile("".join(cell["source"]), str(NOTEBOOK), "exec")
        assert cell["execution_count"] is None and cell["outputs"] == []


def test_week9_notebook_defaults_to_no_training_with_frozen_checkout():
    settings = ast.parse("".join(code_cells()[0]["source"]))
    literals = {}
    for statement in settings.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant):
            literals[statement.targets[0].id] = statement.value.value
    assert literals["RUN_TRAINING"] is False
    assert literals["MODE"] == "smoke"
    assert literals["CONFIRM_PROTOCOL_FROZEN"] is False
    assert re.fullmatch(r"[0-9a-f]{40}", literals["REPO_REF"])
    frozen = subprocess.check_output(["git", "show", literals["REPO_REF"] + ":scripts/run_week9_generation.py"], cwd=ROOT)
    assert b"--retry-incomplete" in frozen and b"--plan-only" in frozen
    for phase in ("development", "confirmation"):
        matrix = json.loads(subprocess.check_output([
            "git", "show", literals["REPO_REF"] + f":configs/week9_generation_{phase}_matrix.json"
        ], cwd=ROOT))
        assert matrix["name"].endswith("-v4")
        assert len(matrix["methods"]) == 7
        assert {"spectral_surgery", "alignfed_calibration"}.issubset(matrix["gates"]["baselines"])
    generation = subprocess.check_output(
        ["git", "show", literals["REPO_REF"] + ":src/riftlora/scale/generation.py"], cwd=ROOT)
    assert b"return_dict=False" in generation
    subprocess.check_output(
        ["git", "show", literals["REPO_REF"] + ":tests/test_week9_real_tokenizer.py"], cwd=ROOT)
    source = "\n".join("".join(cell["source"]) for cell in code_cells())
    assert 'REQUIRE_WEEK9_TOKENIZER="1"' in source
    assert "tests/test_week9_real_tokenizer.py" in source
    assert '"week9_v4_"' in source


def test_week9_training_and_weight_download_are_guarded():
    for cell in code_cells():
        source = "".join(cell["source"])
        if "subprocess.Popen(" in source or "snapshot_download(" in source:
            node = ast.parse(source).body[0]
            assert isinstance(node, ast.If) and isinstance(node.test, ast.Name)
            assert node.test.id == "RUN_TRAINING"


def mode_profiles():
    settings = ast.parse("".join(code_cells()[0]["source"]))
    return next(ast.literal_eval(node.value) for node in settings.body
                if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "MODE_PROFILES")


def notebook_command(profile, output):
    source = ast.parse("".join(code_cells()[2]["source"]))
    statements = [node for node in source.body
                  if isinstance(node, (ast.Assign, ast.AugAssign))
                  and isinstance(target := (node.targets[0] if isinstance(node, ast.Assign)
                                            else node.target), ast.Name)
                  and target.id == "base_command"]
    scope = {"sys": sys, "RUNNER": ROOT / "scripts/run_week9_generation.py",
             "OUTPUT_ROOT": output, "PROFILE": profile}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(NOTEBOOK), "exec"), scope)
    return scope["base_command"]


@pytest.mark.parametrize("mode,count", [
    ("preflight", 42), ("smoke", 7), ("pilot", 7),
    ("development", 42), ("confirmation", 84),
])
def test_notebook_mode_commands_plan_real_jobs_without_training(mode, count, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_week9_generation as runner

    profile = mode_profiles()[mode]
    command = notebook_command(profile, tmp_path)
    assert all(isinstance(arg, str) for arg in command)
    assert profile["jobs"] == count
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("must not train"))
    monkeypatch.setattr(sys, "argv", command[2:] + ["--plan-only"])
    runner.main()
    plan = json.loads((tmp_path / "job_plan.json").read_text())
    manifest = json.loads((tmp_path / "matrix.json").read_text())
    assert len(plan) == count
    assert {job["method"] for job in plan} == set(manifest["methods"])
    if mode == "pilot":
        assert {job["seed"] for job in plan} == {9101}
        assert {job["regime"] for job in plan} == {"noniid_high_staleness"}
        assert manifest["seeds"] == [9101, 9102, 9103]
        assert len(manifest["regimes"]) == 2
    for job in plan:
        args = job["command"]
        config = json.loads(Path(args[args.index("--config") + 1]).read_text())
        assert config["model"]["name"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_pilot_and_development_share_unchanged_cohort(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_week9_generation as runner

    profiles = mode_profiles()
    assert profiles["pilot"]["root"] == profiles["development"]["root"]
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("must not train"))
    plans, manifests = {}, {}
    for mode in ("pilot", "development"):
        command = notebook_command(profiles[mode], tmp_path)
        monkeypatch.setattr(sys, "argv", command[2:] + ["--plan-only"])
        runner.main()
        plans[mode] = json.loads((tmp_path / "job_plan.json").read_text())
        manifests[mode] = json.loads((tmp_path / "matrix.json").read_text())
    assert manifests["pilot"] == manifests["development"]
    assert all(job in plans["development"] for job in plans["pilot"])


@pytest.mark.parametrize("mode,train,frozen,error", [
    ("unknown", False, False, ValueError),
    ("preflight", True, False, ValueError),
    ("confirmation", True, False, RuntimeError),
    ("confirmation", True, True, None),
    ("smoke", True, False, None),
])
def test_notebook_settings_guards(mode, train, frozen, error):
    tree = ast.parse("".join(code_cells()[0]["source"]))
    guards = [node for node in tree.body if isinstance(node, ast.If)]
    scope = {"MODE": mode, "RUN_TRAINING": train,
             "CONFIRM_PROTOCOL_FROZEN": frozen, "MODE_PROFILES": mode_profiles()}
    code = compile(ast.Module(body=guards, type_ignores=[]), str(NOTEBOOK), "exec")
    if error:
        with pytest.raises(error):
            exec(code, scope)
    else:
        exec(code, scope)
