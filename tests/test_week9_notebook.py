import ast
import json
from pathlib import Path
import re
import subprocess


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
    assert literals["CONFIRM_PROTOCOL_FROZEN"] is False
    assert re.fullmatch(r"[0-9a-f]{40}", literals["REPO_REF"])
    frozen = subprocess.check_output(["git", "show", literals["REPO_REF"] + ":scripts/run_week9_generation.py"], cwd=ROOT)
    assert b"--retry-incomplete" in frozen and b"--plan-only" in frozen


def test_week9_training_and_weight_download_are_guarded():
    for cell in code_cells():
        source = "".join(cell["source"])
        if "subprocess.Popen(" in source or "snapshot_download(" in source:
            node = ast.parse(source).body[0]
            assert isinstance(node, ast.If) and isinstance(node.test, ast.Name)
            assert node.test.id == "RUN_TRAINING"
