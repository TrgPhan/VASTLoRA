import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle_qwen_7b_fedlora_week9.ipynb"


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source_text(notebook):
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_is_clean_and_all_code_cells_compile():
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_notebook_pins_audited_release_and_is_safe_by_default():
    text = source_text(load_notebook())
    assert "d9c732313ee7b93cf01bf6c22f6a60fcbf4fe465" in text
    assert "MODE = 'preflight'" in text
    assert "RUN_TRAINING = False" in text
    assert "--profile', 'qwen7b-fedlora'" in text
    assert "scripts/run_week9_7b_fedlora.sh" not in text
    assert "torchao" in text
    assert "torch=={torch_version}" in text


def test_notebook_selects_exact_suites_and_one_worker_per_gpu():
    text = source_text(load_notebook())
    assert "'extended': ['fedavg_lora', 'ffa_lora', 'flora_lora']" in text
    assert "'spectral': ['flexlora', 'florist', 'florg']" in text
    assert "GPU_IDS = [0, 1]" in text
    assert "command.extend(['--workers-per-gpu', '1'])" in text
    assert "command.extend(['--gpu', str(gpu)])" in text
    assert "--retry-incomplete" in text
    assert "--resume-root" in text


def test_notebook_preflights_and_validates_artifacts():
    text = source_text(load_notebook())
    for flag in ("--dry-run", "--prepare-only", "--plan-only"):
        assert flag in text
    assert "validate_run" in text
    assert "data/schedule identity differs across paired methods" in text
    assert "final_rouge_l_precision" in text
    assert "late_harmful_update_rate" in text
    assert "test_week8_spectral_integration.py" not in text
    assert "stdout=subprocess.PIPE" in text
    assert "--tb=long" in text
    assert "full output: {log_path}" in text
