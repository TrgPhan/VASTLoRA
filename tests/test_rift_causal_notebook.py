import ast
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_rift_causal_notebook import build


@pytest.fixture(scope="module")
def notebook():
    return build()


def test_cells_compile_and_no_training_defaults(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
    settings = ast.parse("".join(notebook["cells"][1]["source"]))
    values = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in settings.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
    }
    assert values["RUN_MODE"] == "preflight"
    assert values["RUN_TRAINING"] is False
    assert values["HELDOUT_AUDIT_NOTE"] == ""
    assert values["REQUIRE_SMOKE"] is False


def test_bundle_integrity_and_dependency_closure(notebook, tmp_path):
    ns = {}
    exec("".join(notebook["cells"][2]["source"]), ns)
    payload = zlib.decompress(base64.b64decode(ns["RELEASE_B64"]))
    assert hashlib.sha256(payload).hexdigest() == ns["RELEASE_SHA256"]
    files = json.loads(payload)
    for name, content in files.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert (ROOT / name).read_text(encoding="utf-8") == content
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for name in [
        "src/riftlora/scale/server_adaptation.py",
        "src/riftlora/scale/causal_study.py",
        "scripts/run_kaggle_3b.py",
        "scripts/run_rift_causal_study.py",
        "scripts/analyze_rift_causal_study.py",
        "tests/test_paper_baseline_integration.py",
        "tests/test_rift_causal_data_preflight.py",
        "configs/rift_core_causal_study.json",
        "configs/local_1_5b_rift_development.json",
        "configs/rift_core_heldout_confirmation_matrix.json",
    ]:
        assert name in files
    from run_rift_causal_study import source_manifest

    root_manifest = source_manifest(ROOT)
    bundled_manifest = source_manifest(tmp_path)
    assert bundled_manifest
    assert all(root_manifest[name] == digest for name, digest in bundled_manifest.items())
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "scripts/run_rift_causal_study.py"),
            "--phase",
            "smoke",
            "--task",
            "qnli",
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["jobs"] == 6


def test_resume_and_gpu_guards_are_in_notebook(notebook):
    source = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert "CUDA_VISIBLE_DEVICES=str(gpu)" in source
    assert "start_new_session=True" in source and "os.killpg" in source
    assert "valid_result(smoke_job, path)" in source
    assert "if REQUIRE_SMOKE and phase != 'smoke'" in source
    assert "REQUIRE_SMOKE=True" in source
    assert "RIFT_REAL_DATA_PREFLIGHT='1'" in source
    assert (
        "'baseline_eval_details.csv'" in source and "'final_eval_details.csv'" in source
    )
    assert "--force" not in source
