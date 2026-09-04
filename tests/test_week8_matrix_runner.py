from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_week8_classification_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("run_week8_classification_matrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_matrix_fingerprint_is_key_order_invariant() -> None:
    left = {"name": "matrix", "seeds": [1, 2], "gates": {"a": 1, "b": 2}}
    right = {"gates": {"b": 2, "a": 1}, "seeds": [1, 2], "name": "matrix"}

    assert MODULE._matrix_fingerprint(left) == MODULE._matrix_fingerprint(right)


def test_completed_result_must_match_schema_matrix_and_config(tmp_path: Path) -> None:
    MODULE._RUNNER_MODULE = SimpleNamespace(
        _config_fingerprint=lambda config: config["test_fingerprint"]
    )
    config = {
        "test_fingerprint": "cfg-1",
        "provenance": {"matrix_sha256": "matrix-1"},
    }
    matrix = {"required_schema_version": 3}
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "method": "rift",
                "seed": 4101,
                "provenance": {"matrix_sha256": "matrix-1"},
                "config_fingerprint": "cfg-1",
                "git_worktree_dirty": False,
            }
        ),
        encoding="utf-8",
    )

    assert MODULE._completed_result_matches(
        result_path,
        config=config,
        method="rift",
        seed=4101,
        matrix=matrix,
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["config_fingerprint"] = "stale"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    assert not MODULE._completed_result_matches(
        result_path,
        config=config,
        method="rift",
        seed=4101,
        matrix=matrix,
    )

    payload["config_fingerprint"] = "cfg-1"
    payload["git_worktree_dirty"] = True
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    assert not MODULE._completed_result_matches(
        result_path,
        config=config,
        method="rift",
        seed=4101,
        matrix=matrix,
    )
