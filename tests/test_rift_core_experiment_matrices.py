from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_PATH = ROOT / "configs" / "rift_core_remaining_tasks_local_4gb_matrix.json"
CONFIRMATION_PATH = ROOT / "configs" / "rift_core_heldout_confirmation_matrix.json"
CORE_DEVELOPMENT_PATH = ROOT / "configs" / "rift_core_development_matrix.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_map(matrix: dict) -> dict[str, dict]:
    return {task["name"]: task for task in matrix["tasks"]}


def _window(task: dict) -> set[int]:
    start = int(task["eval_offset"])
    return set(range(start, start + int(task["eval_examples"])))


def test_remaining_development_matrix_only_runs_missing_tasks() -> None:
    matrix = _load(DEVELOPMENT_PATH)

    assert set(_task_map(matrix)) == {"sst2", "mnli_mm"}
    assert matrix["seeds"] == [5101, 5102, 5103]
    assert matrix["methods"] == [
        "spectral_filter",
        "alignfed_calibration",
        "rift",
        "rift_diag",
        "rift_core",
    ]


def test_confirmation_covers_every_week8_task_and_primary_control() -> None:
    matrix = _load(CONFIRMATION_PATH)

    assert set(_task_map(matrix)) == {"sst2", "qnli", "mnli_m", "mnli_mm"}
    assert len(matrix["seeds"]) == 6
    assert set(matrix["methods"]) == {
        "raw",
        "freshness",
        "fedrot",
        "spectral_surgery",
        "alignfed_calibration",
        "rift",
        "rift_diag",
        "rift_core",
    }
    assert matrix["runner"] == {"script": "scripts/run_kaggle_3b.py", "buffer_size": 1, "schedule_mode": "async"}
    assert matrix["experiment"]["fedrot_align_schedule"] == "alternating"
    assert matrix["experiment"]["spectral_surgery"]["preserve_energy"] == "l1"


def test_confirmation_windows_do_not_overlap_core_development() -> None:
    original_development = _task_map(_load(CORE_DEVELOPMENT_PATH))
    remaining_development = _task_map(_load(DEVELOPMENT_PATH))
    confirmation = _task_map(_load(CONFIRMATION_PATH))

    for task_name, confirmation_task in confirmation.items():
        development_task = remaining_development.get(
            task_name, original_development[task_name]
        )
        assert _window(confirmation_task).isdisjoint(_window(development_task))


def test_core_protocol_is_frozen_across_remaining_and_confirmation() -> None:
    development = _load(DEVELOPMENT_PATH)
    confirmation = _load(CONFIRMATION_PATH)
    fixed_fields = (
        "calibration_gradient_examples",
        "calibration_gate_examples",
        "monitor_examples",
        "harm_epsilon",
        "component_score_objective",
        "calibration_gate_objective",
        "monitor_objective",
        "rift_component_gain_mass",
        "rift_gate_min_staleness",
        "rift_gate_selection",
        "rift_gate_confidence_z",
        "rift_max_mean_increase",
        "rift_step_scales",
        "rift_include_freshness_fallback",
        "rift_core",
    )

    for field in fixed_fields:
        assert confirmation["experiment"][field] == development["experiment"][field]

    assert confirmation["experiment"]["warmup_returns"] >= development["experiment"]["warmup_returns"]
    assert confirmation["experiment"]["collected_returns"] >= development["experiment"]["collected_returns"]


def test_scaled_evaluation_protocol_is_large_and_disjoint() -> None:
    development = _task_map(_load(CORE_DEVELOPMENT_PATH))
    confirmation = _task_map(_load(CONFIRMATION_PATH))

    assert all(task["eval_examples"] >= 512 for task in development.values())
    assert all(task["eval_examples"] >= 1024 for task in confirmation.values())
    assert confirmation["sst2"]["eval_split"] == "train"
    assert confirmation["sst2"]["reserve_eval_from_train"] is True
    assert confirmation["sst2"]["eval_shuffle_seed"] == 271828
