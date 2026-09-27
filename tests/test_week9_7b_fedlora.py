import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_kaggle_3b as shared
import run_week9_generation as runner


METHODS = [
    "fedavg_lora",
    "ffa_lora",
    "flora_lora",
    "flexlora",
    "florist",
    "florg",
]


def test_7b_fedlora_profile_is_matched_to_convergence_protocol(tmp_path):
    cohort = runner.load_matrix(profile="qwen7b-fedlora")
    reference = runner.load_matrix()
    assert cohort["phase"] == "development"
    assert cohort["methods"] == METHODS
    assert cohort["seeds"] == [9101, 9102]
    assert [regime["name"] for regime in cohort["regimes"]] == ["noniid_high_staleness"]
    assert cohort["primary_target"] == "flexlora"
    cohort_config = list(runner.specs(cohort, tmp_path / "fedlora"))[0][2]
    reference_config = list(runner.specs(reference, tmp_path / "reference"))[0][2]
    for key in (
        "warmup_returns",
        "local_steps",
        "local_learning_rate",
        "server_update_weight",
        "server_max_rank",
        "calibration_gradient_examples",
        "calibration_gate_examples",
        "monitor_examples",
    ):
        assert cohort_config["experiment"][key] == reference_config["experiment"][key]
    assert cohort_config["experiment"]["collected_returns"] == 512
    assert cohort_config["experiment"]["generation_eval_returns"] == [64, 128, 256, 512]
    assert cohort["tasks"][0]["eval_examples"] == 128
    assert cohort["tasks"][0]["prompt_format"] == "chat_concise_v1"
    assert cohort["tasks"][0]["eligibility_prompt_formats"] == ["chat_v1", "chat_concise_v1"]

    jobs = list(runner.specs(cohort, tmp_path / "fedlora"))
    assert len(jobs) == 12
    assert {(method, seed) for method, seed, _, _ in jobs} == {
        (method, seed) for method in METHODS for seed in (9101, 9102)
    }
    for method, _, config, _ in jobs:
        assert config["model"]["name"] == "Qwen/Qwen2.5-7B-Instruct"
        assert config["model"]["load_in_4bit"] is True
        shared._validate_config(config, method)


def test_7b_fedlora_profile_refuses_confirmation_and_excludes_inexact_fedex():
    with pytest.raises(ValueError, match="requires development"):
        runner.load_matrix("confirmation", profile="qwen7b-fedlora")
    cohort = runner.load_matrix(profile="qwen7b-fedlora")
    assert "fedex_lora" not in cohort["methods"]
    assert any("FedEx-LoRA is intentionally excluded" in note for note in cohort["notes"])


def test_7b_fedlora_smoke_keeps_all_six_operators(tmp_path):
    smoke = runner.load_matrix(smoke=True, profile="qwen7b-fedlora")
    assert smoke["phase"] == "smoke"
    assert smoke["methods"] == METHODS
    assert smoke["seeds"] == [9001]
    assert smoke["experiment"]["collected_returns"] == 5
    assert len(list(runner.specs(smoke, tmp_path / "smoke"))) == 6


def test_remote_launcher_is_isolated_and_resumable():
    script = (ROOT / "scripts/run_week9_7b_fedlora.sh").read_text(encoding="utf-8")
    assert "profile=qwen7b-fedlora" in script
    assert "week9_7b_fedlora.lock" in script
    assert "--retry-incomplete" in script
    assert "WEEK9_FEDLORA_WORKERS_PER_GPU" in script
    assert "--target flexlora" in script
    assert "fedex_lora" not in script


def test_recipe_is_json_and_documents_source_boundary():
    recipe = json.loads((ROOT / "configs/week9_generation_7b_fedlora.json").read_text())
    assert recipe["methods"] == METHODS
    assert recipe["experiment_overrides"]["florist_energy"] == 0.9
    assert any("immediate-async adaptations" in note for note in recipe["notes"])
