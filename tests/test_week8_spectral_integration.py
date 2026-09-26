"""Offline real tiny-Qwen classification, plus the exact Kaggle job contract."""
import json
import math
import os
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_kaggle_3b as runner
from run_week8_classification_matrix import _build_config
from week8_spectral_suite import METHODS, build_matrix, job_list
from riftlora.baselines import attach_florg_adapters


class Tokenizer:
    eos_token, eos_token_id, pad_token_id = "<eos>", 2, 0

    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens:
            return {"input_ids": [1, 3, 4]}
        text = text.strip().replace("<eos>", "")
        index = {"negative": 5, "positive": 6, "yes": 5, "no": 6,
                 "true": 5, "unknown": 6, "false": 7}.get(text, 5)
        return {"input_ids": [index]}


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("task", ["sst2", "qnli", "mnli_m", "mnli_mm"])
@pytest.mark.parametrize("method", METHODS)
def test_classification_runner_real_tiny_qwen(monkeypatch, task, method):
    from datasets import Dataset, DatasetDict
    import datasets
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, get_peft_model

    matrix = build_matrix()
    spec = next(t for t in matrix["tasks"] if t["name"] == task)
    base = json.loads((ROOT / spec["base_config"]).read_text())
    cfg = _build_config(base, spec, matrix["regimes"][0], matrix)
    cfg["model"].update(load_in_4bit=False, max_length=16)
    cfg["dataset"].update(max_train_examples=12, eval_examples=6, eval_offset=0,
                          eval_split="validation", reserve_eval_from_train=False,
                          validation_split="validation")
    cfg["experiment"].update(num_clients=2, client_ranks=[1, 2], server_max_rank=2,
                              adaptive_max_rank=2, reference_rank=1, compute_times=[1, 3],
                              warmup_returns=1, collected_returns=3, local_steps=1,
                              calibration_gradient_examples=3, calibration_gate_examples=3,
                              monitor_examples=6, eval_batch_size=2)
    nlabels = len(cfg["dataset"]["label_texts"])
    def data(start, n):
        return Dataset.from_list([{"sentence": f"text {i}", "question": f"question {i}",
                                   "premise": f"premise {i}", "hypothesis": f"hypothesis {i}",
                                   "label": i % nlabels} for i in range(start, start + n)])
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: DatasetDict(train=data(0, 48), validation=data(100, 6)))
    torch.manual_seed(3)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                       max_position_embeddings=64, use_cache=False))
    if method == "florg":
        model.requires_grad_(False)
        attach_florg_adapters(model, target_modules=["q_proj", "v_proj"], rank=2, seed=3)
        monkeypatch.setattr(runner, "_load_florg_model", lambda c: (Tokenizer(), model))
    else:
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"],
                                                 lora_dropout=0., task_type="CAUSAL_LM"))
        monkeypatch.setattr(runner, "_load_model", lambda c: (Tokenizer(), model))
    runner._validate_config(cfg, method)
    result = runner.run_experiment(cfg, method=method, seed=6101)
    assert result["method"] == method and "async" in result["competitor_fidelity"]
    assert len(result["events"]) == 4 and len(result["final_eval_details"]) == 6
    assert math.isfinite(result["metrics"]["final_class_nll"])
    assert result["metrics"]["measured_event_count"] == 3
    if method != "florg":
        measured = result["events"][1:]
        assert all(e["spectral_rank_after_cap"] <= 4 for e in measured)
        assert all(e["route"] == method + "_async_full_state" for e in measured)
    json.dumps(result, allow_nan=False)


def test_suite_preserves_confirmation_and_excludes_other_methods():
    matrix = build_matrix()
    original = json.loads((ROOT / "configs/rift_core_heldout_confirmation_matrix.json").read_text())
    for field in ("tasks", "regimes", "seeds", "runner"):
        assert matrix[field] == original[field]
    for key, value in original["experiment"].items():
        assert matrix["experiment"][key] == value
    jobs = job_list(matrix)
    assert len(jobs) == len(set(jobs)) == 72
    assert {j[2] for j in jobs} == set(METHODS)
    assert len(job_list(build_matrix(smoke=True))) == 3
    matrix["methods"].append("rift_core")
    with pytest.raises(ValueError):
        job_list(matrix)


def test_notebook_compiles_and_only_uses_suite_methods():
    path = ROOT / "notebooks/kaggle_qwen_1_5b_week8_heldout_classification.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            source = "".join(c["source"])
            compile(source, str(path), "exec")
            assert c["outputs"] == [] and c["execution_count"] is None
            sources.append(source)
    all_source = "\n".join(sources)
    assert "RUN_MODE = 'confirmation'" in all_source
    assert "RUN_TRAINING = False" in all_source
    assert "jobs = all_jobs[SHARD_INDEX::SHARD_COUNT]" in all_source
    assert "all_jobs = job_list(matrix)" in all_source
    assert "selected_methods" not in all_source and "core_methods" not in all_source
    assert "--force" not in all_source


def test_notebook_launcher_builds_only_72_flat_commands(monkeypatch, tmp_path):
    import subprocess
    path = ROOT / "notebooks/kaggle_qwen_1_5b_week8_heldout_classification.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in nb["cells"]
                  if c["cell_type"] == "code" and "subprocess.Popen(" in "".join(c["source"]))
    commands = []
    class Process:
        returncode = 0
        def poll(self):
            return 0
    def popen(command, **kwargs):
        assert all(isinstance(arg, str) for arg in command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] in {"0", "1"}
        commands.append(command)
        return Process()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    scope = {"RUN_TRAINING": True, "GPU_IDS": [0, 1], "OUTPUT_ROOT": tmp_path,
             "jobs": job_list(build_matrix()), "METHODS": METHODS, "SHARD_INDEX": 0,
             "sys": sys, "os": os, "json": json, "MATRIX": tmp_path / "matrix.json",
             "RUNNER": ROOT / "scripts/run_week8_classification_matrix.py", "REPO_DIR": ROOT,
             "subprocess": types.SimpleNamespace(Popen=popen, STDOUT=subprocess.STDOUT)}
    exec(compile(source, str(path), "exec"), scope)
    assert len(commands) == 72
    assert {c[c.index("--method") + 1] for c in commands} == set(METHODS)
    assert all("--force" not in c for c in commands)


def test_completion_rejects_old_commit_and_accepts_same_run(tmp_path, monkeypatch):
    import run_week8_classification_matrix as matrix_runner
    matrix = build_matrix()
    spec, regime = matrix["tasks"][0], matrix["regimes"][0]
    base = json.loads((ROOT / spec["base_config"]).read_text())
    cfg = _build_config(base, spec, regime, matrix)
    cfg["provenance"] = {"matrix_sha256": matrix_runner._matrix_fingerprint(matrix)}
    monkeypatch.setattr(matrix_runner, "_runner_git_commit", lambda: "tested-release")
    result = {"schema_version": 5, "method": "florist", "seed": 6101,
              "provenance": cfg["provenance"], "git_commit": "tested-release",
              "git_worktree_dirty": False, "config_fingerprint": runner._config_fingerprint(cfg)}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    assert matrix_runner._completed_result_matches(path, config=cfg, method="florist", seed=6101, matrix=matrix)
    result["git_commit"] = "old-release"
    path.write_text(json.dumps(result))
    assert not matrix_runner._completed_result_matches(path, config=cfg, method="florist", seed=6101, matrix=matrix)
