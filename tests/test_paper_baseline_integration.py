"""Offline tiny-Qwen tests of real PEFT factors, autograd and runner wiring."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from riftlora.scale.alignfed_peft import PenultimateFeatures, features_at_factors, snapshot_factors
from riftlora.scale.alignfed_reference import fit_semantic_transform

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("alignfed_runner_test", ROOT / "scripts/run_alignfed_reference.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class TinyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens:
            return {"input_ids": [1, 3, 4]}
        label = 5 if "negative" in text else 6
        ids = [label] if text != "<eos>" else []
        if "<eos>" in text:
            ids.append(2)
        return {"input_ids": ids}


@pytest.fixture
def tiny_model():
    peft = pytest.importorskip("peft")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(101)
    config = Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                         use_cache=False, attention_dropout=0.0)
    model = peft.get_peft_model(Qwen2ForCausalLM(config), peft.LoraConfig(
        r=2, lora_alpha=2, lora_dropout=0, target_modules=["q_proj", "v_proj", "o_proj", "down_proj"],
        task_type=peft.TaskType.CAUSAL_LM,
    ))
    model.eval()
    return model


def _batch():
    return {"input_ids": torch.tensor([[1, 3, 4, 5, 2]]), "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "labels": torch.tensor([[-100, -100, -100, 5, 2]])}


def test_real_peft_functional_features_keep_original_parameters_and_backpropagate(tiny_model):
    source = snapshot_factors(tiny_model)
    extractor = PenultimateFeatures(tiny_model)
    expected = extractor(_batch()).detach()
    actual = features_at_factors(extractor, source, _batch())
    torch.testing.assert_close(actual, expected)
    deltas = {k: torch.randn_like(v) * 0.01 for k, v in source.items()}
    sides = {k: "a" if ".lora_A." in k else "b" for k in source}
    transform = fit_semantic_transform(
        source, [deltas], expected.mean(0), lambda state: features_at_factors(extractor, state, _batch()).mean(0),
        factor_sides=sides, steps=1, learning_rate=0.01,
    )
    edited = transform(deltas)
    features = features_at_factors(extractor, {k: source[k] + edited[k] for k in source}, _batch())
    gradients = torch.autograd.grad((features - expected).square().sum(), tuple(transform.parameters()), allow_unused=True)
    assert any(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in gradients)
    for k, v in snapshot_factors(tiny_model).items():
        torch.testing.assert_close(v, source[k])
    assert all(not module._forward_hooks for module in tiny_model.get_base_model().model.layers)


def test_suffix_answer_loss_and_gradients_match_full_teacher_forcing(tiny_model):
    batch = _batch()
    full = tiny_model(**batch).loss
    parameters = tuple(p for p in tiny_model.parameters() if p.requires_grad)
    full_gradients = torch.autograd.grad(full, parameters)
    suffix = RUNNER.shared._answer_token_nll_loss(tiny_model, batch)
    suffix_gradients = torch.autograd.grad(suffix, parameters)
    torch.testing.assert_close(suffix, full)
    for actual, expected in zip(suffix_gradients, full_gradients):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def _config():
    matrix = json.loads((ROOT / "configs/paper_baseline_audit_matrix.json").read_text())
    config = RUNNER.build_config(matrix, "sst2")
    config["model"].update(max_length=12, load_in_4bit=False,
                           target_modules=["q_proj", "v_proj", "o_proj", "down_proj"])
    config["dataset"].update(max_train_examples=4, eval_examples=4)
    config["experiment"].update(num_clients=2, client_ranks=[2, 2], compute_times=[1., 2.],
                                server_max_rank=2, reference_rank=1, adaptive_max_rank=2,
                                warmup_returns=1, collected_returns=4, local_steps=2,
                                calibration_gradient_examples=2, calibration_gate_examples=2,
                                monitor_examples=2, eval_batch_size=1)
    config["alignfed_reference"].update(max_pending=2, max_wait=1.5, steps=1)
    return config


@pytest.mark.parametrize("method", ["alignfed_reference", "factor_fedbuff"])
def test_buffered_runner_completes_real_tiny_model_and_serializes(monkeypatch, tiny_model, method):
    from datasets import Dataset
    data = Dataset.from_list([{"sentence": f"text {i}", "label": i % 2} for i in range(4)])
    monkeypatch.setattr(RUNNER, "prepare_data", lambda c, s: (data, data, data.select([0, 1]),
                                                             data.select([2, 3]), [[0, 2], [1, 3]]))
    monkeypatch.setattr(RUNNER.shared, "_load_model", lambda c: (TinyTokenizer(), tiny_model))
    result = RUNNER.run(_config(), method=method, seed=7201)
    assert result["metrics"]["return_count"] == 5
    assert sum(row["buffer_returns"] for row in result["events"]) == 5
    assert [row["version"] for row in result["events"]] == list(range(1, len(result["events"]) + 1))
    assert result["protocol"]["harm_unit"] == "aggregation buffer"
    assert len(result["final_eval_details"]) == 4
    assert 0 <= result["metrics"]["final_accuracy"] <= 1
    json.dumps(result, allow_nan=False)


def test_posthoc_runner_edits_once_after_training_and_records_pre_edit(monkeypatch, tiny_model):
    import datasets
    from datasets import Dataset, DatasetDict
    data = Dataset.from_list([{"sentence": f"text {i}", "label": i % 2} for i in range(20)])
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: DatasetDict(train=data, validation=data.select([0, 1, 2, 3])))
    monkeypatch.setattr(RUNNER.shared, "_load_model", lambda c: (TinyTokenizer(), tiny_model))
    original_edit = RUNNER.shared.edit_trained_adapter
    edits = []

    def edit(*args, **kwargs):
        for batch, weight in args[2]:
            assert batch["labels"].eq(TinyTokenizer.eos_token_id).any()
            assert weight == batch["labels"].ne(-100).sum().item()
        edits.append(1)
        return original_edit(*args, **kwargs)

    monkeypatch.setattr(RUNNER.shared, "edit_trained_adapter", edit)
    result = RUNNER.shared.run_experiment(_config(), method="spectral_surgery_posthoc", seed=7201)
    assert len(edits) == 1
    assert result["spectral_posthoc"]["harmful_metrics_scope"] == "training_returns_before_posthoc_edit"
    assert "accuracy" in result["spectral_posthoc"]["pre_edit_metrics"]
    assert result["spectral_posthoc"]["calibration_examples"] == 2


def test_all_tasks_validate_and_missing_edit_modules_fail():
    matrix = json.loads((ROOT / "configs/paper_baseline_audit_matrix.json").read_text())
    for task in matrix["tasks"]:
        config = RUNNER.build_config(matrix, task["name"])
        RUNNER.validate_config(config)
        RUNNER.shared._validate_config(config, "spectral_surgery_posthoc")
    config["model"]["target_modules"] = ["q_proj", "v_proj"]
    with pytest.raises(ValueError, match="edit targets"):
        RUNNER.shared._validate_config(config, "spectral_surgery_posthoc")
