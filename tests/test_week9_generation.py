import json
import math
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_kaggle_3b as shared
from run_week9_generation import load_matrix, specs
from riftlora.scale import generation


class Tokenizer:
    pad_token_id, eos_token_id = 0, 2
    chat_template = "test-template"
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert tokenize and add_generation_prompt and not return_dict and messages[0]["role"] == "user"
        return [1, 3, 4]
    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": [1, 3, 4] if add_special_tokens else [5] * len(text.split())}
    def decode(self, ids, **kwargs):
        return "answer"


def row(i, response="answer"):
    return {"instruction": f"question {i}", "context": "", "response": response,
            "category": "open_qa" if i % 2 else "closed_qa", "label": i % 2,
            "source_id": i, "group_id": str(i)}


@pytest.fixture
def tiny_model():
    from peft import get_peft_model, LoraConfig
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(9)
    config = Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=64, use_cache=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield get_peft_model(Qwen2ForCausalLM(config), LoraConfig(
            r=2, lora_alpha=2, lora_dropout=0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    finally:
        torch.set_num_threads(threads)


def config():
    c = next(specs(load_matrix(smoke=True), ROOT / "outputs/test_week9"))[2]
    c["model"].update(load_in_4bit=False, max_length=16)
    c["dataset"].update(max_train_examples=8, eval_examples=4, max_prompt_tokens=8,
                        max_response_tokens=8, max_new_tokens=8)
    c["experiment"].update(num_clients=2, client_ranks=[2, 2], server_max_rank=2,
                            adaptive_max_rank=2, reference_rank=1, compute_times=[1, 4])
    return c


def test_no_prompt_leakage_and_response_nll_matches_full_teacher_forcing(tiny_model):
    c = config()
    batch = generation.collate_generation(Tokenizer(), [(row(0), 0), (row(1, "longer answer here"), 1)], c["dataset"], 16)
    assert torch.all(batch["labels"][:, :3] == -100)
    assert (batch["labels"] == 2).sum() == 2
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    tiny_model.eval()
    full = tiny_model(**batch).loss
    sums, counts = generation.response_loss_values(tiny_model, batch, shared._supervised_suffix_logits)
    torch.testing.assert_close(sums.sum() / counts.sum(), full)
    expected = generation.response_mean_loss(tiny_model, batch, shared._supervised_suffix_logits)
    grads = torch.autograd.grad(expected, tuple(p for p in tiny_model.parameters() if p.requires_grad))
    assert any(torch.count_nonzero(g) for g in grads)


def test_group_split_dedup_and_length_filter_are_seed_independent():
    rows = [row(i) for i in range(400)]
    rows[1]["context"] = rows[0]["context"] = "shared context"
    rows += [dict(rows[0]), row(500, " ".join(["x"] * 20))]
    ds = config()["dataset"]
    splits, audit = generation.prepare_records(rows, Tokenizer(), ds, 16)
    assert audit["duplicate_prompts"] == 1
    assert audit["overlength_rows"] == 1
    sets = [set(r["group_id"] for r in group) for group in splits.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    where = {r["source_id"]: name for name, group in splits.items() for r in group}
    assert where[0] == where[1]
    assert audit == generation.prepare_records(rows, Tokenizer(), ds, 16)[1]


@pytest.mark.parametrize("method", ["raw", "freshness", "fedavg_lora", "fedex_lora", "flora_lora", "ffa_lora", "alignfed_calibration", "spectral_surgery", "rift", "rift_diag", "rift_core"])
def test_generation_runs_shared_methods_with_real_peft(monkeypatch, tiny_model, method, tmp_path):
    from datasets import Dataset, DatasetDict
    data = DatasetDict(train=Dataset.from_list([row(i) for i in range(40)]),
                       validation=Dataset.from_list([row(i) for i in range(100, 104)]))
    monkeypatch.setattr(generation, "load_instruction_data", lambda _: (data, {}))
    monkeypatch.setattr(shared, "_load_model", lambda _: (Tokenizer(), tiny_model))
    c = config()
    calls = []
    if method == "spectral_surgery":
        original = shared.reweight_compact_spectra
        def record_edit(updates, sensitivities, **kwargs):
            assert kwargs["config"].policy == "smooth_abs"
            assert kwargs["config"].preserve_energy == "l1"
            edited = original(updates, sensitivities, **kwargs)
            for name in updates:
                torch.testing.assert_close(edited[name].s.sum(), updates[name].s.sum())
            calls.append("spectral")
            return edited
        monkeypatch.setattr(shared, "reweight_compact_spectra", record_edit)
    elif method == "alignfed_calibration":
        original_gate = shared._whole_update_gate_state
        def record_gate(*args, **kwargs):
            assert kwargs["experiment"]["alignfed_calibration_scales"] == [1., .5, .25, .125]
            assert kwargs["experiment"]["calibration_gate_objective"] == "label_nll"
            calls.append("alignfed")
            return original_gate(*args, **kwargs)
        monkeypatch.setattr(shared, "_whole_update_gate_state", record_gate)
    shared._validate_config(c, method)
    result = shared.run_experiment(c, method=method, seed=9001)
    m = result["metrics"]
    assert m["final_accuracy"] is None and m["final_class_nll"] is None
    assert math.exp(m["final_token_nll"]) == pytest.approx(m["final_perplexity"])
    details = result["final_eval_details"]
    assert m["final_token_nll"] == pytest.approx(sum(r["nll_sum"] for r in details) / sum(r["response_tokens"] for r in details))
    assert len(result["events"]) == 6
    if method in {"spectral_surgery", "alignfed_calibration"}:
        assert len(calls) == c["experiment"]["collected_returns"]
    selected = result["data_diagnostics"]["generation"]["selected_source_ids"]
    assert len(set().union(*map(set, selected.values()))) == sum(map(len, selected.values()))
    # Exercise the artifact checker against the actual runner, not just fabricated rows.
    import pandas as pd
    from week9_artifacts import validate_run
    for key in ("events", "baseline_eval_details", "final_eval_details"):
        pd.DataFrame(result.pop(key)).to_csv(tmp_path / f"{key}.csv", index=False)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    validate_run(path, config=c, method=method, seed=9001, matrix=load_matrix(smoke=True))
    json.dumps(result, allow_nan=False)


def test_generation_runs_real_florg_backend(monkeypatch, tiny_model, tmp_path):
    from datasets import Dataset, DatasetDict
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from riftlora.baselines import attach_florg_adapters

    data = DatasetDict(
        train=Dataset.from_list([row(i) for i in range(40)]),
        validation=Dataset.from_list([row(i) for i in range(100, 104)]),
    )
    monkeypatch.setattr(generation, "load_instruction_data", lambda _: (data, {}))

    def load_florg(config):
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, use_cache=False,
        ))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        attach_florg_adapters(model, target_modules=["q_proj", "v_proj"], rank=2, seed=4)
        return Tokenizer(), model

    monkeypatch.setattr(shared, "_load_florg_model", load_florg)
    c = config()
    shared._validate_config(c, "florg")
    result = shared.run_experiment(c, method="florg", seed=9001)
    assert result["method"] == "florg"
    assert len(result["events"]) == 6
    assert result["metrics"]["final_token_nll"] > 0
    assert len(result["final_eval_details"]) == 4
    import pandas as pd
    from week9_artifacts import validate_run
    for key in ("events", "baseline_eval_details", "final_eval_details"):
        pd.DataFrame(result.pop(key)).to_csv(tmp_path / f"{key}.csv", index=False)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    validate_run(path, config=c, method="florg", seed=9001, matrix=load_matrix(smoke=True))


@pytest.mark.parametrize("phase", ["development", "confirmation"])
def test_week9_uses_final_board_controls_and_parameters(phase):
    week8 = json.loads((ROOT / "configs/rift_core_heldout_confirmation_matrix.json").read_text())
    base = json.loads((ROOT / "configs/local_1_5b_rift_development.json").read_text())
    matrix = load_matrix(phase)
    assert "spectral_surgery" in matrix["methods"]
    assert "alignfed_calibration" in matrix["methods"]
    assert not {"spectral_surgery_posthoc", "alignfed_reference", "spectral_filter"} & set(matrix["methods"])
    for baseline in ("spectral_surgery", "alignfed_calibration"):
        assert baseline in matrix["gates"]["baselines"]
    for method, _, c, _ in specs(matrix, ROOT / "outputs/test_week9"):
        assert c["experiment"]["spectral_surgery"] == week8["experiment"]["spectral_surgery"]
        assert c["experiment"]["alignfed_calibration_scales"] == base["experiment"]["alignfed_calibration_scales"]
        shared._validate_config(c, method)


def test_generation_spectral_requires_per_example_sensitivity():
    c = config()
    c["experiment"]["calibration_gradient_batch_size"] = 2
    with pytest.raises(ValueError, match="per-example"):
        shared._validate_config(c, "spectral_surgery")


def test_generation_refuses_classification_objective():
    c = config()
    c["experiment"]["component_score_objective"] = "class_nll"
    with pytest.raises(ValueError, match="label_nll"):
        shared._validate_config(c, "rift_core")


def test_chat_uses_instruction_without_reference_and_keeps_model_stop_ids(tiny_model):
    class InspectTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "question 1\n\nContext:\ncontext"}]
            return super().apply_chat_template(messages, **kwargs)
    item = {**row(1, "DO NOT LEAK"), "context": "context"}
    assert generation.encode_example(InspectTokenizer(), item, config()["dataset"], 16)[0] == [1, 3, 4]
    tiny_model.generation_config.eos_token_id = [2, 7]
    assert generation.stop_token_ids(tiny_model, Tokenizer()) == [2, 7]


def test_broken_chat_tokenizer_fails_before_length_filter():
    tokenizer = Tokenizer()
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        generation.prepare_records([row(1)], tokenizer, config()["dataset"], 16)


def test_unexpected_chat_return_type_is_not_counted_as_overlength():
    from transformers import BatchEncoding
    class BrokenTokenizer(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return BatchEncoding({"input_ids": [1, 3, 4], "attention_mask": [1, 1, 1]})
    with pytest.raises(TypeError, match="flat list"):
        generation.prepare_records([row(1)], BrokenTokenizer(), config()["dataset"], 16)


def test_group_reservations_are_disjoint_and_reproducible():
    from datasets import Dataset
    data = Dataset.from_list([{**row(i), "group_id": str(i // 3)} for i in range(120)])
    kwargs = dict(reserve_fn=shared._reserve_calibration_splits, split_sizes=(8, 8, 8),
                  max_train_examples=20, label_column="label", seed=19, stratified=True)
    clients, calibration = generation.reserve_grouped_splits(data, **kwargs)
    sets = [set(d["group_id"]) for d in [clients, *calibration]]
    assert len(set.union(*sets)) == sum(map(len, sets))
    clients2, calibration2 = generation.reserve_grouped_splits(data, **kwargs)
    assert clients["source_id"] == clients2["source_id"]
    assert [d["source_id"] for d in calibration] == [d["source_id"] for d in calibration2]


@pytest.mark.parametrize("last_token,hit_limit", [(7, False), (5, True)])
def test_generation_limit_respects_alternative_model_stop_token(monkeypatch, tiny_model, last_token, hit_limit):
    tiny_model.generation_config.eos_token_id = [2, 7]
    def generate(input_ids, **kwargs):
        assert kwargs["eos_token_id"] == [2, 7]
        assert kwargs["max_new_tokens"] == 8
        return torch.cat([input_ids, torch.tensor([[5] * 7 + [last_token]])], dim=1)
    monkeypatch.setattr(tiny_model, "generate", generate)
    metrics, _ = generation.evaluate_generation(tiny_model, Tokenizer(), [row(1)],
        dataset_config=config()["dataset"], max_length=16, batch_size=1,
        suffix_logits=shared._supervised_suffix_logits)
    assert metrics["generation_limit_rate"] == float(hit_limit)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_generation_budget(budget):
    c = config()
    c["dataset"]["max_new_tokens"] = budget
    with pytest.raises(ValueError, match="positive integer"):
        shared._validate_config(c, "raw")


@pytest.mark.parametrize("method", ["raw", "spectral_surgery", "alignfed_calibration", "rift_core"])
def test_learning_curve_does_not_change_training_and_saves_valid_checkpoints(monkeypatch, tiny_model, method, tmp_path):
    import copy
    import random
    import numpy as np
    import pandas as pd
    from datasets import Dataset, DatasetDict
    from week9_artifacts import validate_milestone
    data = DatasetDict(train=Dataset.from_list([row(i) for i in range(40)]),
                       validation=Dataset.from_list([row(i) for i in range(100, 104)]))
    monkeypatch.setattr(generation, "load_instruction_data", lambda _: (data, {}))
    untouched = copy.deepcopy(tiny_model)
    monkeypatch.setattr(shared, "_load_model", lambda _: (Tokenizer(), copy.deepcopy(untouched)))
    original_eval = shared.evaluate_task
    def noisy_eval(*args, **kwargs):
        result = original_eval(*args, **kwargs)
        random.random()
        np.random.random()
        torch.rand(3)
        return result
    monkeypatch.setattr(shared, "evaluate_task", noisy_eval)
    c = config()
    baseline = shared.run_experiment(copy.deepcopy(c), method=method, seed=9001)
    c["experiment"]["generation_eval_returns"] = [2, 5]
    artifact = tmp_path / method
    with_curve = shared.run_experiment(c, method=method, seed=9001, artifact_dir=artifact)
    pd.testing.assert_frame_equal(pd.DataFrame(baseline["events"]), pd.DataFrame(with_curve["events"]), check_exact=True)
    assert baseline["final_eval_details"] == with_curve["final_eval_details"]
    assert with_curve["metrics"]["client_sample_presentations"] == 6
    assert with_curve["metrics"]["client_response_tokens"] == 12
    assert [r["measured_returns"] for r in with_curve["development_learning_curve"]] == [2, 5]
    for budget in (2, 5):
        directory = artifact / "milestones" / f"returns_{budget:04d}"
        entry, _ = validate_milestone(directory, config=c, method=method, seed=9001)
        assert entry["total_returns"] == budget + 1
        state = torch.load(directory / "adapter.pt", weights_only=True)
        assert state["kind"] == "evaluation_adapter_only" and state["adapter"]
    # An independent shorter run must match the first checkpoint of the long trajectory.
    shorter = copy.deepcopy(c)
    shorter["experiment"].update(collected_returns=2, generation_eval_returns=[])
    short_result = shared.run_experiment(shorter, method=method, seed=9001)
    assert short_result["metrics"]["final_nll"] == with_curve["development_learning_curve"][0]["metrics"]["nll"]
    import analyze_week9_learning_curve as curves
    (tmp_path / "matrix.json").write_text(json.dumps(load_matrix(smoke=True)))
    monkeypatch.setattr(curves, "specs", lambda *_: [(method, 9001, c, artifact / "result.json")])
    report = curves.analyze(tmp_path)
    assert len(report["rows"]) == 2 and not report["missing"]
    assert len(report["review_rows"]) == 8
    assert "method" not in report["review_rows"][0]
    if method == "raw":
        monkeypatch.setattr(sys, "argv", ["analyze", "--input-dir", str(tmp_path)])
        curves.main()
        assert (tmp_path / "learning_curve_analysis/learning_curves.png").exists()
        review_path = tmp_path / "learning_curve_analysis/correctness_template.csv"
        review = pd.read_csv(review_path, keep_default_na=False)
        review.loc[0, "notes"] = "keep annotation"
        review.to_csv(review_path, index=False)
        curves.main()
        assert pd.read_csv(review_path, keep_default_na=False).loc[0, "notes"] == "keep annotation"
        import evaluate_week9_checkpoint as checkpoint_eval
        monkeypatch.setattr(sys, "argv", ["eval", "--checkpoint-dir", str(artifact / "milestones/returns_0002"), "--dry-run"])
        checkpoint_eval.main()
    details_file = artifact / "milestones/returns_0002/eval_details.csv"
    details_file.write_text("corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_milestone(details_file.parent, config=c, method=method, seed=9001)


def test_learning_curve_cannot_read_confirmation_test():
    c = config()
    c["dataset"]["eval_split"] = "test"
    c["experiment"]["generation_eval_returns"] = [5]
    with pytest.raises(ValueError, match="development validation"):
        shared._validate_config(c, "raw")


@pytest.mark.parametrize("milestones", [[5, 2], [2, 2, 5], [True, 5], [2, 4], [0, 5]])
def test_invalid_learning_curve_milestones(milestones):
    c = config()
    c["experiment"]["generation_eval_returns"] = milestones
    with pytest.raises(ValueError, match="milestones"):
        shared._validate_config(c, "raw")
