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


@pytest.mark.parametrize("method", ["raw", "freshness", "alignfed_calibration", "rift", "rift_diag", "rift_core"])
def test_generation_runs_shared_methods_with_real_peft(monkeypatch, tiny_model, method, tmp_path):
    from datasets import Dataset, DatasetDict
    data = DatasetDict(train=Dataset.from_list([row(i) for i in range(40)]),
                       validation=Dataset.from_list([row(i) for i in range(100, 104)]))
    monkeypatch.setattr(generation, "load_instruction_data", lambda _: (data, {}))
    monkeypatch.setattr(shared, "_load_model", lambda _: (Tokenizer(), tiny_model))
    c = config()
    shared._validate_config(c, method)
    result = shared.run_experiment(c, method=method, seed=9001)
    m = result["metrics"]
    assert m["final_accuracy"] is None and m["final_class_nll"] is None
    assert math.exp(m["final_token_nll"]) == pytest.approx(m["final_perplexity"])
    details = result["final_eval_details"]
    assert m["final_token_nll"] == pytest.approx(sum(r["nll_sum"] for r in details) / sum(r["response_tokens"] for r in details))
    assert len(result["events"]) == 6
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
