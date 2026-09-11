"""Actual pinned Qwen tokenizer contract; no pretrained model weights are loaded."""
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from riftlora.scale import generation
from run_kaggle_3b import _supervised_suffix_logits
from run_week9_generation import load_matrix, specs


@pytest.fixture(scope="module")
def real_setup():
    from transformers import AutoTokenizer
    config = next(specs(load_matrix(), ROOT / "outputs/test_week9_tokenizer"))[2]
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["name"], revision=config["model"]["revision"], local_files_only=True)
    except OSError:
        if os.environ.get("REQUIRE_WEEK9_TOKENIZER") == "1":
            raise
        pytest.skip("Cache the pinned Qwen tokenizer first; Kaggle requires this test")
    return tokenizer, config


def example(response="Paris.", context=""):
    return {"instruction": "What is the capital of France?", "context": context,
            "response": response, "category": "open_qa", "source_id": 0}


def test_pinned_chat_tokenizer_collates_response_only(real_setup):
    tokenizer, config = real_setup
    ds = config["dataset"]
    item = example()
    prompt, target = generation.encode_example(tokenizer, item, ds, config["model"]["max_length"])
    assert isinstance(prompt, list) and len(prompt) > 2
    assert all(type(token) is int for token in prompt)
    assert prompt == tokenizer.apply_chat_template(
        [{"role": "user", "content": item["instruction"]}], tokenize=True,
        add_generation_prompt=True, return_dict=True)["input_ids"]
    assert target[-1] == tokenizer.eos_token_id
    batch = generation.collate_generation(tokenizer, [(item, 0), (example("The city is Paris."), 0)], ds, 256)
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"][0, :len(prompt)].tolist() == prompt
    assert batch["labels"][0, :len(prompt)].eq(-100).all()
    assert batch["labels"][batch["attention_mask"].eq(0)].eq(-100).all()
    assert batch["labels"][0].ne(-100).sum() == len(target)


def test_actual_prompt_length_is_filtered_and_reference_not_leaked(real_setup):
    tokenizer, config = real_setup
    short, long = example(), example(context="This is a long context. " * 200)
    assert generation.prompt_ids(tokenizer, short, config["dataset"]) == generation.prompt_ids(
        tokenizer, example("A different answer entirely."), config["dataset"])
    assert len(generation.prompt_ids(tokenizer, long, config["dataset"])) > config["dataset"]["max_prompt_tokens"]
    with pytest.raises(generation.TokenBudgetError):
        generation.encode_example(tokenizer, long, config["dataset"], 256)
    splits, audit = generation.prepare_records([short, long], tokenizer, config["dataset"], 256)
    assert audit["overlength_rows"] == 1 and sum(map(len, splits.values())) == 1


def test_real_tokenizer_tiny_peft_response_loss_and_gradient(real_setup):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    tokenizer, config = real_setup
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(12)
        model = get_peft_model(Qwen2ForCausalLM(Qwen2Config(
            vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=256, use_cache=False,
        )), LoraConfig(r=2, lora_alpha=2, lora_dropout=0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        model.eval()
        batch = generation.collate_generation(tokenizer, [(example(), 0)], config["dataset"], 256)
        sums, counts = generation.response_loss_values(model, batch, _supervised_suffix_logits)
        with torch.no_grad():
            expected = model(**batch).loss
        torch.testing.assert_close(sums.sum() / counts.sum(), expected)
        (sums.sum() / counts.sum()).backward()
        assert any(p.grad is not None and p.grad.ne(0).any() for p in model.parameters() if p.requires_grad)
    finally:
        torch.set_num_threads(threads)
