from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_kaggle_3b.py"
SPEC = importlib.util.spec_from_file_location("run_kaggle_3b", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_qnli_prompt_and_label_texts() -> None:
    config = {
        "hub_path": "nyu-mll/glue",
        "subset": "qnli",
        "task": "qnli",
        "question_column": "question",
        "sentence_column": "sentence",
        "label_column": "label",
        "label_texts": [" yes", " no"],
    }
    item = {
        "question": "What city is the Eiffel Tower in?",
        "sentence": "The Eiffel Tower is located in Paris.",
        "label": 0,
    }

    prompt = MODULE._prompt_for_example(item, config)

    assert "Question: What city" in prompt
    assert "Sentence: The Eiffel Tower" in prompt
    assert MODULE._label_texts(config) == [" yes", " no"]


def test_sst2_prompt_uses_sentence() -> None:
    config = {
        "hub_path": "nyu-mll/glue",
        "subset": "sst2",
        "task": "sst2",
        "text_column": "sentence",
        "label_column": "label",
    }

    prompt = MODULE._prompt_for_example({"sentence": "warm and smart"}, config)

    assert "Review: warm and smart" in prompt
    assert MODULE._label_texts(config) == [" negative", " positive"]


def test_mnli_prompt_uses_three_labels_and_pair_columns() -> None:
    config = {
        "hub_path": "nyu-mll/glue",
        "subset": "mnli",
        "task": "mnli",
        "premise_column": "premise",
        "hypothesis_column": "hypothesis",
        "label_column": "label",
        "label_texts": [" entailment", " neutral", " contradiction"],
    }
    item = {
        "premise": "A person is reading a book.",
        "hypothesis": "Someone is reading.",
        "label": 0,
    }

    prompt = MODULE._prompt_for_example(item, config)

    assert "Premise: A person is reading" in prompt
    assert "Hypothesis: Someone is reading" in prompt
    assert MODULE._label_texts(config) == [
        " entailment",
        " neutral",
        " contradiction",
    ]


def test_mnli_evaluator_supports_three_class_scores() -> None:
    class FakeTokenizer:
        eos_token = "<eos>"
        eos_token_id = 4
        pad_token_id = 0

        def __call__(self, text, *, add_special_tokens):
            token = {
                " entailment": 2,
                " neutral": 3,
                " contradiction": 5,
                "<eos>": 4,
            }.get(text, 1)
            if isinstance(token, int):
                ids = [token]
            else:
                ids = token
            return {"input_ids": ([1] + ids) if add_special_tokens else ids}

    class FakeModel:
        def eval(self):
            return self

        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def __call__(self, *, input_ids, attention_mask):
            shape = (*input_ids.shape, 8)
            return SimpleNamespace(logits=torch.zeros(shape))

    config = {
        "hub_path": "nyu-mll/glue",
        "subset": "mnli",
        "task": "mnli",
        "premise_column": "premise",
        "hypothesis_column": "hypothesis",
        "label_column": "label",
        "label_texts": [" entailment", " neutral", " contradiction"],
    }
    dataset = [
        {"premise": "p1", "hypothesis": "h1", "label": 0},
        {"premise": "p2", "hypothesis": "h2", "label": 1},
        {"premise": "p3", "hypothesis": "h3", "label": 2},
    ]

    metrics, details = MODULE.evaluate_classification(
        FakeModel(),
        FakeTokenizer(),
        dataset,
        dataset_config=config,
        max_length=16,
        batch_size=3,
    )

    assert metrics["binary_nll"] is None
    assert 0.0 <= metrics["brier"] <= 2.0
    assert len(details) == 3
    assert "prob_label_2" in details[0]


def test_classification_uses_label_loss_not_eos_loss() -> None:
    class FakeTokenizer:
        eos_token = "<eos>"
        eos_token_id = 4
        pad_token_id = 0

        def __call__(self, text, *, add_special_tokens):
            if add_special_tokens:
                return {"input_ids": [1]}
            tokens = {" yes<eos>": [2, 4], " no<eos>": [3, 4]}
            return {"input_ids": tokens[text]}

    class FakeModel:
        def eval(self):
            return self

        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def __call__(self, *, input_ids, attention_mask):
            logits = torch.zeros((*input_ids.shape, 8))
            # Label 0 is better at the label position, while its EOS is worse.
            logits[:, 0, 2] = 10.0
            logits[:, 0, 3] = 8.0
            for row in range(input_ids.shape[0]):
                logits[row, 1, 4] = 10.0 if input_ids[row, 1].item() == 3 else -10.0
            return SimpleNamespace(logits=logits)

    config = {
        "hub_path": "nyu-mll/glue",
        "subset": "qnli",
        "task": "qnli",
        "question_column": "question",
        "sentence_column": "sentence",
        "label_column": "label",
        "label_texts": [" yes", " no"],
    }
    metrics, details = MODULE.evaluate_classification(
        FakeModel(),
        FakeTokenizer(),
        [{"question": "q", "sentence": "s", "label": 0}],
        dataset_config=config,
        max_length=16,
        batch_size=1,
    )

    assert metrics["binary_nll"] is not None
    assert details[0]["predicted_label"] == 0
