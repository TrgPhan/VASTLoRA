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
    assert metrics["class_nll"] == torch.log(torch.tensor(3.0)).item()
    assert 0.0 <= metrics["brier"] <= 2.0
    assert len(details) == 3
    assert "class_nll" in details[0]
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


def test_class_nll_objective_compares_all_candidate_labels() -> None:
    class FakeModel:
        def __call__(self, *, input_ids, attention_mask):
            logits = torch.zeros((*input_ids.shape, 8))
            logits[0, 0, 2] = 5.0
            logits[3, 0, 3] = 5.0
            return SimpleNamespace(logits=logits)

    batch = {
        "input_ids": torch.tensor([[1, 2], [1, 3], [1, 2], [1, 3]]),
        "attention_mask": torch.ones((4, 2), dtype=torch.long),
        "labels": torch.tensor([[-100, 2], [-100, 3], [-100, 2], [-100, 3]]),
        "class_labels": torch.tensor([0, 1]),
    }

    losses = MODULE._classification_candidate_nll_values(
        FakeModel(),
        batch,
        eos_token_id=4,
    )

    assert losses.shape == (2,)
    assert torch.all(losses < 0.2)


def test_candidate_label_nll_sparse_path_matches_dense_loss_and_gradient() -> None:
    class FakeModel:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits

        def __call__(self, *, input_ids, attention_mask):
            return SimpleNamespace(logits=self.logits)

    labels = torch.tensor(
        [
            [-100, -100, 2, 7],
            [-100, 3, 7, -100],
            [-100, -100, -100, 4],
            [-100, 5, 6, 7],
        ]
    )
    class_labels = torch.tensor([0, 1])
    batch = {
        "input_ids": torch.ones_like(labels),
        "attention_mask": torch.ones_like(labels),
        "labels": labels,
        "class_labels": class_labels,
    }
    sparse_logits = torch.randn(4, 4, 11, requires_grad=True)
    sparse = MODULE._classification_candidate_label_nlls(
        FakeModel(sparse_logits), batch, eos_token_id=7
    )
    sparse.sum().backward()
    sparse_gradient = sparse_logits.grad.detach().clone()

    dense_logits = sparse_logits.detach().clone().requires_grad_(True)
    shifted_logits = dense_logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    token_loss = torch.nn.functional.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        ignore_index=-100,
        reduction="none",
    )
    mask = shifted_labels.ne(-100) & shifted_labels.ne(7)
    dense = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    dense.sum().backward()

    assert torch.allclose(sparse.flatten(), dense)
    assert torch.allclose(sparse_gradient, dense_logits.grad)


def test_label_histogram_is_stable_and_string_keyed() -> None:
    assert MODULE._label_histogram([1, 0, 1, 2, 0, 1]) == {
        "0": 2,
        "1": 3,
        "2": 1,
    }


def test_pair_task_truncation_preserves_prompt_head_and_tail() -> None:
    prompt = list(range(10))

    assert MODULE._truncate_prompt_ids(prompt, 6, task="qnli") == [0, 1, 2, 7, 8, 9]
    assert MODULE._truncate_prompt_ids(prompt, 6, task="mnli") == [0, 1, 2, 7, 8, 9]
    assert MODULE._truncate_prompt_ids(prompt, 6, task="sst2") == [0, 1, 2, 3, 4, 5]


def test_stratified_calibration_balances_every_reserved_split() -> None:
    labels = [0] * 20 + [1] * 20 + [2] * 20

    segments, remaining = MODULE._stratified_segment_indices(
        labels,
        [6, 9, 12],
        seed=7,
    )

    assert [len(segment) for segment in segments] == [6, 9, 12]
    for segment in segments:
        counts = MODULE._label_histogram([labels[index] for index in segment])
        assert max(counts.values()) - min(counts.values()) <= 1
    flattened = [index for segment in segments for index in segment]
    assert len(set(flattened)) == len(flattened)
    assert set(flattened).isdisjoint(remaining)
    assert len(flattened) + len(remaining) == len(labels)


def test_mnli_truth_value_prompt_matches_single_token_verbalizers() -> None:
    config = {
        "hub_path": "nyu-mll/glue",
        "subset": "mnli",
        "task": "mnli",
        "premise_column": "premise",
        "hypothesis_column": "hypothesis",
        "label_column": "label",
        "label_texts": [" true", " unknown", " false"],
        "prompt_style": "truth_value",
    }

    prompt = MODULE._prompt_for_example(
        {"premise": "A person reads.", "hypothesis": "Someone reads."},
        config,
    )

    assert "true if entailed" in prompt
    assert "unknown if neutral" in prompt
    assert "false if contradicted" in prompt


def test_paired_upper_confidence_gate_penalizes_uncertain_updates() -> None:
    deltas = torch.tensor([-0.20, 0.10, -0.10, 0.20])

    mean_only = MODULE._paired_upper_confidence_bound(deltas, confidence_z=0.0)
    risk_bound = MODULE._paired_upper_confidence_bound(deltas, confidence_z=1.0)

    assert mean_only == 0.0
    assert risk_bound > mean_only


def test_rift_gradient_batch_masks_eos_from_classification_objective() -> None:
    class FakeTokenizer:
        eos_token = "<eos>"
        eos_token_id = 4
        pad_token_id = 0

        def __call__(self, text, *, add_special_tokens):
            if add_special_tokens:
                return {"input_ids": [1]}
            return {"input_ids": [2, 4] if "negative" in text else [3, 4]}

    class FakeModel:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

    config = {
        "hub_path": "nyu-mll/glue",
        "subset": "sst2",
        "task": "sst2",
        "text_column": "sentence",
        "label_column": "label",
        "label_texts": [" negative", " positive"],
    }

    batch = MODULE._make_classification_batch(
        FakeModel(),
        FakeTokenizer(),
        [{"sentence": "bad", "label": 0}],
        dataset_config=config,
        max_length=16,
    )

    assert 2 in batch["labels"]
    assert 4 not in batch["labels"]
