from __future__ import annotations

import importlib.util
from pathlib import Path


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
