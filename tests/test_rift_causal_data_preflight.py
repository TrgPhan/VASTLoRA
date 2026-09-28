"""Opt-in real cached GLUE preflight, stopped before any model load/training.

PowerShell: $env:RIFT_REAL_DATA_PREFLIGHT='1'; python -m pytest tests/test_rift_causal_data_preflight.py
This gate intentionally fails if the proposed experiment has overlapping data.
It does not inspect model quality and is not an independent-heldout certificate.
"""

import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_rift_causal_study import build_jobs
import run_kaggle_3b as runner

pytestmark = pytest.mark.skipif(
    os.environ.get("RIFT_REAL_DATA_PREFLIGHT") != "1",
    reason="opt-in real GLUE cache check; no model is loaded",
)

CASES = [
    (phase, task, seed)
    for phase, seeds in (
        ("development", range(8101, 8104)),
        ("confirmation", range(8201, 8207)),
    )
    for task in ("sst2", "qnli", "mnli_m", "mnli_mm")
    for seed in seeds
]


@pytest.mark.parametrize("phase,task,seed", CASES)
def test_real_data_reaches_model_boundary_without_overlap(
    monkeypatch, phase, task, seed
):
    # Set offline mode in the library as well as the environment, because other
    # tests may already have imported datasets/huggingface_hub in this process.
    import datasets.config
    import huggingface_hub.constants

    monkeypatch.setattr(datasets.config, "HF_DATASETS_OFFLINE", True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    spec = json.loads((ROOT / "configs/rift_core_causal_study.json").read_text())
    job = next(
        j
        for j in build_jobs(
            spec, suite="attribution", phase=phase, tasks=[task], variants=["core"]
        )
        if j["seed"] == seed
    )

    class ReachedModelBoundary(Exception):
        pass

    def stop_before_loading(*args, **kwargs):
        raise ReachedModelBoundary

    monkeypatch.setattr(runner, "_load_model", stop_before_loading)
    with pytest.raises(ReachedModelBoundary):
        runner.run_experiment(job["config"], method=job["method"], seed=seed)
