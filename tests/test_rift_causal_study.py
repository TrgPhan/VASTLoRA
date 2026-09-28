import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from riftlora.lowrank import CompactSVD
from riftlora.scale.causal_study import (
    attach_source_ids,
    exclude_reserved_windows,
    experiment_audit,
)
from riftlora.scale.core_repair import CoreRepairConfig
from riftlora.scale.peft_bridge import empty_adapter_state
from riftlora.scale.server_adaptation import ServerAdaptationConfig, fit_server_adapter

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "causal_runner_test", ROOT / "scripts/run_rift_causal_study.py"
)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
import run_kaggle_3b as runner


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 2, bias=False)})
        self.scaling = {"default": 2.0}
        nn.init.zeros_(self.base.weight)

    def forward(self, x):
        return self.base(x) + 2 * self.lora_B["default"](self.lora_A["default"](x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Layer()

    def forward(self, x):
        return self.layer(x)


def dense(c):
    return (c.u * c.s) @ c.v.T


def test_server_fit_escapes_zero_adapter_restores_state_and_scales_once():
    model = Model()
    model.layer.base.eval()
    model.layer.base.weight.requires_grad_(False)
    p = model.layer.lora_A["default"].weight
    p.grad = torch.ones_like(p)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    flags = [p.requires_grad for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    rng = torch.get_rng_state().clone()
    current = empty_adapter_state(model)
    batch = {"x": torch.eye(2), "y": torch.eye(2)}
    loss = lambda m, b: (m(b["x"]) - b["y"]).square().mean()
    common = dict(
        loss_fn=loss,
        rank=2,
        seed=123,
        rank_rtol=1e-7,
        config=ServerAdaptationConfig(steps=25, learning_rate=0.05),
    )
    u1, d1 = fit_server_adapter(
        model, current, current, [(batch, 2.0)], server_weight=0.5, **common
    )
    u2, _ = fit_server_adapter(
        model, current, current, [(batch, 2.0)], server_weight=1.0, **common
    )
    torch.testing.assert_close(
        0.5 * dense(u1["layer"]), dense(u2["layer"]), atol=1e-5, rtol=1e-5
    )
    assert torch.norm(dense(u2["layer"])) > 0.1
    assert d1["server_fit_loss_last_pre_step"] < d1["server_fit_loss_initial"]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert flags == [p.requires_grad for p in model.parameters()]
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(p.grad, torch.ones_like(p))
    assert torch.equal(rng, torch.get_rng_state())


def test_server_fit_failure_restores_factors():
    model = Model()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    zero = empty_adapter_state(model)
    with pytest.raises(ValueError, match="nonfinite"):
        fit_server_adapter(
            model,
            zero,
            zero,
            [({}, 1.0)],
            loss_fn=lambda m, b: torch.tensor(float("nan")),
            rank=2,
            server_weight=0.5,
            seed=1,
            rank_rtol=1e-5,
            config=ServerAdaptationConfig(),
        )
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


def study_spec():
    return json.loads((ROOT / "configs/rift_core_causal_study.json").read_text())


def test_job_matrix_complete_and_legacy_config_unchanged():
    before = (ROOT / "configs/rift_core_heldout_confirmation_matrix.json").read_bytes()
    s = study_spec()
    jobs = suite.build_jobs(s, suite="attribution", phase="confirmation")
    assert len(jobs) == 4 * 6 * 6
    assert {j["task"] for j in jobs} == {"sst2", "qnli", "mnli_m", "mnli_mm"}
    for job in jobs:
        e = job["config"]["experiment"]
        assert e["study_fixed_gate_budget"]
        assert (
            e["component_score_objective"]
            == e["calibration_gate_objective"]
            == "class_nll"
        )
        if job["variant"] == "no_repair":
            assert e["rift_core"]["radius"] == 0
            assert e["rift_core"]["steps"] == 3
    assert (
        before
        == (ROOT / "configs/rift_core_heldout_confirmation_matrix.json").read_bytes()
    )
    delay = suite.build_jobs(s, suite="delay", phase="development", tasks=["qnli"])
    assert len(delay) == 4 * 4 * 3
    assert {
        j["config"]["experiment"]["rift_core"].get("radius_mode")
        for j in delay
        if j["variant"] == "core_constant"
    } == {"constant"}
    objective = suite.build_jobs(s, suite="objective", phase="smoke")
    assert all(
        j["config"]["dataset"]["client_objective"] == "class_nll" for j in objective
    )


def test_manifest_rejects_tampering_confirmation_without_audit_and_bad_selectors():
    with pytest.raises(ValueError, match="audit note"):
        suite.freeze(study_spec(), suite="attribution", phase="confirmation")
    with pytest.raises(ValueError, match="unknown task"):
        suite.build_jobs(
            study_spec(), suite="attribution", phase="smoke", tasks=["typo"]
        )
    plan = suite.freeze(
        study_spec(), suite="attribution", phase="smoke", tasks=["sst2"]
    )
    suite.validate_manifest(plan)
    plan["jobs"][0]["seed"] += 1
    with pytest.raises(ValueError, match="edited"):
        suite.validate_manifest(plan)


def test_sst2_reserves_both_windows_in_both_phases():
    from datasets import Dataset

    raw = attach_source_ids(
        {
            "train": Dataset.from_list(
                [{"sentence": str(i), "label": i % 2} for i in range(40)]
            )
        }
    )
    data = dict(
        train_split="train",
        validation_split="validation",
        eval_split="train",
        eval_shuffle_seed=271828,
        study_reserved_train_windows=[[5, 4], [20, 4]],
        eval_examples=4,
    )
    first = exclude_reserved_windows(raw, raw["train"], {**data, "eval_offset": 5})
    second = exclude_reserved_windows(raw, raw["train"], {**data, "eval_offset": 20})
    assert list(first["_rift_source_id"]) == list(second["_rift_source_id"])
    assert len(first) == 32


def test_audit_rejects_content_leakage_even_across_source_splits():
    from datasets import Dataset
    from riftlora.asyncfl.simulator import SimulationTrace

    data = Dataset.from_list([{"sentence": "same", "label": 0}])
    raw = attach_source_ids({"train": data, "validation": data})
    with pytest.raises(ValueError, match="content overlap"):
        experiment_audit(
            {"dataset": {"label_column": "label"}},
            "raw",
            SimulationTrace(()),
            [],
            train=raw["train"],
            evaluation=raw["validation"],
        )


def test_zero_candidate_is_rejection_and_fixed_gate_budget(monkeypatch):
    model = Model()
    zero = empty_adapter_state(model)
    monkeypatch.setattr(
        runner, "_per_example_classification_losses", lambda *a, **k: torch.ones(2)
    )
    experiment = {
        "rift_step_scales": [1, 0.5, 0.25],
        "rift_include_freshness_fallback": False,
        "rift_gate_selection": "min_risk",
        "server_update_weight": 0.5,
        "server_max_rank": 2,
        "rank_rtol": 1e-5,
        "study_fixed_gate_budget": True,
    }
    diagnostics = {}
    _, _, scale, _, route = runner._rift_gate_state(
        model,
        None,
        zero,
        zero,
        zero,
        [],
        dataset_config={},
        max_length=8,
        batch_size=1,
        experiment=experiment,
        freshness=1.0,
        comparator_updates=zero,
        diagnostics=diagnostics,
    )
    assert diagnostics["gate_candidate_count"] == 4
    assert diagnostics["gate_loss_evaluations"] == 5
    assert scale == 0 and route == "reject_noop"


@pytest.mark.parametrize("mode", ["constant", "staleness"])
def test_radius_modes(mode):
    from test_core_repair import Model as CoreModel, run

    batch = {"x": torch.eye(2), "target": torch.zeros(2, 2)}
    result = run(
        CoreModel(),
        [(batch, 2.0)],
        staleness=24,
        config=CoreRepairConfig(radius=0.1, radius_mode=mode),
    )
    assert result.diagnostics["core_radius"] == pytest.approx(
        0.1 if mode == "constant" else 0.05
    )


from test_paper_baseline_integration import TinyTokenizer, _config
from test_paper_baseline_integration import tiny_model as tiny_model


@pytest.mark.parametrize(
    "variant",
    ["core", "diag", "no_repair", "server_lora", "server_only", "core_constant"],
)
@pytest.mark.parametrize("objective", ["label_nll", "class_nll"])
def test_real_tiny_qwen_end_to_end(monkeypatch, tiny_model, variant, objective):
    import datasets
    from datasets import Dataset, DatasetDict

    train = Dataset.from_list(
        [{"sentence": f"train {i}", "label": i % 2} for i in range(20)]
    )
    evaluation = Dataset.from_list(
        [{"sentence": f"eval {i}", "label": i % 2} for i in range(4)]
    )
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, **k: DatasetDict(train=train, validation=evaluation),
    )
    monkeypatch.setattr(runner, "_load_model", lambda c: (TinyTokenizer(), tiny_model))
    config = _config()
    config["dataset"]["client_objective"] = objective
    config["causal_study"] = {"phase": "smoke"}
    config["experiment"].update(
        study_fixed_gate_budget=True,
        collected_returns=2,
        rift_step_scales=[1, 0.5, 0.25],
    )
    v = study_spec()["variants"][variant]
    suite.merge(config["experiment"], v["experiment"])
    calls = []
    train_client = runner._train_client

    def spy(*args, **kwargs):
        calls.append(1)
        return train_client(*args, **kwargs)

    monkeypatch.setattr(runner, "_train_client", spy)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        runner._validate_config(config, v["method"])
        result = runner.run_experiment(config, method=v["method"], seed=8101)
    finally:
        torch.set_num_threads(threads)
    assert result["metrics"]["measured_event_count"] == 2
    assert len(calls) == (0 if variant == "server_only" else 3)
    assert all(
        row["gate_candidate_count"] == 4 for row in result["events"] if row["measured"]
    )
    assert result["causal_study"]["client_training"] == (variant != "server_only")
    assert len(result["causal_study"]["selected_data"]["train"]["source_ids"]) == 4
    assert result["causal_study"]["effective_gate"]["selection"] == "min_risk"
    assert result["metrics"]["final_class_nll"] >= 0
    json.dumps(
        {k: v for k, v in result.items() if k not in {"events"}}, allow_nan=False
    )


def test_server_fit_microbatch_weights_match_full_batch():
    model = Model()
    zero = empty_adapter_state(model)
    x, y = torch.eye(2), torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    kwargs = dict(
        loss_fn=lambda m, b: (m(b["x"]) - b["y"]).square().mean(),
        rank=2,
        server_weight=0.5,
        seed=123,
        rank_rtol=1e-7,
        config=ServerAdaptationConfig(steps=3, learning_rate=0.02),
    )
    full, _ = fit_server_adapter(model, zero, zero, [({"x": x, "y": y}, 2.0)], **kwargs)
    micro, _ = fit_server_adapter(
        model,
        zero,
        zero,
        [({"x": x[i : i + 1], "y": y[i : i + 1]}, 1.0) for i in range(2)],
        **kwargs,
    )
    torch.testing.assert_close(
        dense(full["layer"]), dense(micro["layer"]), atol=1e-5, rtol=1e-5
    )


def test_seed_duplicates_are_not_replications():
    spec = study_spec()
    spec["phases"]["development"]["seeds"] = [1, 1]
    with pytest.raises(ValueError, match="duplicate seeds"):
        suite.build_jobs(spec, suite="attribution", phase="development")


def synthetic_result(plan, job, root):
    parent = (
        root
        / plan["phase"]
        / plan["suite"]
        / job["task"]
        / job["regime"]
        / job["variant"]
    )
    config = {**job["config"], "output_dir": str(parent.resolve())}
    result = {
        "method": job["method"],
        "variant": job["variant"],
        "seed": job["seed"],
        "config_fingerprint": runner._config_fingerprint(config),
        "causal_study": {
            "protocol": config["causal_study"],
            "data_sha256": "data",
            "schedule_sha256": "schedule",
            "partitions_sha256": "partitions",
            "environment": {"torch": "test"},
        },
        "metrics": {
            "final_accuracy": 0.7,
            "final_class_nll": 0.6,
            "harmful_update_rate": 0.1,
            "acceptance_rate": 0.5,
            "late_event_count": 0,
            "late_harmful_update_rate": 0.0,
            "measured_event_count": config["experiment"]["collected_returns"],
        },
    }
    path = parent / f"{job['variant']}_seed{job['seed']}" / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    return path, result


def test_analysis_missing_pairs_and_resume_refuses_bad_results(tmp_path, monkeypatch):
    import analyze_rift_causal_study as analysis

    spec = study_spec()
    spec["phases"]["development"]["seeds"] = [8101, 8102]
    plan = suite.freeze(
        spec,
        suite="attribution",
        phase="development",
        tasks=["qnli"],
        variants=["core", "diag"],
    )
    for job in plan["jobs"][:-1]:
        synthetic_result(plan, job, tmp_path)
    report = analysis.analyze(plan, tmp_path)
    assert report["completed"] == 3 and len(report["missing"]) == 1
    assert report["pairs"][0]["complete"] is False
    assert all(s["late_harmful_mean"] is None for s in report["summaries"])
    last, result = synthetic_result(plan, plan["jobs"][-1], tmp_path)
    report = analysis.analyze(plan, tmp_path)
    assert report["pairs"][0]["complete"] and report["pairs"][0]["seeds"] == [
        8101,
        8102,
    ]
    assert report["pairs"][0]["metrics"]["final_class_nll"][
        "pointwise_95pct_t_interval"
    ] == [0.0, 0.0]
    assert "Accuracy" in analysis.markdown(report)
    monkeypatch.setattr(
        suite.subprocess,
        "run",
        lambda *a, **k: pytest.fail("completed jobs must not run"),
    )
    suite.execute(plan, tmp_path)
    result["metrics"]["final_class_nll"] = float("nan")
    last.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing overwrite"):
        suite.execute(plan, tmp_path)
    with pytest.raises(ValueError, match="mismatched"):
        analysis.analyze(plan, tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("data_sha256", "other"),
        ("schedule_sha256", "other"),
        ("environment", {"torch": "different"}),
    ],
)
def test_analysis_rejects_unpaired_runs(tmp_path, key, value):
    import analyze_rift_causal_study as analysis

    plan = suite.freeze(
        study_spec(),
        suite="attribution",
        phase="smoke",
        tasks=["qnli"],
        variants=["core", "diag"],
    )
    for job in plan["jobs"]:
        path, result = synthetic_result(plan, job, tmp_path)
    result["causal_study"][key] = value
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="unpaired"):
        analysis.analyze(plan, tmp_path)


def test_no_repair_is_positive_component_filter():
    from riftlora.scale.objective import filter_compact_by_scores

    # The fixture below supplies a signed score and a full-rank innovation.
    model = Model()
    innovation = {
        "layer": CompactSVD(torch.eye(2), torch.tensor([2.0, 1.0]), torch.eye(2))
    }
    scores = {"layer": torch.tensor([1.0, -1.0])}
    from riftlora.scale.core_repair import repair_compact_core

    actual = repair_compact_core(
        model,
        innovation,
        scores,
        [({"x": torch.eye(2)}, 2.0)],
        loss_fn=lambda m, b: m(b["x"]).square().mean(),
        server_weight=0.5,
        staleness=20,
        config=CoreRepairConfig(radius=0),
    )
    expected = filter_compact_by_scores(innovation, scores, keep_nonpositive=False)
    torch.testing.assert_close(dense(actual.updates["layer"]), dense(expected["layer"]))


@pytest.mark.parametrize("eval_split", ["train", "validation"])
def test_content_groups_are_disjoint_and_phase_independent(eval_split):
    from datasets import Dataset
    from riftlora.scale.causal_study import prepare_content_disjoint_splits, content_key

    rows = [{"sentence": f"item {i}", "label": i % 2} for i in range(40)]
    rows += [
        {"sentence": "  item  1 ", "label": 1},
        {"sentence": "conflict", "label": 0},
        {"sentence": "conflict", "label": 1},
    ]
    raw = attach_source_ids(
        {"train": Dataset.from_list(rows), "validation": Dataset.from_list(rows)}
    )
    data = {
        "train_split": "train",
        "validation_split": "validation",
        "eval_split": eval_split,
        "label_column": "label",
        "eval_shuffle_seed": 7,
        "eval_examples": 4,
        "study_content_policy": "nfkc_whitespace_groups_v1",
        "study_eval_windows": [[0, 4], [4, 4]],
    }
    t1, e1, audit = prepare_content_disjoint_splits(raw, {**data, "eval_offset": 0})
    t2, e2, _ = prepare_content_disjoint_splits(raw, {**data, "eval_offset": 4})
    keys = lambda data: {content_key(r, "label") for r in data}
    assert list(t1["_rift_source_id"]) == list(t2["_rift_source_id"])
    assert not (keys(t1) & (keys(e1) | keys(e2)))
    assert not (keys(e1) & keys(e2))
    assert len(t1) == 32 and audit["train"]["canonical_rows"] == 40
    assert audit["train"]["conflicting_content_groups"] == 1
