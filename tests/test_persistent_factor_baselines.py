"""Paper algebra and state invariants, independent of benchmark scores."""
import json
import sys
from pathlib import Path

import pytest
import torch

from riftlora.baselines.factor_averaging import (
    FACTOR_IMPLEMENTATION,
    average_factor_states,
    fedavg_aggregate_factor_state,
    load_factor_state,
)
from riftlora.scale.peft_bridge import FactorSnapshot, capture_factor_snapshot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def state(a, b):
    return {"layer": FactorSnapshot(torch.tensor(a, dtype=torch.float32),
                                    torch.tensor(b, dtype=torch.float32), 1.)}


@pytest.mark.parametrize("freeze_a", [False, True])
def test_sample_weighted_mean_matches_parameter_dictionary_reference(freeze_a):
    clients = [state([[2.]], [[3.]]), state([[2. if freeze_a else 5.]], [[7.]])]
    result = average_factor_states(clients, weights=[2, 3], freeze_a=freeze_a)["layer"]
    # Same normalized sample-weight arithmetic as FedIT model_aggregation.py.
    weights = torch.nn.functional.normalize(torch.tensor([2., 3.]), p=1, dim=0)
    for key in ("a", "b"):
        expected = sum(getattr(client["layer"], key) * w for client, w in zip(clients, weights))
        torch.testing.assert_close(getattr(result, key), expected)
    result.b.zero_()
    assert clients[0]["layer"].b.item() == 3  # Server history cannot alias clients.


def test_multireturn_fedavg_keeps_original_factors_and_stale_snapshots():
    initial = state([[2.]], [[3.]])
    same = fedavg_aggregate_factor_state(initial, initial, active_rank=1, max_rank=1, weight=.5)
    assert (same["layer"].b @ same["layer"].a).item() == 6.
    current = fedavg_aggregate_factor_state(same, state([[4.]], [[5.]]),
                                           active_rank=1, max_rank=1, weight=.5)
    returned_stale_client = state([[3.]], [[4.]])
    final = fedavg_aggregate_factor_state(current, returned_stale_client,
                                         active_rank=1, max_rank=1, weight=.25)
    assert final["layer"].a.item() == 3.
    assert final["layer"].b.item() == 4.
    assert initial["layer"].a.item() == 2.
    assert initial["layer"].b.item() == 3.


def test_ffa_inactive_rank_shrinks_only_b_not_both_factors():
    server = state([[1., 0.], [0., 1.]], [[2., 0.], [0., 1.]])
    client = state([[1., 0.], [0., 1.]], [[2., 0.], [0., 0.]])
    result = fedavg_aggregate_factor_state(server, client, active_rank=1, max_rank=2,
                                           weight=.5, freeze_a=True)["layer"]
    assert torch.equal(result.a, server["layer"].a)
    torch.testing.assert_close(result.b @ result.a, torch.diag(torch.tensor([2., .5])))
    client["layer"].a[0, 0] += 1
    with pytest.raises(ValueError, match="A0 changed"):
        fedavg_aggregate_factor_state(server, client, active_rank=1, max_rank=2,
                                     weight=.5, freeze_a=True)


def test_freeze_survives_optimizer_dispatch_and_isolated_evaluation():
    from peft import LoraConfig, get_peft_model
    from run_kaggle_3b import _freeze_lora_a
    from riftlora.scale.generation_study import evaluate_server
    from riftlora.scale.peft_bridge import mask_inactive_rank_gradients

    torch.manual_seed(5)
    model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False)),
                          LoraConfig(r=2, lora_alpha=2, target_modules=["0"]))
    _freeze_lora_a(model)
    initial = capture_factor_snapshot(model)
    server = initial
    for rank in (1, 2, 1, 2):
        load_factor_state(model, server, active_rank=rank, freeze_a=True)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                     lr=.01, weight_decay=.1)
        loss = model(torch.ones(2, 3)).square().mean()
        loss.backward()
        mask_inactive_rank_gradients(model, active_rank=rank)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        client = capture_factor_snapshot(model)
        server = fedavg_aggregate_factor_state(server, client, active_rank=rank,
                                               max_rank=2, weight=.5, freeze_a=True)
        evaluate_server(model, {}, rank=2, factor_state=server, freeze_a=True,
                        evaluate=lambda: model(torch.ones(1, 3)))
        for name, value in capture_factor_snapshot(model).items():
            assert torch.equal(value.a, initial[name].a)
            assert torch.equal(value.b, client[name].b)  # Evaluation restores local state.


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [float("nan"), 1], [1e308, 1e308]])
def test_invalid_weight_rejection(weights):
    with pytest.raises(ValueError):
        average_factor_states([state([[2.]], [[3.]])] * 2, weights=weights)


@pytest.mark.parametrize("method", ["fedavg_lora", "ffa_lora"])
def test_resume_rejects_legacy_result_even_if_config_and_commit_match(tmp_path, monkeypatch, method):
    import run_week8_classification_matrix as runner
    monkeypatch.setattr(runner, "_runner_git_commit", lambda: "same")
    monkeypatch.setattr(runner, "_runner_config_fingerprint", lambda c: "config")
    config = {"provenance": {"matrix_sha256": "matrix"}}
    payload = dict(method=method, seed=6101, schema_version=5, git_commit="same",
                   git_worktree_dirty=False, config_fingerprint="config", provenance=config["provenance"])
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    assert not runner._completed_result_matches(path, config=config, method=method, seed=6101, matrix={})
    payload["factor_implementation"] = FACTOR_IMPLEMENTATION
    path.write_text(json.dumps(payload))
    assert runner._completed_result_matches(path, config=config, method=method, seed=6101, matrix={})
