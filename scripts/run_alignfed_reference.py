"""Buffered AlignFed reference experiment; distinct from alignfed_calibration.

Uses homogeneous, persistent factor coordinates and an explicit rank-linear T.
This is a documented reimplementation, not an official paper reproduction.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint

import run_kaggle_3b as shared
from run_week8_classification_matrix import _build_config
from riftlora.scale.alignfed_peft import (
    PenultimateFeatures, factor_parameters, features_at_factors, load_factors, snapshot_factors,
)
from riftlora.scale.alignfed_reference import (
    AlignFedBuffer, FactorReturn, factor_difference, fit_semantic_transform, representation_penalty,
)


def build_config(matrix, task_name):
    tasks = {task["name"]: task for task in matrix["tasks"]}
    if task_name not in tasks:
        raise ValueError(f"unknown task {task_name}")
    task = tasks[task_name]
    base = json.loads((ROOT / task["base_config"]).read_text(encoding="utf-8"))
    config = _build_config(base, task, matrix["regimes"][0], matrix)
    config["alignfed_reference"] = dict(matrix["alignfed_reference"])
    config["provenance"] = {"matrix_name": matrix["name"], "phase": matrix.get("phase", "development"),
                            "matrix_sha256": shared._config_fingerprint(matrix)}
    return config


def validate_config(config):
    experiment, alignment = config["experiment"], config["alignfed_reference"]
    rank = experiment["server_max_rank"]
    if experiment["client_ranks"] != [rank] * experiment["num_clients"]:
        raise ValueError("AlignFed reference requires identical persistent factor ranks")
    if any(experiment.get(key, 0) for key in ("network_time", "jitter")):
        raise ValueError("reference scheduler currently requires network_time=jitter=0")
    if experiment.get("server_update_weight", 1.0) != 1.0:
        raise ValueError("equation (8) uses server_update_weight=1")
    if any(float(t) <= 0 or not math.isfinite(float(t)) for t in experiment["compute_times"]):
        raise ValueError("compute times must be positive")
    AlignFedBuffer(max_pending=alignment["max_pending"], max_wait=alignment["max_wait"])
    if not isinstance(alignment["steps"], int) or alignment["steps"] < 1 or alignment["learning_rate"] <= 0:
        raise ValueError("positive alignment steps and learning rate required")
    for key in ("gamma", "representation_weight", "learning_rate"):
        if not math.isfinite(alignment[key]) or alignment[key] < 0:
            raise ValueError(f"invalid alignment {key}")
    semantic_batch_size = alignment.get("semantic_batch_size", 1)
    if not isinstance(semantic_batch_size, int) or semantic_batch_size < 1:
        raise ValueError("semantic_batch_size must be a positive integer")
    # Reuse task/data validation; this runner owns its own buffered schedule.
    compatible = json.loads(json.dumps(config))
    compatible["experiment"]["buffer_size"] = 1
    shared._validate_config(compatible, "freshness")
    if experiment["calibration_gradient_examples"] < 1:
        raise ValueError("semantic alignment needs calibration examples")


def prepare_data(config, seed):
    from datasets import load_dataset
    ds, exp = config["dataset"], config["experiment"]
    raw = load_dataset(ds["hub_path"], ds.get("subset"), revision=ds.get("revision"))
    train_split = ds["train_split"]
    eval_split = ds.get("eval_split", ds["validation_split"])
    offset, count = int(ds.get("eval_offset", 0)), int(ds["eval_examples"])
    if eval_split == train_split:
        if not ds.get("reserve_eval_from_train"):
            raise ValueError("train-split evaluation must be explicitly reserved")
        train_ids, eval_ids = shared.reserved_train_eval_indices(
            len(raw[train_split]), eval_offset=offset, eval_examples=count,
            shuffle_seed=ds.get("eval_shuffle_seed", seed),
        )
        train, evaluation = raw[train_split].select(train_ids), raw[train_split].select(eval_ids)
    else:
        train = raw[train_split]
        evaluation = raw[eval_split].shuffle(seed=ds.get("eval_shuffle_seed", seed))
        if offset < 0 or offset + count > len(evaluation):
            raise ValueError("requested evaluation range exceeds split")
        evaluation = evaluation.select(range(offset, offset + count))
    sizes = (exp["calibration_gradient_examples"], exp["calibration_gate_examples"], exp["monitor_examples"])
    if sum(sizes) + ds["max_train_examples"] > len(train):
        raise ValueError("not enough training data for disjoint client/calibration/monitor sets")
    for name, data in (("training", train), ("evaluation", evaluation)):
        shared._validate_dataset_labels(data, label_column=ds["label_column"],
                                        label_count=len(shared._label_texts(ds)), split_name=name)
    train, (calibration, _, monitor) = shared._reserve_calibration_splits(
        train, label_column=ds["label_column"], split_sizes=sizes,
        max_train_examples=ds["max_train_examples"], seed=seed,
        stratified=exp.get("calibration_sampling", "random") == "stratified",
    )
    partitions = shared._build_partitions(train, label_column=ds["label_column"], experiment=exp, seed=seed)
    return train, evaluation, calibration, monitor, partitions


def train_client(model, tokenizer, extractor, source, dataset, indices, *, config, rng, regularization):
    exp, ds = config["experiment"], config["dataset"]
    parameters = list(factor_parameters(model).values())
    optimizer = torch.optim.AdamW(parameters, lr=exp["local_learning_rate"], weight_decay=exp["weight_decay"])
    losses = []
    for _ in range(exp["local_steps"]):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(exp["gradient_accumulation_steps"]):
            examples = [dataset[rng.choice(indices)] for _ in range(exp["local_batch_size"])]
            batch = shared._collate_examples(tokenizer, [(x, int(x[ds["label_column"]])) for x in examples],
                                             dataset_config=ds, max_length=config["model"]["max_length"])
            batch = shared._move_batch(batch, shared._model_input_device(model))
            model.train()
            logits, labels = shared._supervised_suffix_logits(model, batch)
            token_loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels,
                                                          ignore_index=-100, reduction="none")
            mask = labels.ne(-100)
            if tokenizer.eos_token_id is not None:
                mask &= labels.ne(tokenizer.eos_token_id)
            loss = (token_loss * mask).sum() / mask.sum().clamp_min(1)
            if regularization:
                model.eval()
                with torch.no_grad():
                    reference = features_at_factors(extractor, source, batch)
                local_features = extractor(batch)
                loss = loss + representation_penalty(local_features, reference, weight=regularization)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite local AlignFed loss")
            losses.append(float(loss.detach()))
            (loss / exp["gradient_accumulation_steps"]).backward()
        torch.nn.utils.clip_grad_norm_(parameters, exp["gradient_clip_norm"])
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return sum(losses) / len(losses)


def run(config, *, method, seed):
    if method not in {"alignfed_reference", "factor_fedbuff"}:
        raise ValueError(f"unknown reference method {method}")
    shared._seed_everything(seed)
    validate_config(config)
    exp, ds, alignment = config["experiment"], config["dataset"], config["alignfed_reference"]
    train, evaluation, calibration, monitor, partitions = prepare_data(config, seed)
    tokenizer, model = shared._load_model(config)
    extractor = PenultimateFeatures(model)
    global_state = snapshot_factors(model)
    snapshots = {0: global_state}
    eval_args = dict(dataset_config=ds, max_length=config["model"]["max_length"], batch_size=exp["eval_batch_size"])
    baseline, _ = shared.evaluate_classification(model, tokenizer, evaluation, **eval_args)
    batches = shared._make_classification_batches(
        model,
        tokenizer,
        calibration,
        dataset_config=ds,
        max_length=config["model"]["max_length"],
        batch_size=alignment.get("semantic_batch_size", 1),
    )
    # Semantic feature means are example-weighted. The shared helper's weight is
    # supervised-token based because it normally serves NLL aggregation.
    batches = [(batch, float(batch["input_ids"].shape[0])) for batch, _ in batches]
    sides = {k: "a" if ".lora_A." in k else "b" for k in global_state}
    buffer = AlignFedBuffer(max_pending=alignment["max_pending"], max_wait=alignment["max_wait"])
    current_version = 0
    queue = [(float(duration), client, 0) for client, duration in enumerate(exp["compute_times"])]
    heapq.heapify(queue)
    rngs = [random.Random(seed * 10000 + client) for client in range(exp["num_clients"])]
    rows, returned = [], 0
    total_returns = exp["warmup_returns"] + exp["collected_returns"]
    start_time = time.perf_counter()
    last_local_loss = None

    def mean_features(state, *, differentiable):
        total, weight_sum = None, 0.0
        keys = tuple(state)
        for batch, weight in batches:
            def compute(*values, selected_batch=batch):
                return features_at_factors(extractor, dict(zip(keys, values)), selected_batch).mean(0)
            if differentiable:
                features = checkpoint(compute, *[state[k] for k in keys], use_reentrant=False)
            else:
                with torch.no_grad():
                    features = compute(*[state[k] for k in keys])
            total = features * weight if total is None else total + features * weight
            weight_sum += weight
        return total / weight_sum

    def flush(now, *, returns_seen):
        nonlocal global_state, current_version
        model.eval()
        load_factors(model, global_state)
        before_loss = shared._mean_classification_loss(model, tokenizer, monitor, **eval_args,
                                                       objective=exp.get("monitor_objective", "class_nll"))
        pending = list(buffer.pending)
        stalenesses = [current_version - item.base_version for item in pending]
        needs_alignment = method == "alignfed_reference" and any(
            item.base_version != current_version for item in pending
        )
        target = mean_features(global_state, differentiable=False) if needs_alignment else None

        def align_group(version, deltas):
            device_parameters = factor_parameters(model)
            source = {k: v.to(device_parameters[k]) for k, v in snapshots[version].items()}
            deltas = [{k: v.to(device_parameters[k]) for k, v in delta.items()} for delta in deltas]
            transform = fit_semantic_transform(
                source, deltas, target, lambda state: mean_features(state, differentiable=True),
                factor_sides=sides, steps=alignment["steps"], learning_rate=alignment["learning_rate"],
            )
            with torch.no_grad():
                return [{k: v.cpu() for k, v in transform(delta).items()} for delta in deltas]

        if method == "alignfed_reference":
            global_state, diagnostics = buffer.flush(global_state, current_version=current_version, now=now,
                                                     align_group=align_group, gamma=alignment["gamma"])
        else:
            # Matched factor-coordinate buffered control: no centering/alignment/local penalty.
            global_state = {k: v + torch.stack([x.delta[k] for x in pending]).mean(0)
                            for k, v in global_state.items()}
            diagnostics = {"buffer_returns": len(pending), "singleton_groups": None, "zero_centered_returns": None}
            buffer.pending.clear()
            buffer.last_time = buffer.last_aggregation = now
        current_version += 1
        snapshots[current_version] = global_state
        load_factors(model, global_state)
        after_loss = shared._mean_classification_loss(model, tokenizer, monitor, **eval_args,
                                                      objective=exp.get("monitor_objective", "class_nll"))
        harm = after_loss > before_loss + exp.get("harm_epsilon", 1e-6)
        return_start = returns_seen - len(pending) + 1
        rows.append({"version": current_version, "time": now, "current_loss": before_loss,
                     "accepted_loss": after_loss, "harmful": harm,
                     "contains_late": max(stalenesses) >= exp.get("late_tau", 4),
                     "return_start": return_start, "return_end": returns_seen,
                     "measured": return_start > exp["warmup_returns"],
                     "stalenesses": stalenesses, "local_loss": last_local_loss, **diagnostics})

    while returned < total_returns or buffer.pending:
        next_arrival = queue[0][0] if returned < total_returns else math.inf
        deadline = max(buffer.last_time, buffer.last_aggregation + buffer.max_wait) if buffer.pending else math.inf
        if deadline <= next_arrival:
            flush(deadline, returns_seen=returned)
            continue
        now, client, base_version = heapq.heappop(queue)
        source = snapshots[base_version]
        load_factors(model, source)
        last_local_loss = train_client(
            model, tokenizer, extractor, source, train, partitions[client], config=config, rng=rngs[client],
            regularization=alignment["representation_weight"] if method == "alignfed_reference" else 0,
        )
        delta = factor_difference(snapshot_factors(model), source)
        if buffer.add(FactorReturn(str(client), base_version, delta), now=now):
            flush(now, returns_seen=returned + 1)
        returned += 1
        heapq.heappush(queue, (now + float(exp["compute_times"][client]), client, current_version))
    load_factors(model, global_state)
    final, details = shared.evaluate_classification(model, tokenizer, evaluation, **eval_args)
    measured = [row for row in rows if row["measured"]]
    late = [row for row in measured if row["contains_late"]]
    if not measured:
        raise RuntimeError("no complete post-warmup aggregation buffer was measured")
    return {
        "schema_version": 5,
        "method": method, "seed": seed, "task": ds.get("run_name", ds["task"]),
        "regime": exp.get("regime_name", "default"),
        "fidelity": "explicit rank-linear buffered reimplementation; not official reproduction",
        "config": config, "config_fingerprint": shared._config_fingerprint(config),
        "provenance": dict(config.get("provenance", {})),
        "git_commit": shared._git_commit(), "git_worktree_dirty": shared._git_worktree_dirty(),
        "protocol": {"warmup": "method active for all returns; safety metrics exclude buffers touching warmup returns",
                     "rank": "homogeneous persistent factors",
                     "timeout": "simulated time; alignment compute not added to schedule",
                     "harm_unit": "aggregation buffer", "late_definition": "buffer contains a late return",
                     "feature_pooling": "mean unmasked prompt tokens at penultimate decoder block",
                     "transform": "separate rank x rank maps on A and B; Adam; best calibration discrepancy iterate",
                     "calibration_examples": len(calibration)},
        "baseline": baseline, "final": final,
        "metrics": {"final_accuracy": final["accuracy"], "final_class_nll": final["class_nll"],
                    "harmful_buffer_rate": sum(row["harmful"] for row in measured) / len(measured),
                    "late_harmful_buffer_rate": sum(row["harmful"] for row in late) / len(late) if late else None,
                    "late_buffer_count": len(late), "measured_buffer_count": len(measured),
                    "aggregation_count": len(rows), "return_count": returned,
                    "runtime_seconds": time.perf_counter() - start_time},
        "events": rows, "final_eval_details": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs/paper_baseline_audit_matrix.json")
    parser.add_argument("--task", choices=["sst2", "qnli", "mnli_m", "mnli_mm"], default="sst2")
    parser.add_argument("--method", choices=["alignfed_reference", "factor_fedbuff"], default="alignfed_reference")
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/paper_baseline_development")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = build_config(json.loads(args.matrix.read_text(encoding="utf-8")), args.task)
    validate_config(config)
    destination = args.output_dir / args.task / args.method / f"seed{args.seed}"
    if args.dry_run:
        print(json.dumps({"valid": True, "method": args.method, "config": config,
                          "output": str(destination)}, indent=2))
        return
    if (destination / "result.json").exists():
        raise FileExistsError(f"result already exists: {destination}; choose a new output directory")
    result = run(config, method=args.method, seed=args.seed)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.pop("events")).to_csv(destination / "events.csv", index=False)
    pd.DataFrame(result.pop("final_eval_details")).to_csv(destination / "final_eval_details.csv", index=False)
    (destination / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
