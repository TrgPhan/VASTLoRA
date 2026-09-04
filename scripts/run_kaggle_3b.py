from __future__ import annotations

import argparse
import copy
from collections import deque
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from riftlora.asyncfl import AsyncEventSimulator, ClientProfile
from riftlora.data import iid_partition_indices, label_shard_partition_indices
from riftlora.lowrank import CompactSVD
from riftlora.scale import (
    TransportConfig,
    aggregate_compact_state,
    capture_factor_snapshot,
    compact_factor_innovations,
    empty_adapter_state,
    fedrot_aggregate_factor_state,
    filter_compact_by_scores,
    load_compact_adapter_state,
    mask_inactive_rank_gradients,
    scale_compact_update,
    score_compact_components_microbatched,
    transport_compact_update,
)
from riftlora.scale.tradeoff import reserved_train_eval_indices


DEFAULT_LABEL_TEXTS = {
    "sst2": [" negative", " positive"],
    "qnli": [" yes", " no"],
    "mnli": [" entailment", " neutral", " contradiction"],
}
METHODS = (
    "raw",
    "fedex",
    "freshness",
    "fedrot",
    "vast",
    "mtip",
    "mtip_adaptive",
    "mtip_hybrid",
    "mtip_routed",
    "rift",
    "spectral_filter",
    "alignfed_calibration",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory-bounded 3B MTIP Kaggle benchmark")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/kaggle_3b_pilot.json")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--collected-returns", type=int)
    parser.add_argument("--local-steps", type=int)
    parser.add_argument("--eval-examples", type=int)
    parser.add_argument("--eval-offset", type=int)
    parser.add_argument("--eval-shuffle-seed", type=int)
    parser.add_argument("--eval-split")
    parser.add_argument("--partition-mode", choices=("iid", "label_shard"))
    parser.add_argument("--reserve-eval-from-train", action="store_true")
    parser.add_argument("--residual-beta", type=float)
    parser.add_argument("--residual-staleness-center", type=float)
    parser.add_argument("--residual-staleness-temperature", type=float)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.variant is not None and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.variant) is None:
        raise ValueError("variant must contain only letters, numbers, dot, underscore, or dash")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _apply_overrides(config, args)
    _validate_config(config, args.method)
    if args.dry_run:
        print(json.dumps(_dry_run_summary(config, args), indent=2))
        return

    result = run_experiment(config, method=args.method, seed=args.seed)
    variant = args.variant or args.method
    result["variant"] = variant
    output_dir = Path(config["output_dir"]) / f"{variant}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.pop("events")).to_csv(output_dir / "events.csv", index=False)
    pd.DataFrame(result.pop("baseline_eval_details")).to_csv(
        output_dir / "baseline_eval_details.csv", index=False
    )
    pd.DataFrame(result.pop("final_eval_details")).to_csv(
        output_dir / "final_eval_details.csv", index=False
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True, allow_nan=False))


def run_experiment(config: dict[str, Any], *, method: str, seed: int) -> dict[str, Any]:
    from datasets import load_dataset

    _seed_everything(seed)
    config = _resolve_single_regime_config(config)
    dataset_config = config["dataset"]
    raw = load_dataset(dataset_config["hub_path"], dataset_config.get("subset"))
    train_split = dataset_config["train_split"]
    eval_split = dataset_config.get("eval_split", dataset_config["validation_split"])
    eval_shuffle_seed = dataset_config.get("eval_shuffle_seed", seed)
    eval_offset = int(dataset_config.get("eval_offset", 0))
    eval_examples = int(dataset_config["eval_examples"])
    if eval_split == train_split and dataset_config.get("reserve_eval_from_train", False):
        train_indices, eval_indices = reserved_train_eval_indices(
            len(raw[train_split]),
            eval_offset=eval_offset,
            eval_examples=eval_examples,
            shuffle_seed=eval_shuffle_seed,
        )
        train = raw[train_split].select(train_indices)
        validation = raw[train_split].select(eval_indices)
    else:
        validation = raw[eval_split].shuffle(seed=eval_shuffle_seed)
        eval_end = min(eval_offset + eval_examples, len(validation))
        if eval_offset < 0 or eval_offset >= eval_end:
            raise ValueError("eval_offset must select at least one evaluation example")
        validation = validation.select(range(eval_offset, eval_end))
        train = raw[train_split]
    experiment = config["experiment"]
    calibration_gradient_examples = int(experiment.get("calibration_gradient_examples", 0))
    calibration_gate_examples = int(experiment.get("calibration_gate_examples", 0))
    monitor_examples = int(experiment.get("monitor_examples", 0))
    shuffled_train = train.shuffle(seed=seed)
    reserved_total = calibration_gradient_examples + calibration_gate_examples + monitor_examples
    if reserved_total + int(dataset_config["max_train_examples"]) > len(shuffled_train):
        raise ValueError("reserved calibration/monitor plus max_train_examples exceeds train data")
    calibration_gradient = (
        shuffled_train.select(range(calibration_gradient_examples))
        if calibration_gradient_examples
        else None
    )
    calibration_gate = (
        shuffled_train.select(
            range(
                calibration_gradient_examples,
                calibration_gradient_examples + calibration_gate_examples,
            )
        )
        if calibration_gate_examples
        else None
    )
    monitor = (
        shuffled_train.select(
            range(
                calibration_gradient_examples + calibration_gate_examples,
                reserved_total,
            )
        )
        if monitor_examples
        else None
    )
    train = shuffled_train.select(
        range(reserved_total, reserved_total + dataset_config["max_train_examples"])
    )

    label_column = dataset_config["label_column"]
    partitions = _build_partitions(
        train,
        label_column=label_column,
        experiment=experiment,
        seed=seed,
    )
    train_labels = [int(value) for value in train[label_column]]
    partition_diagnostics = {
        str(client_id): {
            "num_examples": len(indices),
            "label_histogram": _label_histogram(
                [train_labels[index] for index in indices]
            ),
            "rank": int(experiment["client_ranks"][client_id]),
            "compute_time": float(experiment["compute_times"][client_id]),
        }
        for client_id, indices in enumerate(partitions)
    }
    clients = _build_clients(experiment, partitions)
    total_returns = experiment["warmup_returns"] + experiment["collected_returns"]
    trace = AsyncEventSimulator(
        clients,
        seed=seed,
        buffer_size=int(experiment.get("buffer_size", 1)),
        schedule_mode=str(experiment.get("schedule_mode", "async")),
    ).run(max_returns=total_returns)

    tokenizer, model = _load_model(config)
    component_score_objective = str(
        experiment.get("component_score_objective", "label_nll")
    )
    gradient_batch_size = int(
        experiment.get(
            "calibration_gradient_batch_size",
            calibration_gradient_examples,
        )
    )
    if calibration_gradient is None:
        gradient_batches = None
        component_score_loss_fn = None
    elif component_score_objective == "class_nll":
        gradient_batches = _make_classification_candidate_batches(
            model,
            tokenizer,
            calibration_gradient,
            dataset_config=dataset_config,
            max_length=config["model"]["max_length"],
            batch_size=gradient_batch_size,
        )
        component_score_loss_fn = lambda target_model, batch: (
            _classification_candidate_nll_loss(
                target_model,
                batch,
                eos_token_id=tokenizer.eos_token_id,
            )
        )
    else:
        gradient_batches = _make_classification_batches(
            model,
            tokenizer,
            calibration_gradient,
            dataset_config=dataset_config,
            max_length=config["model"]["max_length"],
            batch_size=gradient_batch_size,
        )
        component_score_loss_fn = None
    server_state = _state_to_cpu(empty_adapter_state(model))
    snapshots: dict[int, dict[str, CompactSVD]] = {0: server_state}
    histories = {
        name: deque(maxlen=experiment["history_size"]) for name in server_state
    }
    transport_config = TransportConfig(
        freshness_lambda=experiment["freshness_lambda"],
        reference_rank=experiment["reference_rank"],
        reference_decay=experiment["reference_decay"],
        reference_singular_power=experiment["reference_singular_power"],
        adaptive_energy=experiment["adaptive_energy"],
        adaptive_min_rank=experiment["adaptive_min_rank"],
        adaptive_max_rank=experiment["adaptive_max_rank"],
        adaptive_singular_power=experiment["adaptive_singular_power"],
        residual_beta=experiment.get("residual_beta", 0.5),
        residual_staleness_center=experiment.get("residual_staleness_center", 4.0),
        residual_staleness_temperature=experiment.get(
            "residual_staleness_temperature", 1.0
        ),
        rank_rtol=experiment["rank_rtol"],
    )

    baseline, baseline_details = evaluate_classification(
        model,
        tokenizer,
        validation,
        dataset_config=dataset_config,
        max_length=config["model"]["max_length"],
        batch_size=experiment["eval_batch_size"],
    )
    start_time = time.perf_counter()
    event_rows: list[dict[str, Any]] = []
    rng = random.Random(seed)

    for event_index, event in enumerate(trace.records):
        client_id = int(event.client_id)
        active_rank = event.rank
        stale_state = snapshots[event.base_version]
        load_compact_adapter_state(
            model,
            stale_state,
            active_rank=active_rank,
            seed=seed * 10000 + event_index,
            initialize_free_directions=True,
        )
        before = capture_factor_snapshot(model)
        local_loss = _train_client(
            model,
            tokenizer,
            train,
            partitions[client_id],
            rng=rng,
            active_rank=active_rank,
            dataset_config=dataset_config,
            max_length=config["model"]["max_length"],
            local_steps=experiment["local_steps"],
            gradient_accumulation_steps=experiment["gradient_accumulation_steps"],
            batch_size=experiment["local_batch_size"],
            learning_rate=experiment["local_learning_rate"],
            weight_decay=experiment["weight_decay"],
            gradient_clip_norm=experiment["gradient_clip_norm"],
        )
        after = capture_factor_snapshot(model)
        innovations = compact_factor_innovations(
            before,
            after,
            active_rank=active_rank,
            rank_rtol=experiment["rank_rtol"],
        )

        accepted_method = (
            "freshness" if event_index < experiment["warmup_returns"] else method
        )
        current_state = snapshots[event.arrival_version]
        next_state: dict[str, CompactSVD] = {}
        rhos: list[float] = []
        left_ranks: list[int] = []
        right_ranks: list[int] = []
        freshness_values: list[float] = []
        residual_scales: list[float] = []
        accepted_scales: list[float] = []
        retained_ranks: list[int] = []
        total_ranks: list[int] = []
        predicted_gains: list[float] = []
        gate_mean_deltas: list[float] = []
        accepted_routes: list[str] = []
        if accepted_method == "fedrot":
            next_state = fedrot_aggregate_factor_state(
                current_state,
                after,
                active_rank=active_rank,
                weight=experiment["server_update_weight"],
                max_rank=experiment["server_max_rank"],
                rank_rtol=experiment["rank_rtol"],
            )
            freshness = math.exp(-transport_config.freshness_lambda * event.staleness)
            rhos.append(1.0)
            freshness_values.append(freshness)
            residual_scales.append(1.0)
            left_ranks.append(experiment["server_max_rank"])
            right_ranks.append(experiment["server_max_rank"])
            accepted_scales.append(1.0)
            accepted_routes.append("fedrot")
            for name in next_state:
                histories[name].append(next_state[name])
        elif accepted_method in {"rift", "spectral_filter", "alignfed_calibration"}:
            load_compact_adapter_state(
                model,
                current_state,
                active_rank=experiment["server_max_rank"],
                initialize_free_directions=False,
            )
            freshness = math.exp(-transport_config.freshness_lambda * event.staleness)
            if accepted_method == "alignfed_calibration":
                next_state, accepted_updates, scale, mean_delta, route = (
                    _whole_update_gate_state(
                        model,
                        tokenizer,
                        current_state,
                        innovations,
                        calibration_gate,
                        dataset_config=dataset_config,
                        max_length=config["model"]["max_length"],
                        batch_size=experiment["eval_batch_size"],
                        experiment=experiment,
                        freshness=freshness,
                    )
                )
                accepted_scales.append(scale)
                gate_mean_deltas.append(mean_delta)
                accepted_routes.append(route)
                retained_ranks.append(sum(update.rank for update in accepted_updates.values()))
                total_ranks.append(sum(update.rank for update in innovations.values()))
                predicted_gains.append(float("nan"))
            else:
                if gradient_batches is None:
                    raise ValueError(f"{accepted_method} requires calibration_gradient_examples")
                scores = score_compact_components_microbatched(
                    model,
                    innovations,
                    gradient_batches,
                    loss_fn=component_score_loss_fn,
                )
                filtered = filter_compact_by_scores(
                    innovations,
                    scores.scores,
                    minimum_predicted_gain=float(
                        experiment.get("rift_minimum_predicted_gain", 0.0)
                    ),
                    keep_nonpositive=False,
                )
                selected_rank = sum(update.rank for update in filtered.values())
                retained_ranks.append(selected_rank)
                total_ranks.append(scores.total_rank)
                predicted_gains.append(scores.predicted_gain)
                if accepted_method == "spectral_filter":
                    if selected_rank:
                        scale = float(experiment.get("spectral_filter_scale", 1.0))
                        next_state = _aggregate_scaled_updates(
                            current_state,
                            filtered,
                            scale=scale,
                            experiment=experiment,
                        )
                        accepted_updates = filtered
                        accepted_scales.append(scale)
                        accepted_routes.append("gradient_filter_no_gate")
                    else:
                        next_state = dict(current_state)
                        accepted_updates = filtered
                        accepted_scales.append(0.0)
                        accepted_routes.append("reject")
                else:
                    next_state, accepted_updates, scale, mean_delta, route = (
                        _rift_gate_state(
                            model,
                            tokenizer,
                            current_state,
                            filtered,
                            innovations,
                            calibration_gate,
                            dataset_config=dataset_config,
                            max_length=config["model"]["max_length"],
                            batch_size=experiment["eval_batch_size"],
                            experiment=experiment,
                            freshness=freshness,
                        )
                    )
                    accepted_scales.append(scale)
                    gate_mean_deltas.append(mean_delta)
                    accepted_routes.append(route)
            freshness_values.append(freshness)
            rhos.append(1.0)
            residual_scales.append(accepted_scales[-1] if accepted_scales else 0.0)
            for name, update in accepted_updates.items():
                if update.rank:
                    histories[name].append(update)
        else:
            for name, innovation in innovations.items():
                transported = transport_compact_update(
                    innovation,
                    list(histories[name]),
                    method=accepted_method,
                    staleness=event.staleness,
                    config=transport_config,
                    max_rank=experiment["server_max_rank"],
                )
                next_state[name] = aggregate_compact_state(
                    current_state[name],
                    transported.update,
                    weight=experiment["server_update_weight"],
                    max_rank=experiment["server_max_rank"],
                    rank_rtol=experiment["rank_rtol"],
                )
                histories[name].append(transported.update)
                rhos.append(transported.rho)
                freshness_values.append(transported.freshness)
                residual_scales.append(transported.residual_scale)
                accepted_scales.append(transported.residual_scale)
                accepted_routes.append(accepted_method)
                if transported.left_rank:
                    left_ranks.append(transported.left_rank)
                    right_ranks.append(transported.right_rank)

        current_monitor_loss = float("nan")
        accepted_monitor_loss = float("nan")
        if monitor is not None:
            load_compact_adapter_state(
                model,
                current_state,
                active_rank=experiment["server_max_rank"],
                initialize_free_directions=False,
            )
            current_monitor_loss = _mean_classification_loss(
                model,
                tokenizer,
                monitor,
                dataset_config=dataset_config,
                max_length=config["model"]["max_length"],
                batch_size=experiment["eval_batch_size"],
                objective=str(experiment.get("monitor_objective", "label_nll")),
            )
            load_compact_adapter_state(
                model,
                next_state,
                active_rank=experiment["server_max_rank"],
                initialize_free_directions=False,
            )
            accepted_monitor_loss = _mean_classification_loss(
                model,
                tokenizer,
                monitor,
                dataset_config=dataset_config,
                max_length=config["model"]["max_length"],
                batch_size=experiment["eval_batch_size"],
                objective=str(experiment.get("monitor_objective", "label_nll")),
            )

        update_accepted = int(any(route != "reject" for route in accepted_routes))
        server_state = _state_to_cpu(next_state)
        snapshots[event.new_server_version] = server_state
        event_rows.append(
            {
                "event": event_index,
                "client_id": client_id,
                "client_rank": active_rank,
                "base_version": event.base_version,
                "arrival_version": event.arrival_version,
                "staleness": event.staleness,
                "group_id": event.group_id,
                "group_version": event.group_version,
                "group_position": event.group_position,
                "buffer_size": event.buffer_size,
                "group_closed": event.group_closed,
                "method": accepted_method,
                "measured": event_index >= experiment["warmup_returns"],
                "local_loss": local_loss,
                "current_loss": current_monitor_loss,
                "accepted_loss": accepted_monitor_loss,
                "update_accepted": update_accepted,
                "harmful_update": (
                    accepted_monitor_loss > current_monitor_loss + 1e-12
                    if math.isfinite(current_monitor_loss)
                    and math.isfinite(accepted_monitor_loss)
                    else False
                ),
                "late_harmful_update": (
                    event.staleness >= int(experiment.get("late_tau", 8))
                    and accepted_monitor_loss > current_monitor_loss + 1e-12
                    if math.isfinite(current_monitor_loss)
                    and math.isfinite(accepted_monitor_loss)
                    else False
                ),
                "extreme_harmful_update": (
                    event.staleness >= int(experiment.get("extreme_tau", 16))
                    and accepted_monitor_loss > current_monitor_loss + 1e-12
                    if math.isfinite(current_monitor_loss)
                    and math.isfinite(accepted_monitor_loss)
                    else False
                ),
                "freshness": _mean(freshness_values),
                "rho": _mean(rhos),
                "residual_scale": _mean(residual_scales),
                "accepted_scale": _mean(accepted_scales),
                "retained_rank": _mean(retained_ranks),
                "total_rank": _mean(total_ranks),
                "retained_fraction": (
                    _mean(retained_ranks) / _mean(total_ranks)
                    if _mean(total_ranks)
                    else 0.0
                ),
                "predicted_gain": _mean(predicted_gains),
                "gate_mean_delta": _mean(gate_mean_deltas),
                "route": ";".join(sorted(set(accepted_routes))),
                "mean_left_rank": _mean(left_ranks),
                "mean_right_rank": _mean(right_ranks),
            }
        )

    load_compact_adapter_state(
        model,
        server_state,
        active_rank=experiment["server_max_rank"],
        initialize_free_directions=False,
    )
    final, final_details = evaluate_classification(
        model,
        tokenizer,
        validation,
        dataset_config=dataset_config,
        max_length=config["model"]["max_length"],
        batch_size=experiment["eval_batch_size"],
    )
    runtime = time.perf_counter() - start_time
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    measured_rows = [
        row
        for row in event_rows
        if row["measured"]
        and math.isfinite(row["current_loss"])
        and math.isfinite(row["accepted_loss"])
    ]
    late_tau = int(experiment.get("late_tau", 8))
    extreme_tau = int(experiment.get("extreme_tau", max(16, late_tau + 1)))
    late_deltas = [
        row["accepted_loss"] - row["current_loss"]
        for row in measured_rows
        if row["staleness"] >= late_tau
    ]
    extreme_deltas = [
        row["accepted_loss"] - row["current_loss"]
        for row in measured_rows
        if row["staleness"] >= extreme_tau
    ]
    accepted_rows = [row for row in measured_rows if row["update_accepted"]]
    accepted_utilities = [
        row["current_loss"] - row["accepted_loss"] for row in accepted_rows
    ]
    returned_utilities = [
        row["current_loss"] - row["accepted_loss"] for row in measured_rows
    ]
    num_clients = int(experiment["num_clients"])
    measured_return_counts = {
        str(client_id): sum(int(row["client_id"]) == client_id for row in measured_rows)
        for client_id in range(num_clients)
    }
    measured_accept_counts = {
        str(client_id): sum(
            int(row["client_id"]) == client_id and bool(row["update_accepted"])
            for row in measured_rows
        )
        for client_id in range(num_clients)
    }
    staleness_values = [int(row["staleness"]) for row in measured_rows]
    route_values = [str(row["route"]) for row in measured_rows]
    retained_values = [
        float(row["retained_fraction"])
        for row in measured_rows
        if math.isfinite(float(row["retained_fraction"]))
    ]
    predicted_values = [
        float(row["predicted_gain"])
        for row in measured_rows
        if math.isfinite(float(row["predicted_gain"]))
    ]
    accepted_losses = [float(row["accepted_loss"]) for row in measured_rows]
    best_monitor_offset = (
        min(range(len(accepted_losses)), key=accepted_losses.__getitem__)
        if accepted_losses
        else None
    )
    metrics = {
        "baseline_accuracy": baseline["accuracy"],
        "baseline_balanced_accuracy": baseline["balanced_accuracy"],
        "baseline_brier": baseline["brier"],
        "baseline_nll": baseline["nll"],
        "baseline_binary_nll": baseline["binary_nll"],
        "baseline_class_nll": baseline["class_nll"],
        "baseline_label_nll": baseline["label_nll"],
        "baseline_eos_nll": baseline["eos_nll"],
        "final_accuracy": final["accuracy"],
        "final_balanced_accuracy": final["balanced_accuracy"],
        "final_brier": final["brier"],
        "final_nll": final["nll"],
        "final_binary_nll": final["binary_nll"],
        "final_class_nll": final["class_nll"],
        "final_label_nll": final["label_nll"],
        "final_eos_nll": final["eos_nll"],
        "accuracy_change_pp": 100.0 * (final["accuracy"] - baseline["accuracy"]),
        "nll_change": final["nll"] - baseline["nll"],
        "label_nll_change": final["label_nll"] - baseline["label_nll"],
        "binary_nll_change": _optional_difference(
            final["binary_nll"], baseline["binary_nll"]
        ),
        "class_nll_change": final["class_nll"] - baseline["class_nll"],
        "mean_local_loss": _mean([row["local_loss"] for row in event_rows]),
        "mean_staleness": _mean(staleness_values),
        "max_staleness": max(staleness_values, default=0),
        "p90_staleness": _nearest_rank_percentile(staleness_values, 0.90),
        "harmful_update_rate": _mean(
            [row["harmful_update"] for row in event_rows if row["measured"]]
        ),
        "late_harmful_update_rate": _mean(
            [
                row["late_harmful_update"]
                for row in event_rows
                if row["measured"] and row["staleness"] >= late_tau
            ]
        ),
        "extreme_harmful_update_rate": _mean(
            [
                row["extreme_harmful_update"]
                for row in event_rows
                if row["measured"] and row["staleness"] >= extreme_tau
            ]
        ),
        "monitor_loss_change": _mean(
            [
                row["accepted_loss"] - row["current_loss"]
                for row in event_rows
                if row["measured"]
                and math.isfinite(row["current_loss"])
                and math.isfinite(row["accepted_loss"])
            ]
        ),
        "acceptance_rate": _mean([row["update_accepted"] for row in measured_rows]),
        "client_return_coverage": _mean(
            [
                measured_return_counts[str(client_id)] > 0
                for client_id in range(num_clients)
            ]
        ),
        "min_client_returns": min(
            measured_return_counts.values(), default=0
        ),
        "measured_event_count": len(measured_rows),
        "measured_return_counts": measured_return_counts,
        "measured_accept_counts": measured_accept_counts,
        "late_event_count": len(late_deltas),
        "cumulative_late_harm": sum(max(delta, 0.0) for delta in late_deltas),
        "normalized_cumulative_late_harm": _mean(
            [max(delta, 0.0) for delta in late_deltas]
        ),
        "worst_step_loss_increase": max(late_deltas, default=0.0),
        "extreme_event_count": len(extreme_deltas),
        "cumulative_extreme_harm": sum(
            max(delta, 0.0) for delta in extreme_deltas
        ),
        "normalized_cumulative_extreme_harm": _mean(
            [max(delta, 0.0) for delta in extreme_deltas]
        ),
        "worst_extreme_loss_increase": max(extreme_deltas, default=0.0),
        "utility_per_accepted_update": _mean(accepted_utilities),
        "utility_per_returned_update": _mean(returned_utilities),
        "cumulative_monitor_utility": sum(returned_utilities),
        "rank_filtered_route_rate": _mean(
            [route == "rank_filtered" for route in route_values]
        ),
        "freshness_fallback_route_rate": _mean(
            [route == "freshness_fallback" for route in route_values]
        ),
        "rejection_rate": _mean([route == "reject" for route in route_values]),
        "mean_retained_fraction": _mean(retained_values),
        "mean_predicted_gain": _mean(predicted_values),
        "best_monitor_label_nll": (
            accepted_losses[best_monitor_offset]
            if best_monitor_offset is not None
            and str(experiment.get("monitor_objective", "label_nll")) == "label_nll"
            else None
        ),
        "best_monitor_class_nll": (
            accepted_losses[best_monitor_offset]
            if best_monitor_offset is not None
            and str(experiment.get("monitor_objective", "label_nll")) == "class_nll"
            else None
        ),
        "best_monitor_objective_loss": (
            accepted_losses[best_monitor_offset]
            if best_monitor_offset is not None
            else None
        ),
        "monitor_objective": str(experiment.get("monitor_objective", "label_nll")),
        "best_monitor_event": (
            int(measured_rows[best_monitor_offset]["event"])
            if best_monitor_offset is not None
            else None
        ),
        "mean_rho_after_warmup": _mean(
            [row["rho"] for row in event_rows[experiment["warmup_returns"] :]]
        ),
        "mean_adaptive_left_rank": _mean(
            [row["mean_left_rank"] for row in event_rows if row["mean_left_rank"]]
        ),
        "mean_adaptive_right_rank": _mean(
            [row["mean_right_rank"] for row in event_rows if row["mean_right_rank"]]
        ),
        "runtime_seconds": runtime,
        "peak_cuda_memory_gib": peak_memory,
    }
    return {
        "schema_version": 4,
        "method": method,
        "seed": seed,
        "model": config["model"]["name"],
        "task": dataset_config.get(
            "run_name", dataset_config.get("task", dataset_config["hub_path"])
        ),
        "regime": experiment.get("regime_name", "default"),
        "git_commit": _git_commit(),
        "git_worktree_dirty": _git_worktree_dirty(),
        "config_fingerprint": _config_fingerprint(config),
        "provenance": dict(config.get("provenance", {})),
        "config": config,
        "data_diagnostics": {
            "calibration_gradient_labels": _dataset_label_histogram(
                calibration_gradient, label_column
            ),
            "calibration_gate_labels": _dataset_label_histogram(
                calibration_gate, label_column
            ),
            "monitor_labels": _dataset_label_histogram(monitor, label_column),
            "federated_client_partitions": partition_diagnostics,
        },
        "metrics": metrics,
        "events": event_rows,
        "baseline_eval_details": baseline_details,
        "final_eval_details": final_details,
    }


def _load_model(config: Mapping[str, Any]):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_config = config["model"]
    experiment = config["experiment"]
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs: dict[str, Any] = {"torch_dtype": torch.float16}
    if model_config["load_in_4bit"]:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA requires a CUDA GPU; use --no-4bit for a CPU smoke test")
        kwargs["device_map"] = {"": 0}
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_config["name"], **kwargs)
    if model_config["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif torch.cuda.is_available():
        model = model.to("cuda")

    max_rank = experiment["server_max_rank"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=max_rank,
            lora_alpha=max_rank,
            lora_dropout=0.0,
            target_modules=model_config["target_modules"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.config.use_cache = False
    return tokenizer, model


def _train_client(
    model,
    tokenizer,
    dataset,
    client_indices: Sequence[int],
    *,
    rng: random.Random,
    active_rank: int,
    dataset_config: Mapping[str, Any],
    max_length: int,
    local_steps: int,
    gradient_accumulation_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    model.train()
    losses: list[float] = []
    label_column = dataset_config["label_column"]
    for _ in range(local_steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(gradient_accumulation_steps):
            selected = [rng.choice(client_indices) for _ in range(batch_size)]
            examples = [dataset[index] for index in selected]
            batch = _collate_examples(
                tokenizer,
                [(item, int(item[label_column])) for item in examples],
                dataset_config=dataset_config,
                max_length=max_length,
            )
            batch = _move_batch(batch, _model_input_device(model))
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            shifted_logits = outputs.logits[:, :-1, :].float()
            shifted_labels = batch["labels"][:, 1:]
            token_loss = F.cross_entropy(
                shifted_logits.transpose(1, 2),
                shifted_labels,
                ignore_index=-100,
                reduction="none",
            )
            label_mask = shifted_labels.ne(-100)
            if tokenizer.eos_token_id is not None:
                label_mask &= shifted_labels.ne(tokenizer.eos_token_id)
            loss = (
                (token_loss * label_mask).sum()
                / label_mask.sum().clamp_min(1)
                / gradient_accumulation_steps
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite local loss: {float(loss)}")
            loss.backward()
            losses.append(float(loss.detach()) * gradient_accumulation_steps)
        mask_inactive_rank_gradients(model, active_rank=active_rank)
        torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return _mean(losses)


@torch.no_grad()
def evaluate_classification(
    model,
    tokenizer,
    dataset,
    *,
    dataset_config: Mapping[str, Any],
    max_length: int,
    batch_size: int,
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    model.eval()
    correct = 0
    label_texts = _label_texts(dataset_config)
    num_labels = len(label_texts)
    label_totals = {label: 0 for label in range(num_labels)}
    label_correct = {label: 0 for label in range(num_labels)}
    brier_scores: list[float] = []
    true_nlls: list[float] = []
    class_nlls: list[float] = []
    true_label_nlls: list[float] = []
    true_eos_nlls: list[float] = []
    detail_rows: list[dict[str, Any]] = []
    device = _model_input_device(model)
    label_column = dataset_config["label_column"]
    examples = [(item, int(item[label_column])) for item in dataset]
    for start in range(0, len(examples), batch_size):
        group = examples[start : start + batch_size]
        candidates = [
            (item, candidate)
            for item, _ in group
            for candidate in range(num_labels)
        ]
        batch = _collate_examples(
            tokenizer,
            candidates,
            dataset_config=dataset_config,
            max_length=max_length,
        )
        labels = batch["labels"]
        model_inputs = _move_batch(batch, device)
        logits = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
        ).logits
        shifted_logits = logits[:, :-1, :].float()
        shifted_labels = labels[:, 1:].to(logits.device)
        token_loss = F.cross_entropy(
            shifted_logits.transpose(1, 2),
            shifted_labels,
            ignore_index=-100,
            reduction="none",
        )
        mask = shifted_labels.ne(-100)
        nll = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        eos_mask = mask & shifted_labels.eq(tokenizer.eos_token_id)
        label_mask = mask & ~eos_mask
        label_nll = (token_loss * label_mask).sum(dim=1) / label_mask.sum(dim=1).clamp_min(1)
        eos_nll = (token_loss * eos_mask).sum(dim=1) / eos_mask.sum(dim=1).clamp_min(1)
        scores = nll.view(len(group), num_labels).cpu()
        label_scores = label_nll.view(len(group), num_labels).cpu()
        eos_scores = eos_nll.view(len(group), num_labels).cpu()
        # Classification must compare candidate label likelihoods.  Including
        # EOS in the decision makes formatting likelihood dominate the task.
        probabilities = torch.softmax(-label_scores, dim=1)
        for offset, (row, label_row, eos_row, probs, (item, true_label)) in enumerate(
            zip(scores, label_scores, eos_scores, probabilities, group)
        ):
            prediction = int(torch.argmin(label_row).item())
            correct += int(prediction == true_label)
            label_totals[true_label] += 1
            label_correct[true_label] += int(prediction == true_label)
            true_nlls.append(float(row[true_label].item()))
            class_nll = float(-torch.log(probs[true_label].clamp_min(1e-12)).item())
            class_nlls.append(class_nll)
            binary_nll = class_nll if num_labels == 2 else None
            target = torch.zeros_like(probs)
            target[true_label] = 1.0
            brier = float(torch.sum((probs - target) ** 2).item())
            brier_scores.append(brier)
            true_label_nlls.append(float(label_row[true_label].item()))
            true_eos_nlls.append(float(eos_row[true_label].item()))
            wrong_label = int(
                torch.argmin(
                    label_row.masked_fill(
                        torch.arange(num_labels) == true_label,
                        float("inf"),
                    )
                ).item()
            )
            detail = {
                "eval_index": start + offset,
                "text": _example_text_for_details(item, dataset_config),
                "task": dataset_config.get("task", dataset_config["hub_path"]),
                "true_label": true_label,
                "predicted_label": prediction,
                "is_correct": int(prediction == true_label),
                "true_nll": float(row[true_label].item()),
                "binary_nll": binary_nll,
                "class_nll": class_nll,
                "brier": brier,
                "label_nll": float(label_row[true_label].item()),
                "eos_nll": float(eos_row[true_label].item()),
                "wrong_nll": float(label_row[wrong_label].item()),
                "nll_margin": float(
                    label_row[wrong_label].item() - label_row[true_label].item()
                ),
                "true_probability": float(probs[true_label].item()),
                "prediction_confidence": float(probs[prediction].item()),
            }
            for label in range(num_labels):
                detail[f"nll_label_{label}"] = float(label_row[label].item())
                detail[f"prob_label_{label}"] = float(probs[label].item())
            detail_rows.append(detail)
    balanced_accuracy = _mean(
        [
            label_correct[label] / label_totals[label]
            for label in range(num_labels)
            if label_totals[label]
        ]
    )
    return {
        "accuracy": correct / len(examples),
        "balanced_accuracy": balanced_accuracy,
        "brier": _mean(brier_scores),
        "nll": _mean(true_nlls),
        "class_nll": _mean(class_nlls),
        "binary_nll": _mean(class_nlls) if num_labels == 2 else None,
        "label_nll": _mean(true_label_nlls),
        "eos_nll": _mean(true_eos_nlls),
    }, detail_rows


def _make_classification_batch(
    model,
    tokenizer,
    dataset,
    *,
    dataset_config: Mapping[str, Any],
    max_length: int,
):
    return _make_classification_batches(
        model,
        tokenizer,
        dataset,
        dataset_config=dataset_config,
        max_length=max_length,
        batch_size=len(dataset) if dataset is not None else 0,
    )[0][0]


def _make_classification_batches(
    model,
    tokenizer,
    dataset,
    *,
    dataset_config: Mapping[str, Any],
    max_length: int,
    batch_size: int,
) -> list[tuple[dict[str, torch.Tensor], float]]:
    if dataset is None or len(dataset) == 0:
        raise ValueError("calibration dataset must be non-empty")
    if batch_size <= 0:
        raise ValueError("calibration gradient batch size must be positive")
    label_column = dataset_config["label_column"]
    examples = [(item, int(item[label_column])) for item in dataset]
    batches: list[tuple[dict[str, torch.Tensor], float]] = []
    for start in range(0, len(examples), batch_size):
        batch = _collate_examples(
            tokenizer,
            examples[start : start + batch_size],
            dataset_config=dataset_config,
            max_length=max_length,
        )
        if tokenizer.eos_token_id is not None:
            batch["labels"] = batch["labels"].masked_fill(
                batch["labels"].eq(tokenizer.eos_token_id),
                -100,
            )
        weight = float(batch["labels"].ne(-100).sum().item())
        if weight <= 0.0:
            raise ValueError("calibration batch has no supervised label tokens")
        batches.append((_move_batch(batch, _model_input_device(model)), weight))
    return batches


def _make_classification_candidate_batches(
    model,
    tokenizer,
    dataset,
    *,
    dataset_config: Mapping[str, Any],
    max_length: int,
    batch_size: int,
) -> list[tuple[dict[str, torch.Tensor], float]]:
    if dataset is None or len(dataset) == 0:
        raise ValueError("calibration dataset must be non-empty")
    if batch_size <= 0:
        raise ValueError("calibration gradient batch size must be positive")
    label_column = dataset_config["label_column"]
    label_count = len(_label_texts(dataset_config))
    examples = [(item, int(item[label_column])) for item in dataset]
    batches: list[tuple[dict[str, torch.Tensor], float]] = []
    for start in range(0, len(examples), batch_size):
        group = examples[start : start + batch_size]
        candidates = [
            (item, candidate)
            for item, _ in group
            for candidate in range(label_count)
        ]
        batch = _collate_examples(
            tokenizer,
            candidates,
            dataset_config=dataset_config,
            max_length=max_length,
        )
        batch["class_labels"] = torch.tensor(
            [true_label for _, true_label in group],
            dtype=torch.long,
        )
        batches.append(
            (_move_batch(batch, _model_input_device(model)), float(len(group)))
        )
    return batches


def _classification_candidate_label_nlls(
    model,
    batch: Mapping[str, torch.Tensor],
    *,
    eos_token_id: int | None,
) -> torch.Tensor:
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).logits
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = batch["labels"][:, 1:].to(logits.device)
    token_loss = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        ignore_index=-100,
        reduction="none",
    )
    label_mask = shifted_labels.ne(-100)
    if eos_token_id is not None:
        label_mask &= shifted_labels.ne(eos_token_id)
    candidate_nll = (
        (token_loss * label_mask).sum(dim=1) / label_mask.sum(dim=1).clamp_min(1)
    )
    num_examples = int(batch["class_labels"].numel())
    if num_examples <= 0 or candidate_nll.numel() % num_examples:
        raise ValueError("candidate batch does not contain a complete label grid")
    return candidate_nll.view(num_examples, -1)


def _classification_candidate_nll_values(
    model,
    batch: Mapping[str, torch.Tensor],
    *,
    eos_token_id: int | None,
) -> torch.Tensor:
    candidate_nll = _classification_candidate_label_nlls(
        model,
        batch,
        eos_token_id=eos_token_id,
    )
    return F.cross_entropy(
        -candidate_nll,
        batch["class_labels"],
        reduction="none",
    )


def _classification_candidate_nll_loss(
    model,
    batch: Mapping[str, torch.Tensor],
    *,
    eos_token_id: int | None,
) -> torch.Tensor:
    return _classification_candidate_nll_values(
        model,
        batch,
        eos_token_id=eos_token_id,
    ).mean()


def _per_example_classification_losses(
    model,
    tokenizer,
    dataset,
    *,
    dataset_config: Mapping[str, Any],
    max_length: int,
    batch_size: int,
    objective: str = "label_nll",
) -> torch.Tensor:
    if dataset is None or len(dataset) == 0:
        raise ValueError("calibration gate dataset must be non-empty")
    model.eval()
    values: list[torch.Tensor] = []
    device = _model_input_device(model)
    label_column = dataset_config["label_column"]
    examples = [(item, int(item[label_column])) for item in dataset]
    if objective not in {"label_nll", "class_nll"}:
        raise ValueError("classification objective must be 'label_nll' or 'class_nll'")
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            group = examples[start : start + batch_size]
            if objective == "class_nll":
                candidates = [
                    (item, candidate)
                    for item, _ in group
                    for candidate in range(len(_label_texts(dataset_config)))
                ]
                batch = _collate_examples(
                    tokenizer,
                    candidates,
                    dataset_config=dataset_config,
                    max_length=max_length,
                )
                batch["class_labels"] = torch.tensor(
                    [true_label for _, true_label in group],
                    dtype=torch.long,
                )
                model_inputs = _move_batch(batch, device)
                values.append(
                    _classification_candidate_nll_values(
                        model,
                        model_inputs,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    .detach()
                    .cpu()
                )
                continue
            batch = _collate_examples(
                tokenizer,
                group,
                dataset_config=dataset_config,
                max_length=max_length,
            )
            labels = batch["labels"]
            model_inputs = _move_batch(batch, device)
            logits = model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
            ).logits
            shifted_logits = logits[:, :-1, :].float()
            shifted_labels = labels[:, 1:].to(logits.device)
            token_loss = F.cross_entropy(
                shifted_logits.transpose(1, 2),
                shifted_labels,
                ignore_index=-100,
                reduction="none",
            )
            mask = shifted_labels.ne(-100)
            label_mask = mask
            if tokenizer.eos_token_id is not None:
                label_mask &= shifted_labels.ne(tokenizer.eos_token_id)
            values.append(
                ((token_loss * label_mask).sum(dim=1) / label_mask.sum(dim=1).clamp_min(1))
                .detach()
                .cpu()
            )
    return torch.cat(values)


def _mean_classification_loss(
    model,
    tokenizer,
    dataset,
    *,
    dataset_config: Mapping[str, Any],
    max_length: int,
    batch_size: int,
    objective: str = "label_nll",
) -> float:
    losses = _per_example_classification_losses(
        model,
        tokenizer,
        dataset,
        dataset_config=dataset_config,
        max_length=max_length,
        batch_size=batch_size,
        objective=objective,
    )
    return float(losses.mean().item())


def _aggregate_scaled_updates(
    current_state: Mapping[str, CompactSVD],
    updates: Mapping[str, CompactSVD],
    *,
    scale: float,
    experiment: Mapping[str, Any],
) -> dict[str, CompactSVD]:
    return {
        name: aggregate_compact_state(
            current_state[name],
            scale_compact_update(update, scale),
            weight=experiment["server_update_weight"],
            max_rank=experiment["server_max_rank"],
            rank_rtol=experiment["rank_rtol"],
        )
        for name, update in updates.items()
    }


def _rift_gate_state(
    model,
    tokenizer,
    current_state: Mapping[str, CompactSVD],
    filtered_updates: Mapping[str, CompactSVD],
    raw_updates: Mapping[str, CompactSVD],
    calibration_gate,
    *,
    dataset_config: Mapping[str, Any],
    max_length: int,
    batch_size: int,
    experiment: Mapping[str, Any],
    freshness: float,
) -> tuple[dict[str, CompactSVD], dict[str, CompactSVD], float, float, str]:
    candidates: list[tuple[str, float, Mapping[str, CompactSVD]]] = []
    if sum(update.rank for update in filtered_updates.values()):
        scales = sorted(
            {float(value) for value in experiment.get("rift_step_scales", [1.0, 0.5, 0.25, 0.125])},
            reverse=True,
        )
        if not scales or any(value <= 0.0 or value > 1.0 for value in scales):
            raise ValueError("rift_step_scales must be in (0, 1]")
        candidates.extend(("rank_filtered", scale, filtered_updates) for scale in scales)
    if bool(experiment.get("rift_include_freshness_fallback", True)):
        candidates.append(("freshness_fallback", freshness, raw_updates))

    current_losses = _per_example_classification_losses(
        model,
        tokenizer,
        calibration_gate,
        dataset_config=dataset_config,
        max_length=max_length,
        batch_size=batch_size,
        objective=str(experiment.get("calibration_gate_objective", "label_nll")),
    )
    selected_state = {name: value for name, value in current_state.items()}
    selected_updates = {
        name: scale_compact_update(update, 0.0) for name, update in raw_updates.items()
    }
    selected_scale = 0.0
    selected_delta = 0.0
    selected_route = "reject"
    for route, scale, updates in candidates:
        candidate_state = _aggregate_scaled_updates(
            current_state,
            updates,
            scale=scale,
            experiment=experiment,
        )
        load_compact_adapter_state(
            model,
            candidate_state,
            active_rank=experiment["server_max_rank"],
            initialize_free_directions=False,
        )
        candidate_losses = _per_example_classification_losses(
            model,
            tokenizer,
            calibration_gate,
            dataset_config=dataset_config,
            max_length=max_length,
            batch_size=batch_size,
            objective=str(experiment.get("calibration_gate_objective", "label_nll")),
        )
        mean_delta = float((candidate_losses - current_losses).mean().item())
        if mean_delta <= float(experiment.get("rift_max_mean_increase", 0.0)) and (
            selected_route == "reject" or mean_delta < selected_delta
        ):
            selected_state = candidate_state
            selected_updates = {
                name: scale_compact_update(update, scale) for name, update in updates.items()
            }
            selected_scale = scale
            selected_delta = mean_delta
            selected_route = route
    return selected_state, selected_updates, selected_scale, selected_delta, selected_route


def _whole_update_gate_state(
    model,
    tokenizer,
    current_state: Mapping[str, CompactSVD],
    raw_updates: Mapping[str, CompactSVD],
    calibration_gate,
    *,
    dataset_config: Mapping[str, Any],
    max_length: int,
    batch_size: int,
    experiment: Mapping[str, Any],
    freshness: float,
) -> tuple[dict[str, CompactSVD], dict[str, CompactSVD], float, float, str]:
    scales = sorted(
        {
            float(value)
            for value in experiment.get(
                "alignfed_calibration_scales",
                [1.0, 0.5, 0.25, 0.125, freshness],
            )
        },
        reverse=True,
    )
    if not scales or any(value < 0.0 or value > 1.0 for value in scales):
        raise ValueError("alignfed_calibration_scales must be in [0, 1]")
    current_losses = _per_example_classification_losses(
        model,
        tokenizer,
        calibration_gate,
        dataset_config=dataset_config,
        max_length=max_length,
        batch_size=batch_size,
        objective=str(experiment.get("calibration_gate_objective", "label_nll")),
    )
    selected_state = {name: value for name, value in current_state.items()}
    selected_updates = {
        name: scale_compact_update(update, 0.0) for name, update in raw_updates.items()
    }
    selected_scale = 0.0
    selected_delta = 0.0
    selected_route = "reject"
    for scale in scales:
        candidate_state = _aggregate_scaled_updates(
            current_state,
            raw_updates,
            scale=scale,
            experiment=experiment,
        )
        load_compact_adapter_state(
            model,
            candidate_state,
            active_rank=experiment["server_max_rank"],
            initialize_free_directions=False,
        )
        candidate_losses = _per_example_classification_losses(
            model,
            tokenizer,
            calibration_gate,
            dataset_config=dataset_config,
            max_length=max_length,
            batch_size=batch_size,
            objective=str(experiment.get("calibration_gate_objective", "label_nll")),
        )
        mean_delta = float((candidate_losses - current_losses).mean().item())
        if mean_delta <= 0.0 and (selected_route == "reject" or mean_delta < selected_delta):
            selected_state = candidate_state
            selected_updates = {
                name: scale_compact_update(update, scale) for name, update in raw_updates.items()
            }
            selected_scale = scale
            selected_delta = mean_delta
            selected_route = "whole_update_gate"
    return selected_state, selected_updates, selected_scale, selected_delta, selected_route


def _collate_examples(
    tokenizer,
    examples: Sequence[tuple[Mapping[str, Any], int]],
    *,
    dataset_config: Mapping[str, Any],
    max_length: int,
):
    encoded: list[tuple[list[int], list[int]]] = []
    eos = tokenizer.eos_token or ""
    label_texts = _label_texts(dataset_config)
    for item, label in examples:
        prompt = _prompt_for_example(item, dataset_config)
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(label_texts[label] + eos, add_special_tokens=False)[
            "input_ids"
        ]
        prompt_budget = max(1, max_length - len(target_ids))
        prompt_ids = prompt_ids[:prompt_budget]
        input_ids = (prompt_ids + target_ids)[:max_length]
        labels = [-100] * len(prompt_ids) + target_ids
        labels = labels[: len(input_ids)]
        encoded.append((input_ids, labels))

    width = max(len(ids) for ids, _ in encoded)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    for input_ids, labels in encoded:
        padding = width - len(input_ids)
        input_rows.append(input_ids + [tokenizer.pad_token_id] * padding)
        label_rows.append(labels + [-100] * padding)
        attention_rows.append([1] * len(input_ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def _label_texts(dataset_config: Mapping[str, Any]) -> list[str]:
    task = str(dataset_config.get("task", dataset_config.get("subset", ""))).lower()
    expected = 3 if task == "mnli" else 2
    configured = dataset_config.get("label_texts")
    if configured is not None:
        if len(configured) != expected:
            raise ValueError(
                f"{task} requires exactly {expected} label_texts, got {len(configured)}"
            )
        return [str(value) for value in configured]
    if task not in DEFAULT_LABEL_TEXTS:
        raise ValueError(
            "dataset.label_texts is required for unsupported classification task "
            f"{task!r}"
        )
    return DEFAULT_LABEL_TEXTS[task]


def _prompt_for_example(item: Mapping[str, Any], dataset_config: Mapping[str, Any]) -> str:
    task = str(dataset_config.get("task", dataset_config.get("subset", ""))).lower()
    if task == "sst2":
        text_column = dataset_config.get("text_column", "sentence")
        return (
            "Classify the sentiment of this movie review as negative or positive.\n"
            f"Review: {item[text_column]}\nSentiment:"
        )
    if task == "qnli":
        question_column, sentence_column = _qnli_columns(dataset_config)
        return (
            "Answer whether the sentence contains the answer to the question.\n"
            f"Question: {item[question_column]}\n"
            f"Sentence: {item[sentence_column]}\nAnswer:"
        )
    if task == "mnli":
        premise_column, hypothesis_column = _mnli_columns(dataset_config)
        return (
            "Classify the relationship between the premise and hypothesis as "
            "entailment, neutral, or contradiction.\n"
            f"Premise: {item[premise_column]}\n"
            f"Hypothesis: {item[hypothesis_column]}\nRelationship:"
        )
    template = dataset_config.get("prompt_template")
    if template is None:
        raise ValueError(f"unsupported task for prompt construction: {task!r}")
    return str(template).format(**item)


def _example_text_for_details(
    item: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
) -> str:
    task = str(dataset_config.get("task", dataset_config.get("subset", ""))).lower()
    if task == "sst2":
        return str(item[dataset_config.get("text_column", "sentence")])
    if task == "qnli":
        question_column, sentence_column = _qnli_columns(dataset_config)
        return (
            f"Q: {item[question_column]} "
            f"S: {item[sentence_column]}"
        )
    if task == "mnli":
        premise_column, hypothesis_column = _mnli_columns(dataset_config)
        return (
            f"P: {item[premise_column]} "
            f"H: {item[hypothesis_column]}"
        )
    text_column = dataset_config.get("text_column")
    return str(item[text_column]) if text_column is not None else str(dict(item))


def _build_clients(experiment: Mapping[str, Any], partitions: Sequence[Sequence[int]]):
    return [
        ClientProfile(
            client_id=str(client_id),
            rank=experiment["client_ranks"][client_id],
            num_samples=len(partitions[client_id]),
            compute_time=experiment["compute_times"][client_id],
        )
        for client_id in range(experiment["num_clients"])
    ]


def _build_partitions(
    train,
    *,
    label_column: str,
    experiment: Mapping[str, Any],
    seed: int,
) -> list[list[int]]:
    partition_mode = experiment.get("partition_mode", "label_shard")
    if partition_mode == "iid":
        return iid_partition_indices(
            len(train),
            experiment["num_clients"],
            seed=seed,
        )
    if partition_mode == "label_shard":
        return label_shard_partition_indices(
            list(train[label_column]),
            experiment["num_clients"],
            shards_per_client=experiment["shards_per_client"],
            seed=seed,
        )
    raise ValueError(f"unsupported partition_mode: {partition_mode}")


def _state_to_cpu(state: Mapping[str, CompactSVD]) -> dict[str, CompactSVD]:
    return {
        name: CompactSVD(
            value.u.detach().to(device="cpu", dtype=torch.float32).clone(),
            value.s.detach().to(device="cpu", dtype=torch.float32).clone(),
            value.v.detach().to(device="cpu", dtype=torch.float32).clone(),
        )
        for name, value in state.items()
    }


def _model_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device):
    return {name: value.to(device) for name, value in batch.items()}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()


def _mean(values: Sequence[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _optional_difference(
    left: float | None,
    right: float | None,
) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _nearest_rank_percentile(values: Sequence[int | float], quantile: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _label_histogram(labels: Sequence[int | str]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for label in labels:
        key = str(label)
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items()))


def _dataset_label_histogram(dataset, label_column: str) -> dict[str, int]:
    if dataset is None:
        return {}
    return _label_histogram(list(dataset[label_column]))


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    canonical = copy.deepcopy(dict(config))
    canonical.pop("output_dir", None)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_worktree_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.model_name is not None:
        config["model"]["name"] = args.model_name
    if args.collected_returns is not None:
        config["experiment"]["collected_returns"] = args.collected_returns
    if args.local_steps is not None:
        config["experiment"]["local_steps"] = args.local_steps
    if args.eval_examples is not None:
        config["dataset"]["eval_examples"] = args.eval_examples
    if args.eval_offset is not None:
        config["dataset"]["eval_offset"] = args.eval_offset
    if args.eval_shuffle_seed is not None:
        config["dataset"]["eval_shuffle_seed"] = args.eval_shuffle_seed
    if args.eval_split is not None:
        config["dataset"]["eval_split"] = args.eval_split
    if args.partition_mode is not None:
        config["experiment"]["partition_mode"] = args.partition_mode
    if args.reserve_eval_from_train:
        config["dataset"]["reserve_eval_from_train"] = True
    if args.residual_beta is not None:
        config["experiment"]["residual_beta"] = args.residual_beta
    if args.residual_staleness_center is not None:
        config["experiment"]["residual_staleness_center"] = args.residual_staleness_center
    if args.residual_staleness_temperature is not None:
        config["experiment"]["residual_staleness_temperature"] = (
            args.residual_staleness_temperature
        )
    if args.no_4bit:
        config["model"]["load_in_4bit"] = False


def _validate_config(config: Mapping[str, Any], method: str) -> None:
    config = _resolve_single_regime_config(config)
    experiment = config["experiment"]
    dataset = config["dataset"]
    model = config["model"]
    num_clients = experiment["num_clients"]
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    if "target_modules" not in model:
        raise ValueError(
            "3B runner requires model.target_modules; legacy week4 BERT configs "
            "use target_suffixes and must be run with their original runner"
        )
    for field in ("max_train_examples", "eval_examples"):
        if field not in dataset:
            raise ValueError(f"3B runner requires dataset.{field}")
    task = str(dataset.get("task", dataset.get("subset", ""))).lower()
    if task == "sst2" and "text_column" not in dataset:
        raise ValueError("SST-2 config requires text_column")
    if task == "qnli":
        has_legacy_pair = "question_column" in dataset and "sentence_column" in dataset
        has_text_pair = "text_column" in dataset and "text_pair_column" in dataset
        if not (has_legacy_pair or has_text_pair):
            raise ValueError(
                "QNLI config requires either question_column/sentence_column "
                "or text_column/text_pair_column"
            )
    if task == "mnli":
        has_named_pair = "premise_column" in dataset and "hypothesis_column" in dataset
        has_text_pair = "text_column" in dataset and "text_pair_column" in dataset
        if not (has_named_pair or has_text_pair):
            raise ValueError(
                "MNLI config requires premise_column/hypothesis_column "
                "or text_column/text_pair_column"
            )
        if len(_label_texts(dataset)) != 3:
            raise ValueError("MNLI requires exactly three label_texts")
    if experiment.get("partition_mode", "label_shard") not in {"iid", "label_shard"}:
        raise ValueError("partition_mode must be 'iid' or 'label_shard'")
    if len(experiment["client_ranks"]) != num_clients:
        raise ValueError("client_ranks length must equal num_clients")
    if len(experiment["compute_times"]) != num_clients:
        raise ValueError("compute_times length must equal num_clients")
    if max(experiment["client_ranks"]) > experiment["server_max_rank"]:
        raise ValueError("client rank cannot exceed server_max_rank")
    if experiment["adaptive_max_rank"] > experiment["server_max_rank"]:
        raise ValueError("adaptive_max_rank cannot exceed server_max_rank")
    if experiment["warmup_returns"] < 1 or experiment["collected_returns"] < 1:
        raise ValueError("warmup_returns and collected_returns must be positive")
    if int(experiment.get("late_tau", 8)) < 0:
        raise ValueError("late_tau must be non-negative")
    if int(experiment.get("extreme_tau", 16)) <= int(experiment.get("late_tau", 8)):
        raise ValueError("extreme_tau must be greater than late_tau")
    if int(experiment.get("buffer_size", 1)) != 1:
        raise ValueError(
            "run_kaggle_3b currently applies one returned update at a time; "
            "use buffer_size=1 for classification runs until grouped state "
            "aggregation is implemented in the model runner"
        )
    if str(experiment.get("schedule_mode", "async")) != "async":
        raise ValueError(
            "run_kaggle_3b currently supports schedule_mode='async' only"
        )
    if int(experiment.get("monitor_examples", 0)) <= 0:
        raise ValueError("monitor_examples must be positive to compute harmful metrics")
    if method in {"rift", "spectral_filter"} and int(
        experiment.get("calibration_gradient_examples", 0)
    ) <= 0:
        raise ValueError(f"{method} requires calibration_gradient_examples > 0")
    if int(experiment.get("calibration_gradient_batch_size", 1)) <= 0:
        raise ValueError("calibration_gradient_batch_size must be positive")
    for field in (
        "component_score_objective",
        "calibration_gate_objective",
        "monitor_objective",
    ):
        if str(experiment.get(field, "label_nll")) not in {"label_nll", "class_nll"}:
            raise ValueError(f"{field} must be 'label_nll' or 'class_nll'")
    if method in {"rift", "alignfed_calibration"} and int(
        experiment.get("calibration_gate_examples", 0)
    ) <= 0:
        raise ValueError(f"{method} requires calibration_gate_examples > 0")
    if not 0.0 <= experiment.get("residual_beta", 0.5) <= 1.0:
        raise ValueError("residual_beta must be between zero and one")
    if experiment.get("residual_staleness_temperature", 1.0) <= 0.0:
        raise ValueError("residual_staleness_temperature must be positive")


def _dry_run_summary(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = _resolve_single_regime_config(config)
    experiment = config["experiment"]
    clients = [
        ClientProfile(
            client_id=str(index),
            rank=rank,
            num_samples=1,
            compute_time=experiment["compute_times"][index],
        )
        for index, rank in enumerate(experiment["client_ranks"])
    ]
    total = experiment["warmup_returns"] + experiment["collected_returns"]
    trace = AsyncEventSimulator(
        clients,
        seed=args.seed,
        buffer_size=int(experiment.get("buffer_size", 1)),
        schedule_mode=str(experiment.get("schedule_mode", "async")),
    ).run(max_returns=total)
    return {
        "valid": True,
        "method": args.method,
        "model": config["model"]["name"],
        "task": config["dataset"].get("task", config["dataset"].get("subset")),
        "regime": experiment.get("regime_name", "default"),
        "returns": total,
        "mean_staleness": _mean(trace.staleness_values),
        "max_staleness": max(trace.staleness_values),
        "buffer_size": int(experiment.get("buffer_size", 1)),
        "schedule_mode": str(experiment.get("schedule_mode", "async")),
        "groups": len(trace.groups),
        "server_max_rank": experiment["server_max_rank"],
        "partition_mode": experiment.get("partition_mode", "label_shard"),
        "load_in_4bit": config["model"]["load_in_4bit"],
    }


def _qnli_columns(dataset_config: Mapping[str, Any]) -> tuple[str, str]:
    question_column = str(dataset_config.get("question_column", dataset_config.get("text_column", "question")))
    sentence_column = str(
        dataset_config.get("sentence_column", dataset_config.get("text_pair_column", "sentence"))
    )
    return question_column, sentence_column


def _mnli_columns(dataset_config: Mapping[str, Any]) -> tuple[str, str]:
    premise_column = str(
        dataset_config.get("premise_column", dataset_config.get("text_column", "premise"))
    )
    hypothesis_column = str(
        dataset_config.get(
            "hypothesis_column", dataset_config.get("text_pair_column", "hypothesis")
        )
    )
    return premise_column, hypothesis_column


def _resolve_single_regime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    experiment = resolved.get("experiment")
    if not isinstance(experiment, dict):
        return resolved
    if "client_ranks" in experiment and "compute_times" in experiment:
        return resolved
    regimes = experiment.get("regimes")
    if not isinstance(regimes, list) or len(regimes) != 1:
        missing = [
            name
            for name in ("client_ranks", "compute_times")
            if name not in experiment
        ]
        if missing:
            raise ValueError(
                "experiment requires explicit "
                + ", ".join(missing)
                + " or a single regime entry"
            )
        return resolved
    regime = dict(regimes[0])
    if "client_ranks" not in experiment:
        ranks = regime.get("client_ranks", regime.get("ranks"))
        if ranks is None:
            raise ValueError("single regime config requires client_ranks or ranks")
        experiment["client_ranks"] = list(ranks)
    if "compute_times" not in experiment:
        compute_times = regime.get("compute_times")
        if compute_times is None:
            raise ValueError("single regime config requires compute_times")
        experiment["compute_times"] = list(compute_times)
    if "partition_mode" not in experiment:
        partition_mode = regime.get("partition_mode", regime.get("partition"))
        if partition_mode is not None:
            experiment["partition_mode"] = partition_mode
    if "regime_name" not in experiment and "name" in regime:
        experiment["regime_name"] = regime["name"]
    return resolved


if __name__ == "__main__":
    main()

