from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping, Sequence
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

from vastlora.asyncfl import AsyncEventSimulator, ClientProfile
from vastlora.data import iid_partition_indices, label_shard_partition_indices
from vastlora.lowrank import CompactSVD
from vastlora.scale import (
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
    score_compact_components_with_hooks,
    transport_compact_update,
)
from vastlora.scale.tradeoff import reserved_train_eval_indices


LABEL_TEXT = {0: " negative", 1: " positive"}
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
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


def run_experiment(config: dict[str, Any], *, method: str, seed: int) -> dict[str, Any]:
    from datasets import load_dataset

    _seed_everything(seed)
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
    shuffled_train = train.shuffle(seed=seed)
    calibration_total = calibration_gradient_examples + calibration_gate_examples
    if calibration_total + int(dataset_config["max_train_examples"]) > len(shuffled_train):
        raise ValueError("calibration plus max_train_examples exceeds available train data")
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
    train = shuffled_train.select(
        range(calibration_total, calibration_total + dataset_config["max_train_examples"])
    )

    label_column = dataset_config["label_column"]
    partitions = _build_partitions(
        train,
        label_column=label_column,
        experiment=experiment,
        seed=seed,
    )
    clients = _build_clients(experiment, partitions)
    total_returns = experiment["warmup_returns"] + experiment["collected_returns"]
    trace = AsyncEventSimulator(clients, seed=seed).run(max_returns=total_returns)

    tokenizer, model = _load_model(config)
    gradient_batch = (
        _make_sentiment_batch(
            model,
            tokenizer,
            calibration_gradient,
            text_column=dataset_config["text_column"],
            label_column=label_column,
            max_length=config["model"]["max_length"],
        )
        if calibration_gradient is not None
        else None
    )
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

    baseline, baseline_details = evaluate_sentiment(
        model,
        tokenizer,
        validation,
        text_column=dataset_config["text_column"],
        label_column=label_column,
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
            text_column=dataset_config["text_column"],
            label_column=label_column,
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
                        text_column=dataset_config["text_column"],
                        label_column=label_column,
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
                if gradient_batch is None:
                    raise ValueError(f"{accepted_method} requires calibration_gradient_examples")
                scores = score_compact_components_with_hooks(
                    model,
                    innovations,
                    gradient_batch,
                )
                filtered = filter_compact_by_scores(
                    innovations,
                    scores.scores,
                    minimum_predicted_gain=float(
                        experiment.get("rift_minimum_predicted_gain", 0.0)
                    ),
                    keep_nonpositive=False,
                )
                retained_ranks.append(scores.retained_rank)
                total_ranks.append(scores.total_rank)
                predicted_gains.append(scores.predicted_gain)
                if accepted_method == "spectral_filter":
                    next_state = _aggregate_scaled_updates(
                        current_state,
                        filtered,
                        scale=float(experiment.get("spectral_filter_scale", 1.0)),
                        experiment=experiment,
                    )
                    accepted_updates = filtered
                    accepted_scales.append(float(experiment.get("spectral_filter_scale", 1.0)))
                    accepted_routes.append("gradient_filter_no_gate")
                else:
                    next_state, accepted_updates, scale, mean_delta, route = (
                        _rift_gate_state(
                            model,
                            tokenizer,
                            current_state,
                            filtered,
                            innovations,
                            calibration_gate,
                            text_column=dataset_config["text_column"],
                            label_column=label_column,
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
                "method": accepted_method,
                "local_loss": local_loss,
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
    final, final_details = evaluate_sentiment(
        model,
        tokenizer,
        validation,
        text_column=dataset_config["text_column"],
        label_column=label_column,
        max_length=config["model"]["max_length"],
        batch_size=experiment["eval_batch_size"],
    )
    runtime = time.perf_counter() - start_time
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    metrics = {
        "baseline_accuracy": baseline["accuracy"],
        "baseline_balanced_accuracy": baseline["balanced_accuracy"],
        "baseline_brier": baseline["brier"],
        "baseline_nll": baseline["nll"],
        "baseline_binary_nll": baseline["binary_nll"],
        "baseline_label_nll": baseline["label_nll"],
        "baseline_eos_nll": baseline["eos_nll"],
        "final_accuracy": final["accuracy"],
        "final_balanced_accuracy": final["balanced_accuracy"],
        "final_brier": final["brier"],
        "final_nll": final["nll"],
        "final_binary_nll": final["binary_nll"],
        "final_label_nll": final["label_nll"],
        "final_eos_nll": final["eos_nll"],
        "accuracy_change_pp": 100.0 * (final["accuracy"] - baseline["accuracy"]),
        "nll_change": final["nll"] - baseline["nll"],
        "binary_nll_change": final["binary_nll"] - baseline["binary_nll"],
        "mean_local_loss": _mean([row["local_loss"] for row in event_rows]),
        "mean_staleness": _mean([row["staleness"] for row in event_rows]),
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
        "schema_version": 1,
        "method": method,
        "seed": seed,
        "model": config["model"]["name"],
        "git_commit": _git_commit(),
        "config": config,
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
    text_column: str,
    label_column: str,
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
    for _ in range(local_steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(gradient_accumulation_steps):
            selected = [rng.choice(client_indices) for _ in range(batch_size)]
            examples = [dataset[index] for index in selected]
            batch = _collate_examples(
                tokenizer,
                [(item[text_column], int(item[label_column])) for item in examples],
                max_length=max_length,
            )
            batch = _move_batch(batch, _model_input_device(model))
            loss = model(**batch).loss / gradient_accumulation_steps
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
def evaluate_sentiment(
    model,
    tokenizer,
    dataset,
    *,
    text_column: str,
    label_column: str,
    max_length: int,
    batch_size: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    correct = 0
    label_totals = {0: 0, 1: 0}
    label_correct = {0: 0, 1: 0}
    brier_scores: list[float] = []
    true_nlls: list[float] = []
    binary_nlls: list[float] = []
    true_label_nlls: list[float] = []
    true_eos_nlls: list[float] = []
    detail_rows: list[dict[str, Any]] = []
    device = _model_input_device(model)
    examples = [(item[text_column], int(item[label_column])) for item in dataset]
    for start in range(0, len(examples), batch_size):
        group = examples[start : start + batch_size]
        candidates = [(text, candidate) for text, _ in group for candidate in (0, 1)]
        batch = _collate_examples(tokenizer, candidates, max_length=max_length)
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
        scores = nll.view(len(group), 2).cpu()
        label_scores = label_nll.view(len(group), 2).cpu()
        eos_scores = eos_nll.view(len(group), 2).cpu()
        probabilities = torch.softmax(-scores, dim=1)
        for offset, (row, label_row, eos_row, probs, (text, true_label)) in enumerate(
            zip(scores, label_scores, eos_scores, probabilities, group)
        ):
            prediction = int(torch.argmin(row).item())
            correct += int(prediction == true_label)
            label_totals[true_label] += 1
            label_correct[true_label] += int(prediction == true_label)
            true_nlls.append(float(row[true_label].item()))
            binary_nll = float(-torch.log(probs[true_label].clamp_min(1e-12)).item())
            binary_nlls.append(binary_nll)
            brier = float((probs[1].item() - true_label) ** 2)
            brier_scores.append(brier)
            true_label_nlls.append(float(label_row[true_label].item()))
            true_eos_nlls.append(float(eos_row[true_label].item()))
            wrong_label = 1 - true_label
            detail_rows.append(
                {
                    "eval_index": start + offset,
                    "text": text,
                    "true_label": true_label,
                    "predicted_label": prediction,
                    "is_correct": int(prediction == true_label),
                    "nll_negative": float(row[0].item()),
                    "nll_positive": float(row[1].item()),
                    "true_nll": float(row[true_label].item()),
                    "binary_nll": binary_nll,
                    "brier": brier,
                    "label_nll": float(label_row[true_label].item()),
                    "eos_nll": float(eos_row[true_label].item()),
                    "wrong_nll": float(row[wrong_label].item()),
                    "nll_margin": float(row[wrong_label].item() - row[true_label].item()),
                    "prob_negative": float(probs[0].item()),
                    "prob_positive": float(probs[1].item()),
                    "true_probability": float(probs[true_label].item()),
                    "prediction_confidence": float(probs[prediction].item()),
                }
            )
    balanced_accuracy = _mean(
        [label_correct[label] / label_totals[label] for label in (0, 1) if label_totals[label]]
    )
    return {
        "accuracy": correct / len(examples),
        "balanced_accuracy": balanced_accuracy,
        "brier": _mean(brier_scores),
        "nll": _mean(true_nlls),
        "binary_nll": _mean(binary_nlls),
        "label_nll": _mean(true_label_nlls),
        "eos_nll": _mean(true_eos_nlls),
    }, detail_rows


def _make_sentiment_batch(
    model,
    tokenizer,
    dataset,
    *,
    text_column: str,
    label_column: str,
    max_length: int,
):
    if dataset is None or len(dataset) == 0:
        raise ValueError("calibration dataset must be non-empty")
    examples = [(item[text_column], int(item[label_column])) for item in dataset]
    return _move_batch(
        _collate_examples(tokenizer, examples, max_length=max_length),
        _model_input_device(model),
    )


def _per_example_sentiment_losses(
    model,
    tokenizer,
    dataset,
    *,
    text_column: str,
    label_column: str,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    if dataset is None or len(dataset) == 0:
        raise ValueError("calibration gate dataset must be non-empty")
    model.eval()
    values: list[torch.Tensor] = []
    device = _model_input_device(model)
    examples = [(item[text_column], int(item[label_column])) for item in dataset]
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            group = examples[start : start + batch_size]
            batch = _collate_examples(tokenizer, group, max_length=max_length)
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
            values.append(
                ((token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1))
                .detach()
                .cpu()
            )
    return torch.cat(values)


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
    text_column: str,
    label_column: str,
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

    current_losses = _per_example_sentiment_losses(
        model,
        tokenizer,
        calibration_gate,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
        batch_size=batch_size,
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
        candidate_losses = _per_example_sentiment_losses(
            model,
            tokenizer,
            calibration_gate,
            text_column=text_column,
            label_column=label_column,
            max_length=max_length,
            batch_size=batch_size,
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
    text_column: str,
    label_column: str,
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
    current_losses = _per_example_sentiment_losses(
        model,
        tokenizer,
        calibration_gate,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
        batch_size=batch_size,
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
        candidate_losses = _per_example_sentiment_losses(
            model,
            tokenizer,
            calibration_gate,
            text_column=text_column,
            label_column=label_column,
            max_length=max_length,
            batch_size=batch_size,
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


def _collate_examples(tokenizer, examples: Sequence[tuple[str, int]], *, max_length: int):
    encoded: list[tuple[list[int], list[int]]] = []
    eos = tokenizer.eos_token or ""
    for sentence, label in examples:
        prompt = (
            "Classify the sentiment of this movie review as negative or positive.\n"
            f"Review: {sentence}\nSentiment:"
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(LABEL_TEXT[label] + eos, add_special_tokens=False)["input_ids"]
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


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


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
    experiment = config["experiment"]
    num_clients = experiment["num_clients"]
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
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
    if method in {"rift", "spectral_filter"} and int(
        experiment.get("calibration_gradient_examples", 0)
    ) <= 0:
        raise ValueError(f"{method} requires calibration_gradient_examples > 0")
    if method in {"rift", "alignfed_calibration"} and int(
        experiment.get("calibration_gate_examples", 0)
    ) <= 0:
        raise ValueError(f"{method} requires calibration_gate_examples > 0")
    if not 0.0 <= experiment.get("residual_beta", 0.5) <= 1.0:
        raise ValueError("residual_beta must be between zero and one")
    if experiment.get("residual_staleness_temperature", 1.0) <= 0.0:
        raise ValueError("residual_staleness_temperature must be positive")


def _dry_run_summary(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
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
    trace = AsyncEventSimulator(clients, seed=args.seed).run(max_returns=total)
    return {
        "valid": True,
        "method": args.method,
        "model": config["model"]["name"],
        "returns": total,
        "mean_staleness": _mean(trace.staleness_values),
        "max_staleness": max(trace.staleness_values),
        "server_max_rank": experiment["server_max_rank"],
        "partition_mode": experiment.get("partition_mode", "label_shard"),
        "load_in_4bit": config["model"]["load_in_4bit"],
    }


if __name__ == "__main__":
    main()
