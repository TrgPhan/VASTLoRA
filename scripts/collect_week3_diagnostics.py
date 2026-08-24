from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vastlora.asyncfl import AsyncEventSimulator, ClientProfile
from vastlora.data import iid_partition_indices, label_shard_partition_indices
from vastlora.diagnostics import (
    analyze_innovation_geometry,
    persistent_temporal_projection,
    residual_budget_transport,
    subspace_lattice_transport,
    transport_innovations,
    validate_diagnostic_dataframe,
)
from vastlora.lora import (
    add_dense_innovation,
    get_local_innovations,
    get_server_adapter_state,
    inject_diagnostic_lora,
    local_adapter_parameters,
    named_lora_modules,
    reset_local_adapters,
    reset_local_adapters_from_server,
    set_server_adapter_state,
    zero_local_adapters,
)
from vastlora.lowrank import CompactSVD, compact_svd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect replayable Week-3 LoRA diagnostics")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/week4_killtest.json")
    parser.add_argument("--regime", action="append", help="Run only the named regime")
    parser.add_argument("--seed", action="append", type=int, help="Run only the given seed")
    parser.add_argument("--collected-returns", type=int, help="Override returns per run")
    parser.add_argument("--history-size", type=int, help="Override reference history size")
    parser.add_argument("--reference-rank", type=int, help="Override reference rank")
    parser.add_argument(
        "--reference-singular-power",
        type=float,
        help="Weight temporal reference directions by singular value to this power",
    )
    parser.add_argument("--freshness-lambda", type=float, help="Override freshness lambda")
    parser.add_argument(
        "--residual-freshness-power",
        type=float,
        help="Exponent applied only to VAST residual freshness",
    )
    parser.add_argument(
        "--residual-budget",
        type=float,
        help="Residual norm budget relative to projected norm",
    )
    parser.add_argument(
        "--projection-scale-cap",
        type=float,
        help="Maximum projection energy compensation scale",
    )
    parser.add_argument("--persistent-max-rank", type=int)
    parser.add_argument("--persistent-short-history", type=int)
    parser.add_argument("--persistent-overlap-threshold", type=float)
    parser.add_argument("--risk-threshold", type=float)
    parser.add_argument("--server-update-weight", type=float, help="Override server update weight")
    parser.add_argument("--local-steps", type=int, help="Override local optimizer steps")
    parser.add_argument("--local-learning-rate", type=float, help="Override local learning rate")
    parser.add_argument(
        "--accept-method",
        choices=(
            "raw",
            "freshness",
            "projection",
            "projection_left",
            "projection_right",
            "persistent_projection",
            "projection_scaled",
            "residual_budget",
            "risk_switch",
            "saber",
            "union",
            "lattice",
            "lattice_sq",
            "vast",
        ),
        help="Method that advances the server trajectory",
    )
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--model-name", help="Override model checkpoint")
    parser.add_argument("--tokenizer-name", help="Override tokenizer checkpoint")
    parser.add_argument("--eval-examples", type=int, help="Override validation sample count")
    parser.add_argument("--federated-train-start", type=int, help="Exclude an initial train prefix")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--candidate-suite",
        choices=("core", "full"),
        default="full",
        help="Core skips expensive exploratory candidate evaluations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    experiment["candidate_suite"] = args.candidate_suite
    regimes = [
        regime
        for regime in experiment["regimes"]
        if not args.regime or regime["name"] in args.regime
    ]
    seeds = list(args.seed) if args.seed else list(experiment["seeds"])
    if not regimes or not seeds:
        raise ValueError("regime/seed filters selected no runs")
    if args.collected_returns is not None:
        experiment["collected_returns"] = args.collected_returns
    if args.history_size is not None:
        experiment["history_size"] = args.history_size
        experiment["warmup_returns"] = max(experiment["warmup_returns"], args.history_size)
    if args.reference_rank is not None:
        experiment["reference_rank"] = args.reference_rank
    if args.reference_singular_power is not None:
        experiment["reference_singular_power"] = args.reference_singular_power
    if args.freshness_lambda is not None:
        experiment["freshness_lambda"] = args.freshness_lambda
    if args.residual_freshness_power is not None:
        experiment["residual_freshness_power"] = args.residual_freshness_power
    if args.residual_budget is not None:
        experiment["residual_budget"] = args.residual_budget
    if args.projection_scale_cap is not None:
        experiment["projection_scale_cap"] = args.projection_scale_cap
    if args.persistent_max_rank is not None:
        experiment["persistent_max_rank"] = args.persistent_max_rank
    if args.persistent_short_history is not None:
        experiment["persistent_short_history"] = args.persistent_short_history
    if args.persistent_overlap_threshold is not None:
        experiment["persistent_overlap_threshold"] = args.persistent_overlap_threshold
    if args.risk_threshold is not None:
        experiment["risk_threshold"] = args.risk_threshold
    if args.server_update_weight is not None:
        experiment["server_update_weight"] = args.server_update_weight
    if args.local_steps is not None:
        experiment["local_steps"] = args.local_steps
    if args.local_learning_rate is not None:
        experiment["local_learning_rate"] = args.local_learning_rate
    if args.accept_method is not None:
        experiment["accept_method"] = args.accept_method
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.model_name is not None:
        config["model"]["name"] = args.model_name
    if args.tokenizer_name is not None:
        config["model"]["tokenizer_name"] = args.tokenizer_name
    if args.eval_examples is not None:
        experiment["eval_examples"] = args.eval_examples
    if args.federated_train_start is not None:
        config["dataset"]["federated_train_start"] = args.federated_train_start

    device = _resolve_device(args.device)
    _set_determinism(min(seeds))
    output_dir = ROOT / config["output_dir"]
    replay_dir = output_dir / "replay"
    run_dir = output_dir / "runs"
    replay_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = config["dataset"]
    raw = load_dataset(dataset_config["hub_path"], dataset_config["subset"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer_name"])
    tokenized_train = _tokenize(raw[dataset_config["train_split"]], tokenizer, config)
    tokenized_validation = _tokenize(
        raw[dataset_config["validation_split"]], tokenizer, config
    )
    eval_indices = _sample_indices(
        list(range(len(tokenized_validation))),
        experiment["eval_examples"],
        experiment["eval_seed"],
    )
    eval_batch = _make_batch(tokenized_validation, eval_indices, device)
    validation_hash = _hash_indices(eval_indices)
    dataset_fingerprint = _dataset_fingerprint(config, raw)

    all_frames: list[pd.DataFrame] = []
    for regime in regimes:
        for seed in seeds:
            frame = run_diagnostic(
                config=config,
                regime=regime,
                seed=seed,
                raw_train=raw[dataset_config["train_split"]],
                tokenized_train=tokenized_train,
                eval_batch=eval_batch,
                eval_indices=eval_indices,
                validation_hash=validation_hash,
                dataset_fingerprint=dataset_fingerprint,
                device=device,
                replay_dir=replay_dir,
            )
            frame.to_csv(run_dir / f"{regime['name']}_seed{seed}.csv", index=False)
            all_frames.append(frame)
            print(
                f"completed {regime['name']} seed={seed}: rows={len(frame)}, "
                f"harmful={(frame['raw_update_utility'] < 0).mean():.3f}"
            )

    combined = pd.concat(all_frames, ignore_index=True)
    output_csv = output_dir / "week3_diagnostics.csv"
    combined.to_csv(output_csv, index=False)
    validation = validate_diagnostic_dataframe(
        combined,
        min_stale_updates=min(100, max(1, int(0.9 * len(combined)))),
        artifact_root=output_dir,
    )
    (output_dir / "week3_diagnostics_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))
    if not validation["valid"]:
        raise RuntimeError("diagnostic dataframe did not pass validation")


def run_diagnostic(
    *,
    config: Mapping[str, Any],
    regime: Mapping[str, Any],
    seed: int,
    raw_train: Dataset,
    tokenized_train: Dataset,
    eval_batch: Mapping[str, torch.Tensor],
    eval_indices: Sequence[int],
    validation_hash: str,
    dataset_fingerprint: str,
    device: torch.device,
    replay_dir: Path,
) -> pd.DataFrame:
    experiment = config["experiment"]
    model_config = config["model"]
    dataset_config = config["dataset"]
    accept_method = str(experiment.get("accept_method", "raw"))
    run_id = f"{regime['name']}_seed{seed}_{accept_method}"
    _set_determinism(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["name"],
        attn_implementation="eager",
    )
    model.to(device)
    target_names = inject_diagnostic_lora(
        model,
        target_suffixes=tuple(model_config["target_suffixes"]),
    )
    initial_state = get_server_adapter_state(model, cpu=True)
    train_start = int(dataset_config.get("federated_train_start", 0))
    eligible_indices = list(range(train_start, len(raw_train)))
    labels = [
        int(raw_train[index][dataset_config["label_column"]])
        for index in eligible_indices
    ]
    local_partitions = _build_partitions(labels, regime, experiment["num_clients"], seed)
    partitions = [
        [eligible_indices[local_index] for local_index in partition]
        for partition in local_partitions
    ]
    ranks = [int(value) for value in regime["ranks"]]
    compute_times = [float(value) for value in regime["compute_times"]]
    if not (len(partitions) == len(ranks) == len(compute_times) == experiment["num_clients"]):
        raise ValueError("client partition, rank, and latency profiles must have equal length")

    clients = [
        ClientProfile(
            client_id=f"c{index:02d}",
            rank=ranks[index],
            num_samples=len(partitions[index]),
            compute_time=compute_times[index],
        )
        for index in range(experiment["num_clients"])
    ]
    total_returns = experiment["warmup_returns"] + experiment["collected_returns"]
    trace = AsyncEventSimulator(clients, seed=seed, buffer_size=1).run(max_returns=total_returns)
    snapshots: dict[int, dict[str, torch.Tensor]] = {0: initial_state}
    history: dict[str, list[CompactSVD]] = {name: [] for name in target_names}
    artifacts: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    current_loss: float | None = None
    start_time = time.perf_counter()
    artifact_relative = f"replay/{run_id}.pt"

    for event_index, event in enumerate(trace.records):
        base_state = snapshots[event.base_version]
        set_server_adapter_state(model, base_state)
        local_seed = seed * 1_000_000 + event_index * 101 + int(event.client_id[1:])
        factor_alpha = experiment["lora_scale_per_rank"] * event.rank
        if experiment.get("factor_init", "server") == "server":
            reset_local_adapters_from_server(
                model,
                event.rank,
                alpha=factor_alpha,
                seed=int(experiment.get("factor_seed", 2026)),
            )
        elif experiment["factor_init"] == "fresh":
            reset_local_adapters(
                model,
                event.rank,
                alpha=factor_alpha,
                seed=local_seed,
            )
        else:
            raise ValueError(f"unsupported factor_init: {experiment['factor_init']}")
        client_index = int(event.client_id[1:])
        sample_count = experiment["local_steps"] * experiment["local_batch_size"]
        local_indices = _sample_indices(partitions[client_index], sample_count, local_seed)
        training = _train_local_update(
            model,
            tokenized_train,
            local_indices,
            device=device,
            steps=experiment["local_steps"],
            batch_size=experiment["local_batch_size"],
            learning_rate=experiment["local_learning_rate"],
            weight_decay=experiment["weight_decay"],
            gradient_clip_norm=experiment["gradient_clip_norm"],
        )
        innovations = get_local_innovations(model)
        compact_updates = {
            name: compact_svd(update, rtol=experiment["rank_rtol"])
            for name, update in innovations.items()
        }
        zero_local_adapters(model)
        current_state = snapshots[event.arrival_version]

        if event_index >= experiment["warmup_returns"]:
            set_server_adapter_state(model, current_state)
            if current_loss is None:
                current_loss, current_accuracy = _evaluate_metrics(model, eval_batch)
            geometry = analyze_innovation_geometry(
                innovations,
                history,
                reference_rank=experiment["reference_rank"],
                history_size=experiment["history_size"],
                reference_decay=experiment["reference_decay"],
                reference_singular_power=float(
                    experiment.get("reference_singular_power", 0.0)
                ),
                rank_rtol=experiment["rank_rtol"],
            )
            persistent = persistent_temporal_projection(
                innovations,
                history,
                max_rank=int(experiment.get("persistent_max_rank", 4)),
                short_history_size=int(
                    experiment.get("persistent_short_history", 4)
                ),
                long_history_size=experiment["history_size"],
                overlap_threshold=float(
                    experiment.get("persistent_overlap_threshold", 0.7)
                ),
                reference_decay=experiment["reference_decay"],
                rank_rtol=experiment["rank_rtol"],
            )
            freshness = float(torch.exp(torch.tensor(-experiment["freshness_lambda"] * event.staleness)))
            raw_state = add_dense_innovation(
                current_state,
                innovations,
                weight=experiment["server_update_weight"],
            )
            freshness_state = add_dense_innovation(
                current_state,
                innovations,
                weight=experiment["server_update_weight"] * freshness,
            )
            vast_updates = transport_innovations(
                innovations,
                geometry.projected_updates,
                freshness=freshness
                ** float(experiment.get("residual_freshness_power", 1.0)),
            )
            residual_budget_updates = residual_budget_transport(
                innovations,
                geometry.projected_updates,
                freshness=freshness,
                residual_budget=float(experiment.get("residual_budget", 0.5)),
                projection_scale_cap=1.0,
            )
            projection_scaled_updates = residual_budget_transport(
                innovations,
                geometry.projected_updates,
                freshness=0.0,
                residual_budget=0.0,
                projection_scale_cap=float(
                    experiment.get("projection_scale_cap", 2.0)
                ),
            )
            saber_updates = residual_budget_transport(
                innovations,
                geometry.projected_updates,
                freshness=freshness,
                residual_budget=float(experiment.get("residual_budget", 0.5)),
                projection_scale_cap=float(
                    experiment.get("projection_scale_cap", 2.0)
                ),
            )
            union_updates = subspace_lattice_transport(
                innovations,
                geometry.projected_left_updates,
                geometry.projected_right_updates,
                geometry.projected_updates,
                single_side_weight=1.0,
            )
            lattice_updates = subspace_lattice_transport(
                innovations,
                geometry.projected_left_updates,
                geometry.projected_right_updates,
                geometry.projected_updates,
                single_side_weight=freshness,
            )
            lattice_sq_updates = subspace_lattice_transport(
                innovations,
                geometry.projected_left_updates,
                geometry.projected_right_updates,
                geometry.projected_updates,
                single_side_weight=freshness**2,
            )
            if event.staleness == 0:
                fresh_updates = {
                    name: update.dense() for name, update in innovations.items()
                }
                residual_budget_updates = fresh_updates
                saber_updates = fresh_updates
                lattice_updates = fresh_updates
                lattice_sq_updates = fresh_updates
            vast_state = add_dense_innovation(
                current_state,
                vast_updates,
                weight=experiment["server_update_weight"],
            )
            raw_loss = _loss_for_state(model, raw_state, eval_batch)
            freshness_loss = _loss_for_state(model, freshness_state, eval_batch)
            vast_loss = _loss_for_state(model, vast_state, eval_batch)
            projection_state = add_dense_innovation(
                current_state,
                geometry.projected_updates,
                weight=experiment["server_update_weight"],
            )
            persistent_projection_state = add_dense_innovation(
                current_state,
                persistent.projected_updates,
                weight=experiment["server_update_weight"],
            )
            projection_scaled_state = add_dense_innovation(
                current_state,
                projection_scaled_updates,
                weight=experiment["server_update_weight"],
            )
            residual_budget_state = add_dense_innovation(
                current_state,
                residual_budget_updates,
                weight=experiment["server_update_weight"],
            )
            saber_state = add_dense_innovation(
                current_state,
                saber_updates,
                weight=experiment["server_update_weight"],
            )
            projection_left_state = add_dense_innovation(
                current_state,
                geometry.projected_left_updates,
                weight=experiment["server_update_weight"],
            )
            projection_right_state = add_dense_innovation(
                current_state,
                geometry.projected_right_updates,
                weight=experiment["server_update_weight"],
            )
            union_state = add_dense_innovation(
                current_state,
                union_updates,
                weight=experiment["server_update_weight"],
            )
            lattice_state = add_dense_innovation(
                current_state,
                lattice_updates,
                weight=experiment["server_update_weight"],
            )
            lattice_sq_state = add_dense_innovation(
                current_state,
                lattice_sq_updates,
                weight=experiment["server_update_weight"],
            )
            projection_loss = _loss_for_state(model, projection_state, eval_batch)
            persistent_projection_loss = _loss_for_state(
                model, persistent_projection_state, eval_batch
            )
            if experiment.get("candidate_suite", "full") == "full":
                projection_scaled_loss = _loss_for_state(
                    model, projection_scaled_state, eval_batch
                )
                residual_budget_loss = _loss_for_state(
                    model, residual_budget_state, eval_batch
                )
                saber_loss = _loss_for_state(model, saber_state, eval_batch)
                projection_left_loss = _loss_for_state(
                    model, projection_left_state, eval_batch
                )
                projection_right_loss = _loss_for_state(
                    model, projection_right_state, eval_batch
                )
                union_loss = _loss_for_state(model, union_state, eval_batch)
                lattice_loss = _loss_for_state(model, lattice_state, eval_batch)
                lattice_sq_loss = _loss_for_state(model, lattice_sq_state, eval_batch)
            else:
                projection_scaled_loss = float("nan")
                residual_budget_loss = float("nan")
                saber_loss = float("nan")
                projection_left_loss = float("nan")
                projection_right_loss = float("nan")
                union_loss = float("nan")
                lattice_loss = float("nan")
                lattice_sq_loss = float("nan")
            candidate_states = {
                "raw": raw_state,
                "freshness": freshness_state,
                "projection": projection_state,
                "projection_left": projection_left_state,
                "projection_right": projection_right_state,
                "persistent_projection": persistent_projection_state,
                "projection_scaled": projection_scaled_state,
                "residual_budget": residual_budget_state,
                "saber": saber_state,
                "union": union_state,
                "lattice": lattice_state,
                "lattice_sq": lattice_sq_state,
                "vast": vast_state,
            }
            candidate_losses = {
                "raw": raw_loss,
                "freshness": freshness_loss,
                "projection": projection_loss,
                "projection_left": projection_left_loss,
                "projection_right": projection_right_loss,
                "persistent_projection": persistent_projection_loss,
                "projection_scaled": projection_scaled_loss,
                "residual_budget": residual_budget_loss,
                "saber": saber_loss,
                "union": union_loss,
                "lattice": lattice_loss,
                "lattice_sq": lattice_sq_loss,
                "vast": vast_loss,
            }
            candidate_updates = {
                "raw": {name: update.dense() for name, update in innovations.items()},
                "freshness": {
                    name: freshness * update.dense() for name, update in innovations.items()
                },
                "projection": geometry.projected_updates,
                "projection_left": geometry.projected_left_updates,
                "projection_right": geometry.projected_right_updates,
                "persistent_projection": persistent.projected_updates,
                "projection_scaled": projection_scaled_updates,
                "residual_budget": residual_budget_updates,
                "saber": saber_updates,
                "union": union_updates,
                "lattice": lattice_updates,
                "lattice_sq": lattice_sq_updates,
                "vast": vast_updates,
            }
            risk_score = training.grad_norm_first / max(geometry.fro_norm, 1e-12)
            risk_uses_projection = risk_score >= float(
                experiment.get("risk_threshold", 18.0)
            )
            candidate_states["risk_switch"] = (
                projection_state if risk_uses_projection else freshness_state
            )
            candidate_losses["risk_switch"] = (
                projection_loss if risk_uses_projection else freshness_loss
            )
            candidate_updates["risk_switch"] = (
                geometry.projected_updates
                if risk_uses_projection
                else {
                    name: freshness * update.dense()
                    for name, update in innovations.items()
                }
            )
            accepted_state = candidate_states[accept_method]
            accepted_loss, accepted_accuracy = _metrics_for_state(
                model, accepted_state, eval_batch
            )
            update_id = f"u{event_index:04d}"
            artifact_id = f"{artifact_relative}#{update_id}"
            rows.append(
                {
                    "run_id": run_id,
                    "regime": regime["name"],
                    "seed": seed,
                    "update_id": update_id,
                    "client_id": event.client_id,
                    "base_version": event.base_version,
                    "current_version": event.arrival_version,
                    "tau": event.staleness,
                    "rank": event.rank,
                    "num_samples": len(local_indices),
                    "virtual_latency": event.finish_time - event.dispatch_time,
                    "update_fro_norm": geometry.fro_norm,
                    "effective_rank": geometry.effective_rank,
                    "rho_left": geometry.rho_left,
                    "rho_right": geometry.rho_right,
                    "rho_two_sided": geometry.rho_two_sided,
                    "raw_update_utility": current_loss - raw_loss,
                    "freshness_update_utility": current_loss - freshness_loss,
                    "vast_update_utility": current_loss - vast_loss,
                    "projection_update_utility": current_loss - projection_loss,
                    "projection_left_update_utility": current_loss
                    - projection_left_loss,
                    "projection_right_update_utility": current_loss
                    - projection_right_loss,
                    "persistent_projection_update_utility": current_loss
                    - persistent_projection_loss,
                    "projection_scaled_update_utility": current_loss
                    - projection_scaled_loss,
                    "residual_budget_update_utility": current_loss
                    - residual_budget_loss,
                    "saber_update_utility": current_loss - saber_loss,
                    "union_update_utility": current_loss - union_loss,
                    "lattice_update_utility": current_loss - lattice_loss,
                    "lattice_sq_update_utility": current_loss - lattice_sq_loss,
                    "current_loss": current_loss,
                    "current_accuracy": current_accuracy,
                    "raw_candidate_loss": raw_loss,
                    "freshness_candidate_loss": freshness_loss,
                    "vast_candidate_loss": vast_loss,
                    "projection_candidate_loss": projection_loss,
                    "projection_left_candidate_loss": projection_left_loss,
                    "projection_right_candidate_loss": projection_right_loss,
                    "persistent_projection_candidate_loss": persistent_projection_loss,
                    "projection_scaled_candidate_loss": projection_scaled_loss,
                    "residual_budget_candidate_loss": residual_budget_loss,
                    "saber_candidate_loss": saber_loss,
                    "union_candidate_loss": union_loss,
                    "lattice_candidate_loss": lattice_loss,
                    "lattice_sq_candidate_loss": lattice_sq_loss,
                    "accepted_method": accept_method,
                    "accepted_loss": accepted_loss,
                    "accepted_accuracy": accepted_accuracy,
                    "risk_score": risk_score,
                    "risk_threshold": float(experiment.get("risk_threshold", 18.0)),
                    "risk_uses_projection": risk_uses_projection,
                    "local_probe_loss_before": training.probe_loss_before,
                    "local_probe_loss_after": training.probe_loss_after,
                    "local_probe_loss_change": training.probe_loss_after
                    - training.probe_loss_before,
                    "local_train_loss_first": training.train_loss_first,
                    "local_train_loss_last": training.train_loss_last,
                    "local_train_loss_mean": training.train_loss_mean,
                    "local_grad_norm_first": training.grad_norm_first,
                    "local_grad_norm_last": training.grad_norm_last,
                    "a_relative_change": training.a_relative_change,
                    "a_subspace_retained": training.a_subspace_retained,
                    "b_relative_change": training.b_relative_change,
                    "freshness": freshness,
                    "residual_freshness": freshness
                    ** float(experiment.get("residual_freshness_power", 1.0)),
                    "dataset_name": f"{dataset_config['hub_path']}/{dataset_config['subset']}",
                    "dataset_fingerprint_sha256": dataset_fingerprint,
                    "partition_seed": seed,
                    "partition_artifact": f"{artifact_relative}#partitions",
                    "client_indices_artifact": f"{artifact_relative}#partitions.{event.client_id}",
                    "base_snapshot_id": f"{artifact_relative}#snapshot.v{event.base_version}",
                    "current_snapshot_id": f"{artifact_relative}#snapshot.v{event.arrival_version}",
                    "update_artifact_id": artifact_id,
                    "validation_split": dataset_config["validation_split"],
                    "validation_indices_sha256": validation_hash,
                    "metric": "cross_entropy_loss",
                    "model_name": model_config["name"],
                    "model_commit": getattr(model.config, "_commit_hash", None) or "unknown",
                    "reference_rank": experiment["reference_rank"],
                    "history_size": experiment["history_size"],
                    "reference_decay": experiment["reference_decay"],
                    "reference_singular_power": float(
                        experiment.get("reference_singular_power", 0.0)
                    ),
                    "freshness_lambda": experiment["freshness_lambda"],
                    "residual_freshness_power": float(
                        experiment.get("residual_freshness_power", 1.0)
                    ),
                    "residual_budget": float(
                        experiment.get("residual_budget", 0.5)
                    ),
                    "projection_scale_cap": float(
                        experiment.get("projection_scale_cap", 2.0)
                    ),
                    "persistent_max_rank": int(
                        experiment.get("persistent_max_rank", 4)
                    ),
                    "persistent_short_history": int(
                        experiment.get("persistent_short_history", 4)
                    ),
                    "persistent_overlap_threshold": float(
                        experiment.get("persistent_overlap_threshold", 0.7)
                    ),
                    "persistent_left_rank_mean": sum(persistent.left_ranks.values())
                    / len(persistent.left_ranks),
                    "persistent_right_rank_mean": sum(persistent.right_ranks.values())
                    / len(persistent.right_ranks),
                    "server_update_weight": experiment["server_update_weight"],
                    "local_learning_rate": experiment["local_learning_rate"],
                    "local_steps": experiment["local_steps"],
                }
            )
            artifacts[update_id] = {
                "current_version": event.arrival_version,
                "raw_loss": raw_loss,
                "current_loss": current_loss,
                "innovation": {
                    name: {"left": update.left.cpu(), "right": update.right.cpu()}
                    for name, update in innovations.items()
                },
            }
            next_state = accepted_state
            current_loss = accepted_loss
            current_accuracy = accepted_accuracy
            compact_updates = {
                name: _compact_dense(update, rtol=experiment["rank_rtol"])
                for name, update in candidate_updates[accept_method].items()
            }
        else:
            next_state = add_dense_innovation(
                current_state,
                innovations,
                weight=experiment["server_update_weight"],
            )

        snapshots[event.new_server_version] = {
            name: value.detach().cpu().clone() for name, value in next_state.items()
        }
        for name, update in compact_updates.items():
            history[name].append(update)

        if (event_index + 1) % 10 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"{run_id}: {event_index + 1}/{total_returns} returns ({elapsed:.1f}s)")

    bundle = {
        "format_version": 1,
        "run_id": run_id,
        "config": dict(config),
        "regime": dict(regime),
        "seed": seed,
        "target_names": target_names,
        "partitions": {f"c{index:02d}": values for index, values in enumerate(partitions)},
        "validation_indices": list(eval_indices),
        "eval_batch": {name: value.cpu() for name, value in eval_batch.items()},
        "snapshots": snapshots,
        "updates": artifacts,
        "trace": [record.__dict__ for record in trace.records],
    }
    torch.save(bundle, replay_dir / f"{run_id}.pt")
    return pd.DataFrame(rows)


def _build_partitions(
    labels: Sequence[int],
    regime: Mapping[str, Any],
    num_clients: int,
    seed: int,
) -> list[list[int]]:
    if regime["partition"] == "iid":
        return iid_partition_indices(len(labels), num_clients, seed=seed)
    if regime["partition"] == "label_shard":
        partitions = label_shard_partition_indices(
            labels,
            num_clients,
            shards_per_client=int(regime.get("shards_per_client", 2)),
            seed=seed,
        )
        for offset, values in enumerate(partitions):
            random.Random(seed + offset).shuffle(values)
        return partitions
    raise ValueError(f"unsupported partition: {regime['partition']}")


def _tokenize(dataset: Dataset, tokenizer: Any, config: Mapping[str, Any]) -> Dataset:
    text_column = config["dataset"]["text_column"]
    text_pair_column = config["dataset"].get("text_pair_column")
    label_column = config["dataset"]["label_column"]
    max_length = int(config["model"]["max_length"])

    def tokenize_batch(batch: Mapping[str, Sequence[Any]]) -> Mapping[str, Any]:
        texts = [batch[text_column]]
        if text_pair_column:
            texts.append(batch[text_pair_column])
        result = tokenizer(
            *texts, padding="max_length", truncation=True, max_length=max_length
        )
        result["labels"] = batch[label_column]
        return result

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {config['dataset']['subset']}",
    )


def _make_batch(dataset: Dataset, indices: Sequence[int], device: torch.device) -> dict[str, torch.Tensor]:
    values = dataset.select(list(indices))[:]
    return {
        name: torch.tensor(value, dtype=torch.long, device=device)
        for name, value in values.items()
    }


@dataclass(frozen=True)
class LocalTrainingDiagnostics:
    probe_loss_before: float
    probe_loss_after: float
    train_loss_first: float
    train_loss_last: float
    train_loss_mean: float
    grad_norm_first: float
    grad_norm_last: float
    a_relative_change: float
    a_subspace_retained: float
    b_relative_change: float


def _train_local_update(
    model: torch.nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
) -> LocalTrainingDiagnostics:
    parameters = list(local_adapter_parameters(model))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    probe_indices = list(indices[:batch_size])
    probe_batch = _make_batch(dataset, probe_indices, device)
    probe_loss_before = _evaluate_loss(model, probe_batch)
    losses: list[float] = []
    grad_norms: list[float] = []
    for step in range(steps):
        model.train()
        start = step * batch_size
        batch = _make_batch(dataset, indices[start : start + batch_size], device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        grad_norms.append(float(grad_norm.detach().item()))
    probe_loss_after = _evaluate_loss(model, probe_batch)
    a_relative_change, a_subspace_retained, b_relative_change = _factor_change_metrics(model)
    return LocalTrainingDiagnostics(
        probe_loss_before=probe_loss_before,
        probe_loss_after=probe_loss_after,
        train_loss_first=losses[0],
        train_loss_last=losses[-1],
        train_loss_mean=sum(losses) / len(losses),
        grad_norm_first=grad_norms[0],
        grad_norm_last=grad_norms[-1],
        a_relative_change=a_relative_change,
        a_subspace_retained=a_subspace_retained,
        b_relative_change=b_relative_change,
    )


def _factor_change_metrics(model: torch.nn.Module) -> tuple[float, float, float]:
    a_delta_sq = 0.0
    a_initial_sq = 0.0
    a_final_sq = 0.0
    a_retained_sq = 0.0
    b_delta_sq = 0.0
    b_initial_sq = 0.0
    for module in named_lora_modules(model).values():
        a0 = module.initial_lora_a.detach()
        a1 = module.lora_a.detach()
        b0 = module.initial_lora_b.detach()
        b1 = module.lora_b.detach()
        a_delta_sq += float(torch.sum((a1 - a0).square()).item())
        a_initial_sq += float(torch.sum(a0.square()).item())
        a_final_sq += float(torch.sum(a1.square()).item())
        b_delta_sq += float(torch.sum((b1 - b0).square()).item())
        b_initial_sq += float(torch.sum(b0.square()).item())
        if torch.count_nonzero(a0):
            q, _ = torch.linalg.qr(a0.T, mode="reduced")
            a_retained_sq += float(torch.sum((a1 @ q).square()).item())
    eps = 1e-12
    return (
        (a_delta_sq / max(a_initial_sq, eps)) ** 0.5,
        a_retained_sq / max(a_final_sq, eps),
        (b_delta_sq / max(b_initial_sq, eps)) ** 0.5,
    )


def _evaluate_loss(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> float:
    model.eval()
    with torch.inference_mode():
        return float(model(**batch).loss.item())


def _evaluate_metrics(
    model: torch.nn.Module, batch: Mapping[str, torch.Tensor]
) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        output = model(**batch)
        accuracy = (output.logits.argmax(dim=-1) == batch["labels"]).float().mean()
        return float(output.loss.item()), float(accuracy.item())


def _loss_for_state(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> float:
    set_server_adapter_state(model, state)
    return _evaluate_loss(model, batch)


def _metrics_for_state(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> tuple[float, float]:
    set_server_adapter_state(model, state)
    return _evaluate_metrics(model, batch)


def _compact_dense(matrix: torch.Tensor, *, rtol: float) -> CompactSVD:
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    keep = s >= (rtol * s[0]) if s.numel() and s[0] > 0 else torch.zeros_like(s, dtype=torch.bool)
    return CompactSVD(u[:, keep], s[keep], vh.T[:, keep])


def _sample_indices(values: Sequence[int], count: int, seed: int) -> list[int]:
    if count <= 0 or not values:
        raise ValueError("cannot sample an empty or non-positive batch")
    rng = random.Random(seed)
    if count <= len(values):
        return rng.sample(list(values), count)
    return [rng.choice(values) for _ in range(count)]


def _hash_indices(indices: Sequence[int]) -> str:
    payload = ",".join(str(value) for value in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _dataset_fingerprint(config: Mapping[str, Any], dataset: Mapping[str, Dataset]) -> str:
    audit_path = ROOT / config["dataset"]["fingerprint_audit"]
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        train_name = config["dataset"]["train_split"]
        if isinstance(audit.get("splits"), Mapping):
            return str(audit["splits"][train_name]["fingerprint_sha256"])
        if train_name in audit:
            return str(audit[train_name]["fingerprint_sha256"])
    return str(dataset[config["dataset"]["train_split"]]._fingerprint)


def _resolve_device(choice: str) -> torch.device:
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == "__main__":
    main()
