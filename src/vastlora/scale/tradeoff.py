from __future__ import annotations

import random
from typing import Any, Iterable, Sequence


DEFAULT_TRADEOFF_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"name": "freshness", "method": "freshness", "cli_args": []},
    {"name": "hybrid_beta005", "method": "mtip_hybrid", "cli_args": ["--residual-beta", "0.05"]},
    {"name": "hybrid_beta010", "method": "mtip_hybrid", "cli_args": ["--residual-beta", "0.1"]},
    {"name": "hybrid_beta020", "method": "mtip_hybrid", "cli_args": ["--residual-beta", "0.2"]},
    {"name": "hybrid_beta040", "method": "mtip_hybrid", "cli_args": ["--residual-beta", "0.4"]},
    {"name": "hybrid_beta070", "method": "mtip_hybrid", "cli_args": ["--residual-beta", "0.7"]},
    {
        "name": "routed_c2_t1",
        "method": "mtip_routed",
        "cli_args": ["--residual-staleness-center", "2", "--residual-staleness-temperature", "1"],
    },
    {
        "name": "routed_c4_t1",
        "method": "mtip_routed",
        "cli_args": ["--residual-staleness-center", "4", "--residual-staleness-temperature", "1"],
    },
    {
        "name": "routed_c6_t2",
        "method": "mtip_routed",
        "cli_args": ["--residual-staleness-center", "6", "--residual-staleness-temperature", "2"],
    },
)


def reserved_train_eval_indices(
    length: int,
    *,
    eval_offset: int,
    eval_examples: int,
    shuffle_seed: int,
) -> tuple[list[int], list[int]]:
    if length <= 0 or eval_offset < 0 or eval_examples <= 0:
        raise ValueError("invalid development holdout bounds")
    ordered_indices = list(range(length))
    random.Random(shuffle_seed).shuffle(ordered_indices)
    eval_end = min(eval_offset + eval_examples, length)
    eval_indices = ordered_indices[eval_offset:eval_end]
    if not eval_indices:
        raise ValueError("development holdout must contain at least one example")
    eval_index_set = set(eval_indices)
    train_indices = [index for index in range(length) if index not in eval_index_set]
    return train_indices, eval_indices


def select_tradeoff(
    rows: list[dict[str, Any]],
    candidates: Sequence[dict[str, Any]] = DEFAULT_TRADEOFF_CANDIDATES,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate"]), []).append(row)
    if "freshness" not in grouped:
        raise ValueError("freshness development baseline is missing")

    baseline_by_seed = {int(row["seed"]): row for row in grouped["freshness"]}
    summaries: list[dict[str, Any]] = []
    for candidate in candidates[1:]:
        candidate_rows = grouped.get(str(candidate["name"]), [])
        paired = [
            (row, baseline_by_seed[int(row["seed"])])
            for row in candidate_rows
            if int(row["seed"]) in baseline_by_seed
        ]
        if not paired:
            continue
        accuracy_gain = _mean(
            100.0 * (row["final_accuracy"] - base["final_accuracy"])
            for row, base in paired
        )
        balanced_accuracy_gain = _mean(
            100.0
            * (row["final_balanced_accuracy"] - base["final_balanced_accuracy"])
            for row, base in paired
        )
        sequence_nll_gain = _mean(
            base["final_nll"] - row["final_nll"] for row, base in paired
        )
        binary_nll_gain = _mean(
            base["final_binary_nll"] - row["final_binary_nll"]
            for row, base in paired
        )
        balanced_wins = sum(
            row["final_balanced_accuracy"] > base["final_balanced_accuracy"]
            for row, base in paired
        )
        sequence_nll_relative_change = _mean(
            row["final_nll"] / base["final_nll"] - 1.0 for row, base in paired
        )
        binary_nll_relative_change = _mean(
            row["final_binary_nll"] / base["final_binary_nll"] - 1.0
            for row, base in paired
        )
        brier_relative_change = _mean(
            row["final_brier"] / base["final_brier"] - 1.0 for row, base in paired
        )
        dev_gate_pass = (
            balanced_accuracy_gain >= 0.5
            and sequence_nll_relative_change <= 0.10
            and binary_nll_relative_change <= 0.05
            and brier_relative_change <= 0.05
            and balanced_wins == len(paired)
        )
        normalized_gate_deficit = (
            max(0.0, 0.5 - balanced_accuracy_gain) / 0.5
            + max(0.0, sequence_nll_relative_change - 0.10) / 0.10
            + max(0.0, binary_nll_relative_change - 0.05) / 0.05
            + max(0.0, brier_relative_change - 0.05) / 0.05
            + (len(paired) - balanced_wins) / len(paired)
        )
        summaries.append(
            {
                **candidate,
                "dev_seed_count": len(paired),
                "accuracy_gain_pp": accuracy_gain,
                "balanced_accuracy_gain_pp": balanced_accuracy_gain,
                "sequence_nll_gain": sequence_nll_gain,
                "binary_nll_gain": binary_nll_gain,
                "sequence_nll_relative_change": sequence_nll_relative_change,
                "binary_nll_relative_change": binary_nll_relative_change,
                "brier_relative_change": brier_relative_change,
                "balanced_accuracy_wins": balanced_wins,
                "dev_gate_pass": dev_gate_pass,
                "normalized_gate_deficit": normalized_gate_deficit,
            }
        )
    if not summaries:
        raise ValueError("no complete MTIP trade-off candidate was found")

    feasible = [row for row in summaries if row["dev_gate_pass"]]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row["balanced_accuracy_gain_pp"],
                row["binary_nll_gain"],
                row["sequence_nll_gain"],
            ),
        )
    else:
        selected = min(
            summaries,
            key=lambda row: (
                row["normalized_gate_deficit"],
                -row["balanced_accuracy_gain_pp"],
            ),
        )
    return {
        "status": "DEV_GATE_PASS" if feasible else "DEV_GATE_MISS",
        "selection_rule": (
            "maximize balanced accuracy among candidates clearing paired accuracy, "
            "sequence-NLL, binary-NLL, and Brier non-inferiority gates; otherwise "
            "minimize normalized gate deficit"
        ),
        "selected": selected,
        "candidates": summaries,
    }


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return float(sum(collected) / len(collected))
