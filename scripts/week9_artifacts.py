"""Validate finished generative runs before resuming or drawing conclusions."""
from __future__ import annotations

import json
import math
import hashlib

import numpy as np
import pandas as pd

import run_week8_classification_matrix as matrix_runner


EVAL_COLUMNS = (
    "source_id", "reference", "nll_sum", "response_tokens", "response_nll",
    "rouge_l", "exact_match", "generated_tokens", "hit_generation_limit",
)
EVENT_COLUMNS = (
    "event", "measured", "current_loss", "accepted_loss", "staleness",
    "harmful_update", "late_harmful_update", "update_accepted",
    "client_id", "base_version", "arrival_version",
)


def _read_csv(path, required, **kwargs):
    frame = pd.read_csv(path, **kwargs)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {', '.join(missing)}")
    return frame


def _close(actual, expected, label):
    if not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"{label} does not match artifacts")


def _booleans(series):
    values = series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if values.isna().any():
        raise ValueError(f"invalid boolean column {series.name}")
    return values.astype(bool)


def validate_run(path, *, config, method, seed, matrix):
    """Return verified payload and data identity; dirty provenance is checked by callers."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method in {"fedavg_lora", "ffa_lora"}:
        from riftlora.baselines.factor_averaging import FACTOR_IMPLEMENTATION
        if payload.get("factor_implementation") != FACTOR_IMPLEMENTATION:
            raise ValueError("obsolete factor implementation; rerun FedAvg/FFA-LoRA")
    expected = matrix_runner._runner_config_fingerprint(config)
    if (payload.get("config_fingerprint") != expected
            or matrix_runner._runner_config_fingerprint(payload["config"]) != expected
            or payload.get("method") != method or payload.get("seed") != seed
            or payload.get("schema_version", 0) < matrix.get("required_schema_version", 5)
            or payload.get("provenance") != config["provenance"]):
        raise ValueError("config/schema/identity mismatch")
    metrics, exp, ds = payload["metrics"], config["experiment"], config["dataset"]
    selected = payload["data_diagnostics"]["generation"]["selected_source_ids"]
    counts = {"clients": ds["max_train_examples"], "gradient": exp["calibration_gradient_examples"],
              "gate": exp["calibration_gate_examples"], "monitor": exp["monitor_examples"],
              "evaluation": ds["eval_examples"]}
    if {key: len(value) for key, value in selected.items()} != counts:
        raise ValueError("sample manifest counts do not match config")
    if len(set().union(*map(set, selected.values()))) != sum(map(len, selected.values())):
        raise ValueError("client/calibration/monitor/eval rows overlap")
    groups = payload["data_diagnostics"]["generation"].get("selected_group_ids")
    if ds.get("reserve_context_groups", False):
        if groups is None or {key: len(value) for key, value in groups.items()} != counts:
            raise ValueError("missing or invalid context-group manifest")
        sets = [set(value) for value in groups.values()]
        if len(set().union(*sets)) != sum(map(len, sets)):
            raise ValueError("contexts overlap between client/calibration/monitor/eval roles")
    reference_identity = None
    for stage in ("baseline", "final"):
        details = _read_csv(path.parent / f"{stage}_eval_details.csv", EVAL_COLUMNS, keep_default_na=False)
        if len(details) != ds["eval_examples"] or details.source_id.tolist() != selected["evaluation"]:
            raise ValueError("evaluation count/IDs differ from sample manifest")
        numeric = details[["nll_sum", "response_tokens", "response_nll", "rouge_l", "exact_match", "generated_tokens"]].astype(float)
        if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any().any():
            raise ValueError("nonfinite or negative evaluation values")
        if (numeric.response_tokens <= 0).any() or not np.equal(numeric.response_tokens, np.floor(numeric.response_tokens)).all():
            raise ValueError("invalid response token counts")
        if (numeric.response_tokens > ds["max_response_tokens"]).any():
            raise ValueError("response exceeds locked token budget")
        if not np.allclose(numeric.response_nll, numeric.nll_sum / numeric.response_tokens, rtol=1e-6, atol=1e-8):
            raise ValueError("per-example NLL does not match token sums")
        computed = numeric.nll_sum.sum() / numeric.response_tokens.sum()
        _close(metrics[f"{stage}_token_nll"], computed, "reported token NLL")
        if metrics[f"{stage}_perplexity"] is None or metrics[f"{stage}_perplexity"] <= 0:
            raise ValueError("invalid perplexity")
        _close(math.log(metrics[f"{stage}_perplexity"]), computed, "log perplexity")
        if (numeric.rouge_l > 1).any() or not numeric.exact_match.isin([0, 1]).all():
            raise ValueError("invalid sequence scores")
        if (numeric.generated_tokens > ds.get("max_new_tokens", ds["max_response_tokens"])).any():
            raise ValueError("generation exceeds budget")
        for key in ("rouge_l", "exact_match"):
            _close(metrics[f"{stage}_{key}"], numeric[key].mean(), key)
        _close(metrics[f"{stage}_generation_limit_rate"], _booleans(details.hit_generation_limit).mean(), "generation limit rate")
        identity = details[["source_id", "reference", "response_tokens"]].to_dict("list")
        if reference_identity is not None and reference_identity != identity:
            raise ValueError("baseline and final references differ")
        reference_identity = identity
        for key, column in (("rouge_l_precision", "rouge_l_precision"), ("rouge_l_recall", "rouge_l_recall"),
                            ("mean_generated_tokens", "generated_tokens")):
            if f"{stage}_{key}" in metrics:
                if column not in details:
                    raise ValueError(f"missing reported generation column: {column}")
                values = details[column].astype(float)
                if not np.isfinite(values).all() or (values < 0).any() or (key != "mean_generated_tokens" and (values > 1).any()):
                    raise ValueError(f"invalid generation values: {key}")
                _close(metrics[f"{stage}_{key}"], values.mean(), key)

    events = _read_csv(path.parent / "events.csv", EVENT_COLUMNS)
    total = exp["warmup_returns"] + exp["collected_returns"]
    if len(events) != total or events.event.tolist() != list(range(total)):
        raise ValueError("incomplete or reordered return trace")
    measured = _booleans(events.measured)
    if measured.tolist() != [False] * exp["warmup_returns"] + [True] * exp["collected_returns"]:
        raise ValueError("incorrect warmup/measured event boundary")
    finite = events.loc[measured, ["current_loss", "accepted_loss", "staleness"]].to_numpy(dtype=float)
    if not np.isfinite(finite).all():
        raise ValueError("missing/nonfinite monitor observations")
    trace = events.loc[measured]
    harm = trace.accepted_loss > trace.current_loss + exp["harm_epsilon"]
    late = trace.staleness >= exp["late_tau"]
    if not (harm.to_numpy() == _booleans(trace.harmful_update).to_numpy()).all():
        raise ValueError("harmful flags do not match monitor deltas")
    if not ((harm & late).to_numpy() == _booleans(trace.late_harmful_update).to_numpy()).all():
        raise ValueError("late harmful flags do not match monitor deltas")
    _close(metrics["harmful_update_rate"], harm.mean(), "harmful rate")
    _close(metrics["late_harmful_update_rate"], harm[late].mean() if late.any() else 0, "late harmful rate")
    _close(metrics["late_event_count"], late.sum(), "late event count")
    _close(metrics["measured_event_count"], len(trace), "measured event count")
    _close(metrics["acceptance_rate"], _booleans(trace.update_accepted).mean(), "acceptance rate")
    for key in ("runtime_seconds", "peak_cuda_memory_gib"):
        if not math.isfinite(metrics[key]) or metrics[key] < 0:
            raise ValueError(f"invalid {key}")
    identity = {"selected": selected, "groups": groups, "references": reference_identity,
                "schedule": events[["client_id", "base_version", "arrival_version", "staleness"]].to_dict("list")}
    milestones = exp.get("generation_eval_returns", [])
    if milestones:
        curve = payload.get("development_learning_curve", [])
        if [r["measured_returns"] for r in curve] != milestones:
            raise ValueError("missing or reordered learning curve")
        for entry in curve:
            record, _ = validate_milestone(path.parent / "milestones" / f"returns_{entry['measured_returns']:04d}",
                                           config=config, method=method, seed=seed)
            if record != entry:
                raise ValueError("embedded learning curve differs from saved milestone")
    return payload, identity


def validate_milestone(directory, *, config, method, seed):
    record = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    exp, ds = config["experiment"], config["dataset"]
    if (record["config_fingerprint"] != matrix_runner._runner_config_fingerprint(config)
            or record["method"] != method or record["seed"] != seed
            or record["measured_returns"] not in exp["generation_eval_returns"]
            or record["phase"] not in {"development", "smoke"}
            or record["total_returns"] != record["measured_returns"] + exp["warmup_returns"]):
        raise ValueError("milestone identity/budget mismatch")
    for name in ("adapter.pt", "eval_details.csv", "events.csv"):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != record["sha256"][name]:
            raise ValueError(f"milestone artifact hash mismatch: {name}")
    details = _read_csv(directory / "eval_details.csv", EVAL_COLUMNS, keep_default_na=False)
    if len(details) != ds["eval_examples"] or details.source_id.duplicated().any():
        raise ValueError("milestone evaluation identity/count mismatch")
    metrics = record["metrics"]
    _close(metrics["nll"], details.nll_sum.sum() / details.response_tokens.sum(), "milestone NLL")
    for key in ("rouge_l", "rouge_l_precision", "rouge_l_recall", "exact_match"):
        values = details[key].astype(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"invalid milestone {key}")
        _close(metrics[key], values.mean(), key)
    _close(metrics["mean_generated_tokens"], details.generated_tokens.mean(), "milestone generation length")
    _close(metrics["generation_limit_rate"], _booleans(details.hit_generation_limit).mean(), "milestone limit rate")
    events = _read_csv(directory / "events.csv", EVENT_COLUMNS)
    if events.event.tolist() != list(range(record["total_returns"])):
        raise ValueError("milestone event trace mismatch")
    measured = _booleans(events.measured)
    if measured.tolist() != [False] * exp["warmup_returns"] + [True] * record["measured_returns"]:
        raise ValueError("milestone warmup/measured boundary mismatch")
    trace = events.loc[measured]
    harm = trace.accepted_loss > trace.current_loss + exp["harm_epsilon"]
    late = trace.staleness >= exp["late_tau"]
    _close(record["safety"]["harmful_update_rate"], harm.mean(), "milestone harmful")
    _close(record["safety"]["late_event_count"], late.sum(), "milestone late events")
    _close(record["safety"]["acceptance_rate"], _booleans(trace.update_accepted).mean(), "milestone acceptance")
    if late.any():
        _close(record["safety"]["late_harmful_update_rate"], harm[late].mean(), "milestone late harmful")
    elif record["safety"]["late_harmful_update_rate"] is not None:
        raise ValueError("no late events must be reported as unavailable")
    presentations = record["total_returns"] * exp["local_steps"] * exp["local_batch_size"] * exp["gradient_accumulation_steps"]
    if record["work"]["client_sample_presentations"] != presentations:
        raise ValueError("milestone training budget mismatch")
    if not 0 < record["work"]["client_unique_examples"] <= min(presentations, ds["max_train_examples"]):
        raise ValueError("milestone unique sample count mismatch")
    return record, details[["source_id", "reference", "response_tokens"]].to_dict("list")
