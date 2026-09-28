"""Opt-in data and schedule provenance for the causal-control experiments."""

from dataclasses import asdict
import importlib.metadata
import hashlib
import itertools
import json
import unicodedata

from riftlora.scale.tradeoff import reserved_train_eval_indices


SOURCE_ID = "_rift_source_id"


def content_key(row, label):
    values = {
        k: " ".join(unicodedata.normalize("NFKC", v).split())
        if isinstance(v, str)
        else v
        for k, v in row.items()
        if k not in {SOURCE_ID, "idx", label}
    }
    return fingerprint(values)


def _canonical_content_rows(data, label):
    first, conflicts = {}, set()
    for index, row in enumerate(data):
        key = content_key(row, label)
        if key in first and first[key][1] != row[label]:
            conflicts.add(key)
        else:
            first.setdefault(key, (index, row[label]))
    indices = [index for key, (index, _) in first.items() if key not in conflicts]
    keys = [key for key in first if key not in conflicts]
    return (
        data.select(indices),
        keys,
        {
            "input_rows": len(data),
            "canonical_rows": len(indices),
            "removed_rows": len(data) - len(indices),
            "conflicting_content_groups": len(conflicts),
        },
    )


def prepare_content_disjoint_splits(raw, dataset):
    """Group content before window selection; reserve both phases without model scores."""
    if dataset.get("study_content_policy") != "nfkc_whitespace_groups_v1":
        raise ValueError("unknown study content policy")
    train_name = dataset["train_split"]
    eval_name = dataset.get("eval_split", dataset["validation_split"])
    train, train_keys, train_stats = _canonical_content_rows(
        raw[train_name], dataset["label_column"]
    )
    if train_name == eval_name:
        eval_pool, eval_keys, eval_stats = train, train_keys, train_stats
    else:
        eval_pool, eval_keys, eval_stats = _canonical_content_rows(
            raw[eval_name], dataset["label_column"]
        )
    windows = dataset["study_eval_windows"]
    start, count = dataset["eval_offset"], dataset["eval_examples"]
    if not any(a <= start and start + count <= a + n for a, n in windows):
        raise ValueError("active evaluation window is not reserved")
    if any(a < 0 or n < 1 or a + n > len(eval_pool) for a, n in windows):
        raise ValueError("reserved window exceeds canonical evaluation pool")
    if train_name == eval_name:
        _, order = reserved_train_eval_indices(
            len(eval_pool),
            eval_offset=0,
            eval_examples=len(eval_pool),
            shuffle_seed=dataset["eval_shuffle_seed"],
        )
    else:
        ordered_ids = eval_pool.shuffle(seed=dataset["eval_shuffle_seed"])[SOURCE_ID]
        lookup = {value: i for i, value in enumerate(eval_pool[SOURCE_ID])}
        order = [lookup[value] for value in ordered_ids]
    reserved_indices = [order[i] for a, n in windows for i in range(a, a + n)]
    if len(reserved_indices) != len(set(reserved_indices)):
        raise ValueError("development/confirmation evaluation windows overlap")
    reserved_keys = {eval_keys[i] for i in reserved_indices}
    allowed = [i for i, key in enumerate(train_keys) if key not in reserved_keys]
    evaluation = eval_pool.select(order[start : start + count])
    return (
        train.select(allowed),
        evaluation,
        {
            "policy": dataset["study_content_policy"],
            "train": train_stats,
            "evaluation_source": eval_stats,
            "reserved_content_groups": len(reserved_keys),
            "eligible_train_rows": len(allowed),
            "reserved_train_groups_removed": len(train) - len(allowed),
            "windows": windows,
        },
    )


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def attach_source_ids(raw):
    result = {}
    for split, data in raw.items():
        if SOURCE_ID in data.column_names:
            raise ValueError("reserved source ID column already exists")
        result[split] = data.add_column(
            SOURCE_ID, [f"{split}:{i}" for i in range(len(data))]
        )
    return result


def exclude_reserved_windows(raw, train, dataset):
    """Exclude BOTH study eval windows from SST-2 training in BOTH phases.

    Rebuild from the original split in source order, avoiding phase-dependent
    training pools. This does not certify non-exposure in historical runs.
    """
    windows = dataset.get("study_reserved_train_windows", [])
    if not windows:
        return train
    split = dataset["train_split"]
    if dataset.get("eval_split", dataset["validation_split"]) != split:
        raise ValueError("reserved train windows require evaluation from train")
    excluded = set()
    for offset, count in windows:
        if offset < 0 or count < 1 or offset + count > len(raw[split]):
            raise ValueError("reserved evaluation window exceeds source split")
        _, indices = reserved_train_eval_indices(
            len(raw[split]),
            eval_offset=offset,
            eval_examples=count,
            shuffle_seed=dataset["eval_shuffle_seed"],
        )
        excluded.update(indices)
    start, end = (
        dataset["eval_offset"],
        dataset["eval_offset"] + dataset["eval_examples"],
    )
    if not any(offset <= start and end <= offset + count for offset, count in windows):
        raise ValueError("active eval window must be covered by study reservation")
    return raw[split].select([i for i in range(len(raw[split])) if i not in excluded])


def experiment_audit(config, method, trace, partitions, **datasets):
    selections = {}
    content_sets = {}
    label = config["dataset"]["label_column"]
    for name, data in datasets.items():
        rows = list(data) if data is not None else []
        ids = [r[SOURCE_ID] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate source IDs inside {name}")
        content = [
            content_key(r, label)
            if config["dataset"].get("study_content_policy")
            else fingerprint(
                {k: v for k, v in r.items() if k not in {SOURCE_ID, "idx", label}}
            )
            for r in rows
        ]
        content_sets[name] = set(content)
        selections[name] = {
            "source_ids": ids,
            "content_sha256": content,
            "rows_sha256": fingerprint(rows),
            "count": len(rows),
        }
    for left, right in itertools.combinations(selections, 2):
        if set(selections[left]["source_ids"]) & set(selections[right]["source_ids"]):
            raise ValueError(f"source overlap: {left}/{right}")
        if content_sets[left] & content_sets[right]:
            raise ValueError(
                f"exact content overlap: {left}/{right}; revise splits on development"
            )
    records = [asdict(r) for r in trace.records]
    return {
        "protocol": dict(config["causal_study"]),
        "selected_data": selections,
        "data_sha256": fingerprint(selections),
        "schedule_sha256": fingerprint(records),
        "schedule": records,
        "partitions_sha256": fingerprint(partitions),
        "partitions": partitions,
        "effective_gate": (
            {
                "selection": "min_risk",
                "freshness_fallback": False,
                "positive_filter_comparator": method != "server_only",
                "noop_comparator": method == "server_only",
                "scales": config["experiment"]["rift_step_scales"],
            }
            if method in {"rift_core", "rift_diag", "server_lora", "server_only"}
            else None
        ),
        "client_training": method != "server_only",
        "environment": {
            name: _version(name)
            for name in ("torch", "transformers", "peft", "datasets", "bitsandbytes")
        },
        "server_only_warmup": "no-op; same measured fitting opportunities, no client contribution",
        "scope": "Source/exact-content disjointness, not a semantic leakage or historical exposure certificate.",
    }


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
