from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from vastlora.data.partitioning import iid_partition_indices


Record = Mapping[str, Any]


@dataclass(frozen=True)
class TextDatasetSpec:
    name: str
    hub_path: str
    subset: str
    text_column: str
    label_column: str
    index_column: str
    train_split: str
    validation_split: str
    unlabeled_splits: tuple[str, ...]
    metric: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TextDatasetSpec":
        return cls(
            name=str(value["name"]),
            hub_path=str(value["hub_path"]),
            subset=str(value["subset"]),
            text_column=str(value["text_column"]),
            label_column=str(value["label_column"]),
            index_column=str(value["index_column"]),
            train_split=str(value["train_split"]),
            validation_split=str(value["validation_split"]),
            unlabeled_splits=tuple(str(item) for item in value.get("unlabeled_splits", ())),
            metric=str(value["metric"]),
        )


def audit_text_split(
    records: Iterable[Record],
    *,
    split: str,
    text_column: str,
    label_column: str,
    index_column: str,
    allow_unlabeled: bool = False,
) -> dict[str, Any]:
    rows = 0
    null_texts = 0
    empty_texts = 0
    unlabeled_rows = 0
    missing_indices = 0
    duplicate_indices = 0
    label_counts: Counter[str] = Counter()
    index_counts: Counter[int] = Counter()
    normalized_counts: Counter[str] = Counter()
    normalized_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    normalized_labels: dict[str, set[str]] = defaultdict(set)
    min_chars: int | None = None
    max_chars = 0
    fingerprint = hashlib.sha256()

    for record in records:
        rows += 1
        raw_text = record.get(text_column)
        if raw_text is None:
            null_texts += 1
            text = ""
        else:
            text = str(raw_text)

        normalized = _normalize_text(text)
        if not normalized:
            empty_texts += 1
        normalized_counts[normalized] += 1

        label = record.get(label_column)
        if label is None or label == -1:
            unlabeled_rows += 1
            label_key = "unlabeled"
        else:
            label_key = str(label)
            label_counts[label_key] += 1
        normalized_labels[normalized].add(label_key)

        index = record.get(index_column)
        if index is None:
            missing_indices += 1
        else:
            index_counts[int(index)] += 1
        if len(normalized_examples[normalized]) < 5:
            normalized_examples[normalized].append(
                {
                    "idx": index,
                    "label": label_key,
                    "text": text,
                }
            )

        char_count = len(text)
        min_chars = char_count if min_chars is None else min(min_chars, char_count)
        max_chars = max(max_chars, char_count)
        fingerprint.update(
            json.dumps(
                {
                    "text": text,
                    "label": label_key,
                    "idx": index,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        )
        fingerprint.update(b"\n")

    duplicate_indices = sum(count - 1 for count in index_counts.values() if count > 1)
    duplicate_texts = sum(count - 1 for count in normalized_counts.values() if count > 1)
    conflicting_texts = sum(
        1
        for text, labels in normalized_labels.items()
        if text and len(labels - {"unlabeled"}) > 1
    )
    conflict_examples = [
        {
            "normalized_text": text,
            "examples": normalized_examples[text],
        }
        for text, labels in normalized_labels.items()
        if text and len(labels - {"unlabeled"}) > 1
    ][:20]
    issues = _split_issues(
        split=split,
        rows=rows,
        null_texts=null_texts,
        empty_texts=empty_texts,
        unlabeled_rows=unlabeled_rows,
        allow_unlabeled=allow_unlabeled,
        missing_indices=missing_indices,
        duplicate_indices=duplicate_indices,
        conflicting_texts=conflicting_texts,
    )

    return {
        "split": split,
        "rows": rows,
        "label_counts": dict(sorted(label_counts.items())),
        "unlabeled_rows": unlabeled_rows,
        "null_texts": null_texts,
        "empty_texts": empty_texts,
        "duplicate_indices": duplicate_indices,
        "duplicate_normalized_texts": duplicate_texts,
        "conflicting_duplicate_texts": conflicting_texts,
        "conflicting_duplicate_examples": conflict_examples,
        "min_chars": min_chars if min_chars is not None else 0,
        "max_chars": max_chars,
        "fingerprint_sha256": fingerprint.hexdigest(),
        "issues": issues,
    }


def build_iid_partition_manifest(
    *,
    num_items: int,
    labels: Sequence[int],
    num_clients: int,
    seeds: Sequence[int],
    rank_schedule: Sequence[int],
) -> dict[str, Any]:
    if len(rank_schedule) != num_clients:
        raise ValueError("rank_schedule length must equal num_clients")

    manifests = []
    for seed in seeds:
        partitions = iid_partition_indices(num_items, num_clients, seed=seed)
        client_records = []
        for client_index, indices in enumerate(partitions):
            label_counts = Counter(str(labels[index]) for index in indices)
            client_records.append(
                {
                    "client_id": f"c{client_index:02d}",
                    "rank": rank_schedule[client_index],
                    "num_samples": len(indices),
                    "label_counts": dict(sorted(label_counts.items())),
                    "indices": indices,
                }
            )
        manifests.append(
            {
                "seed": seed,
                "partition": "iid",
                "clients": client_records,
                "complete": sorted(index for part in partitions for index in part) == list(range(num_items)),
            }
        )

    return {
        "num_items": num_items,
        "num_clients": num_clients,
        "seeds": list(seeds),
        "rank_schedule": list(rank_schedule),
        "partitions": manifests,
    }


def summarize_partition_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for partition in manifest["partitions"]:
        sizes = [client["num_samples"] for client in partition["clients"]]
        positive_rates = [
            _positive_rate(client["label_counts"], client["num_samples"])
            for client in partition["clients"]
            if "1" in client["label_counts"]
        ]
        rows.append(
            {
                "seed": partition["seed"],
                "complete": partition["complete"],
                "min_client_samples": min(sizes),
                "max_client_samples": max(sizes),
                "client_sample_delta": max(sizes) - min(sizes),
                "min_positive_rate": min(positive_rates) if positive_rates else None,
                "max_positive_rate": max(positive_rates) if positive_rates else None,
            }
        )
    return rows


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _positive_rate(label_counts: Mapping[str, int], num_samples: int) -> float:
    if num_samples <= 0:
        return 0.0
    return float(label_counts.get("1", 0)) / num_samples


def _split_issues(
    *,
    split: str,
    rows: int,
    null_texts: int,
    empty_texts: int,
    unlabeled_rows: int,
    allow_unlabeled: bool,
    missing_indices: int,
    duplicate_indices: int,
    conflicting_texts: int,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if rows == 0:
        issues.append({"level": "error", "code": "empty_split", "message": f"{split} has no rows"})
    if null_texts:
        issues.append({"level": "error", "code": "null_text", "message": f"{split} has null texts"})
    if empty_texts:
        issues.append({"level": "error", "code": "empty_text", "message": f"{split} has empty texts"})
    if unlabeled_rows and not allow_unlabeled:
        issues.append(
            {
                "level": "error",
                "code": "unlabeled_rows",
                "message": f"{split} has unlabeled rows but should be labeled",
            }
        )
    if missing_indices:
        issues.append({"level": "error", "code": "missing_idx", "message": f"{split} has missing indices"})
    if duplicate_indices:
        issues.append({"level": "error", "code": "duplicate_idx", "message": f"{split} has duplicate indices"})
    if conflicting_texts:
        issues.append(
            {
                "level": "warning",
                "code": "conflicting_duplicate_text",
                "message": f"{split} has repeated normalized texts with conflicting labels",
            }
        )
    return issues
