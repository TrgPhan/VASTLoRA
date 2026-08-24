from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets import load_dataset  # noqa: E402

from vastlora.data import (  # noqa: E402
    TextDatasetSpec,
    audit_text_split,
    build_iid_partition_manifest,
    summarize_partition_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and audit the Week-3 diagnostic dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "week3_dataset.json",
        help="Path to the Week-3 dataset JSON config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "week3",
        help="Directory for audit and partition artifacts.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    spec = TextDatasetSpec.from_mapping(config["dataset"])
    diagnostic = config["diagnostic"]

    dataset = load_dataset(spec.hub_path, spec.subset)
    audits = {}
    for split_name, split_data in dataset.items():
        audits[split_name] = audit_text_split(
            split_data,
            split=split_name,
            text_column=spec.text_column,
            label_column=spec.label_column,
            index_column=spec.index_column,
            allow_unlabeled=split_name in spec.unlabeled_splits,
        )

    train_labels = list(dataset[spec.train_split][spec.label_column])
    partition_manifest = build_iid_partition_manifest(
        num_items=dataset[spec.train_split].num_rows,
        labels=train_labels,
        num_clients=int(diagnostic["num_clients"]),
        seeds=[int(seed) for seed in diagnostic["seeds"]],
        rank_schedule=[int(rank) for rank in diagnostic["rank_schedule"]],
    )

    report = {
        "dataset": {
            "name": spec.name,
            "hub_path": spec.hub_path,
            "subset": spec.subset,
            "text_column": spec.text_column,
            "label_column": spec.label_column,
            "index_column": spec.index_column,
            "train_split": spec.train_split,
            "validation_split": spec.validation_split,
            "unlabeled_splits": list(spec.unlabeled_splits),
            "metric": spec.metric,
        },
        "diagnostic": diagnostic,
        "splits": audits,
        "partition_summary": summarize_partition_manifest(partition_manifest),
        "blocking_issues": [
            issue
            for split_audit in audits.values()
            for issue in split_audit["issues"]
            if issue["level"] == "error"
        ],
        "warnings": [
            issue
            for split_audit in audits.values()
            for issue in split_audit["issues"]
            if issue["level"] == "warning"
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / f"{spec.name}_dataset_audit.json"
    partition_path = args.output_dir / f"{spec.name}_iid_partitions.json"
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    partition_path.write_text(json.dumps(partition_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"audit": str(audit_path), "partitions": str(partition_path)}, indent=2))


if __name__ == "__main__":
    main()
