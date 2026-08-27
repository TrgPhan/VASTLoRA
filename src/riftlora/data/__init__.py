"""Data partitioning helpers for reproducible VAST-LoRA simulations."""

from riftlora.data.partitioning import iid_partition_indices, label_shard_partition_indices
from riftlora.data.week3 import (
    TextDatasetSpec,
    audit_text_split,
    build_iid_partition_manifest,
    summarize_partition_manifest,
)

__all__ = [
    "TextDatasetSpec",
    "audit_text_split",
    "build_iid_partition_manifest",
    "iid_partition_indices",
    "label_shard_partition_indices",
    "summarize_partition_manifest",
]

