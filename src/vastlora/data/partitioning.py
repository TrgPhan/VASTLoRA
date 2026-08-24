from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence


def iid_partition_indices(
    num_items: int,
    num_clients: int,
    *,
    seed: int,
) -> list[list[int]]:
    """Deterministically shuffle indices and split them as evenly as possible."""

    if num_items < 0:
        raise ValueError("num_items must be non-negative")
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    indices = list(range(num_items))
    rng = random.Random(seed)
    rng.shuffle(indices)

    partitions = [[] for _ in range(num_clients)]
    for offset, index in enumerate(indices):
        partitions[offset % num_clients].append(index)
    return partitions


def label_shard_partition_indices(
    labels: Sequence[int | str],
    num_clients: int,
    *,
    shards_per_client: int,
    seed: int,
) -> list[list[int]]:
    """Create deterministic label-sorted shards for a simple non-IID split."""

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if shards_per_client <= 0:
        raise ValueError("shards_per_client must be positive")

    total_shards = num_clients * shards_per_client
    if len(labels) < total_shards:
        raise ValueError("not enough items to allocate at least one item per shard")

    by_label: dict[int | str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)

    sorted_indices: list[int] = []
    for label in sorted(by_label, key=str):
        sorted_indices.extend(by_label[label])

    shards = [[] for _ in range(total_shards)]
    for offset, index in enumerate(sorted_indices):
        shards[offset % total_shards].append(index)

    rng = random.Random(seed)
    rng.shuffle(shards)

    partitions = [[] for _ in range(num_clients)]
    for client_id in range(num_clients):
        start = client_id * shards_per_client
        stop = start + shards_per_client
        for shard in shards[start:stop]:
            partitions[client_id].extend(shard)
    return partitions

