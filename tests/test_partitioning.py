from vastlora.data import iid_partition_indices, label_shard_partition_indices


def test_iid_partition_is_deterministic_and_complete() -> None:
    first = iid_partition_indices(23, 5, seed=11)
    second = iid_partition_indices(23, 5, seed=11)

    assert first == second
    assert sorted(index for part in first for index in part) == list(range(23))
    assert max(len(part) for part in first) - min(len(part) for part in first) <= 1


def test_label_shard_partition_is_deterministic_and_complete() -> None:
    labels = ["a"] * 10 + ["b"] * 10 + ["c"] * 10

    first = label_shard_partition_indices(labels, 3, shards_per_client=2, seed=19)
    second = label_shard_partition_indices(labels, 3, shards_per_client=2, seed=19)

    assert first == second
    assert sorted(index for part in first for index in part) == list(range(30))
    assert all(part for part in first)

